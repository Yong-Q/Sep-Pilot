"""Atomic, process-locked state transactions. Errors propagate (fail closed)."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import time

_lock = threading.RLock()


class ConversationLock:
    """threading.Lock-compatible conversation lease shared by API processes."""
    def __init__(self, path):
        self.path = Path(path)
        self.local = threading.Lock()
        self.stream = None

    def acquire(self, blocking=True, timeout=-1):
        started = time.monotonic()
        if not blocking:
            acquired = self.local.acquire(False)
        elif timeout >= 0:
            acquired = self.local.acquire(timeout=timeout)
        else:
            acquired = self.local.acquire()
        if not acquired:
            return False
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            stream = open(self.path, 'a+')
            import fcntl
        except BaseException:
            self.local.release()
            raise
        try:
            while True:
                try:
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    self.stream = stream
                    return True
                except BlockingIOError:
                    if not blocking or timeout >= 0 and time.monotonic() - started >= timeout:
                        stream.close()
                        self.local.release()
                        return False
                    time.sleep(.01)
        except BaseException:
            stream.close()
            self.local.release()
            raise

    def release(self):
        if self.stream is None:
            raise RuntimeError('conversation lease is not held')
        import fcntl
        fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
        self.stream.close()
        self.stream = None
        self.local.release()


@contextmanager
def json_transaction(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, open(str(path) + '.lock', 'a+') as guard:
        import fcntl
        fcntl.flock(guard.fileno(), fcntl.LOCK_EX)
        try:
            data = json.loads(path.read_text()) if path.exists() else {}
            if not isinstance(data, dict):
                raise ValueError('state file must contain a JSON object')
            yield data
            fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
            try:
                with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                    json.dump(data, stream, ensure_ascii=False, default=str)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, path)
                dirfd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dirfd)
                finally:
                    os.close(dirfd)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        finally:
            fcntl.flock(guard.fileno(), fcntl.LOCK_UN)


def write_checkpoint(path, state):
    with json_transaction(path) as data:
        data.clear()
        data.update(state)


def compact_checkpoint_evidence(state, evidence_root):
    """Store full tool facts once; checkpoints retain bounded previews + refs."""
    root = Path(evidence_root)
    memory = state.get('memory', {})
    logs = [memory.get('tool_call_log', [])]
    logs.extend(a.get('tool_call_log', []) for a in memory.get('agent_memories', {}).values())
    references = {}
    for log in logs:
        for call in log:
            call_id = call.get('call_id')
            if not call_id or not isinstance(call_id, str) or not call_id.isalnum():
                continue
            path = root / (call_id + '.json')
            if 'result' in call:
                if not path.exists():
                    write_checkpoint(path, call)
                else:
                    original = json.loads(path.read_text())
                    if original.get('call_id') != call_id or original.get('result') != call['result']:
                        raise ValueError('tool evidence collision or modification')
                call['evidence_path'] = str(path)
                call.pop('result', None)
            if call.get('evidence_path'):
                references[call_id] = call['evidence_path']
    return references


def apply_evidence_references(memory, references):
    logs = [memory.tool_call_log]
    logs.extend(a.get('tool_call_log', []) for a in memory.agent_memories.values())
    for log in logs:
        for call in log:
            if call.get('call_id') in references:
                call['evidence_path'] = references[call['call_id']]
                call.pop('result', None)
