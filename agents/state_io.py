"""Atomic, process-locked state transactions. Errors propagate (fail closed)."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import threading
import time
import stat
import weakref

_registry_lock = threading.Lock()
_path_locks = weakref.WeakValueDictionary()
_held_paths = threading.local()
_guard_fds = set()


def _before_fork():
    # Do not inherit an unregistered flock descriptor or a half-updated table.
    _registry_lock.acquire()


def _after_fork_child():
    global _registry_lock, _path_locks, _held_paths, _guard_fds
    # These descriptions also belong to the parent. Closing our copies is
    # required; LOCK_UN would release the parent's real transaction lock.
    for fd in _guard_fds:
        os.close(fd)
    _registry_lock = threading.Lock()
    _path_locks = weakref.WeakValueDictionary()
    _held_paths = threading.local()
    _guard_fds = set()


os.register_at_fork(before=_before_fork, after_in_parent=lambda: _registry_lock.release(),
                    after_in_child=_after_fork_child)


@contextmanager
def _transaction_guard(path):
    import fcntl
    key = str(path)
    held = getattr(_held_paths, 'paths', ())
    if key in held:
        raise RuntimeError('same state transaction cannot be reentered')
    if held and key < held[-1]:
        raise RuntimeError('nested state transactions must acquire canonical path order')
    with _registry_lock:
        lock = _path_locks.get(key)
        if lock is None:
            lock = threading.RLock()
            _path_locks[key] = lock
    lock.acquire()
    pid = os.getpid()
    fd = None
    try:
        with _registry_lock:
            fd = os.open(key + '.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            _guard_fds.add(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError('state lock must be a regular file')
        fcntl.flock(fd, fcntl.LOCK_EX)
        _held_paths.paths = (*held, key)
        yield pid
    finally:
        # A fork inside the caller's transaction must not unlock/close a
        # parent's descriptor or release its inherited threading lock.
        if os.getpid() == pid:
            _held_paths.paths = held
            try:
                if fd is not None:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_UN)
                    finally:
                        with _registry_lock:
                            _guard_fds.discard(fd)
                            os.close(fd)
            finally:
                lock.release()


def _read_state(path):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False, {}
    with os.fdopen(fd, 'r', encoding='utf-8') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('state file must be a regular file')
        return True, json.load(stream)


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
    # Parent aliases share one lock and one publication namespace. A leaf
    # symlink is never followed: replacing the alias differs from replacing
    # its target, so silently resolving it would change transaction semantics.
    path = path.parent.resolve() / path.name
    path.parent.mkdir(parents=True, exist_ok=True)
    with _transaction_guard(path) as owner_pid:
        existed, data = _read_state(path)
        if not isinstance(data, dict):
            raise ValueError('state file must contain a JSON object')
        before = json.dumps(data, ensure_ascii=False, default=str)
        yield data
        if os.getpid() != owner_pid:
            raise RuntimeError('inherited state transaction cannot commit after fork')
        if path.is_symlink():
            raise ValueError('state leaf became a symlink during transaction')
        encoded = json.dumps(data, ensure_ascii=False, default=str)
        # Many callers inspect/reject a candidate under this lock. Avoid
        # rewriting and fsyncing the entire store when nothing changed.
        # Compare JSON, not Python equality (True, 1 and 1.0 differ).
        if existed and encoded == before:
            return
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
        try:
            with os.fdopen(fd, 'w', encoding='utf-8') as stream:
                stream.write(encoded)
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
