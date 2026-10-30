"""Owner-scoped, recoverable display work; never execution authority.

Call request only after scientific state and outbox are durable. The trusted
renderer must load fresh authoritative state; tickets carry no scientific facts.
This standalone lane does not alter checkpoint, TaskLine or event delivery.
"""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import copy
import fcntl
import json
import os
from pathlib import Path
import stat
import threading
import time
import uuid


class DisplayMarkerError(ValueError):
    """Invalid advisory bytes, distinct from a foreign/changed scope."""


class DisplaySync:
    MAX_BYTES = 8192
    LOCK_TIMEOUT = 1.0

    def __init__(self, root, owner, render):
        self.root = Path(os.path.abspath(root))
        if (not isinstance(owner, dict) or set(owner) != {'username', 'conv_id'}
                or any(not isinstance(v, str) or not v or len(v) > 128
                       or v in {'.', '..'} or '/' in v or '\\' in v for v in owner.values())
                or (self.root.parent.name, self.root.name) != (owner['username'], owner['conv_id'])):
            raise PermissionError('display owner does not match conversation root')
        if self.root.resolve(strict=True) != self.root or not callable(render):
            raise ValueError('display requires a real scope and trusted renderer')
        self.owner, self.render = copy.deepcopy(owner), render
        self._root_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        self._root_identity = os.fstat(self._root_fd).st_ino
        self._pid = os.getpid()
        self._guard = threading.Lock()
        self._future = None
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='display-sync')
        try:
            with self._file_lock('display_sync.lock'):
                state = self._read()
                if state is None:
                    self._write({'schema_version': 1, 'owner': self.owner,
                                 'revision': 0, 'rendered_revision': 0, 'reason': '', 'last_error': None})
        except BaseException:
            self._pool.shutdown(wait=False)
            os.close(self._root_fd)
            raise

    def _check_root(self):
        if os.getpid() != self._pid:
            raise RuntimeError('display lane must be reconstructed after fork')
        current = os.stat(self.root, follow_symlinks=False)
        if (not stat.S_ISDIR(current.st_mode) or current.st_ino != self._root_identity
                or self.root.resolve(strict=True) != self.root):
            raise PermissionError('display conversation root changed')

    @contextmanager
    def _file_lock(self, name):
        self._check_root()
        fd = os.open(name, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o600, dir_fd=self._root_fd)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('display lock must be regular')
            deadline = time.monotonic() + self.LOCK_TIMEOUT
            while True:
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError('display lock is busy; durable dirty remains recoverable')
                    time.sleep(.005)
            self._check_root()
            yield
        finally:
            os.close(fd)  # Closing only this description releases its flock.

    def _read(self):
        try:
            fd = os.open('display_sync.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK,
                         dir_fd=self._root_fd)
        except FileNotFoundError:
            return None
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('display marker must be a regular file')
            if info.st_size > self.MAX_BYTES:
                raise DisplayMarkerError('invalid display marker size')
            raw = stream.read(self.MAX_BYTES + 1)
        if len(raw) > self.MAX_BYTES:
            raise DisplayMarkerError('display marker exceeds bounded format')
        try:
            state = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError) as error:
            raise DisplayMarkerError('invalid display marker JSON') from error
        if not isinstance(state, dict) or 'owner' not in state:
            raise DisplayMarkerError('display marker has no owner schema')
        if state['owner'] != self.owner:
            raise PermissionError('display marker belongs to a different owner')
        revision, rendered = state.get('revision'), state.get('rendered_revision')
        if (state.get('schema_version') != 1 or type(revision) is not int or type(rendered) is not int
                or not 0 <= rendered <= revision or revision > 2**63 - 1
                or not isinstance(state.get('reason'), str) or len(state['reason']) > 256):
            raise DisplayMarkerError('invalid display revision')
        return state

    def _write(self, state):
        raw = json.dumps(state, ensure_ascii=False, sort_keys=True).encode()
        if len(raw) > self.MAX_BYTES:
            raise ValueError('display marker exceeds bounded format')
        self._check_root()
        try:
            info = os.stat('display_sync.json', dir_fd=self._root_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode):
                raise ValueError('display marker must be regular')
        except FileNotFoundError:
            pass
        temporary = '.display-sync-' + uuid.uuid4().hex
        fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                     0o600, dir_fd=self._root_fd)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            self._check_root()
            os.replace(temporary, 'display_sync.json', src_dir_fd=self._root_fd, dst_dir_fd=self._root_fd)
            os.fsync(self._root_fd)
        finally:
            try: os.unlink(temporary, dir_fd=self._root_fd)
            except FileNotFoundError: pass

    def status(self):
        with self._file_lock('display_sync.lock'):
            state = self._read()
            if state is None:
                raise ValueError('display marker disappeared')
            return {**state, 'dirty': state['revision'] != state['rendered_revision']}

    def request(self, reason):
        self._check_root()
        if not isinstance(reason, str) or not reason or len(reason) > 256:
            raise ValueError('display reason must be a bounded nonempty label')
        with self._guard:
            if self._closed: raise RuntimeError('display lane is closed')
            with self._file_lock('display_sync.lock'):
                state = self._read()
                if state is None: raise ValueError('display marker disappeared')
                state.update(revision=state['revision'] + 1, reason=reason, last_error=None)
                self._write(state)
            return self._kick_locked()

    def _kick_locked(self):
        if self._future is None or self._future.done():
            self._future = self._pool.submit(self._drain)
        return self._future

    def kick(self):
        """Resume durable dirty after restart; no polling or science retry."""
        self._check_root()
        with self._guard:
            if self._closed: raise RuntimeError('display lane is closed')
            return self._kick_locked()

    def _drain(self):
        with self._file_lock('display_sync.render.lock'):
            while True:
                state = self.status()
                if not state['dirty']:
                    # request's durable update and its choice of in-flight
                    # future share this guard. Re-read after acquiring it so
                    # the old future cannot strand a just-published request.
                    with self._guard:
                        latest = self.status()
                        if not latest['dirty']:
                            self._future = None
                            return latest
                    continue
                revision = state['revision']
                ticket = {key: copy.deepcopy(state[key]) for key in ('owner', 'revision', 'reason')}
                try:
                    self.render(ticket)
                except BaseException as error:
                    with self._file_lock('display_sync.lock'):
                        latest = self._read()
                        if latest is not None and latest['revision'] == revision:
                            latest['last_error'] = type(error).__name__ + ': ' + str(error)[:256]
                            self._write(latest)
                    raise
                with self._file_lock('display_sync.lock'):
                    latest = self._read()
                    if latest is None: raise ValueError('display marker disappeared')
                    if latest['revision'] == revision:
                        latest.update(rendered_revision=revision, last_error=None)
                        self._write(latest)
                    # A newer ticket stays dirty and is rendered from fresh state.

    def flush(self, timeout=None):
        """Explicitly wait for display only. Timeout never cancels scientific work."""
        return self.kick().result(timeout=timeout)

    def close(self):
        # Cleanup closes only our descriptor, even when its pathname moved.
        # It must never close a descriptor inherited from a different process.
        if os.getpid() != self._pid:
            raise RuntimeError('display lane must be reconstructed after fork')
        with self._guard:
            if self._closed:
                return
            self._closed = True
        self._pool.shutdown(wait=True, cancel_futures=False)
        os.close(self._root_fd)
