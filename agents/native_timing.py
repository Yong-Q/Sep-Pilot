"""Bounded, content-free diagnostic timings. Never an execution/result receipt."""
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import fcntl
import json
import os
import stat
import time
import uuid

MAX_FILE_BYTES = 8 * 1024 * 1024
_PHASES = frozenset({'total', 'partial_admission', 'recorded_preparation_witness',
    'original_attempt', 'original_input_sha', 'timeout_subset', 'material_quality',
    'timeout_script_joblog', 'retry_subset_sha', 'adapter_bootstrap', 'inventory_refresh',
    'child_subprocess', 'jobwatch_registration', 'taskline_update', 'recovery_collection'})
_CURRENT = ContextVar('native_diagnostic_timing', default=None)


@contextmanager
def phase(name):
    report = _CURRENT.get()
    if report is None or name not in _PHASES:
        yield
        return
    try:
        wall, cpu = time.perf_counter_ns(), time.thread_time_ns()
    except Exception:
        yield
        return
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        try:
            sample = report['phases'].setdefault(name, {'count': 0, 'wall_ms': 0., 'thread_cpu_ms': 0., 'ok': True})
            sample['count'] = min(1000000, sample['count'] + 1)
            sample['wall_ms'] += max(0, time.perf_counter_ns() - wall) / 1e6
            sample['thread_cpu_ms'] += max(0, time.thread_time_ns() - cpu) / 1e6
            sample['ok'] = sample['ok'] and ok
        except Exception:
            pass  # Diagnostic clocks must never change scientific/control behavior.


def timed_call(name, function, *args, **kwargs):
    with phase(name): return function(*args, **kwargs)


def timed_phase(name):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            return timed_call(name, function, *args, **kwargs)
        return wrapped
    return decorate


def timed_native(tool):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            if _CURRENT.get() is not None or tool != 'run_cdft':
                return function(*args, **kwargs)
            try:
                report = {'schema_version': 1, 'trace_id': uuid.uuid4().hex, 'tool': tool,
                          'started_at': time.time(), 'outcome': 'raised', 'diagnostic_only': True, 'phases': {}}
                token = _CURRENT.set(report)
            except Exception:
                return function(*args, **kwargs)
            try:
                with phase('total'):
                    result = function(*args, **kwargs)
                    report['outcome'] = 'returned'
                    return result
            finally:
                _CURRENT.reset(token)
                try: _write_report(report)
                except Exception: pass
        return wrapped
    return decorate


def _scope():
    from .watch_context import get_context
    from .config import get_config
    owner = {key: get_context().get(key) for key in ('username', 'conv_id')}
    if not all(isinstance(value, str) and value and '/' not in value and value not in {'.', '..'}
               for value in owner.values()): return None
    return get_config().project_root / 'runs' / owner['username'] / owner['conv_id'], owner


def _write_report(report):
    """One bounded append, one dedicated lock, no shared-state lock or UI output."""
    from .report_recovery import _OwnedFiles
    raw = (json.dumps(report, separators=(',', ':'), allow_nan=False) + '\n').encode()
    if len(raw) > 8192: return
    scoped = _scope()
    if scoped is None: return
    root, owner = scoped
    owned = _OwnedFiles(root, owner)
    directory = fd = None
    try:
        try: os.mkdir('performance', 0o700, dir_fd=owned.fd)
        except FileExistsError: pass
        directory = os.open('performance', os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=owned.fd)
        fd = os.open('native_timing.jsonl', os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                     0o600, dir_fd=directory)
        if not stat.S_ISREG(os.fstat(fd).st_mode): return
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        observed = os.fstat(fd)
        if (stat.S_ISREG(observed.st_mode) and observed.st_nlink == 1
                and observed.st_size + len(raw) <= MAX_FILE_BYTES):
            os.write(fd, raw)
    finally:
        if fd is not None: os.close(fd)
        if directory is not None: os.close(directory)
        owned.close()
