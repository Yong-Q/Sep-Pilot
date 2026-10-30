"""Isolate application defaults before collection imports API/auth singletons.

Scientific fixtures remain free to select their own tmp_path/configuration.
The audit guard fails on production storage access rather than hiding writes.
Task-owned roots are retained as test evidence, including when a test fails.
"""
import os
import hashlib
from pathlib import Path
import sys
import tempfile
import threading

import pytest


PRODUCTION_ROOT = Path(__file__).resolve().parents[1]
_isolation_parent = Path(os.environ.get('BIMEM_TEST_ISOLATION_DIR') or PRODUCTION_ROOT / 'tmp').expanduser().resolve()
_isolation_parent.mkdir(parents=True, exist_ok=True)
ISOLATION_ROOT = Path(tempfile.mkdtemp(prefix='pytest-isolation-', dir=_isolation_parent))
os.environ['BIMEM_BACKGROUND_MONITOR'] = '0'
os.environ['BIMEM_AUTH_STATE_DIR'] = str(ISOLATION_ROOT / 'collection-auth')
os.environ['BIMEM_JOB_WATCH_FILE'] = str(ISOLATION_ROOT / 'collection-watch.json')
os.environ['BIMEM_TASK_LINE_FILE'] = str(ISOLATION_ROOT / 'collection-task-lines.json')

# CPython's `open` audit event omits os.open's dir_fd. Preserve that argument
# around the real call so fd-relative pytest/shutil cleanup is checked against
# its actual directory, rather than the repository cwd.
_open_context = threading.local()
_blocked_storage_accesses = []
_os_open = os.open
def _open_with_directory_context(path, flags, mode=0o777, *, dir_fd=None):
    previous = getattr(_open_context, 'dir_fd', None)
    _open_context.dir_fd = dir_fd
    try:
        return _os_open(path, flags, mode, dir_fd=dir_fd)
    finally:
        _open_context.dir_fd = previous
os.open = _open_with_directory_context


def _production_storage(path, dir_fd=None):
    if not isinstance(path, (str, bytes, os.PathLike)):
        return False
    target = Path(os.fsdecode(path))
    if not target.is_absolute() and isinstance(dir_fd, int) and dir_fd >= 0:
        target = Path(os.readlink('/proc/self/fd/' + str(dir_fd))) / target
    target = target.resolve()
    return (any(target.is_relative_to(PRODUCTION_ROOT / name) for name in ('data/state', 'runs'))
            or target.parent == PRODUCTION_ROOT and any(target.name.startswith(name) for name in
                ('users.json', 'tokens.json', 'conversation_logs.jsonl')))


def _guard_production_storage(event, arguments):
    if event == 'open' and _production_storage(arguments[0], getattr(_open_context, 'dir_fd', None)):
        _blocked_storage_accesses.append((event, str(arguments[0])))
        raise AssertionError('test accessed production storage: ' + str(arguments[0]))
    if event in {'os.remove', 'os.rmdir', 'os.mkdir', 'os.chmod', 'os.rename', 'os.link', 'os.symlink'}:
        if event in {'os.rename', 'os.link'}:
            paths = [(arguments[0], arguments[2]), (arguments[1], arguments[3])]
        elif event == 'os.symlink':
            paths = [(arguments[0], arguments[2]), (arguments[1], arguments[2])]
        else:
            offset = 2 if event in {'os.mkdir', 'os.chmod'} else 1
            paths = [(arguments[0], arguments[offset])]
        if any(_production_storage(path, dir_fd) for path, dir_fd in paths):
            _blocked_storage_accesses.append((event, str(paths)))
            raise AssertionError('test mutated production storage: ' + str(paths))


sys.addaudithook(_guard_production_storage)

# Seed defaults before test modules import Session/API at collection time.
from agents import config as agent_config
agent_config.set_config(agent_config.AgentConfig(api_key='test', project_root=ISOLATION_ROOT / 'collection'))


def pytest_configure(config):
    config.addinivalue_line('markers', 'production_guard_probe: intentionally verifies forbidden production storage access')

@pytest.fixture(autouse=True)
def isolated_storage(monkeypatch, request):
    """Fresh caches and state for each test; explicit test monkeypatches win."""
    violations_before = len(_blocked_storage_accesses)
    from agents import job_watch, task_line, watch_context
    import auth
    root = ISOLATION_ROOT / ('test-' + hashlib.sha256(request.node.nodeid.encode()).hexdigest()[:16])
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(agent_config, '_config', agent_config.AgentConfig(api_key='test', project_root=root))
    monkeypatch.setenv('BIMEM_AUTH_STATE_DIR', str(root))
    monkeypatch.setenv('BIMEM_JOB_WATCH_FILE', str(root / 'job_watch.json'))
    monkeypatch.setenv('BIMEM_TASK_LINE_FILE', str(root / 'task_lines.json'))
    monkeypatch.setattr(task_line, 'LINE_FILE', str(root / 'task_lines.json'))
    monkeypatch.setattr(task_line, 'LEGACY_LINE_FILE', str(root / 'absent-legacy-task-lines.json'))
    monkeypatch.setenv('BIMEM_BACKGROUND_MONITOR', '0')
    for key, path in {'USER_DB': root / 'users.json', 'TOKEN_DB': root / 'tokens.json',
                      'CONVERSATION_LOG': root / 'conversation_logs.jsonl',
                      'CREDENTIAL_KEY_FILE': root / 'credential_master.key'}.items():
        monkeypatch.setattr(auth, key, path)
    manager = auth.UserSessionManager()
    monkeypatch.setattr(auth, 'session_manager', manager)
    watch = job_watch.JobWatch(str(root / 'job_watch.json'))
    monkeypatch.setattr(job_watch, '_watch', watch)
    monkeypatch.setattr(task_line, '_store', task_line.TaskLineStore(str(root / 'task_lines.json')))
    watch_context.clear_context()
    if 'api' in sys.modules:
        api = sys.modules['api']
        monkeypatch.setattr(api, 'session_manager', manager)
        monkeypatch.setattr(api, 'USER_DB', auth.USER_DB)
        monkeypatch.setattr(api, 'CONVERSATION_LOG', auth.CONVERSATION_LOG)
        monkeypatch.setattr(api, '_watch', watch)
        for key, value in {'_workflow_runtimes': {}, '_conv_locks': {}, '_notification_backlog': [],
                           '_lifecycle_tasks': set()}.items():
            monkeypatch.setattr(api, key, value)
    from bimem_agent import paths
    paths.load_runtime_config.cache_clear()
    yield root
    watch.stop()
    watch_context.clear_context()
    paths.load_runtime_config.cache_clear()
    if (len(_blocked_storage_accesses) != violations_before
            and request.node.get_closest_marker('production_guard_probe') is None):
        pytest.fail('application caught a forbidden production storage access: '
                    + repr(_blocked_storage_accesses[violations_before:]))
