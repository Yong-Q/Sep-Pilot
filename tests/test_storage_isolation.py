"""Storage isolation checks; never read production accounts or job records."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def test_auth_storage_override_is_applied_before_import(tmp_path):
    production = Path(__file__).resolve().parents[1]
    script = '''
import builtins, io, os
from pathlib import Path
root = Path(os.environ['TEST_PRODUCTION_ROOT'])
real_open, real_io_open = builtins.open, io.open
def guarded(delegate):
    def call(file, *args, **kwargs):
        if not isinstance(file, int):
            path = Path(file).absolute()
            if path.parent == root and path.name.startswith(('users.json', 'tokens.json')):
                raise AssertionError('production authentication storage accessed')
        return delegate(file, *args, **kwargs)
    return call
builtins.open, io.open = guarded(real_open), guarded(real_io_open)
import auth
assert auth.USER_DB == Path(os.environ['BIMEM_AUTH_STATE_DIR']) / 'users.json'
assert auth.TOKEN_DB == Path(os.environ['BIMEM_AUTH_STATE_DIR']) / 'tokens.json'
assert auth.CREDENTIAL_KEY_FILE.is_relative_to(Path(os.environ['BIMEM_AUTH_STATE_DIR']))
auth.migrate_auth_storage()
assert auth.USER_DB.is_file() and auth.TOKEN_DB.is_file()
print('isolated')
'''
    environment = dict(os.environ, BIMEM_AUTH_STATE_DIR=str(tmp_path), TEST_PRODUCTION_ROOT=str(production))
    result = subprocess.run([sys.executable, '-c', script], env=environment,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'isolated'


def test_defaults_and_api_startup_use_isolated_storage(isolated_storage):
    import asyncio
    import auth
    import api
    from agents.config import get_config
    from agents.job_watch import get_watch
    from agents.task_line import get_store
    assert get_config().project_root == isolated_storage
    assert Path(get_store()._file).is_relative_to(isolated_storage)
    assert Path(get_watch()._watch_file).is_relative_to(isolated_storage)
    assert auth.USER_DB.is_relative_to(isolated_storage)
    assert api.session_manager is auth.session_manager
    asyncio.run(api._startup_watch())
    assert auth.USER_DB.is_file() and auth.TOKEN_DB.is_file()
    get_watch().register('9')
    get_watch().register('10')
    assert {'9', '10'} <= set(json.loads(Path(get_watch()._watch_file).read_text()))


@pytest.mark.parametrize('relative', ['users.json', 'tokens.json', 'data/state/isolation-proof',
                                     'runs/isolation-proof/session_checkpoint.json'])
@pytest.mark.production_guard_probe
def test_production_storage_writes_fail_before_touching_disk(relative):
    production = Path(__file__).resolve().parents[1]
    with pytest.raises(AssertionError, match='production storage'):
        (production / relative).write_text('must never be written')


def test_temp_cleanup_and_rename_use_actual_dir_fd(tmp_path):
    for name in ('tokens.json.lock', 'users.json.lock'):
        (tmp_path / name).write_text('synthetic lock')
    descriptor = os.open(tmp_path, os.O_RDONLY)
    try:
        handle = os.open('tokens.json.lock', os.O_RDONLY, dir_fd=descriptor)
        os.close(handle)
        os.rename('tokens.json.lock', 'renamed.lock', src_dir_fd=descriptor, dst_dir_fd=descriptor)
        os.unlink('renamed.lock', dir_fd=descriptor)
        os.unlink('users.json.lock', dir_fd=descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.production_guard_probe
def test_production_dir_fd_still_rejects_mutation(tmp_path):
    production = Path(__file__).resolve().parents[1]
    descriptor = os.open(production, os.O_RDONLY)
    isolated = os.open(tmp_path, os.O_RDONLY)
    try:
        with pytest.raises(AssertionError, match='production storage'):
            os.unlink('tokens.json.lock', dir_fd=descriptor)
        with pytest.raises(AssertionError, match='production storage'):
            os.open('tokens.json', os.O_WRONLY | os.O_CREAT, dir_fd=descriptor)
        with pytest.raises(AssertionError, match='production storage'):
            os.rename('uncreated-source', 'users.json', src_dir_fd=isolated, dst_dir_fd=descriptor)
    finally:
        os.close(descriptor)
        os.close(isolated)
