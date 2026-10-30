"""No-op transactions avoid disk commits; actual mutations stay durable."""
import json

import pytest

from agents import state_io


def test_existing_noop_transaction_does_not_create_or_sync_a_replacement(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    state_io.write_checkpoint(path, {'nodes': {'a': {'status': 'pending'}}})
    before = path.stat()
    monkeypatch.setattr(state_io.tempfile, 'mkstemp', lambda *a, **k: pytest.fail('no-op must not create a replacement'))
    monkeypatch.setattr(state_io.os, 'fsync', lambda *a: pytest.fail('no-op must not fsync'))
    for _ in range(20):
        with state_io.json_transaction(path) as data:
            assert data['nodes']['a']['status'] == 'pending'
    assert (path.stat().st_ino, path.stat().st_mtime_ns) == (before.st_ino, before.st_mtime_ns)


def test_identical_checkpoint_and_reverted_mutation_do_not_commit(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    state_io.write_checkpoint(path, {'counter': 1})
    monkeypatch.setattr(state_io.os, 'fsync', lambda *a: pytest.fail('unchanged content must not fsync'))
    state_io.write_checkpoint(path, {'counter': 1})
    with state_io.json_transaction(path) as data:
        data['counter'] = 2
        data['counter'] = 1


@pytest.mark.parametrize('initial,changed', [(1, True), (1, 1.0), ('pending', 'running')])
def test_real_json_changes_still_sync_file_and_directory(tmp_path, monkeypatch, initial, changed):
    path = tmp_path / 'state.json'
    state_io.write_checkpoint(path, {'value': initial})
    original = state_io.os.fsync
    synced = []
    def sync(fd):
        synced.append(fd)
        original(fd)
    monkeypatch.setattr(state_io.os, 'fsync', sync)
    with state_io.json_transaction(path) as data:
        data['value'] = changed
    stored = json.loads(path.read_text())['value']
    assert type(stored) is type(changed) and stored == changed
    assert len(synced) == 2


def test_missing_empty_state_is_created_durably(tmp_path, monkeypatch):
    path = tmp_path / 'new.json'
    original = state_io.os.fsync
    synced = []
    monkeypatch.setattr(state_io.os, 'fsync', lambda fd: synced.append(fd) or original(fd))
    with state_io.json_transaction(path):
        pass
    assert json.loads(path.read_text()) == {}
    assert len(synced) == 2


def test_failed_transaction_keeps_original_state_without_sync(tmp_path, monkeypatch):
    path = tmp_path / 'state.json'
    state_io.write_checkpoint(path, {'value': 'original'})
    monkeypatch.setattr(state_io.os, 'fsync', lambda *a: pytest.fail('aborted changes must not sync'))
    with pytest.raises(RuntimeError, match='abort'):
        with state_io.json_transaction(path) as data:
            data['value'] = 'uncommitted'
            raise RuntimeError('abort')
    assert json.loads(path.read_text()) == {'value': 'original'}
