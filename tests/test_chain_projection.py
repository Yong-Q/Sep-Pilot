"""Current-state reads must not parse accumulated revision payloads."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from agents import orchestration_chain as chain
from tests.test_orchestration_chain import fixture

OWNER = {'username': 'u', 'conv_id': 'c'}


def test_repeated_transitions_keep_large_history_out_of_current_manifest(tmp_path):
    line, runtime = fixture()
    history = [{'attempt': 1, 'result': 'scientific evidence' * 20000}]
    runtime['nodes']['compute']['attempt_history'] = history
    for n in range(8):
        runtime['nodes']['compute']['status'] = 'running' if n % 2 else 'pending'
        chain.sync_chain(tmp_path, line, runtime, str(n))
    path = tmp_path / 'orchestration_chain.json'
    assert path.stat().st_size < 100_000
    data = json.loads(path.read_text())
    node = data['current']['nodes']['compute']
    assert chain.read_chain_reference(tmp_path, OWNER, node['attempt_history_ref']) == history
    assert len(data['revisions']) == 8
    for revision in data['revisions']:
        assert 'changes' not in revision
        assert chain.read_chain_reference(tmp_path, OWNER, revision['changes_ref'])


def test_graph_read_uses_small_sidecar_without_reading_chain_or_evidence(tmp_path, monkeypatch):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    original = Path.read_text
    reads = []
    def tracked(path, *args, **kwargs):
        reads.append(path.name)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', tracked)
    graph = chain.read_last_graph(tmp_path, OWNER)
    assert graph['nodes'][0]['step_id'] == 'compute'
    assert 'orchestration_chain.json' not in reads
    assert all(not name.endswith('.json') or name == 'orchestration_current.json' for name in reads)
    assert chain.read_last_graph(tmp_path, {**OWNER, 'username': 'other'}) == {}


def test_legacy_migration_preserves_exact_bytes_and_revision_audit(tmp_path):
    line, runtime = fixture()
    legacy = {'schema_version': 1, 'owner': OWNER, 'current': {},
              'revisions': [{'sequence': 1, 'parent_hash': None,
                             'changes': {'old': {'before': None, 'after': 'x' * 200000}},
                             'hash': 'legacy-recorded-hash'}]}
    raw = json.dumps(legacy, indent=3).encode()
    path = tmp_path / 'orchestration_chain.json'
    path.write_bytes(raw)
    chain.sync_chain(tmp_path, line, runtime, 'migration')
    data = json.loads(path.read_text())
    archive = tmp_path / data['legacy_archive_ref']['path']
    assert archive.read_bytes() == raw
    assert data['legacy_archive_ref']['sha256'] == hashlib.sha256(raw).hexdigest()
    assert data['revisions'][0]['hash'] == 'legacy-recorded-hash'
    assert chain.read_chain_revision(tmp_path, OWNER, 1)['changes'] == legacy['revisions'][0]['changes']
    assert data['revisions'][1]['parent_hash'] == 'legacy-recorded-hash'
    assert path.stat().st_size < 20000


def test_noop_does_not_rewrite_manifest_sidecar_or_append_evidence(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial', turn_id='turn')
    paths = list(tmp_path.rglob('*.json'))
    before = {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in paths}
    result = chain.sync_chain(tmp_path, line, runtime, 'initial', turn_id='turn')
    assert not result['updated']
    assert {str(p): (p.read_bytes(), p.stat().st_mtime_ns) for p in tmp_path.rglob('*.json')} == before


def test_reference_rejects_owner_escape_and_tampered_content(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    data = json.loads((tmp_path / 'orchestration_chain.json').read_text())
    reference = data['revisions'][0]['changes_ref']
    with pytest.raises(PermissionError):
        chain.read_chain_reference(tmp_path, {**OWNER, 'username': 'other'}, reference)
    with pytest.raises((ValueError, PermissionError)):
        chain.read_chain_reference(tmp_path, OWNER, {**reference, 'path': '../outside.json'})
    path = tmp_path / reference['path']
    path.write_text('{}')
    with pytest.raises(ValueError):
        chain.read_chain_reference(tmp_path, OWNER, reference)


def test_stale_sidecar_falls_back_to_current_manifest(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    sidecar = (tmp_path / 'orchestration_current.json').read_bytes()
    runtime['nodes']['compute']['status'] = 'succeeded'
    chain.sync_chain(tmp_path, line, runtime, 'completed')
    (tmp_path / 'orchestration_current.json').write_bytes(sidecar)
    graph = chain.read_last_graph(tmp_path, OWNER)
    assert graph['nodes'][0]['status'] == 'succeeded'
    assert graph['revision'] == 2


def test_strict_graph_reader_distinguishes_missing_corrupt_and_foreign(tmp_path):
    assert chain.read_last_graph(tmp_path, OWNER, strict=True) == {}
    path = tmp_path / 'orchestration_chain.json'
    path.write_text('bad json')
    with pytest.raises(ValueError):
        chain.read_last_graph(tmp_path, OWNER, strict=True)
    path.write_text(json.dumps({'owner': {**OWNER, 'username': 'other'}}))
    with pytest.raises(PermissionError):
        chain.read_last_graph(tmp_path, OWNER, strict=True)


def test_empty_legacy_restore_compacts_retained_graph_and_preserves_history(tmp_path):
    history = [{'full_result': 'old evidence' * 20000}]
    legacy = {'schema_version': 1, 'owner': OWNER,
              'current': {'approved_plan_version': 2, 'nodes': {
                  'compute': {'step_id': 'compute', 'attempt_history': history}}}}
    (tmp_path / 'orchestration_chain.json').write_text(json.dumps(legacy))
    chain.sync_chain(tmp_path, {**OWNER, 'steps': []}, {}, 'restore')
    assert (tmp_path / 'orchestration_chain.json').stat().st_size < 20000
    node = chain.read_last_graph(tmp_path, OWNER)['nodes'][0]
    assert chain.read_chain_reference(tmp_path, OWNER, node['attempt_history_ref']) == history


def test_legacy_revision_hydrates_original_hash_material(tmp_path):
    line, runtime = fixture()
    revision = {'sequence': 1, 'parent_hash': None, 'time': 1.0,
                'reason': 'old', 'changes': {'a': {'before': None, 'after': [1, 2]}}}
    revision['hash'] = hashlib.sha256(json.dumps(revision, sort_keys=True, default=str).encode()).hexdigest()
    (tmp_path / 'orchestration_chain.json').write_text(json.dumps(
        {'schema_version': 1, 'owner': OWNER, 'current': {}, 'revisions': [revision]}))
    chain.sync_chain(tmp_path, line, runtime, 'migration')
    hydrated = chain.read_chain_revision(tmp_path, OWNER, 1)
    assert hydrated == revision


def test_failed_evidence_publication_does_not_commit_manifest(tmp_path, monkeypatch):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    before = {name: (tmp_path / name).read_bytes() for name in
              ('orchestration_chain.json', 'orchestration_current.json')}
    runtime['nodes']['compute']['status'] = 'succeeded'
    def failed(*args, **kwargs):
        raise OSError('injected disk failure')
    monkeypatch.setattr(chain, 'publish', failed)
    with pytest.raises(OSError):
        chain.sync_chain(tmp_path, line, runtime, 'completed')
    assert {name: (tmp_path / name).read_bytes() for name in before} == before


def test_evidence_symlink_is_never_followed(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    data = json.loads((tmp_path / 'orchestration_chain.json').read_text())
    ref = data['revisions'][0]['changes_ref']
    target = tmp_path / ref['path']
    saved = tmp_path / 'saved-evidence.json'
    target.rename(saved)
    target.symlink_to(saved)
    with pytest.raises((OSError, PermissionError)):
        chain.read_chain_reference(tmp_path, OWNER, ref)


def test_revision_references_are_hash_linked(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    runtime['nodes']['compute']['status'] = 'succeeded'
    chain.sync_chain(tmp_path, line, runtime, 'completed')
    data = json.loads((tmp_path / 'orchestration_chain.json').read_text())
    parent = None
    for revision in data['revisions']:
        material = {k: v for k, v in revision.items() if k != 'hash'}
        assert revision['parent_hash'] == parent
        assert revision['hash'] == hashlib.sha256(json.dumps(material, sort_keys=True, default=str).encode()).hexdigest()
        parent = revision['hash']


def test_history_reference_removal_is_recorded_in_audit(tmp_path):
    line, runtime = fixture()
    runtime['nodes']['compute']['attempt_history'] = [{'result': 'x' * 20000}]
    runtime['plan_history'] = [{'result': 'x' * 20000}]
    chain.sync_chain(tmp_path, line, runtime, 'large')
    runtime['plan_history'] = []
    chain.sync_chain(tmp_path, line, runtime, 'cleared')
    changes = chain.read_chain_revision(tmp_path, OWNER, 2)['changes']
    assert changes['execution_history_ref']['before']['sha256']
    assert changes['execution_history_ref']['after_present'] is False


def test_stale_plan_cannot_replace_last_complete_graph_with_old_nodes(tmp_path):
    line, runtime = fixture()
    revised = copy.deepcopy(line)
    revised['plan_version'] = 3
    revised['steps'].append({**copy.deepcopy(revised['steps'][0]), 'step_id': 'report'})
    chain.sync_chain(tmp_path, revised, runtime, 'new')
    chain.sync_chain(tmp_path, line, runtime, 'stale')
    data = json.loads((tmp_path / 'orchestration_chain.json').read_text())
    assert [n['step_id'] for n in data['last_complete_graph']['nodes']] == ['compute', 'report']
