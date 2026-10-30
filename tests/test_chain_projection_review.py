"""Independent boundary regressions for immutable chain references/projections."""
import json
from pathlib import Path

import pytest

from agents import chain_evidence
from agents import orchestration_chain as chain
from tests.test_orchestration_chain import fixture

OWNER = {'username': 'u', 'conv_id': 'c'}


def test_malformed_sidecar_graph_falls_back_to_manifest(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    sidecar = tmp_path / 'orchestration_current.json'
    projection = json.loads(sidecar.read_text())
    projection['graph'] = 'not a graph object'
    sidecar.write_text(json.dumps(projection))
    graph = chain.read_last_graph(tmp_path, OWNER, strict=True)
    assert isinstance(graph, dict)
    assert graph['nodes'][0]['step_id'] == 'compute'


def test_symlink_sidecar_is_not_accepted_as_owned_projection(tmp_path):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    original_status = chain.read_last_graph(tmp_path, OWNER, strict=True)['nodes'][0]['status']
    assert original_status == runtime['nodes']['compute']['status']
    sidecar = tmp_path / 'orchestration_current.json'
    projection = json.loads(sidecar.read_text())
    projection['graph']['nodes'][0]['status'] = 'unverified-external-cache'
    external = tmp_path / 'external-projection.json'
    external.write_text(json.dumps(projection))
    sidecar.unlink()
    sidecar.symlink_to(external)
    graph = chain.read_last_graph(tmp_path, OWNER, strict=True)
    assert graph['nodes'][0]['status'] == original_status


def test_reference_read_rejects_directory_replaced_with_symlink_after_precheck(tmp_path, monkeypatch):
    root, outside = tmp_path / 'session', tmp_path / 'outside'
    outside.mkdir()
    reference = chain_evidence.publish(root, OWNER, 'review', {'scientific_fact': 42})
    evidence = root / reference['path']
    (outside / evidence.name).write_bytes(evidence.read_bytes())
    original = chain_evidence._directory
    def swap_after_check(actual_root):
        directory = original(actual_root)
        directory.rename(root / 'parked-evidence')
        directory.symlink_to(outside, target_is_directory=True)
        return directory
    monkeypatch.setattr(chain_evidence, '_directory', swap_after_check)
    with pytest.raises((PermissionError, OSError)):
        chain_evidence.read_reference(root, OWNER, reference)


def test_reference_publish_rejects_directory_replaced_with_symlink_after_precheck(tmp_path, monkeypatch):
    root, outside = tmp_path / 'session', tmp_path / 'outside'
    root.mkdir(); outside.mkdir(); (root / 'chain_evidence').mkdir()
    original = chain_evidence._directory
    def swap_after_check(actual_root):
        directory = original(actual_root)
        directory.rename(root / 'parked-evidence')
        directory.symlink_to(outside, target_is_directory=True)
        return directory
    monkeypatch.setattr(chain_evidence, '_directory', swap_after_check)
    with pytest.raises((PermissionError, OSError)):
        chain_evidence.publish(root, OWNER, 'review', {'scientific_fact': 42})
    assert list(outside.iterdir()) == []


def test_sidecar_publication_failure_does_not_hide_new_manifest_or_scientific_contract(tmp_path, monkeypatch):
    line, runtime = fixture()
    chain.sync_chain(tmp_path, line, runtime, 'initial')
    original_graph = chain.read_last_graph(tmp_path, OWNER, strict=True)
    def fail_projection(*args, **kwargs):
        raise OSError('injected sidecar publication failure')
    monkeypatch.setattr(chain, 'write_checkpoint', fail_projection)
    runtime['nodes']['compute']['status'] = 'succeeded'
    with pytest.raises(OSError, match='sidecar publication'):
        chain.sync_chain(tmp_path, line, runtime, 'complete')
    graph = chain.read_last_graph(tmp_path, OWNER, strict=True)
    assert graph['nodes'][0]['status'] == 'succeeded'
    assert graph['nodes'][0]['arguments'] == original_graph['nodes'][0]['arguments']
    assert graph['nodes'][0]['expected_outputs'] == original_graph['nodes'][0]['expected_outputs']
