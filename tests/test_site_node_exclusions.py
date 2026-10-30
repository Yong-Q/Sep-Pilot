"""Private operator exclusions are additive and project-scoped."""
import json

import pytest

from agents.node_inventory import load_policy


def setup_policy(tmp_path):
    folder = tmp_path / 'env'; folder.mkdir()
    value = {'excluded_nodes': ['login'], 'preferred_nodes': [], 'allowed_states': ['idle', 'mix'],
             'max_age_seconds': 60, 'tool_profiles': {}}
    (folder / 'node_policy.json').write_text(json.dumps(value))
    return value


def test_site_exclusion_is_added_without_changing_public_policy(tmp_path, monkeypatch):
    original = setup_policy(tmp_path)
    from bimem_agent import paths
    monkeypatch.setattr(paths, 'get_runtime_section', lambda _: {
        'scope_root': str(tmp_path), 'excluded_nodes': ['node06']})
    result = load_policy(tmp_path)
    assert result['excluded_nodes'] == ['login', 'node06']
    assert json.loads((tmp_path / 'env/node_policy.json').read_text()) == original


def test_private_policy_for_another_project_does_not_affect_current_workspace(tmp_path, monkeypatch):
    original = setup_policy(tmp_path)
    from bimem_agent import paths
    monkeypatch.setattr(paths, 'get_runtime_section', lambda _: {
        'scope_root': str(tmp_path / 'other'), 'excluded_nodes': ['node06']})
    assert load_policy(tmp_path) == original


@pytest.mark.parametrize('excluded', ['node06', [False], [''], ['node*']])
def test_invalid_operator_exclusions_fail_closed(tmp_path, monkeypatch, excluded):
    setup_policy(tmp_path)
    from bimem_agent import paths
    monkeypatch.setattr(paths, 'get_runtime_section', lambda _: {
        'scope_root': str(tmp_path), 'excluded_nodes': excluded})
    with pytest.raises(ValueError): load_policy(tmp_path)
