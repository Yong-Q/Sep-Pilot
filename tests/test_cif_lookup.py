"""Portable, bounded input discovery; no scientific executions or site data."""
from pathlib import Path

import pytest

from agents.cif_lookup import find_cif


@pytest.fixture(autouse=True)
def no_site_dataset(monkeypatch):
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', lambda _: {})


def cif(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('data_framework\n')
    return path


def test_project_dataset_works_without_private_configuration(tmp_path):
    source = cif(tmp_path / 'cifs/FRAME-A.CIF')
    result = find_cif('frame a', tmp_path, 'alice', 'one')
    assert result['status'] == 'found' and result['read_only']
    assert result['cif_path'] == str(source)
    assert result['search_complete']


def test_production_registry_and_lead_can_resolve_before_planning(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agents import registry
    from agents.config import AgentConfig
    from agents.defns import ORCHESTRATOR
    from agents.session import Session
    from agents.watch_context import set_context, clear_context
    source = cif(tmp_path / 'runs/alice/one/uploads/framework.cif')
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path))
    monkeypatch.setattr(registry, '_lazy', lambda *a, **kw: pytest.fail('legacy helper must not be imported'))
    set_context('alice', 'one', 'lead-orchestrator', 'one')
    try:
        result = registry.get_registry().execute_dict('find_cif', {'name': 'framework'})
        assert result['cif_path'] == str(source)
        session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
        session._current_line_id = 'one';session._on_workflow_start = lambda *_: None
        assert 'find_cif' in ORCHESTRATOR.function_map()
        assert not session._chain_required_block('find_cif')
    finally:
        clear_context()


def test_own_session_uploads_and_explicit_project_paths(tmp_path):
    source = cif(tmp_path / 'runs/alice/one/uploads/framework.cif')
    assert find_cif('framework', tmp_path, 'alice', 'one')['cif_path'] == str(source)
    assert find_cif(str(source), tmp_path, 'alice', 'one')['cif_path'] == str(source)
    shared = cif(tmp_path / 'cifs/other.cif')
    assert find_cif('cifs/other.cif', tmp_path, 'alice', 'one')['cif_path'] == str(shared)


def test_operator_configured_dataset_is_optional_and_shared(tmp_path, monkeypatch):
    project = tmp_path / 'project'
    dataset = tmp_path / 'shared-inputs'
    source = cif(dataset / 'database/frame_b.cif')
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', lambda _: {'cif_dir': str(dataset)})
    monkeypatch.setattr('bimem_agent.paths.get_config_path', lambda *_: dataset)
    result = find_cif('FRAME-B', project, 'alice', 'one')
    assert result['cif_path'] == str(source)


def test_overlapping_configured_root_does_not_duplicate_one_file(tmp_path, monkeypatch):
    source = cif(tmp_path / 'cifs/frame.cif')
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', lambda _: {'cif_dir': str(tmp_path / 'cifs')})
    monkeypatch.setattr('bimem_agent.paths.get_config_path', lambda *_: tmp_path / 'cifs' / '..' / 'cifs')
    assert find_cif('frame', tmp_path, 'alice', 'one')['candidates'] == [str(source)]


def test_duplicate_normalized_names_are_ambiguous_without_picking_a_version(tmp_path):
    original = cif(tmp_path / 'cifs/frame-a.cif')
    duplicate = cif(tmp_path / 'runs/alice/one/uploads/frame_a.cif')
    cif(tmp_path / 'cifs/frame-a_pacmof.cif')
    result = find_cif('frame a', tmp_path, 'alice', 'one')
    assert result['status'] == 'ambiguous' and not result['ok']
    assert 'cif_path' not in result
    assert set(result['candidates']) == {str(original), str(duplicate)}


def test_charged_or_expanded_variants_are_not_silently_selected(tmp_path):
    cif(tmp_path / 'cifs/frame-a_charged.cif')
    cif(tmp_path / 'cifs/frame-a_2x2x2.cif')
    result = find_cif('frame-a', tmp_path, 'alice', 'one')
    assert result['status'] == 'not_found' and not result['candidates']


def test_substrings_and_globs_are_not_exact_material_identities(tmp_path):
    cif(tmp_path / 'cifs/frame.cif')
    assert find_cif('fram', tmp_path, 'alice', 'one')['status'] == 'not_found'
    assert find_cif('frame*', tmp_path, 'alice', 'one')['status'] == 'not_found'


def test_other_sessions_are_not_searched_or_returned(tmp_path, monkeypatch):
    forbidden = cif(tmp_path / 'runs/bob/two/uploads/private.cif')
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', lambda _: {'cif_dir': str(tmp_path / 'runs')})
    monkeypatch.setattr('bimem_agent.paths.get_config_path', lambda *_: tmp_path / 'runs')
    result = find_cif('private', tmp_path, 'alice', 'one')
    assert result['status'] == 'not_found' and result['candidates'] == []
    explicit = find_cif(str(forbidden), tmp_path, 'alice', 'one')
    assert explicit['status'] == 'forbidden' and 'cif_path' not in explicit


@pytest.mark.parametrize('link_kind', ['file', 'directory', 'session_root'])
def test_symlinks_cannot_leak_another_session(tmp_path, link_kind):
    secret = cif(tmp_path / 'runs/bob/two/private.cif')
    shared = tmp_path / 'cifs'
    shared.mkdir()
    if link_kind == 'file':
        (shared / 'private.cif').symlink_to(secret)
    elif link_kind == 'directory':
        (shared / 'linked').symlink_to(secret.parent, target_is_directory=True)
    else:
        own = tmp_path / 'runs/alice'
        own.mkdir()
        (own / 'one').symlink_to(secret.parent, target_is_directory=True)
    result = find_cif('private', tmp_path, 'alice', 'one')
    assert not result.get('cif_path') and result['candidates'] == []
    assert str(secret) not in str(result)


def test_explicit_cif_path_must_stay_in_declared_input_roots(tmp_path):
    source = cif(tmp_path / 'unregistered/frame.cif')
    result = find_cif(str(source), tmp_path, 'alice', 'one')
    assert result['status'] == 'forbidden' and 'cif_path' not in result


def test_workspace_scope_guard_checks_candidates_before_returning(tmp_path, monkeypatch):
    source = cif(tmp_path / 'cifs/frame.cif')
    checked = []
    def deny(params, username, conv_id, project_root):
        checked.append(params)
        return ['blocked by workspace scope'] if params.get('cif_path') == str(source) else []
    monkeypatch.setattr('agents.workspace.tool_scope_issues', deny)
    result = find_cif('frame', tmp_path, 'alice', 'one')
    assert result['status'] == 'not_found' and not result.get('cif_path')
    assert {'cif_path': str(source)} in checked


def test_search_limit_never_claims_unique_result_from_incomplete_search(tmp_path, monkeypatch):
    source = cif(tmp_path / 'cifs/frame.cif')
    for index in range(4):
        cif(tmp_path / f'cifs/other-{index}.cif')
    monkeypatch.setattr('agents.cif_lookup.MAX_SEARCH_ENTRIES', 2)
    result = find_cif('frame', tmp_path, 'alice', 'one')
    assert result['status'] == 'search_limit' and not result['search_complete']
    assert result['scanned_entries'] <= 2
    assert 'cif_path' not in result


def test_unscoped_lookup_does_not_search_private_sessions(tmp_path):
    cif(tmp_path / 'runs/alice/one/uploads/private.cif')
    assert find_cif('private', tmp_path)['status'] == 'not_found'


def test_invalid_scope_is_structured_and_does_not_escape(tmp_path):
    result = find_cif('frame', tmp_path, '../bob', 'one')
    assert result['status'] == 'forbidden' and 'cif_path' not in result


def test_invalid_explicit_path_is_structured(tmp_path):
    assert find_cif('cifs/bad\x00.cif', tmp_path, 'alice', 'one')['status'] == 'forbidden'


def test_invalid_operator_configuration_is_not_silently_ignored(tmp_path, monkeypatch):
    def broken(_):
        raise ValueError('invalid configured path')
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', broken)
    result = find_cif('frame', tmp_path, 'alice', 'one')
    assert result['status'] == 'configuration_error' and not result['search_complete']


def test_general_agents_can_discover_inputs_without_a_compute_handoff():
    from agents.defns import ORCHESTRATOR
    assert 'find_cif' in ORCHESTRATOR.function_map()
