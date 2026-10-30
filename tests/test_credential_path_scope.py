"""Fake credentials only: scoped file tools must reject before file reads."""
import json
from pathlib import Path

import pytest

from agents import registry
from agents.config import AgentConfig
from agents.watch_context import set_context


@pytest.fixture
def credential_tree(tmp_path, monkeypatch):
    project = tmp_path / 'project'; project.mkdir()
    home = tmp_path / 'home'; home.mkdir()
    state = tmp_path / 'private-account-state'; state.mkdir()
    codex = home / 'custom-codex-home'; codex.mkdir()
    monkeypatch.setenv('CODEX_HOME', str(codex))
    monkeypatch.setenv('BIMEM_AUTH_STATE_DIR', str(state))
    monkeypatch.setattr(Path, 'home', classmethod(lambda cls: home))
    config = AgentConfig(api_key='test', project_root=project)
    monkeypatch.setattr('agents.config._config', config)
    set_context('alice', 'conversation', 'lead-orchestrator', '')
    return project, home, state, codex


@pytest.mark.parametrize('target', ['default_auth', 'configured_auth', 'configured_child', 'users', 'tokens',
                                  'credential_key', 'claude_settings', 'claude_local_settings', 'symlink',
                                  'relative_codex', 'relative_state', 'auth_db_alias'])
@pytest.mark.parametrize('tool', ['read_file', 'grep_search'])
def test_native_read_blocks_fake_credentials_before_open(credential_tree, monkeypatch, target, tool):
    project, home, state, codex = credential_tree
    paths = {'default_auth': home / '.codex/auth.json', 'configured_auth': codex / 'auth.json',
        'configured_child': codex / 'nested/private.json', 'users': state / 'users.json',
        'tokens': state / 'tokens.json', 'credential_key': state / 'data/state/credential_master.key',
        'claude_settings': home / '.claude/settings.json',
        'claude_local_settings': home / '.claude/settings.local.json',
        'relative_codex': codex / 'config.toml', 'relative_state': state / 'users.json',
        'auth_db_alias': home / 'private-account-db.json'}
    secret = paths.get(target, codex / 'auth.json')
    secret.parent.mkdir(parents=True, exist_ok=True)
    secret.write_text('SYNTHETIC_PRIVATE_MARKER')
    if target == 'auth_db_alias':
        (state / 'users.json').symlink_to(secret)
    path = secret
    if target == 'symlink':
        path = project / 'apparently-public.json'; path.symlink_to(secret)
    elif target == 'relative_codex':
        path = secret.relative_to(home)
    elif target == 'relative_state':
        path = secret.relative_to(project.parent)
    original = Path.read_text
    reads = []
    def intercepted(file, *args, **kwargs):
        if file.resolve() == secret.resolve():
            reads.append(str(file))
            return 'SYNTHETIC_PRIVATE_MARKER'
        return original(file, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', intercepted)
    searches = []
    monkeypatch.setattr('subprocess.run', lambda *a, **k: searches.append(a) or pytest.fail('credential grep must not start'))
    arguments = {'path': str(path)}
    if tool == 'grep_search': arguments['pattern'] = 'SYNTHETIC_PRIVATE_MARKER'
    result = registry.get_registry().execute_dict(tool, arguments)
    assert reads == []
    assert searches == []
    assert result.get('error')
    assert 'SYNTHETIC_PRIVATE_MARKER' not in json.dumps(result)


@pytest.mark.parametrize('layout', ['ordinary', 'database_alias', 'codex_glob_characters'])
def test_recursive_grep_skips_credentials_but_keeps_source_skills_and_cif(credential_tree, monkeypatch, layout):
    project, home, state, codex = credential_tree
    if layout == 'codex_glob_characters':
        codex = home / 'codex[private]?*'
        monkeypatch.setenv('CODEX_HOME', str(codex))
    secret_paths = [codex / 'auth.json', home / '.codex/auth.json', home / '.claude/settings.json',
                    state / 'users.json', state / 'tokens.json', state / 'data/state/credential_master.key']
    for path in secret_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if layout == 'database_alias' and path == state / 'users.json':
            target = home / 'private-account-db.json'
            target.write_text('MATCH SYNTHETIC_PRIVATE_MARKER'); path.symlink_to(target)
        else:
            path.write_text('MATCH SYNTHETIC_PRIVATE_MARKER')
    public_paths = [project / 'module.py', project / '.claude/skills/example/SKILL.md', home / 'datasets/public.cif']
    for path in public_paths:
        path.parent.mkdir(parents=True, exist_ok=True); path.write_text('MATCH PUBLIC_SCIENCE_MARKER')
    result = registry.get_registry().execute_dict('grep_search', {'path': str(project.parent), 'pattern': 'MATCH'})
    serialized = json.dumps(result)
    assert 'SYNTHETIC_PRIVATE_MARKER' not in serialized
    assert result['total_matches'] == len(public_paths)
    for path in public_paths:
        assert str(path) in serialized
        assert registry.get_registry().execute_dict('read_file', {'path': str(path)})['content'] == 'MATCH PUBLIC_SCIENCE_MARKER'
