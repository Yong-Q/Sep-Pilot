import sys
from pathlib import Path
from types import SimpleNamespace

from agents import registry


def test_installation_profile_preserves_family_and_method_but_ignores_workload():
    from agents.installations import installation_profile
    assert installation_profile('generate_structure', {'material_type': 'MOF'}) == installation_profile(
        'generate_structure', {'material_type': 'u-MOF', 'n_structures': 50, 'memory_mb': 8192})
    assert installation_profile('generate_structure', {'material_type': 'HOF'}) != installation_profile('generate_structure', {})
    assert installation_profile('run_pacman_charge', {'method': 'pacman'}) != installation_profile('run_pacman_charge', {})


def test_recovery_state_exposes_real_changed_executor_as_read_only_capability(tmp_path):
    from agents.config import AgentConfig
    from agents.session import Session
    from agents.recovery import input_fingerprint
    source = tmp_path / 'agents/registry.py';source.parent.mkdir();source.write_text('old = True\n')
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._pending_jobs_for_conv = lambda: []
    args = {'material_type': 'MOF', 'n_structures': 3, 'output_dir': str(tmp_path / 'output')}
    aid, _ = s.recovery_gate.claim('key', 'generate_structure', args, input_fingerprint(args, tmp_path), project_root=tmp_path)
    s.recovery_gate.outcome('key', aid, {'error': 'failed'})
    assert 'verification_capability' not in s._recovery_state()['recovery_guidance']['key']
    source.write_text('repaired = True\n')
    guidance = s._recovery_state()['recovery_guidance']['key']
    assert guidance['executor_change_observed']
    assert guidance['verification_capability']['arguments'] == {'tool_name': 'generate_structure', 'arguments': args}
    assert s.recovery_gate.snapshot()['key']['approved_fingerprint'] is None


def test_executor_digest_tracks_selected_site_config(tmp_path, monkeypatch):
    from agents.installations import executor_fingerprint
    config = tmp_path / 'site.json'
    config.write_text('{"software": {}}')
    monkeypatch.setenv('BIMEM_RUNTIME_CONFIG', str(config))
    before = executor_fingerprint('generate_structure', tmp_path / 'repo')
    config.write_text('{"software": {"python": "changed"}}')
    assert executor_fingerprint('generate_structure', tmp_path / 'repo') != before


def test_diagnosis_restores_owned_terminal_receipt_after_scheduler_purge(tmp_path, monkeypatch):
    from agents.watch_context import set_context, clear_context
    logs = tmp_path / 'logs'
    logs.mkdir()
    (logs / 'gen_42.err').write_text('FileNotFoundError: missing database')
    record = {'job_id': '42', 'username': 'alice', 'conv_id': 'conv', 'work_dir': str(tmp_path),
              'terminal': True, 'failed': True, 'state': 'FAILED'}
    watch = SimpleNamespace(get=lambda _: record, mark_diagnosed=lambda _: True)
    monkeypatch.setattr('agents.job_watch.get_watch', lambda: watch)
    monkeypatch.setattr(registry.slurm, 'diagnose_job', lambda *a, **kw: {'state': 'UNKNOWN', 'terminal': False, 'error': 'Invalid job id'})
    set_context('alice', 'conv', 'lead-orchestrator', '')
    try:
        result = registry._exec_diagnose_job({'job_id': '42'})
        assert result['terminal'] and result['failed']
        assert 'missing database' in result['stderr_tail']
        assert result['state_source'] == 'durable_jobwatch_receipt'
        record['username'] = 'bob'
        assert registry._exec_diagnose_job({'job_id': '42'})['blocked']
    finally:
        clear_context()


def test_native_inspection_checks_installation_without_submitting(tmp_path, monkeypatch):
    import bimem_agent.paths as paths
    executable = tmp_path / 'python'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    database = tmp_path / 'database'
    (database / 'bbs_MOF').mkdir(parents=True)
    monkeypatch.setattr(paths, 'get_runtime_section', lambda _: {'pormake_database_dir': str(database)})
    monkeypatch.setattr(paths, 'get_config_path', lambda section, key: database if key == 'pormake_database_dir' else executable)
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path, pormake_python=str(executable)))
    for name in ('pormake_generate_topo.py', 'tools/stage_pormake_outputs.py'):
        file = tmp_path / name
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('pass\n')
    result = registry.get_registry().execute_dict('inspect_runtime', {'tool_name': 'generate_structure', 'arguments': {'material_type': 'MOF'}})
    assert result['ok'] and result['read_only']
    assert result['verified_tool'] == 'generate_structure'
    assert result['runtime_fingerprint']
    executable.unlink()
    assert not registry.get_registry().execute_dict('inspect_runtime', {'tool_name': 'generate_structure', 'arguments': {}})['ok']


def test_charge_adapter_uses_configured_interpreter_in_submitted_script(tmp_path, monkeypatch):
    import bimem_agent.paths as paths
    monkeypatch.setattr(paths, 'get_runtime_section', lambda section: {'pacmof_python': str(tmp_path / 'installed-python')})
    monkeypatch.setattr(paths, 'get_config_path', lambda section, key: tmp_path / 'installed-python')
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path))
    captured = {}
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: captured.update(kw) or {'submitted': False})
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    registry._exec_charge({'cif_dir': str(tmp_path / 'cifs'), 'output_dir': str(tmp_path / 'charged')})
    assert str(tmp_path / 'installed-python') in captured['command']
    assert '/home/user/' not in captured['command']


def test_xtb_adapter_uses_configured_executable(tmp_path, monkeypatch):
    import bimem_agent.paths as paths
    monkeypatch.setattr(paths, 'get_runtime_section', lambda section: {'xtb_executable': str(tmp_path / 'xtb')})
    monkeypatch.setattr(paths, 'get_config_path', lambda section, key: tmp_path / 'xtb')
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path))
    captured = {}
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: captured.update(kw) or {'submitted': False})
    registry._exec_xtb({'input_path': str(tmp_path / 'input.xyz'), 'output_dir': str(tmp_path / 'result')})
    assert str(tmp_path / 'xtb') in captured['command']
    assert '/home/user/' not in captured['command']
