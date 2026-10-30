"""Integration of same-actor correction and task-scoped installation."""
import json
from types import SimpleNamespace
from tests.test_parallel_workflow import factory, node, eventually
from tests.test_online_sandbox import fake_index


def test_missing_dependency_is_returned_to_same_actor_and_installed_in_task(factory, tmp_path, monkeypatch):
    from agents import registry, workspace
    from agents.report_authoring_session import ReportStageSession
    root = tmp_path / 'runs/tester/network-correction'; source = root / 'source.csv'
    out = root / 'derived'; artifact = out / 'artifacts/result.json'
    producer = node('analyze', tool='run_analysis_sandbox', args={
        'script': 'import sandboxprobe',
        'input_files': [{'path': str(source), 'input_id': 'T1', 'format': 'csv'}],
        'output_dir': str(out), 'declared_outputs': [{'name': 'result.json', 'format': 'json'}]},
        outputs=[str(artifact)])
    rt = factory([producer], cid='network-correction')
    monkeypatch.setattr(registry, 'get_config', lambda: rt.main.config)
    monkeypatch.setattr(workspace, 'get_config', lambda: rt.main.config)
    source.write_text('x\n1\n'); calls = []
    server, stop, thread = fake_index()
    monkeypatch.setenv('HTTPS_PROXY', 'http://127.0.0.1:' + str(server.getsockname()[1]))
    def model(actor, role):
        request = actor._report_stage_request
        assert 'ModuleNotFoundError' in str(request['messages'])
        assert 'sandbox_network_enabled' in str(request['messages'])
        calls.append(role.name)
        script = """import subprocess,sys,json,pathlib
subprocess.run([sys.executable,'-m','pip','install','--no-deps','--no-cache-dir','--index-url','http://example.invalid/simple/','--trusted-host','example.invalid','sandboxprobe==1.0.0'],check=True)
import sandboxprobe
pathlib.Path('/output/result.json').write_text(json.dumps({'value':sandboxprobe.VALUE}))
"""
        return SimpleNamespace(id='model-test', content=[SimpleNamespace(type='tool_use',
            id='tool-test', name='return_report_stage_data', input={'correction': {'script': script}})])
    monkeypatch.setattr(ReportStageSession, '_call_api', model)
    try:
        rt.start(1); rt.tick()
        eventually(lambda: rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed','uncertain'}, timeout=90)
        state = rt.snapshot()['nodes']['analyze']
        assert state['status'] == 'succeeded', state.get('error')
        assert calls == ['analyst'] and len(state['result']['correction_attempts']) == 2
        assert json.loads(artifact.read_text()) == {'value': 42}
        assert state['result']['sandbox']['network'] is True
        assert state['result']['sandbox']['network_connectivity_verified'] is False
        assert state['result']['scientific_acceptance'] is False
        assert source.read_text() == 'x\n1\n'
    finally:
        stop.set(); server.close(); thread.join(timeout=2)
