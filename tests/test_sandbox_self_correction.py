from pathlib import Path
import hashlib
import pytest
from agents import sandbox_self_correction as module
from tests.test_parallel_workflow import factory, node, eventually


def failure():
    return {'ok': False, 'status': 'execution_failed', 'error_kind': 'data_execution_failed',
            'exit_code': 1, 'sandbox': {'os_isolated': True}, 'output_dir': '/attempt/one', 'inputs': []}


def test_stderr_is_returned_to_same_correction_flow_then_succeeds():
    calls = []
    def execute(index, script):
        calls.append((index, script))
        return failure() if index == 0 else {'ok': True, 'status': 'data_ready_needs_review', 'inputs': []}
    def correct(data):
        assert 'ModuleNotFoundError' in data['stderr']
        assert data['stderr_is_untrusted'] and data['preserve_inputs_and_output_contract']
        return {'script': 'print(2)'}
    result = module.run_self_correction('print(1)', execute_attempt=execute,
        request_correction=correct, control_valid=lambda: True,
        read_stderr=lambda _: 'ModuleNotFoundError: missing library', verify_inputs=lambda _: True)
    assert result['ok'] and calls == [(0, 'print(1)'), (1, 'print(2)')]
    assert [item['attempt'] for item in result['attempts']] == [1, 2]


@pytest.mark.parametrize('changed', [
    {'error_kind': 'data_sandbox_unavailable'}, {'sandbox': {'os_isolated': False}},
    {'timed_out': True}, {'exit_code': None}, {'exit_code': 0},
    {'error_kind': 'plot_source_changed'},
])
def test_uncertain_or_unsafe_failures_never_invoke_model(changed):
    raw = failure(); raw.update(changed)
    result = module.run_self_correction('print(1)', execute_attempt=lambda *_: raw,
        request_correction=lambda _: pytest.fail('unsafe correction'),
        control_valid=lambda: True, read_stderr=lambda _: 'error', verify_inputs=lambda _: True)
    assert not result['ok'] and len(result['attempts']) == 1


def test_revoked_control_never_dispatches_again():
    checks = iter([True, True, False])
    calls = []
    with pytest.raises(PermissionError):
        module.run_self_correction('print(1)', execute_attempt=lambda *x: calls.append(x) or failure(),
            request_correction=lambda _: {'script': 'print(2)'},
            control_valid=lambda: next(checks), read_stderr=lambda _: 'error', verify_inputs=lambda _: True)
    assert len(calls) == 1


def test_output_contract_cannot_be_changed_by_correction():
    with pytest.raises(ValueError):
        module.run_self_correction('print(1)', execute_attempt=lambda *_: failure(),
            request_correction=lambda _: {'script': 'print(2)', 'output_dir': '/other'},
            control_valid=lambda: True, read_stderr=lambda _: 'error', verify_inputs=lambda _: True)


def test_successful_attempt_is_published_at_approved_path_without_overwrite(tmp_path):
    root = tmp_path / 'owner' / 'conversation'; root.mkdir(parents=True)
    source = root / 'input.txt'; source.write_text('source')
    attempt = root / 'attempt'; attempt.mkdir()
    artifact = attempt / 'result.txt'; artifact.write_text('result')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'ok': True, 'status': 'data_ready_needs_review', 'output_dir': str(attempt),
        'inputs': [{'source_path': str(source), 'sha256': digest(source), 'bytes': source.stat().st_size}],
        'artifacts': [{'name': 'result.txt', 'path': str(artifact), 'sha256': digest(artifact),
                       'bytes': artifact.stat().st_size}], 'scientific_acceptance': False}
    out = root / 'delivery'
    receipt = module.publish_corrected_result(root, out, result,
        [{'name': 'result.txt', 'format': 'txt'}], [{'attempt': 1}], lambda: True, lambda: True, [])
    assert (out / 'artifacts' / 'result.txt').read_text() == 'result'
    assert receipt['artifacts'][0]['path'] == str(out / 'artifacts' / 'result.txt')
    with pytest.raises(FileExistsError):
        module.publish_corrected_result(root, out, result,
            [{'name': 'result.txt', 'format': 'txt'}], [], lambda: True, lambda: True, [])
    assert (out / 'artifacts' / 'result.txt').read_text() == 'result'


def test_source_changed_after_attempt_cannot_publish(tmp_path):
    root = tmp_path / 'owner' / 'conversation'; root.mkdir(parents=True)
    source = root / 'input.txt'; source.write_text('changed')
    attempt = root / 'attempt'; attempt.mkdir()
    artifact = attempt / 'result.txt'; artifact.write_text('result')
    result = {'ok': True, 'inputs': [{'source_path': str(source), 'sha256': '0' * 64, 'bytes': 7}],
        'artifacts': [{'name': 'result.txt', 'path': str(artifact),
                       'sha256': hashlib.sha256(artifact.read_bytes()).hexdigest(), 'bytes': 6}]}
    with pytest.raises(ValueError, match='source changed'):
        module.publish_corrected_result(root, root / 'delivery', result,
            [{'name': 'result.txt', 'format': 'txt'}], [], lambda: True, lambda: True, [])
    assert not (root / 'delivery').exists()


def test_cancel_racing_publication_keeps_files_but_never_emits_success(tmp_path, monkeypatch):
    from agents import plot_sandbox
    root = tmp_path / 'owner' / 'conversation'; root.mkdir(parents=True)
    source = root / 'input.txt'; source.write_text('source')
    attempt = root / 'attempt'; attempt.mkdir()
    artifact = attempt / 'result.txt'; artifact.write_text('result')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    result = {'ok': True, 'inputs': [{'source_path': str(source), 'sha256': digest(source), 'bytes': 6}],
        'artifacts': [{'name': 'result.txt', 'path': str(artifact), 'sha256': digest(artifact), 'bytes': 6}]}
    active = [True]
    original = plot_sandbox._publish_data_outputs
    def publish(*args):
        answer = original(*args)
        active[0] = False
        return answer
    monkeypatch.setattr(plot_sandbox, '_publish_data_outputs', publish)
    out = root / 'delivery'
    with pytest.raises(PermissionError, match='authorization changed'):
        module.publish_corrected_result(root, out, result,
            [{'name': 'result.txt', 'format': 'txt'}], [], lambda: active[0], lambda: True, [])
    assert (out / 'artifacts/result.txt').read_text() == 'result'
    assert not (out / 'execution.json').exists()


@pytest.mark.parametrize('source_changed', [False, True])
def test_native_dag_returns_stderr_to_assigned_actor_and_publishes(factory, tmp_path, monkeypatch, source_changed):
    from types import SimpleNamespace
    from agents import registry, workspace
    from agents.report_authoring_session import ReportStageSession
    root = tmp_path / 'runs/tester/correct-data'; source = root / 'source.csv'
    out = root / 'derived'; artifact = out / 'artifacts/result.json'
    producer = node('analyze', tool='run_analysis_sandbox', args={
        'script': 'import absent_test_package_xyz',
        'input_files': [{'path': str(source), 'input_id': 'T1', 'format': 'csv'}],
        'output_dir': str(out), 'declared_outputs': [{'name': 'result.json', 'format': 'json'}]},
        outputs=[str(artifact)])
    rt = factory([producer], cid='correct-data')
    monkeypatch.setattr(registry, 'get_config', lambda: rt.main.config)
    monkeypatch.setattr(workspace, 'get_config', lambda: rt.main.config)
    source.write_text('x\n1\n')
    calls = []
    def model(actor, role):
        request = actor._report_stage_request
        assert 'ModuleNotFoundError' in str(request['messages'])
        assert 'absent_test_package_xyz' in str(request['messages'])
        calls.append(role.name)
        if source_changed:source.write_text('x\n9\n')
        script = "import json,pathlib\npathlib.Path('/output/result.json').write_text(json.dumps({'value':1}))"
        return SimpleNamespace(id='model-test', content=[SimpleNamespace(type='tool_use',
            id='test-call-1', name='return_report_stage_data', input={'correction': {'script': script}})])
    monkeypatch.setattr(ReportStageSession, '_call_api', model)
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['analyze']['status'] in {'succeeded', 'failed', 'uncertain'}, timeout=30)
    state = rt.snapshot()['nodes']['analyze']
    if source_changed:
        assert state['status'] == 'failed' and 'source changed' in state['error']
        assert calls == ['analyst'] and not artifact.exists()
        return
    assert state['status'] == 'succeeded', state
    assert calls == ['analyst']
    assert artifact.read_text() == '{"value": 1}'
    assert len(state['result']['correction_attempts']) == 2
    assert state['result']['scientific_acceptance'] is False


def test_multiple_corrections_reuse_actor_and_pair_tool_result(factory, tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agents import registry, workspace
    from agents.report_authoring_session import ReportStageSession
    root = tmp_path / 'runs/tester/multi-correct'; source = root / 'source.csv'
    out = root / 'derived'; artifact = out / 'artifacts/result.json'
    producer = node('analyze', tool='run_analysis_sandbox', args={
        'script': 'import absent_first_xyz',
        'input_files': [{'path': str(source), 'input_id': 'T1', 'format': 'csv'}],
        'output_dir': str(out), 'declared_outputs': [{'name': 'result.json', 'format': 'json'}]},
        outputs=[str(artifact)])
    rt = factory([producer], cid='multi-correct')
    monkeypatch.setattr(registry, 'get_config', lambda: rt.main.config)
    monkeypatch.setattr(workspace, 'get_config', lambda: rt.main.config)
    source.write_text('x\n1\n'); actors = []
    def model(actor, role):
        actors.append(id(actor)); messages = actor._report_stage_request['messages']
        if len(actors) == 1:
            assert 'absent_first_xyz' in str(messages)
            script = 'import absent_second_xyz'
        else:
            assert 'absent_first_xyz' in str(messages) and 'absent_second_xyz' in str(messages)
            assert messages[-1]['content'][0]['type'] == 'tool_result'
            assert messages[-1]['content'][0]['tool_use_id'] == 'test-call-1'
            script = "open('/output/result.json','w').write('{\"value\":1}')"
        return SimpleNamespace(id='model-test', content=[SimpleNamespace(type='tool_use',
            id='test-call-' + str(len(actors)), name='return_report_stage_data',
            input={'correction': {'script': script}})])
    monkeypatch.setattr(ReportStageSession, '_call_api', model)
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['analyze']['status'] in {'succeeded', 'failed', 'uncertain'}, timeout=30)
    state = rt.snapshot()['nodes']['analyze']
    assert state['status'] == 'succeeded', state['status']
    assert len(actors) == 2 and actors[0] == actors[1]
    assert len(state['result']['correction_attempts']) == 3 and artifact.is_file()


def test_loaded_implementation_change_blocks_publication(factory, tmp_path, monkeypatch):
    from agents import registry, workspace, executor_identity
    root = tmp_path / 'runs/tester/code-change'; source = root / 'source.csv'
    out = root / 'derived'; artifact = out / 'artifacts/result.json'
    producer = node('analyze', tool='run_analysis_sandbox', args={
        'script': "open('/output/result.json','w').write('{}')",
        'input_files': [{'path': str(source), 'input_id': 'T1', 'format': 'csv'}],
        'output_dir': str(out), 'declared_outputs': [{'name': 'result.json', 'format': 'json'}]},
        outputs=[str(artifact)])
    rt = factory([producer], cid='code-change')
    monkeypatch.setattr(registry, 'get_config', lambda: rt.main.config)
    monkeypatch.setattr(workspace, 'get_config', lambda: rt.main.config)
    calls = []
    def identity(function):
        calls.append(function.__name__)
        return {'verified': True, 'matches': True, 'qualified_name': function.__name__,
                'source_sha256': 'a' if len(calls) <= 7 else 'b',
                'loaded_code_sha256': 'c', 'source_code_sha256': 'c'}
    monkeypatch.setattr(executor_identity, 'executor_identity', identity)
    source.write_text('x\n1\n')
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['analyze']['status'] in {'succeeded', 'failed', 'uncertain'}, timeout=30)
    state = rt.snapshot()['nodes']['analyze']
    assert state['status'] == 'uncertain' and 'implementation changed' in state['error']
    assert not artifact.exists()


def test_provider_failure_is_terminal_known_sandbox_failure_not_uncertain(factory, tmp_path, monkeypatch):
    from agents import registry, workspace
    from agents.report_authoring_session import ReportStageSession
    root = tmp_path / 'runs/tester/no-provider'; source = root / 'source.csv'
    out = root / 'derived'; artifact = out / 'artifacts/result.json'
    producer = node('analyze', tool='run_analysis_sandbox', args={
        'script': 'import absent_test_package_xyz',
        'input_files': [{'path': str(source), 'input_id': 'T1', 'format': 'csv'}],
        'output_dir': str(out), 'declared_outputs': [{'name': 'result.json', 'format': 'json'}]},
        outputs=[str(artifact)])
    rt = factory([producer], cid='no-provider')
    monkeypatch.setattr(registry, 'get_config', lambda: rt.main.config)
    monkeypatch.setattr(workspace, 'get_config', lambda: rt.main.config)
    def failed_provider(*_):raise RuntimeError('provider-error-private-marker')
    monkeypatch.setattr(ReportStageSession, '_call_api', failed_provider)
    source.write_text('x\n1\n')
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['analyze']['status'] in {'succeeded', 'failed', 'uncertain'}, timeout=30)
    state = rt.snapshot()['nodes']['analyze']
    assert state['status'] == 'failed', state['status']
    assert state['result']['error_kind'] == 'sandbox_model_correction_unavailable'
    assert 'provider-error-private-marker' not in str(state['result'])
    assert not artifact.exists()
