import json
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.recovery import RecoveryGate, failure_reason, input_fingerprint, _input_fingerprint
from agents.registry import _exec_run_bash, _exec_submit_job, get_registry
from agents.session import Session
from agents.state_io import write_checkpoint


def test_authoritative_errors_not_quoted_failure_words():
    assert not failure_reason({'ok': True, 'steps': [{'status': 'FAILED'}]})
    assert not failure_reason({'exit_code': 0, 'stdout': 'FAILED in old source'})
    assert not failure_reason({'skipped': True, 'submitted': False})
    assert failure_reason({'exit_code': 2, 'stderr': 'bad input'}) == 'bad input'
    assert failure_reason({'error': 'bad path'})
    assert failure_reason({'submitted': True, 'stdout': '{"submitted": false, "error":"scheduler refused"}'}) == 'scheduler refused'


def test_file_query_and_unattributed_result_never_invent_failure():
    s = Session(config=AgentConfig(api_key='test'))
    content = [ToolUseBlock(type='tool_use', id='query1', name='task_line_query', input={})]
    assert not s._detect_tool_failures([
        {'tool_use_id': 'query1', 'content': json.dumps({'ok': True, 'steps': [{'status': 'FAILED'}]})},
        {'tool_use_id': 'unknown', 'content': '{"error": "quoted source"}'},
    ], content)
    content = [ToolUseBlock(type='tool_use', id='bash1', name='run_bash', input={})]
    assert s._detect_tool_failures([{'tool_use_id': 'bash1', 'content': '{"exit_code": 1, "stderr":"no file"}'}], content) == [('run_bash', 'no file')]


def test_persistent_single_claim_and_uncertain_outcome(tmp_path):
    path = tmp_path / 'ledger.json'
    a, b = RecoveryGate(path=path), RecoveryGate(path=path)
    aid, reason = a.claim('task', 'run_henry', {'gas': 'Kr'}, 'fp1')
    assert aid and not reason
    assert b.claim('task', 'run_henry', {'gas': 'Kr'}, 'fp1')[0] is None
    a.outcome('task', aid, {'error': 'timeout'}, uncertain=True)
    assert b.claim('task', 'run_henry', {'gas': 'Kr'}, 'fp2')[0] is None
    with pytest.raises(ValueError, match='confirmed failed'):
        b.approve('task', 'fp2', {})


def test_verified_executor_repair_allows_same_inputs_but_invalidates_on_further_edit(tmp_path):
    from agents.installations import executor_fingerprint
    adapter = tmp_path / 'agents/registry.py'
    adapter.parent.mkdir()
    adapter.write_text('broken = True\n')
    gate = RecoveryGate()
    aid, _ = gate.claim('task', 'generate_structure', {}, 'inputs', project_root=tmp_path)
    gate.outcome('task', aid, {'error': 'adapter failed'})
    adapter.write_text('broken = False\n')
    proof = {'ok': True, 'read_only': True, 'verified_tool': 'generate_structure', 'checked_arguments': {},
             'runtime_fingerprint': executor_fingerprint('generate_structure', tmp_path)}
    gate.approve('task', 'inputs', {}, runtime_verification=proof)
    adapter.write_text('another = True\n')
    assert gate.claim('task', 'generate_structure', {}, 'inputs', project_root=tmp_path)[0] is None
    adapter.write_text('broken = False\n')
    assert gate.claim('task', 'generate_structure', {}, 'inputs', project_root=tmp_path)[0]


def test_unchanged_runtime_or_wrong_tool_cannot_approve_same_inputs(tmp_path):
    from agents.installations import executor_fingerprint
    gate = RecoveryGate()
    aid, _ = gate.claim('task', 'generate_structure', {}, 'inputs', project_root=tmp_path)
    gate.outcome('task', aid, {'error': 'failed'})
    proof = {'ok': True, 'read_only': True, 'verified_tool': 'generate_structure', 'checked_arguments': {},
             'runtime_fingerprint': executor_fingerprint('generate_structure', tmp_path)}
    with pytest.raises(ValueError, match='changed executor'):
        gate.approve('task', 'inputs', {}, runtime_verification=proof)


def test_changed_interpreter_recovers_only_explicitly_unexecuted_preflight(tmp_path):
    gate = RecoveryGate()
    attempt, _ = gate.claim('structure', 'generate_structure', {}, 'inputs')
    gate.outcome('structure', attempt, {'blocked': True, 'executed': False,
        'error': f'configured pormake interpreter is unavailable: {tmp_path / "missing"}'})
    executable = tmp_path / 'python'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    proof = gate.approve_configured_preflight('structure', attempt, str(executable))
    assert proof['ok']
    assert gate.claim('structure', 'generate_structure', {}, 'inputs')[0]
    assert gate.claim('structure', 'generate_structure', {}, 'inputs')[0] is None


@pytest.mark.parametrize('outcome', [
    {'error': 'scheduler outcome unknown'},
    {'blocked': True, 'executed': False, 'error': 'some other failure'},
    {'blocked': True, 'executed': False, 'job_id': '123',
     'error': 'configured pormake interpreter is unavailable: /missing'},
])
def test_environment_recovery_never_retries_unknown_or_submitted_work(tmp_path, outcome):
    gate = RecoveryGate()
    attempt, _ = gate.claim('structure', 'generate_structure', {}, 'inputs')
    gate.outcome('structure', attempt, outcome)
    with pytest.raises(ValueError):
        gate.approve_configured_preflight('structure', attempt, '/usr/bin/python3')


def test_once_only_permit_and_budget_survive_wording_changes(tmp_path):
    gate = RecoveryGate(path=tmp_path / 'ledger.json')
    for attempt in range(3):
        if attempt:
            gate.approve('task', f'fp{attempt}', {'diagnosis': 'verified'})
        aid, reason = gate.claim('task', 'run_henry', {'gas': 'Kr'}, f'fp{attempt}')
        assert aid and not reason
        gate.outcome('task', aid, {'error': f'different error text {attempt}'})
    fresh = RecoveryGate(path=tmp_path / 'ledger.json')
    assert fresh.snapshot()['task']['attempts'] == 3
    with pytest.raises(ValueError, match='budget'):
        fresh.approve('task', 'fp3', {})


def test_permit_invalidated_by_input_change_after_review(tmp_path):
    input_file = tmp_path / 'sample.cif'
    input_file.write_text('first')
    params = {'cif': str(input_file), 'gas': 'Kr', 'temperature': 298}
    gate = RecoveryGate()
    fp1 = input_fingerprint(params, tmp_path)
    aid, _ = gate.claim('task', 'run_henry', params, fp1)
    gate.outcome('task', aid, {'error': 'bad CIF'})
    with pytest.raises(ValueError, match='unchanged'):
        gate.approve('task', fp1, {})
    input_file.write_text('fixed')
    fixed = input_fingerprint(params, tmp_path)
    gate.approve('task', fixed, {})
    input_file.write_text('unverified new edit')
    assert gate.claim('task', 'run_henry', params, input_fingerprint(params, tmp_path))[0] is None
    input_file.write_text('fixed')
    second, _ = gate.claim('task', 'run_henry', params, fixed)
    assert second
    assert gate.claim('task', 'run_henry', params, fixed)[0] is None


def test_submitted_retry_does_not_resolve_calculation_branch():
    s = Session(config=AgentConfig(api_key='test'))
    branch = s._create_error_branch('run_henry', 'failed')
    s._resolve_error_branches_for_tool('run_henry', '{"submitted":true,"job_id":"42"}')
    assert s._error_branches[branch]['status'] == 'open'


def test_nested_scheduler_and_shell_escape_are_blocked(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail('scheduler/subprocess must not be called')
    monkeypatch.setattr('agents.registry.slurm.submit_and_return', fail)
    monkeypatch.setattr('subprocess.run', fail)
    assert _exec_submit_job({'command': 'sbatch /absolute/submit.sh'})['blocked']
    assert _exec_submit_job({'command': 'cd /tmp && qsub ./job.pbs'})['blocked']
    assert _exec_run_bash({'command': 'sbatch jobs.sh'})['blocked']


def make_session(tmp_path, submit):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path), registry=get_registry())
    s.messages = [{'role': 'user', 'content': 'hello'}]
    s.goal_contract = GoalContract.from_user_message('hello')
    submit.__name__ = 'submit_job'
    s.current_agent = Agent(name='lead-orchestrator', instructions='', functions=[submit])
    s.recovery_gate.path = tmp_path / 'recovery_state.json'
    return s


def test_actual_loop_blocks_second_submit_and_checkpoints_results(tmp_path):
    dispatched = []
    def submit(**params):
        dispatched.append(params)
        return {'failed': True, 'error': 'real missing executable', 'submitted': False}
    s = make_session(tmp_path, submit)
    params = {'command': 'python compute.py', 'work_dir': str(tmp_path)}
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='one', name='submit_job', input=params)]),
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='two', name='submit_job', input={**params, 'command': 'python compute_v2.py'})]),
        SimpleNamespace(content=[TextBlock(type='text', text='Stopped; diagnosis required.')]),
    ])
    s._call_api = lambda agent: next(responses)
    snapshots = []
    def save(state, reason):
        snapshots.append((state, reason))
        write_checkpoint(tmp_path / 'session_checkpoint.json', state)
    s._on_checkpoint = save
    s._execute_loop(s.current_agent)
    assert len(dispatched) == 1
    assert any(reason == 'after_tool_result' for _, reason in snapshots)
    saved = json.loads((tmp_path / 'session_checkpoint.json').read_text())
    assert saved['memory']['tool_call_log'][0]['failed']
    assert saved['recovery_gate']
    for state, _ in snapshots:
        msgs = state['messages']
        for i, msg in enumerate(msgs):
            blocks = msg.get('content')
            if msg['role'] == 'assistant' and isinstance(blocks, list):
                ids = {b['id'] for b in blocks if b.get('type') == 'tool_use'}
                if ids:
                    assert ids <= {b.get('tool_use_id') for b in msgs[i+1]['content']}


def test_storage_failure_blocks_before_dispatch(tmp_path):
    dispatched = []
    def submit(**params):
        dispatched.append(params)
        return {'submitted': True, 'job_id': '42'}
    s = make_session(tmp_path, submit)
    s._call_api = lambda agent: SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='one', name='submit_job', input={'command': 'python compute.py', 'work_dir': str(tmp_path)})])
    def unavailable(*args):
        raise OSError('disk full')
    s._on_checkpoint = unavailable
    with pytest.raises(OSError, match='disk full'):
        s._execute_loop(s.current_agent)
    assert dispatched == []
    assert next(iter(s.recovery_gate.snapshot().values()))['status'] == 'reserved'


def test_unknown_or_unrelated_diagnostic_cannot_authorize_retry(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract = GoalContract.from_user_message('Kr GCMC 298K')
    args = {'cif': str(tmp_path / 'x.cif'), 'gas': 'Kr', 'temperature': 298}
    key = s.recovery_gate.key('run_henry', args, s.goal_contract.version, project_root=tmp_path)
    aid, _ = s.recovery_gate.claim(key, 'run_henry', args, input_fingerprint(args, tmp_path))
    s.recovery_gate.outcome(key, aid, {'job_id': '42', 'failed': True, 'error': 'failed'})
    s.memory.record_tool_call('adsorption', 'diagnose_job', {'job_id': 'other'}, '{"status":"UNKNOWN","terminal":true,"failed":false}')
    diag_id = s.memory.tool_call_log[-1]['call_id']
    s.memory.record_tool_call('adsorption', 'run_bash', {'command': f'test -r {args["cif"]}'}, '{"exit_code":0}')
    verify_id = s.memory.tool_call_log[-1]['call_id']
    with pytest.raises(ValueError, match='failed job ID'):
        s._prepare_retry(key, 'run_henry', args, 'genuine failure diagnosis', diag_id, 'concrete correction implemented', verify_id)


def test_full_review_allows_verified_file_only_fix(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract = GoalContract.from_user_message('Kr GCMC 298K')
    source = tmp_path / 'x.cif'
    source.write_text('bad original input')
    args = {'cif': str(source), 'gas': 'Kr', 'temperature': 298}
    key = s.recovery_gate.key('run_henry', args, s.goal_contract.version, project_root=tmp_path)
    aid, _ = s.recovery_gate.claim(key, 'run_henry', args, input_fingerprint(args, tmp_path))
    s.recovery_gate.outcome(key, aid, {'failed': True, 'error': 'CIF parse error'})
    s.memory.record_tool_call('adsorption', 'read_file', {'path': str(source)}, '{"content":"missing cell parameter"}')
    diag_id = s.memory.tool_call_log[-1]['call_id']
    source.write_text('corrected real cell parameters')
    s.memory.record_tool_call('adsorption', 'run_bash', {'command': f'test -r {source}'}, '{"exit_code":0}')
    verify_id = s.memory.tool_call_log[-1]['call_id']
    s.memory.tool_call_log[-1]['result'] = '{"exit_code":0,"validation_passed":false}'
    with pytest.raises(ValueError, match='verification did not pass'):
        s._prepare_retry(key, 'run_henry', args, 'CIF file was missing cell parameters', diag_id,
                         'Corrected the cell parameters in the input file', verify_id)
    s.memory.tool_call_log[-1]['result'] = '{"exit_code":0}'
    permit = s._prepare_retry(key, 'run_henry', args, 'CIF file was missing cell parameters', diag_id,
                              'Corrected the cell parameters in the input file', verify_id)
    assert permit['ok'] and permit['single_use']
    attempt, reason = s.recovery_gate.claim(key, 'run_henry', args, input_fingerprint(args, tmp_path))
    assert attempt and not reason


@pytest.mark.parametrize('reviewed_resources', [False, True])
def test_prepare_retry_accepts_real_stdout_trace_and_native_runtime_verification(tmp_path, reviewed_resources):
    from agents.installations import executor_fingerprint
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract = GoalContract.from_user_message('生成MOF')
    args = {'material_type': 'MOF', 'n_structures': 1, 'output_dir': str(tmp_path / 'output')}
    step_id = 'generate' if reviewed_resources else ''
    if reviewed_resources:
        s.goal_contract.approved_nodes = [{'step_id': step_id, 'tool': 'generate_structure', 'arguments': args.copy()}]
        args.update(memory_mb=8192, cpus_per_task=1, nodelist='node1', partition='compute')
    key = s.recovery_gate.key('generate_structure', args, s.goal_contract.version, node=step_id, project_root=tmp_path)
    source = tmp_path / 'agents/registry.py'
    source.parent.mkdir()
    source.write_text('old = True\n')
    aid, _ = s.recovery_gate.claim(key, 'generate_structure', args, input_fingerprint(args, tmp_path), project_root=tmp_path)
    s.recovery_gate.outcome(key, aid, {'job_id': '42', 'failed': True, 'error': 'missing database'})
    s.memory.record_tool_call('lead-orchestrator', 'diagnose_job', {'job_id': '42'},
                             json.dumps({'terminal': True, 'failed': True, 'stdout_tail': 'Traceback: missing database'}))
    diag = s.memory.tool_call_log[-1]['call_id']
    source.write_text('repaired = True\n')
    proof = {'ok': True, 'read_only': True, 'verified_tool': 'generate_structure', 'checked_arguments': args,
             'runtime_fingerprint': executor_fingerprint('generate_structure', tmp_path)}
    s.memory.record_tool_call('lead-orchestrator', 'inspect_runtime', {'tool_name': 'generate_structure', 'arguments': args}, json.dumps(proof))
    verify = s.memory.tool_call_log[-1]['call_id']
    if reviewed_resources:
        with pytest.raises(ValueError, match='approved task identity'):
            s._prepare_retry(key, 'generate_structure', {**args, 'n_structures': 2}, 'Database path was wrong', diag,
                             'Configured database path is now resolved', verify)
        assert s.recovery_gate.snapshot()[key]['approved_fingerprint'] is None
    retry_args = {k: v for k, v in args.items() if k not in {'memory_mb', 'cpus_per_task', 'nodelist', 'partition'}} if reviewed_resources else args
    assert s._prepare_retry(key, 'generate_structure', retry_args, 'Database path was wrong', diag,
                            'Configured database path is now resolved', verify)['ok']
    assert s.recovery_gate.snapshot()[key]['approved_fingerprint'] == input_fingerprint(args, tmp_path)


def test_legacy_science_only_permit_can_reuse_only_original_resources(tmp_path):
    source = tmp_path / 'input.cif';source.write_text('original input')
    args = {'cif': str(source), 'gas': 'Kr', 'memory_mb': 8192, 'cpus_per_task': 1}
    gate = RecoveryGate()
    aid, _ = gate.claim('task', 'run_henry', args, input_fingerprint(args, tmp_path), project_root=tmp_path)
    gate.outcome('task', aid, {'error': 'failed'})
    science = {'cif': str(source), 'gas': 'Kr'}
    gate.approve('task', input_fingerprint(science, tmp_path), {}, arguments=science)
    changed = {**args, 'memory_mb': 16384}
    assert gate.claim('task', 'run_henry', changed, input_fingerprint(changed, tmp_path), project_root=tmp_path)[0] is None
    source.write_text('unverified edit')
    assert gate.claim('task', 'run_henry', args, input_fingerprint(args, tmp_path), project_root=tmp_path)[0] is None
    source.write_text('original input')
    assert gate.claim('task', 'run_henry', args, input_fingerprint(args, tmp_path), project_root=tmp_path)[0]


def test_cosmetic_and_ignored_argument_changes_are_not_fixes(tmp_path):
    args = {'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298}
    assert input_fingerprint(args, tmp_path) == input_fingerprint({**args, 'output_csv': 'newname.csv'}, tmp_path)
    assert input_fingerprint(args, tmp_path) == input_fingerprint({**args, 'resource_review_id': 'new-receipt'}, tmp_path)
    issues = get_registry().validate_params('run_henry', {**args, 'made_up_fix': 'yes'})
    assert any('ignored' in issue for issue in issues)


def test_legacy_resource_receipt_fingerprint_migrates_without_blocking_retry(tmp_path):
    old = {'gas': 'CO2', 'memory_mb': 8192, 'resource_review_id': 'old-receipt'}
    legacy = _input_fingerprint(old, tmp_path, include_resource_review=True)
    gate = RecoveryGate(state={'task': {
        'key': 'task', 'tool': 'run_cdft', 'params': old,
        'fingerprint': legacy, 'approved_fingerprint': legacy,
        'attempts': 1, 'status': 'failed', 'project_root': str(tmp_path),
    }})
    current = {**old, 'resource_review_id': 'new-receipt'}

    attempt, reason = gate.claim('task', 'run_cdft', current,
                                 input_fingerprint(current, tmp_path))

    assert attempt and not reason


def test_trimmed_checkpoint_does_not_replay_old_goal_over_newer_one(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    newest = '改用cDFT计算Kr/Xe，298K'
    s.messages = [{'role': 'user', 'content': newest}]
    s.goal_contract = GoalContract.from_user_message(newest)
    s.reconcile_transcript([
        {'role': 'user', 'content': '用GCMC计算CO2，298K'},
        {'role': 'assistant', 'content': 'old answer'},
        {'role': 'user', 'content': newest},
    ])
    assert s.goal_contract.method == 'CDFT'
    assert s.goal_contract.gases == ['Kr', 'Xe']


def test_case_and_path_aliases_do_not_change_submission_identity(tmp_path):
    gate = RecoveryGate()
    first = gate.key('run_henry', {'cif': 'x.cif', 'gas': 'Kr'}, 1, project_root=tmp_path)
    second = gate.key('run_henry', {'cif_path': str(tmp_path / 'x.cif'), 'gas': 'KR'}, 1, project_root=tmp_path)
    assert first == second


def test_running_future_prevents_new_agent_and_tool_calls(tmp_path):
    from concurrent.futures import Future
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._active_tool_futures = [Future()]
    s._call_api = lambda agent: pytest.fail('must not call LLM while earlier tool is still running')
    result = s._execute_loop(Agent(name='lead-orchestrator'))
    assert '仍在后台运行' in result


def test_failed_pipeline_cannot_be_bypassed_with_raw_submission(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract = GoalContract.from_user_message('Kr cDFT 298K')
    s._pending_jobs_for_conv = lambda: []
    args = {'action': 'pipeline', 'cif_dir': 'cifs', 'gas': 'Kr', 'temperature': 298}
    key, aid, reason = s._claim_submission('run_cdft', args)
    assert aid and not reason
    s.recovery_gate.outcome(key, aid, {'failed': True, 'error': 'input error'})
    assert s._claim_submission('submit_job', {'command': 'python new.py', 'work_dir': str(tmp_path)})[2]
    assert s._claim_submission('run_cdft', {'action': 'submit', 'input_dir': 'generated_inputs'})[2]


def test_no_fresh_prose_claim_can_replace_diagnostic_evidence(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract = GoalContract.from_user_message('Kr GCMC 298K')
    args = {'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298}
    key = s.recovery_gate.key('run_henry', args, s.goal_contract.version, project_root=tmp_path)
    aid, _ = s.recovery_gate.claim(key, 'run_henry', args, 'old')
    s.recovery_gate.outcome(key, aid, {'failed': True, 'error': 'real failure'})
    with pytest.raises(ValueError, match='actual diagnostic'):
        s._prepare_retry(key, 'run_henry', args, 'I reflected carefully on the failure', 'invented',
                         'I have fixed all of the issues already', 'invented2')


def test_autogenerated_submission_script_is_not_a_verified_fix(tmp_path):
    gate = RecoveryGate()
    args = {'command': 'python compute.py', 'work_dir': str(tmp_path)}
    aid, _ = gate.claim('task', 'submit_job', args, input_fingerprint(args, tmp_path), project_root=tmp_path)
    (tmp_path / 'submit.sh').write_text('generated by submitter, not repaired')
    gate.outcome('task', aid, {'failed': True, 'error': 'job failed'})
    with pytest.raises(ValueError, match='unchanged'):
        gate.approve('task', input_fingerprint(args, tmp_path), {})


def test_waiting_chain_continues_only_after_prerequisite_terminal(tmp_path):
    gate = RecoveryGate()
    aid, _ = gate.claim('chain', 'run_henry_chain', {'material': 'm'}, 'fp')
    gate.outcome('chain', aid, {'chain_status': 'waiting', 'context': {'charge_job_id': '42'}})
    assert gate.claim('chain', 'run_henry_chain', {'material': 'm'}, 'fp')[0] is None
    gate.sync_jobs([{'job_id': '42', 'terminal': True, 'failed': False, 'state': 'COMPLETED'}])
    resumed, reason = gate.claim('chain', 'run_henry_chain', {'material': 'm'}, 'fp')
    assert resumed and not reason
    assert gate.snapshot()['chain']['attempts'] == 2


def test_scheduler_receipt_binds_job_before_parent_tool_returns(tmp_path, monkeypatch):
    from agents import slurm, job_watch
    from agents.watch_context import set_context, clear_context
    gate = RecoveryGate(path=tmp_path / 'recovery.json')
    aid, _ = gate.claim('task', 'run_henry', {}, 'fp')
    watch = job_watch.JobWatch(watch_file=str(tmp_path / 'jobs.json'))
    monkeypatch.setattr(job_watch, 'get_watch', lambda: watch)
    monkeypatch.setattr(slurm.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=0, stdout='Submitted batch job 42', stderr=''))
    set_context('u', 'c1', 'adsorption', 'c1', tool_name='run_henry', recovery_key='task',
                attempt_id=aid, recovery_path=str(tmp_path / 'recovery.json'))
    try:
        receipt = slurm.submit_sbatch_local('#!/bin/bash\npython compute.py', str(tmp_path / 'work'))
    finally:
        clear_context()
    assert receipt['submitted']
    durable = RecoveryGate(path=tmp_path / 'recovery.json').snapshot()['task']
    assert durable['job_ids'] == ['42'] and durable['status'] == 'submitted'


def test_script_submitting_more_jobs_is_rejected(tmp_path, monkeypatch):
    script = tmp_path / 'batch.py'
    script.write_text("import subprocess\nsubprocess.run(['sbatch', 'child.sh'])\n")
    monkeypatch.setattr('subprocess.run', lambda *a, **kw: pytest.fail('must not execute nested submitter'))
    assert _exec_submit_job({'command': f'python {script}', 'work_dir': str(tmp_path)})['blocked']


def test_unconfirmed_terminal_record_cannot_pass_success_report_gate(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    gate = s._gate_jobs_before_report([{'job_id': '42', 'tool': 'run_henry', 'state': 'UNKNOWN',
                                      'terminal': True, 'failed': False}])
    assert not gate['ok']
    assert '未确认' in gate['running'][0]
