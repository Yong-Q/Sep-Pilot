"""Failures observed in evaluator5: no real LLM or scheduler submissions."""
import json
from types import SimpleNamespace

import pytest
from anthropic.types import ToolUseBlock, TextBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.goal_contract import GoalContract
from agents.output_contract import normalize_output, artifact_fact
from agents.parallel_workflow import resources_for, conflicts, artifact_base
from agents.registry import _build_default_registry, _exec_cdft, _exec_run_bash
from agents.session import Session
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context
from agents.workspace import scoped_shell_contract
from agents.workflow_patch import patched_graph


@pytest.fixture(autouse=True)
def clean_context():
    clear_context()
    yield
    clear_context()


def session(tmp_path):
    return Session(config=AgentConfig(api_key='test', project_root=tmp_path), registry=_build_default_registry())


def plan_node(tmp_path):
    return {'agent': 'analyst', 'tool': 'run_cdft',
            'arguments': {'action': 'pipeline', 'cif_dir': str(tmp_path / 'cifs'), 'gas': 'Kr',
                          'temperature': 298, 'job_work_dir': str(tmp_path / 'runs' / 'tester' / 'c1' / 'cdft' / 'Kr')},
            'depends_on': [], 'expected_outputs': ['results.csv']}


def test_approval_activates_runtime_after_checkpoint_not_legacy_dispatch(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = session(tmp_path)
    s._current_line_id = 'c1'
    s.current_agent = ORCHESTRATOR
    s.goal_contract = GoalContract.from_user_message('用cDFT计算Kr Henry，298K，直接执行')
    order = []
    s._on_checkpoint = lambda state, reason: order.append((reason, state['goal_contract']['approved_plan_version']))
    s._on_workflow_start = lambda version: order.append(('start', version)) or {'status': 'scheduled'}
    s._record_workflow_draft([{'step_id': 'kr', 'description': 'Run and validate Kr cDFT',
        'agent': 'analyst', 'tool': 'run_cdft', 'depends_on': [], 'missing_parameters': []}],
        completion_criteria='Verified Kr cDFT result table at 298 K')
    result = s._propose_workflow_patch(0, 'Compile a complete executable scientific plan', [
        {'operation': 'upsert', 'step_id': 'kr', 'node': plan_node(tmp_path)}])
    assert result['execution']['scheduled']
    assert order[-2:] == [('approved_workflow_before_activation', 1), ('start', 1)]
    assert 'WORKFLOW_BLOCK' in s._claim_submission('run_cdft', plan_node(tmp_path)['arguments'])[2]


def test_activation_failure_returns_receipt_and_never_serial_fallback(tmp_path):
    s = session(tmp_path)
    s.goal_contract.approved_nodes = [plan_node(tmp_path)]
    s.goal_contract.approved_plan_version = 2
    def blocked(version):
        raise ValueError('legacy dispatch has unresolved outcomes')
    s._on_workflow_start = blocked
    receipt = s._activate_approved_workflow()
    assert receipt['status'] == 'activation_blocked' and not receipt['legacy_fallback_allowed']
    assert 'legacy serial' in s._claim_submission('run_cdft', plan_node(tmp_path)['arguments'])[2]


def test_hidden_virtual_tool_cannot_bypass_agent_capabilities(tmp_path):
    s = session(tmp_path)
    s.messages = [{'role': 'user', 'content': 'hello'}]
    s._request_user_decision = lambda **kw: pytest.fail('hidden main-only tool executed')
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='bad', name='request_user_decision', input={
            'question': 'Should we repair the missing input directory?', 'reason': 'missing inputs', 'related_nodes': ['xe']})]),
        SimpleNamespace(content=[TextBlock(type='text', text='Return facts to main chat.')]),
        SimpleNamespace(content=[TextBlock(type='text', text='Main chat has the facts.')])])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(Agent(name='analyst', functions=[]))
    result = next(m['content'][0] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], list))
    assert json.loads(result['content'])['executed'] is False
    assert not s._pending_user_interaction
    assert s.context['user_decision_proposal']['agent'] == 'analyst'
    assert s.current_agent.name == 'lead-orchestrator'


def test_force_handoff_mode_does_not_block_workflow_draft_control(tmp_path):
    s = session(tmp_path)
    s.messages = [{'role': 'user', 'content': 'execute the approved scientific request'}]
    for index in range(4):
        s.memory.record_tool_call('lead-orchestrator', 'task_line_query', {'index': index}, '{}')
    draft = {
        'completion_criteria': 'A verified COF screening report',
        'nodes': [{
            'step_id': 'screen', 'description': 'screen COFs', 'agent': 'analyst',
            'tool': 'read_file', 'arguments': {'path': 'results.csv'},
            'depends_on': [], 'expected_outputs': [],
        }],
    }
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(
            type='tool_use', id='draft', name='record_workflow_draft', input=draft,
        )]),
        SimpleNamespace(content=[TextBlock(type='text', text='Draft persisted.')]),
    ])
    s._call_api = lambda agent: next(responses)

    assert s._execute_loop(ORCHESTRATOR) == 'Draft persisted.'
    assert s.context['workflow_draft']['nodes'][0]['step_id'] == 'screen'


def test_internal_preflight_deadlock_recovers_persisted_draft_instead_of_asking_user(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    nodes = [{
        'step_id': 'screen', 'description': 'screen COFs', 'agent': 'analyst',
        'tool': 'read_file', 'arguments': {'path': 'results.csv'},
        'depends_on': [], 'expected_outputs': [],
    }]
    store.upsert_step(
        'c1', 'record_workflow_draft', tool='record_workflow_draft', status='running', done=False,
        arguments={'nodes': nodes, 'completion_criteria': 'A verified COF screening report'},
        validation={'goal_contract': 'passed', 'execution': 'passed'},
    )
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = session(tmp_path)
    s._current_line_id = 'c1'
    s.current_agent = ORCHESTRATOR
    s.goal_contract = GoalContract.from_user_message('筛选COF并直接执行')
    s.goal_contract.execution_authorized = True

    receipt = s._request_user_decision(
        '工作流初始化守卫发生循环，是否请管理员处理？',
        '内部工作流工具无法初始化', ['record_workflow_draft', 'screen'],
    )

    assert receipt['status'] == 'autonomous_repair'
    assert not s._waiting_for_user_input and s._pending_user_interaction is None
    assert s.context['workflow_draft']['nodes'][0]['step_id'] == 'screen'
    step = next(item for item in store.get_line('c1')['steps'] if item['step_id'] == 'record_workflow_draft')
    assert step['done'] and step['status'] == 'completed'


def test_checkpoint_import_clears_obsolete_internal_preflight_question(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    nodes = [{
        'step_id': 'screen', 'description': 'screen COFs', 'agent': 'analyst',
        'tool': 'read_file', 'arguments': {'path': 'results.csv'},
        'depends_on': [], 'expected_outputs': [],
    }]
    store.upsert_step(
        'c1', 'record_workflow_draft', tool='record_workflow_draft', status='running', done=False,
        arguments={'nodes': nodes, 'completion_criteria': 'A verified COF screening report'},
        validation={'goal_contract': 'passed', 'execution': 'passed'},
    )
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    original = session(tmp_path)
    original._current_line_id = 'c1'
    original.current_agent = ORCHESTRATOR
    original.goal_contract = GoalContract.from_user_message('筛选COF并直接执行')
    original.goal_contract.execution_authorized = True
    original._waiting_for_user_input = True
    original._pending_user_interaction = {
        'tool': 'user_decision',
        'params': {'question': '内部守卫循环，是否请管理员处理？', 'reason': '工作流无法初始化',
                   'related_nodes': ['record_workflow_draft', 'screen'],
                   'recovery_key': '', 'candidate_job_id': ''},
        'prompt': '内部守卫循环，是否请管理员处理？',
    }

    restored = session(tmp_path)
    assert restored.import_state(original.export_state())
    assert not restored._waiting_for_user_input and restored._pending_user_interaction is None
    assert restored.context['workflow_draft']['nodes'][0]['step_id'] == 'screen'
    assert restored.context['preflight_recovery_ready']['next_action'] == 'propose_workflow_patch'


def test_initial_compile_excludes_preflight_control_record_from_executable_dag(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = session(tmp_path)
    s._current_line_id = 'c1'
    s.current_agent = ORCHESTRATOR
    s.goal_contract = GoalContract.from_user_message('读取筛选结果并直接执行')
    s.goal_contract.execution_authorized = True
    node = {
        'step_id': 'screen', 'description': 'read verified screening results',
        'agent': 'analyst', 'tool': 'read_file',
        'arguments': {'path': str(tmp_path / 'results.csv')},
        'depends_on': [], 'expected_outputs': [],
    }
    s._record_workflow_draft([node], 'A verified COF screening report')
    store.upsert_step(
        'c1', 'record_workflow_draft', tool='record_workflow_draft', status='completed', done=True,
        arguments={'nodes': [node], 'completion_criteria': 'A verified COF screening report'},
        validation={'goal_contract': 'passed', 'schema': 'passed'},
    )
    s._on_workflow_start = lambda version: {'status': 'scheduled', 'plan_version': version}

    receipt = s._propose_workflow_patch(0, 'compile restored complete draft', [
        {'operation': 'upsert', 'step_id': 'screen', 'node': node},
    ])

    assert receipt['execution']['scheduled']
    assert [item['step_id'] for item in s.goal_contract.approved_nodes] == ['screen']


def test_tool_reply_exposes_persisted_evidence_identity(tmp_path):
    s = session(tmp_path)
    s.messages = [{'role': 'user', 'content': 'hello'}]
    def read_file(**kwargs):
        return {'path': kwargs['path'], 'content': 'actual evidence'}
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='provider-id', name='read_file', input={'path': 'x.cif'})]),
        SimpleNamespace(content=[TextBlock(type='text', text='Done.')])])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(Agent(name='lead-orchestrator', functions=[read_file]))
    result = next(m['content'][0] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], list))
    ref = json.loads(result['content'])['evidence_ref']
    assert ref['call_id'] == s.memory.tool_call_log[-1]['call_id'] != 'provider-id'
    assert 'evidence_ref' not in json.loads(s.memory.tool_call_log[-1]['result'])


@pytest.mark.parametrize('claim', [
    'Kr 气体的 cDFT 输入文件生成完成', '10 个材料的 Kr cDFT 输入文件生成完成（n_inputs=10）',
    'Kr cDFT 计算作业已提交，job_ids 列表'])
def test_text_claims_are_not_artifact_contracts(claim):
    with pytest.raises(ValueError, match='prose'):
        normalize_output(claim)


def test_typed_directory_requires_all_declared_fresh_nonempty_files(tmp_path):
    contract = {'kind': 'directory', 'path': 'inputs', 'pattern': '*.dat', 'min_count': 2}
    folder = tmp_path / 'inputs'
    folder.mkdir()
    (folder / 'a.dat').write_text('Kr params')
    assert artifact_fact(contract, tmp_path) is None
    (folder / 'b.dat').write_text('')
    assert artifact_fact(contract, tmp_path) is None
    (folder / 'b.dat').write_text('Kr params')
    fact = artifact_fact(contract, tmp_path)
    assert len(fact['files']) == 2 and fact['size'] > 0
    (folder / 'escaped.dat').symlink_to(tmp_path / 'outside.dat')
    (tmp_path / 'outside.dat').write_text('another scope')
    assert artifact_fact(contract, tmp_path) is None


@pytest.mark.parametrize('fault', ['wrong_directory', 'missing_directory', 'wrong_gas', 'wrong_temperature'])
def test_inputs_submit_edges_carry_real_directory_and_science_scope(tmp_path, fault):
    inputs = {**plan_node(tmp_path), 'step_id': 'inputs', 'arguments': {
        'action': 'inputs', 'cif_dir': 'cifs', 'gas': 'Xe', 'temperature': 298, 'input_dir': 'inputs_xe'}}
    submit = {**plan_node(tmp_path), 'step_id': 'submit', 'depends_on': ['inputs'],
              'arguments': {'action': 'submit', 'gas': 'Xe', 'temperature': 298, 'input_dir': 'inputs_xe'}}
    patched_graph([inputs, submit], [], tmp_path)
    if fault == 'wrong_directory': submit['arguments']['input_dir'] = 'inputs'
    if fault == 'missing_directory': inputs['arguments'].pop('input_dir')
    if fault == 'wrong_gas': submit['arguments']['gas'] = 'Kr'
    if fault == 'wrong_temperature': submit['arguments']['temperature'] = 300
    with pytest.raises(ValueError, match='inputs|producer|temperature'):
        patched_graph([inputs, submit], [], tmp_path)


@pytest.mark.parametrize('command', [
    'mkdir -p runs/evaluator4/REDACTED/charged && cp /tmp/input.cif runs/evaluator4/REDACTED/charged/',
    'cat ../../../users.json', 'python3 -c "print(1)"', 'ls $(pwd)', 'ls; sh script.sh',
    'find . -exec cat {} ;', 'grep --dereference-recursive password .', 'cp -RL src dst',
    'echo data > ../../other-session/out.txt', 'ls\npython3 script.py', 'cat /proc/self/environ',
    'rg --pre="sh script.sh" .', 'sort --compress-program=script.sh data',
    'wc --files0-from=targets.txt', 'find -files0-from targets.txt', 'file -m magic -f targets'])
def test_scoped_shell_rejects_cross_tenant_and_code_escape(tmp_path, command):
    own = tmp_path / 'runs' / 'tester' / 'c1'
    own.mkdir(parents=True)
    _, issues = scoped_shell_contract(command, '', 'tester', 'c1', tmp_path)
    assert issues


def test_scoped_shell_prepares_own_files_and_blocks_other_session(tmp_path, monkeypatch):
    monkeypatch.setattr('agents.registry.get_config', lambda: AgentConfig(api_key='test', project_root=tmp_path))
    own = tmp_path / 'runs' / 'tester' / 'c1'
    own.mkdir(parents=True)
    (own / 'source.dat').write_text('Xe params')
    set_context('tester', 'c1', 'analyst')
    result = _exec_run_bash({'command': 'mkdir -p inputs_xe && cp source.dat inputs_xe/ && wc -l inputs_xe/source.dat'})
    assert result['exit_code'] == 0 and (own / 'inputs_xe' / 'source.dat').exists()
    result = _exec_run_bash({'command': f'mkdir -p {tmp_path}/runs/tester/another-session/charged'})
    assert result['blocked'] and not result['executed']
    assert not (tmp_path / 'runs' / 'tester' / 'another-session').exists()


def test_scoped_shell_leases_do_not_serialize_other_users(tmp_path):
    node = {'tool': 'run_bash', 'arguments': {'command': 'pwd'}, 'expected_outputs': []}
    a = resources_for(node, tmp_path, tmp_path / 'runs' / 'alice' / 'c1')
    b = resources_for(node, tmp_path, tmp_path / 'runs' / 'bob' / 'c1')
    assert not any(conflicts(x, y) for x in a for y in b)
    own_write = {'key': 'path:' + str(tmp_path / 'runs' / 'alice' / 'c1' / 'inputs'), 'mode': 'write'}
    assert any(conflicts(x, own_write) for x in a)


def test_cdft_artifacts_follow_timestamped_job_receipt_not_preparation_root(tmp_path):
    node = plan_node(tmp_path)
    preparation = tmp_path / 'runs' / 'tester' / 'c1' / 'cdft' / 'Kr'
    actual = preparation / 'data' / '20260916_231137'
    assert artifact_base(node, tmp_path, preparation.parent) == preparation
    assert artifact_base(node, tmp_path, preparation.parent, {'work_dir': str(actual)}) == actual


def test_zero_inputs_is_failure_and_gas_defaults_are_disjoint(tmp_path, monkeypatch):
    monkeypatch.setattr('agents.registry.get_config', lambda: AgentConfig(api_key='test', project_root=tmp_path))
    captured = []
    def process(argv, **kw):
        captured.append(argv[-1])
        return SimpleNamespace(returncode=0, stdout='{"n_inputs": 0}', stderr='')
    monkeypatch.setattr('subprocess.run', process)
    set_context('tester', 'c1', 'analyst')
    a = _exec_cdft({'action': 'inputs', 'cif_dir': 'source', 'gas': 'Kr', 'temperature': 298})
    b = _exec_cdft({'action': 'inputs', 'cif_dir': 'source', 'gas': 'Xe', 'temperature': 298})
    assert a['failed'] and b['failed'] and not a['submitted']
    import ast
    def directory(code):
        tree = ast.parse(code)
        call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'generate_cdft_inputs')
        return ast.literal_eval(next(k.value for k in call.keywords if k.arg == 'output_dir'))
    assert directory(captured[0]) != directory(captured[1])
    assert 'n = len(generated)' in captured[0]  # don't count stale .dat files


def test_supervisor_corrects_false_running_from_plan_label(tmp_path):
    s = session(tmp_path)
    args = plan_node(tmp_path)['arguments']
    aid, _ = s.recovery_gate.claim('xe', 'run_cdft', args, 'fingerprint')
    s.recovery_gate.outcome('xe', aid, {'failed': True, 'error': 'input_dat_dir not found'})
    decision = s._audit_supervisor_decision({'next_action': 'wait_existing', 'reason': 'Xe is running'},
                                           {'kind': 'tool_failed', 'payload': {}})
    assert decision['next_action'] == 'diagnose_and_fix' and decision['audited']
    assert decision['authoritative_facts']['jobs'] == []


def test_supervisor_user_negotiation_and_complete_need_real_receipts(tmp_path):
    s = session(tmp_path)
    receipt = s._audit_supervisor_decision({'next_action': 'wait_existing', 'reason': 'wait'},
                                          {'payload': {'requires_user': True}})
    assert receipt['next_action'] == 'ask_user'
    receipt = s._audit_supervisor_decision({'next_action': 'verified_complete', 'reason': 'done'}, {'payload': {}})
    assert receipt['next_action'] != 'verified_complete'


def test_supervisor_cannot_escalate_internal_contract_error_to_user(tmp_path):
    s = session(tmp_path)
    receipt = s._audit_supervisor_decision(
        {'next_action': 'ask_user', 'reason': 'report schema failed'},
        {'kind': 'tool_failed', 'payload': {
            'requires_user': False, 'category': 'contract_validation',
        }},
    )
    assert receipt['next_action'] == 'diagnose_and_fix'
    assert receipt['audited'] is True


def completed_cdft(s, tmp_path):
    args = {'action': 'submit', 'input_dir': str(tmp_path / 'inputs'), 'gas': 'Kr', 'temperature': 298}
    aid, _ = s.recovery_gate.claim('kr', 'run_cdft', args, 'original')
    work = tmp_path / 'job-76964'
    s.recovery_gate.outcome('kr', aid, {'submitted': True, 'job_id': '76964', 'work_dir': str(work), 'n_mofs': 1})
    s.recovery_gate.sync_jobs([{'job_id': '76964', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])
    return work


def test_real_collect_then_full_csv_read_can_accept_without_echo_loop(tmp_path):
    s = session(tmp_path)
    work = completed_cdft(s, tmp_path)
    output = tmp_path / 'kr_results.csv'
    s.memory.record_tool_call('analyst', 'run_cdft', {'action': 'collect', 'job_work_dir': str(work)}, json.dumps({
        'rows': [{'MOF': 'u-HOF', 'Kr_henry_mol_L_atm': 0.25}], 'output_csv': str(output)}))
    s.memory.record_tool_call('analyst', 'read_file', {'path': str(output)}, json.dumps({
        'content': 'MOF,Kr_henry_mol_L_atm\nu-HOF,0.25\n', 'total_lines': 2, 'offset': 0, 'truncated': False}))
    receipt = s._accept_recovered_result('kr', s.memory.tool_call_log[-1]['call_id'], 'Collected all actual Kr job results with finite nonnegative Henry values')
    assert receipt['status'] == 'completed'


@pytest.mark.parametrize('kind', ['other_job', 'empty', 'nan', 'wrong_gas', 'partial', 'duplicate'])
def test_bad_csv_or_wrong_collection_cannot_mark_complete(tmp_path, kind):
    s = session(tmp_path)
    work = completed_cdft(s, tmp_path)
    params = {'action': 'collect', 'job_work_dir': str(work if kind != 'other_job' else tmp_path / 'other-job')}
    rows = [{'MOF': 'u-HOF', 'Kr_henry_mol_L_atm': 0.25}]
    if kind == 'empty': rows = []
    if kind == 'nan': rows[0]['Kr_henry_mol_L_atm'] = float('nan')
    if kind == 'wrong_gas': rows = [{'MOF': 'u-HOF', 'Xe_henry_mol_L_atm': 0.25}]
    if kind == 'duplicate': rows *= 2
    if kind == 'partial':
        s.memory.record_tool_call('analyst', 'read_file', {'path': str(work / 'results.csv')},
                                  '{"content":"MOF,Kr_henry_mol_L_atm\nu-HOF,0.25", "truncated":true}')
    else:
        s.memory.record_tool_call('analyst', 'run_cdft', params, json.dumps({'rows': rows}))
    with pytest.raises(ValueError):
        s._accept_recovered_result('kr', s.memory.tool_call_log[-1]['call_id'], 'This purported CSV would claim task completion')
    assert s.recovery_gate.snapshot()['kr']['status'] == 'completed_unverified'
