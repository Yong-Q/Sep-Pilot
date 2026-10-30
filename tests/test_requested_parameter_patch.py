import copy

import pytest

from agents.goal_contract import GoalContract
from agents.requested_parameters import requested_parameter_patch


def sample():
    before = {'step_id': 'validate', 'tool': 'run_gcmc_batch', 'agent': 'adsorption',
              'arguments': {'cif_dir': 'top2', 'gases': ['CO2', 'N2'],
                            'temperature': 298, 'pressure': 1, 'cycles': 10000},
              'depends_on': ['rank'], 'expected_outputs': ['results.csv']}
    after = copy.deepcopy(before)
    after['arguments']['cycles'] = 1000
    goal = GoalContract.from_user_message('用GCMC计算CO2/N2，298K，直接执行')
    goal.parameters['cycles'] = 1000
    changes = [{'operation': 'upsert', 'step_id': 'validate', 'node': after}]
    return before, changes, goal


def test_explicit_cycle_request_authorizes_only_its_typed_delta():
    before, changes, goal = sample()
    receipt = requested_parameter_patch([before], changes, goal)
    assert receipt['goal_version'] == goal.version
    assert receipt['changes'] == [{'step_id': 'validate', 'argument': 'cycles',
                                   'before': 10000, 'after': 1000, 'goal_field': 'cycles'}]
    assert before['arguments']['cycles'] == 10000


@pytest.mark.parametrize('mutation', ['no_request', 'wrong_cycles', 'input', 'method',
                                    'memory', 'remove', 'new_node', 'output_floor',
                                    'dependencies', 'owner', 'completed', 'submitted'])
def test_parameter_request_does_not_authorize_unrelated_mutations(mutation):
    before, changes, goal = sample()
    after = changes[0]['node']
    if mutation == 'no_request': goal.parameters.pop('cycles')
    elif mutation == 'wrong_cycles': after['arguments']['cycles'] = 500
    elif mutation == 'input': after['arguments']['cif_dir'] = 'different'
    elif mutation == 'method': after['tool'] = 'run_cdft'
    elif mutation == 'memory': after['arguments']['memory_mb'] = 999999
    elif mutation == 'remove': changes[0] = {'operation': 'remove', 'step_id': 'validate'}
    elif mutation == 'new_node': changes[0]['step_id'] = after['step_id'] = 'new'
    elif mutation == 'output_floor': after['expected_outputs'] = []
    elif mutation == 'dependencies': after['depends_on'] = []
    elif mutation == 'owner': after['agent'] = 'harness'
    elif mutation == 'completed': before.update(status='completed', done=True)
    elif mutation == 'submitted': before['job_ids'] = ['test-job']
    assert requested_parameter_patch([before], changes, goal) is None


def test_temperature_request_and_operational_repair_share_one_version():
    before, changes, goal = sample()
    goal.parameters['temperature_K'] = 310
    after = changes[0]['node']
    after['arguments'].update(temperature=310, output_dir='new-attempt')
    assert {item['argument'] for item in requested_parameter_patch([before], changes, goal)['changes']} == {'cycles', 'temperature'}


def test_gcmc_cycle_request_cannot_change_string_steps():
    before, changes, goal = sample()
    before['tool'] = changes[0]['node']['tool'] = 'run_string_tst'
    assert requested_parameter_patch([before], changes, goal) is None


def test_reconciliation_exposes_exact_pending_delta_without_changing_goal():
    from agents.requested_parameters import parameter_reconciliation
    before, _, goal = sample()
    goal.approved_nodes = [before]
    saved = copy.deepcopy(goal.to_dict())
    state = {'nodes': {'validate': {'status': 'pending'}}}
    receipt = parameter_reconciliation(goal, state)
    assert receipt['applied'] is False
    assert receipt['differences'] == [{'step_id': 'validate', 'tool': 'run_gcmc_batch', 'argument': 'cycles',
        'planned_value': 10000, 'current_user_constraint': 1000, 'status': 'pending', 'job_ids': []}]
    assert goal.to_dict() == saved
    before['arguments']['cycles'] = 1000
    assert parameter_reconciliation(goal, state) is None


def test_user_parameter_proposal_is_applied_with_durable_exact_scope(tmp_path, monkeypatch):
    from agents.config import AgentConfig
    from agents.session import Session
    from agents.defns import ORCHESTRATOR
    from agents.task_line import TaskLineStore
    from agents.watch_context import set_context, clear_context
    from agents.parallel_workflow import _approval_digest
    before, changes, goal = sample()
    before['depends_on'] = changes[0]['node']['depends_on'] = []
    before['arguments']['cif_dir'] = changes[0]['node']['arguments']['cif_dir'] = str(tmp_path / 'cifs')
    (tmp_path / 'cifs').mkdir()
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c', username='u', conv_id='c')
    store.apply_workflow_patch('c', [{'operation': 'upsert', 'step_id': 'validate', 'node': before}],
                               0, 1, 'original approved plan', tmp_path)
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session.current_agent = ORCHESTRATOR
    session._current_line_id = 'c'
    session.goal_contract = goal
    session.goal_contract.approved_plan_version = 1
    session.goal_contract.approved_nodes = [before]
    session._on_workflow_start = lambda version: {'scheduled': True, 'plan_version': version}
    checkpoints = []
    session._on_checkpoint = lambda state, reason: checkpoints.append(state)
    set_context(username='u', conv_id='c')
    try:
        session._set_control_source('GCMC按1000 cycles测试，继续执行')
        proposal = session._propose_workflow_patch(1, 'Apply user cycle budget', changes)
        assert proposal['status'] == 'ready_to_apply'
        assert not session._waiting_for_user_input
        session._runtime_snapshot = lambda: {'nodes': {'validate': {'status': 'succeeded', 'job_ids': ['completed-job']}}}
        with pytest.raises(ValueError, match='parameter.*unexecuted|parameter.*started'):
            session._apply_workflow_patch(proposal['patch']['new_version'])
        assert store.get_line('c')['plan_version'] == 1
        session._runtime_snapshot = lambda: {'nodes': {}}
        result = session._apply_workflow_patch(proposal['patch']['new_version'])
        assert result['execution']['scheduled']
        applied = store.get_line('c')['steps'][0]
        assert applied['arguments']['cycles'] == 1000
        proof = session.context['approved_execution_proof']
        assert proof['owner'] == {'username': 'u', 'conv_id': 'c'}
        assert proof['approval_sha256'] == _approval_digest(goal.to_dict())
        assert proof['control_source']['origin'] == 'user'
        assert any(s['context'].get('approved_execution_proof') == proof for s in checkpoints)
    finally:
        clear_context()


def test_formal_completed_node_is_not_downgraded_by_older_failed_job_history(tmp_path, monkeypatch):
    from agents.config import AgentConfig
    from agents.session import Session
    from agents.defns import ORCHESTRATOR
    from agents.task_line import TaskLineStore
    from agents.watch_context import set_context, clear_context
    before, changes, goal = sample()
    output = str(tmp_path / 'cifs'); (tmp_path / 'cifs').mkdir()
    before['depends_on'] = changes[0]['node']['depends_on'] = ['generate']
    before['arguments']['cif_dir'] = changes[0]['node']['arguments']['cif_dir'] = output
    generate = {'step_id':'generate', 'agent':'harness', 'tool':'generate_structure',
                'arguments':{'material_type':'MOF','n_structures':2,'output_dir':output},
                'depends_on':[], 'expected_outputs':[{'kind':'directory','path':output,'pattern':'*.cif','min_count':2}]}
    store = TaskLineStore(str(tmp_path / 'lines.json')); store.begin_line('c',username='u',conv_id='c')
    store.apply_workflow_patch('c',[{'operation':'upsert','step_id':n['step_id'],'node':n} for n in (generate,before)],0,1,'formal plan',tmp_path)
    store.upsert_step('c','generate',status='completed',done=True,job_ids=['older-failed','verified-completed'])
    monkeypatch.setattr('agents.task_line.get_store',lambda:store)
    s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.current_agent=ORCHESTRATOR;s._current_line_id='c';s.goal_contract=goal
    goal.approved_plan_version=1;goal.approved_nodes=[generate,before]
    s._runtime_snapshot=lambda:{'nodes':{'generate':{'status':'succeeded','job_ids':['verified-completed']},'validate':{'status':'pending'}}}
    s._pending_jobs_for_conv=lambda:[{'job_id':'older-failed','terminal':True,'failed':True,'state':'FAILED'},
                                   {'job_id':'verified-completed','terminal':True,'failed':False,'state':'COMPLETED'}]
    set_context(username='u',conv_id='c')
    try:
        s._set_control_source('GCMC按1000 cycles测试，继续执行')
        s._propose_workflow_patch(1,'user changes future budget',changes)
        current=next(n for n in store.get_line('c')['steps'] if n['step_id']=='generate')
        assert current['status']=='completed' and current['done'] is True
    finally:clear_context()
