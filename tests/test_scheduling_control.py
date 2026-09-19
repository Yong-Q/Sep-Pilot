import copy
import json
from types import SimpleNamespace

import pytest

from agents.goal_contract import GoalContract
from agents.parallel_workflow import WorkflowStore


def test_handoff_background_is_not_requested_method():
    goal=GoalContract.from_user_message('用cDFT计算Xe/Kr，298K，直接执行')
    assert goal.guard_tool_call('handoff_to_harness',{'task':'调配生成结构的计算节点','context':'历史GCMC流程示例与CO2分析结果'})[0]
    assert goal.guard_tool_call('handoff_to_adsorption',{'task':'用GCMC计算Xe/Kr'})[0]
    assert not goal.guard_tool_call('run_gcmc_batch',{'gases':['Xe','Kr']})[0]
    assert goal.guard_tool_call('handoff_to_harness',{'task':'用generate_structure在/home/user/gcmc_agent/BiMemAgent-claude-sdk/runs/11/REDACTED生成100个MOF'})[0]
    assert not goal.guard_tool_call('handoff_to_harness',{'task':'执行委派','context':json.dumps({'node':{'tool':'run_gcmc_batch','arguments':{'gases':['Xe','Kr']}}})})[0]
    assert goal.guard_tool_call('generate_structure',{'material_type':'MOF','description':'GCMC/cDFT均可用的输入'})[0]


def test_unmodified_live_branch_carries_to_future_plan_with_same_lease(tmp_path):
    store=WorkflowStore(tmp_path/'store.json')
    current={'step_id':'gen','agent':'harness','tool':'generate_structure','arguments':{},'depends_on':[],
             'expected_outputs':['result.cif'],'resources':[{'key':'path:/protected','mode':'write'}]}
    future={'step_id':'future','agent':'analyst','tool':'read_file','arguments':{'path':'result.cif'},'depends_on':['gen'],'expected_outputs':[],'resources':[]}
    goal={'version':1,'approved_plan_version':1}
    store.activate('wf',[current,future],goal,'a','c')
    claimed=store.claim('wf','gen');token=claimed['token']
    leases=copy.deepcopy(store.snapshot()['leases'])
    newfuture={**future,'arguments':{'path':'updated.cif'}}
    state=store.activate('wf',[current,newfuture],{'version':2,'approved_plan_version':2},'a','c')
    assert state['nodes']['gen']['token']==token and state['nodes']['gen']['status']=='running'
    assert store.snapshot()['leases']==leases
    with pytest.raises(ValueError):store.activate('wf',[{**current,'arguments':{'changed':True}},newfuture],{'version':3,'approved_plan_version':3},'a','c')


def test_queued_retarget_preserves_job_cpu_memory_and_uses_control_not_shell(tmp_path,monkeypatch):
    from agents.job_control import retarget_queued_job
    from agents.watch_context import set_context,clear_context
    from agents.job_watch import JobWatch
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c',tool='generate_structure')
    monkeypatch.setattr('agents.job_watch.get_watch',lambda:watch)
    constraints={'JobState':'PENDING','ReqNodeList':'(null)','Partition':'compute','NumCPUs':'4','ReqTRES':'cpu=4,mem=1024M,node=1'}
    inventory={'verified':True,'stale':False,'policy':{'excluded_nodes':[],'preferred_nodes':[],'allowed_states':['idle'],'tool_profiles':{}},
        'nodes':{'node07':{'name':'node07','states':['idle'],'partitions':['bigcpu'],'cpus':{'idle':8,'total':8},'memory':{'verified':True,'available_for_scheduling_mib':4096}}}}
    monkeypatch.setattr('agents.resource_review.assess_job_resources',lambda job_id:{'effective_constraints':copy.deepcopy(constraints),'node_inventory':inventory})
    commands=[]
    def run(argv,**kwargs):
        commands.append(argv);constraints.update(ReqNodeList='node07',Partition='bigcpu')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    monkeypatch.setattr('subprocess.run',run)
    set_context('a','c','lead-orchestrator')
    try:result=retarget_queued_job('123','node07','bigcpu','a','c',tmp_path)
    finally:clear_context()
    assert result['ok'] and result['job_id']=='123' and result['not_resubmitted']
    assert commands==[['scontrol','update','JobId=123','ReqNodeList=node07','Partition=bigcpu','MinMemoryNode=1024']]
    assert result['scientific_parameters_unchanged']


def test_generation_scheduler_fields_reach_submission_not_just_description(tmp_path,monkeypatch):
    from agents.registry import _exec_structure_gen
    captured=[]
    monkeypatch.setattr('agents.registry.slurm.submit_and_return',lambda **kwargs:captured.append(kwargs) or {'submitted':False})
    _exec_structure_gen({'material_type':'MOF','n_structures':2,'output_dir':str(tmp_path),'partition':'bigcpu','nodelist':'node07','memory_mb':4096,'cpus_per_task':2,'walltime':'00:05:00'})
    assert captured and captured[0]['partition']=='bigcpu' and captured[0]['nodelist']=='node07'
    assert captured[0]['memory_mb']==4096 and captured[0]['cpus_per_task']==2


def test_completed_generation_native_guard_never_submits_into_existing_dataset(tmp_path, monkeypatch):
    from agents.registry import _exec_structure_gen
    (tmp_path / 'small').mkdir()
    original = tmp_path / 'small' / 'original.cif'
    original.write_text('data_original_fixture')
    monkeypatch.setattr('agents.registry.slurm.submit_and_return', lambda **kwargs: pytest.fail('must not replay'))
    result = _exec_structure_gen({'material_type': 'MOF', 'n_structures': 100, 'output_dir': str(tmp_path)})
    assert result['blocked'] and not result['executed']
    assert original.read_text() == 'data_original_fixture'


def test_same_plan_can_reconcile_dialogue_version_but_not_science(tmp_path):
    store = WorkflowStore(tmp_path / 'state.json')
    contract = {'step_id': 'calc', 'agent': 'analyst', 'tool': 'read_file', 'arguments': {'path': 'input.cif'},
        'depends_on': [], 'expected_outputs': [], 'resources': []}
    goal = {'version': 1, 'approved_plan_version': 1, 'method': 'CDFT', 'gases': ['Xe', 'Kr'],
        'parameters': {'temperature_K': 298}, 'execution_mode': 'workflow'}
    store.activate('wf', [contract], goal, 'user', 'session')
    newer = {**goal, 'version': 2, 'active_goal': 'How is the existing calculation progressing?'}
    state = store.activate('wf', [contract], newer, 'user', 'session')
    assert state['goal_contract']['version'] == 2 and state['nodes']['calc']['status'] == 'pending'
    assert state['goal_reconciliation_history']
    with pytest.raises(ValueError, match='goal changed'):
        store.activate('wf', [contract], {**newer, 'version': 3, 'parameters': {'temperature_K': 310}}, 'user', 'session')


def test_control_roles_do_not_grant_root_file_access():
    from agents.control_policy import role_denial, bypasses_data_lease
    for role in ['monitor', 'supervisor', 'harness-maintainer', '']:
        assert role_denial(role, 'cancel_watched_job')
        assert not bypasses_data_lease(role, 'retarget_queued_job')
    for role in ['lead-orchestrator', 'monitor', 'supervisor']:
        assert bypasses_data_lease(role, 'diagnose_job')
        assert not bypasses_data_lease(role, 'run_bash')
    assert bypasses_data_lease('lead-orchestrator', 'cancel_watched_job')
    assert role_denial('monitor', 'write_file')
    assert not role_denial('supervisor', 'supervisor_decision')


def test_control_checks_provenance_not_natural_language():
    from agents.control_policy import require_user_control_source
    owner={'username':'a','conv_id':'c'}
    # These are source-validation checks, not expectations about intent.
    # Actual question/approval/withdrawal decisions are exercised by the SDK.
    for text in ['嗯','按刚才那个来','不要取消','can it be stopped?']:
        require_user_control_source({'origin':'user','owner':owner,'text':text},owner)
    with pytest.raises(PermissionError):require_user_control_source({'origin':'system','owner':owner},owner)
    with pytest.raises(PermissionError):require_user_control_source({'origin':'user','owner':{'username':'b','conv_id':'c'}},owner)


@pytest.mark.parametrize('returncode,state,terminal,confirmed', [(1,'PENDING',False,False),(0,'RUNNING',False,False),(0,'UNKNOWN',False,False),(0,'CANCELLED',True,True)])
def test_cancel_requires_scheduler_proof_and_stays_local(tmp_path,monkeypatch,returncode,state,terminal,confirmed):
    from agents.job_watch import JobWatch
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c')
    commands=[]
    def run(argv,**kwargs):
        commands.append(argv)
        return SimpleNamespace(returncode=returncode,stdout='',stderr='denied' if returncode else '')
    monkeypatch.setattr('subprocess.run',run)
    monkeypatch.setattr('agents.slurm.check_job_status',lambda *args,**kwargs:{'status':state,'terminal':terminal,'failed':terminal})
    result=watch.cancel('123',control_root=tmp_path)
    assert result['cancelled'] == confirmed and result['request_sent'] == (returncode==0)
    assert commands == [['scancel','123']]
    assert watch.get('123')['terminal'] == confirmed
    assert bool(watch.drain_notifications()) == confirmed
    if returncode:
        assert not watch.get('123').get('cancelled_by_user')
    elif not confirmed:
        watch.cancel('123',control_root=tmp_path)
        assert commands == [['scancel','123']]


def test_cancel_and_retarget_share_control_lock_without_removing_data_lease(tmp_path,monkeypatch):
    from agents.job_watch import JobWatch
    from agents.state_io import ConversationLock
    watch=JobWatch(str(tmp_path/'watch.json'));watch.register('123',username='a',conv_id='c')
    lock=ConversationLock(tmp_path/'job_control/123.lock')
    assert lock.acquire(blocking=False)
    try:
        result=watch.cancel('123',control_root=tmp_path)
        assert result['blocked'] and not result['request_sent']
        assert not watch.get('123')['terminal']
    finally:lock.release()


def test_user_stop_freezes_descendants_but_preserves_live_claim(tmp_path):
    from agents.job_control import pause_owned_workflow
    store=WorkflowStore(tmp_path/'data/state/parallel_workflows.json')
    identity=store.identity('a','c')
    node={'step_id':'gen','agent':'harness','tool':'generate_structure','arguments':{},'depends_on':[],
          'expected_outputs':['x.cif'],'resources':[{'key':'path:/protected','mode':'write'}]}
    store.activate(identity,[node],{'version':1,'approved_plan_version':1},'a','c')
    claim=store.claim(identity,'gen');leases=copy.deepcopy(store.snapshot()['leases'])
    pause_owned_workflow(tmp_path,'a','c')
    state=store.snapshot(identity)
    assert state['user_paused'] and state['nodes']['gen']['token']==claim['token']
    assert store.snapshot()['leases']==leases
