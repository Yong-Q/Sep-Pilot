import json
import time
from types import SimpleNamespace
import pytest
from agents.recovery import SUBMISSION_TOOLS

from agents.job_watch import JobWatch
from agents.node_inventory import candidates
from agents.registry import _build_default_registry, _exec_cdft


def test_pending_warning_is_durable_nonterminal_and_rate_limited(tmp_path, monkeypatch):
    from agents import slurm
    watch=JobWatch(str(tmp_path/'watch.json'))
    watch.register('123',username='a',conv_id='c')
    watch._jobs['123']['pending_since_epoch']=time.time()-600
    monkeypatch.setattr(slurm,'batch_fetch_states',lambda ids:{'123':{'status':'PENDING','terminal':False,'failed':False,'pending_reason':'Resources'}})
    watch._poll_once()
    notification=watch.drain_notifications()[0]
    assert notification['kind']=='job_pending_warning'
    assert not watch.get('123')['terminal'] and not watch.get('123')['failed']
    assert not watch.get('123')['notified']
    restored=JobWatch(str(tmp_path/'watch.json'))
    assert restored.get('123')['pending_reason']=='Resources'
    watch.ack_notification('123',notification['notification_id'])
    watch._poll_once()
    assert not watch.drain_notifications()


def test_warning_ack_cannot_erase_later_completion(tmp_path):
    watch=JobWatch(str(tmp_path/'watch.json'))
    watch.register('123',username='a',conv_id='c')
    watch._queue_notification(watch._jobs['123'],{'job_id':'123','kind':'job_pending_warning','state':'PENDING'})
    warning=watch.drain_notifications()[0]
    watch._notify_completion('123','COMPLETED')
    completion=watch.get('123')['pending_notification']
    assert completion['notification_id']!=warning['notification_id']
    watch.ack_notification('123',warning['notification_id'])
    assert watch.get('123')['pending_notification']['state']=='COMPLETED'
    assert watch.drain_notifications()[0]['state']=='COMPLETED'


def test_cpu_availability_does_not_hide_insufficient_ram():
    snapshot={'verified':True,'stale':False,'policy':{'preferred_nodes':[],'excluded_nodes':[],'allowed_states':['mix']},
        'nodes':{'n':{'name':'n','states':['mix'],'cpus':{'idle':208,'total':256},'memory':{'verified':True,'available_for_scheduling_mib':512000}}}}
    assert candidates(snapshot,cpus=40,memory_mb=16384)
    assert not candidates(snapshot,cpus=40,memory_mb=758098)


def test_resource_tools_exposed_only_as_readonly_proposals():
    from agents.defns import MONITOR
    assert {'resource_health','assess_job_resources','resource_review_decision'}<=set(MONITOR.function_map())
    registry=_build_default_registry()
    assert not registry.validate_params('run_cdft',{'action':'pipeline','cif_dir':'structures','gas':'Kr','temperature':298,'memory_mb':16384})
    assert registry.validate_params('run_cdft',{'action':'pipeline','cif_dir':'structures','gas':'Kr','memory_mb':0})


def test_native_missing_ram_budget_never_invokes_helper(monkeypatch):
    monkeypatch.setattr('subprocess.run',lambda *a,**k:(_ for _ in ()).throw(AssertionError('must not submit without budget')))
    result=_exec_cdft({'action':'pipeline','gas':'Kr'})
    assert result['executed'] is False and result['status']=='needs_resource_review'


def test_resource_reviewer_rejects_unverified_decision_then_uses_actual_ids(tmp_path,monkeypatch):
    from anthropic.types import ToolUseBlock
    from agents.session import Session
    from agents.config import AgentConfig
    from agents.resource_review import run_resource_review
    session=Session(config=AgentConfig(api_key='test',project_root=tmp_path))
    monkeypatch.setattr(session.registry,'execute_dict',lambda *a,**k:{'read_only':True,'cluster_verified':True})
    step=0
    def response(agent):
        nonlocal step
        step+=1
        if step==2:name,args='resource_health',{}
        else:
            name='resource_review_decision'
            args={'status':'needs_user','reason':'unknown memory budget','evidence_call_ids':['invented'] if step==1 else [c['call_id'] for c in session.memory.tool_call_log if c['tool']=='resource_health']}
        return SimpleNamespace(content=[ToolUseBlock(type='tool_use',id=str(step),name=name,input=args)])
    session._call_api=response
    receipt=run_resource_review(session,{'kind':'before_submission'})
    assert step==3 and receipt['resource_review_id']
    assert session.memory.tool_call_log[0]['failed']
    assert receipt['status']=='needs_user' and receipt['not_resubmitted']


def test_missing_resource_budget_stays_with_resource_agent_without_user_veto(tmp_path,monkeypatch):
    from agents.session import Session
    from agents.config import AgentConfig
    from agents.defns import ORCHESTRATOR
    from agents.goal_contract import GoalContract
    from agents.task_line import TaskLineStore
    store=TaskLineStore(str(tmp_path/'lines.json'));store.begin_line('c1')
    monkeypatch.setattr('agents.task_line.get_store',lambda:store)
    session=Session(config=AgentConfig(api_key='test',project_root=tmp_path));session._current_line_id='c1';session.current_agent=ORCHESTRATOR
    session.goal_contract=GoalContract.from_user_message('用cDFT计算Kr，298K，直接执行')
    session._on_resource_review=lambda request:{'status':'needs_user','reason':'finite memory budget unknown'}
    session._on_workflow_start=lambda *a:(_ for _ in ()).throw(AssertionError('must not dispatch'))
    node={'agent':'analyst','tool':'run_cdft','arguments':{'action':'pipeline','cif_dir':'cifs','gas':'Kr','temperature':298},'depends_on':[],'expected_outputs':[]}
    session._record_workflow_draft([{'step_id':'kr','description':'Run and validate Kr cDFT','agent':'analyst',
        'tool':'run_cdft','depends_on':[],'missing_parameters':[]}],completion_criteria='Verified Kr result')
    result=session._propose_workflow_patch(0,'Compile bounded resource workflow',[{'operation':'upsert','step_id':'kr','node':node}])
    assert result['status']=='waiting_resources' and not session._waiting_for_user_input
    assert session.context['pending_resource_plan']['changes']
    assert not session.goal_contract.approved_nodes and not store.get_line('c1')['steps']


def test_busy_compatible_node_is_eligible_for_slurm_queue_not_immediate_capacity():
    snapshot={'verified':True,'stale':False,'policy':{'preferred_nodes':[], 'excluded_nodes':[], 'allowed_states':['idle','mix']},
        'nodes':{'n':{'name':'n','states':['alloc'],'cpus':{'idle':0,'total':40}, 'memory_mib':64000,
                     'memory':{'verified':True,'available_for_scheduling_mib':0}}}}
    assert not candidates(snapshot, cpus=4, memory_mb=8192)
    assert candidates(snapshot, cpus=4, memory_mb=8192, for_queue=True)
    assert not candidates(snapshot, cpus=4, memory_mb=128000, for_queue=True)
    snapshot['nodes']['n']['states']=['down']
    assert not candidates(snapshot, cpus=4, memory_mb=8192, for_queue=True)


@pytest.mark.parametrize('tool', sorted(SUBMISSION_TOOLS | {'run_cdft', 'run_vasp'}))
def test_resource_agent_budget_can_submit_to_busy_capacity(tool):
    from agents.resource_review import allocate_reviewed_resources
    snapshot={'verified':True,'stale':False,'policy':{'preferred_nodes':[], 'excluded_nodes':[], 'allowed_states':['idle','mix'],
        'tool_profiles':{'run_cdft':{'default_cpus':4,'partition_preference':['compute']}}},
        'nodes':{'n':{'name':'n','partitions':['compute'],'states':['alloc'],'cpus':{'idle':0,'total':40},
                     'memory_mib':64000,'memory':{'verified':True,'available_for_scheduling_mib':0}}}}
    session=SimpleNamespace(memory=SimpleNamespace(tool_call_log=[{'tool':'resource_health','call_id':'proof'}]),
                            _load_evidence_call=lambda c:{'result':json.dumps({'node_inventory':snapshot})})
    receipt=allocate_reviewed_resources(session, {'nodes':[{'step_id':'x','tool':tool,'arguments':{'action':'submit'}}]},
        {'status':'ready','resource_review_id':'review','evidence_call_ids':['proof'],'suggested_resources':{'memory_mb':8192}})
    assert receipt['status']=='ready'
    assert receipt['resource_allocations']['x']['scheduling_mode']=='queue'
    assert receipt['resource_allocations']['x']['memory_mb']==8192


def test_common_scheduler_receives_reviewed_memory_without_cross_session_leak():
    from agents.slurm import render_sbatch_script
    from agents.watch_context import set_context, clear_context
    set_context('u', 'c', resource_allocation={'resource_review_id':'proof', 'memory_mb':8192,
                                             'partition':'compute', 'nodelist':'n', 'cpus':4})
    try:
        script = render_sbatch_script('probe', 'true', '/tmp/probe')
        assert '#SBATCH --mem=8192M' in script
        assert '#SBATCH --cpus-per-task=4' in script
        assert '#SBATCH --nodelist=n' in script
    finally:
        clear_context()
    assert '#SBATCH --mem=8192M' not in render_sbatch_script('probe', 'true', '/tmp/probe')
