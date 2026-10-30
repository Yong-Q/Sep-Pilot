"""Dead readonly dispatches recover through normal claim; no scientific submits."""
import copy
import os
import pytest
from agents.state_io import json_transaction
from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
from agents.registry import _build_default_registry
from agents.session import Session
from agents.task_line import TaskLineStore


def node(key,tool='read_file'):
    return {'step_id':key,'agent':'analyst','tool':tool,'arguments':{'path':'input.txt'},
            'depends_on':[],'expected_outputs':[]}


@pytest.fixture
def factory(tmp_path):
    runtimes=[]
    def build(nodes):
        root=tmp_path/'runs/tester/c1';root.mkdir(parents=True)
        main=Session(config=AgentConfig(api_key='test',project_root=tmp_path),registry=_build_default_registry())
        main.goal_contract=GoalContract.from_user_message('hello')
        main.goal_contract.execution_authorized=True
        main.goal_contract.approved_plan_version=1
        main.goal_contract.approved_nodes=copy.deepcopy(nodes)
        lines=TaskLineStore(str(tmp_path/'lines.json'))
        line=lines.begin_line('c1',username='tester',conv_id='c1')
        main._current_line_id=line['line_id']
        lines.apply_workflow_patch(line['line_id'],[{'operation':'upsert','step_id':n['step_id'],'node':n} for n in nodes],0,1,'test approved DAG')
        mailbox=LifecycleStore(root/'lifecycle.json');mailbox.initialize()
        rt=ParallelWorkflow(main,root,WorkflowStore(tmp_path/'workflow.json'),mailbox,lines,'tester','c1')
        main._runtime_snapshot=rt.snapshot
        runtimes.append(rt)
        return rt
    yield build
    for rt in runtimes:rt.shutdown()


DEAD_PID=2147483647

def orphan(factory, status='pending', tool='read_file'):
    rt=factory([node('verify',tool=tool)])
    rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'verify')
    with json_transaction(rt.store.path) as data:
        w=data['workflows'][rt.workflow_id];n=w['nodes']['verify']
        n.update(status=status,owner_pid=DEAD_PID,dispatch_phase='entered_tool',
                 tool_returned=False,error='worker disappeared; unknown mutations are never redispatched')
        data['leases'][ticket['token']]['owner_pid']=DEAD_PID
    rt.task_lines.upsert_step(rt.main._current_line_id,'verify',username=rt.username,conv_id=rt.conv_id,status='running')
    return rt,ticket['token']

def recover(rt,original_token,**overrides):
    args=dict(workflow_id=rt.workflow_id,step_id='verify',token=original_token,
              goal=rt.main.goal_contract.to_dict(),username=rt.username,conv_id=rt.conv_id)
    args.update(overrides)
    return rt.store.recover_readonly_orphan(**args)

@pytest.mark.parametrize('status',['running','pending'])
def test_dead_readonly_dispatch_clears_identity_and_preserves_audit(factory,status):
    rt,token=orphan(factory,status)
    old=copy.deepcopy(rt.snapshot()['nodes']['verify'])
    assert recover(rt,token)
    recovered=rt.snapshot()['nodes']['verify']
    assert recovered['status']=='pending'
    assert not recovered.get('dispatch_phase') and not recovered.get('token')
    assert recovered['attempt_history'][-1]['dispatch_phase']=='entered_tool'
    assert recovered['attempt_history'][-1]['error']==old['error']
    assert recovered['attempt_history'][-1]['owner_pid']==DEAD_PID
    assert token not in rt.store.snapshot()['leases']
    assert not rt.store.update(rt.workflow_id,'verify',token,{'status':'succeeded'})
    assert not recover(rt,token)
    new=rt.store.claim(rt.workflow_id,'verify')
    assert new and new['token']!=token and new['dispatch_phase']=='preflight'

@pytest.mark.parametrize('change',['live_pid','user_pause','token','owner','goal','plan','contract','job','returned','foreign_lease','nonreadonly','legacy_error','origin_goal','execution_plan','pause_kind','second_lease'])
def test_readonly_recovery_rejects_changed_authority_without_mutation(factory,change):
    rt,token=orphan(factory)
    args={}
    with json_transaction(rt.store.path) as data:
        w=data['workflows'][rt.workflow_id];n=w['nodes']['verify']
        if change=='live_pid':n['owner_pid']=os.getpid()
        elif change=='user_pause':w['user_paused']=True
        elif change=='token':args['token']='other'
        elif change=='owner':args['username']='other'
        elif change=='goal':args['goal']={**rt.main.goal_contract.to_dict(),'version':99}
        elif change=='plan':w['plan_version']=99
        elif change=='contract':n['contract']['arguments']['path']='changed'
        elif change=='job':n['job_ids']=['42']
        elif change=='returned':n['tool_returned']=True
        elif change=='foreign_lease':data['leases'][token]['workflow_id']='other'
        elif change=='nonreadonly':n['contract']['tool']='run_pore_analysis'
        elif change=='legacy_error':n['error']='unrelated failure'
        elif change=='origin_goal':n['origin_goal_version']=99
        elif change=='execution_plan':n['execution_plan_version']=99
        elif change=='pause_kind':w['pause_kind']='user_change'
        elif change=='second_lease':data['leases']['new-owner']={'workflow_id':rt.workflow_id,'step_id':'verify','owner_pid':os.getpid()}
    before=rt.store.snapshot()
    assert not recover(rt,token,**args)
    assert rt.store.snapshot()==before


def test_tick_dead_readonly_owner_does_not_leave_unknown_dispatch_identity(factory,monkeypatch):
    rt,token=orphan(factory,'running')
    monkeypatch.setattr(rt.store,'claim',lambda *args:None)
    rt.tick()
    n=rt.snapshot()['nodes']['verify']
    assert n['status']=='pending'
    assert not n.get('dispatch_phase') and not n.get('token')
    assert n['attempt_history'][-1]['token']==token
    assert next(s for s in rt._line()['steps'] if s['step_id']=='verify')['status']=='pending'



def test_recovered_dispatch_can_take_formal_patch_without_stale_live_identity(factory):
    rt,token=orphan(factory)
    assert recover(rt,token)
    replacement=copy.deepcopy(rt.main.goal_contract.approved_nodes[0])
    replacement['agent']='lead-orchestrator'
    line=rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation':'upsert','step_id':'verify','node':replacement}],1,2,'formal agent assignment repair',
        runtime_nodes=rt.snapshot()['nodes'])
    assert line['plan_version']==2
    assert line['steps'][0]['status']=='pending'


def test_tick_dispatches_one_new_readonly_attempt_through_normal_claim(factory,monkeypatch):
    rt,token=orphan(factory)
    observed=[]
    def fake_readonly_worker(step_id,ticket):
        observed.append((step_id,ticket['token'],ticket['dispatch_phase']))
        rt.store.update(rt.workflow_id,step_id,ticket['token'],{'status':'prefinish','tool_returned':True},release=True)
    monkeypatch.setattr(rt,'_worker',fake_readonly_worker)
    rt.tick()
    for future in list(rt.futures.values()):future.result(timeout=3)
    assert len(observed)==1 and observed[0][1]!=token and observed[0][2]=='preflight'
    assert rt.snapshot()['nodes']['verify']['attempt_history'][-1]['token']==token
    assert not rt.store.update(rt.workflow_id,'verify',token,{'status':'succeeded'})


@pytest.mark.parametrize('pause_kind',['failure','reconciliation'])
def test_readonly_recovery_releases_only_its_own_global_dead_worker_pause(factory,monkeypatch,pause_kind):
    rt,token=orphan(factory)
    with json_transaction(rt.store.path) as data:
        w=data['workflows'][rt.workflow_id]
        w.update(status='needs_user',pause_kind=pause_kind,pause_reason='dead worker requires dispatch reconciliation')
    submitted=[]
    monkeypatch.setattr(rt.pool,'submit',lambda *args:submitted.append(args) or __import__('concurrent.futures').futures.Future())
    rt.tick()
    assert len(submitted)==1
    assert rt.snapshot()['status']=='active'


def test_readonly_recovery_keeps_other_unknown_global_pause(factory):
    rt,token=orphan(factory)
    with json_transaction(rt.store.path) as data:
        w=data['workflows'][rt.workflow_id]
        w.update(status='needs_user',pause_kind='reconciliation')
        w['nodes']['science']={'status':'uncertain','contract':{'tool':'run_cdft'}}
    assert recover(rt,token)
    assert rt.snapshot()['status']=='needs_user'
    assert rt.snapshot()['nodes']['science']['status']=='uncertain'


def test_retired_readonly_attempt_cannot_settle_or_pause_current_execution(factory):
    rt,token=orphan(factory)
    assert recover(rt,token)
    before=rt.store.snapshot()
    before_line=copy.deepcopy(rt._line())
    with pytest.raises(ValueError,match='ownership changed'):
        rt._settle('verify',token,{'error':'late result from retired dispatch'})
    assert rt.store.snapshot()==before
    assert rt._line()==before_line
