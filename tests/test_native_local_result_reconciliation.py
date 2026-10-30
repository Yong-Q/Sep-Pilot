"""Proof-backed local receipt recovery never executes a writer or scientific job."""
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import pytest
from agents.parallel_workflow import WorkflowStore
from agents.state_io import json_transaction

DEAD_PID=2147483647

@pytest.fixture
def proven(tmp_path,monkeypatch):
    contract={'step_id':'fragment','agent':'analyst','tool':'generate_scientific_report',
        'arguments':{'report_mode':'fragment','output_path':str(tmp_path/'fragment.md')},
        'expected_outputs':[str(tmp_path/'fragment.md')],'depends_on':[],
        'resources':[{'key':'path:'+str(tmp_path/'fragment.md'),'mode':'write'}]}
    goal={'version':1,'approved_plan_version':1,'execution_authorized':True,'approved_nodes':[copy.deepcopy(contract)]}
    n={'contract':contract,'status':'uncertain','token':'current','owner_pid':DEAD_PID,
       'origin_goal_version':1,'execution_plan_version':1,'dispatch_phase':'entered_tool',
       'tool_returned':False,'job_ids':[],'error':'worker disappeared; unknown mutations are never redispatched',
       'attempt_history':[],'started_at':100}
    w={'username':'tester','conv_id':'c1','plan_version':1,'goal_contract':copy.deepcopy(goal),
       'status':'needs_user','pause_kind':'reconciliation','pause_reason':'dead worker requires dispatch reconciliation',
       'nodes':{'fragment':n}}
    p=tmp_path/'store.json';p.write_text(json.dumps({'workflows':{'w':w},'leases':{
        'current':{'workflow_id':'w','step_id':'fragment','owner_pid':DEAD_PID,'resources':contract['resources']}}}))
    (tmp_path/'fragment.md').write_text('Immutable verified native fragment')
    proof={'kind':'archived_verified_report_fragment','proof_sha256':'exact-proof',
        'username':'tester','conv_id':'c1','goal_version':1,'plan_version':1,'current_token':'current',
        'result':{'path':str(tmp_path/'fragment.md')},'result_ref':{'call_id':'originalcall'},
        'output_baseline':{},'started_at':50,'artifacts':{str(tmp_path/'fragment.md'):{'size':34}},
        'no_resubmission':True,'no_output_write':True}
    calls=[]
    def verify(state,step_id,session_root,approved_goal=None):
        calls.append((state['nodes'][step_id]['token'],session_root))
        return copy.deepcopy(proof)
    monkeypatch.setitem(sys.modules,'agents.report_recovery',SimpleNamespace(
        verify_archived_report_fragment=verify,verify_returned_readonly=verify))
    store=WorkflowStore(p)
    def commit(**overrides):
        args=dict(workflow_id='w',step_id='fragment',token='current',goal=goal,username='tester',conv_id='c1',session_root=tmp_path,proof=copy.deepcopy(proof))
        args.update(overrides)
        return store.reconcile_proven_local_result(**args)
    return SimpleNamespace(store=store,goal=goal,proof=proof,calls=calls,commit=commit,root=tmp_path)


def test_verified_archived_report_becomes_prefinish_not_completed(proven):
    old=proven.store.snapshot('w')['nodes']['fragment']
    assert proven.commit()
    current=proven.store.snapshot('w')['nodes']['fragment']
    assert current['status']=='prefinish' and current['phase']=='prefinish'
    assert not current.get('node_verification')
    assert current['result_ref']['call_id']=='originalcall'
    assert current['attempt_history'][-1]['error']==old['error']
    assert current['started_at']==50
    assert current['restoration']['no_output_write'] is True
    assert not proven.store.snapshot()['leases']
    assert len(proven.calls)==1
    assert not proven.commit()


@pytest.mark.parametrize('history', ['valid', 'missing', 'unverified', 'gap'])
def test_reconciled_metadata_origin_requires_contiguous_native_proof(proven, history):
    """A metadata-only revision must not strand a completed native receipt."""
    next_goal={**proven.goal,'version':2}
    proven.proof['goal_version']=2
    entry={'from_version':1,'to_version':2,
        'basis':'identical protected scope and approved node contracts',
        'metadata_only_verified':True}
    if history=='unverified':entry['metadata_only_verified']=False
    if history=='gap':entry['from_version']=0
    with json_transaction(proven.store.path) as data:
        workflow=data['workflows']['w'];workflow['goal_contract']=copy.deepcopy(next_goal)
        if history!='missing':workflow['goal_reconciliation_history']=[entry]
    before=proven.store.snapshot()
    assert proven.commit(goal=next_goal) is (history=='valid')
    if history=='valid':
        assert proven.store.snapshot('w')['nodes']['fragment']['status']=='prefinish'
    else:
        assert proven.store.snapshot()==before


@pytest.mark.parametrize('change',['token','owner','goal','plan','pause','live_pid','job','foreign_lease','source_or_file_proof','mutator','changed_contract'])
def test_native_proof_reconciliation_denies_changed_authority_without_writes(proven,change):
    args={}
    with json_transaction(proven.store.path) as data:
        w=data['workflows']['w'];n=w['nodes']['fragment']
        if change=='token':args['token']='stale'
        elif change=='owner':args['username']='other'
        elif change=='goal':args['goal']={**proven.goal,'version':99}
        elif change=='plan':w['plan_version']=99
        elif change=='pause':w['user_paused']=True
        elif change=='live_pid':n['owner_pid']=os.getpid()
        elif change=='job':n['job_ids']=['42']
        elif change=='foreign_lease':data['leases']['other']={'workflow_id':'different','step_id':'x','owner_pid':os.getpid(),'resources':n['contract']['resources']}
        elif change=='source_or_file_proof':args['proof']={**proven.proof,'proof_sha256':'stale-proof'}
        elif change=='mutator':n['contract']['tool']='write_file'
        elif change=='changed_contract':n['contract']['arguments']['output_path']='changed.md'
    before=proven.store.snapshot()
    assert not proven.commit(**args)
    assert proven.store.snapshot()==before


@pytest.mark.parametrize('kind',['report','readonly'])
def test_real_native_proof_produces_bounded_current_attempt_dossier(tmp_path,kind,monkeypatch):
    from tests.test_report_recovery import build_report_recovery_case,build_returned_readonly_case
    from agents.report_recovery import verify_archived_report_fragment,verify_returned_readonly
    from agents.parallel_workflow import ParallelWorkflow
    from agents import result_validation
    c=(build_report_recovery_case if kind=='report' else build_returned_readonly_case)(tmp_path)
    step='report' if kind=='report' else 'qc';w=c['state'];n=w['nodes'][step]
    w.update(status='needs_user',pause_kind='reconciliation',pause_reason='dead worker requires dispatch reconciliation')
    n.update(owner_pid=DEAD_PID,dispatch_phase='entered_tool')
    goal=w['goal_contract'];path=tmp_path/'workflow.json'
    path.write_text(json.dumps({'workflows':{'w':w},'leases':{}}));store=WorkflowStore(path)
    verify=verify_archived_report_fragment if kind=='report' else verify_returned_readonly
    proof=verify(w,step,c['root'],approved_goal=goal)
    assert store.reconcile_proven_local_result('w',step,n['token'],goal,'owner','conv',c['root'],proof)
    current=store.snapshot('w')['nodes'][step]
    rt=object.__new__(ParallelWorkflow);rt.store=store;rt.workflow_id='w';rt.root=c['root']
    rt.main=SimpleNamespace(goal_contract=SimpleNamespace(to_dict=lambda:copy.deepcopy(goal)))
    rt.username='owner';rt.conv_id='conv'
    monkeypatch.setattr(result_validation,'_inventory',lambda *args:pytest.fail('restored native proof must not scan a directory'))
    dossier=rt._create_validation_dossier(step,current,current['path_manifest'],current['result'])
    assert dossier['verdict']=='pass'
    assert dossier['requires_agent_review'] is True
    assert dossier['attempt_id']==current['token']
    assert dossier['contract']==current['contract']
    assert dossier['method_facts']['native_result_restoration']['proof_sha256']==proof['proof_sha256']
    evidence=json.loads(Path(dossier['evidence_path']).read_text())
    assert evidence['tool']=='inspect_workflow_result' and evidence['result']['step_id']==step
    assert store.snapshot('w')['nodes'][step]['status']=='prefinish'


@pytest.mark.parametrize('kind,tamper',[('report',None),('readonly',None),('report','report'),('report','source_path'),('readonly','call'),('report','commit_report'),('report','commit_source'),('readonly','pause_commit'),('report','none_pending'),('readonly','none_pending')])
def test_real_proof_dossier_can_finish_via_normal_scoped_workflow(tmp_path,kind,tamper,monkeypatch):
    from tests.test_report_recovery import build_report_recovery_case,build_returned_readonly_case
    from agents.report_recovery import verify_archived_report_fragment,verify_returned_readonly
    from agents.parallel_workflow import ParallelWorkflow
    from agents.config import AgentConfig
    from agents.goal_contract import GoalContract
    from agents.registry import _build_default_registry
    from agents.session import Session
    from agents.task_line import TaskLineStore
    from agents.lifecycle import LifecycleStore
    c=(build_report_recovery_case if kind=='report' else build_returned_readonly_case)(tmp_path)
    w=c['state'];step='report' if kind=='report' else 'qc';n=w['nodes'][step]
    if kind=='report':
        source_contract={'step_id':'source','agent':'analyst','tool':'read_file',
            'arguments':{'path':str(c['source_path'])},'depends_on':[],'expected_outputs':[]}
        w['nodes']['source']['contract']=source_contract
        w['goal_contract']['approved_nodes'].insert(0,copy.deepcopy(source_contract))
        source_call=json.loads(c['source_path'].read_text())
        source_call.update(tool='read_file',params=source_contract['arguments'],result={'content':'Synthetic source fact'})
        c['source_path'].write_text(json.dumps(source_call,sort_keys=True))
    if tamper=='none_pending':
        child={'step_id':'downstream','agent':'analyst','tool':'read_file',
               'arguments':{'path':str(c['report'])},'depends_on':[step],'expected_outputs':[]}
        w['nodes']['downstream']={'status':'pending','contract':child}
        w['goal_contract']['approved_nodes'].append(copy.deepcopy(child))
    main=Session(config=AgentConfig(api_key='test',project_root=tmp_path),registry=_build_default_registry())
    main.goal_contract=GoalContract.from_dict({**w['goal_contract'],'schema_version':4})
    w['goal_contract']=main.goal_contract.to_dict()
    if kind=='report':
        import hashlib
        from agents.parallel_workflow import execution_fingerprint
        fingerprint=execution_fingerprint(source_contract,main)
        w['nodes']['source'].update(execution_fingerprint=fingerprint,phase='finish',artifacts={},node_verification={'evidence_call_ids':['source_call']})
        c['history']['result'].update(source_attempt_fingerprint=fingerprint,
            evidence_fingerprint=hashlib.sha256(c['source_path'].read_bytes()).hexdigest())
        call=json.loads(c['call_path'].read_text());call['result']=json.dumps(c['history']['result'])
        c['call_path'].write_text(json.dumps(call))
    if tamper=='none_pending':
        from agents.parallel_workflow import resources_for
        child['resources']=resources_for(child,tmp_path,c['root'])
    w.update(status='needs_user',pause_kind='reconciliation',pause_reason='dead worker requires dispatch reconciliation')
    if tamper=='none_pending':
        w.pop('pause_kind');w.pop('pause_reason')
    n.update(owner_pid=DEAD_PID,dispatch_phase='entered_tool')
    path=tmp_path/'workflow.json';store=WorkflowStore(path);wid=store.identity('owner','conv')
    path.write_text(json.dumps({'workflows':{wid:w},'leases':{}}))
    lines=TaskLineStore(str(tmp_path/'lines.json'));line=lines.begin_line('conv',username='owner',conv_id='conv')
    main._current_line_id=line['line_id']
    lines.apply_workflow_patch(line['line_id'],[{'operation':'upsert','step_id':a['step_id'],'node':a} for a in main.goal_contract.approved_nodes],0,7,'approved fixture')
    for key in w['nodes']:
        status='pending' if key=='downstream' else 'running' if key==step else 'completed'
        lines.upsert_step(line['line_id'],key,username='owner',conv_id='conv',status=status,done=status=='completed',validation={'output_contract':'passed'} if status=='completed' else {})
    lifecycle=LifecycleStore(c['root']/'lifecycle.json');lifecycle.initialize()
    rt=ParallelWorkflow(main,c['root'],store,lifecycle,lines,'owner','conv');main._runtime_snapshot=rt.snapshot
    # Completion may check readiness, but the unit test forbids any scientific dispatch.
    dispatched=[]
    def fake_submit(*args):
        assert tamper=='none_pending' and args[1]=='downstream'
        dispatched.append(args[1])
        from concurrent.futures import Future
        future=Future();future.set_result(None);return future
    monkeypatch.setattr(rt.pool,'submit',fake_submit)
    verify=verify_archived_report_fragment if kind=='report' else verify_returned_readonly
    try:
        proof=verify(w,step,c['root'],approved_goal=main.goal_contract.to_dict())
        assert store.reconcile_proven_local_result(wid,step,n['token'],main.goal_contract.to_dict(),'owner','conv',c['root'],proof)
        current=rt.snapshot()['nodes'][step]
        dossier=rt._create_validation_dossier(step,current,current['path_manifest'],current['result'])
        assert dossier['verdict']=='pass',dossier.get('reasons')
        if tamper in {'commit_report','commit_source','pause_commit'}:
            original_commit=store.finish_proven_local_result
            def change_at_commit(*args,**kwargs):
                if tamper=='pause_commit':
                    store.user_pause(wid)
                else:
                    target=c['report'] if tamper=='commit_report' else c['source_path']
                    before_stat=target.stat();data=target.read_bytes()
                    target.write_bytes(bytes([data[0]^1])+data[1:])
                    os.utime(target,ns=(before_stat.st_atime_ns,before_stat.st_mtime_ns))
                return original_commit(*args,**kwargs)
            monkeypatch.setattr(store,'finish_proven_local_result',change_at_commit)
            with pytest.raises(ValueError,match='before finish commit'):
                rt.finish_node(step,[dossier['evidence_call_id']],'A last-boundary proof or user pause change must block completion.')
            assert rt.snapshot()['nodes'][step]['status']=='prefinish'
            assert next(s for s in rt._line()['steps'] if s['step_id']==step)['status']!='completed'
            if tamper=='pause_commit':assert rt.snapshot()['user_paused'] is True
            return
        if tamper and tamper!='none_pending':
            target=c['root']/'evidence/readonly_call.json' if tamper=='call' else c[tamper]
            before_stat=target.stat();data=target.read_bytes()
            target.write_bytes(bytes([data[0]^1])+data[1:])
            os.utime(target,ns=(before_stat.st_atime_ns,before_stat.st_mtime_ns))
            with pytest.raises((ValueError,PermissionError)):
                rt.finish_node(step,[dossier['evidence_call_id']],'Earlier dossier must not approve changed native source or bytes.')
            assert rt.snapshot()['nodes'][step]['status']=='prefinish'
            return
        result=rt.finish_node(step,[dossier['evidence_call_id']],'Verified exact native source identity and unchanged owned output; no new scientific execution.')
        assert result['ok'] and result['phase']=='finish'
        assert rt.snapshot()['nodes'][step]['status']=='succeeded'
        assert rt.snapshot()['nodes'][step]['node_verification']['evidence_call_ids']==[dossier['evidence_call_id']]
        if tamper=='none_pending':
            assert dispatched==['downstream']
            assert rt.snapshot()['status']=='active'
            assert rt.snapshot()['nodes']['downstream']['status']=='running'
    finally:rt.shutdown()
