"""Completed native evidence survives a pure actor assignment change."""
import copy
import json
from types import SimpleNamespace
import pytest
from agents.parallel_workflow import execution_fingerprint
from agents.state_io import json_transaction,write_checkpoint
from tests.test_parallel_workflow import factory,node,eventually


def completed_local(factory,tmp_path):
    source=tmp_path/'source.txt';source.write_text('stable source')
    rt=factory([node('read',path=str(source))])
    seen=[]
    def read(arguments):seen.append(arguments);return {'content':'stable source'}
    rt.main.registry.get('read_file').execute=read
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['read']['status']=='succeeded')
    return rt,source,seen


def reassign(rt,step_id,agent):
    old=copy.deepcopy(rt.snapshot()['nodes'][step_id]['contract']);old.pop('resources',None);old['agent']=agent
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,[{'operation':'upsert','step_id':step_id,'node':old}],1,2,'approved role assignment')
    rt.main.goal_contract.approved_plan_version=2
    rt.main.goal_contract.approved_nodes=[old]
    return old


def test_completed_readonly_role_change_keeps_original_result_and_producer(factory,tmp_path):
    rt,source,seen=completed_local(factory,tmp_path)
    before=copy.deepcopy(rt.snapshot()['nodes']['read'])
    reassign(rt,'read','lead-orchestrator')
    rt.start(2);rt.tick()
    current=rt.snapshot()['nodes']['read']
    assert current['status']=='succeeded' and len(seen)==1
    for key in ('result','result_ref','artifacts','token','execution_fingerprint'):
        assert current[key]==before[key]
    assert current['contract']['agent']=='lead-orchestrator'
    assert current['execution_producer']['agent']=='analyst'


@pytest.mark.parametrize('change',['input','tool_source','artifact','live_lease','native_evidence','native_evidence_result'])
def test_role_change_cannot_reuse_changed_completed_proof(factory,tmp_path,change):
    rt,source,seen=completed_local(factory,tmp_path)
    reassign(rt,'read','lead-orchestrator')
    if change=='input':source.write_text('changed source')
    elif change=='tool_source':
        def different(arguments):return {'content':'different adapter'}
        rt.main.registry.get('read_file').execute=different
    elif change=='artifact':
        with json_transaction(rt.store.path) as data:data['workflows'][rt.workflow_id]['nodes']['read']['artifacts']={'missing':{'size':1}}
    elif change=='live_lease':
        with json_transaction(rt.store.path) as data:data['leases']['foreign']={'workflow_id':rt.workflow_id,'step_id':'read','owner_pid':123,'resources':[]}
    elif change in ('native_evidence','native_evidence_result'):
        p=rt.snapshot()['nodes']['read']['result_ref']['evidence_path'];record=json.loads(open(p).read())
        if change=='native_evidence':record['failed']=True
        else:record['result']=json.dumps({'content':'tampered successful native result'})
        open(p,'w').write(json.dumps(record))
    with pytest.raises(ValueError):rt.start(2)
    assert len(seen)==1


def test_verified_native_job_role_change_preserves_scheduler_and_inspection_receipts(factory,tmp_path):
    output=tmp_path/'generated';output.mkdir();(output/'a.cif').write_text('native stable artifact')
    spec=node('generate',tool='generate_structure',args={'material_type':'MOF','n_structures':1,'output_dir':str(output)},outputs=[{'kind':'directory','path':str(output),'pattern':'*.cif','min_count':1}]);spec['agent']='harness'
    rt=factory([spec]);rt.start(1)
    contract=rt.snapshot()['nodes']['generate']['contract'];call_id='nativeownedexecution'
    receipt={'submitted':True,'job_id':'42','output_dir':str(output)}
    path=rt.root/'evidence'/f'{call_id}.json'
    write_checkpoint(path,{'call_id':call_id,'time':1,'tool':'generate_structure','params':contract['arguments'],'result':json.dumps(receipt),'failed':False})
    with json_transaction(rt.store.path) as data:
        n=data['workflows'][rt.workflow_id]['nodes']['generate']
        n.update(status='succeeded',phase='finish',token='native-token',attempt_id='native-attempt',job_ids=['42'],
            node_verification={'owner':'harness','evidence_call_ids':['native-inspection']},result=receipt,
            result_ref={'call_id':call_id,'evidence_path':str(path)},execution_fingerprint=execution_fingerprint(contract,rt.main))
        n['artifacts']=rt._artifacts(contract,receipt)
    rt.job_watch=SimpleNamespace(list=lambda:[],get=lambda job:{'job_id':job,'username':rt.username,'conv_id':rt.conv_id,'terminal':True,'failed':False,'state':'COMPLETED'})
    before=copy.deepcopy(rt.snapshot()['nodes']['generate'])
    reassign(rt,'generate','harness-maintainer')
    rt.start(2)
    after=rt.snapshot()['nodes']['generate']
    assert after['status']=='succeeded'
    for key in ('job_ids','attempt_id','token','result_ref','node_verification','execution_fingerprint'):
        assert after[key]==before[key]
    assert after['execution_producer']['agent']=='harness'


def test_restored_finish_clears_only_its_resolved_branch_blocker(tmp_path,monkeypatch):
    from tests.test_native_local_result_reconciliation import proven
    # Use the real proof-protocol fixture setup without executing any writer.
    fixture=proven.__wrapped__(tmp_path,monkeypatch)
    assert fixture.commit()
    with json_transaction(fixture.store.path) as data:
        w=data['workflows']['w'];w['nodes']['fragment']['node_verification']={'evidence_call_ids':['fresh']}
        w['branch_blockers']={'fragment':{'kind':'reconciliation'},'unrelated':{'kind':'failure'}}
    fields={'artifacts':fixture.proof['artifacts'],'path_manifest':{},'execution_fingerprint':'owned-fp'}
    assert fixture.store.finish_proven_local_result('w','fragment','current',fixture.goal,'tester','c1',fixture.root,fields)
    assert fixture.store.snapshot('w')['branch_blockers']=={'unrelated':{'kind':'failure'}}
