import copy
import pytest
from tests.test_repair_event_isolation import setup
from agents.state_io import json_transaction


def install(machine, store):
    feedback={'contract_code':'possible_report_constraint_conflict','missing_table_ids':['T2'],
        'source_tables':[{'id':'T2','available_columns':['unit','quality']}],
        'independent_review_issues':['competing instructions'],
        'resolution':'Preserve the evidence; seek a plan revision or user decision.'}
    proof={'call_id':'call-one','call_sha256':'a'*64,'checkpoint_sha256':'b'*64}
    with json_transaction(store.path) as data:
        state=data['workflows']['workflow'];node=state['nodes']['one']
        node['result']={'error_kind':'report_authoring_failed','executed':False,'report_published':False,'contract_feedback':feedback}
        state['failure_recovery']['incidents']['incident-one']['evidence']=proof
    return feedback,proof


def test_scoped_verified_feedback_reaches_repair_planner(tmp_path,monkeypatch):
    machine,store,actor,root=setup(tmp_path);feedback,proof=install(machine,store)
    monkeypatch.setattr(machine,'_source',lambda state,step: proof if step=='one' else pytest.fail('cross incident source'))
    seen=[]
    result=machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:(seen.append(e) or {'status':'blocked'}))
    assert seen[0]['payload']['report_contract_feedback']==feedback
    assert 'not user instruction' in seen[0]['payload']['report_contract_feedback_trust']
    assert seen[0]['payload']['report_contract_feedback_source']==proof
    assert result['retry_granted'] is False
    assert store.snapshot('workflow')['nodes']['one']['status']=='failed'


def test_changed_call_or_checkpoint_rejected_before_delegation(tmp_path,monkeypatch):
    machine,store,actor,root=setup(tmp_path);_,proof=install(machine,store)
    changed=copy.deepcopy(proof);changed['checkpoint_sha256']='c'*64
    monkeypatch.setattr(machine,'_source',lambda state,step:changed)
    with pytest.raises(ValueError,match='source changed'):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:pytest.fail('changed source delegated'))
    assert store.snapshot('workflow')['failure_recovery']['incidents']['incident-one']['phase']=='repair_required'


def test_untrusted_feedback_never_reaches_planner(tmp_path,monkeypatch):
    machine,store,actor,root=setup(tmp_path);_,proof=install(machine,store)
    with json_transaction(store.path) as data:
        data['workflows']['workflow']['nodes']['one']['result']['contract_feedback']['contract_code']='forged'
    monkeypatch.setattr(machine,'_source',lambda state,step:proof)
    with pytest.raises(ValueError,match='untrusted'):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:pytest.fail('forged source delegated'))


def test_real_call_payload_tampering_rejected_before_planner(tmp_path,monkeypatch):
    import json,time
    from agents import local_preparation_recovery
    machine,store,actor,root=setup(tmp_path);feedback,_=install(machine,store)
    checkpoint=str(root/'checkpoint.json'); witness={'path':checkpoint,'sha256':'b'*64}
    monkeypatch.setattr(local_preparation_recovery,'checkpoint_call_witness',lambda *a,**k:witness)
    cid='call-one';callpath=root/'evidence'/f'{cid}.json'
    with json_transaction(store.path) as data:
        state=data['workflows']['workflow'];node=state['nodes']['one'];node['started_at']=time.time()-1
        node['result_ref']={'call_id':cid,'evidence_path':str(callpath),'checkpoint_path':checkpoint}
        call={'call_id':cid,'time':time.time(),'tool':'generate_scientific_report',
              'params':node['contract']['arguments'],'result':copy.deepcopy(node['result'])}
    callpath.write_text(json.dumps(call))
    proof=machine._source(store.snapshot('workflow'),'one')
    with json_transaction(store.path) as data:
        data['workflows']['workflow']['failure_recovery']['incidents']['incident-one']['evidence']=proof
    call['result']['contract_feedback']['missing_table_ids']=['T9'];callpath.write_text(json.dumps(call))
    with pytest.raises(ValueError):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:pytest.fail('tampered evidence delegated'))
    assert store.snapshot('workflow')['failure_recovery']['incidents']['incident-one']['phase']=='repair_required'


def verified_call(machine,store,root):
    import json,time,hashlib
    install(machine,store); cid='call-one'; path=root/'evidence'/f'{cid}.json'; cp=root/'checkpoint.json'
    with json_transaction(store.path) as data:
        node=data['workflows']['workflow']['nodes']['one'];node['started_at']=time.time()-1
        node['result_ref']={'call_id':cid,'evidence_path':str(path),'checkpoint_path':str(cp)}
        call={'call_id':cid,'time':time.time(),'agent':'communicator','tool':'generate_scientific_report',
              'params':node['contract']['arguments'],'result':copy.deepcopy(node['result']),'failed':True}
    path.write_text(json.dumps(call));record={k:call.get(k) for k in ('call_id','time','agent','tool','params','failed')}
    record.update(evidence_path=str(path),result_preview=str(call['result'])[:500],result_sha256=hashlib.sha256(str(call['result']).encode()).hexdigest())
    checkpoint={'owner':{'username':'owner','conv_id':'case','step_id':'one'},'checkpoint_at':time.time()+1,'memory':{'tool_call_log':[record]}}
    cp.write_text(json.dumps(checkpoint));proof=machine._source(store.snapshot('workflow'),'one')
    with json_transaction(store.path) as data:data['workflows']['workflow']['failure_recovery']['incidents']['incident-one']['evidence']=proof
    return path,cp,call,checkpoint


def test_actual_native_receipt_and_checkpoint_reach_planner(tmp_path):
    machine,store,actor,root=setup(tmp_path);verified_call(machine,store,root);seen=[]
    machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=lambda e:(seen.append(e) or {'status':'blocked'}))
    assert seen[0]['payload']['report_contract_feedback']['missing_table_ids']==['T2']
    assert seen[0]['payload']['report_contract_feedback_source']['call_id']=='call-one'


@pytest.mark.parametrize('tamper',['call','checkpoint','control'])
def test_native_evidence_or_control_change_blocks_dispatch(tmp_path,tamper):
    import json
    machine,store,actor,root=setup(tmp_path);path,cp,call,checkpoint=verified_call(machine,store,root)
    if tamper=='call':
        call['result']['contract_feedback']['missing_table_ids']=['T9'];path.write_text(json.dumps(call))
    if tamper=='checkpoint':
        checkpoint['memory']['tool_call_log'][0]['result_preview']='tampered';cp.write_text(json.dumps(checkpoint))
    checks=[]
    def control():
        checks.append(True);return tamper!='control' or len(checks)<2
    with pytest.raises(ValueError):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',control_valid=control,stage_runner=lambda e:pytest.fail('unsafe dispatch'))
    assert store.snapshot('workflow')['failure_recovery']['incidents']['incident-one']['phase']=='repair_required'
