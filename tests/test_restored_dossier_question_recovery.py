"""A verified current recovery dossier releases an obsolete technical question."""
import copy
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
import pytest
from agents.lifecycle import obsolete_internal_question_for_recovery


def restored_case(tmp_path):
    root=tmp_path/'runs/owner/conv';proof='a'*64;call='restored'+'b'*64;token='c'*32
    contract={'step_id':'verify','agent':'analyst','tool':'read_file','arguments':{'path':str(root/'input.txt')},'expected_outputs':[]}
    goal={'version':5,'approved_plan_version':7,'execution_authorized':True,'execution_mode':'workflow','approved_nodes':[contract]}
    wid=hashlib.sha256(json.dumps(['owner','conv']).encode()).hexdigest()[:24]
    native={'kind':'current_returned_evidence','proof_sha256':proof,'no_resubmission':True,'no_output_write':True}
    dossier={'workflow_id':wid,'plan_version':7,'step_id':'verify','attempt_id':token,'job_ids':[],
        'contract':contract,'evidence_call_id':call,'evidence_path':str(root/'evidence'/f'{call}.json'),
        'verdict':'pass','requires_agent_review':True,'method_facts':{'native_result_restoration':native}}
    node={'status':'prefinish','phase':'prefinish','token':token,'origin_goal_version':5,'execution_plan_version':7,
        'contract':contract,'job_ids':[],'restoration':copy.deepcopy(native),'validation_dossier':copy.deepcopy(dossier)}
    runtime={'username':'owner','conv_id':'conv','plan_version':7,'goal_contract':goal,'status':'needs_user','user_paused':False,'nodes':{'verify':node}}
    checkpoints=[]
    session=SimpleNamespace(_evidence_root=root/'evidence',goal_contract=SimpleNamespace(version=5,approved_plan_version=7,to_dict=lambda:copy.deepcopy(goal)),
        _pending_user_interaction={'tool':'user_decision','params':{'related_nodes':['verify','dependent'],'decision_id':'oldtechnical'}},
        _waiting_for_user_input=True,_awaiting_plan_approval=False,_pending_workflow_patch=None,context={},
        _checkpoint=lambda reason:checkpoints.append(reason),_runtime_snapshot=lambda:copy.deepcopy(runtime))
    event={'kind':'worker_validation_dossier_ready','event_id':f'runtime:{wid}:{token}:validation-dossier:{call}','time':time.time(),
        'main_chat':'waiting_user','supervisor':'delivered','supervisor_receipt':{'reason':'native dossier'},
        'payload':{'step_id':'verify','runtime_id':wid,'origin_goal_version':5,'plan_version':7,'correlation_token':token,
            'validation_dossier':copy.deepcopy(dossier)}}
    return session,event,runtime,checkpoints


def test_current_restored_dossier_obsoletes_only_internal_question(tmp_path):
    session,event,runtime,checkpoints=restored_case(tmp_path)
    before=copy.deepcopy(runtime)
    assert obsolete_internal_question_for_recovery(session,event,runtime)
    assert session._pending_user_interaction is None and not session._waiting_for_user_input
    assert not session._awaiting_plan_approval
    assert runtime==before
    assert checkpoints==['obsolete_internal_question_for_recovery']


@pytest.mark.parametrize('change',['owner','goal','plan','token','call','proof','pause','approval','requires_user','unrelated','ordinary_dossier','current_changed'])
def test_stale_foreign_unverified_or_user_controlled_dossier_does_not_release_question(tmp_path,change):
    s,e,r,checkpoints=restored_case(tmp_path);p=e['payload'];node=r['nodes']['verify']
    if change=='owner':r['username']='other'
    elif change=='goal':r['goal_contract']={**r['goal_contract'],'version':6}
    elif change=='plan':r['plan_version']=8
    elif change=='token':p['correlation_token']='d'*32
    elif change=='call':p['validation_dossier']['evidence_call_id']='different'
    elif change=='proof':p['validation_dossier']['method_facts']['native_result_restoration']['proof_sha256']='e'*64
    elif change=='pause':r['user_paused']=True
    elif change=='approval':s._awaiting_plan_approval=True
    elif change=='requires_user':p['requires_user']=True
    elif change=='unrelated':s._pending_user_interaction['params']['related_nodes']=['other']
    elif change=='ordinary_dossier':node.pop('restoration')
    elif change=='current_changed':node['status']='succeeded'
    pending=copy.deepcopy(s._pending_user_interaction)
    assert not obsolete_internal_question_for_recovery(s,e,r)
    assert s._pending_user_interaction==pending and s._waiting_for_user_input and not checkpoints


def test_natural_dispatch_releases_question_and_reserves_same_scope_main(tmp_path,monkeypatch):
    import api
    s,e,r,checkpoints=restored_case(tmp_path)
    conv=SimpleNamespace(session=s,username='owner',conv_id='conv',interrupt_requested=False,interrupt_message='',
        is_processing=False,last_progress_at=0,current_uid='old-main',current_status='waiting')
    receipts=[];scheduled=[]
    class Store:
        def recover_claims(self):pass
        def heartbeat(self):pass
        def pending(self):return [copy.deepcopy(e)]
        def receipt(self,*args):receipts.append(args)
        def claim(self,*args):return True
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda conv:SimpleNamespace(tick=lambda *args:None))
    monkeypatch.setattr(api._watch,'list',lambda:[])
    monkeypatch.setattr(api,'_schedule_lifecycle',lambda coroutine:(scheduled.append(coroutine),coroutine.close()))
    import asyncio
    asyncio.run(api._dispatch_conversation_lifecycle('owner',conv,Store()))
    assert s._pending_user_interaction is None
    assert conv.is_processing and conv.current_uid.startswith('lifecycle:')
    assert len(scheduled)==1
    assert any(row[1:3]==('main_chat','retry') for row in receipts)
