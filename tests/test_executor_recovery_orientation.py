import copy
import json
from pathlib import Path

import pytest
from agents import executor_recovery
from agents.watch_context import set_context,clear_context
from tests.test_executor_repair_notification import failed_attempt, capabilities
from tests.test_parallel_workflow import factory


def native_executor(params):
    return {'not_dispatched':True}


@pytest.fixture
def current_failure(factory,tmp_path,monkeypatch):
    rt,gate,source=failed_attempt(factory,tmp_path)
    rt.main.registry.get('generate_structure').execute=native_executor
    source.write_text('executor_epoch = 2\n')
    rt.recover_notifications()
    event=capabilities(rt)[-1]
    rt.main.recovery_gate=gate
    calls=[];checkpoints=[]
    rt.main.registry.get('inspect_runtime').execute=lambda params:calls.append(copy.deepcopy(params))or{
        'ok':True,'read_only':True,'verified_tool':params['tool_name'],
        'checked_arguments':copy.deepcopy(params['arguments']),'runtime_fingerprint':event['payload']['recovery']['runtime_fingerprint']}
    rt.main._checkpoint=lambda reason:checkpoints.append(reason)
    rt.main._interrupt_requested=lambda:False
    set_context(rt.username,rt.conv_id,'lead-orchestrator',rt.conv_id)
    yield rt,gate,event,calls,checkpoints
    clear_context()


def test_new_process_epoch_replays_delivered_notification_once_without_retry(current_failure,monkeypatch):
    rt,gate,event,calls,_=current_failure
    for receiver in ['main_chat','supervisor']:
        rt.lifecycle.receipt(event['event_id'],receiver,'delivered',{'status':'handled'})
    before=gate.path.read_bytes();node=copy.deepcopy(rt.snapshot()['nodes']['generate'])
    rt.recover_notifications();assert len(capabilities(rt))==1
    monkeypatch.setattr(executor_recovery,'_PROCESS_STARTED_NS','new-import-epoch')
    rt.recover_notifications();assert len(capabilities(rt))==2
    rt.recover_notifications();assert len(capabilities(rt))==2
    new=capabilities(rt)[-1]
    assert new['event_id']!=event['event_id']
    assert new['payload']['recovery']['executor_epoch']['process_epoch']!=event['payload']['recovery']['executor_epoch']['process_epoch']
    assert gate.path.read_bytes()==before and rt.snapshot()['nodes']['generate']==node and not calls


def test_resume_performs_fresh_native_inspection_and_orients_main_without_authority_change(current_failure,monkeypatch):
    rt,gate,event,calls,checkpoints=current_failure
    actor=rt.main;actor._pending_user_interaction={'tool':'user_decision','params':{'decision_id':'genuine-pending'}}
    actor._waiting_for_user_input=True
    before=copy.deepcopy(actor.goal_contract.to_dict());ledger=gate.path.read_bytes()
    received=[]
    def loop(agent):
        received.append(json.loads(actor.messages[-1]['content'].split('\n',1)[1]));return 'observed'
    monkeypatch.setattr(actor,'_execute_loop',loop)
    assert actor.resume_lifecycle_event(event)=='observed'
    assert len(calls)==1 and checkpoints==['executor_recovery_native_inspection']
    orientation=received[0]['payload']['recovery']['current_executor_orientation']
    assert orientation['inspection_executed']and orientation['status']=='inspection_succeeded'
    assert orientation['loaded_executor']['matches'] is True
    proof=orientation['inspection'];assert proof['call_id']==actor.memory.tool_call_log[-1]['call_id']
    assert proof['result']['ok']is True and proof['time']>0
    assert orientation['retry_granted']is False and orientation['scientific_execution']is False
    assert actor._pending_user_interaction['params']['decision_id']=='genuine-pending'
    assert actor._waiting_for_user_input and actor.goal_contract.to_dict()==before and gate.path.read_bytes()==ledger


@pytest.mark.parametrize('change',['owner','goal','plan','contract','paused','interrupt','attempt','params','jobs','budget','fingerprint'])
def test_stale_or_foreign_event_cannot_trigger_inspection(current_failure,monkeypatch,change):
    rt,gate,event,calls,_=current_failure;actor=rt.main
    state=rt.snapshot();entry=gate.snapshot()
    if change=='owner':set_context('other',rt.conv_id,'lead-orchestrator',rt.conv_id)
    elif change=='goal':state['goal_contract']['version']+=1
    elif change=='plan':state['plan_version']+=1
    elif change=='contract':state['nodes']['generate']['contract']['arguments']['n_structures']=2
    elif change=='paused':state['user_paused']=True
    elif change=='interrupt':actor._interrupt_requested=lambda:True
    elif change=='attempt':event['payload']['recovery']['failed_attempt_id']='different'
    elif change=='params':event['payload']['recovery']['arguments']['arguments']['n_structures']=2
    elif change=='jobs':entry['failed-generation']['job_ids']=['42']
    elif change=='budget':entry['failed-generation']['attempts']=3
    else:event['payload']['recovery']['runtime_fingerprint']='stale'
    monkeypatch.setattr(actor,'_runtime_snapshot',lambda:state)
    monkeypatch.setattr(gate,'snapshot',lambda:copy.deepcopy(entry))
    before=copy.deepcopy(event)
    assert executor_recovery.refresh_executor_event(actor,event)==before
    assert calls==[]


def test_loaded_old_body_requires_reload_and_does_not_produce_syntax_green(current_failure,monkeypatch):
    rt,gate,event,calls,_=current_failure
    monkeypatch.setattr(executor_recovery,'executor_identity',lambda function:{'verified':True,'matches':False})
    result=executor_recovery.refresh_executor_event(rt.main,event)
    orientation=result['payload']['recovery']['current_executor_orientation']
    assert orientation['status']=='backend_reload_required'and not orientation['inspection_executed']
    assert not calls and not rt.main.memory.tool_call_log


def test_failed_fresh_inspection_remains_failed_evidence(current_failure):
    rt,gate,event,calls,_=current_failure
    rt.main.registry.get('inspect_runtime').execute=lambda params:{'ok':False,'error':'configured executable unavailable'}
    result=executor_recovery.refresh_executor_event(rt.main,event)
    orientation=result['payload']['recovery']['current_executor_orientation']
    assert orientation['inspection']['result']['ok']is False
    assert rt.main.memory.tool_call_log[-1]['failed']is True
    assert orientation['retry_granted']is False


def test_same_event_is_one_bounded_native_inspection(current_failure):
    rt,gate,event,calls,checkpoints=current_failure
    first=executor_recovery.refresh_executor_event(rt.main,event)
    second=executor_recovery.refresh_executor_event(rt.main,event)
    assert len(calls)==len(checkpoints)==len(rt.main.memory.tool_call_log)==1
    assert first==second


@pytest.mark.parametrize('race',['source','owner','goal','attempt','token','user_control','interrupt'])
def test_inspection_race_retains_actual_call_but_cannot_label_it_current(current_failure,monkeypatch,race):
    from agents.state_io import json_transaction
    rt,gate,event,calls,_=current_failure;actor=rt.main
    from types import SimpleNamespace
    from agents.workflow_user_control import bind_api_workflow_control, advance_user_control
    control_conv=SimpleNamespace(username=rt.username,conv_id=rt.conv_id,session=actor,
        interrupt_requested=False,interrupt_message='',_workflow_user_control_epoch=0)
    bind_api_workflow_control(control_conv,actor)
    original=actor.registry.get('inspect_runtime').execute
    def inspect(params):
        result=original(params)
        if race=='source':(Path(actor.config.project_root)/'agents/registry.py').write_text('epoch=3\n')
        elif race=='owner':set_context('other',rt.conv_id,'lead-orchestrator',rt.conv_id)
        elif race=='goal':actor.goal_contract.version+=1
        elif race=='attempt':
            with gate.transaction()as data:data['failed-generation']['attempt_id']='new-attempt'
        elif race=='token':
            with json_transaction(rt.store.path)as data:
                data['workflows'][rt.workflow_id]['nodes']['generate']['token']='new-token'
        elif race=='user_control':advance_user_control(control_conv)
        else:
            control_conv.interrupt_requested=True
            actor._interrupt_requested=lambda:True
        return result
    actor.registry.get('inspect_runtime').execute=inspect
    result=executor_recovery.refresh_executor_event(actor,event)
    orientation=result['payload']['recovery']['current_executor_orientation']
    assert orientation['status']=='superseded'and orientation['inspection_executed']
    assert orientation['inspection_current']is False and not orientation['retry_granted']
    assert len(calls)==len(actor.memory.tool_call_log)==1
    assert not actor.__dict__['_executor_recovery_observations']
