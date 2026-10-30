import copy,pytest
from tests.test_report_bounded_retry import candidate,native_call
from tests.test_user_report_retry import auth
from agents.report_bounded_retry import capture_report_retry
from agents.state_io import json_transaction

def legacy_fixture(candidate,monkeypatch):
 from agents import report_failure_plan_bridge
 runtime,goal,_,_=candidate;state=runtime.snapshot();node=state['nodes']['report'];args=node['contract']['arguments']
 failure={'error':'LLM API 连接失败（Connection error.）','traceback':'Traceback\nagents.session.LLMUnavailableError: LLM API 连接失败（Connection error.）\n'}
 call,ref=native_call(runtime.root,'failedcall','report','generate_scientific_report',args,failure,True)
 goal['approved_plan_version']=2;runtime.main.context={'approved_execution_proof':{'fixture':True}};runtime._line=lambda:{}
 with json_transaction(runtime.store.path)as data:
  state=data['workflows']['workflow'];state['plan_version']=2;state['goal_contract']=copy.deepcopy(goal)
  node=state['nodes']['report'];node.update(result=failure,result_ref=ref,started_at=call['time']-1,execution_plan_version=1)
  event=state['outbox']['failed-event']['payload'];event['result_ref']=copy.deepcopy(ref);event['plan_version']=1
  state['failure_recovery']['incidents']['incident']['plan_version']=2
 def bridge(state,step,goal,line,approval,owner):return {'from_plan_version':1,'to_plan_version':2,'no_retry_authority':True}
 monkeypatch.setattr(report_failure_plan_bridge,'prove_failure_plan_bridge',bridge)
 return runtime,goal,auth(runtime,goal)

def test_authorized_legacy_failure_uses_old_event_and_keeps_bridge(candidate,monkeypatch):
 runtime,goal,authorization=legacy_fixture(candidate,monkeypatch)
 proof,packet=capture_report_retry(runtime,runtime.snapshot(),'report',control_valid=lambda:True,return_packet=True,user_authorization=authorization)
 assert proof['legacy_connection_failure'] and proof['legacy_plan_bridge']['no_retry_authority']
 assert proof['legacy_plan_bridge']['from_plan_version']==1 and proof['plan_version']==2
 assert not (runtime.root/'report.md').exists() and runtime.snapshot()['nodes']['report']['status']=='failed'

@pytest.mark.parametrize('change',['no_authority','different_error','bad_event_plan','pause','pending','wrong_goal','target_exists','incident_not_blocked','bridge_denied'])
def test_legacy_path_does_not_bypass_user_or_native_guards(candidate,monkeypatch,change):
 from agents import report_failure_plan_bridge
 runtime,goal,authorization=legacy_fixture(candidate,monkeypatch)
 if change=='no_authority':authorization=None
 elif change=='pending':runtime.main._pending_user_interaction={'tool':'user_decision'}
 elif change=='target_exists':(runtime.root/'report.md').write_text('do not overwrite')
 elif change=='bridge_denied':monkeypatch.setattr(report_failure_plan_bridge,'prove_failure_plan_bridge',lambda *args:(_ for _ in ()).throw(ValueError('denied')))
 else:
  with json_transaction(runtime.store.path) as data:
   state=data['workflows']['workflow']
   if change=='different_error':state['nodes']['report']['result']['error']='unrelated'
   elif change=='bad_event_plan':state['outbox']['failed-event']['payload']['plan_version']=2
   elif change=='pause':state['user_paused']=True
   elif change=='wrong_goal':goal['parameters']['temperature']=500
   elif change=='incident_not_blocked':state['failure_recovery']['incidents']['incident']['phase']='repair_required'
 with pytest.raises((ValueError,PermissionError,KeyError)):
  capture_report_retry(runtime,runtime.snapshot(),'report',control_valid=lambda:True,user_authorization=authorization)
