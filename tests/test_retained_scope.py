import copy, hashlib, importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace
import pytest

from agents import retained_prerequisite_scope as scope
from agents.goal_contract import GoalContract
from agents.parallel_workflow import WorkflowStore,_approval_digest
from agents.recovery import RecoveryGate,input_fingerprint
from agents import task_line

@pytest.fixture
def proofcase(tmp_path,monkeypatch):
 root=tmp_path/'u'/'c'; (root/'evidence').mkdir(parents=True)
 cif=root/'input.cif';cif.write_text('data_test\n')
 args={'cif_path':str(cif),'method':'pacmof'}
 contract={'step_id':'charge','agent':'analyst','tool':'run_pacman_charge','arguments':args}
 old=GoalContract.from_user_message('用 GCMC 计算 CO2 298 K'); old.approved_nodes=[contract];old.approved_plan_version=1
 new=copy.deepcopy(old);new.apply_user_message('温度改为308 K');new.approved_plan_version=2
 oldd=old.to_dict();newd=new.to_dict()
 line={'username':'u','conv_id':'c','plan_version':2,'plan_history':[{'version':1,'steps':[contract]}]}
 approval={'schema_version':1,'kind':'main_model_applied_plan','owner':{'username':'u','conv_id':'c'},'goal_version':new.version,'plan_version':2,'base_plan_version':1,'control_source':{'origin':'user','owner':{'username':'u','conv_id':'c'},'turn_id':'turn'},'approval_sha256':_approval_digest(newd)}
 monkeypatch.setattr(task_line,'get_store',lambda:SimpleNamespace(get_line=lambda *a,**kw:line))
 result={'submitted':True,'job_id':'123'};call={'call_id':'abc','tool':'run_pacman_charge','params':args,'failed':False,'agent':'analyst','time':1.0,'result':result}
 evidence=root/'evidence/abc.json';evidence.write_text(json.dumps(call))
 record={**call,'evidence_path':str(evidence),'result_preview':str(result)[:500],'result_sha256':hashlib.sha256(str(result).encode()).hexdigest()}
 checkpoint=root/'evidence/cp.json';checkpoint.write_text(json.dumps({'owner':{'username':'u','conv_id':'c','step_id':'charge'},'checkpoint_at':2,'memory':{'tool_call_log':[record]},'context':{'assigned_node':contract}}))
 key=RecoveryGate().key('run_pacman_charge',args,1,'charge',tmp_path)
 entry={'key':key,'goal_version':1,'attempt_id':'attempt','tool':'run_pacman_charge','params':args,'fingerprint':input_fingerprint(args,tmp_path),'job_ids':['123'],'status':'submitted'}
 (root/'recovery_state.json').write_text(json.dumps({key:entry}))
 node={'contract':contract,'status':'waiting_jobs','token':'token','attempt_id':'attempt','job_ids':['123'],'recovery_key':key,'result_ref':{'call_id':'abc','evidence_path':str(evidence),'checkpoint_path':str(checkpoint)},'result':result,'effective_arguments':args,'tool_returned':True,'origin_goal_version':1,'execution_plan_version':1}
 runtime={'username':'u','conv_id':'c','status':'active','plan_version':2,'goal_contract':newd,'nodes':{'charge':copy.deepcopy(node)},'plan_history':[{'plan_version':1,'goal_contract':oldd,'nodes':{'charge':copy.deepcopy(node)}}]}
 event={'kind':'worker_completed','event_id':'runtime:'+WorkflowStore.identity('u','c')+':token:succeeded','payload':{'step_id':'charge','runtime_id':WorkflowStore.identity('u','c'),'workflow_id':'c','correlation_token':'token','attempt_id':'attempt','origin_goal_version':1,'plan_version':2}}
 job={'job_id':'123','username':'u','conv_id':'c','tool':'run_pacman_charge','attempt_id':'attempt','recovery_key':key,'state':'RUNNING','terminal':False,'failed':False}
 session=SimpleNamespace(_evidence_root=root/'evidence',_current_line_id='c',goal_contract=new,_pending_workflow_patch=None,_awaiting_plan_approval=False,_interrupt_requested=lambda:False,context={'approved_execution_proof':approval},config=SimpleNamespace(project_root=tmp_path))
 return session,event,runtime,[job],cif

def test_exact_retained_charge_event(proofcase):
 session,event,runtime,jobs,cif=proofcase
 proof=scope.retained_node_event_proof(session,event,runtime,jobs=jobs)
 assert proof and proof['independent_changed_goal_parameters']==['temperature_K']
 assert proof['no_origin_or_ledger_rewrite'] and proof['no_acceptance_or_retry_authority']
 assert scope.event_goal_is_current(session,event,runtime,jobs=jobs)

@pytest.mark.parametrize('mutation',['token','owner','tool','attempt','cif','other_parameter','authorization','old_contract','job','plan'])
def test_reject_stale_or_unproved_event(proofcase,mutation):
 s,e,r,j,c=proofcase
 if mutation=='token':e['payload']['correlation_token']='different'
 if mutation=='owner':r['username']='other'
 if mutation=='tool':r['nodes']['charge']['contract']['tool']='run_gcmc_batch'
 if mutation=='attempt':r['nodes']['charge']['attempt_id']='different'
 if mutation=='cif':c.write_text('changed input')
 if mutation=='other_parameter':
  s.goal_contract.parameters['unknown_scientific_parameter']=7;r['goal_contract']=s.goal_contract.to_dict()
 if mutation=='authorization':s.goal_contract.execution_authorized=False;r['goal_contract']=s.goal_contract.to_dict()
 if mutation=='old_contract':r['plan_history'][0]['nodes']['charge']['contract']['arguments']['method']='other'
 if mutation=='job':j[0]['username']='other'
 if mutation=='plan':r['plan_version']=3
 assert scope.retained_node_event_proof(s,e,r,jobs=j) is None


def pending_edit_case(proofcase):
 s,e,r,j,c=proofcase
 r['plan_version']=1;r['goal_contract']=copy.deepcopy(r['plan_history'][0]['goal_contract'])
 e['payload']['plan_version']=1
 return s,e,r,j,c


def test_old_prerequisite_notice_waits_during_parameter_plan_negotiation(proofcase):
 s,e,r,j,c=pending_edit_case(proofcase);before=copy.deepcopy((e,r))
 assert scope.defer_pending_prerequisite_event(s,e,r)
 assert (e,r)==before
 # Once the new graph is active the complete proof, not the deferral,
 # determines whether the original notice can be delivered.
 r['plan_version']=2;r['goal_contract']=s.goal_contract.to_dict();e['payload']['plan_version']=2
 assert not scope.defer_pending_prerequisite_event(s,e,r)
 assert scope.retained_node_event_proof(s,e,r,jobs=j)

@pytest.mark.parametrize('change',['foreign','token','attempt','unknown_tool','changed_gas','changed_source','unchanged_goal','pause','bad_event','not_old_plan'])
def test_pending_notice_deferral_never_promotes_other_scope(proofcase,change):
 s,e,r,j,c=pending_edit_case(proofcase)
 if change=='foreign':r['username']='other'
 if change=='token':e['payload']['correlation_token']='other'
 if change=='attempt':e['payload']['attempt_id']='other'
 if change=='unknown_tool':r['nodes']['charge']['contract']['tool']='run_gcmc_batch'
 if change=='changed_gas':s.goal_contract.gases=['N2']
 if change=='changed_source':s.goal_contract.parameters['new_input']='different'
 if change=='unchanged_goal':s.goal_contract.version=1
 if change=='pause':r['user_paused']=True
 if change=='bad_event':e['kind']='manual_request'
 if change=='not_old_plan':e['payload']['plan_version']=2
 assert not scope.defer_pending_prerequisite_event(s,e,r)


def test_dispatcher_keeps_completed_supervisor_notice_during_pending_edit(proofcase,tmp_path,monkeypatch):
 import asyncio
 import api
 from agents.lifecycle import LifecycleStore
 s,e,r,j,c=pending_edit_case(proofcase)
 store=LifecycleStore(tmp_path/'lifecycle.json');store.emit(e['kind'],e['payload'],event_id=e['event_id'])
 store.receipt(e['event_id'],'supervisor','delivered',{'next_action':'diagnose_and_fix'})
 s._runtime_snapshot=lambda:copy.deepcopy(r)
 conv=SimpleNamespace(session=s,username='u',conv_id='c',interrupt_requested=False,interrupt_message='',
     is_processing=True,last_progress_at=0,current_agent='lead',current_uid='turn')
 monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda c:SimpleNamespace(tick=lambda *args:None))
 monkeypatch.setattr(api,'_watch',SimpleNamespace(list=lambda:j))
 asyncio.run(api._dispatch_conversation_lifecycle('u',conv,store))
 recorded=store.snapshot()['events'][e['event_id']]
 assert recorded['supervisor']=='delivered' and recorded['main_chat']=='pending'


def test_dispatcher_resumes_same_prerequisite_after_approved_graph(proofcase,tmp_path,monkeypatch):
 import asyncio
 import api
 from agents.lifecycle import LifecycleStore
 s,e,r,j,c=proofcase
 r['nodes']['charge']['status']='succeeded'
 j[0].update(state='COMPLETED',terminal=True,failed=False)
 store=LifecycleStore(tmp_path/'lifecycle.json');store.emit(e['kind'],e['payload'],event_id=e['event_id'])
 store.receipt(e['event_id'],'supervisor','delivered',{'next_action':'advance_dependencies'})
 s._runtime_snapshot=lambda:copy.deepcopy(r)
 s._pending_user_interaction=None
 conv=SimpleNamespace(session=s,username='u',conv_id='c',interrupt_requested=False,interrupt_message='',
     is_processing=False,last_progress_at=0,current_agent='lead',current_uid='turn')
 monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda c:SimpleNamespace(tick=lambda *args:None))
 monkeypatch.setattr(api,'_watch',SimpleNamespace(list=lambda:j))
 delivered=[]
 async def schedule(conv,event,store):delivered.append(event['event_id'])
 monkeypatch.setattr(api,'_reserve_and_schedule_lifecycle_main',schedule)
 asyncio.run(api._dispatch_conversation_lifecycle('u',conv,store))
 assert delivered==[e['event_id']]
 assert store.snapshot()['events'][e['event_id']]['main_chat']=='pending'
