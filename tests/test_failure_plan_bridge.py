import copy,pytest
from agents.parallel_workflow import _approval_digest
from agents.report_failure_plan_bridge import prove_failure_plan_bridge
from tests.test_report_noop_rebase import rebased

def fixture_bridge():
 old,goal,line,approval=rebased();old['pause_kind']='failure';old['pause_reason']='failed node requires diagnosis and plan review'
 node=old['nodes']['report'];node.update(token='a'*32,execution_plan_version=6,origin_goal_version=2,dispatch_phase='entered_tool',tool_returned=True,job_ids=[],attempt_id=None,result_ref={'call_id':'abc'},result={'error':'failure'})
 historical=next(n for n in line['plan_history'][1]['steps'] if n['step_id']=='report');historical['arguments']['instructions']='unentered edit';historical.update(status='pending',done=False,job_ids=[],output_files=[])
 state=copy.deepcopy(old);state['plan_version']=8;state['goal_contract']=copy.deepcopy(goal);state['pause_kind']=None;state['pause_reason']=None
 state['plan_history']=[{'plan_version':6,'goal_contract':copy.deepcopy(old['goal_contract']),'nodes':copy.deepcopy(old['nodes'])}]
 state['activation_recovery']={'owner':['owner','conv'],'formal_approval':copy.deepcopy(approval),'from_plan_version':6,'approved_plan_version':8,'approval_sha256':_approval_digest(goal),'from_approval_sha256':_approval_digest(old['goal_contract']),'status':'blocked','details':{'no_serial_fallback':True,'error':'Unchanged failed nodes remain failed. Restore verified results or provide a verified execution repair; identical upsert is not a retry.'}}
 state['outbox']={'event':{'kind':'worker_failed','payload':{'step_id':'report','plan_version':6,'correlation_token':'a'*32,'result_ref':{'call_id':'abc'}}}}
 return state,goal,line,approval,{'username':'owner','conv_id':'conv'}

def test_owned_archived_failure_is_proven_without_retry_authority():
 state,goal,line,approval,owner=fixture_bridge();before=copy.deepcopy((state,goal,line,approval,owner))
 proof=prove_failure_plan_bridge(state,'report',goal,line,approval,owner)
 assert proof['from_plan_version']==6 and proof['to_plan_version']==8 and proof['no_retry_authority']
 assert (state,goal,line,approval,owner)==before

@pytest.mark.parametrize('mutation',['foreign','token','event','missing_event','duplicate_event','archive','current_contract','history_contract','started_history','old_result','approval','new_goal','source_goal','pause','stop_directive','duplicate_node'])
def test_failure_bridge_refuses_identity_and_authority_changes(mutation):
 state,goal,line,approval,owner=fixture_bridge()
 if mutation=='foreign':owner['username']='other'
 elif mutation=='token':state['nodes']['report']['token']='b'*32
 elif mutation=='event':state['outbox']['event']['payload']['plan_version']=8
 elif mutation=='missing_event':state['outbox'].clear()
 elif mutation=='duplicate_event':state['outbox']['second']=copy.deepcopy(state['outbox']['event'])
 elif mutation=='archive':state['plan_history'][0]['nodes']['report']['result_ref']={'call_id':'other'}
 elif mutation=='current_contract':state['nodes']['report']['contract']['arguments']['instructions']='change'
 elif mutation=='history_contract':line['plan_history'][1]['steps'][0]['arguments']['gas']='CO2'
 elif mutation=='started_history':next(n for n in line['plan_history'][1]['steps'] if n['step_id']=='report')['status']='running'
 elif mutation=='old_result':state['nodes']['report']['result']={'error':'different'}
 elif mutation=='approval':approval['approval_sha256']='invalid'
 elif mutation=='new_goal':goal['parameters']['temperature']=400
 elif mutation=='source_goal':state['plan_history'][0]['goal_contract']['gases']=['CO2']
 elif mutation=='pause':state['user_paused']=True
 elif mutation=='stop_directive':goal['directives'][-1]='不要继续原任务。'
 elif mutation=='duplicate_node':goal['approved_nodes'].append(copy.deepcopy(goal['approved_nodes'][0]))
 with pytest.raises((ValueError,PermissionError,KeyError)):prove_failure_plan_bridge(state,'report',goal,line,approval,owner)
