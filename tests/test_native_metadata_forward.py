"""Only exact executed metadata can pass a forward proposal and application."""
import copy
import json
import os
from types import SimpleNamespace
import pytest
from agents.goal_contract import GoalContract
from agents.state_io import write_checkpoint,json_transaction
from agents.workflow_view import bind_result_handoff
from agents.workflow_metadata_alignment import prove_forward_transition
from tests.test_active_resource_carry import native_case
from tests.test_parallel_workflow import factory,node


@pytest.fixture
def forward_case(native_case,factory,monkeypatch):
 from agents.defns import ORCHESTRATOR
 from agents.watch_context import set_context,clear_context
 state,aligned,root,goal,jobs=native_case
 rt=factory([node('placeholder')],cid='metadata-host',username='owner')
 rt.root=root;rt.conv_id='conv';rt.workflow_id=rt.store.identity('owner','conv')
 rt.main._current_line_id='conv';rt.main._evidence_root=root/'evidence'
 state['user_paused']=False;state['nodes']['n2']['jobs_confirmed_terminal']=True
 state['goal_contract']['approved_nodes']=[copy.deepcopy(state['nodes']['n2']['contract'])]
 wrong=copy.deepcopy(aligned);wrong['arguments']['resource_review_id']='new-unexecuted-review'
 goal['approved_nodes']=[wrong];rt.main.goal_contract=GoalContract.from_dict(goal)
 rt.main.current_agent=ORCHESTRATOR
 rt.main._control_source={'origin':'user','turn_id':'native-user-control','owner':{'username':'owner','conv_id':'conv'},'goal_version':goal['version']}
 rt.main._pending_jobs_for_conv=lambda:list(jobs.values())
 rt.job_watch=SimpleNamespace(get=lambda jid:copy.deepcopy(jobs.get(str(jid),{})),list=lambda:list(jobs.values()))
 token=state['nodes']['n2']['token'];lease={token:{'workflow_id':rt.workflow_id,'step_id':'n2','owner_pid':os.getpid(),'resources':copy.deepcopy(state['nodes']['n2']['contract']['resources'])}}
 write_checkpoint(rt.store.path,{'workflows':{rt.workflow_id:state},'leases':lease})
 line=rt.task_lines.begin_line('conv',username='owner',conv_id='conv');rt.main._current_line_id=line['line_id']
 old=state['nodes']['n2']['contract']
 rt.task_lines.apply_workflow_patch(line['line_id'],[{'operation':'upsert','step_id':'n2','node':old}],0,1,'original scientific task')
 rt.task_lines.apply_workflow_patch(line['line_id'],[{'operation':'upsert','step_id':'n2','node':wrong}],1,2,'already approved wrong receipt metadata')
 monkeypatch.setattr('agents.task_line.get_store',lambda:rt.task_lines)
 reviews=[]
 def forbidden_review(event):reviews.append(event);pytest.fail('terminal executed metadata must not allocate/review another budget')
 rt.main._on_resource_review=forbidden_review
 bind_result_handoff(rt.main,lambda:rt)
 set_context('owner','conv','lead-orchestrator',line['line_id'])
 try:yield rt,aligned,jobs,reviews
 finally:clear_context()


def propose(rt,aligned):
 return rt.main._propose_workflow_patch(2,'Align approved receipt metadata with the original verified executed allocation.',
  [{'operation':'upsert','step_id':'n2','node':copy.deepcopy(aligned)}])


def test_real_session_proposal_apply_and_execute_carries_original_terminal_attempt(forward_case):
 rt,aligned,jobs,reviews=forward_case
 before=rt.snapshot()['nodes']['n2'];ledger=(rt.root/'recovery_state.json').read_bytes()
 propose(rt,aligned);patch=rt.main._pending_workflow_patch
 assert patch and patch['base_version']==2 and patch['new_version']==3
 assert patch['proposed_nodes'][0]['arguments']==aligned['arguments']
 assert rt.main.goal_contract.approved_plan_version==2 and rt.snapshot()['plan_version']==1
 assert reviews==[] and (rt.root/'recovery_state.json').read_bytes()==ledger
 rt.main._apply_workflow_patch(3)
 assert rt.main.goal_contract.approved_plan_version==3 and rt.task_lines.get_line(rt.main._current_line_id)['plan_version']==3
 rt.start(3)
 after=rt.snapshot()['nodes']['n2']
 assert after['status']=='prefinish' and rt.snapshot()['plan_version']==3
 assert after['approved_resource_carry']['no_dispatch'] is True
 for key in ['token','attempt_id','job_ids','result_ref','result','effective_arguments','resource_allocation']:assert after[key]==before[key]
 assert reviews==[] and (rt.root/'recovery_state.json').read_bytes()==ledger


@pytest.mark.parametrize('change',['running','uncertain','unconfirmed','paused','memory','cpu','science','output','input','wrong_receipt','job_owner','nonterminal','producer','checkpoint','ledger'])
def test_forward_alignment_never_reinterprets_science_or_unproved_allocation(forward_case,change):
 rt,aligned,jobs,reviews=forward_case;candidate=copy.deepcopy(aligned);n=rt.snapshot()['nodes']['n2']
 if change in ['running','uncertain','unconfirmed']:
  rt.store.update(rt.workflow_id,'n2',n['token'],{'status':change} if change!='unconfirmed' else {'jobs_confirmed_terminal':False})
 elif change=='paused':rt.store.user_pause(rt.workflow_id,True)
 elif change in ['memory','cpu','science','input','wrong_receipt']:
  key,value={'memory':('memory_mb',50000),'cpu':('num_processes',48),'science':('gas','CO2'),'input':('input_dir',str(rt.root/'new-input')),'wrong_receipt':('resource_review_id','another-review')}[change];candidate['arguments'][key]=value
 elif change=='output':candidate['arguments']['output']=str(rt.root/'output/other.csv')
 elif change=='job_owner':jobs['77']['username']='foreign'
 elif change=='nonterminal':jobs['77'].update(terminal=False,state='RUNNING')
 elif change in ['producer','checkpoint']:
  from pathlib import Path
  field='evidence_path' if change=='producer' else 'checkpoint_path';Path(n['result_ref'][field]).write_text('{}')
 else:
  path=rt.root/'recovery_state.json';value=json.loads(path.read_text());value['recovery']['params']['memory_mb']+=1;path.write_text(json.dumps(value))
 before=rt.store.path.read_bytes();line=rt.task_lines.get_line(rt.main._current_line_id);ledger=(rt.root/'recovery_state.json').read_bytes()
 with pytest.raises((ValueError,PermissionError)):propose(rt,candidate)
 assert rt.main._pending_workflow_patch is None
 assert rt.store.path.read_bytes()==before and rt.task_lines.get_line(rt.main._current_line_id)==line
 assert (rt.root/'recovery_state.json').read_bytes()==ledger and reviews==[]


@pytest.mark.parametrize('change',['pause','job_owner','producer','checkpoint','ledger'])
def test_apply_reproves_metadata_after_proposal_instead_of_trusting_cached_permission(forward_case,change):
 rt,aligned,jobs,reviews=forward_case;propose(rt,aligned);n=rt.snapshot()['nodes']['n2']
 if change=='pause':rt.store.user_pause(rt.workflow_id,True)
 elif change=='job_owner':jobs['77']['conv_id']='foreign'
 elif change in ['producer','checkpoint']:
  from pathlib import Path
  Path(n['result_ref']['evidence_path' if change=='producer' else 'checkpoint_path']).write_text('{}')
 else:
  path=rt.root/'recovery_state.json';value=json.loads(path.read_text());value['recovery']['params']['memory_mb']+=1;path.write_text(json.dumps(value))
 before=rt.task_lines.get_line(rt.main._current_line_id)
 with pytest.raises((ValueError,PermissionError)):rt.main._apply_workflow_patch(3)
 assert rt.main.goal_contract.approved_plan_version==2 and rt.task_lines.get_line(rt.main._current_line_id)==before
 assert reviews==[]
