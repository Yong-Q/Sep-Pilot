import copy,hashlib,json,os
import pytest
from agents.archived_download_recovery import prove_archived_download
from agents.parallel_workflow import WorkflowStore
from agents.state_io import json_transaction

def fixture(tmp_path):
 root=tmp_path/'runs/u/c';(root/'evidence').mkdir(parents=True);(root/'workers').mkdir();source=root/'doc.pdf';source.write_bytes(b'%PDF-1.4 original')
 args={'url':'https://example.org/doc.pdf','output_path':str(source)};contract={'step_id':'doc','agent':'analyst','tool':'download_scientific_file','arguments':args,'depends_on':[],'expected_outputs':[str(source)],'resources':[]}
 oldresult={'status':'success','output_path':str(source),'output_files':[str(source)],'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'bytes':source.stat().st_size,'existing_results_overwritten':False}
 failed={'error':'destination exists','error_kind':'scientific_download_failed','output_published':False,'existing_results_overwritten':False}
 def ref(cid,result,is_failed):
  call={'call_id':cid,'time':1,'agent':'analyst','tool':'download_scientific_file','params':args,'failed':is_failed,'result':json.dumps(result)};path=root/'evidence'/(cid+'.json');path.write_text(json.dumps(call));entry={k:call[k]for k in ('call_id','time','agent','tool','params','failed')};entry.update(evidence_path=str(path),result_preview=call['result'][:500],result_sha256=hashlib.sha256(call['result'].encode()).hexdigest());cp=root/'workers'/(cid+'.json');cp.write_text(json.dumps({'owner':{'username':'u','conv_id':'c','step_id':'doc'},'checkpoint_at':2,'memory':{'tool_call_log':[entry]}}));return {'call_id':cid,'evidence_path':str(path),'checkpoint_path':str(cp)}
 old={'contract':contract,'status':'succeeded','phase':'finish','token':'old','tool_returned':True,'execution_plan_version':1,'origin_goal_version':1,'result':oldresult,'result_ref':ref('oldcall',oldresult,False),'artifacts':{}}
 current={'contract':copy.deepcopy(contract),'status':'failed','token':'current','tool_returned':True,'dispatch_phase':'entered_tool','job_ids':[],'owner_pid':os.getpid(),'execution_plan_version':2,'origin_goal_version':1,'result':failed,'result_ref':ref('badcall',failed,True),'attempt_history':[copy.deepcopy(old)]}
 prior_goal={'version':1,'execution_authorized':True,'approved_plan_version':1,'approved_nodes':[copy.deepcopy(contract)]};goal={**prior_goal,'approved_plan_version':2};state={'username':'u','conv_id':'c','status':'needs_user','plan_version':2,'goal_contract':copy.deepcopy(goal),'nodes':{'doc':current},'plan_history':[{'plan_version':1,'goal_contract':prior_goal,'nodes':{'doc':copy.deepcopy(old)}}]}
 path=tmp_path/'workflow.json';path.write_text(json.dumps({'workflows':{'w':state},'leases':{}}));return root,source,state,goal,WorkflowStore(path)

def test_exact_native_attempt_restores_without_new_download(tmp_path):
 root,source,state,goal,store=fixture(tmp_path);proof=prove_archived_download(state,'doc',root,goal,owner={'username':'u','conv_id':'c'})
 assert store.restore_archived_download('w','doc','current',goal,'u','c',root,proof,control_valid=lambda:True)
 node=store.snapshot('w')['nodes']['doc'];assert node['status']=='succeeded' and node['token']=='old'
 assert node['attempt_history'][-1]['token']=='current' and proof['no_download'] and proof['no_file_write']
 assert source.read_bytes()==b'%PDF-1.4 original'

@pytest.mark.parametrize('tamper',['source','old_call','failed_call','contract','archive','duplicate','owner','pause','proof','control'])
def test_changed_identity_or_control_cannot_restore(tmp_path,tamper):
 root,source,state,goal,store=fixture(tmp_path);proof=prove_archived_download(state,'doc',root,goal,owner={'username':'u','conv_id':'c'})
 if tamper=='source':source.write_bytes(b'%PDF-1.4 changed')
 if tamper=='old_call':(root/'evidence/oldcall.json').write_text('{}')
 if tamper=='failed_call':(root/'evidence/badcall.json').write_text('{}')
 if tamper=='proof':proof['proof_sha256']='0'*64
 with json_transaction(store.path) as d:
  w=d['workflows']['w'];n=w['nodes']['doc']
  if tamper=='contract':n['contract']['arguments']['url']='https://other.invalid'
  if tamper=='archive':w['plan_history'][0]['nodes']['doc']['token']='tampered'
  if tamper=='duplicate':n['attempt_history'].append(copy.deepcopy(n['attempt_history'][0]))
  if tamper=='owner':w['username']='other'
  if tamper=='pause':w['user_paused']=True
 assert not store.restore_archived_download('w','doc','current',goal,'u','c',root,proof,control_valid=lambda:tamper!='control')
 assert store.snapshot('w')['nodes']['doc']['status']=='failed'

def test_user_control_change_during_proof_blocks_commit(tmp_path):
 root,source,state,goal,store=fixture(tmp_path);proof=prove_archived_download(state,'doc',root,goal,owner={'username':'u','conv_id':'c'})
 calls=[]
 def valid():
  calls.append(True)
  return len(calls)==1
 assert not store.restore_archived_download('w','doc','current',goal,'u','c',root,proof,control_valid=valid)
 assert len(calls)==2 and store.snapshot('w')['nodes']['doc']['status']=='failed'
