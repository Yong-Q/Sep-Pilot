import copy,hashlib,json
import pytest
from agents.preparation_identity_bridge import prove_historical_preparation

def fixture(tmp_path):
 root=tmp_path/'runs/u/c';(root/'evidence').mkdir(parents=True);(root/'workers').mkdir();inp=root/'input';inp.mkdir();f=inp/'a.dat';f.write_bytes(b'approved input')
 args={'action':'inputs','input_dir':str(inp),'gas':'CH4'};contract={'step_id':'prep','agent':'analyst','tool':'run_cdft','arguments':args,'depends_on':[],'expected_outputs':[str(inp)]}
 manifest={'verification':'fresh_recipe_byte_comparison','input_dir':str(inp),'n_inputs':1,'items':[{'path':'a.dat','sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'bytes':len(f.read_bytes())}]};result={'submitted':False,'reused_existing_inputs':True,'input_dir':str(inp),'input_manifest':manifest,'input_generation_arguments':args}
 def ref(cid):
  call={'call_id':cid,'time':1,'agent':'analyst','tool':'run_cdft','params':args,'failed':False,'result':json.dumps(result)};path=root/'evidence'/(cid+'.json');path.write_text(json.dumps(call));entry={k:call[k]for k in ('call_id','time','agent','tool','params','failed')};entry.update(evidence_path=str(path),result_preview=call['result'][:500],result_sha256=hashlib.sha256(call['result'].encode()).hexdigest());cp=root/'workers'/(cid+'.json');cp.write_text(json.dumps({'owner':{'username':'u','conv_id':'c','step_id':'prep'},'checkpoint_at':2,'memory':{'tool_call_log':[entry]}}));return {'call_id':cid,'evidence_path':str(path),'checkpoint_path':str(cp)}
 old={'status':'succeeded','phase':'finish','token':'old','contract':copy.deepcopy(contract),'execution_plan_version':2,'origin_goal_version':1,'result':copy.deepcopy(result),'result_ref':ref('oldcall')}
 now={'status':'succeeded','phase':'finish','token':'new','contract':copy.deepcopy(contract),'execution_plan_version':3,'origin_goal_version':1,'result':copy.deepcopy(result),'result_ref':ref('newcall'),'attempt_history':[copy.deepcopy(old)]}
 state={'username':'u','conv_id':'c','nodes':{'prep':now},'plan_history':[{'plan_version':2,'goal_contract':{'approved_nodes':[copy.deepcopy(contract)]},'nodes':{'prep':copy.deepcopy(old)}}]}
 return root,f,state,copy.deepcopy(old['result_ref'])

def test_exact_historic_and_current_receipts_are_bound_without_resubmission(tmp_path):
 root,f,state,dep=fixture(tmp_path);proof=prove_historical_preparation(state,'prep',root,{'username':'u','conv_id':'c'},dep)
 assert proof['old_call']=='oldcall' and proof['current_call']=='newcall' and proof['no_resubmission']

@pytest.mark.parametrize('tamper',['owner','missing_archive','contract','old_manifest','new_manifest','old_call','current_call','checkpoint','dependency','duplicate','changed_hash','file'])
def test_recovery_does_not_equate_unproven_preparations(tmp_path,tamper):
 root,f,state,dep=fixture(tmp_path);n=state['nodes']['prep'];old=n['attempt_history'][0]
 if tamper=='owner':state['username']='other'
 if tamper=='missing_archive':state['plan_history']=[]
 if tamper=='contract':old['contract']['arguments']['gas']='N2'
 if tamper=='old_manifest':old['result']['input_manifest']['items'][0]['sha256']='0'*64
 if tamper=='new_manifest':n['result']['input_manifest']['items'][0]['sha256']='0'*64
 if tamper=='old_call':(root/'evidence/oldcall.json').write_text('{}')
 if tamper=='current_call':(root/'evidence/newcall.json').write_text('{}')
 if tamper=='checkpoint':(root/'workers/oldcall.json').write_text('{}')
 if tamper=='dependency':dep['checkpoint_path']=str(root/'workers/newcall.json')
 if tamper=='duplicate':n['attempt_history'].append(copy.deepcopy(old))
 if tamper=='changed_hash':n['result']['input_generation_arguments']['gas']='N2'
 if tamper=='file':f.write_bytes(b'tampered input')
 with pytest.raises((ValueError,PermissionError,OSError)):prove_historical_preparation(state,'prep',root,{'username':'u','conv_id':'c'},dep)
