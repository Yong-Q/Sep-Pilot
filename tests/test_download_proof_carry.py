import copy,hashlib,json
import pytest
from agents.presentation_transition import completed_transition_proof

def fixture(tmp_path):
 root=tmp_path/'runs'/'u'/'c';(root/'evidence').mkdir(parents=True);(root/'workers').mkdir();target=root/'source.pdf';target.write_bytes(b'%PDF-1.4 immutable original')
 params={'url':'https://example.org/source.pdf','output_path':str(target)};result={'status':'success','output_path':str(target),'output_files':[str(target)],'existing_results_overwritten':False,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'bytes':target.stat().st_size}
 call={'call_id':'abc123','time':1,'agent':'analyst','tool':'download_scientific_file','params':params,'failed':False,'result':json.dumps(result)};evidence=root/'evidence/abc123.json';evidence.write_text(json.dumps(call))
 record={k:call[k]for k in ('call_id','time','agent','tool','params','failed')};record.update(evidence_path=str(evidence),result_preview=call['result'][:500],result_sha256=hashlib.sha256(call['result'].encode()).hexdigest())
 checkpoint=root/'workers/checkpoint.json';checkpoint.write_text(json.dumps({'owner':{'username':'u','conv_id':'c','step_id':'doc'},'checkpoint_at':2,'memory':{'tool_call_log':[record]}}))
 contract={'step_id':'doc','tool':'download_scientific_file','arguments':params};node={'status':'succeeded','phase':'finish','token':'original','contract':contract,'result':result,'result_ref':{'call_id':'abc123','evidence_path':str(evidence),'checkpoint_path':str(checkpoint)}}
 return root,target,{'username':'u','conv_id':'c','goal_contract':{},'nodes':{'doc':node}}

def test_owned_unchanged_download_reuses_actual_bytes_without_network(tmp_path):
 root,target,state=fixture(tmp_path);proof=completed_transition_proof(state,'doc',root,upstream_jobs={})
 assert proof['no_reexecution'] is True and proof['current_token']=='original'
 assert target.read_bytes()==b'%PDF-1.4 immutable original'

@pytest.mark.parametrize('tamper',['file','receipt','contract','failed','foreign'])
def test_changed_download_receipt_or_source_cannot_be_reused(tmp_path,tamper):
 root,target,state=fixture(tmp_path);node=state['nodes']['doc']
 if tamper=='file':target.write_bytes(b'%PDF-1.4 changed')
 if tamper=='receipt':node['result']['sha256']='0'*64
 if tamper=='contract':node['contract']['arguments']['url']='https://other.invalid'
 if tamper=='failed':node['status']='failed'
 if tamper=='foreign':state['username']='other'
 with pytest.raises((ValueError,PermissionError,OSError)):completed_transition_proof(state,'doc',root,upstream_jobs={})
