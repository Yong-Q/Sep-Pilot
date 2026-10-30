import copy,hashlib,json
import pytest
from agents.request_projection import project_control_history

@pytest.mark.parametrize('tamper',['none','owner','result','witness','tool','call'])
def test_large_owned_search_projection_preserves_evidence_and_pairing(tmp_path,tamper):
 root=tmp_path/'runs'/'u'/'c';evidence=root/'evidence';evidence.mkdir(parents=True)
 args={'pattern':'convergence','path':'input','max_results':50};cid='cafe1234'
 result={'total_matches':30,'results':['input.csv:1:'+('x'*1500)]*30,'truncated':False,'read_only':True,'status':'matched','exit_code':0,'process_exit_code':0}
 saved={'call_id':cid,'tool':'grep_search','params':args,'result':json.dumps(result),'failed':False}
 path=evidence/(cid+'.json');path.write_text(json.dumps(saved))
 payload={**result,'evidence_ref':{'call_id':cid,'path':str(path),'tool':'grep_search'}}
 messages=[{'role':'assistant','content':[{'type':'tool_use','id':'tool-1','name':'grep_search','input':args}]},{'role':'user','content':[{'type':'tool_result','tool_use_id':'tool-1','content':json.dumps(payload)}]}]
 original=copy.deepcopy(messages);owner={'username':'u','conv_id':'c'}
 log=[{'call_id':cid,'tool':'grep_search','params':args,'failed':False,'result_sha256':hashlib.sha256(saved['result'].encode()).hexdigest()}]
 if tamper=='owner':owner['conv_id']='other'
 elif tamper=='result':saved['result']=json.dumps({**result,'total_matches':31});path.write_text(json.dumps(saved))
 elif tamper=='witness':log[0]['result_sha256']='0'*64
 elif tamper=='tool':messages[0]['content'][0]['name']='read_file'
 elif tamper=='call':messages[0]['content'][0]['input']={'pattern':'different'}
 source=copy.deepcopy(messages);view,stats=project_control_history(messages,evidence_root=evidence,owner=owner,trusted_calls=log)
 assert messages==source and view[1]['content'][0]['tool_use_id']=='tool-1'
 if tamper=='none':
  assert stats['projected_receipts']==1 and stats['characters_after']<stats['characters_before']/2
  projected=json.loads(view[1]['content'][0]['content']);assert projected['model_view_truncated'] is True
  assert projected['request_view_provenance']['call_id']==cid and projected['total_matches']==30
  assert json.loads(path.read_text())==saved and original==messages
 else:
  assert stats['projected_receipts']==0 and view==messages

def test_new_grep_long_lines_are_bounded_and_labeled(tmp_path):
 from agents.registry import _exec_grep_search
 path=tmp_path/'long.json';path.write_text('{"needle":"'+('x'*20000)+'"}\nnormal needle\n')
 result=_exec_grep_search({'pattern':'needle','path':str(path),'max_results':50})
 assert result['total_matches']==2 and result['truncated'] is True and result['long_lines_truncated']==1
 assert len(result['results'][0])<2200 and result['results'][0].startswith(str(path)+':1:')
 assert result['results'][1].endswith('normal needle') and path.read_text().endswith('normal needle\n')

def test_other_large_tool_result_is_owned_and_recoverable(tmp_path):
 root=tmp_path/'runs'/'u'/'c';evidence=root/'evidence';evidence.mkdir(parents=True)
 args={'path':'/source'};cid='f00d1234';result={'ok':True,'status':'succeeded','rows':[{'value':i,'note':'n'*2500} for i in range(40)]}
 saved={'call_id':cid,'tool':'read_file','params':args,'result':json.dumps(result),'failed':False};path=evidence/(cid+'.json');path.write_text(json.dumps(saved))
 obj={**result,'evidence_ref':{'call_id':cid,'path':str(path)}}
 messages=[{'role':'user','content':'明确任务和科学约束'}, {'role':'assistant','content':[{'type':'tool_use','id':'use','name':'read_file','input':args}]}, {'role':'user','content':[{'type':'tool_result','tool_use_id':'use','content':json.dumps(obj)}]}]
 original=copy.deepcopy(messages);witness=[{'call_id':cid,'tool':'read_file','params':args,'failed':False,'result_sha256':hashlib.sha256(saved['result'].encode()).hexdigest()}]
 view,stats=project_control_history(messages,evidence_root=evidence,owner={'username':'u','conv_id':'c'},trusted_calls=witness)
 assert messages==original and view[0]==original[0] and view[1]==original[1]
 projected=json.loads(view[2]['content'][0]['content']);assert projected['status']=='succeeded' and projected['model_view_truncated']
 assert projected['rows'][-1]['model_view_omitted_items']==10 and projected['request_view_provenance']['call_id']==cid
 assert json.loads(path.read_text())==saved and stats['projected_receipts']==1

def test_over_budget_fails_before_request_without_mutating_user_text():
 from agents.request_budget import ensure_request_budget
 text='科学限制和原始用户要求'*100
 request={'messages':[{'role':'user','content':text}]};original=copy.deepcopy(request)
 with pytest.raises(ValueError,match='未发送模型请求'):
  ensure_request_budget(request,limit=20)
 assert request==original and ensure_request_budget(request,limit=100000)>0
