import json
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.session import Session


def test_model_receives_specific_native_contract_feedback_without_fake_framework_claim(tmp_path):
 s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.current_agent=ORCHESTRATOR
 result={'error':'sandbox output path mismatch','error_kind':'workflow_contract','contract_code':'sandbox_output_declaration','executed':False,
  'step_id':'analysis','required_outputs':['/owned/analysis/artifacts/data.csv'],
  'issues':['Use a file path rather than a recursive directory glob'],'next_action':'Use the exact declared artifact path.'}
 for i in range(3):
  call='call'+str(i);encoded=json.dumps(result)
  s.memory.record_tool_call('lead-orchestrator','propose_workflow_patch',{},encoded,failed=True)
  s._handle_failures([{'type':'tool_result','tool_use_id':call,'content':encoded}],
      [{'type':'tool_use','id':call,'name':'propose_workflow_patch','input':{}}],'lead-orchestrator')
 text='\n'.join(str(x.get('content')or'')for x in s.messages if x.get('role')=='user')
 assert 'sandbox_output_declaration' in text and '/owned/analysis/artifacts/data.csv' in text
 assert 'Use the exact declared artifact path.' in text
 assert '仅在有证据表明实现与 schema 不一致时修改框架' in text
 assert s._pending_handoff.name=='patcher' and '普通参数错误直接建议修订' in s.context['automatic_handoff_task']


def test_generic_failure_does_not_become_contract_feedback(tmp_path):
 s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.current_agent=ORCHESTRATOR
 result={'error':'actual input unavailable','executed':False};encoded=json.dumps(result)
 s.memory.record_tool_call('lead-orchestrator','read_file',{},encoded,failed=True)
 s._handle_failures([{'type':'tool_result','tool_use_id':'call','content':encoded}],
     [{'type':'tool_use','id':'call','name':'read_file','input':{}}],'lead-orchestrator')
 text='\n'.join(str(x.get('content')or'')for x in s.messages if x.get('role')=='user')
 assert '[系统·参数合同修复]' not in text
