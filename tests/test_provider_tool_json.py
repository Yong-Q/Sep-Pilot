from types import SimpleNamespace
import anthropic,httpx,pytest,traceback
from anthropic.types import TextBlock
from agents.agent import Agent
from agents.config import AgentConfig
from agents.provider_tool_json import MalformedProviderToolJSON,malformed_tool_json
from agents.session import Session

def bad_request(message,status=400):
 req=httpx.Request('POST','https://example.invalid');response=httpx.Response(status,request=req)
 cls=anthropic.BadRequestError if status==400 else anthropic.AuthenticationError
 return cls(message,response=response,body={'error':{'message':message}})

def test_only_exact_provider_format_error_is_classified():
 assert malformed_tool_json(bad_request('Unable to parse tool parameter JSON from model.'))
 assert not malformed_tool_json(bad_request('unrelated invalid request'))
 assert not malformed_tool_json(bad_request('Unable to parse tool parameter JSON from model.',401))
 assert not malformed_tool_json(ValueError('Unable to parse tool parameter JSON from model.'))
 injected=bad_request('Unable to parse tool parameter JSON from model.');injected.body={'error':{'message':'unrelated'}}
 assert not malformed_tool_json(injected)

def test_call_api_converts_exact_400_without_exposing_provider_text(tmp_path):
 s=Session(config=AgentConfig(api_key='test',project_root=tmp_path));s.messages=[{'role':'user','content':'status'}]
 def fail(**kwargs):raise bad_request('Unable to parse tool parameter JSON from model. secret-token')
 s._stream_api=fail
 with pytest.raises(MalformedProviderToolJSON) as error:s._call_api(Agent(name='lead-orchestrator',functions=[]))
 assert 'secret-token' not in str(error.value)
 assert 'secret-token' not in ''.join(traceback.format_exception(error.value))

@pytest.mark.parametrize('failures',[1,2])
def test_session_bounded_correction_has_no_speculative_tool_execution(tmp_path,failures):
 calls=[]
 def read_file():calls.append('executed');return {'ok':True}
 agent=Agent(name='lead-orchestrator',instructions='',functions=[read_file]);s=Session(config=AgentConfig(api_key='test',project_root=tmp_path))
 s.messages=[{'role':'user','content':'原任务'}];responses=[MalformedProviderToolJSON('safe')]*failures+[SimpleNamespace(content=[TextBlock(type='text',text='完成已有记录的说明。')])]
 def model(_):
  value=responses.pop(0)
  if isinstance(value,Exception):raise value
  return value
 s._call_api=model;answer=s._execute_loop(agent)
 assert not calls and len(s.context['provider_tool_json_errors'])==failures
 assert all(x['executed'] is False for x in s.context['provider_tool_json_errors'])
 if failures==1:assert '说明' in answer
 else:assert '连续两次' in answer and not s.task_complete
