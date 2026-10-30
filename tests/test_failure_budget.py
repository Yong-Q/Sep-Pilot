from agents.failure_budget import CrossDelegationFailureBudget,progress_epoch

def e(nodes=None,draft=None):
 return progress_epoch({'version':1,'approved_plan_version':0,'parameters':{}}, {'nodes':nodes or {}},draft or {})

def fail(message='compiler guard',kind='workflow_contract'):
 return {'error':message,'error_kind':kind,'executed':False}

def test_identical_error_across_other_successes_is_bounded():
 b=CrossDelegationFailureBudget();epoch=e();args={'base_version':0}
 assert b.observe('compile',args,fail(),epoch) is None
 assert b.observe('read_file',{'path':'a'},{'ok':True},epoch) is None
 assert b.observe('compile',args,fail(),epoch) is None
 r=b.observe('compile',args,fail(),epoch)
 assert r['attempts']==3 and r['automatic_retry'] is False
 assert 'compiler guard' not in str(r)

def test_distinct_targets_or_errors_do_not_conflate():
 b=CrossDelegationFailureBudget();epoch=e()
 assert b.observe('read',{'path':'a'},fail('bad input'),epoch) is None
 assert b.observe('read',{'path':'b'},fail('bad input'),epoch) is None
 assert b.observe('read',{'path':'a'},fail('other error'),epoch) is None
 assert b.observe('read',{'path':'a'},fail('bad input'),epoch) is None
 assert b.observe('read',{'path':'a'},fail('bad input'),epoch)['attempts']==3

def test_different_scientific_parameters_are_not_the_same_failure():
 b=CrossDelegationFailureBudget();epoch=e()
 for temperature in (298,308,318):
  assert b.observe('compute',{'temperature':temperature},fail('solver failed'),epoch) is None
 assert b.observe('compute',{'temperature':298},fail('solver failed'),epoch) is None
 assert b.observe('compute',{'temperature':298},fail('solver failed'),epoch)['attempts']==3

def test_real_progress_or_same_tool_success_resets_budget():
 b=CrossDelegationFailureBudget();a=e();changed=e({'s':{'status':'succeeded','job_ids':['1']}})
 for _ in range(2):assert b.observe('compile',{},fail(),a) is None
 assert b.observe('compile',{},fail(),changed) is None
 assert b.observe('compile',{}, {'ok':True},changed) is None
 for _ in range(2):assert b.observe('compile',{},fail(),changed) is None
 assert b.observe('compile',{},fail(),changed)['attempts']==3

def test_draft_version_only_is_not_real_progress():
 d={'nodes':[{'step_id':'x'}],'completion_criteria':'report','version':1}
 assert e(draft=d)==e(draft={**d,'version':2})
 assert e(draft=d)!=e(draft={**d,'nodes':[{'step_id':'y'}]})

def test_benign_waits_and_non_error_do_not_count():
 b=CrossDelegationFailureBudget();epoch=e()
 for _ in range(5):assert b.observe('watch',{}, {'status':'waiting','blocked':True},epoch) is None

def test_native_session_stops_same_failure_without_claiming_completion(tmp_path):
 from types import SimpleNamespace
 from anthropic.types import ToolUseBlock
 from agents.agent import Agent
 from agents.config import AgentConfig
 from agents.session import Session
 calls=[]
 def probe(reason):
  calls.append(reason)
  return {'error':'same guard','error_kind':'workflow_contract','executed':False}
 session=Session(config=AgentConfig(api_key='test',project_root=tmp_path))
 session.messages=[{'role':'user','content':'Inspect and report honestly.'}]
 agent=Agent(name='analyst',functions=[probe])
 sequence=iter([SimpleNamespace(content=[ToolUseBlock(type='tool_use',id='x'+str(i),name='probe',input={'reason':'explanation '+str(i)})]) for i in range(3)])
 session._call_api=lambda agent: next(sequence)
 answer=session._execute_loop(agent)
 assert len(calls)==3
 assert session.task_complete is False
 assert session.context['bounded_repeated_failure']['attempts']==3
 assert '尚未完成' in answer and '停止本轮自动重试' in answer
