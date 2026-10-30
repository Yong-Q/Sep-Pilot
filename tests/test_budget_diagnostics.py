import pytest
from agents.request_budget import ensure_request_budget

def test_budget_failure_reports_only_sizes_without_user_or_secret_content():
 secret='do-not-emit-this-token';request={'system':'system','messages':[{'role':'user','content':secret*1000}],'tools':[{'name':'tool'}]}
 with pytest.raises(ValueError) as error:ensure_request_budget(request,limit=100)
 message=str(error.value)
 assert '未发送模型请求' in message and '最大单条消息=' in message and '请求字符数=' in message
 assert secret not in message and 'system' not in message
 assert request['messages'][0]['content']==secret*1000
