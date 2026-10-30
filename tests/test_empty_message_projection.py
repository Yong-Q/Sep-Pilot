import copy
import pytest
from agents.request_projection import drop_empty_assistant_messages


def test_empty_assistant_is_removed_only_in_request_view():
    messages=[{'role':'user','content':'source'}, {'role':'assistant','content':[]},
              {'role':'user','content':'continue'}]
    before=copy.deepcopy(messages)
    assert drop_empty_assistant_messages(messages)==[messages[0],messages[2]]
    assert messages==before


def test_tool_pairs_and_thinking_stay_intact():
    messages=[{'role':'assistant','content':[{'type':'thinking','thinking':'reason','signature':'x'},
                {'type':'tool_use','id':'a','name':'inspect','input':{}}]},
              {'role':'user','content':[{'type':'tool_result','tool_use_id':'a','content':'evidence'}]}]
    assert drop_empty_assistant_messages(messages)==messages


@pytest.mark.parametrize('content',['', '  ', []])
def test_empty_user_is_not_deleted(content):
    with pytest.raises(ValueError,match='non-assistant'):
        drop_empty_assistant_messages([{'role':'user','content':content}])


def test_tool_results_after_empty_assistant_are_not_dropped_or_repaired():
    messages=[{'role':'assistant','content':[]},{'role':'user','content':[{'type':'tool_result','tool_use_id':'a','content':'evidence'}]}]
    with pytest.raises(ValueError,match='tool results'):drop_empty_assistant_messages(messages)


def test_unknown_or_none_content_not_invented():
    messages=[{'role':'assistant','content':None}]
    with pytest.raises(ValueError,match='unsupported shape'):drop_empty_assistant_messages(messages)


def test_only_empty_assistant_fails_before_provider():
    with pytest.raises(ValueError,match='no messages'):drop_empty_assistant_messages([{'role':'assistant','content':[]}])


def test_actual_session_provider_request_projects_without_mutating_history(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from agents.agent import Agent
    from agents.config import AgentConfig
    from agents.session import Session
    captured=[]
    monkeypatch.setattr('agents.codex_provider.call_codex_model',lambda **kw:(captured.append(copy.deepcopy(kw['messages'])) or SimpleNamespace(content=[])))
    session=Session(config=AgentConfig(model_provider='codex_cli',project_root=tmp_path))
    session.messages=[{'role':'user','content':'first'},{'role':'assistant','content':[]}, {'role':'user','content':'continue'}]
    session._call_api(Agent(name='lead-orchestrator',functions=[]))
    sent=captured[0]
    assert all(msg['content']!=[] for msg in sent)
    assert any(msg.get('role')=='user' and 'first' in str(msg.get('content')) for msg in sent)
    assert any(msg.get('role')=='user' and 'continue' in str(msg.get('content')) for msg in sent)
    assert any(msg.get('role')=='assistant' and msg.get('content')==[] for msg in session.messages)


def test_completed_tool_exchange_then_empty_assistant_and_user():
    messages=[{'role':'assistant','content':[{'type':'tool_use','id':'a','name':'inspect','input':{}}]},
              {'role':'user','content':[{'type':'tool_result','tool_use_id':'a','content':'evidence'}]},
              {'role':'assistant','content':[]},{'role':'user','content':'continue'}]
    assert drop_empty_assistant_messages(messages)==[messages[0],messages[1],messages[3]]
