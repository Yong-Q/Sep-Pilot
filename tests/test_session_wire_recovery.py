"""An invalid model proposal is corrected without executing any of its batch."""
from types import SimpleNamespace

from agents.agent import Agent
from agents.config import AgentConfig
from agents.codex_provider import CodexToolArgumentError, CodexToolCallError
from agents.session import Session


def test_session_recovers_wire_error_without_executing_empty_argument_tool(tmp_path):
    from anthropic.types import TextBlock, ToolUseBlock
    executed = []
    def list_my_jobs():
        executed.append(True)
        return {'count': 0, 'jobs': []}
    agent = Agent(name='lead-orchestrator', instructions='', functions=[list_my_jobs])
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    responses = iter([
        CodexToolArgumentError((CodexToolCallError('toolu_bad', 'list_my_jobs', 'Invalid JSON; not executed.'),)),
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='toolu_fixed', name='list_my_jobs', input={})]),
        SimpleNamespace(content=[TextBlock(type='text', text='There are no jobs.')]),
    ])
    def model(_):
        response = next(responses)
        if isinstance(response, Exception):
            raise response
        return response
    s._call_api = model
    assert 'no jobs' in s._execute_loop(agent)
    assert executed == [True]
    assert s._api_failures == 0
    observations = [block for msg in s.messages if msg.get('role') == 'user' and isinstance(msg.get('content'), list)
                    for block in msg['content'] if isinstance(block, dict) and block.get('tool_use_id') == 'toolu_bad']
    assert len(observations) == 1 and observations[0]['is_error']
    assert '"executed": false' in observations[0]['content']
    assert len(s.memory.tool_call_log) == 1


def test_call_api_preserves_safe_wire_error_instead_of_provider_outage(tmp_path, monkeypatch):
    import pytest
    error = CodexToolArgumentError((CodexToolCallError('toolu_bad', 'list_my_jobs', 'Invalid JSON; not executed.'),))
    def provider(**kwargs):
        raise error
    monkeypatch.setattr('agents.codex_provider.call_codex_model', provider)
    s = Session(config=AgentConfig(model_provider='codex_cli', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'status'}]
    with pytest.raises(CodexToolArgumentError) as caught:
        s._call_api(Agent(name='lead-orchestrator', functions=[]))
    assert caught.value is error
