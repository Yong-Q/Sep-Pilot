import copy
from types import SimpleNamespace

from agents.agent import Agent
from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.session import Session


def test_changing_runtime_and_clock_preserves_static_provider_prefix(tmp_path, monkeypatch):
    seen = []
    def provider(**kwargs):
        seen.append(copy.deepcopy(kwargs))
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='ok')], stop_reason='end_turn',
                               usage=SimpleNamespace(input_tokens=50, output_tokens=2, cache_read_input_tokens=100))
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    monkeypatch.setattr(session, '_stream_api', provider)
    role = Agent(name='analyst', instructions='stable role policy', functions=[])
    session.goal_contract = GoalContract.from_user_message('用GCMC计算CO2，298K，直接执行')
    session.messages = [{'role': 'user', 'content': 'my actual request'}]
    original = copy.deepcopy(session.messages)
    session.context['execution_budget'] = {'remaining': 8}
    session._call_api(role)
    session.context['execution_budget'] = {'remaining': 7}
    session.goal_contract.parameters['cycles'] = 1000
    session._call_api(role)
    assert seen[0]['system'] == seen[1]['system']
    assert seen[0]['messages'] != seen[1]['messages']
    assert '1000' in str(seen[1]['messages'])
    assert session.messages == original
    assert session.context['model_call_metrics'][-1]['usage']['cache_hit_tokens'] == 100


def test_runtime_view_preserves_paired_tool_result_and_original_history():
    from agents.cache_context import with_runtime_context
    messages = [{'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'a', 'name': 'check_job', 'input': {}}]},
                {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'a', 'content': 'real evidence'}]}]
    original = copy.deepcopy(messages)
    view = with_runtime_context(messages, 'current state')
    assert view[1]['content'][0] == original[1]['content'][0]
    assert view[1]['content'][1]['type'] == 'text'
    assert messages == original
