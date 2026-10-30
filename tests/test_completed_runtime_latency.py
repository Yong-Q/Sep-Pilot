from types import SimpleNamespace
from anthropic.types import TextBlock, ToolUseBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.session import Session


def test_completed_dag_result_inspection_does_not_force_legacy_handoff(tmp_path):
    def read_file(path):
        return {'path': path, 'content': 'native result evidence'}
    def handoff_to_analyst(task, context=''):
        raise AssertionError('completed result inspection does not require another delegation')
    role = Agent(name='lead-orchestrator', functions=[read_file, handoff_to_analyst])
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.goal_contract.approved_nodes = [{'step_id': 'old', 'tool': 'run_md_optimize'}]
    s._runtime_snapshot = lambda: {'status': 'completed', 'nodes': {}}
    s.messages = [{'role': 'user', 'content': '核对已有结果，给出最后阶段。'}]
    replies = iter([SimpleNamespace(content=[ToolUseBlock(type='tool_use', id=f'r{i}', name='read_file',
                    input={'path': str(tmp_path / f'result{i}.txt')})]) for i in range(4)] +
                   [SimpleNamespace(content=[TextBlock(type='text', text='The final result is verified.')])])
    s._call_api = lambda agent: next(replies)
    s._execute_loop(role)
    assert not any('[系统·快速委派]' in str(message.get('content')) for message in s.messages)
