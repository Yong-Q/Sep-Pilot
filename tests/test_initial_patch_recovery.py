import pytest
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.session import Session


@pytest.mark.parametrize('message', [
    '继续执行', '先不要继续执行', '不确认这个方案，解释一下',
    'do not retry', 'continue explaining the methods',
])
def test_recovery_language_goes_to_model_without_implicitly_replaying_patch(tmp_path, monkeypatch, message):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session.current_agent = ORCHESTRATOR
    session._error_branches = {'old': {
        'status': 'open', 'failed_step': 'propose_workflow_patch',
        'failed_arguments': {'base_version': 0, 'changes': [{'step_id': 'old'}]},
    }}
    monkeypatch.setattr(session, '_propose_workflow_patch',
        lambda **kwargs: pytest.fail('Words alone replayed a historical patch'))
    monkeypatch.setattr(session, '_execute_loop', lambda agent, **kwargs: 'model receives the real request')
    assert session.reply(message, verbose=False) == 'model receives the real request'
    assert message in str(session.messages[-1]['content'])
