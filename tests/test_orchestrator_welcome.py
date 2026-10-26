from agents.defns import ORCHESTRATOR


def test_orchestrator_prompt_has_one_time_welcome_contract():
    prompt = ORCHESTRATOR.instructions

    assert "首次与用户交流" in prompt
    assert "SEP-Pilot" in prompt
    assert "不要重复自我介绍" in prompt
