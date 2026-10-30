from agents.defns import ORCHESTRATOR, SUPERVISOR, PATCHER
from bimem_agent.catalog import skill_index


def test_orchestration_guidance_is_discoverable_and_loaded_by_lifetime_roles():
    skill = skill_index()['parallel-task-executor']
    assert skill['automation_level'] == 'metadata'
    path = '.claude/skills/parallel-task-executor/SKILL.md'
    for role in (ORCHESTRATOR, SUPERVISOR):
        assert path in role.skill_files
        tools_before = set(role.function_map())
        assert 'Parallel Task Orchestration' in role.get_instructions()
        assert set(role.function_map()) == tools_before
    assert path not in PATCHER.skill_files
