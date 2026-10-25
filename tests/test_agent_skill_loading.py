import pytest

from agents.agent import Agent


def test_role_skill_is_loaded_when_instructions_are_built(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.agent.PROJECT_ROOT", tmp_path)
    skill = tmp_path / ".claude/skills/delivery/SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("Verify the user interaction before delivery.")
    role = Agent(instructions=lambda ctx: "Task: " + ctx["task"],
                 skill_files=[".claude/skills/delivery/SKILL.md"])
    assert "Verify the user interaction" in role.get_instructions({"task": "fix UI"})
    skill.write_text("Verify restart and dependency continuation.")
    assert "Verify restart" in role.get_instructions({"task": "fix UI"})
    assert "Verify the user interaction" not in role.get_instructions({"task": "fix UI"})


def test_missing_required_role_skill_is_not_silently_ignored(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.agent.PROJECT_ROOT", tmp_path)
    with pytest.raises(FileNotFoundError):
        Agent(skill_files=[".claude/skills/missing/SKILL.md"]).get_instructions()


def test_role_skill_cannot_reference_outside_project(tmp_path, monkeypatch):
    monkeypatch.setattr("agents.agent.PROJECT_ROOT", tmp_path)
    with pytest.raises(ValueError, match="project"):
        Agent(skill_files=["../outside/SKILL.md"]).get_instructions()
