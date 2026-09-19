from __future__ import annotations

import shutil
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bimem_agent.catalog import load_catalog


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _is_generated_file(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "generated: true" in text


def _example_specs(skill: dict) -> list[str]:
    specs_dir = PROJECT_ROOT / "examples" / "specs"
    keys = [skill["id"], *skill.get("aliases", [])]
    normalized = {key.replace("-", "_") for key in keys}
    matches = []
    for path in sorted(specs_dir.glob("*.json")):
        stem = path.stem
        if any(token in stem for token in normalized):
            matches.append(str(path.relative_to(PROJECT_ROOT)))
    return matches


def _skill_md(skill: dict, notice: str) -> str:
    examples = _example_specs(skill)
    lines = [
        "---",
        f"name: {skill['id']}",
        "description: >",
        f"  {skill['use_when']}",
        "generated: true",
        "---",
        "",
        f"# {skill['title']}",
        "",
        f"> {notice}",
        "",
        "## Use When",
        "",
        f"- {skill['use_when']}",
        "",
        "## Do Not Use For",
        "",
        f"- {skill['do_not_use_for']}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend([f"- `{item}`" for item in skill["inputs"]])
    lines.extend(
        [
            "",
            "## Automation",
            "",
            f"- Level: `{skill['automation_level']}`",
            f"- Entry: `{skill['automation']}`",
            "- Spec format: `docs/workflow-spec.md`",
            "",
            "## Legacy Source",
            "",
            f"- `{skill['legacy_source']}`",
        ]
    )
    if examples:
        lines.extend(["", "## Example Specs", ""])
        lines.extend([f"- `{example}`" for example in examples])
    if skill.get("aliases"):
        lines.extend(["", "## Aliases", ""])
        lines.extend([f"- `{alias}`" for alias in skill["aliases"]])
    if skill.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in skill["notes"]])
    return "\n".join(lines) + "\n"


def _skill_alias_md(skill: dict, alias: str, notice: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {alias}",
            "description: >",
            f"  Legacy alias for {skill['id']}.",
            "generated: true",
            "---",
            "",
            f"# Alias: {alias}",
            "",
            f"> {notice}",
            "",
            f"- Canonical skill: `{skill['id']}`",
            f"- Title: {skill['title']}",
            f"- Legacy source: `{skill['legacy_source']}`",
        ]
    ) + "\n"


def _agent_md(agent: dict, notice: str) -> str:
    lines = [
        "---",
        f"name: {agent['id']}",
        "description: >",
        f"  {agent['role']}",
        f"model: {agent['model']}",
        "generated: true",
        "---",
        "",
        f"# {agent['title']}",
        "",
        f"> {notice}",
        "",
        "## Role",
        "",
        f"- {agent['role']}",
        "",
        "## Use When",
        "",
        f"- {agent['use_when']}",
        "",
        "## Inputs",
        "",
    ]
    lines.extend([f"- `{item}`" for item in agent["inputs"]])
    lines.extend(["", "## Outputs", ""])
    lines.extend([f"- `{item}`" for item in agent["outputs"]])
    lines.extend(["", "## Skills", ""])
    lines.extend([f"- `{item}`" for item in agent["skills"]])
    lines.extend(["", "## Workflow", ""])
    lines.extend([f"1. {step}" for step in agent["workflow"]])
    if agent.get("aliases"):
        lines.extend(["", "## Aliases", ""])
        lines.extend([f"- `{alias}`" for alias in agent["aliases"]])
    if agent.get("notes"):
        lines.extend(["", "## Notes", ""])
        lines.extend([f"- {note}" for note in agent["notes"]])
    return "\n".join(lines) + "\n"


def _agent_alias_md(agent: dict, alias: str, notice: str) -> str:
    return "\n".join(
        [
            "---",
            f"name: {alias}",
            "description: >",
            f"  Legacy alias for {agent['id']}.",
            "generated: true",
            "---",
            "",
            f"# Alias: {alias}",
            "",
            f"> {notice}",
            "",
            f"- Canonical agent: `{agent['id']}`",
            f"- Title: {agent['title']}",
        ]
    ) + "\n"


def _skill_index_md(skills: list[dict]) -> str:
    lines = [
        "# Skill Index",
        "",
        "| Canonical Skill | Aliases | Automation | Legacy Source |",
        "|---|---|---|---|",
    ]
    for skill in skills:
        aliases = ", ".join(f"`{item}`" for item in skill.get("aliases", [])) or "-"
        lines.append(
            f"| `{skill['id']}` | {aliases} | `{skill['automation_level']}` | `{skill['legacy_source']}` |"
        )
    return "\n".join(lines) + "\n"


def _agent_index_md(agents: list[dict]) -> str:
    lines = [
        "# Agent Index",
        "",
        "| Canonical Agent | Aliases | Model | Skills |",
        "|---|---|---|---|",
    ]
    for agent in agents:
        aliases = ", ".join(f"`{item}`" for item in agent.get("aliases", [])) or "-"
        skills = ", ".join(f"`{item}`" for item in agent["skills"])
        lines.append(f"| `{agent['id']}` | {aliases} | `{agent['model']}` | {skills} |")
    return "\n".join(lines) + "\n"


def _cleanup_generated(skill_paths: set[Path], agent_paths: set[Path]) -> None:
    skills_root = PROJECT_ROOT / ".claude" / "skills"
    if skills_root.exists():
        for skill_md in skills_root.glob("*/SKILL.md"):
            if skill_md in skill_paths or not _is_generated_file(skill_md):
                continue
            skill_md.unlink()
            try:
                skill_md.parent.rmdir()
            except OSError:
                pass

    agents_root = PROJECT_ROOT / ".claude" / "agents"
    if agents_root.exists():
        for agent_md in agents_root.glob("*.md"):
            if agent_md in agent_paths or not _is_generated_file(agent_md):
                continue
            agent_md.unlink()

    if skills_root.exists():
        for skill_dir in skills_root.glob("*"):
            if skill_dir.is_dir() and not any(skill_dir.iterdir()):
                shutil.rmtree(skill_dir)


def main() -> None:
    catalog = load_catalog()
    notice = catalog["meta"]["generated_notice"]
    expected_skill_paths: set[Path] = set()
    expected_agent_paths: set[Path] = set()

    for skill in catalog["skills"]:
        expected_skill_paths.add(PROJECT_ROOT / ".claude" / "skills" / skill["id"] / "SKILL.md")
        for alias in skill.get("aliases", []):
            expected_skill_paths.add(PROJECT_ROOT / ".claude" / "skills" / alias / "SKILL.md")

    for agent in catalog["agents"]:
        expected_agent_paths.add(PROJECT_ROOT / ".claude" / "agents" / f"{agent['id']}.md")
        for alias in agent.get("aliases", []):
            expected_agent_paths.add(PROJECT_ROOT / ".claude" / "agents" / f"{alias}.md")

    _cleanup_generated(expected_skill_paths, expected_agent_paths)

    for skill in catalog["skills"]:
        path = PROJECT_ROOT / ".claude" / "skills" / skill["id"] / "SKILL.md"
        _write(path, _skill_md(skill, notice))
        for alias in skill.get("aliases", []):
            alias_path = PROJECT_ROOT / ".claude" / "skills" / alias / "SKILL.md"
            _write(alias_path, _skill_alias_md(skill, alias, notice))

    for agent in catalog["agents"]:
        path = PROJECT_ROOT / ".claude" / "agents" / f"{agent['id']}.md"
        _write(path, _agent_md(agent, notice))
        for alias in agent.get("aliases", []):
            alias_path = PROJECT_ROOT / ".claude" / "agents" / f"{alias}.md"
            _write(alias_path, _agent_alias_md(agent, alias, notice))

    _write(PROJECT_ROOT / "docs" / "skill-index.md", _skill_index_md(catalog["skills"]))
    _write(PROJECT_ROOT / "docs" / "agent-index.md", _agent_index_md(catalog["agents"]))


if __name__ == "__main__":
    main()
