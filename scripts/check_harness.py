from __future__ import annotations

from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bimem_agent.catalog import load_catalog


def _check_exists(path: Path, errors: list[str]) -> None:
    if not path.exists():
        errors.append(f"Missing: {path}")


def _validate_claude_pointer(errors: list[str]) -> None:
    path = PROJECT_ROOT / "CLAUDE.md"
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"Unreadable CLAUDE.md: {path} ({exc})")
        return
    for token in ("## Harness:", "**Trigger:**", "registry/catalog.json"):
        if token not in text:
            errors.append(f"CLAUDE.md missing harness pointer token: {token}")


def _validate_example_specs(catalog: dict, errors: list[str]) -> None:
    known = set()
    for skill in catalog["skills"]:
        known.add(skill["id"])
        known.update(skill.get("aliases", []))

    specs_dir = PROJECT_ROOT / "examples" / "specs"
    for path in sorted(specs_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            errors.append(f"Invalid JSON: {path} ({exc})")
            continue

        if "steps" in data:
            if not isinstance(data["steps"], list) or not data["steps"]:
                errors.append(f"Workflow missing steps: {path}")
                continue
            for index, step in enumerate(data["steps"], start=1):
                skill = step.get("skill")
                if skill not in known:
                    errors.append(f"Unknown workflow skill in {path} step {index}: {skill}")
                if "id" not in step:
                    errors.append(f"Workflow step missing id in {path} step {index}")
        else:
            skill = data.get("skill")
            if skill not in known:
                errors.append(f"Unknown skill in {path}: {skill}")


def _validate_legacy_sync(catalog: dict, errors: list[str]) -> None:
    skill_names = {skill["id"] for skill in catalog["skills"]}
    agent_names = {agent["id"] for agent in catalog["agents"]}
    for skill in catalog["skills"]:
        skill_names.update(skill.get("aliases", []))
    for agent in catalog["agents"]:
        agent_names.update(agent.get("aliases", []))

    legacy_skills_root = PROJECT_ROOT.parent / ".claude" / "skills"
    legacy_agents_root = PROJECT_ROOT.parent / ".claude" / "agents"

    for path in sorted(legacy_skills_root.glob("*")):
        if path.is_dir() and path.name not in skill_names:
            errors.append(f"Legacy skill not represented in catalog ids/aliases: {path.name}")

    for path in sorted(legacy_agents_root.glob("*.md")):
        if path.stem not in agent_names:
            errors.append(f"Legacy agent not represented in catalog ids/aliases: {path.stem}")


def main() -> int:
    catalog = load_catalog()
    errors: list[str] = []

    for skill in catalog["skills"]:
        _check_exists(PROJECT_ROOT / ".claude" / "skills" / skill["id"] / "SKILL.md", errors)
        for alias in skill.get("aliases", []):
            _check_exists(PROJECT_ROOT / ".claude" / "skills" / alias / "SKILL.md", errors)

    for agent in catalog["agents"]:
        _check_exists(PROJECT_ROOT / ".claude" / "agents" / f"{agent['id']}.md", errors)
        for alias in agent.get("aliases", []):
            _check_exists(PROJECT_ROOT / ".claude" / "agents" / f"{alias}.md", errors)

    _check_exists(PROJECT_ROOT / "docs" / "skill-index.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "agent-index.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "workflow-spec.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "project-state.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "runtime-config.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "testing-status.md", errors)
    _check_exists(PROJECT_ROOT / "docs" / "upstream-blockers.md", errors)
    _check_exists(PROJECT_ROOT / "config" / "runtime.json", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "collect_vext_results.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "agent_trigger_smoke.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "project_status.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "run_rag_query.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "run_vext_one.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "build_guest_forcefield.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "write_raspa_case_files.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "normalize_cif_charges.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "skill_regression.py", errors)
    _check_exists(PROJECT_ROOT / "scripts" / "smoke_test.py", errors)
    _validate_claude_pointer(errors)
    _validate_example_specs(catalog, errors)
    _validate_legacy_sync(catalog, errors)

    if errors:
        for error in errors:
            print(error)
        return 1

    print("Harness check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
