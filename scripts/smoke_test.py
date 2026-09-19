from __future__ import annotations

import json
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_WORKFLOW = PROJECT_ROOT / "scripts" / "run_workflow.py"

SMOKE_SPECS = [
    ("cif lookup", PROJECT_ROOT / "examples" / "specs" / "cif_lookup.json", False),
    ("literature rag query", PROJECT_ROOT / "examples" / "specs" / "literature_rag_query.json", False),
    ("results inspect path", PROJECT_ROOT / "examples" / "specs" / "results_inspect_path.json", False),
    ("results inspect job", PROJECT_ROOT / "examples" / "specs" / "results_inspect_job.json", False),
    ("structure plan", PROJECT_ROOT / "examples" / "specs" / "structure_gen.json", False),
    ("guest forcefield build", PROJECT_ROOT / "examples" / "specs" / "guest_forcefield_build.json", True),
    ("string asset prep", PROJECT_ROOT / "examples" / "specs" / "string_tst_generate.json", True),
    ("external potential asset prep", PROJECT_ROOT / "examples" / "specs" / "external_potential_generate.json", True),
    ("charge to vext workflow", PROJECT_ROOT / "examples" / "specs" / "workflow_charge_to_external_potential.json", False),
]


def run_one(label: str, spec_path: Path, execute: bool) -> dict:
    command = ["python3", str(RUN_WORKFLOW), "--project", "smoke", "--spec", str(spec_path)]
    if execute:
        command.append("--execute")
    proc = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )
    payload = {
        "label": label,
        "spec": str(spec_path),
        "execute": execute,
        "returncode": proc.returncode,
        "status": "passed" if proc.returncode == 0 else "failed",
    }
    if proc.stdout.strip():
        try:
            payload["stdout_json"] = json.loads(proc.stdout)
        except Exception:
            payload["stdout"] = proc.stdout
    if proc.stderr.strip():
        payload["stderr"] = proc.stderr
    return payload


def main() -> int:
    results = [run_one(label, spec_path, execute) for label, spec_path, execute in SMOKE_SPECS]
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))
    return 0 if all(item["status"] == "passed" for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
