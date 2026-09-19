from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bimem_agent.legacy_bridge import pretty_json
from bimem_agent.run_store import create_run, finalize_run
from bimem_agent.task_runner import run_task_spec


def _load_spec(path: str | None) -> dict:
    if path:
        with Path(path).open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return json.load(sys.stdin)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or dry-run a BiMemAgent workflow spec.")
    parser.add_argument("--spec", help="Path to a JSON task spec. If omitted, read JSON from stdin.")
    parser.add_argument("--execute", action="store_true", help="Execute command-wrapper plans when available.")
    parser.add_argument("--project", default="default", help="Project name used for run-ledger grouping.")
    parser.add_argument("--label", default=None, help="Optional human-readable run label.")
    parser.add_argument("--no-log", action="store_true", help="Skip writing run metadata under BiMemAgent/runs/.")
    args = parser.parse_args()

    spec = _load_spec(args.spec)
    record = None
    try:
        if not args.no_log:
            record = create_run(project=args.project, spec=spec, execute=args.execute, label=args.label)
        result = run_task_spec(spec, execute=args.execute)
        if record is not None:
            record = finalize_run(record, result=result)
            result = {
                **result,
                "_run": {
                    "run_id": record["run_id"],
                    "project": record["project"],
                    "run_dir": record["run_dir"],
                    "status": record["status"],
                },
            }
        print(pretty_json(result))
        return 0
    except Exception as exc:
        if record is not None:
            finalize_run(record, error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
