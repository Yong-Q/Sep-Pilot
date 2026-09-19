from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bimem_agent.run_store import load_run, summarize_runs


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect BiMemAgent run ledger.")
    parser.add_argument("--project", help="Filter by project slug or label.")
    parser.add_argument("--run-id", help="Show one specific run record.")
    parser.add_argument("--limit", type=int, default=10, help="Number of recent runs to show.")
    args = parser.parse_args()

    if args.run_id:
        print(json.dumps(load_run(args.run_id, project=args.project), ensure_ascii=False, indent=2))
        return 0

    print(json.dumps(summarize_runs(project=args.project, limit=args.limit), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
