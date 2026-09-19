from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def collect(run_root: Path, output: Path) -> int:
    rows = []
    ok_count = 0

    for work_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        summary_path = work_dir / "run_summary.json"
        payload = {
            "status": "missing",
            "returncode": "",
            "input_dat": "",
            "vext_path": "",
            "stdout_path": "",
            "stderr_path": "",
            "error": "run_summary.json missing",
        }
        if summary_path.exists():
            try:
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
            except Exception as exc:
                payload = {
                    "status": "bad_summary",
                    "returncode": "",
                    "input_dat": "",
                    "vext_path": "",
                    "stdout_path": "",
                    "stderr_path": "",
                    "error": f"Could not parse run_summary.json: {exc}",
                }

        if payload.get("status") == "ok":
            ok_count += 1

        rows.append(
            {
                "structure": work_dir.name,
                "status": payload.get("status", ""),
                "returncode": payload.get("returncode", ""),
                "input_dat": payload.get("input_dat", ""),
                "vext_path": payload.get("vext_path", ""),
                "stdout_path": payload.get("stdout_path", ""),
                "stderr_path": payload.get("stderr_path", ""),
                "error": payload.get("error", ""),
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "structure",
                "status",
                "returncode",
                "input_dat",
                "vext_path",
                "stdout_path",
                "stderr_path",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(
        json.dumps(
            {
                "run_root": str(run_root.resolve()),
                "output": str(output.resolve()),
                "count": len(rows),
                "ok_count": ok_count,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect per-structure Vext execution summaries into a CSV.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    return collect(Path(args.run_root).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    raise SystemExit(main())
