from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List

from .run_store import load_run


def _detect_kind(path: Path) -> str:
    if path.is_dir():
        if (path / "OUTCAR").exists() or (path / "OSZICAR").exists():
            return "vasp-workdir"
        if (path / "results.csv").exists():
            return "result-dir"
        if path.name == "logs":
            return "log-dir"
        return "directory"

    suffix = path.suffix.lower()
    if suffix == ".csv":
        if path.name == "results.csv":
            return "results-csv"
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix in {".out", ".err", ".log"}:
        return "log-file"
    if suffix == ".pbs":
        return "pbs-script"
    if suffix == ".py":
        return "python-script"
    if suffix == ".png":
        return "plot"
    return "file"


def _preview_text(path: Path, max_lines: int = 8) -> List[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            return [line.rstrip("\n") for _, line in zip(range(max_lines), handle)]
    except Exception:
        return []


def _inspect_csv(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        rows = list(csv.reader(handle))

    return {
        "header": rows[0] if rows else [],
        "n_rows": max(len(rows) - 1, 0),
        "sample_row": rows[1] if len(rows) > 1 else [],
    }


def inspect_path(path: str, max_entries: int = 20) -> Dict[str, Any]:
    target = Path(path).expanduser().resolve()
    info: Dict[str, Any] = {
        "path": str(target),
        "exists": target.exists(),
    }
    if not target.exists():
        return info

    info["kind"] = _detect_kind(target)

    if target.is_file():
        info["size_bytes"] = target.stat().st_size
        if info["kind"] in {"csv", "results-csv"}:
            info["csv"] = _inspect_csv(target)
        elif info["kind"] in {"log-file", "pbs-script", "python-script", "json"}:
            info["preview"] = _preview_text(target)
        return info

    entries = sorted(target.iterdir(), key=lambda item: item.name)
    info["n_entries"] = len(entries)
    info["entries"] = [
        {
            "name": entry.name,
            "path": str(entry),
            "kind": _detect_kind(entry),
        }
        for entry in entries[:max_entries]
    ]

    representative = [
        item for item in entries
        if item.is_file() and item.suffix.lower() in {".csv", ".json", ".out", ".err", ".log", ".pbs", ".png"}
    ]
    if representative:
        info["highlights"] = [str(item) for item in representative[:8]]
    return info


def inspect_job_name(job_name: str, legacy_root: str, max_lines: int = 8) -> Dict[str, Any]:
    root = Path(legacy_root)
    jobs_dir = root / "gcmc_output" / "jobs"
    logs_dir = root / "gcmc_output" / "logs"
    script = jobs_dir / f"{job_name}.pbs"
    py_script = jobs_dir / f"{job_name}.py"
    out_log = logs_dir / f"{job_name}.out"
    err_log = logs_dir / f"{job_name}.err"

    payload = {
        "job_name": job_name,
        "script": inspect_path(str(script)),
        "python_script": inspect_path(str(py_script)),
        "stdout_log": inspect_path(str(out_log)),
        "stderr_log": inspect_path(str(err_log)),
    }
    if out_log.exists():
        payload["stdout_preview"] = _preview_text(out_log, max_lines=max_lines)
    if err_log.exists():
        payload["stderr_preview"] = _preview_text(err_log, max_lines=max_lines)
    return payload


def inspect_run(run_id: str, project: str | None = None) -> Dict[str, Any]:
    record = load_run(run_id, project=project)
    payload: Dict[str, Any] = {"run": record}
    result_path = Path(record["result_path"])
    spec_path = Path(record["spec_path"])
    if result_path.exists():
        payload["result"] = inspect_path(str(result_path))
    if spec_path.exists():
        payload["spec"] = inspect_path(str(spec_path))
    return payload
