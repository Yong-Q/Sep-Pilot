from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .paths import PROJECT_ROOT


RUNS_ROOT = PROJECT_ROOT / "runs"
INDEX_PATH = RUNS_ROOT / "index.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-")
    return slug or "run"


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _workflow_name(spec: Dict[str, Any]) -> str:
    if "workflow_name" in spec:
        return spec["workflow_name"]
    if "skill" in spec:
        return f"{spec['skill']}-{spec.get('action', 'default')}"
    return "workflow"


def load_index() -> List[Dict[str, Any]]:
    return _read_json(INDEX_PATH, [])


def save_index(index: List[Dict[str, Any]]) -> None:
    _write_json(INDEX_PATH, index)


def create_run(project: str, spec: Dict[str, Any], execute: bool, label: Optional[str] = None) -> Dict[str, Any]:
    project_slug = slugify(project)
    name_slug = slugify(label or _workflow_name(spec))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{timestamp}_{name_slug}"
    run_dir = RUNS_ROOT / project_slug / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    record = {
        "run_id": run_id,
        "project": project_slug,
        "label": label or _workflow_name(spec),
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "execute": execute,
        "status": "running",
        "run_dir": str(run_dir),
        "spec_path": str(run_dir / "spec.json"),
        "result_path": str(run_dir / "result.json"),
        "metadata_path": str(run_dir / "run.json"),
    }
    _write_json(run_dir / "spec.json", spec)
    _write_json(run_dir / "run.json", record)

    index = load_index()
    index.append(
        {
            "run_id": run_id,
            "project": project_slug,
            "label": record["label"],
            "status": "running",
            "created_at": record["created_at"],
            "updated_at": record["updated_at"],
            "execute": execute,
            "run_dir": str(run_dir),
        }
    )
    save_index(index)
    return record


def finalize_run(record: Dict[str, Any], result: Optional[Dict[str, Any]] = None, error: Optional[str] = None) -> Dict[str, Any]:
    run_dir = Path(record["run_dir"])
    if result is not None:
        _write_json(run_dir / "result.json", result)

    record = dict(record)
    record["updated_at"] = utc_now()
    record["status"] = "failed" if error else "completed"
    if error:
        record["error"] = error
    if result is not None:
        record["result_status"] = result.get("status") or result.get("workflow_name") or "recorded"
    _write_json(run_dir / "run.json", record)

    index = load_index()
    for item in index:
        if item["run_id"] == record["run_id"]:
            item["updated_at"] = record["updated_at"]
            item["status"] = record["status"]
            if error:
                item["error"] = error
            break
    save_index(index)
    return record


def summarize_runs(project: Optional[str] = None, limit: int = 10) -> Dict[str, Any]:
    index = load_index()
    if project:
        project_slug = slugify(project)
        index = [item for item in index if item["project"] == project_slug]

    index = sorted(index, key=lambda item: item["created_at"], reverse=True)
    recent = index[:limit]

    counts: Dict[str, int] = {}
    projects: Dict[str, int] = {}
    for item in index:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
        projects[item["project"]] = projects.get(item["project"], 0) + 1

    return {
        "project": slugify(project) if project else None,
        "total_runs": len(index),
        "status_counts": counts,
        "projects": projects,
        "recent_runs": recent,
    }


def load_run(run_id: str, project: Optional[str] = None) -> Dict[str, Any]:
    candidates = []
    if project:
        candidates.append(RUNS_ROOT / slugify(project) / run_id / "run.json")
    else:
        for path in RUNS_ROOT.glob(f"*/{run_id}/run.json"):
            candidates.append(path)

    for path in candidates:
        if path.exists():
            return _read_json(path, {})
    raise FileNotFoundError(f"Run not found: {run_id}")
