"""Bounded, provenance-linked validation dossiers for workflow results."""

from __future__ import annotations

from dataclasses import dataclass
import csv
import fnmatch
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
from typing import Any


VALIDATION_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ValidationLimits:
    max_entries: int = 10_000
    max_bytes: int = 64 * 1024 * 1024
    max_depth: int = 12
    max_seconds: float = 10.0
    sample_count: int = 3


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _result_root(node: dict[str, Any]) -> Path:
    manifest = node.get("path_manifest") or {}
    result = node.get("result") or {}
    contract = node.get("contract") or {}
    arguments = contract.get("arguments") or {}
    candidates = [
        manifest.get("result_dir"),
        result.get("work_dir"),
        result.get("output_dir"),
        arguments.get("job_work_dir"),
        arguments.get("work_dir"),
        arguments.get("output_dir"),
    ]
    value = next((item for item in candidates if item), None)
    if not value:
        expected = contract.get("expected_outputs") or []
        first = expected[0] if expected else None
        value = first.get("path") if isinstance(first, dict) else first
    if not value:
        raise ValueError("workflow node has no executor-owned result root")
    return Path(value).expanduser().resolve()


def _inventory(root: Path, limits: ValidationLimits) -> dict[str, Any]:
    started = time.monotonic()
    files: list[dict[str, Any]] = []
    escaped: list[str] = []
    total_bytes = 0
    truncated = False
    reasons: list[str] = []
    if not root.exists() or not root.is_dir():
        return {"root": str(root), "file_count": 0, "total_bytes": 0, "files": [],
                "escaped_paths": [], "truncated": False, "reasons": ["result root is not a directory"]}
    for current, directories, names in os.walk(root, followlinks=False):
        current_path = Path(current)
        depth = len(current_path.relative_to(root).parts)
        directories.sort(); names.sort()
        if depth >= limits.max_depth:
            if directories:
                truncated = True; reasons.append("maximum scan depth reached")
            directories[:] = []
        safe_directories = []
        for name in directories:
            candidate = current_path / name
            resolved = candidate.resolve()
            if not _inside(resolved, root):
                escaped.append(str(candidate.relative_to(root)))
            else:
                safe_directories.append(name)
        directories[:] = safe_directories
        for name in names:
            if time.monotonic() - started > limits.max_seconds:
                truncated = True; reasons.append("maximum scan time reached"); break
            if len(files) >= limits.max_entries:
                truncated = True; reasons.append("maximum entry count reached"); break
            candidate = current_path / name
            resolved = candidate.resolve()
            if not _inside(resolved, root):
                escaped.append(str(candidate.relative_to(root))); continue
            if not resolved.is_file():
                continue
            stat = resolved.stat()
            total_bytes += stat.st_size
            files.append({"path": str(candidate.relative_to(root)), "size": stat.st_size,
                          "mtime_ns": stat.st_mtime_ns})
        if truncated:
            break
    return {"root": str(root), "file_count": len(files), "total_bytes": total_bytes,
            "files": files, "escaped_paths": sorted(set(escaped)),
            "truncated": truncated, "reasons": sorted(set(reasons))}


def _samples(root: Path, inventory: dict[str, Any], attempt_id: str, count: int) -> list[dict[str, Any]]:
    ranked = sorted(inventory["files"], key=lambda item: hashlib.sha256(
        f"{attempt_id}\0{item['path']}".encode()).hexdigest())[:max(0, count)]
    samples = []
    for item in ranked:
        path = root / item["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        samples.append({"path": item["path"], "size": item["size"], "sha256": digest,
                        "facts": {"nonempty": item["size"] > 0}})
    return samples


def _read_text(path: Path, limit: int = 256 * 1024) -> str:
    with path.open("rb") as handle:
        size = path.stat().st_size
        if size <= limit:
            raw = handle.read(limit)
        else:
            head_size = max(1, limit // 2)
            tail_size = max(1, limit - head_size)
            raw = handle.read(head_size)
            handle.seek(max(0, size - tail_size))
            raw += b"\n[... bounded middle omitted ...]\n" + handle.read(tail_size)
    return raw.decode("utf-8", errors="replace")


def _finite(value: str) -> bool:
    try:
        return math.isfinite(float(value.strip()))
    except (TypeError, ValueError):
        return False


def _classify_structure(text: str) -> tuple[bool, dict[str, Any], str]:
    valid = "_cell_length_" in text.lower() and "_atom_site_" in text.lower()
    return valid, {"valid_cif": valid}, "missing CIF cell or atom-site content" if not valid else ""


def _classify_charge(text: str) -> tuple[bool, dict[str, Any], str]:
    lower = text.lower()
    if "_atom_site_charge" not in lower:
        return False, {"finite_charge_column": False}, "missing _atom_site_charge column"
    values = []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) >= 2 and not fields[0].startswith("_") and _finite(fields[-1]):
            values.append(float(fields[-1]))
    valid = bool(values) and all(math.isfinite(value) for value in values)
    return valid, {"finite_charge_column": valid, "charge_values_checked": len(values)}, \
        "charge column has no finite values" if not valid else ""


def _classify_gcmc(text: str) -> tuple[bool, dict[str, Any], str] | None:
    lower = text.lower()
    relevant = any(word in lower for word in ("loading", "simulation finished", "simulation failed", "convergence"))
    if not relevant:
        return None
    completed = any(word in lower for word in ("simulation finished", "simulation completed", "normal termination"))
    failed_marker = any(word in lower for word in ("simulation failed", "fatal", "convergence error"))
    loading_tokens = re.findall(r"(?:loading|uptake)[^\n\r]*?([-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))",
                                text, flags=re.IGNORECASE)
    finite_loading = any(_finite(token) for token in loading_tokens)
    valid = completed and finite_loading and not failed_marker
    return valid, {"finite_loading_and_completion": valid, "completion_marker": completed,
                   "finite_loading": finite_loading}, "missing finite loading or completion marker" if not valid else ""


def _classify_cdft_table(text: str) -> tuple[list[bool], dict[str, Any]] | None:
    first = next((line for line in text.splitlines() if line.strip()), "")
    lower = first.lower()
    if "," not in first or not any(word in lower for word in ("density", "henry", "adsorp", "uptake")):
        return None
    rows = list(csv.reader(io.StringIO(text)))
    if len(rows) < 2:
        return [], {"finite_adsorption_or_vext": False, "rows_checked": 0}
    statuses = []
    for row in rows[1:]:
        if not any(field.strip() for field in row):
            continue
        numeric = [field for field in row[1:] if field.strip()]
        statuses.append(bool(numeric) and any(_finite(field) for field in numeric))
    return statuses, {"finite_adsorption_or_vext": any(statuses), "rows_checked": len(statuses)}


def _classify_md(text: str) -> tuple[bool, dict[str, Any], str] | None:
    lower = text.lower()
    if not any(marker in lower for marker in ("loop time", "toteng", "total energy", "simulation completed")):
        return None
    completed = any(marker in lower for marker in ("loop time", "simulation completed", "normal termination"))
    tokens = re.findall(r"(?:toteng|total energy)\s*(?:=)?\s*([-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))",
                        text, flags=re.IGNORECASE)
    if "toteng" in lower and not tokens:
        lines = text.splitlines()
        for index, line in enumerate(lines[:-1]):
            if "toteng" in line.lower():
                tokens.extend(field for field in lines[index + 1].split() if _finite(field))
                break
    finite_energy = any(_finite(token) for token in tokens)
    valid = completed and finite_energy
    return valid, {"finite_energy_and_completion": valid, "completion_marker": completed,
                   "finite_energy": finite_energy}, "missing MD completion marker or finite energy" if not valid else ""


def _classify_dft(text: str) -> tuple[bool, dict[str, Any], str] | None:
    lower = text.lower()
    if not any(marker in lower for marker in ("toten", "total energy", "scf converged", "required accuracy")):
        return None
    completed = any(marker in lower for marker in (
        "general timing and accounting", "normal termination", "scf converged", "reached required accuracy"))
    tokens = re.findall(r"(?:toten|total energy)\s*(?:=)?\s*([-+]?(?:nan|inf|\d+(?:\.\d*)?(?:[eE][-+]?\d+)?))",
                        text, flags=re.IGNORECASE)
    finite_energy = any(_finite(token) for token in tokens)
    valid = completed and finite_energy
    return valid, {"finite_total_energy_and_completion": valid, "completion_marker": completed,
                   "finite_total_energy": finite_energy}, "missing DFT completion marker or finite total energy" if not valid else ""


def _diagnostic_excerpts(node: dict[str, Any], root: Path, inventory: dict[str, Any],
                         samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Collect context without deciding which words mean failure."""
    selected = [sample["path"] for sample in samples]
    selected.extend(item["path"] for item in sorted(
        inventory["files"], key=lambda item: (-item["mtime_ns"], item["path"]))[:5])
    excerpts: list[dict[str, Any]] = []
    remaining = 32 * 1024
    seen = set()
    result = node.get("result") or {}
    for field in ("stdout", "stderr", "run_log_tail", "output_tail"):
        value = result.get(field)
        if isinstance(value, str) and value.strip():
            excerpt = value[-min(8000, remaining):]
            excerpts.append({"path": f"execution_receipt.{field}", "text": excerpt})
            remaining -= len(excerpt)
            if remaining <= 0:
                return excerpts
    for relative in selected:
        if relative in seen:
            continue
        seen.add(relative)
        path = root / relative
        text = _read_text(path, min(8000, remaining))
        printable = sum(character.isprintable() or character in "\n\r\t" for character in text)
        if text and printable / len(text) >= 0.80:
            excerpts.append({"path": relative, "text": text})
            remaining -= len(text)
            if remaining <= 0:
                break
        if len(excerpts) >= 10:
            break
    return excerpts


def _profile_results(node: dict[str, Any], root: Path, inventory: dict[str, Any],
                     samples: list[dict[str, Any]], limits: ValidationLimits) -> dict[str, Any]:
    tool = (node.get("contract") or {}).get("tool", "")
    arguments = (node.get("contract") or {}).get("arguments") or {}
    sample_by_path = {sample["path"]: sample for sample in samples}
    outcomes: list[tuple[str, bool, str]] = []
    aggregate_outcomes: list[tuple[str, bool, str]] | None = None
    method_facts: dict[str, Any] = {}
    bytes_read = 0
    for item in inventory["files"]:
        if bytes_read >= limits.max_bytes:
            inventory["truncated"] = True
            inventory["reasons"] = sorted(set(inventory["reasons"] + ["maximum parse byte count reached"]))
            break
        path = root / item["path"]
        text = _read_text(path, min(256 * 1024, max(0, limits.max_bytes - bytes_read)))
        bytes_read += len(text.encode("utf-8", errors="ignore"))
        classified = None
        if tool == "generate_structure":
            classified = _classify_structure(text)
        elif tool in {"run_pacman_charge", "validate_framework_charges"}:
            classified = _classify_charge(text)
        elif tool == "run_cdft":
            table = _classify_cdft_table(text)
            if table is not None:
                statuses, facts = table
                aggregate_outcomes = [(f"{item['path']}#row-{index + 1}", status,
                                       "non-finite or missing cDFT result")
                                      for index, status in enumerate(statuses)]
                classified = (bool(statuses) and all(statuses), facts,
                              "cDFT table contains invalid rows" if statuses and not all(statuses) else "")
        elif tool.startswith("run_gcmc") or tool in {"run_henry", "run_henry_chain", "run_isotherm_chain"} or "gcmc" in tool:
            classified = _classify_gcmc(text)
        elif tool == "run_md_optimize" or tool.startswith("run_md"):
            classified = _classify_md(text)
        elif tool in {"run_vasp", "run_xtb_optimize"} or "dft" in tool or "vasp" in tool or "xtb" in tool:
            classified = _classify_dft(text)
        if classified is None:
            continue
        valid, facts, reason = classified
        for key, value in facts.items():
            if isinstance(value, bool):
                method_facts[key] = method_facts.get(key, False) or value
            elif isinstance(value, int):
                method_facts[key] = method_facts.get(key, 0) + value
            else:
                method_facts[key] = value
        if item["path"] in sample_by_path:
            sample_by_path[item["path"]]["facts"].update(facts)
        outcomes.append((item["path"], valid, reason))
    if aggregate_outcomes is not None:
        outcomes = aggregate_outcomes
    if not outcomes:
        expected = (node.get("contract") or {}).get("expected_outputs") or []
        selected: list[dict[str, Any]] = []
        for value in expected:
            spec = value if isinstance(value, dict) else {"kind": "file", "path": value}
            target = Path(spec.get("path") or "")
            target = (target if target.is_absolute() else root / target).resolve()
            if target.is_file() and _inside(target, root):
                relative = str(target.relative_to(root))
                selected.extend(item for item in inventory["files"] if item["path"] == relative)
            elif target.is_dir() and _inside(target, root):
                prefix = str(target.relative_to(root))
                pattern = spec.get("pattern", "*")
                selected.extend(item for item in inventory["files"]
                                if _inside((root / item["path"]).resolve(), target)
                                and fnmatch.fnmatch(Path(item["path"]).name, pattern))
        unique = {item["path"]: item for item in selected}
        candidates = list(unique.values()) or inventory["files"]
        outcomes = [(item["path"], item["size"] > 0, "empty output") for item in candidates]
    attempted = len(outcomes)
    succeeded_rows = [path for path, valid, _ in outcomes if valid]
    failed_rows = [{"path": path, "reason": reason or "native validation failed"}
                   for path, valid, reason in outcomes if not valid]
    threshold = float(arguments.get("min_success_ratio", (node.get("contract") or {}).get(
        "min_success_ratio", 0.60)))
    ratio = len(succeeded_rows) / attempted if attempted else 0.0
    verdict = "pass" if succeeded_rows and ratio >= threshold else "recover"
    if inventory["truncated"] or inventory["escaped_paths"]:
        verdict = "invalid"
    successful_artifacts = list(dict.fromkeys(
        str(root / path.split("#", 1)[0]) for path in succeeded_rows))
    return {"counts": {"attempted": attempted, "succeeded": len(succeeded_rows), "failed": len(failed_rows)},
            "success_ratio": ratio, "required_success_ratio": threshold,
            "successful_artifacts": successful_artifacts,
            "failed_items": failed_rows, "method_facts": method_facts, "verdict": verdict}


def _identity(node: dict[str, Any], attempt_id: str, inventory: dict[str, Any]) -> str:
    payload = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "workflow_id": node.get("workflow_id"), "plan_version": node.get("plan_version"),
        "step_id": node.get("step_id") or (node.get("contract") or {}).get("step_id"),
        "attempt_id": attempt_id, "job_ids": node.get("job_ids") or [],
        "contract": node.get("contract") or {},
        "root": inventory["root"], "files": inventory["files"],
        "escaped_paths": inventory["escaped_paths"], "truncated": inventory["truncated"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:32]


def write_validation_evidence(evidence_root: Path, dossier: dict[str, Any], evidence_call_id: str) -> Path:
    evidence_root.mkdir(parents=True, exist_ok=True)
    path = evidence_root / f"{evidence_call_id}.json"
    if path.exists():
        return path
    record = {
        "call_id": evidence_call_id,
        "time": dossier["generated_at"],
        "agent": dossier.get("verification_owner", "workflow-executor"),
        "tool": "inspect_workflow_result",
        "failed": False,
        "params": {"step_id": dossier["step_id"], "path": dossier["inventory"]["root"]},
        "result": dossier,
    }
    temporary = evidence_root / f".{evidence_call_id}.{os.getpid()}.tmp"
    temporary.write_text(json.dumps(record, ensure_ascii=False, indent=2, sort_keys=True))
    os.replace(temporary, path)
    return path


def build_validation_dossier(node: dict[str, Any], attempt_id: str, evidence_root: Path,
                             limits: ValidationLimits | None = None) -> dict[str, Any]:
    limits = limits or ValidationLimits()
    root = _result_root(node)
    inventory = _inventory(root, limits)
    evidence_call_id = _identity(node, attempt_id, inventory)
    evidence_path = Path(evidence_root) / f"{evidence_call_id}.json"
    if evidence_path.exists():
        stored = json.loads(evidence_path.read_text()).get("result")
        if isinstance(stored, dict):
            return stored
    samples = _samples(root, inventory, attempt_id, limits.sample_count)
    profile = _profile_results(node, root, inventory, samples, limits)
    diagnostic_excerpts = _diagnostic_excerpts(node, root, inventory, samples)
    invalid = inventory["escaped_paths"] or inventory["truncated"] or not inventory["file_count"]
    dossier = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "workflow_id": node.get("workflow_id", ""),
        "plan_version": node.get("plan_version"),
        "step_id": node.get("step_id") or (node.get("contract") or {}).get("step_id", ""),
        "attempt_id": attempt_id,
        "job_ids": list(node.get("job_ids") or []),
        "tool": (node.get("contract") or {}).get("tool", ""),
        "contract": node.get("contract") or {},
        "generated_at": time.time(),
        "verification_owner": (node.get("contract") or {}).get("agent", "workflow-executor"),
        "inventory": inventory,
        "samples": samples,
        "counts": profile["counts"],
        "success_ratio": profile["success_ratio"],
        "required_success_ratio": profile["required_success_ratio"],
        "successful_artifacts": profile["successful_artifacts"],
        "failed_items": profile["failed_items"],
        "method_facts": profile["method_facts"],
        "diagnostic_excerpts": diagnostic_excerpts,
        "verdict": "invalid" if invalid else profile["verdict"],
        "requires_agent_review": True,
        "reasons": inventory["reasons"] + (["path escaped approved result root"] if inventory["escaped_paths"] else []),
        "evidence_call_id": evidence_call_id,
        "evidence_path": str(evidence_path),
    }
    write_validation_evidence(Path(evidence_root), dossier, evidence_call_id)
    return dossier
