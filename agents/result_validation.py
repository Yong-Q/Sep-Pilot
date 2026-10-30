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
import stat as stat_module
import time
from typing import Any


VALIDATION_SCHEMA_VERSION = 9


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


def _internal_projection_temporary(candidate: Path, owned_root: Path | None, expected_outputs) -> bool:
    """Only native owner-local publication names, never arbitrary dot files.

    A broad conversation-root output declaration does not make internal display
    publication bytes scientific results. An explicit file or internal-directory
    contract targeting this reserved namespace is rejected explicitly.
    """
    if owned_root is None:
        return False
    owner = Path(owned_root).absolute()
    parent = candidate.parent
    native_name = ((parent == owner / "chain_evidence" and re.fullmatch(r"\.publish-[a-f0-9]{32}", candidate.name))
                   or (parent == owner and (re.fullmatch(r"\.display-sync-[a-f0-9]{32}", candidate.name)
                       or re.fullmatch(r"(?:orchestration_(?:chain|current)|task_manifest)\.json\.[a-z0-9_]{8}", candidate.name))))
    if not native_name:
        return False
    for value in expected_outputs or []:
        spec = value if isinstance(value, dict) else {"path": value}
        name = spec.get("path")
        if not isinstance(name, str):
            continue
        target = Path(name)
        target = target if target.is_absolute() else owner / target
        if target == candidate:
            raise ValueError("declared scientific output targets an internal projection temporary")
        pattern = spec.get("pattern", "*")
        if (target == parent and isinstance(pattern, str) and fnmatch.fnmatch(candidate.name, pattern)
                and (target != owner or pattern != "*")):
            raise ValueError("declared scientific output targets an internal projection temporary")
    try:
        for directory in {owner, parent}:
            info = directory.lstat()
            if not stat_module.S_ISDIR(info.st_mode) or info.st_uid != os.getuid():
                return False
    except FileNotFoundError:
        return False
    try:
        info = candidate.lstat()
    except FileNotFoundError:
        # The native publisher may already have unlinked its exact temporary.
        return True
    return stat_module.S_ISREG(info.st_mode) and info.st_uid == os.getuid()


def _inventory(root: Path, limits: ValidationLimits, *, owned_root: Path | None = None,
               expected_outputs=()) -> dict[str, Any]:
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
            if _internal_projection_temporary(candidate, owned_root, expected_outputs):
                continue
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


def cdft_completion_assessment(node, quality, collected_rows=None):
    """Coverage counts materials once; table integrity is a separate hard gate.

    Only an explicit approved best_effort policy accepts a partial subset.
    A successful file, owner assertion, or generic fraction cannot imply all
    materials or scientific convergence.
    """
    contract=node.get('contract') or {};args=contract.get('arguments') or {}
    policies=[value for value in (contract.get('completion_policy'),args.get('completion_policy')) if value is not None]
    best_effort=bool(policies) and all(value=='best_effort' for value in policies)
    mode='best_effort' if best_effort else 'strict'
    integrity=[]
    if any(value not in {'strict','best_effort'} for value in policies):integrity.append('invalid completion policy')
    ratios=[value for value in (contract.get('min_success_ratio'),args.get('min_success_ratio')) if value is not None]
    if any(type(value) not in (int,float) or not math.isfinite(value) or not 0<=value<=1 for value in ratios):
        integrity.append('invalid minimum success ratio');ratios=[]
    threshold=max(ratios) if ratios else .60
    if not best_effort:threshold=1.0
    names=quality.get('expected_materials') or []
    verified=(quality.get('expected_materials_verified') is True and isinstance(names,list) and bool(names)
              and len(names)==len(set(names)) and all(isinstance(name,str) and name for name in names))
    if not verified:
        names=[];integrity.append('complete original material identities are not verified')
    materials=quality.get('materials') or {};faults={}
    for name in names:
        item=materials.get(name) or {};errors=list(item.get('errors') or [])
        if item.get('output_valid') is not True:errors.append('missing or invalid native density output')
        if item.get('technical_completion') is False:errors.append('native subprocess did not complete successfully')
        if item.get('native_convergence') is False:errors.append('native optimizer reported failure')
        if name not in quality.get('input_materials',[]):errors.append('original material is missing from copied batch inputs')
        if name not in quality.get('run_materials',[]):errors.append('original material is missing from native run directories')
        if errors:faults[name]=errors
    for field in ('input_materials','run_materials'):
        if set(quality.get(field) or [])-set(names):integrity.append(field+' contains undeclared material identities')
    if not quality.get('collected_csv_sha256'):integrity.append('collected CSV was not observed')
    row_map = {}
    if isinstance(collected_rows,list):
        for row in collected_rows:
            if isinstance(row,dict) and isinstance(row.get('MOF'),str):
                row_map.setdefault(row['MOF'],[]).append(row)
    for error in quality.get('collection_errors') or []:
        material,separator,detail=error.partition('/')
        field=detail.split(':',1)[0] if separator else ''
        item=materials.get(material) or {};expected=None;known_field=False
        if field.endswith('_density_mol_L'):
            gas=field[:-len('_density_mol_L')];known_field=True
            expected=item.get('output_facts',{}).get('components',{}).get(gas,{}).get('density_mol_L')
        elif field.endswith('_henry_mol_L_atm'):
            gas=field[:-len('_henry_mol_L_atm')];known_field=True
            expected=item.get('henry_mol_L_atm',{}).get(gas)
        records=row_map.get(material,[])
        # Only a genuinely absent native value paired with an observed blank
        # CSV cell is a declared missing-case value. NaN, fabricated numbers,
        # finite-native disagreement and absent CSV proof remain hard errors.
        missing_blank=(material in faults and known_field and expected is None and len(records)==1
                       and field in records[0] and isinstance(records[0][field],str) and not records[0][field].strip())
        if missing_blank:
            faults[material].append(error)
        else:
            integrity.append(error)
            if material in names:faults.setdefault(material,[]).append(error)
    succeeded=[name for name in names if name not in faults]
    failed=[{'path':name,'reason':'; '.join(dict.fromkeys(faults[name]))} for name in names if name in faults]
    count=len(names);ratio=len(succeeded)/count if count else 0.0
    complete=bool(count) and not failed and not integrity
    status='complete' if complete else 'partial' if succeeded else 'unknown' if not count else 'failed'
    verdict=('recover' if not verified else 'invalid' if integrity else 'pass' if complete or best_effort and succeeded and ratio>=threshold else 'recover')
    return {'schema_version':1,'policy_mode':mode,'explicit_best_effort':best_effort,
            'completion_status':status,'all_materials_valid':complete,
            'counts':{'attempted':count,'succeeded':len(succeeded),'failed':len(failed)},
            'success_ratio':ratio,'required_success_ratio':threshold,'verdict':verdict,
            'successful_materials':succeeded,'failed_items':failed,'integrity_errors':integrity,
            'scientific_convergence':quality.get('scientific_convergence'),
            'note':'Coverage describes observed valid material outputs, not authenticated per-case exit, optimizer/grid convergence, or identical Hamiltonians.'}


def cdft_collected_rows(work_dir, expected_sha):
    """Read the exact bounded CSV tied to native quality, or retain unknown."""
    if not expected_sha:return None
    from .workload_context import _read_regular
    try:
        raw,checksum=_read_regular(Path(work_dir)/'results.csv',limit=8*1024*1024)
        if checksum != expected_sha:return None
        return list(csv.DictReader(io.StringIO(raw.decode('utf-8'))))
    except (OSError,ValueError,UnicodeError):return None


def _profile_results(node: dict[str, Any], root: Path, inventory: dict[str, Any],
                     samples: list[dict[str, Any]], limits: ValidationLimits,
                     owned_root: Path | None = None) -> dict[str, Any]:
    tool = (node.get("contract") or {}).get("tool", "")
    arguments = (node.get("contract") or {}).get("arguments") or {}
    sample_by_path = {sample["path"]: sample for sample in samples}
    if tool in {'run_gcmc_isotherm', 'run_gcmc_batch'}:
        from .gcmc_results import inspect_gcmc_results
        native = inspect_gcmc_results(root, tool, arguments, native_receipt=node.get("result"))
        counts = native['counts']
        ratio = counts['succeeded'] / counts['attempted'] if counts['attempted'] else 0.0
        threshold = float(arguments.get('min_success_ratio', (node.get('contract') or {}).get('min_success_ratio', 0.60)))
        for item in native['cases']:
            data = item.get('data') or {}
            relative = str(Path(data['path']).relative_to(root)) if data.get('path') else ''
            if relative in sample_by_path:
                sample_by_path[relative]['facts'].update(finite_loading_and_completion=item['ok'],
                    completion_marker=item.get('completion_marker', False), finite_loading=item.get('finite_loading', False))
        return {'counts': counts, 'success_ratio': ratio, 'required_success_ratio': threshold,
            'successful_artifacts': native['successful_artifacts'],
            'failed_items': [{'path': item['case_dir'], 'reason': item.get('reason', 'native result invalid')}
                             for item in native['cases'] if not item['ok']],
            'method_facts': {'gcmc_native': native, 'finite_loading_and_completion': bool(counts['succeeded'])},
            'verdict': ('invalid' if native['errors'] or native.get('integrity_errors') or inventory['truncated'] or inventory['escaped_paths']
                        else 'pass' if counts['succeeded'] and ratio >= threshold else 'recover')}
    outcomes: list[tuple[str, bool, str]] = []
    aggregate_outcomes: list[tuple[str, bool, str]] | None = None
    method_facts: dict[str, Any] = {}
    if tool == 'run_md_optimize' or tool.startswith('run_md'):
        from bimem_agent.result_inspector import inspect_md_run
        method_facts['md_native'] = inspect_md_run(
            root, mode=arguments.get('mode', 'single'), max_log_bytes=min(4_000_000, limits.max_bytes))
    if tool == 'run_string_tst':
        from bimem_agent.result_inspector import inspect_string_run
        method_facts['string_native'] = inspect_string_run(root,
            receipt_path=(node.get('result') or {}).get('result_json_path'), gas=arguments.get('gas'),
            approved_walltime=arguments.get('walltime'), approved_walltime_source='workflow_contract',
            max_bytes=min(16_000_000, limits.max_bytes))
    if tool == 'run_cdft' and arguments.get('action', 'pipeline') in {'submit', 'pipeline', 'collect'}:
        from .cdft_result_facts import inspect_cdft_batch_quality
        effective = node.get('effective_arguments') or arguments
        gases = effective.get('gases') or ([effective['gas']] if effective.get('gas') else [])
        method_facts['cdft_native'] = inspect_cdft_batch_quality(
            root, owned_root=owned_root or root, gases=gases,
            source_input_dir=effective.get('input_dir'))
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
    cdft_quality = method_facts.get('cdft_native')
    if tool == 'run_cdft' and cdft_quality is not None:
        collected_rows=cdft_collected_rows(root,cdft_quality.get('collected_csv_sha256'))
        completion=cdft_completion_assessment(node,cdft_quality,collected_rows)
        method_facts['cdft_completion']=completion
        verdict=completion['verdict']
        if inventory['truncated'] or inventory['escaped_paths']:verdict='invalid'
        successful_artifacts=[str(root/'input_dir'/name/'output.dat') for name in completion['successful_materials']]
        # The collected CSV is a native output in its own right. A passing,
        # nonempty, hash-bound table may repair an old managed output path;
        # invalid/unknown/empty collections must never acquire that authority.
        if verdict=='pass' and completion['successful_materials'] and collected_rows:
            successful_artifacts.insert(0,str(root/'results.csv'))
        return {'counts':completion['counts'],'success_ratio':completion['success_ratio'],
                'required_success_ratio':completion['required_success_ratio'],
                'successful_artifacts':successful_artifacts,
                'failed_items':completion['failed_items'],'method_facts':method_facts,'verdict':verdict,
                'completion_status':completion['completion_status'],'all_materials_valid':completion['all_materials_valid']}
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
    if (node.get('contract') or {}).get('tool') == 'run_cdft':
        payload['cdft_completion_policy_version'] = 1
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
    inventory = _inventory(root, limits, owned_root=Path(evidence_root).absolute().parent,
                           expected_outputs=(node.get("contract") or {}).get("expected_outputs") or [])
    evidence_call_id = _identity(node, attempt_id, inventory)
    evidence_path = Path(evidence_root) / f"{evidence_call_id}.json"
    if evidence_path.exists():
        stored = json.loads(evidence_path.read_text()).get("result")
        if isinstance(stored, dict):
            return stored
    samples = _samples(root, inventory, attempt_id, limits.sample_count)
    profile = _profile_results(node, root, inventory, samples, limits, owned_root=Path(evidence_root).parent)
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
    if 'completion_status' in profile:
        dossier.update(completion_status=profile['completion_status'],all_materials_valid=profile['all_materials_valid'])
        dossier['reasons'] += profile['method_facts']['cdft_completion']['integrity_errors']
    write_validation_evidence(Path(evidence_root), dossier, evidence_call_id)
    return dossier
