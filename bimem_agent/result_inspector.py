from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shlex
import stat
from typing import Any, Dict, List

from .run_store import load_run


def _string_duration(value):
    """Parse explicit [days-]HH:MM:SS; unsupported forms remain unknown."""
    if not isinstance(value, str): return None
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):([0-5]\d):([0-5]\d)', value.strip())
    if not match: return None
    days, hours, minutes, seconds = match.groups()
    return int(days or 0)*86400 + int(hours)*3600 + int(minutes)*60 + int(seconds)


def _string_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0: return None
    return None if isinstance(value, float) and not math.isfinite(value) else value


def _read_string_artifact(root, value, limit):
    """Anchor every path component with NOFOLLOW, including nested inputs."""
    path = Path(value)
    relative = path.relative_to(root) if path.is_absolute() else path
    if not relative.parts or any(part in {'', '.', '..'} for part in relative.parts):
        raise ValueError('invalid String artifact path')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts[:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory); directory = child
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError('String artifact is not a bounded regular file')
        raw = stream.read(limit + 1)
    if len(raw) > limit: raise ValueError('String artifact exceeds read bound')
    return raw, {'path': str(root/relative), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def inspect_string_run(work_dir, *, receipt_path=None, gas=None, approved_walltime=None,
                       approved_walltime_source=None, scheduler=None, max_bytes=16_000_000):
    """Read one owned native String attempt; never infer physics from shape."""
    root = Path(os.path.abspath(Path(work_dir).expanduser()))
    result = {'ok': False, 'result_dir': str(root), 'adapter_reported_success': None,
        'energy_unit': None, 'engine_reported_convergence': None, 'rows': [], 'artifacts': {},
        'column_semantics': {'status': 'unknown', 'columns': None, 'provenance': {
            'declared_by_adapter': False, 'source_verified': False, 'binary_source_binding': False}},
        'timing': {'approved_walltime_seconds': _string_duration(approved_walltime),
            'approved_walltime_source': approved_walltime_source if _string_duration(approved_walltime) is not None else None,
            'submitted_walltime_seconds': None, 'scheduler_elapsed_seconds': None,
            'adapter_native_wall_seconds': None, 'native_process_cpu_seconds': None},
        'errors': [], 'warnings': [], 'requires_scientific_review': True}
    if root.is_symlink() or isinstance(max_bytes, bool) or not isinstance(max_bytes, int) or max_bytes < 1:
        result['errors'].append('unsafe String result root or invalid read bound'); return result
    budget = max_bytes
    def read(value):
        nonlocal budget
        raw, fact = _read_string_artifact(root, value, budget)
        budget -= len(raw)
        result['artifacts'][fact['path']] = fact
        return raw, fact
    try:
        if receipt_path:
            selected = Path(receipt_path)
        elif gas and isinstance(gas, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_+\-]*', gas):
            selected = root/f'string_input_{gas}'/'string_results.json'
        else:
            candidates = [root/'string_results.json'] if (root/'string_results.json').exists() else []
            if not candidates:
                for candidate in root.glob('string_input_*/string_results.json'):
                    candidates.append(candidate)
                    if len(candidates) > 32: break
            if len(candidates) != 1: raise ValueError('String result receipt is missing or ambiguous')
            selected = candidates[0]
        raw, receipt_fact = read(selected)
        receipt = json.loads(raw)
        if (not isinstance(receipt, dict) or not isinstance(receipt.get('results'), list)
                or not 1 <= len(receipt['results']) <= 256 or not all(isinstance(row, dict) for row in receipt['results'])):
            raise ValueError('String receipt has invalid or unbounded native rows')
        result['receipt'] = receipt_fact
        result['adapter_reported_success'] = receipt.get('success') if isinstance(receipt.get('success'), bool) else None
        protocol = receipt.get('output_protocol') or {}
        columns = ['fractional_a', 'fractional_b', 'fractional_c', 'orientation_alpha',
                   'orientation_beta', 'orientation_gamma', 'potential']
        if (isinstance(protocol, dict) and protocol.get('id') == 'tutrast_string_seven_columns_v1'
                and protocol.get('column_count') == 7 and protocol.get('columns') == columns):
            result['column_semantics'].update(status='declared_by_adapter', columns=columns)
            result['column_semantics']['provenance']['declared_by_adapter'] = True
        fingerprint = receipt.get('executable_sha256')
        result['executable_sha256_at_run'] = fingerprint if isinstance(fingerprint, str) and re.fullmatch(r'[0-9a-f]{64}', fingerprint) else None
        manifest_raw, manifest_fact = read(receipt.get('manifest_path') or '')
        if receipt.get('manifest_sha256') != manifest_fact['sha256']:
            raise ValueError('String manifest hash does not match its receipt')
        manifest_rows = list(csv.DictReader(manifest_raw.decode('utf-8').splitlines()))
        manifest = {(row.get('structure'), str(row.get('direction'))): row.get('input_dat') for row in manifest_rows}
        if len(manifest) != len(manifest_rows) or len(manifest) != len(receipt['results']):
            raise ValueError('String manifest and native receipt rows do not agree')
        seen, native_flags, native_wall = set(), [], []
        for row in receipt['results']:
            identity = (row.get('structure'), str(row.get('direction')))
            source = Path(row.get('input_path') or '')
            if identity in seen or identity not in manifest or str(source) != manifest[identity]:
                raise ValueError('String native row does not match its manifest input')
            seen.add(identity)
            if (Path(row.get('output_path') or '') != source.parent/'string_path.dat'
                    or Path(row.get('run_log_path') or '') != source.parent/'run.log'):
                raise ValueError('String row does not name its own canonical output and log')
            input_raw, input_fact = read(source)
            output_raw, output_fact = read(row['output_path'])
            log_raw, log_fact = read(row['run_log_path'])
            if any(row.get(key) != fact['sha256'] for key, fact in (
                    ('input_sha256', input_fact), ('output_sha256', output_fact), ('run_log_sha256', log_fact))):
                raise ValueError('String native artifact hash does not match its receipt')
            points = [line.split() for line in output_raw.decode('utf-8').splitlines() if line.strip()]
            if len(points) < 2 or any(len(point) != 7 or not all(math.isfinite(float(v)) for v in point) for point in points):
                raise ValueError('String path must contain finite seven-column native points')
            if row.get('point_count') != len(points) or (row.get('expected_point_count') is not None and row['expected_point_count'] != len(points)):
                raise ValueError('String native point counts do not match the receipt')
            markers = re.findall(r'\binfo:\s*([01])\s+(\d+)\s+(\d+)', log_raw.decode('utf-8', errors='replace'))
            flag = markers[-1] if markers else None
            native = {'converged': flag[0] == '1' if flag else None,
                'iterations': int(flag[1]) if flag else None, 'expanded_atoms': int(flag[2]) if flag else None}
            if isinstance(row.get('native_info'), dict) and row['native_info'] != native:
                raise ValueError('String native convergence receipt disagrees with its actual log')
            native_flags.append(native['converged'])
            input_text = input_raw.decode('utf-8', errors='replace')
            declared_points = re.search(r'(?m)^#_of_points[^\n]*\n\s*(\d+)\b', input_text)
            if declared_points and int(declared_points[1]) != len(points):
                raise ValueError('String native point count disagrees with its actual input')
            budget_match = re.search(r'(?m)^([^\n]*\bRunning_steps\b[^\n]*)\n([^\n]+)', input_text)
            iterations = None
            if budget_match:
                header, values = budget_match[1].split(), budget_match[2].split()
                index = header.index('Running_steps')
                if index < len(values) and values[index].isdigit(): iterations = int(values[index])
            elapsed = _string_number(row.get('adapter_native_wall_elapsed_seconds'))
            native_wall.append(elapsed)
            result['rows'].append({'structure': row.get('structure'), 'direction': row.get('direction'),
                'point_count': len(points), 'column_count': 7, 'native_info': native,
                'declared_iteration_budget': iterations, 'output_path': output_fact['path'],
                'adapter_native_wall_elapsed_seconds': elapsed})
        result['engine_reported_convergence'] = False if False in native_flags else True if all(flag is True for flag in native_flags) else None
        if native_wall and all(value is not None for value in native_wall):
            result['timing']['adapter_native_wall_seconds'] = sum(native_wall)
        if result['engine_reported_convergence'] is not True:
            result['warnings'].append('Native convergence is false or unknown; finite paths do not establish converged transport physics')
    except (OSError, ValueError, TypeError, KeyError, UnicodeError):
        result['errors'].append('String receipt/artifact is missing, unsafe, inconsistent or exceeds the read bound')
    try:
        raw, fact = read('submit.sh')
        matches = re.findall(r'(?m)^#SBATCH\s+(?:-t\s+|--time(?:=|\s+))(\S+)', raw.decode('utf-8', errors='replace'))
        if len(matches) == 1:
            result['timing']['submitted_walltime_seconds'] = _string_duration(matches[0])
            result['timing']['submitted_walltime_source'] = fact
    except (OSError, ValueError, TypeError):
        pass
    if isinstance(scheduler, dict):
        elapsed = _string_number(scheduler.get('elapsed_seconds'))
        if elapsed is None: elapsed = _string_duration(scheduler.get('elapsed'))
        result['timing']['scheduler_elapsed_seconds'] = elapsed
        result['timing']['scheduler_elapsed_source'] = 'owned_scheduler_receipt' if elapsed is not None else None
    result['ok'] = not result['errors']
    return result


def _read_local_md_file(root: Path, name: str, limit: int):
    """Exact regular file, anchored to this run; never follow a symlink."""
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    finally:
        os.close(directory)
    with os.fdopen(fd, 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('MD artifact must be a regular file')
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('MD artifact exceeds bounded read')
    return raw, {'path': str(root / name), 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def inspect_md_run(work_dir: str | Path, mode: str | None = None, max_log_bytes: int = 4_000_000) -> Dict[str, Any]:
    """Read standard MD artifacts in one caller-owned run, not its workspace.

    Successful parsing and adapter success are independent of scientific
    convergence. The caller is responsible for authenticating run ownership.
    """
    from agents.md_result_facts import parse_md_result_facts
    root = Path(os.path.abspath(Path(work_dir).expanduser()))
    result = {'ok': False, 'result_dir': str(root), 'mode': mode, 'mode_source': 'workflow_contract' if mode else None,
        'adapter_reported_success': None, 'solver_log': {}, 'line_refs': {}, 'artifacts': {},
        'thermo_scope': 'last minimization stage only; never label it final NPT/dynamics thermo',
        'errors': [], 'warnings': [], 'requires_scientific_review': True}
    if isinstance(max_log_bytes, bool) or not isinstance(max_log_bytes, int) or max_log_bytes < 1:
        result['errors'].append('invalid MD log read bound'); return result
    if root.is_symlink():
        result['errors'].append('MD run directory cannot be a symlink'); return result
    recorded_mode = None
    try:
        raw, artifact = _read_local_md_file(root, 'md_result.json', 256 * 1024)
        data = json.loads(raw)
        if not isinstance(data, dict): raise ValueError('MD result must be a JSON object')
        result['artifacts']['md_result.json'] = artifact
        if isinstance(data.get('success'), bool): result['adapter_reported_success'] = data['success']
        energy = data.get('energy')
        if isinstance(energy, (int, float)) and not isinstance(energy, bool) and math.isfinite(energy):
            result['adapter_energy'] = energy
        if data.get('mode') in {'single', 'md'}:
            recorded_mode = data['mode']
            if not mode: result.update(mode=recorded_mode, mode_source='md_result.json')
    except (OSError, ValueError, UnicodeError):
        result['errors'].append('standard MD result JSON unavailable or invalid')
    # Older MD result JSON omitted mode. The exact owned submission receipt is
    # a bounded fallback; parse its JSON argument without executing any shell.
    if recorded_mode is None:
        try:
            raw, artifact = _read_local_md_file(root, 'submit.sh', 256 * 1024)
            for line_number, line in enumerate(raw.decode('utf-8').splitlines(), 1):
                if line.lstrip().startswith('#') or '--parameters' not in line: continue
                words = shlex.split(line)
                arguments = json.loads(words[words.index('--parameters') + 1])
                if isinstance(arguments, dict) and arguments.get('mode') in {'single', 'md'}:
                    recorded_mode = arguments['mode']
                    result['artifacts']['submit.sh'] = artifact
                    result['line_refs']['mode'] = {**artifact, 'line': line_number}
                    if not mode: result.update(mode=recorded_mode, mode_source='submit.sh parameters')
                    break
        except (OSError, ValueError, IndexError, UnicodeError):
            pass
    if mode and recorded_mode and mode != recorded_mode:
        result['warnings'].append('workflow mode differs from recorded MD artifact mode')
    try:
        raw, artifact = _read_local_md_file(root, 'solver.log', max_log_bytes)
        facts = parse_md_result_facts(raw.decode('utf-8', errors='replace'))
        facts.update(source_path=artifact['path'], source_sha256=artifact['sha256'], source_bytes=artifact['bytes'])
        result['solver_log'] = facts
        result['artifacts']['solver.log'] = artifact
        for key, line in {'final_minimization': facts['final_minimization']['line_start'],
                          'final_minimization_thermo': facts['final_minimization_thermo']['last_row_line']}.items():
            if line is not None: result['line_refs'][key] = {**artifact, 'line': line}
        result['errors'].extend(facts['errors'])
        result['warnings'].extend(facts['warnings'])
    except (OSError, ValueError):
        result['errors'].append('exact solver.log unavailable, unsafe or exceeds read bound')
    result['ok'] = not result['errors']
    return result


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
    root = Path(record.get('run_dir') or result_path.parent)
    for key, path in (('result', result_path), ('spec', spec_path)):
        if path.exists():
            if root.is_symlink() or path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                payload.setdefault('inspection_errors', []).append(key + ' path escapes the selected run or is a symlink')
            else:
                payload[key] = inspect_path(str(path))
    if (root / 'md_result.json').exists():
        payload['md_native'] = inspect_md_run(root)
    if result_path.name == 'string_results.json' or next(root.glob('string_input_*/string_results.json'), None) is not None:
        payload['string_native'] = inspect_string_run(root,
            receipt_path=result_path if result_path.name == 'string_results.json' else None)
    return payload
