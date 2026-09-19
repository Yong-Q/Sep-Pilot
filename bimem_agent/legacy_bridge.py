from __future__ import annotations

import json
import re
import shlex
import shutil
import subprocess
import textwrap
from datetime import datetime
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, List

from .paths import (
    AGENT_ENV_PYTHON,
    LEGACY_ROOT,
    PROJECT_ROOT,
    ensure_project_path,
    get_config_path,
    get_runtime_section,
    get_scheduler_defaults,
)


def _normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    return value


def _all_cifs_already_charged(cif_dir: str) -> bool:
    path = Path(cif_dir)
    if not path.is_dir():
        return False
    cifs = sorted(path.glob("*.cif"))
    return bool(cifs) and all(item.name.endswith("_pacman.cif") for item in cifs)


def _split_charge_inputs(cif_dir: str) -> tuple[list[Path], list[Path]]:
    path = Path(cif_dir)
    if not path.is_dir():
        return [], []
    raw = []
    charged = []
    for cif in sorted(path.glob("*.cif")):
        if cif.name.endswith("_pacman.cif"):
            charged.append(cif)
        else:
            raw.append(cif)
    return raw, charged


def _preferred_charged_files(cif_dir: str) -> list[Path]:
    _, charged = _split_charge_inputs(cif_dir)
    best: Dict[str, Path] = {}
    for path in charged:
        stem = path.name[:-4]
        base = stem.split("_pacman", 1)[0]
        current = best.get(base)
        if current is None or path.name.count("_pacman") < current.name.count("_pacman"):
            best[base] = path
    return sorted(best.values())


def _timestamped_name(prefix: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%m%d%H%M%S%f')}"


def _shell_join(parts: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def _remote_slurm_config() -> Dict[str, Any]:
    return get_runtime_section("remote_slurm")


def _remote_slurm_work_root(plan_name: str) -> Path:
    root = _remote_slurm_config().get("work_root", "tmp/remote_slurm")
    return (PROJECT_ROOT / root / plan_name).resolve()


def _remote_slurm_registry_path() -> Path:
    return _remote_slurm_work_root("registry").parent / "jobs.json"


def _load_remote_job_registry() -> Dict[str, Any]:
    path = _remote_slurm_registry_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _register_remote_job(job_id: str, metadata: Dict[str, Any]) -> None:
    path = _remote_slurm_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    registry = _load_remote_job_registry()
    registry[str(job_id)] = metadata
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")


def helper_call(function_name: str, **kwargs: Any) -> Any:
    ensure_project_path()
    import helper  # type: ignore

    func = getattr(helper, function_name)
    return func(**kwargs)


def resolve_cif_path(cif: str) -> str:
    ensure_project_path()
    import helper  # type: ignore

    path = helper._resolve_cif(cif)
    if path is None:
        raise FileNotFoundError(f"Could not resolve CIF: {cif}")
    return path


def run_pore_analysis(**kwargs: Any) -> Dict[str, Any]:
    ensure_project_path()
    from tools import zeopp  # type: ignore

    if "cif_path" in kwargs:
        result = zeopp.run_single(**kwargs)
        return _normalize(asdict(result))
    result = zeopp.run_batch(**kwargs)
    return {"results": _normalize(result)}


def build_pore_analysis_command(params: Dict[str, Any]) -> Dict[str, Any]:
    runner = PROJECT_ROOT / "scripts" / "run_pore_analysis.py"
    scheduler_defaults = get_scheduler_defaults().get("pore_analysis", {})
    output_root = params.get("output_root")
    if output_root is None:
        label = Path(params.get("cif_path") or params.get("cif_dir")).stem
        output_root = str((PROJECT_ROOT / "tmp" / f"pore_analysis_{label}").resolve())

    command = ["python3", str(runner)]
    if "cif_path" in params:
        command.extend(["--cif-path", params["cif_path"]])
    else:
        command.extend(["--cif-dir", params["cif_dir"]])
    command.extend(["--probe-radius", str(params.get("probe_radius", 1.525))])
    command.extend(["--n-samples", str(params.get("n_samples", 5000))])
    if "cif_dir" in params:
        output_csv = params.get("output_csv", f"{output_root}/batch_results.csv")
        command.extend(["--n-threads", str(params.get("n_threads", 4))])
        command.extend(["--output-csv", output_csv])
    else:
        output_csv = None

    return {
        "plan_name": "pore-analysis",
        "cwd": None,
        "command": command,
        "scheduler": params.get("scheduler", "local"),
        "queue": params.get("queue", scheduler_defaults.get("queue", "tiny")),
        "ppn": int(params.get("ppn", scheduler_defaults.get("ppn", 1))),
        "walltime": params.get("walltime", scheduler_defaults.get("walltime", "00:30:00")),
        "job_name": params.get("job_name", scheduler_defaults.get("job_name", "zeopp_batch")),
        "output_root": output_root,
        "output_csv": output_csv,
        "submit_script": f"{output_root}/submit_pore_analysis.pbs",
        "logs_dir": f"{output_root}/logs",
        "note": "Zeo++ can run locally or through a tiny PBS wrapper for style consistency.",
    }


def run_material_props(**kwargs: Any) -> Dict[str, Any]:
    ensure_project_path()
    from tools import mof_features  # type: ignore

    if "cif_path" in kwargs:
        result = mof_features.extract_features(kwargs["cif_path"])
        return _normalize(asdict(result))
    results = mof_features.extract_features_batch(
        cif_dir=kwargs["cif_dir"],
        output_csv=kwargs.get("output_csv"),
        n_threads=kwargs.get("n_threads", 8),
    )
    return {"results": _normalize(results)}


def run_interaction(**kwargs: Any) -> Dict[str, Any]:
    ensure_project_path()
    from analysis.interaction import calc_binding_energy  # type: ignore

    result = calc_binding_energy(**kwargs)
    return _normalize(asdict(result))


def run_md_optimize(**kwargs: Any) -> Any:
    ensure_project_path()
    from tools import lammps_optimize  # type: ignore

    mode = kwargs.pop("mode", "single")
    output_mode = kwargs.pop("output_mode", None)
    submit_pbs = kwargs.pop("submit_pbs", False)
    scheduler_queue = kwargs.pop("queue", "normal")
    scheduler_node = kwargs.pop("node", None)
    scheduler_walltime = kwargs.pop("walltime", "72:00:00")
    scheduler_n_procs = kwargs.get("n_procs", 96)

    if output_mode is None:
        output_mode = "native" if kwargs.get("skip_convert") else "interop"
    if output_mode not in {"native", "interop"}:
        raise ValueError(f"Unsupported md-optimize output_mode: {output_mode}")
    kwargs.setdefault("skip_convert", output_mode == "native")

    def _submit_pbs(pbs_path: Path) -> Dict[str, Any]:
        proc = subprocess.run(
            ["qsub", str(pbs_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "pbs_script": str(pbs_path),
            "submitted": proc.returncode == 0,
            "job_id": proc.stdout.strip() if proc.returncode == 0 else None,
            "stderr": proc.stderr.strip(),
            "stdout": proc.stdout.strip(),
            "returncode": proc.returncode,
        }

    def _fallback_to_pbs(local_result: Any, is_md: bool) -> Dict[str, Any]:
        cif_path = kwargs["cif_path"]
        output_dir = kwargs.get("output_dir")
        if is_md:
            pbs_dir = lammps_optimize.generate_md_pbs(
                cif_path=cif_path,
                output_dir=output_dir,
                cutoff=kwargs.get("cutoff", 6.0),
                force_field=kwargs.get("force_field", "UFF"),
                temperature=kwargs.get("temperature", 298.0),
                pressure=kwargs.get("pressure", 1.0),
                npt_ps=kwargs.get("npt_ps", 700.0),
                timestep=kwargs.get("timestep", 1.0),
                n_procs=scheduler_n_procs,
                queue=scheduler_queue,
                node=scheduler_node,
                walltime=scheduler_walltime,
                skip_convert=kwargs.get("skip_convert", False),
            )
            pbs_path = Path(pbs_dir) / f"md_{Path(cif_path).stem}.pbs"
        else:
            pbs_dir = lammps_optimize.generate_pbs_scripts(
                cif_dir=kwargs["cif_dir"],
                output_dir=output_dir,
                cutoff=kwargs.get("cutoff", 6.0),
                force_field=kwargs.get("force_field", "UFF"),
                n_iter=kwargs.get("n_iter", 10),
                n_procs=scheduler_n_procs,
                queue=scheduler_queue,
                skip_convert=kwargs.get("skip_convert", False),
            )
            pbs_path = Path(pbs_dir)

        payload = {
            "local_result": _normalize(asdict(local_result)) if is_dataclass(local_result) else _normalize(local_result),
            "pbs_generated": True,
            "pbs_dir": str(pbs_dir),
            "output_mode": output_mode,
            "artifact_format": "lammps-data" if kwargs.get("skip_convert", False) else "cif",
        }
        if pbs_path.is_file():
            payload.update(_submit_pbs(pbs_path) if submit_pbs else {"pbs_script": str(pbs_path), "submitted": False})
        return payload

    if mode == "single":
        result = lammps_optimize.run_single(**kwargs)
        if not result.success and "libfftw3.so.3" in (result.error or ""):
            return _fallback_to_pbs(result, is_md=True)
        payload = _normalize(asdict(result))
        payload["output_mode"] = output_mode
        payload["artifact_format"] = "lammps-data" if kwargs.get("skip_convert", False) else "cif"
        return payload
    if mode == "md":
        result = lammps_optimize.run_md(**kwargs)
        if not result.success and "libfftw3.so.3" in (result.error or ""):
            return _fallback_to_pbs(result, is_md=True)
        payload = _normalize(asdict(result))
        payload["output_mode"] = output_mode
        payload["artifact_format"] = "lammps-data" if kwargs.get("skip_convert", False) else "cif"
        return payload
    if mode == "batch":
        payload = _normalize(lammps_optimize.run_batch(**kwargs))
        return {
            "results": payload,
            "output_mode": output_mode,
            "artifact_format": "lammps-data" if kwargs.get("skip_convert", False) else "cif",
        }
    raise ValueError(f"Unsupported md-optimize mode: {mode}")


def run_literature_rag(query: str, n_results: int = 5, query_type: str = "auto") -> Dict[str, Any]:
    if not AGENT_ENV_PYTHON.exists():
        raise FileNotFoundError(f"Agent environment python not found: {AGENT_ENV_PYTHON}")

    command = [
        str(AGENT_ENV_PYTHON),
        str(PROJECT_ROOT / "scripts" / "run_rag_query.py"),
        "--query",
        query,
        "--n-results",
        str(n_results),
        "--query-type",
        query_type,
    ]
    result = subprocess.run(
        command,
        cwd=str(LEGACY_ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"RAG query failed with code {result.returncode}: {stderr}")
    try:
        return json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"Could not parse RAG query output: {exc}") from exc


def run_jobs_workflow(action: str, **kwargs: Any) -> Dict[str, Any]:
    ensure_project_path()
    from utils import pbs  # type: ignore

    if action == "check":
        job_id = str(kwargs["job_id"])
        scheduler = kwargs.get("scheduler")
        if scheduler in {"slurm", "remote_slurm"} or job_id.isdigit():
            return _normalize(check_remote_slurm_job(job_id, host=kwargs.get("host")))
        return _normalize(pbs.check_job_status(job_id))

    if action == "submit-single":
        cif_path = resolve_cif_path(kwargs.pop("cif"))
        job_id, job_name = pbs.submit_gcmc(cif=cif_path, **kwargs)
        return {"job_id": job_id, "job_name": job_name, "mode": "submit-single"}

    if action == "submit-batch-gcmc":
        job_id, job_name = pbs.submit_raspa_gcmc_batch(**kwargs)
        return {"job_id": job_id, "job_name": job_name, "mode": action}

    if action == "submit-batch-henry":
        job_id, job_name = pbs.submit_raspa_henry_batch(**kwargs)
        return {"job_id": job_id, "job_name": job_name, "mode": action}

    if action == "submit-batch-isotherm":
        job_id, job_name = pbs.submit_raspa_isotherm_batch(**kwargs)
        return {"job_id": job_id, "job_name": job_name, "mode": action}

    if action == "submit-charge":
        scheduler = kwargs.get("scheduler", get_scheduler_defaults().get("charge", {}).get("mode", "remote_slurm"))
        if scheduler in {"slurm", "remote_slurm"}:
            plan = build_charge_command(kwargs)
            payload = execute_charge_plan(plan)
            if payload["returncode"] != 0:
                raise RuntimeError(payload.get("stderr") or payload.get("stdout") or "Remote Slurm charge submission failed")
            return {
                "job_id": payload.get("job_id"),
                "job_name": plan["job_name"],
                "mode": action,
                "scheduler": "remote_slurm",
                "remote_host": payload.get("remote_host"),
                "submit_script": payload.get("submit_script"),
            }
        job_id, job_name = pbs.submit_pacman(**kwargs)
        return {"job_id": job_id, "job_name": job_name, "mode": action, "scheduler": "pbs"}

    raise ValueError(f"Unsupported jobs action: {action}")


def _extract_prefixed_paths(text: str) -> Dict[str, str]:
    patterns = {
        "csv_path": r"^CSV:\s+(.*)$",
        "json_path": r"^JSON:\s+(.*)$",
        "plot_path": r"^图表:\s+(.*)$",
        "summary_csv": r"^汇总CSV:\s+(.*)$",
        "comparison_plot": r"^对比图:\s+(.*)$",
        "output_dir": r"^输出目录\s*:\s+(.*)$",
    }
    out: Dict[str, str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match:
            out[key] = match.group(1).strip()
    return out


def run_adsorption_workflow(action: str, **kwargs: Any) -> Dict[str, Any]:
    if action == "isotherm":
        text = helper_call("run_isotherm", **kwargs)
        payload = {"report": text, **_extract_prefixed_paths(text)}
        return payload
    if action == "batch":
        text = helper_call("run_batch", **kwargs)
        payload = {"report": text, **_extract_prefixed_paths(text)}
        return payload
    if action == "henry":
        text = helper_call("run_henry", **kwargs)
        payload = {"report": text, **_extract_prefixed_paths(text)}
        return payload
    raise ValueError(f"Unsupported adsorption action: {action}")


def run_vasp_workflow(action: str, **kwargs: Any) -> Dict[str, Any]:
    ensure_project_path()
    from tools import vasp  # type: ignore

    if action == "setup":
        work_dir = vasp.setup_calc(
            cif_path=kwargs["cif_path"],
            calc_type=kwargs.get("calc_type", "static"),
            work_dir=kwargs.get("work_dir"),
            pp_path=kwargs.get("pp_path"),
            kpoints=kwargs.get("kpoints", "gamma"),
        )
        return {"work_dir": work_dir, "mode": action}

    if action == "submit":
        job_id = vasp.submit_vasp(
            work_dir=kwargs["work_dir"],
            queue=kwargs.get("queue", "batch"),
            walltime=kwargs.get("walltime", "04:00:00"),
            ncpus=kwargs.get("ncpus", 10),
            node=kwargs.get("node"),
            job_name=kwargs.get("job_name"),
        )
        return {"job_id": job_id, "work_dir": kwargs["work_dir"], "mode": action}

    if action == "parse":
        result = vasp.parse_results(kwargs["work_dir"])
        payload: Dict[str, Any] = {"result": _normalize(asdict(result)), "mode": action}
        if kwargs.get("csv"):
            payload["csv"] = vasp.export_csv([result], kwargs["csv"])
        return payload

    if action == "setup-batch":
        work_dirs = vasp.setup_batch(
            cif_dir=kwargs["cif_dir"],
            calc_type=kwargs.get("calc_type", "static"),
            output_dir=kwargs.get("output_dir"),
            pp_path=kwargs.get("pp_path"),
            kpoints=kwargs.get("kpoints", "gamma"),
        )
        return {"work_dirs": _normalize(work_dirs), "mode": action}

    if action == "run":
        result = vasp.run_vasp(
            work_dir=kwargs["work_dir"],
            vasp_bin=kwargs.get("vasp_bin"),
            ncpus=kwargs.get("ncpus", 1),
            source_env=kwargs.get("source_env", True),
        )
        return {"result": _normalize(asdict(result)), "mode": action}

    raise ValueError(f"Unsupported vasp action: {action}")


def build_charge_command(params: Dict[str, Any]) -> Dict[str, Any]:
    charge_type = params.get("charge_type", "DDEC6")
    cif_dir = params["cif_dir"]
    default_output_dir = f"{LEGACY_ROOT}/cifs/{params['task_name']}" if params.get("task_name") else f"{LEGACY_ROOT}/cifs/pacman_manual"
    output_dir = params.get("output_dir", default_output_dir)
    pacman_dir = get_config_path("software", "pacman_dir")
    pacman_python = get_config_path("software", "pacman_python")
    scheduler_defaults = get_scheduler_defaults().get("charge", {})
    scheduler = params.get("scheduler", scheduler_defaults.get("mode", "remote_slurm"))
    remote_cfg = _remote_slurm_config()
    job_name = params.get("job_name", _timestamped_name("pacman"))
    submit_script = _remote_slurm_work_root("charge") / f"{job_name}.sbatch"
    stage_dir = (PROJECT_ROOT / "tmp" / "charge_stage" / job_name).resolve()
    raw_cifs, charged_cifs = _split_charge_inputs(cif_dir)
    skip_execution = bool(charged_cifs) and not raw_cifs
    command = [
        str(pacman_python),
        "pmcharge.py",
        str(stage_dir),
        "--charge_type",
        charge_type,
    ]
    if "digits" in params:
        command.extend(["--digits", str(params["digits"])])
    return {
        "plan_name": "charge",
        "cwd": str(pacman_dir),
        "command": command,
        "job_name": job_name,
        "source_cif_dir": cif_dir,
        "run_cif_dir": str(stage_dir),
        "raw_input_files": [str(path) for path in raw_cifs],
        "existing_charged_files": [str(path) for path in charged_cifs],
        "output_dir": output_dir,
        "scheduler": scheduler,
        "remote_host": params.get("remote_host", remote_cfg.get("host", "gpu2")),
        "partition": params.get("partition", scheduler_defaults.get("partition", remote_cfg.get("partition", "gpu"))),
        "nodelist": params.get("nodelist", scheduler_defaults.get("nodelist", remote_cfg.get("nodelist"))),
        "gres": params.get("gres", scheduler_defaults.get("gres", remote_cfg.get("gres", "gpu:1"))),
        "cpus_per_task": int(params.get("cpus_per_task", scheduler_defaults.get("cpus_per_task", remote_cfg.get("cpus_per_task", 4)))),
        "walltime": params.get("walltime", scheduler_defaults.get("walltime", remote_cfg.get("walltime", "12:00:00"))),
        "submit_script": str(submit_script),
        "logs_dir": str(submit_script.parent / "logs"),
        "status_file": str(submit_script.parent / "status" / f"{job_name}.status"),
        "skip_execution": skip_execution,
        "postprocess_note": (
            f"Input CIFs already appear charged; skip PACMAN and copy *_pacman.cif files into {output_dir}"
            if skip_execution
            else f"Stage raw CIFs into {stage_dir}, run PACMAN there, then copy generated *_pacman.cif files into {output_dir}"
        ),
    }


def _render_charge_remote_sbatch(plan: Dict[str, Any]) -> str:
    command = _shell_join([str(item) for item in plan["command"]])
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {plan['job_name']}",
        f"#SBATCH -p {plan['partition']}",
    ]
    if plan.get("nodelist"):
        lines.append(f"#SBATCH --nodelist={plan['nodelist']}")
    lines.append(f"#SBATCH --cpus-per-task={int(plan['cpus_per_task'])}")
    if plan.get("gres"):
        lines.append(f"#SBATCH --gres={plan['gres']}")
    lines.extend(
        [
            f"#SBATCH -t {plan['walltime']}",
            f"#SBATCH -o {plan['logs_dir']}/%x_%j.out",
            f"#SBATCH -e {plan['logs_dir']}/%x_%j.err",
            "",
            "set -euo pipefail",
            f"STATUS_FILE={shlex.quote(plan['status_file'])}",
            "mkdir -p \"$(dirname \"$STATUS_FILE\")\"",
            "echo RUNNING > \"$STATUS_FILE\"",
            "trap 'echo FAILED > \"$STATUS_FILE\"' ERR",
            "",
            f"mkdir -p {shlex.quote(plan['logs_dir'])}",
            f"mkdir -p {shlex.quote(plan['output_dir'])}",
            f"mkdir -p {shlex.quote(plan['run_cif_dir'])}",
            "",
            f"cd {shlex.quote(plan['cwd'])}",
            command,
            "",
            "shopt -s nullglob",
            f"for cif in {shlex.quote(plan['run_cif_dir'])}/*_pacman.cif; do",
            f"    cp \"$cif\" {shlex.quote(plan['output_dir'])}/",
            "done",
            "shopt -u nullglob",
            "echo COMPLETED > \"$STATUS_FILE\"",
            "",
        ]
    )
    return "\n".join(lines)


def _submit_remote_slurm_script(script_path: Path, host: str) -> Dict[str, Any]:
    cfg = _remote_slurm_config()
    submit_command = cfg.get("submit_command", "sbatch")
    proc = subprocess.run(
        ["ssh", host, submit_command, str(script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    stdout = proc.stdout.strip()
    stderr = proc.stderr.strip()
    job_id = None
    match = re.search(r"Submitted batch job\s+(\d+)", stdout)
    if match:
        job_id = match.group(1)
    return {
        "returncode": proc.returncode,
        "stdout": stdout,
        "stderr": stderr,
        "job_id": job_id,
        "submitted": proc.returncode == 0 and job_id is not None,
        "command": ["ssh", host, submit_command, str(script_path)],
        "scheduler": "remote_slurm",
        "remote_host": host,
        "submit_script": str(script_path),
    }


def check_remote_slurm_job(job_id: str, host: str | None = None) -> Dict[str, Any]:
    cfg = _remote_slurm_config()
    remote_host = host or cfg.get("host", "gpu2")
    queue_cmd = cfg.get("queue_query_command", "squeue")
    acct_cmd = cfg.get("accounting_command", "sacct")

    queue_proc = subprocess.run(
        ["ssh", remote_host, queue_cmd, "-j", job_id, "-h", "-o", "%T|%M|%N"],
        capture_output=True,
        text=True,
        check=False,
    )
    queue_out = queue_proc.stdout.strip()
    if queue_proc.returncode == 0 and queue_out:
        state, elapsed, node = (queue_out.split("|", 2) + ["", ""])[:3]
        return {
            "scheduler": "remote_slurm",
            "remote_host": remote_host,
            "job_id": job_id,
            "status": state,
            "elapsed": elapsed,
            "node": node,
            "raw": queue_out,
        }

    acct_proc = subprocess.run(
        ["ssh", remote_host, acct_cmd, "-j", job_id, "--format=JobIDRaw,State,Elapsed,NodeList", "-P", "-n"],
        capture_output=True,
        text=True,
        check=False,
    )
    acct_lines = [line.strip() for line in acct_proc.stdout.splitlines() if line.strip()]
    if acct_proc.returncode == 0 and acct_lines:
        fields = acct_lines[0].split("|")
        return {
            "scheduler": "remote_slurm",
            "remote_host": remote_host,
            "job_id": fields[0] if fields else job_id,
            "status": fields[1] if len(fields) > 1 else "UNKNOWN",
            "elapsed": fields[2] if len(fields) > 2 else "",
            "node": fields[3] if len(fields) > 3 else "",
            "raw": acct_lines[0],
        }

    registry = _load_remote_job_registry()
    metadata = registry.get(str(job_id), {})
    status_file = metadata.get("status_file")
    if status_file and Path(status_file).exists():
        status_text = Path(status_file).read_text(encoding="utf-8").strip() or "UNKNOWN"
        return {
            "scheduler": "remote_slurm",
            "remote_host": remote_host,
            "job_id": job_id,
            "status": status_text,
            "raw": status_text,
            "submit_script": metadata.get("submit_script"),
            "output_dir": metadata.get("output_dir"),
        }

    return {
        "scheduler": "remote_slurm",
        "remote_host": remote_host,
        "job_id": job_id,
        "status": "UNKNOWN",
        "raw": acct_proc.stderr.strip() or queue_proc.stderr.strip(),
    }


def _normalize_gases(params: Dict[str, Any]) -> List[str]:
    gases = params.get("gases")
    if gases is None:
        gas = params.get("gas")
        if gas is None:
            raise KeyError("Expected 'gas' or 'gases' for String-TST/external-potential workflows")
        gases = [gas]
    if isinstance(gases, str):
        gases = [gases]
    normalized = [str(item) for item in gases]
    if not normalized:
        raise ValueError("Gas list must not be empty")
    return normalized


def build_string_tst_command(params: Dict[str, Any]) -> Dict[str, Any]:
    work_dir = params["output_dir"]
    gases = _normalize_gases(params)
    generate_script = get_config_path("software", "string_tst_generate_script")
    submit_template = get_config_path("software", "string_tst_submit_template")
    generation_steps: List[Dict[str, Any]] = []

    for gas in gases:
        input_dir = f"{work_dir}/string_input_{gas}"
        command = [
            "python3",
            str(generate_script),
            "--cif-dir",
            params["cif_dir"],
            "--output-dir",
            input_dir,
            "--gas",
            gas,
        ]
        if "temperature" in params:
            command.extend(["--temperature", str(params["temperature"])])
        if "directions" in params:
            command.extend(["--directions", *[str(item) for item in params["directions"]]])
        if "nmax" in params:
            command.extend(["--nmax", str(params["nmax"])])
        if "spacing" in params:
            command.extend(["--spacing", str(params["spacing"])])
        if "cutoff" in params:
            command.extend(["--cutoff", str(params["cutoff"])])
        if "fh_signal" in params:
            command.extend(["--fh-signal", str(params["fh_signal"])])
        if "running_steps" in params:
            command.extend(["--running-steps", str(params["running_steps"])])
        if "n_string" in params:
            command.extend(["--n-string", str(params["n_string"])])
        if "move_frac" in params:
            command.extend(["--move-frac", str(params["move_frac"])])
        if "move_angle" in params:
            command.extend(["--move-angle", str(params["move_angle"])])
        if "convergence" in params:
            command.extend(["--convergence", str(params["convergence"])])
        generation_steps.append(
            {
                "gas": gas,
                "command": command,
                "input_dir": input_dir,
                "manifest": f"{input_dir}/manifest.csv",
            }
        )

    return {
        "plan_name": "string-tst",
        "cwd": None,
        "work_dir": work_dir,
        "gases": gases,
        "generation_steps": generation_steps,
        "submit_template": str(submit_template),
        "submit_script": f"{work_dir}/submit_string_auto.sbatch",
        "logs_dir": f"{work_dir}/logs",
        "note": "When run with --execute, BiMemAgent generates String inputs for one or two gases and renders a ready-to-submit Slurm script."
    }


def build_structure_gen_command(params: Dict[str, Any]) -> Dict[str, Any]:
    pormake_python = get_config_path("software", "pormake_python")
    runner = PROJECT_ROOT / "scripts" / "run_structure_gen.py"
    scheduler_defaults = get_scheduler_defaults().get("structure_gen", {})
    material_type = str(params["material_type"]).upper()
    command = [
        str(pormake_python),
        str(runner),
        "--material-type",
        material_type,
        "--n-structures",
        str(params["n_structures"]),
        "--output-dir",
        params["output_dir"],
    ]
    if "max_atoms" in params:
        command.extend(["--max-atoms", str(params["max_atoms"])])
    if "cutoff" in params:
        command.extend(["--cutoff", str(params["cutoff"])])
    if "max_cell" in params:
        command.extend(["--max-cell", str(params["max_cell"])])
    if params.get("small"):
        command.append("--small")
    if "bb_dir" in params:
        command.extend(["--bb-dir", params["bb_dir"]])
    return {
        "plan_name": "structure-gen",
        "cwd": None,
        "command": command,
        "output_dir": params["output_dir"],
        "scheduler": params.get("scheduler", "local"),
        "queue": params.get("queue", scheduler_defaults.get("queue", "tiny")),
        "ppn": int(params.get("ppn", scheduler_defaults.get("ppn", 1))),
        "walltime": params.get("walltime", scheduler_defaults.get("walltime", "01:00:00")),
        "job_name": params.get("job_name", scheduler_defaults.get("job_name", "pormake_gen")),
        "submit_script": f"{params['output_dir']}/submit_structure_gen.pbs",
        "logs_dir": f"{params['output_dir']}/logs",
        "note": "This command writes generated CIFs plus candidates.txt into the target output directory."
    }


def build_xtb_optimize_command(params: Dict[str, Any]) -> Dict[str, Any]:
    xtb_executable = get_config_path("software", "xtb_executable")
    runner = PROJECT_ROOT / "scripts" / "run_xtb_optimize.py"
    output_dir = params["output_dir"]
    scheduler_defaults = get_scheduler_defaults().get("xtb", {})
    command = [
        "python3",
        str(runner),
        "--input",
        params["input_path"],
        "--output-dir",
        output_dir,
        "--xtb-executable",
        str(xtb_executable),
        "--gfn",
        str(params.get("gfn", 2)),
        "--opt-level",
        str(params.get("opt_level", "normal")),
        "--charge",
        str(params.get("charge", 0)),
        "--uhf",
        str(params.get("uhf", 0)),
        "--parallel",
        str(params.get("parallel", 1)),
        "--output-format",
        str(params.get("output_format", "xyz")),
    ]
    if "namespace" in params:
        command.extend(["--namespace", str(params["namespace"])])
    return {
        "plan_name": "xtb-optimize",
        "cwd": None,
        "command": command,
        "output_dir": output_dir,
        "scheduler": params.get("scheduler", "local"),
        "queue": params.get("queue", scheduler_defaults.get("queue", "tiny")),
        "ppn": int(params.get("ppn", scheduler_defaults.get("ppn", 1))),
        "walltime": params.get("walltime", scheduler_defaults.get("walltime", "00:30:00")),
        "job_name": params.get("job_name", scheduler_defaults.get("job_name", "xtb_opt")),
        "submit_script": f"{output_dir}/submit_xtb_optimize.pbs",
        "logs_dir": f"{output_dir}/logs",
        "note": "xTB currently targets molecule/cluster optimization. For periodic framework cleanup, prefer md-optimize.",
    }


def build_external_potential_command(params: Dict[str, Any]) -> Dict[str, Any]:
    work_dir = params["output_dir"]
    input_dir = f"{work_dir}/vext_inputs"
    scheduler_defaults = get_scheduler_defaults().get("external_potential", {})
    generate_script = get_config_path("software", "tustrast_generate_script")
    default_input_param = get_config_path("software", "tustrast_input_param")
    default_vext_executable = get_config_path("software", "tustrast_executable")
    command = [
        "python3",
        str(generate_script),
        "--cif-dir",
        params["cif_dir"],
        "--output-dir",
        input_dir,
        "--gas",
        params.get("gas", "CO2"),
    ]
    if "bulk_density" in params:
        command.extend(["--bulk-density", str(params["bulk_density"])])
    if "temperature" in params:
        command.extend(["--temperature", str(params["temperature"])])
    if "ngrid" in params:
        command.extend(["--ngrid", str(params["ngrid"])])
    return {
        "plan_name": "external-potential",
        "cwd": None,
        "command": command,
        "work_dir": work_dir,
        "input_dir": input_dir,
        "run_root": f"{work_dir}/vext_runs",
        "summary_csv": f"{work_dir}/vext_runs/vext_summary.csv",
        "input_param": params.get(
            "input_param",
            str(default_input_param),
        ),
        "vext_executable": params.get(
            "vext_executable",
            str(default_vext_executable),
        ),
        "max_parallel": int(params.get("max_parallel", scheduler_defaults.get("max_parallel", 14))),
        "queue": params.get("queue", scheduler_defaults.get("queue", "normal")),
        "node": params.get("node", scheduler_defaults.get("node", "node05.hpc.local")),
        "ppn": int(params.get("ppn", scheduler_defaults.get("ppn", 40))),
        "walltime": params.get("walltime", scheduler_defaults.get("walltime", "72:00:00")),
        "job_name": params.get("job_name", scheduler_defaults.get("job_name", "external_potential")),
        "submit_script": f"{work_dir}/submit_external_potential.pbs",
        "runner_script": str(PROJECT_ROOT / "scripts" / "run_vext_one.py"),
        "collector_script": str(PROJECT_ROOT / "scripts" / "collect_vext_results.py"),
        "note": "When run with --execute, BiMemAgent generates Vext inputs, renders a PBS batch script, and wires in local run/collect helpers."
    }


def build_command_wrapper(plan_name: str, command: List[str], cwd: str | None = None, note: str | None = None, **extra: Any) -> Dict[str, Any]:
    return {
        "plan_name": plan_name,
        "cwd": cwd,
        "command": command,
        "note": note,
        **extra,
    }


def build_guest_forcefield_command(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
    scripts_root = PROJECT_ROOT / "scripts"
    if action in {"default", "build"}:
        script = scripts_root / "build_guest_forcefield.py"
        command = [
            "python3",
            str(script),
            "--name",
            params["name"],
            "--outdir",
            params["output_dir"],
        ]
        if params.get("input_path"):
            command.extend(["--input", params["input_path"]])
        if params.get("target"):
            command.extend(["--target", params["target"]])
        if params.get("mode"):
            command.extend(["--mode", params["mode"]])
        if "charge" in params:
            command.extend(["--charge", str(params["charge"])])
        if params.get('allow_charge_rounding_correction') is True:
            command.append('--allow-charge-rounding-correction')
        if "opt" in params:
            command.extend(["--opt", str(params["opt"])])
        if params.get("lbcc") is False:
            command.append("--no-lbcc")
        if params.get("resname"):
            command.extend(["--resname", params["resname"]])
        if params.get("critical_temperature") is not None:
            command.extend(["--critical-temperature", str(params["critical_temperature"])])
        if params.get("critical_pressure") is not None:
            command.extend(["--critical-pressure", str(params["critical_pressure"])])
        if params.get("acentric_factor") is not None:
            command.extend(["--acentric-factor", str(params["acentric_factor"])])
        if params.get("from_existing_ligpargen"):
            command.extend(["--from-existing-ligpargen", params["from_existing_ligpargen"]])
        if params.get("forcefield_dir"):
            command.extend(["--forcefield-dir", params["forcefield_dir"]])
        return build_command_wrapper(
            "guest-forcefield:build",
            command,
            output_dir=params["output_dir"],
            manifest_path=str(Path(params["output_dir"]) / "package" / "guest.json"),
            note="Builds a reusable RASPA guest force-field package from a single molecule input.",
        )

    if action == "case":
        script = scripts_root / "write_raspa_case_files.py"
        command = [
            "python3",
            str(script),
            "--output-dir",
            params["output_dir"],
            "--framework-name",
            params["framework_name"],
            "--unit-cells",
            *[str(value) for value in params["unit_cells"]],
        ]
        for component in params["components"]:
            command.extend(["--component", component])
        optional_flags = {
            "temperature": "--temperature",
            "pressure": "--pressure",
            "cutoff_vdw": "--cutoff-vdw",
            "charge_method": "--charge-method",
            "ewald_precision": "--ewald-precision",
            "cycles": "--cycles",
            "init_cycles": "--init-cycles",
            "print_every": "--print-every",
            "input_name": "--input-name",
            "pbs_name": "--pbs-name",
            "pbs_queue": "--pbs-queue",
            "pbs_walltime": "--pbs-walltime",
            "pbs_ppn": "--pbs-ppn",
            "pbs_script_name": "--pbs-script-name",
            "charge_type": "--charge-type",
            "run_script": "--run-script",
            "wrapper_script_name": "--wrapper-script-name",
            "framework_cif": "--framework-cif",
            "expected_net_charge": "--expected-net-charge",
            "framework_charge_source": "--framework-charge-source",
            "pacman_dir": "--pacman-dir",
            "pacman_python": "--pacman-python",
            "raspa_sim": "--raspa-sim",
            "normalize_cif_script": "--normalize-cif-script",
            "prepare_script": "--prepare-script",
            "pacman_digits": "--pacman-digits",
        }
        for key, flag in optional_flags.items():
            if params.get(key) is not None:
                command.extend([flag, str(params[key])])
        return build_command_wrapper(
            "guest-forcefield:case",
            command,
            output_dir=params["output_dir"],
            note="Writes RASPA case files and an optional PACMAN-to-RASPA wrapper for mixed-guest GCMC runs.",
        )

    if action == "normalize-cif":
        script = scripts_root / "normalize_cif_charges.py"
        command = ["python3", str(script), params["cif_path"]]
        if params.get('expected_net_charge') is None or not params.get('charge_source'):
            raise ValueError('Read-only charge validation requires expected_net_charge and approved charge_source.')
        command.extend(['--expected-net-charge', str(params['expected_net_charge']), '--charge-source', params['charge_source']])
        if params.get("decimals") is not None:
            command.extend(["--decimals", str(params["decimals"])])
        return build_command_wrapper(
            "guest-forcefield:normalize-cif",
            command,
            cif_path=params["cif_path"],
            note="Validates approved CIF charges against declared cell net charge, without modifying CIF.",
        )

    raise ValueError(f"Unsupported guest-forcefield action: {action}")


def run_subprocess_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    command = plan["command"]
    result = subprocess.run(
        command,
        cwd=plan.get("cwd"),
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
        "cwd": plan.get("cwd"),
        "plan_name": plan.get("plan_name"),
    }


def execute_charge_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    source_dir = Path(plan["source_cif_dir"])
    run_dir = Path(plan.get("run_cif_dir", plan["source_cif_dir"]))
    copied: list[str] = []
    skipped = bool(plan.get("skip_execution"))

    if not skipped:
        run_dir.mkdir(parents=True, exist_ok=True)
        for raw_path in plan.get("raw_input_files", []):
            src = Path(raw_path)
            if src.exists():
                shutil.copy2(src, run_dir / src.name)

    if skipped:
        payload = {
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "command": plan["command"],
            "cwd": plan.get("cwd"),
            "plan_name": plan.get("plan_name"),
            "skipped_execution": True,
        }
    elif plan.get("scheduler") == "remote_slurm":
        submit_script_path = Path(plan["submit_script"])
        submit_script_path.parent.mkdir(parents=True, exist_ok=True)
        Path(plan["logs_dir"]).mkdir(parents=True, exist_ok=True)
        Path(plan["status_file"]).parent.mkdir(parents=True, exist_ok=True)
        submit_script_path.write_text(_render_charge_remote_sbatch(plan), encoding="utf-8")
        payload = _submit_remote_slurm_script(submit_script_path, plan["remote_host"])
        if payload.get("job_id"):
            _register_remote_job(
                str(payload["job_id"]),
                {
                    "submit_script": str(submit_script_path),
                    "status_file": plan["status_file"],
                    "output_dir": str(output_dir),
                    "remote_host": plan["remote_host"],
                },
            )
    else:
        payload = run_subprocess_plan(plan)

    if payload["returncode"] == 0 and plan.get("scheduler") != "remote_slurm":
        output_dir.mkdir(parents=True, exist_ok=True)
        for cif_path in sorted(run_dir.glob("*_pacman.cif")):
            dest = output_dir / cif_path.name
            shutil.copy2(cif_path, dest)
            copied.append(str(dest))

    if skipped:
        output_dir.mkdir(parents=True, exist_ok=True)
        for cif_path in _preferred_charged_files(str(source_dir)):
            dest = output_dir / cif_path.name
            shutil.copy2(cif_path, dest)
            copied.append(str(dest))

    payload["output_dir"] = str(output_dir)
    payload["copied_files"] = copied
    return payload


def _render_string_submit_script(plan: Dict[str, Any]) -> str:
    template_path = Path(plan["submit_template"])
    template = template_path.read_text(encoding="utf-8")
    gases = plan["gases"]
    gas_a = gases[0]
    gas_b = gases[1] if len(gases) > 1 else gases[0]
    work_dir = plan["work_dir"]
    string_exe = get_config_path("software", "string_tst_executable")
    diff_exe = get_config_path("software", "string_tst_diffusivity_executable")

    rendered = (
        template
        .replace("#SBATCH --nodelist=gpu", "#SBATCH -p gpu\n#SBATCH --nodelist=gpu2")
        .replace("#SBATCH --chdir=WORK_DIR", f"#SBATCH --chdir={work_dir}")
        .replace("#SBATCH --output=WORK_DIR/logs/%x_%j.out", f"#SBATCH --output={work_dir}/logs/%x_%j.out")
        .replace("#SBATCH --error=WORK_DIR/logs/%x_%j.err", f"#SBATCH --error={work_dir}/logs/%x_%j.err")
        .replace("WORK_DIR=\"WORK_DIR\"", f"WORK_DIR={json.dumps(work_dir)}")
        .replace('STRING_EXE="${TST_DIR}/bin/GPU_string_polyatmoic_gpu"', f"STRING_EXE={json.dumps(str(string_exe))}")
        .replace('DIFF_EXE="${TST_DIR}/bin/cal_diffusivity"', f"DIFF_EXE={json.dumps(str(diff_exe))}")
        .replace("GAS_A=\"C2H4\"", f"GAS_A={json.dumps(gas_a)}")
        .replace("GAS_B=\"C2H6\"", f"GAS_B={json.dumps(gas_b)}")
    )
    rendered = (
        rendered
        .replace("#SBATCH --gres=gpu:4", "#SBATCH --gres=gpu:1")
        .replace("#SBATCH --cpus-per-task=32", "#SBATCH --cpus-per-task=8")
        .replace('echo "=== Starting: GPU 0,1=${GAS_A}, GPU 2,3=${GAS_B} ==="', 'echo "=== Starting: GPU 0 only ==="')
        .replace("run_gpu 1 ${GAS_A} &\n", "")
        .replace("run_gpu 2 ${GAS_B} &\n", "")
        .replace("run_gpu 3 ${GAS_B} &\n", "")
        .replace("run_gpu 0 ${GAS_A} &\nwait\n", "run_gpu 0 ${GAS_A}\nif [[ \"${GAS_B}\" != \"${GAS_A}\" ]]; then\n    run_gpu 0 ${GAS_B}\nfi\n")
        .replace('for gas in ${GAS_A} ${GAS_B}; do', 'for gas in ${GAS_A} ${GAS_B}; do')
    )
    if len(gases) == 1:
        rendered = rendered.replace('for gas in ${GAS_A} ${GAS_B}; do', 'for gas in ${GAS_A}; do')
    return rendered


def execute_string_tst_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    work_dir = Path(plan["work_dir"])
    work_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["logs_dir"]).mkdir(parents=True, exist_ok=True)

    generation_results: List[Dict[str, Any]] = []
    overall_returncode = 0
    manifests: Dict[str, str] = {}

    for step in plan["generation_steps"]:
        step_plan = {
            "plan_name": f"string-tst:{step['gas']}",
            "cwd": plan.get("cwd"),
            "command": step["command"],
        }
        payload = run_subprocess_plan(step_plan)
        payload["gas"] = step["gas"]
        payload["input_dir"] = step["input_dir"]
        payload["manifest"] = step["manifest"]
        generation_results.append(payload)
        manifests[step["gas"]] = step["manifest"]
        if payload["returncode"] != 0:
            overall_returncode = payload["returncode"]

    submit_script_path = Path(plan["submit_script"])
    submit_script_path.write_text(_render_string_submit_script(plan), encoding="utf-8")

    note = plan["note"]
    if len(plan["gases"]) == 1:
        note += " Single-gas mode reuses the same gas in both GAS_A and GAS_B slots of the legacy scheduler template."

    return {
        "plan_name": plan["plan_name"],
        "returncode": overall_returncode,
        "work_dir": str(work_dir),
        "gases": plan["gases"],
        "generation_results": generation_results,
        "manifests": manifests,
        "submit_script": str(submit_script_path),
        "logs_dir": plan["logs_dir"],
        "note": note,
    }


def _render_external_potential_submit_script(plan: Dict[str, Any]) -> str:
    return textwrap.dedent(
        f"""\
        #!/bin/bash
        #PBS -N {plan["job_name"]}
        #PBS -q {plan["queue"]}
        #PBS -l nodes={plan["node"]}:ppn={plan["ppn"]}
        #PBS -l walltime={plan["walltime"]}
        #PBS -o {plan["work_dir"]}/logs/{plan["job_name"]}.out
        #PBS -e {plan["work_dir"]}/logs/{plan["job_name"]}.err

        set -euo pipefail

        INPUT_ROOT={shlex.quote(plan["input_dir"])}
        RUN_ROOT={shlex.quote(plan["run_root"])}
        SUMMARY_CSV={shlex.quote(plan["summary_csv"])}
        VEXT_EXE={shlex.quote(plan["vext_executable"])}
        RUNNER_SCRIPT={shlex.quote(plan["runner_script"])}
        COLLECT_SCRIPT={shlex.quote(plan["collector_script"])}
        MAX_PARALLEL={int(plan["max_parallel"])}

        mkdir -p {shlex.quote(plan["work_dir"] + "/logs")} "${{RUN_ROOT}}"

        shopt -s nullglob
        inputs=("${{INPUT_ROOT}}"/*.dat)
        shopt -u nullglob

        if [[ "${{#inputs[@]}}" -eq 0 ]]; then
            echo "No .dat inputs found under ${{INPUT_ROOT}}" >&2
            exit 1
        fi

        pids=()
        status=0

        run_one() {{
            local input_dat="$1"
            local structure
            structure="$(basename "${{input_dat}}" .dat)"
            local work_dir="${{RUN_ROOT}}/${{structure}}"
            mkdir -p "${{work_dir}}"
            python3 "${{RUNNER_SCRIPT}}" \\
                --input-dat "${{input_dat}}" \\
                --output-dir "${{work_dir}}" \\
                --vext-executable "${{VEXT_EXE}}" \\
                > "${{work_dir}}/run.stdout" 2> "${{work_dir}}/run.stderr"
        }}

        for input_dat in "${{inputs[@]}}"; do
            while [[ "$(jobs -rp | wc -l)" -ge "${{MAX_PARALLEL}}" ]]; do sleep 5; done
            run_one "${{input_dat}}" &
            pids+=("$!")
        done

        for pid in "${{pids[@]}}"; do
            if ! wait "${{pid}}"; then status=1; fi
        done

        python3 "${{COLLECT_SCRIPT}}" --run-root "${{RUN_ROOT}}" --output "${{SUMMARY_CSV}}"

        echo "Run root: ${{RUN_ROOT}}"
        echo "Summary CSV: ${{SUMMARY_CSV}}"
        exit "${{status}}"
        """
    )


def execute_external_potential_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    work_dir = Path(plan["work_dir"])
    logs_dir = work_dir / "logs"
    work_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    payload = run_subprocess_plan(plan)

    submit_script_path = Path(plan["submit_script"])
    submit_script_path.write_text(_render_external_potential_submit_script(plan), encoding="utf-8")

    input_count = len(list(Path(plan["input_dir"]).glob("*.dat")))
    payload.update(
        {
            "work_dir": str(work_dir),
            "input_dir": plan["input_dir"],
            "input_count": input_count,
            "run_root": plan["run_root"],
            "summary_csv": plan["summary_csv"],
            "submit_script": str(submit_script_path),
            "runner_script": plan["runner_script"],
            "collector_script": plan["collector_script"],
            "logs_dir": str(logs_dir),
        }
    )
    return payload


def _render_xtb_submit_script(plan: Dict[str, Any]) -> str:
    command = _shell_join(plan["command"])
    return textwrap.dedent(
        f"""\
        #!/bin/sh
        #PBS -r n
        #PBS -q {plan["queue"]}
        #PBS -N {plan["job_name"]}
        #PBS -l nodes=1:ppn={int(plan["ppn"])}
        #PBS -l walltime={plan["walltime"]}
        #PBS -o {plan["logs_dir"]}/{plan["job_name"]}.out
        #PBS -e {plan["logs_dir"]}/{plan["job_name"]}.err

        set -euo pipefail
        cd {shlex.quote(plan["output_dir"])}
        {command}
        """
    )


def _render_simple_pbs_script(plan: Dict[str, Any]) -> str:
    command = _shell_join(plan["command"])
    return textwrap.dedent(
        f"""\
        #!/bin/sh
        #PBS -r n
        #PBS -q {plan["queue"]}
        #PBS -N {plan["job_name"]}
        #PBS -l nodes=1:ppn={int(plan["ppn"])}
        #PBS -l walltime={plan["walltime"]}
        #PBS -o {plan["logs_dir"]}/{plan["job_name"]}.out
        #PBS -e {plan["logs_dir"]}/{plan["job_name"]}.err

        set -euo pipefail
        mkdir -p {shlex.quote(plan["logs_dir"])}
        cd {shlex.quote(plan.get("output_dir", plan.get("output_root", str(PROJECT_ROOT))))}
        {command}
        """
    )


def _submit_pbs_wrapper(plan: Dict[str, Any]) -> Dict[str, Any]:
    submit_script_path = Path(plan["submit_script"])
    submit_script_path.parent.mkdir(parents=True, exist_ok=True)
    Path(plan["logs_dir"]).mkdir(parents=True, exist_ok=True)
    submit_script_path.write_text(_render_simple_pbs_script(plan), encoding="utf-8")
    proc = subprocess.run(
        ["qsub", str(submit_script_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "plan_name": plan["plan_name"],
        "scheduler": "pbs",
        "submit_script": str(submit_script_path),
        "logs_dir": plan["logs_dir"],
        "job_id": proc.stdout.strip() if proc.returncode == 0 else None,
        "submitted": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "command": plan["command"],
    }


def execute_xtb_optimize_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(plan["logs_dir"])
    logs_dir.mkdir(parents=True, exist_ok=True)

    if plan.get("scheduler") == "pbs":
        submit_script_path = Path(plan["submit_script"])
        submit_script_path.write_text(_render_xtb_submit_script(plan), encoding="utf-8")
        proc = subprocess.run(
            ["qsub", str(submit_script_path)],
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "plan_name": plan["plan_name"],
            "scheduler": "pbs",
            "submit_script": str(submit_script_path),
            "logs_dir": str(logs_dir),
            "job_id": proc.stdout.strip() if proc.returncode == 0 else None,
            "submitted": proc.returncode == 0,
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "output_dir": str(output_dir),
            "command": plan["command"],
        }

    payload = run_subprocess_plan(plan)
    payload.update(
        {
            "output_dir": str(output_dir),
            "logs_dir": str(logs_dir),
        }
    )
    return payload


def execute_pore_analysis_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    output_root = Path(plan["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    Path(plan["logs_dir"]).mkdir(parents=True, exist_ok=True)
    if plan.get("scheduler") == "pbs":
        payload = _submit_pbs_wrapper(plan)
        payload["output_root"] = str(output_root)
        payload["output_csv"] = plan.get("output_csv")
        return payload
    payload = run_subprocess_plan(plan)
    payload["output_root"] = str(output_root)
    payload["output_csv"] = plan.get("output_csv")
    payload["logs_dir"] = plan["logs_dir"]
    return payload


def execute_structure_gen_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(plan["logs_dir"]).mkdir(parents=True, exist_ok=True)
    if plan.get("scheduler") == "pbs":
        payload = _submit_pbs_wrapper(plan)
        payload["output_dir"] = str(output_dir)
        return payload
    payload = run_subprocess_plan(plan)
    payload["output_dir"] = str(output_dir)
    payload["logs_dir"] = plan["logs_dir"]
    return payload


def pretty_json(data: Any) -> str:
    return json.dumps(_normalize(data), indent=2, ensure_ascii=False)
