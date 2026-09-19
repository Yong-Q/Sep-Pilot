"""SLURM job submission helpers.

All computation tools submit through this module.
Supports both local sbatch and remote SSH sbatch.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import get_config


# ── SLURM Script Renderer ──────────────────────────────────────────

def get_resource_config(task_type: str = "default") -> Dict[str, Any]:
    """根据任务类型返回资源配置，尽量使用完整节点"""
    # 默认配置：使用完整节点（16核）
    configs = {
        "default": {"ntasks": 1, "cpus_per_task": 16, "walltime": "24:00:00"},
        "henry": {"ntasks": 1, "cpus_per_task": 16, "walltime": "04:00:00"},
        "isotherm": {"ntasks": 1, "cpus_per_task": 16, "walltime": "08:00:00"},
        "pore": {"ntasks": 1, "cpus_per_task": 16, "walltime": "02:00:00"},
        "literature": {"ntasks": 1, "cpus_per_task": 16, "walltime": "01:00:00"},
        "multi_step": {"ntasks": 1, "cpus_per_task": 16, "walltime": "12:00:00"},
        "inverse_design": {"ntasks": 1, "cpus_per_task": 16, "walltime": "24:00:00"},
        "forward_design": {"ntasks": 1, "cpus_per_task": 16, "walltime": "12:00:00"},
        "machine_learning": {"ntasks": 1, "cpus_per_task": 16, "walltime": "24:00:00"},
        "database": {"ntasks": 1, "cpus_per_task": 16, "walltime": "04:00:00"},
    }
    return configs.get(task_type, configs["default"])


def render_sbatch_script(
    job_name: str,
    command: str,
    work_dir: str,
    partition: str = "compute",
    ntasks: int = 1,
    cpus_per_task: int = 16,  # 默认使用完整节点（16核）
    gres: str = "",
    walltime: str = "24:00:00",
    nodelist: str = "",
    memory_mb: Optional[int] = None,
    output: str = "",
    error: str = "",
    modules: List[str] | None = None,
    env_exports: Dict[str, str] | None = None,
    extra_directives: List[str] | None = None,
    task_type: str = "default",  # 新增：任务类型，用于自动配置资源
) -> str:
    """Render a SLURM sbatch script.

    资源管理策略：
    - 尽量使用完整节点（16核），避免碎片化
    - 根据任务类型自动调整资源和时间限制
    - 不要"东占一个节点西占一个节点"
    """
    # 根据任务类型自动配置资源
    if task_type != "default":
        config = get_resource_config(task_type)
        ntasks = config.get("ntasks", ntasks)
        cpus_per_task = config.get("cpus_per_task", cpus_per_task)
        walltime = config.get("walltime", walltime)

    # All SDK compute workers share the same reviewed scheduling envelope.
    # Tool-specific default arguments must not discard the resource agent's
    # estimate. Explicit user choices were preserved when that envelope formed.
    from .watch_context import get_context
    allocation = get_context().get('resource_allocation') or {}
    if allocation.get('resource_review_id'):
        memory_mb = allocation.get('memory_mb', memory_mb)
        partition = allocation.get('partition', partition)
        nodelist = allocation.get('nodelist', nodelist)
        cpus_per_task = allocation.get('cpus', cpus_per_task)

    # 防止递归调用 sbatch submit.sh 导致无限提交
    if "sbatch submit.sh" in command or "sbatch ./submit.sh" in command:
        raise ValueError(
            "递归调用错误: command 包含 'sbatch submit.sh'，这会导致无限递归提交。"
            "请使用正确的模拟命令，例如: run_sim simulation.input"
        )

    logs_dir = os.path.join(work_dir, "logs")
    output = output or f"{logs_dir}/{job_name}_%j.out"
    error = error or f"{logs_dir}/{job_name}_%j.err"
    
    lines = [
        "#!/bin/bash",
        f"#SBATCH -J {job_name}",
        f"#SBATCH -p {partition}",
        f"#SBATCH -n {ntasks}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
    ]
    if gres:
        lines.append(f"#SBATCH --gres={gres}")
    if nodelist:
        lines.append(f"#SBATCH --nodelist={nodelist}")
    if memory_mb is not None:
        if isinstance(memory_mb,bool) or not isinstance(memory_mb,int) or memory_mb<64:raise ValueError('finite memory_mb >=64 MiB required')
        lines.append(f'#SBATCH --mem={memory_mb}M')
    lines.extend([
        f"#SBATCH -t {walltime}",
        f"#SBATCH -o {output}",
        f"#SBATCH -e {error}",
        "",
        "set -euo pipefail",
        f"mkdir -p {shlex.quote(logs_dir)}",
        f"cd {shlex.quote(work_dir)}",
    ])
    
    if modules:
        for mod in modules:
            lines.append(f"module load {mod} 2>/dev/null || true")
    
    if env_exports:
        for key, val in env_exports.items():
            lines.append(f"export {key}={shlex.quote(val)}")
    
    if extra_directives:
        for d in extra_directives:
            lines.append(f"#SBATCH {d}")
    
    lines.extend(["", command, ""])
    return "\n".join(lines)


# ── Job Submission ──────────────────────────────────────────────────

def _record_submission_receipt(func):
    from functools import wraps
    @wraps(func)
    def submit(*args, **kwargs):
        result = func(*args, **kwargs)
        if result.get('submitted') and result.get('job_id'):
            from .watch_context import get_context
            context = get_context()
            if context.get('username') and context.get('conv_id'):
                work_dir = kwargs.get('work_dir') or (args[1] if len(args) > 1 else '')
                try:
                    from .job_watch import get_watch
                    get_watch().register(result['job_id'], work_dir=str(work_dir),
                                         username=context['username'], conv_id=context['conv_id'],
                                         agent_name=context.get('agent_name', ''), tool=context.get('tool_name', 'submit_job'))
                except Exception as error:
                    # The scheduler DID accept the job. Do not turn a storage
                    # failure into a retryable "not submitted" receipt.
                    result['receipt_error'] = str(error)
        return result
    return submit

@_record_submission_receipt
def submit_sbatch_local(script_content: str, work_dir: str) -> Dict[str, Any]:
    """Submit a SLURM job locally via sbatch."""
    script_path = os.path.join(work_dir, "submit.sh")
    os.makedirs(work_dir, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)
    
    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True, text=True, timeout=30,
        )
        job_id = None
        if result.returncode == 0:
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
        return {
            "submitted": result.returncode == 0,
            "job_id": job_id,
            "script": script_path,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except FileNotFoundError:
        return {
            "submitted": False,
            "error": "sbatch not found - SLURM not available on this node",
            "script": script_path,
        }
    except subprocess.TimeoutExpired:
        return {"submitted": False, "error": "sbatch timeout"}


@_record_submission_receipt
def submit_sbatch_remote(
    script_content: str,
    work_dir: str,
    host: str = "",
    submit_command: str = "sbatch",
) -> Dict[str, Any]:
    """Submit a SLURM job to a remote host via SSH + sbatch."""
    config = get_config()
    from .config import get_config as _gc
    # Load runtime config for remote settings
    runtime_path = config.project_root / "config" / "runtime.json"
    remote_cfg = {}
    if runtime_path.exists():
        with open(runtime_path) as f:
            remote_cfg = json.load(f).get("remote_slurm", {})
    
    host = host or remote_cfg.get("host", "gpu2")
    submit_command = remote_cfg.get("submit_command", "sbatch")
    
    script_path = os.path.join(work_dir, "submit.sh")
    os.makedirs(work_dir, exist_ok=True)
    
    with open(script_path, "w") as f:
        f.write(script_content)
    
    try:
        result = subprocess.run(
            ["ssh", host, submit_command, script_path],
            capture_output=True, text=True, timeout=30,
        )
        job_id = None
        if result.returncode == 0:
            match = re.search(r"Submitted batch job (\d+)", result.stdout)
            if match:
                job_id = match.group(1)
        return {
            "submitted": result.returncode == 0,
            "job_id": job_id,
            "host": host,
            "script": script_path,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "returncode": result.returncode,
        }
    except Exception as e:
        return {"submitted": False, "error": str(e), "script": script_path}


def submit_job(
    job_name: str,
    command: str,
    work_dir: str,
    mode: str = "local",
    **slurm_kwargs,
) -> Dict[str, Any]:
    """Submit a SLURM job. mode: 'local', 'remote', or 'auto'.

    'auto' tries local first, falls back to remote.
    """
    # 防止递归调用 sbatch submit.sh 导致无限提交
    if "sbatch submit.sh" in command or "sbatch ./submit.sh" in command:
        return {
            "submitted": False,
            "error": "递归调用错误: command 包含 'sbatch submit.sh'，这会导致无限递归提交。"
                    "请使用正确的模拟命令，例如: run_sim simulation.input",
            "hint": "Henry系数计算应使用: run_henry 工具或 run_sim simulation.input",
            "work_dir": work_dir,
        }

    # `host` is a submission-target option (remote SSH host), NOT an
    # sbatch/slurm option — pop it so it doesn't leak into render_sbatch_script.
    host = slurm_kwargs.pop("host", "")
    # `timeout_minutes` is a BLOCKING-wait / polling concept (used by
    # submit_and_wait), NOT an sbatch directive. submit_job never blocks, so a
    # caller passing timeout_minutes (e.g. run_cdft/run_vasp/run_md_optimize)
    # must not leak it into render_sbatch_script — that crashes with
    # "unexpected keyword argument 'timeout_minutes'". Pop it silently.
    slurm_kwargs.pop("timeout_minutes", None)
    script = render_sbatch_script(
        job_name=job_name,
        command=command,
        work_dir=work_dir,
        **slurm_kwargs,
    )

    if mode == "remote":
        return submit_sbatch_remote(script, work_dir, host=host)

    if mode == "local":
        return submit_sbatch_local(script, work_dir)

    # auto mode
    result = submit_sbatch_local(script, work_dir)
    if result.get("submitted"):
        return result
    return submit_sbatch_remote(script, work_dir, host=host)


# ── Job Status ──────────────────────────────────────────────────────

# Terminal (non-running) SLURM states that mean the job is no longer executing.
TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "OUT_OF_ME+"}
# States that indicate the job FAILED (as opposed to cleanly COMPLETED).
FAILURE_STATES = {"FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "OUT_OF_ME+", "PREEMPTED", "REVOKED", "BOOT_FAIL", "DEADLINE"}


# ── squeue short-code → canonical job state (ai2-kit absorb) ──────────
# Single source of truth for what a `squeue --format='%i %T'` code means,
# including the terminal/failed flags. Mirrors ai2-kit's JobState enum design
# (each state carries `terminal`). `failed` is the subset of terminal states
# that did NOT cleanly complete (CANCELLED counts as failure here, matching
# FAILURE_STATES above — a cancelled job still must be surfaced to the agent).
SQUEUE_TRANSLATE = {
    'PD': ("PENDING", False, False),     # queued
    'CF': ("PENDING", False, False),     # configuring
    'S':  ("PENDING", False, False),     # suspended
    'R':  ("RUNNING", False, False),
    'CG': ("RUNNING", False, False),     # completing
    'CD': ("COMPLETED", True, False),
    'CA': ("CANCELLED", True, True),
    'F':  ("FAILED", True, True),
    'NF': ("NODE_FAIL", True, True),
    'RV': ("REVOKED", True, True),
    'SE': ("SPECIAL_EXIT", True, True),
    'TO': ("TIMEOUT", True, True),
    'DL': ("DEADLINE", True, True),
}

# squeue %T emits long names (e.g. "RUNNING") while %t emits short codes
# ("R"). Accept BOTH so callers can't silently misclassify a running job as
# terminal (the bug that happened when %T was fed into the short-code table).
_SQUEUE_LONG_TO_SHORT = {
    'PENDING': 'PD', 'CONFIGURING': 'CF', 'SUSPENDED': 'S',
    'RUNNING': 'R', 'COMPLETING': 'CG', 'COMPLETED': 'CD',
    'CANCELLED': 'CA', 'FAILED': 'F', 'NODE_FAIL': 'NF',
    'REVOKED': 'RV', 'SPECIAL_EXIT': 'SE', 'TIMEOUT': 'TO',
    'DEADLINE': 'DL', 'OUT_OF_MEMORY': 'F', 'BOOT_FAIL': 'NF',
    'PREEMPTED': 'CA',
}


def translate_squeue_state(short: str) -> Dict[str, Any]:
    """Map a squeue state token — short code ('R', 'CD', 'F') OR long name
    ('RUNNING', 'COMPLETED', 'FAILED') — to a canonical {status, terminal,
    failed} dict. Unknown tokens are treated as terminal (the job is not
    running); the caller decides how to classify further."""
    norm = _SQUEUE_LONG_TO_SHORT.get(short, short)
    status, terminal, failed = SQUEUE_TRANSLATE.get(
        norm, (short, True, False))
    return {"status": status, "terminal": terminal, "failed": failed}


# ── Batch squeue query with TTL cache (ai2-kit absorb) ────────────────
# ai2-kit polls with ONE `squeue --noheader --format='%i %t' -u $USER` per
# polling interval and caches the result, instead of spawning one subprocess
# per job. We do the same: JobWatch asks once per poll cycle for all its jobs.
# The cache lives here so agent tools (check_job etc.) can also reuse it.
_BATCH_CACHE_LOCK = threading.RLock()
_batch_state_cache: Dict[str, str] = {}   # job_id -> squeue short code
_batch_state_ts = 0.0
BATCH_STATE_TTL = 5.0                     # seconds (local squeue is cheap; JobWatch polls every 5s)


def normalize_job_id(job_id: Any) -> str:
    """Extract the pure numeric SLURM job id from any string that may carry
    an sbatch prefix (e.g. 'Submitted batch job 3758' -> '3758').

    JobWatch registers whatever job_id string it is handed; historically an
    sbatch wrapper returned the whole 'Submitted batch job <N>' line, which
    broke squeue lookups (-> UNKNOWN -> false terminal -> spurious
    '✅ completion' notifications) and broke TaskLine matching
    ('Submitted batch job 3758' never equals '3758').  Normalizing at the
    boundary (register + check + taskline match) fixes both.  Returns the
    input unchanged when no digits are present (keeps non-numeric keys).
    """
    s = str(job_id or "").strip()
    m = re.search(r"(\d+)", s)
    return m.group(1) if m else s


def batch_fetch_states(job_ids: Optional[List[str]] = None, ttl: float = BATCH_STATE_TTL) -> Dict[str, Dict[str, Any]]:
    """One `squeue --noheader --format='%i %T' -u $USER` for many jobs, TTL-cached.

    Returns {job_id: {"status": <long>, "terminal": bool, "failed": bool}}
    for every job found in the queue (or, if job_ids is given, the subset
    that is currently queued). Jobs ABSENT from the result are not running —
    the caller must decide (e.g. fall back to check_job_status, which reads
    the job.done success indicator / .err logs to classify them)."""
    global _batch_state_cache, _batch_state_ts
    with _BATCH_CACHE_LOCK:
        if not (time.time() - _batch_state_ts < ttl) or not _batch_state_cache:
            try:
                proc = subprocess.run(
                    ["squeue", "--noheader", "--format=%i|%t|%R|%S", "-u",
                     os.environ.get("USER", "")],
                    capture_output=True, text=True, timeout=15,
                )
                raw: Dict[str, str] = {}
                if proc.returncode == 0:
                    for line in proc.stdout.splitlines():
                        parts = line.strip().split('|') if '|' in line else line.split()
                        if len(parts) >= 2:
                            raw[parts[0]] = {'short':parts[1], 'pending_reason':parts[2] if len(parts)>2 else '',
                                            'estimated_start':parts[3] if len(parts)>3 else ''}
                _batch_state_cache = raw
                _batch_state_ts = time.time()
            except Exception:
                pass  # reuse last cache on query failure
        states = {k: v for k, v in _batch_state_cache.items()}
    if job_ids is not None:
        wanted = {str(j) for j in job_ids}
        states = {k: v for k, v in states.items() if k in wanted}
    return {jid: {**translate_squeue_state(value['short']), 'pending_reason':value.get('pending_reason',''),
                  'estimated_start':value.get('estimated_start','')} if isinstance(value,dict) else translate_squeue_state(value)
            for jid, value in states.items()}


def check_job_status(job_id: str, host: str = "", work_dir: str = "") -> Dict[str, Any]:
    """Check SLURM job status via squeue/sacct.

    Returns an enriched dict with:
      - status: raw SLURM state string (RUNNING / COMPLETED / FAILED / CANCELLED / ...)
      - terminal: True if the job is no longer running
      - failed: True if the state is a failure state (not COMPLETED)
      - error: stderr tail for terminal failure states (read from the job's .err file)
    """

    job_id = normalize_job_id(job_id)
    config = get_config()
    runtime_path = config.project_root / "config" / "runtime.json"
    remote_cfg = {}
    if runtime_path.exists():
        with open(runtime_path) as f:
            remote_cfg = json.load(f).get("remote_slurm", {})

    # Local-first: jobs are normally submitted via local sbatch (this machine
    # IS the SLURM login node). Only fall back to SSH (remote host) when the
    # job is not found locally AND a host was explicitly given.
    # NOTE: do NOT default host to remote_cfg host here — that caused false
    # "job not found" failures for locally-submitted jobs (checked on gpu2).
    queue_cmd = remote_cfg.get("queue_query_command", "squeue")
    acct_cmd = remote_cfg.get("accounting_command", "sacct")

    status = "UNKNOWN"
    elapsed = ""
    node = ""
    pending_reason = ""

    def _query_local(command_args: List[str]):
        try:
            proc = subprocess.run(command_args, capture_output=True, text=True, timeout=15)
            return proc
        except Exception:
            return None

    def _query_remote(host_: str, remote_cmd: str):
        try:
            return subprocess.run(["ssh", host_, remote_cmd], capture_output=True, text=True, timeout=15)
        except Exception:
            return None

    # 1) LOCAL squeue
    proc = _query_local([queue_cmd, "-j", job_id, "-h", "-o", "%T|%M|%N|%R"])
    if proc and proc.returncode == 0 and proc.stdout.strip():
        fields = proc.stdout.strip().split("|")
        status = fields[0] if fields else "UNKNOWN"
        elapsed = fields[1] if len(fields) > 1 else ""
        node = fields[2] if len(fields) > 2 else ""
        pending_reason = fields[3] if len(fields) > 3 else ""

    # 2) LOCAL sacct (terminal state for finished jobs)
    if status in ("", "UNKNOWN"):
        proc = _query_local([acct_cmd, "-j", job_id, "--format=JobIDRaw,State,Elapsed,NodeList", "-P", "-n", "--noheader"])
        if proc and proc.returncode == 0 and proc.stdout.strip():
            fields = proc.stdout.strip().split("|")
            status = fields[1] if len(fields) > 1 else status
            elapsed = fields[2] if len(fields) > 2 else elapsed
            node = fields[3] if len(fields) > 3 else node

    # 2b) LOCAL scontrol — works even when sacct's accounting storage is
    #     disabled (the common case here: "Slurm accounting storage is
    #     disabled"). Gives the authoritative JobState + ExitCode.
    _scontrol_exit_code = ""
    if status in ("", "UNKNOWN"):
        proc = _query_local(["scontrol", "show", "job", job_id])
        if proc and proc.returncode == 0 and proc.stdout.strip():
            _sc = proc.stdout
            import re as _re
            m = _re.search(r"JobState=(\S+)", _sc)
            if m:
                status = m.group(1)
            m = _re.search(r"ExitCode=(\S+)", _sc)
            if m:
                _scontrol_exit_code = m.group(1)
            m = _re.search(r"Elapsed=(\S+)", _sc)
            if m:
                elapsed = m.group(1)
            m = _re.search(r"BatchHost=(\S+)", _sc)
            if m:
                node = m.group(1)

    # 2c) LOCAL job.done success indicator (ai2-kit absorb) — the sbatch
    #     scripts write `echo "$SLURM_JOB_ID" > <work_dir>/job.done` ONLY after
    #     all output validation passed. When accounting is disabled (gpu2),
    #     a finished job vanishes from squeue/sacct/scontrol and reports
    #     UNKNOWN — a present indicator proves it truly completed, whereas a
    #     vanished job WITHOUT the indicator is a failure (fell over, killed,
    #     node died). This is strictly more reliable than guessing from output
    #     files. Local jobs only (remote job.done lives on the remote host).
    if status in ("", "UNKNOWN") and work_dir:
        try:
            _done = os.path.join(work_dir, "job.done")
            if os.path.isfile(_done):
                _content = Path(_done).read_text(errors="ignore").strip()
                if _content and job_id in _content:
                    status = "COMPLETED"
                    node = node or "local"
        except Exception:
            pass

    # 3) REMOTE fallback — only if the job is NOT found locally AND the caller
    #    explicitly asked for a remote host (e.g. remote_slurm workloads).
    if status in ("", "UNKNOWN") and host:
        # NOTE: ssh concatenates args into ONE remote shell command, so the format
        # string with '|' MUST be quoted or the remote shell splits it into commands.
        remote_cmd = f"{queue_cmd} -j {shlex.quote(job_id)} -h -o '%T|%M|%N'"
        proc = _query_remote(host, remote_cmd)
        if proc and proc.returncode == 0 and proc.stdout.strip():
            fields = proc.stdout.strip().split("|")
            status = fields[0] if fields else "UNKNOWN"
            elapsed = fields[1] if len(fields) > 1 else ""
            node = fields[2] if len(fields) > 2 else ""
        if status in ("", "UNKNOWN"):
            remote_cmd = f"{acct_cmd} -j {shlex.quote(job_id)} --format=JobIDRaw,State,Elapsed,NodeList -P -n --noheader"
            proc = _query_remote(host, remote_cmd)
            if proc and proc.returncode == 0 and proc.stdout.strip():
                fields = proc.stdout.strip().split("|")
                status = fields[1] if len(fields) > 1 else status
                elapsed = fields[2] if len(fields) > 2 else elapsed
                node = fields[3] if len(fields) > 3 else node

    result_host = host if host else "local"

    # If still unknown, the job may have finished and fallen out of accounting.
    # Treat "not found in squeue nor sacct" as terminal (finished).
    is_terminal = status in TERMINAL_STATES or status in ("", "UNKNOWN")

    result: Dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "pending_reason": pending_reason,
        "elapsed": elapsed,
        "node": node,
        "host": result_host,
        "terminal": is_terminal,
        "failed": status in FAILURE_STATES,
    }
    if _scontrol_exit_code:
        result["exit_code"] = _scontrol_exit_code

    # For terminal failures / finished-but-unaccounted jobs, read stderr +
    # run.log tail. On gpu2 sacct is DISABLED, so a failed job often reports
    # "UNKNOWN" instead of "FAILED" — we detect failure from the OUTPUT:
    #   - the job's .err file contains a failure marker, or
    #   - the work_dir run.log shows "FATAL"/"ERROR: RASPA simulation FAILED",
    #   - no valid loading data was produced.
    if is_terminal and (result["failed"] or status in ("", "UNKNOWN")):
        stderr_tail = _read_slurm_stderr(job_id, result_host, work_dir)
        if stderr_tail:
            result["error"] = stderr_tail
        try:
            from .raspa_errors import summarize_run_log
            combined = stderr_tail
            if work_dir:
                if result_host != "local" and result_host:
                    _rl_proc = subprocess.run(
                        ["ssh", result_host, f"tail -c 2000 {work_dir}/P_*/run.log 2>/dev/null | tail -c 2000"],
                        capture_output=True, text=True, timeout=15,
                    )
                else:
                    _rl_proc = subprocess.run(
                        ["bash", "-c", f"tail -c 2000 {work_dir}/P_*/run.log 2>/dev/null | tail -c 2000"],
                        capture_output=True, text=True, timeout=15,
                    )
                rl = _rl_proc.stdout.strip()
                if rl:
                    combined += "\n" + rl[-2000:]
                    result["run_log_tail"] = rl[-2000:]
            # ── Data-verified completion short-circuit ─────────────────────
            # On sacct-disabled hosts a finished job leaves squeue and is
            # reported UNKNOWN even when it SUCCEEDED.  This RASPA build only
            # writes the CIF preamble into run.log (fully-buffered stdout);
            # the authoritative log incl. "Average loading absolute" /
            # "Simulation finished" lives in P_*/Output/System_0/*.data.
            # So when .err is clean AND >=1 pressure .data has non-zero
            # loading AND run.log carries no real failure marker, declare
            # COMPLETED — do NOT let the shared short run.log tail trip the
            # raspa_error classifier into a fake "CIF parse failure"
            # (that false positive caused ~20 pointless resubmissions).
            if status in ("", "UNKNOWN"):
                try:
                    _zv = _all_pressures_zero_loading(work_dir, result_host)
                    _data_ok = _zv is False
                except Exception:
                    _data_ok = False
                _rl_low = (rl or "").lower()
                _rl_hard_fail = any(m in _rl_low for m in (
                    "raspa simulation failed", "fatal", "segmentation fault",
                    "core dumped", "traceback"))
                if _data_ok and not _rl_hard_fail and not (stderr_tail or "").strip():
                    result["status"] = "COMPLETED"
                    result["failed"] = False
                    result["terminal"] = True
                    result["note"] = (
                        "sacct 不可用：作业已离开 squeue 故报 UNKNOWN；但 .err 为空且 .data "
                        "含非零 loading（按数据判定为完成）。run.log 仅含 CIF 前导属该 RASPA "
                        "构建 stdout 缓冲的正常行为，不是解析失败。")
                    return result
            ra = summarize_run_log(combined)
            if ra["raspa_error"]:
                result["diagnosis"] = {
                    "kind": "raspa",
                    "cause": ra["cause"],
                    "fixes": ra["fixes"],
                    "critical_lines": ra["critical_lines"],
                }
                # RASPA error in the output = definitively a failure even if
                # the SLURM state came back UNKNOWN (sacct disabled).
                result["failed"] = True
        except Exception:
            pass
        # Generic output-based failure markers (script echoes "FATAL" on failure)
        if not result["failed"] and status in ("", "UNKNOWN"):
            low = (stderr_tail or "").lower()
            if any(m in low for m in ("fatal", "traceback", "segmentation fault",
                                      "core dumped", "error:", "slurmstepd: error",
                                      "raspa simulation failed", "no valid gcmc")):
                result["failed"] = True

    # Zero-loading silent failure: the job COMPLETED (or otherwise reached a
    # terminal state) but every pressure point produced loading = 0. This is
    # the classic "ran but produced nothing" failure (pressure-unit bug,
    # forcefield mismatch, blocked pores). Detect it from the .data files so
    # JobWatch can wake the agent — runs for ANY terminal job, not just ones
    # SLURM already marked FAILED.
    if is_terminal and not result["failed"] and work_dir:
        try:
            _zero = _all_pressures_zero_loading(work_dir, result_host)
            if _zero is not None and _zero is True:
                result["failed"] = True
                result["diagnosis"] = {
                    "kind": "zero_loading",
                    "cause": "作业 COMPLETED 但所有压力点 loading 均为 0（静默失败）——RASPA 运行完成但没有任何吸附。",
                    "fixes": [
                        "检查压力单位：RASPA ExternalPressure 必须为 Pa（1 bar = 1e5 Pa）",
                        "检查力场/伪原子是否匹配该气体",
                        "检查框架孔隙是否被堵（原子重叠）",
                    ],
                    "critical_lines": ["FATAL: All pressures gave ZERO loading"],
                }
                result["error"] = (result.get("error") or "") + "\n" + (
                    "FATAL: All pressures gave ZERO loading (silent failure). "
                    "Simulation ran but nothing adsorbed."
                )
        except Exception:
            pass
    # RUNNING-progress signal: a RASPA job can be RUNNING at 100% CPU yet be
    # frozen (MC loop never advances). Observed repeatedly (MOF-74 2x2x2
    # charged/Ewald): squeue=RUNNING, run.log silent, .data stuck at the
    # initial energy header with 0 block averages. Enrich every non-terminal
    # check with objective output progress so agents stop guessing health
    # from log verbosity or CPU load.
    if not is_terminal and work_dir and not result["failed"]:
        try:
            _prog = _raspa_progress(work_dir)
            if _prog is not None:
                result["progress"] = _prog
        except Exception:
            pass
    return result


def _all_pressures_zero_loading(work_dir: str, host: str = "") -> Optional[bool]:
    """Return True if every RASPA pressure-point .data file reports ZERO
    loading, False if at least one is non-zero, None if no .data files found.

    Only counts a pressure point as a data file when it contains an
    'Average loading absolute' line with a parseable value, so this never
    mistakes a missing/empty run for 'zero loading'."""
    try:
        wd = Path(work_dir)
        data_files = sorted(wd.glob("P_*/Output/System_0/*.data"))
        if not data_files:
            return None
        found_any = False
        for d in data_files:
            try:
                text = d.read_text(errors="ignore")
            except OSError:
                continue
            m = None
            for ln in text.splitlines():
                if "Average loading absolute" in ln:
                    parts = ln.split()
                    # RASPA format: 'Average loading absolute  <val> +/- <err> [unit]'
                    # The numeric value is the 4th whitespace-separated field.
                    if len(parts) >= 4:
                        try:
                            val = float(parts[3])
                            m = val
                        except ValueError:
                            continue
                    if m is not None:
                        break
            if m is None:
                continue  # not a valid data point → ignore
            found_any = True
            if abs(m) > 1e-12:
                return False  # at least one point adsorbed
        if not found_any:
            return None
        return True  # all valid points are zero
    except Exception:
        return None


def _raspa_progress(work_dir: str) -> Optional[Dict[str, Any]]:
    """Probe the freshest RASPA .data file under work_dir and report output
    progress — the objective liveness signal for RUNNING jobs.

    RASPA appends one block-average section to Output/System_0/*.data every
    PrintEvery cycles (init + production alike), so data_lines grows
    monotonically while the MC loop advances. A frozen job leaves .data stuck
    at the initial energy header: block_averages == 0 and data_age_sec grows.

    Layout covered: single-point  <wd>/Output/System_0/*.data  and isotherm
    <wd>/P_*/Output/System_0/*.data. Returns None when no .data exists yet
    (job still in RASPA setup phase) or on error. Local reads only.
    """
    try:
        wd = Path(work_dir)
        files = []
        for pat in ("Output/System_0/*.data", "P_*/Output/System_0/*.data"):
            files.extend(wd.glob(pat))
        if not files:
            return None
        newest = max(files, key=lambda p: p.stat().st_mtime)
        st = newest.stat()
        text = newest.read_text(errors="ignore")
        lines = text.count("\n") + 1
        blocks = sum(1 for ln in text.splitlines()
                     if "Average loading absolute" in ln)
        age_sec = max(0.0, time.time() - st.st_mtime)
        return {
            "data_file": str(newest),
            "data_age_sec": int(age_sec),
            "data_lines": lines,
            "block_averages": blocks,
            # >15 min without completing even ONE block-average while RUNNING
            # (elapsed >> init PrintEvery) = strong freeze indicator.
            # A RUNNING RASPA job keeps appending block-averages to .data;
            # a frozen job stays parked at the initial energy header
            # (~<2000 lines, 0 completed PrintEvery blocks). >30 min without
            # growing past header scale while RUNNING => freeze indicator.
            "stalled": bool(age_sec > 1800 and lines < 2000),
        }
    except Exception:
        return None


def _read_slurm_stderr(job_id: str, host: str, work_dir: str = "") -> str:
    """Read the .err file tail for a job from the SLURM host (or local).

    host == "" or "local" → read locally (jobs submitted via local sbatch);
    otherwise → ssh to the remote host."""
    try:
        # Local read first if work_dir is local
        if work_dir and host in ("", "local"):
            for suffix in (f"_{job_id}.err", "*.err"):
                matches = list(Path(work_dir).glob(f"logs/*{suffix}"))
                matches += list(Path(work_dir).glob(f"logs/{job_id}.err"))
                if matches:
                    content = matches[0].read_text(errors="ignore")
                    return content[-1200:]
        # Remote read via ssh (job logs live on the host)
        if host and host != "local":
            proc = subprocess.run(
                ["ssh", host, f"cat {work_dir}/logs/*_{job_id}.err 2>/dev/null | tail -c 1200"],
                capture_output=True, text=True, timeout=15,
            )
            if proc.returncode == 0 and proc.stdout.strip():
                return proc.stdout.strip()[-1200:]
    except Exception:
        pass
    return ""


def diagnose_job(job_id: str, work_dir: str = "", host: str = "") -> Dict[str, Any]:
    """Diagnose a (possibly failed) SLURM job.

    Returns a structured report: state, terminal, failed, elapsed, plus
    stderr/stdout tails and any obvious error keywords. This is the agent's
    dedicated recovery tool — call it whenever a computation job fails.
    """
    st = check_job_status(job_id, host=host, work_dir=work_dir)
    report = dict(st)

    # Try to pull stdout + stderr tails
    try:
        cfg = get_config()
        runtime_path = cfg.project_root / "config" / "runtime.json"
        remote_cfg = {}
        if runtime_path.exists():
            with open(runtime_path) as f:
                remote_cfg = json.load(f).get("remote_slurm", {})
        host = host or remote_cfg.get("host", "gpu2")

        if work_dir:
            out = subprocess.run(
                ["ssh", host, f"tail -c 2000 {work_dir}/logs/*_{job_id}.out 2>/dev/null"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            err = subprocess.run(
                ["ssh", host, f"tail -c 2000 {work_dir}/logs/*_{job_id}.err 2>/dev/null"],
                capture_output=True, text=True, timeout=15,
            ).stdout.strip()
            if out:
                report["stdout_tail"] = out[-2000:]
            if err:
                report["stderr_tail"] = err[-2000:]
            # Also look for RASPA run.log in each pressure subdir
            try:
                runlog = subprocess.run(
                    ["ssh", host, f"tail -c 2500 {work_dir}/P_*/run.log 2>/dev/null | head -c 2500"],
                    capture_output=True, text=True, timeout=15,
                ).stdout.strip()
                if runlog:
                    report["run_log_tail"] = runlog[-2500:]
            except Exception:
                pass
    except Exception:
        pass

    # ── RASPA-aware diagnosis: match known software error signatures ──
    try:
        from .raspa_errors import summarize_run_log, match_raspa_error
        combined = " ".join([
            report.get("stderr_tail", ""),
            report.get("run_log_tail", ""),
            report.get("stdout_tail", ""),
            report.get("error", ""),
        ])
        ra = summarize_run_log(combined)
        if ra["raspa_error"]:
            report["diagnosis"] = {
                "kind": "raspa",
                "cause": ra["cause"],
                "fixes": ra["fixes"],
                "critical_lines": ra["critical_lines"],
            }
        else:
            # Generic keyword scan for non-RASPA issues
            entry = match_raspa_error(combined)
            if entry:
                report["diagnosis"] = {"kind": "raspa", "cause": entry["cause"], "fixes": entry["fixes"]}
            else:
                text = json.dumps(report, ensure_ascii=False)
                causes = []
                for kw in ["Error", "error", "FATAL", "Segmentation", "bus error",
                           "Cannot open", "not found", "No such file", "permission denied",
                           "memory", "MKL", "lib", "core dumped", "killed", "Killed",
                           "out of memory", "exit code", "returned non-zero"]:
                    if kw.lower() in text.lower():
                        causes.append(kw)
                if causes:
                    report["suspected_causes"] = causes[:8]
    except Exception:
        pass
    return report


def submit_and_wait(
    job_name: str,
    command: str,
    work_dir: str,
    timeout_minutes: int = 5,
    poll_interval: int = 5,
    mode: str = "local",
    **slurm_kwargs,
) -> Dict[str, Any]:
    """Submit a SLURM job, wait for completion, and return result.

    Returns dict with keys: submitted, job_id, status, elapsed, output, error.
    """
    # 防止递归调用 sbatch submit.sh 导致无限提交
    if "sbatch submit.sh" in command or "sbatch ./submit.sh" in command:
        return {
            "submitted": False,
            "failed": True,
            "status": "REJECTED",
            "error": "递归调用错误: command 包含 'sbatch submit.sh'，这会导致无限递归提交。"
                    "请使用正确的模拟命令，例如: run_sim simulation.input",
            "hint": "Henry系数计算应使用: run_henry 工具或 run_sim simulation.input",
            "work_dir": work_dir,
            "message": "❌ Job rejected: 递归调用错误，使用 run_henry 工具而不是手动创建 submit.sh",
        }

    result = submit_job(
        job_name=job_name,
        command=command,
        work_dir=work_dir,
        mode=mode,
        **slurm_kwargs,
    )

    if not result.get("submitted"):
        return result

    job_id = result.get("job_id", "")
    host = result.get("host", "")
    start_time = time.time()
    max_wait = timeout_minutes * 60
    terminal_states = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY"}

    # Quick initial check (job may fail immediately, e.g. command not found)
    time.sleep(2)
    status = check_job_status(job_id, host)
    st = status.get("status", "UNKNOWN")
    if st in terminal_states:
        result["status"] = st
        result["elapsed"] = status.get("elapsed", "")
        # Read output/error on immediate failure too
        _read_job_logs(result, job_name, job_id, work_dir)
        return result

    # Poll for completion
    while time.time() - start_time < max_wait:
        time.sleep(poll_interval)
        status = check_job_status(job_id, host)
        st = status.get("status", "UNKNOWN")

        if st in terminal_states:
            result["status"] = st
            result["elapsed"] = status.get("elapsed", "")
            break
    else:
        result["status"] = "TIMEOUT"
        result["error"] = f"Job {job_id} did not complete within {timeout_minutes} minutes"

    _read_job_logs(result, job_name, job_id, work_dir)
    return result


def _read_job_logs(result: Dict, job_name: str, job_id: str, work_dir: str):
    """Read SLURM output/error logs into result dict."""
    work_path = Path(work_dir)
    out_dir = work_path / "Output" / "System_0"
    if out_dir.exists():
        result["output_dir"] = str(out_dir)

    log_dir = work_path / "logs"
    if log_dir.exists():
        out_file = log_dir / f"{job_name}_{job_id}.out"
        err_file = log_dir / f"{job_name}_{job_id}.err"
        if out_file.exists():
            result["stdout"] = out_file.read_text()[-2000:]
        if err_file.exists():
            result["stderr"] = err_file.read_text()[-1000:]


def submit_and_return(
    job_name: str,
    command: str,
    work_dir: str,
    mode: str = "local",
    task_type: str = "default",  # 新增：任务类型，用于自动配置资源
    **slurm_kwargs,
) -> Dict[str, Any]:
    """Submit a SLURM job and return immediately with job ID (non-blocking).

    Use this for long-running computations (GCMC, cDFT, etc.) where the agent
    should not block waiting for completion. The user can check results later.

    Performs a short post-submission health check: if the job is REJECTED by
    the scheduler or crashes instantly (e.g. bad partition, missing executable),
    the result is marked failed=True with the stderr attached — so the agent
    can immediately invoke its fault-recovery path instead of waiting forever.

    资源管理策略：
    - 尽量使用完整节点（16核），避免碎片化
    - 根据任务类型自动调整资源和时间限制
    - 不要"东占一个节点西占一个节点"
    """
    # 防止递归调用 sbatch submit.sh 导致无限提交
    if "sbatch submit.sh" in command or "sbatch ./submit.sh" in command:
        return {
            "submitted": False,
            "failed": True,
            "status": "REJECTED",
            "error": "递归调用错误: command 包含 'sbatch submit.sh'，这会导致无限递归提交。"
                    "请使用正确的模拟命令，例如: run_sim simulation.input",
            "hint": "Henry系数计算应使用: run_henry 工具或 run_sim simulation.input",
            "work_dir": work_dir,
            "message": "❌ Job rejected: 递归调用错误，使用 run_henry 工具而不是手动创建 submit.sh",
        }

    # 根据任务类型自动配置资源
    if task_type != "default":
        resource_config = get_resource_config(task_type)
        # 将资源配置合并到slurm_kwargs
        for key, value in resource_config.items():
            if key not in slurm_kwargs:
                slurm_kwargs[key] = value

    result = submit_job(
        job_name=job_name,
        command=command,
        work_dir=work_dir,
        mode=mode,
        task_type=task_type,  # 传递task_type给render_sbatch_script
        **slurm_kwargs,
    )
    if not result.get("submitted"):
        result["failed"] = True
        result["status"] = "REJECTED"
        result["error"] = result.get("stderr") or result.get("error") or "sbatch submission failed"
        result["message"] = (
            f"❌ Job '{job_name}' was REJECTED by the scheduler: {result['error']}. "
            "Check the partition / resources requested and resubmit."
        )
        return result

    job_id = result.get("job_id", "")
    result["status"] = "SUBMITTED"

    # Short post-submission check: catch instant terminal failures (bad command,
    # partition limit, node down). A real run won't finish in <4s.
    try:
        time.sleep(3)
        st = check_job_status(job_id, work_dir=work_dir)
        st_state = st.get("status", "")
        if st.get("failed") or (st.get("terminal") and st_state not in ("COMPLETED", "")):
            result["status"] = st_state or "FAILED"
            result["failed"] = True
            err = st.get("error") or st.get("stderr_tail") or ""
            result["error"] = (
                err[-800:]
                if err else f"Job {job_id} entered terminal state '{st_state}' within seconds of submission."
            )
            result["message"] = (
                f"❌ Job '{job_name}' (ID {job_id}) FAILED immediately. "
                f"State: {st_state}. Reason: {result['error'][:300]}. "
                "Diagnose with check_job/diagnose_job, fix parameters, and resubmit."
            )
            return result
    except Exception:
        pass  # health check is best-effort; job may still be queued

    result["message"] = (
        f"✅ Job '{job_name}' submitted (ID: {job_id}). "
        f"Use check_job(job_id='{job_id}') to check status, "
        f"or inspect_run(run_dir='{work_dir}') to collect results when done."
    )
    return result


# ── GCMC-specific helpers ──────────────────────────────────────────

RASPA_SHARE = "/home/user/RASPA2/simulations/share/raspa"
# Project-LOCAL mirror of the RASPA force-field + molecule-definition trees.
# Everything GCMC needs is copied in here once; jobs then stage per-run LOCAL
# copies into their working directory (RASPA2 checks ./force_field.def,
# ./pseudo_atoms.def, ./force_field_mixing_rules.def, ./<gas>.def FIRST), so the
# external RASPA2 share under /home/user is NEVER modified by agent fixes.
LOCAL_RASPA_SHARE = str(Path(__file__).resolve().parents[1] / "forcefields" / "raspa")


def ensure_local_raspa_share() -> str:
    """Make sure the project-local FF mirror exists; sync incrementally from
    installed RASPA2 share (copies any subdirs/files missing locally).
    Returns the mirror path."""
    import shutil
    ff_mirror = Path(LOCAL_RASPA_SHARE) / "forcefield"
    mol_mirror = Path(LOCAL_RASPA_SHARE) / "molecules"
    ff_mirror.parent.mkdir(parents=True, exist_ok=True)

    # Incremental sync: for each top-level dir, copy missing subdirs/files
    for subdir_name in ("forcefield", "molecules"):
        src_dir = os.path.join(RASPA_SHARE, subdir_name)
        dst_dir = os.path.join(LOCAL_RASPA_SHARE, subdir_name)
        if not os.path.isdir(src_dir):
            continue
        if not os.path.isdir(dst_dir):
            shutil.copytree(src_dir, dst_dir)
        else:
            # Incremental: copy each entry that doesn't exist locally
            try:
                for entry in os.listdir(src_dir):
                    s = os.path.join(src_dir, entry)
                    d = os.path.join(dst_dir, entry)
                    if not os.path.exists(d):
                        if os.path.isdir(s):
                            shutil.copytree(s, d)
                        else:
                            shutil.copy2(s, d)
            except OSError:
                pass

    return LOCAL_RASPA_SHARE


def stage_raspa_local_files(
    work_dir: str,
    forcefield: str,
    mol_name: str,
    mol_def: str,
    cif_path: str = "",
) -> List[str]:
    """Python-side staging of LOCAL RASPA files into a run folder.

    Copies the force-field defs, the molecule .def and (optionally) the CIF
    into `work_dir` from the project-local mirror — so the agent can inspect /
    edit the copies BEFORE the job runs and RASPA resolves them from ./ at run
    time. The external RASPA2 share is never touched. Returns the staged file
    list.
    """
    import shutil
    ensure_local_raspa_share()
    os.makedirs(work_dir, exist_ok=True)
    staged: List[str] = []
    ff_src = os.path.join(LOCAL_RASPA_SHARE, "forcefield", forcefield)
    for fname in ("pseudo_atoms.def", "force_field.def", "force_field_mixing_rules.def"):
        src = os.path.join(ff_src, fname)
        if os.path.isfile(src):
            dst = os.path.join(work_dir, fname)
            try:
                shutil.copy2(src, dst)
                staged.append(dst)
            except OSError:
                pass
    mol_src = os.path.join(LOCAL_RASPA_SHARE, "molecules", mol_def, f"{mol_name}.def")
    if os.path.isfile(mol_src):
        dst = os.path.join(work_dir, f"{mol_name}.def")
        try:
            shutil.copy2(mol_src, dst)
            staged.append(dst)
        except OSError:
            pass
    if cif_path and os.path.isfile(cif_path):
        try:
            dst = os.path.join(work_dir, os.path.basename(cif_path))
            shutil.copy2(cif_path, dst)
            staged.append(dst)
        except OSError:
            pass
    return staged


def stage_raspa_local_shell(
    work_var: str = "WORK_DIR",
    ff_var: str = "FORCEFIELD",
    gas_var: str = "GAS",
    gasdef_var: str = "GAS_DEF",
) -> str:
    """Shell snippet that stages LOCAL force-field + molecule-definition copies
    into a job working directory (RASPA2 'local mode').

    RASPA2 resolves ./pseudo_atoms.def, ./force_field.def,
    ./force_field_mixing_rules.def and ./<gas>.def in the CWD FIRST and only
    falls back to $RASPA_DIR/share/raspa/... afterwards. By copying the files
    from the project-local mirror into the run dir we guarantee the agent's
    force-field fixes only ever touch per-job copies — never the external
    RASPA2 installation.
    """
    ensure_local_raspa_share()
    return "\n".join([
        f"  # Local-mode: stage force-field + molecule defs from project mirror",
        f"  cp \"{LOCAL_RASPA_SHARE}/forcefield/${ff_var}/pseudo_atoms.def\" \"${work_var}/\" 2>/dev/null || true",
        f"  cp \"{LOCAL_RASPA_SHARE}/forcefield/${ff_var}/force_field.def\" \"${work_var}/\" 2>/dev/null || true",
        f"  cp \"{LOCAL_RASPA_SHARE}/forcefield/${ff_var}/force_field_mixing_rules.def\" \"${work_var}/\" 2>/dev/null || true",
        f"  cp \"{LOCAL_RASPA_SHARE}/molecules/${gasdef_var}/${gas_var}.def\" \"${work_var}/\" 2>/dev/null || true",
    ])

# Per-gas molecule definition / force field / charge settings.
# CRITICAL: not every gas exists in every molecule-definition dir —
# TraPPE has CO2/N2/CO/Xe/Kr but NOT CH4/H2 (those live in
# ExampleDefinitions). Using the wrong definition = guaranteed job failure.
#
# forcefield must be one whose pseudo_atoms.def actually CONTAINS the
# gas's atom types, otherwise RASPA dies with "ReturnPseudoAtomNumber".
# Verified against the installed RASPA:
#   GenericMOFs  → CH4, CO2, N2, CO, H2, H2O, SO2  (has C_co2/O_co2/N_n2/
#                   N_com/C_CO/O_CO/O001/H002/S_so2/O_so2/H/CH4)
#   wbao         → Xe, Kr, O2                       (also has Ni_/framework;
#                   GenericMOFs lacks Xe/Kr/O_o2)
GAS_PRESETS = {
    "CH4": {"molecule_name": "methane", "molecule_definition": "ExampleDefinitions", "forcefield": "GenericMOFs", "use_charges": False},
    "CO2": {"molecule_name": "CO2", "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": True},
    "N2":  {"molecule_name": "N2", "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": True},
    "CO":  {"molecule_name": "CO", "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": False},
    "H2":  {"molecule_name": "H2", "molecule_definition": "ExampleDefinitions", "forcefield": "GenericMOFs", "use_charges": False},
    "Xe":  {"molecule_name": "Xe", "molecule_definition": "TraPPE", "forcefield": "wbao", "use_charges": False},
    "Kr":  {"molecule_name": "Kr", "molecule_definition": "TraPPE", "forcefield": "wbao", "use_charges": False},
    "H2O": {"molecule_name": "H2O", "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": False},
    "SO2": {"molecule_name": "SO2", "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": True},
    "O2":  {"molecule_name": "O2", "molecule_definition": "TraPPE", "forcefield": "wbao", "use_charges": False},
}
SUPPORTED_GASES = list(GAS_PRESETS.keys())

# Upper-case normalized lookup (Xe → XE, Kr → KR, co2 → CO2)
_GAS_PRESETS_UPPER = {k.upper(): dict(v) for k, v in GAS_PRESETS.items()}


def _gas_preset(gas: str) -> Dict[str, Any]:
    """Return the molecule preset for a gas (case-insensitive), or the raw
    name in TraPPE as a fallback so unknown gases still get a reasonable shot."""
    g = str(gas).upper()
    if g in _GAS_PRESETS_UPPER:
        return dict(_GAS_PRESETS_UPPER[g])
    # Unknown gas → assume same-name def in TraPPE + GenericMOFs
    return {"molecule_name": g, "molecule_definition": "TraPPE", "forcefield": "GenericMOFs", "use_charges": True}


def is_supported_gas(gas: str) -> bool:
    """Case-insensitive support check."""
    return str(gas).upper() in _GAS_PRESETS_UPPER


def _raspa_molecule_def(gas: str, molecule_def: str = "", molecule_name: str = "") -> str:
    """Path to the correct RASPA molecule definition file for a gas.

    Uses GAS_PRESETS so CH4 resolves to ExampleDefinitions/methane.def,
    CO2 to TraPPE/CO2.def, etc. (the mature mapping).
    """
    preset = _gas_preset(gas)
    md = molecule_def or preset["molecule_definition"]
    mn = molecule_name or preset["molecule_name"]
    return os.path.join(ensure_local_raspa_share(), "molecules", md, f"{mn}.def")


def _parse_molecule_atoms(def_path: str) -> List[str]:
    """Extract atom-type names from a RASPA .def molecule file.

    Format of the '# atomic positions' block:
        0 O_co2     0.0           0.0           1.16
        1 C_co2     0.0           0.0           0.0
        2 O_co2     0.0           0.0          -1.16
    Only lines inside that block (until the next '#' comment) are read.
    """
    atoms: List[str] = []
    try:
        lines = Path(def_path).read_text().splitlines()
        in_positions = False
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith("#"):
                if in_positions:
                    break  # positions block ended
                if "atomic positions" in ln.lower():
                    in_positions = True
                continue
            if not in_positions:
                continue
            parts = stripped.split()
            if len(parts) >= 2:
                try:
                    int(parts[0])  # must start with an integer index
                    atoms.append(parts[1])
                except ValueError:
                    break  # malformed → stop
    except Exception:
        pass
    return atoms


def _force_field_pseudo_atoms(force_field: str) -> List[str]:
    """Atom-type names defined in a force field's pseudo_atoms.def.

    Reads the project-LOCAL mirror (same content as the installed RASPA2, but
    self-contained and consistent with what jobs actually stage).
    """
    def_path = os.path.join(ensure_local_raspa_share(), "forcefield", force_field, "pseudo_atoms.def")
    try:
        if not os.path.exists(def_path):
            return []
        atoms = []
        for ln in Path(def_path).read_text().splitlines():
            parts = ln.split()
            if parts and not parts[0].startswith("#"):
                atoms.append(parts[0])
        return atoms
    except Exception:
        return []


def _validate_pressure_list(p_min: float, p_max: float, n_points: int) -> List[str]:
    """Sanity checks on the pressure grid."""
    errs = []
    if p_min <= 0 or p_max <= 0:
        errs.append(f"压力必须为正: p_min={p_min}, p_max={p_max}")
    if p_max < p_min:
        errs.append(f"压力范围无效: p_max ({p_max}) < p_min ({p_min})")
    if n_points < 1:
        errs.append(f"压力点数无效: {n_points} (需要 ≥1)")
    return errs


def _cif_structure_sanity(cif_path: str, min_ok: float = 1.0) -> Dict[str, Any]:
    """Detect physically-broken CIFs BEFORE wasting a SLURM slot.

    The failure we're guarding against (seen in real runs): a PACMAN-derived
    CIF whose atoms overlap (< 1 Å apart under PBC), which makes RASPA
    compute surface area = 0 and Rosenbluth factor = 0 — i.e. every insertion
    attempt fails, the job "completes" with all-zero loading, and nobody
    notices. Parsing the fractional coordinates and checking pairwise
    minimum-image distances catches this instantly.

    Returns {ok, min_distance, overlapping_pairs, errors}.
    """
    import math as _math

    try:
        lines = Path(cif_path).read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return {"ok": False, "min_distance": None, "overlapping_pairs": 0,
                "errors": [f"无法读取 CIF: {e}"]}

    cell = {}
    atoms = []
    for ln in lines:
        p = ln.split()
        if len(p) >= 2 and (p[0].startswith("_cell_length") or p[0].startswith("_cell_angle")):
            try:
                cell[p[0]] = float(p[1])
            except ValueError:
                pass
        # P1 atom rows: symbol label mult fx fy fz occ
        if len(p) == 7 and p[0] in ("Ni", "C", "O", "H", "N", "Zn", "Cu", "Co", "Fe", "Al", "Zr", "Mg", "Na", "K", "Ca", "Si", "P", "S", "Cl", "F", "B", "Cr", "Mn", "Mo", "Ti", "V", "W", "Li", "Be", "Sr", "Ba", "Pb", "Sn", "In", "Ga", "Ge", "Se", "Br", "I"):
            try:
                atoms.append((p[0], float(p[3]), float(p[4]), float(p[5])))
            except ValueError:
                pass

    a = cell.get("_cell_length_a")
    b = cell.get("_cell_length_b")
    c = cell.get("_cell_length_c")
    if not a or not b or not c or not atoms:
        # Can't check structure — don't block, just report unverifiable.
        return {"ok": True, "min_distance": None, "overlapping_pairs": 0,
                "errors": [], "note": "结构校验跳过（无法解析 CIF 晶胞/原子）"}

    # Generic triclinic lattice vectors from cell lengths + angles.
    alpha = _math.radians(cell.get("_cell_angle_alpha", 90.0))
    beta = _math.radians(cell.get("_cell_angle_beta", 90.0))
    gamma = _math.radians(cell.get("_cell_angle_gamma", 90.0))
    ca, cb, cg = _math.cos(alpha), _math.cos(beta), _math.cos(gamma)
    sg = _math.sin(gamma)
    va = [a, 0.0, 0.0]
    vb = [b * cg, b * sg, 0.0]
    # Standard triclinic: a along x, b in xy-plane, c general.
    vc = [c * cb,
          c * (ca - cb * cg) / sg,
          c * _math.sqrt(max(0.0, 1.0 - cb * cb - ((ca - cb * cg) / sg) ** 2))]

    cart = []
    for (_, x, y, z) in atoms:
        cart.append([
            x * va[0] + y * vb[0] + z * vc[0],
            x * va[1] + y * vb[1] + z * vc[1],
            x * va[2] + y * vb[2] + z * vc[2],
        ])
    # Precompute cell vectors for minimum-image.
    L = [va, vb, vc]

    def _min_image_dist(i: int, j: int) -> float:
        d = [cart[i][k] - cart[j][k] for k in range(3)]
        # Solve d = L^T f, wrap f to [-0.5,0.5], recompute.
        # Invert 3x3 (cell must be non-degenerate).
        m = [[L[0][0], L[1][0], L[2][0]],
             [L[0][1], L[1][1], L[2][1]],
             [L[0][2], L[1][2], L[2][2]]]
        det = (m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
               - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
               + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]))
        if abs(det) < 1e-12:
            return 1e9
        # inverse of m
        inv = [[(m[1][1] * m[2][2] - m[1][2] * m[2][1]) / det,
                (m[0][2] * m[2][1] - m[0][1] * m[2][2]) / det,
                (m[0][1] * m[1][2] - m[0][2] * m[1][1]) / det],
               [(m[1][2] * m[2][0] - m[1][0] * m[2][2]) / det,
                (m[0][0] * m[2][2] - m[0][2] * m[2][0]) / det,
                (m[0][2] * m[1][0] - m[0][0] * m[1][2]) / det],
               [(m[1][0] * m[2][1] - m[1][1] * m[2][0]) / det,
                (m[0][1] * m[2][0] - m[0][0] * m[2][1]) / det,
                (m[0][0] * m[1][1] - m[0][1] * m[1][0]) / det]]
        f = [inv[0][0] * d[0] + inv[0][1] * d[1] + inv[0][2] * d[2],
             inv[1][0] * d[0] + inv[1][1] * d[1] + inv[1][2] * d[2],
             inv[2][0] * d[0] + inv[2][1] * d[1] + inv[2][2] * d[2]]
        f = [x - round(x) for x in f]
        # d2 = L^T f
        d2 = [L[0][0] * f[0] + L[1][0] * f[1] + L[2][0] * f[2],
              L[0][1] * f[0] + L[1][1] * f[1] + L[2][1] * f[2],
              L[0][2] * f[0] + L[1][2] * f[1] + L[2][2] * f[2]]
        return _math.sqrt(d2[0] * d2[0] + d2[1] * d2[1] + d2[2] * d2[2])

    min_d = 1e9
    overlap = 0
    n = len(cart)
    for i in range(n):
        for j in range(i + 1, n):
            d = _min_image_dist(i, j)
            if d < min_d:
                min_d = d
            if d < min_ok:
                overlap += 1

    if min_d < min_ok:
        return {
            "ok": False,
            "min_distance": round(min_d, 3),
            "overlapping_pairs": overlap,
            "errors": [
                f"CIF 结构异常: 存在 {overlap} 对原子间距 < {min_ok} Å "
                f"(最近 {round(min_d, 3)} Å)。"
                "原子严重重叠 → RASPA 孔隙面积为 0、所有插入尝试失败、"
                "loading 恒为 0 但作业仍报'成功'。"
            ],
        }
    return {"ok": True, "min_distance": round(min_d, 3),
            "overlapping_pairs": overlap, "errors": []}




def _cif_has_charges(cif_path: str) -> bool:
    """Check if a CIF file has atomic charges (_atom_site_charge)."""
    try:
        text = Path(cif_path).read_text(encoding="utf-8")
        return "_atom_site_charge" in text
    except Exception:
        return False


def _find_charged_cif(cif_path: str) -> str:
    """Find a charged version of the CIF file.
    
    Searches for *_pacman.cif, *_pacman_pacman.cif, *_pacman_pacman_pacman.cif
    in the same directory and parent directories.
    Returns the path to the charged CIF, or original path if none found.
    """
    from pathlib import Path
    
    path = Path(cif_path)
    stem = path.stem
    parent = path.parent
    
    # Check if current file has charges
    if _cif_has_charges(cif_path):
        return cif_path
    
    # Search for charged versions in same directory
    patterns = [
        f"{stem}_pacman.cif",
        f"{stem}_pacman_pacman.cif", 
        f"{stem}_pacman_pacman_pacman.cif",
    ]
    
    for pattern in patterns:
        candidate = parent / pattern
        if candidate.exists() and _cif_has_charges(str(candidate)):
            return str(candidate)
    
    # Search in parent directory
    parent_parent = parent.parent
    if parent_parent.exists():
        for pattern in patterns:
            candidate = parent_parent / pattern
            if candidate.exists() and _cif_has_charges(str(candidate)):
                return str(candidate)
    
    # Search in cifs directory
    cifs_dir = Path("/home/user/gcmc_agent/cifs")
    if cifs_dir.exists():
        for cif_file in cifs_dir.rglob("*.cif"):
            if stem in cif_file.stem and _cif_has_charges(str(cif_file)):
                return str(cif_file)
    
    return cif_path

def validate_gcmc_inputs(
    cif_path: str,
    gas: str,
    temperature: float = 298.0,
    p_min: float = 0.1,
    p_max: float = 10.0,
    n_points: int = 10,
    n_cycles: int = 50000,
    force_field: str = "GenericMOFs",
    unit_cells: str = "2 2 2",
    molecule_def: str = "TraPPE",
) -> Dict[str, Any]:
    """PRE-SUBMISSION validation of a GCMC job.

    Catches jobs that are guaranteed to fail BEFORE wasting a SLURM slot:
      - missing CIF
      - gas molecule .def not installed (e.g. TraPPE/CH4.def absent)
      - force field not installed
      - force field missing the gas's pseudoatom parameters
      - invalid temperature / pressures / cycles / unit cells

    Returns {ok: bool, errors: [...], warnings: [...], fixes: {param: hint}}.
    """
    errors: List[str] = []
    warnings: List[str] = []
    fixes: Dict[str, str] = {}

    # 1. CIF
    if not cif_path:
        errors.append("缺少 CIF 文件路径 (cif 参数)")
        fixes["cif"] = "提供有效的 CIF 路径，如 'MOF-5_pacman.cif' 或绝对路径"
    elif not os.path.exists(cif_path):
        errors.append(f"CIF 文件不存在: {cif_path}")
        fixes["cif"] = "先用 find_cif 解析出正确的 CIF 路径"
    else:
        struct = _cif_structure_sanity(cif_path)
        if not struct["ok"]:
            errors.extend(struct["errors"])
            fixes["cif"] = (
                "CIF 结构损坏（原子重叠）会导致所有插入尝试失败、loading 恒为 0。"
                "请换用未损坏的 CIF（如 pacman_04201447_clean/ 下的版本），"
                "或检查 PACMAN 生成/后处理步骤。"
            )
        
        # Check if CIF has charges when gas requires them
        if gas:
            preset = _gas_preset(str(gas).upper())
            if preset.get("use_charges") and not _cif_has_charges(cif_path):
                # Try to find a charged version
                charged_cif = _find_charged_cif(cif_path)
                if charged_cif != cif_path:
                    warnings.append(
                        f"CIF 文件缺少电荷信息 (_atom_site_charge)，但 {gas} 需要电荷。"
                        f"找到带电荷的版本: {charged_cif}"
                    )
                else:
                    errors.append(
                        f"CIF 文件缺少电荷信息 (_atom_site_charge)，但 {gas} 需要电荷进行静电计算。"
                        f"请先运行 run_pacman_charge 为 CIF 添加电荷，或使用已带电荷的 CIF 文件。"
                    )
                    fixes["cif"] = (
                        "运行 run_pacman_charge(cif_dir=..., method='pacmof') 为 CIF 添加电荷，"
                        "或使用已带电荷的 CIF（如 *_pacman_pacman_pacman.cif）"
                    )

    # 2. Gas + molecule definition
    g_up = str(gas).upper() if gas else ""
    if g_up not in _GAS_PRESETS_UPPER:
        errors.append(f"不支持的气体: {gas!r}")
        fixes["gas"] = f"支持的气体: {', '.join(SUPPORTED_GASES)}"
    else:
        preset = _gas_preset(g_up)
        gas_def = _raspa_molecule_def(g_up, molecule_def=molecule_def)
        if not os.path.exists(gas_def):
            installed = []
            _local_share = ensure_local_raspa_share()
            mol_dir = os.path.join(_local_share, "molecules", preset["molecule_definition"])
            if os.path.isdir(mol_dir):
                installed = sorted(f.stem for f in Path(mol_dir).glob("*.def"))
            errors.append(
                f"分子定义文件不存在: {gas_def} "
                f"(本地 forcefields/raspa 缺少 {preset['molecule_definition']}/{preset['molecule_name']}.def)"
            )
            fixes["gas"] = (
                f"改用已安装的分子: {', '.join(installed[:12]) or '(无)'}，"
                f"或复制 {preset['molecule_name']}.def 到 {mol_dir}（不要改外部 RASPA2）"
            )
        # 3. Force field must have the gas's pseudoatoms. If none given,
        #    use the per-gas preset default (GenericMOFs, or wbao for Xe/Kr/O2).
        ff = force_field or preset.get("forcefield", "GenericMOFs")
        if ff:
            _local_share = ensure_local_raspa_share()
            ff_dir = os.path.join(_local_share, "forcefield", ff)
            if not os.path.isdir(ff_dir):
                installed_ff = []
                if os.path.isdir(os.path.join(_local_share, "forcefield")):
                    installed_ff = sorted(
                        d for d in os.listdir(os.path.join(_local_share, "forcefield"))
                        if os.path.isdir(os.path.join(_local_share, "forcefield", d))
                    )
                errors.append(f"力场不存在: {ff}")
                fixes["force_field"] = f"改用已安装力场: {', '.join(installed_ff) or '(无)'}"
            else:
                ff_atoms = _force_field_pseudo_atoms(ff)
                gas_atoms = _parse_molecule_atoms(gas_def)
                missing = [a for a in gas_atoms if a not in ff_atoms]
                if missing:
                    errors.append(
                        f"力场 '{ff}' 缺少客体分子的伪原子参数: {', '.join(missing)}"
                    )
                    fixes["force_field"] = (
                        f"气体 {gas!r} 的默认力场是 {preset.get('forcefield', 'GenericMOFs')}，"
                        f"若已指定力场请改用它；或在 {ff}/pseudo_atoms.def 中补充这些原子"
                    )

                # 3b. Check if force field has framework atom types from CIF
                # This catches cases like GenericMOFs missing Ni for Ni-MOF-74
                # NOTE: This is a WARNING, not an error. RASPA assigns zero LJ
                # parameters for unknown atoms → zero interactions → meaningless
                # results. But the job WILL still run and complete.
                if cif_path and os.path.exists(cif_path):
                    try:
                        with open(cif_path, 'r') as f:
                            cif_content = f.read()

                        # Extract unique atom types from CIF
                        import re

                        # CIF atom_site loop format:
                        # loop_
                        #   _atom_site_type_symbol
                        #   _atom_site_label
                        #   _atom_site_symmetry_multiplicity
                        #   _atom_site_fract_x
                        #   _atom_site_fract_y
                        #   _atom_site_fract_z
                        #   _atom_site_occupancy
                        #   Ni  Ni1       1.0  0.6484999999999999  0.6198  0.6452000000000001  1.0000

                        # Find the atom_site loop block
                        # Match: loop_\n  _atom_site_*\n  _atom_site_*\n ... data lines
                        loop_pattern = r'loop_\s*\n(?:\s*_atom_site_\w+\s*\n)+'
                        loop_match = re.search(loop_pattern, cif_content)

                        if loop_match:
                            # Get the data after the loop header
                            start_pos = loop_match.end()
                            data_lines = []
                            for line in cif_content[start_pos:].split('\n'):
                                line = line.strip()
                                if not line or line.startswith('#') or line.startswith('loop_') or line.startswith('_'):
                                    break
                                data_lines.append(line)

                            # Extract type_symbol (first column in each data line)
                            type_symbols = []
                            for line in data_lines:
                                parts = line.split()
                                if parts:
                                    # First column is type_symbol
                                    type_symbols.append(parts[0])

                            # Get unique framework atom types
                            framework_atoms = list(set(type_symbols))

                            # Check if force field has these atom types
                            # Metal atoms (Ni, Zn, Cu, etc.) are often missing from
                            # generic force fields - this is a WARNING, not an error
                            missing_framework = [a for a in framework_atoms if a not in ff_atoms]
                            if missing_framework:
                                # Separate metals from non-metals
                                metal_set = {'Ni', 'Zn', 'Cu', 'Fe', 'Co', 'Mn', 'Cr', 'V', 'Ti',
                                            'Al', 'Mg', 'Ca', 'Sr', 'Ba', 'Cd', 'Pt', 'Ag', 'Sc',
                                            'Nb', 'Mo', 'Ru', 'Rh', 'Pd', 'In', 'Sn', 'Sb', 'La',
                                            'Ce', 'Pr', 'Nd', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho',
                                            'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os',
                                            'Ir', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Zr'}
                                missing_metals = [a for a in missing_framework if a in metal_set]
                                missing_non_metals = [a for a in missing_framework if a not in metal_set]

                                if missing_metals:
                                    warnings.append(
                                        f"力场 '{ff}' 缺少金属原子参数: {', '.join(missing_metals)}。"
                                        f"RASPA将分配零LJ参数→零相互作用→loading可能为零。"
                                        f"建议使用UFF力场或自定义力场文件。"
                                    )
                                if missing_non_metals:
                                    warnings.append(
                                        f"力场 '{ff}' 缺少非金属原子参数: {', '.join(missing_non_metals)}。"
                                        f"这可能导致不准确的结果。"
                                    )
                    except Exception:
                        pass  # CIF parsing failed, skip framework atom check

    # 4. Temperature
    if temperature <= 0:
        errors.append(f"温度无效: {temperature} K")
        fixes["temperature"] = "提供正的温度值 (如 298)"

    # 5. Pressures
    errors += _validate_pressure_list(p_min, p_max, n_points)

    # 6. Cycles
    if n_cycles < 1000:
        warnings.append(f"MC 循环数偏少: {n_cycles} (统计可能不足，建议 ≥10000)")

    # 7. Unit cells
    try:
        ucs = [int(x) for x in unit_cells.split()]
        if len(ucs) == 3 and all(u > 0 for u in ucs):
            n_uc = ucs[0] * ucs[1] * ucs[2]
            if n_uc > 27:
                warnings.append(f"UnitCells 乘积 {n_uc} 较大，内存/时间开销高，建议 ≤27")

            # Check if unit cell size is sufficient for cutoff radius
            # RASPA requires: unit_cell_length * n_cells >= 2 * cutoff_radius
            # Default cutoff is12 Å, so minimum box size is24 Å
            cutoff_radius =12.0  # Default RASPA cutoff
            min_box_size =2 * cutoff_radius  #24 Å

            # Try to read CIF to get unit cell parameters
            if cif_path and os.path.exists(cif_path):
                try:
                    with open(cif_path, 'r') as f:
                        cif_content = f.read()

                    # Extract unit cell parameters from CIF
                    import re
                    a_match = re.search(r'_cell_length_a\s+([\d.]+)', cif_content)
                    b_match = re.search(r'_cell_length_b\s+([\d.]+)', cif_content)
                    c_match = re.search(r'_cell_length_c\s+([\d.]+)', cif_content)

                    if a_match and b_match and c_match:
                        a = float(a_match.group(1))
                        b = float(b_match.group(1))
                        c = float(c_match.group(1))

                        # Check each direction
                        box_a = a * ucs[0]
                        box_b = b * ucs[1]
                        box_c = c * ucs[2]

                        if box_a < min_box_size:
                            warnings.append(
                                f"a方向盒子尺寸不足: {a}×{ucs[0]}={box_a:.1f} Å < {min_box_size} Å (2×cutoff)。"
                                f"建议增加a方向unit cells至 ≥{int(min_box_size/a)+1}"
                            )
                        if box_b < min_box_size:
                            warnings.append(
                                f"b方向盒子尺寸不足: {b}×{ucs[1]}={box_b:.1f} Å < {min_box_size} Å (2×cutoff)。"
                                f"建议增加b方向unit cells至 ≥{int(min_box_size/b)+1}"
                            )
                        if box_c < min_box_size:
                            warnings.append(
                                f"c方向盒子尺寸不足: {c}×{ucs[2]}={box_c:.1f} Å < {min_box_size} Å (2×cutoff)。"
                                f"建议增加c方向unit cells至 ≥{int(min_box_size/c)+1}"
                            )
                except Exception:
                    pass  # CIF parsing failed, skip unit cell size check
        else:
            warnings.append(f"UnitCells 格式异常: '{unit_cells}' (期望如 '2 2 2')")
    except Exception:
        warnings.append(f"UnitCells 格式异常: '{unit_cells}' (期望如 '2 2 2')")

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "fixes": fixes,
    }


def _charge_method_lines(charge_method: str) -> List[str]:
    """Return the RASPA simulation.input lines for a charge method.

    Ewald -> classic Ewald summation (template default, unchanged behaviour);
    Wolf  -> Wolf summation, 12 A cutoff (matches CutOffVDW 12.0; cheaper for
             small supercells / metal-containing frameworks where Ewald can
             be overkill or finicky);
    None   -> no electrostatics block at all; the callers force
             UseChargesFromCIFFile=no so RASPA runs dispersion-only.
    """
    if charge_method == "Ewald":
        return [
            "ChargeMethod                  Ewald",
            "EwaldPrecision                1e-6",
        ]
    if charge_method == "Wolf":
        return [
            "ChargeMethod                  Wolf",
            "WolfCutoff                    12.0",
        ]
    if charge_method == "None":
        return []
    raise ValueError(f"unknown charge_method {charge_method!r} (expected Ewald/Wolf/None)")

def submit_gcmc_isotherm(
    cif_path: str,
    gas: str,
    temperature: float = 298.0,
    p_min: float = 0.1,
    p_max: float = 10.0,
    n_points: int = 10,
    n_cycles: int = 50000,
    force_field: str = "",
    guest_ff: str = "",
    unit_cells: str = "2 2 2",
    charge_method: str = "Ewald",  # Ewald | Wolf | None (None => dispersion-only)
    output_base: str = "",
    mode: str = "auto",
    **slurm_kwargs,
) -> Dict[str, Any]:
    """Submit a GCMC isotherm job via SLURM."""
    if charge_method not in ("Ewald", "Wolf", "None"):
        return {
            "submitted": False,
            "failed": True,
            "status": "INVALID_INPUTS",
            "error": f"charge_method \u4e0d\u6cd5: {charge_method!r} (\u53ef\u9009 'Ewald' \u9ed8\u8ba4 / 'Wolf' / 'None')",
            "message": "\u274c charge_method \u53c2\u6570\u975e\u6cd5\uff0c\u672a\u63d0\u4ea4\u4f5c\u4e1a\u3002\u53ef\u9009 'Ewald' / 'Wolf' / 'None'\u3002",
        }
    from .registry import _normalize_cif_for_raspa

    # ── PRE-SUBMISSION validation: catch guaranteed-to-fail jobs before
    # they waste a SLURM slot (missing CIF, uninstalled gas .def, force field
    # without the guest's pseudoatoms, bad pressures, etc.). ──
    validation = validate_gcmc_inputs(
        cif_path=cif_path, gas=gas, temperature=temperature,
        p_min=p_min, p_max=p_max, n_points=n_points,
        n_cycles=n_cycles, force_field=force_field, unit_cells=unit_cells,
        molecule_def=guest_ff,
    )
    if not validation["ok"]:
        return {
            "submitted": False,
            "failed": True,
            "status": "INVALID_INPUTS",
            "error": "参数校验失败，未提交作业:\n- " + "\n- ".join(validation["errors"]),
            "validation": validation,
            "message": (
                "❌ 提交前参数校验失败，未提交 SLURM 作业。"
                "请根据 errors/fixes 修正参数后重新提交。"
            ),
        }

    config = get_config()
    if not output_base:
        # Session-isolated workspace: GCMC outputs under runs/{user}/{conv}/gcmc
        try:
            from .workspace import session_dir
            output_base = str(session_dir("gcmc"))
        except Exception:
            output_base = str(config.project_root / "gcmc_output")

    # Normalize CIF (may be written to a one-shot temp dir that gets cleaned
    # before SLURM runs — so we stage it into the STABLE work_dir instead).
    normalized_cif = _normalize_cif_for_raspa(cif_path)
    cif_name = Path(cif_path).stem
    work_dir = os.path.join(output_base, gas, f"{temperature}K", cif_name)
    try:
        os.makedirs(work_dir, exist_ok=True)
    except OSError:
        pass
    stable_cif = os.path.join(work_dir, f"{cif_name}.cif")
    # Skip the copy when the (already-normalized) source IS the staged target
    # (common when the user passes a CIF that lives inside the work_dir RASPA
    # derives — copy2 to the same path raises OSError "Same file", which the
    # agent previously hit as CIF_STAGING_FAILED).
    if os.path.abspath(normalized_cif) != os.path.abspath(stable_cif):
        try:
            import shutil as _shutil
            _shutil.copy2(normalized_cif, stable_cif)
        except OSError as e:
            return {
                "submitted": False,
                "failed": True,
                "status": "CIF_STAGING_FAILED",
                "error": f"无法将归一化 CIF 拷入 work_dir: {e}",
                "message": "❌ CIF 暂存失败，未提交作业。请检查 CIF 文件是否可读。",
            }
    # Script references the stable staged copy (survives temp-dir cleanup).
    script_cif = stable_cif
    
    # Generate pressure points. Interface is in **bar**; RASPA's
    # `ExternalPressure` is in **Pa**, so convert bar → Pa (× 1e5) before
    # writing the simulation.input. Passing bar values straight through made
    # every run execute at ~1e-6 bar → zero loading while still reporting
    # "success" (exactly the silent-failure class this validator exists for).
    import numpy as np
    pressures = np.geomspace(p_min, p_max, n_points).tolist()
    pressures_pa = [p * 1e5 for p in pressures]

    # Resolve per-gas molecule name/definition/force field from the proven
    # GAS_PRESETS table (e.g. CH4 → ExampleDefinitions/methane, not TraPPE).
    preset = _gas_preset(gas)
    mol_name = preset["molecule_name"]
    mol_def = guest_ff or preset["molecule_definition"]  # user override wins
    use_charges = preset["use_charges"]
    # charge_method="None" => dispersion-only run: framework electrostatics
    # OFF even for charged presets (CO2/N2). No ChargeMethod block is emitted
    # and UseChargesFromCIFFile is forced to no.
    if charge_method == "None":
        use_charges = False
    # force_field override: user-specified wins, else the per-gas preset
    # default (GenericMOFs, or wbao for Xe/Kr/O2 which GenericMOFs lacks).
    ff = force_field or preset["forcefield"]

    # Pre-stage LOCAL FF + molecule .def + CIF into every pressure run dir at
    # SUBMISSION time (agent can inspect/edit copies before the job runs; the
    # sbatch script re-stages them too as a safety net). External RASPA2 share
    # is never touched.
    for _p in pressures_pa:
        _pdir = os.path.join(work_dir, "P_" + f"{_p:.2e}")
        try:
            stage_raspa_local_files(_pdir, ff, mol_name, mol_def, script_cif)
        except Exception as e:
            print(f"  ⚠️ pre-stage failed for {cif_name} P={_p:.2e}: {e}", flush=True)

    # Build RASPA simulation script
    raspa_dir = "/home/user/RASPA2/simulations"

    script_lines = [
        "#!/bin/bash",
        f"set -euo pipefail",
        f"RASPA_DIR={raspa_dir}",
        f"export LD_LIBRARY_PATH=$RASPA_DIR/lib:$LD_LIBRARY_PATH",
        "run_sim() {",
        "  if command -v stdbuf >/dev/null 2>&1; then",
        '    stdbuf -oL -eL "$RASPA_DIR/bin/simulate" "$@"',
        "  else",
        '    "$RASPA_DIR/bin/simulate" "$@"',
        "  fi",
        "}",
        f"CIF={script_cif}",
        f"CIF_NAME={cif_name}",
        f"GAS={mol_name}",
        f"GAS_DEF={mol_def}",
        f"FORCEFIELD={ff}",
        f"USE_CHARGES={'yes' if use_charges else 'no'}",
        f"TEMP={temperature}",
        f"NCYCLES={n_cycles}",
        f"UNIT_CELLS=({unit_cells})",
        f"OUT_BASE={output_base}",
        # WORK_BASE is the Python-side work_dir (gas-name path, e.g. .../CH4/298K/xxx)
        # — NOT $OUT_BASE/$GAS/... (which uses the RASPA molecule_name, e.g.
        # methane for CH4). The two differ whenever gas != molecule_name; using
        # the Python path keeps staging, the job, and JobWatch's work_dir in sync.
        f"WORK_BASE={work_dir}",
        "PRESSURE_LIST=" + '"' + " ".join(f"{p:.2e}" for p in pressures_pa) + '"',
        "PRESSURE_COUNT=" + str(len(pressures_pa)),
        "FAILED_PRESSURES=\"\"",
        "",
        "for PRESSURE in $PRESSURE_LIST; do",
        "  WORK_DIR=$WORK_BASE/P_$PRESSURE",
        "  mkdir -p $WORK_DIR/Output/System_0",
        "  cp $CIF $WORK_DIR/",
        # Local-mode: stage FF + molecule defs into the run dir (never the
        # external RASPA2 share — agent fixes only touch per-job copies).
        *stage_raspa_local_shell().split("\n"),
        "  cat > $WORK_DIR/simulation.input << SIMEOF",
        "SimulationType                MonteCarlo",
        "NumberOfCycles                $NCYCLES",
        "NumberOfInitializationCycles  $((NCYCLES * 2))",
        "PrintEvery                    1000",
        "RestartFile                   no",
        "",
        "Forcefield                    $FORCEFIELD",
        "UseChargesFromCIFFile         $USE_CHARGES",
        *_charge_method_lines(charge_method),
        "",
        "Framework 0",
        "FrameworkName $CIF_NAME",
        "CutOffVDW                     12.0",
        f"UnitCells                     {unit_cells}",
        "ExternalTemperature           $TEMP",
        "ExternalPressure              $PRESSURE",
        "",
        "Component 0 MoleculeName             $GAS",
        "            MoleculeDefinition       $GAS_DEF",
        "            TranslationProbability   0.5",
        "            RotationProbability      0.5",
        "            ReinsertionProbability   0.5",
        "            SwapProbability          1.0",
        "            CreateNumberOfMolecules  0",
        "SIMEOF",
        "  cd $WORK_DIR",
        "  # Capture RASPA failures instead of masking them",
        "  if ! run_sim simulation.input > run.log 2>&1; then",
        "    echo \"ERROR: RASPA simulation FAILED at P=$PRESSURE (exit $?)\" | tee -a run.log",
        "    FAILED_PRESSURES=\"$FAILED_PRESSURES $PRESSURE\"",
        "  fi",
        "done",
        "",
        "## ── Output validation: no valid loading data = silent failure → FAIL the job ──",
        "## Zero-loading across ALL points is ALSO a silent failure (e.g. wrong",
        "## pressure units gave 0.1 Pa instead of 0.1 bar → physically zero",
        "## adsorption while the job still 'completes'). Detect it here. ──",
        "VALID_COUNT=0",
        "NONZERO_COUNT=0",
        "ZERO_SUMMARY=\"\"",
        "for P in $PRESSURE_LIST; do",
        "  W=$WORK_BASE/P_$P/Output/System_0",
        "  D=$(ls $W/*.data 2>/dev/null | head -1)",
        "  if [ -n \"$D\" ] && grep -q 'Average loading absolute' \"$D\" 2>/dev/null; then",
        "    VALID_COUNT=$((VALID_COUNT + 1))",
        "    if awk '/Average loading absolute/{v=$4+0; exit (v!=0 ? 0 : 1)}' \"$D\" 2>/dev/null; then",
        "      NONZERO_COUNT=$((NONZERO_COUNT + 1))",
        "    else",
        "      ZERO_SUMMARY=\"$ZERO_SUMMARY $P\"",
        "    fi",
        "  fi",
        "done",
        "if [ \"$VALID_COUNT\" = \"0\" ]; then",
        "  echo \"FATAL: No pressure point produced valid GCMC loading data (${VALID_COUNT}/${PRESSURE_COUNT}).\" >&2",
        "  echo \"GCMC simulation failed - check run.log files under $OUT_BASE/$GAS/${TEMP}K/$CIF_NAME\" >&2",
        "  exit 1",
        "fi",
        "if [ \"$NONZERO_COUNT\" = \"0\" ]; then",
        "  echo \"FATAL: All ${PRESSURE_COUNT} pressures gave ZERO loading (${VALID_COUNT} data files present).\" >&2",
        "  echo \"This is a silent failure - the simulation ran but nothing adsorbed. Check for:\" >&2",
        "  echo \"  - pressure units (RASPA ExternalPressure must be in Pa, not bar)\" >&2",
        "  echo \"  - forcefield/pseudo-atom mismatch for $GAS\" >&2",
        "  echo \"  - framework pore accessibility (overlapping atoms)\" >&2",
        "  echo \"Offending pressures:$ZERO_SUMMARY\" >&2",
        "  exit 1",
        "fi",
        "if [ -n \"$FAILED_PRESSURES\" ]; then",
        "  echo \"WARNING: $VALID_COUNT/${PRESSURE_COUNT} pressures OK; failed:$FAILED_PRESSURES\"",
        "fi",
        "if [ -n \"$ZERO_SUMMARY\" ]; then",
        "  echo \"WARNING: $NONZERO_COUNT/${PRESSURE_COUNT} pressures have non-zero loading; zero-loading pressures:$ZERO_SUMMARY\"",
        "fi",
        "",
        "## -- Magnitude-type silent-failure gate: framework charges requested from CIF --",
        "## (UseChargesFromCIFFile=yes) but the CIF has no partial-charge column: RASPA marks",
        "## each framework pseudo-atom '(charge definition not found)' in the .data file and",
        "## silently assigns q=0 -> electrostatics OFF -> loading can be 20-50x too low for",
        "## polar/OMS frameworks (CO2@M-MOF-74 at 0.1 bar: ~0.1 mol/kg instead of ~2.5+),",
        "## yet still non-zero, so the zero-loading gate above cannot catch it.",
        "CHARGE_MISSING_SUMMARY=\"\"",
        "LOWLOAD_SUMMARY=\"\"",
        "for P in $PRESSURE_LIST; do",
        "  W=$WORK_BASE/P_$P/Output/System_0",
        "  D=$(ls $W/*.data 2>/dev/null | head -1 || true)",
        "  if [ -n \"$D\" ] && grep -q 'Average loading absolute' \"$D\" 2>/dev/null && grep -q 'charge definition not found' \"$D\" 2>/dev/null; then",
        "    CHARGE_MISSING_SUMMARY=\"$CHARGE_MISSING_SUMMARY $P\"",
        "  fi",
        "  if [ -n \"$D\" ] && [ \"$USE_CHARGES\" = \"yes\" ] && echo \"$CIF_NAME\" | grep -qiE 'MOF-74|CPO-27' && [ \"$GAS\" = \"CO2\" ]; then",
        "    MK=$(awk 'index($0,\"[mol/kg framework]\"){print $6+0; exit}' \"$D\" 2>/dev/null || true)",
        "    if [ -n \"$MK\" ] && awk -v p=\"$P\" -v mk=\"$MK\" 'BEGIN{exit !(p<=20000 && mk<0.5)}' 2>/dev/null; then",
        "      LOWLOAD_SUMMARY=\"$LOWLOAD_SUMMARY $P(${MK}mol/kg)\"",
        "    fi",
        "  fi",
        "done",
        "if [ \"$USE_CHARGES\" = \"yes\" ] && [ -n \"$CHARGE_MISSING_SUMMARY\" ]; then",
        "  echo \"FATAL: framework atom charge definitions NOT FOUND in CIF at pressures:$CHARGE_MISSING_SUMMARY\" >&2",
        "  echo \"  Cause: UseChargesFromCIFFile=yes but CIF has no partial-charge column;\" >&2",
        "  echo \"         every framework atom got q=0.0 -> electrostatics silently disabled.\" >&2",
        "  echo \"         Non-zero loading on polar/OMS frameworks is NOT physically meaningful.\" >&2",
        "  echo \"  Fix: add partial charges (QEq/DDEC) to the CIF, or rerun with use_charges=no\" >&2",
        "  echo \"       ONLY if the model is intentionally uncharged.\" >&2",
        "  exit 1",
        "fi",
        "if [ -n \"$LOWLOAD_SUMMARY\" ]; then",
        "  echo \"WARNING: CO2 loading on MOF-74/CPO-27 at <=0.2 bar implausibly low (<0.5 mol/kg):$LOWLOAD_SUMMARY\"",
        "fi",
        "echo 'GCMC isotherm job completed'",
        # ── Success indicator (ai2-kit absorb): written ONLY after all
        # validation passed, so a job that later vanishes from the scheduler
        # (gpu2 has no accounting) can be proven to have truly COMPLETED via
        # `test -f $WORK_BASE/job.done` — vs. a vanished job that failed. ──
        "echo \"$SLURM_JOB_ID\" > \"$WORK_BASE/job.done\"",
    ]
    
    command = "\n".join(script_lines)
    job_name = slurm_kwargs.pop("job_name", f"gcmc_{cif_name}_{gas}")

    result = submit_and_return(
        job_name=job_name,
        command=command,
        work_dir=work_dir,
        mode=mode,
        partition=slurm_kwargs.pop("partition", "compute"),
        cpus_per_task=slurm_kwargs.pop("cpus_per_task", 4),
        walltime=slurm_kwargs.pop("walltime", "04:00:00"),
        **slurm_kwargs,
    )

    # Add helpful info for the agent
    if result.get("submitted"):
        result["status"] = "SUBMITTED"
        result["work_dir"] = os.path.join(output_base, gas, f"{temperature}K", cif_name)
        result["note"] = (
            f"GCMC isotherm job submitted for {cif_name} + {gas} at {temperature}K. "
            f"Job ID: {result.get('job_id', '?')}. "
            f"Results will be in: {result['work_dir']}. "
            f"Use check_job(job_id='{result.get('job_id', '')}', work_dir='{result['work_dir']}') "
            f"to monitor, or diagnose_job(...) if it fails."
        )

        # Try to parse existing results (in case job already completed from a previous run)
        result["isotherm_data"] = []
        base_path = Path(output_base) / gas / f"{temperature}K" / cif_name
        for p_dir in sorted(base_path.glob("P_*")):
            out_dir = p_dir / "Output" / "System_0"
            if out_dir.exists():
                for f in out_dir.glob("*.data"):
                    content = f.read_text()
                    pressure = p_dir.name.replace("P_", "")
                    for line in content.split("\n"):
                        if "Average loading absolute" in line or "Loading" in line:
                            result["isotherm_data"].append({
                                "pressure": pressure,
                                "loading": line.strip(),
                            })

    return result


def submit_gcmc_batch(
    cif_dir: str,
    gases: List[str] = ("CO2",),
    temperature: float = 298.0,
    pressure: float = 1.0,
    n_cycles: int = 10000,
    forcefield: str = "",
    mode: str = "local",
    unit_cells: str = "2 2 2",
    charge_method: str = "Ewald",  # Ewald | Wolf | None (None => dispersion-only)
    output_dir: str = "",
    **slurm_kwargs,
) -> Dict[str, Any]:
    """Submit ONE SLURM job that screens a whole CIF directory × gases.

    Unlike the old `_exec_gcmc_batch` (which submitted N separate jobs —
    exactly the waste the user complained about), this builds a single
    sbatch script that loops over every CIF × gas and runs RASPA at a
    single pressure per material. One job ID and one queue entry; independent
    material/gas branches run concurrently up to SLURM_CPUS_PER_TASK.

    Returns:
        dict with submitted flag, job_id, and the list of (cif, gas) pairs.
    """
    config = get_config()
    try:
        ucs = [int(x) for x in str(unit_cells).split()]
        assert len(ucs) == 3 and all(u > 0 for u in ucs)
    except Exception:
        return {"submitted": False, "failed": True,
                "error": f"UnitCells 格式异常: '{unit_cells}' (期望如 '2 2 2', 正整数x3)"}
    if charge_method not in ("Ewald", "Wolf", "None"):
        return {"submitted": False, "failed": True,
                "error": f"charge_method 非法: '{charge_method}' (可选 'Ewald' 默认 / 'Wolf' / 'None')"}
    cifs = sorted(str(f) for f in Path(cif_dir).glob("*.cif"))
    if not cifs:
        return {"submitted": False, "failed": True, "error": f"No CIF files found in {cif_dir}"}

    raspa_dir = "/home/user/RASPA2/simulations"
    if output_dir:
        work_dir = str(Path(output_dir).resolve())
    else:
        try:
            from .workspace import session_dir
            work_dir = str(session_dir("gcmc") / "batch")
        except Exception:
            work_dir = str(config.project_root / "gcmc_output" / "batch")

    # A completed/scientifically populated branch is immutable unless a caller
    # explicitly chooses a new branch path.  This prevents retries or parallel
    # variants from silently mixing evidence in one folder.
    work_path = Path(work_dir)
    if work_path.is_dir():
        existing_result = (work_path / "job.done").is_file()
        if not existing_result:
            for index, candidate in enumerate(work_path.rglob("*.data")):
                if index >= 10_000:
                    break
                if "Output" in candidate.parts:
                    existing_result = True
                    break
        if existing_result:
            return {
                "submitted": False,
                "blocked": True,
                "status": "EXISTING_RESULTS_REQUIRE_REUSE",
                "error_code": "EXISTING_RESULTS_REQUIRE_REUSE",
                "error": "The output branch already contains scientific results; reuse it or select a new output_dir.",
                "work_dir": work_dir,
            }
    p_pa = float(pressure) * 1e5  # bar → Pa

    # Resolve per (cif, gas) settings now (Python side, validated once).
    # forcefield override applies to every (cif, gas) in the batch; otherwise
    # the per-gas preset default (GenericMOFs / wbao) is used.
    jobs = []  # (cif_abs, cif_name, gas_label, mol_name, mol_def, ff, use_charges)
    for cif in cifs:
        cif_name = Path(cif).stem
        for gas in gases:
            preset = _gas_preset(gas)
            jobs.append((
                cif, cif_name, gas,
                preset["molecule_name"], preset["molecule_definition"],
                forcefield or preset["forcefield"],
                preset["use_charges"] and charge_method != "None",
            ))

    # Pre-stage LOCAL FF + molecule .def + CIF into each (cif, gas) run dir at
    # SUBMISSION time, so the agent can inspect/edit the copies before the job
    # runs. The sbatch script re-stages them too (idempotent) as a safety net.
    staged_note = []
    for cif, cif_name, gas, mol_name, mol_def, ff, use_charges in jobs:
        out_dir = os.path.join(work_dir, gas, f"{temperature}K", cif_name)
        try:
            stage_raspa_local_files(out_dir, ff, mol_name, mol_def, cif)
            staged_note.append(cif_name)
        except Exception as e:
            print(f"  ⚠️ pre-stage failed for {cif_name}+{gas}: {e}", flush=True)

    # One self-contained run_one() function; one call block per (cif, gas).
    run_body = []
    for cif, cif_name, gas, mol_name, mol_def, ff, use_charges in jobs:
        out_dir = os.path.join(work_dir, gas, f"{temperature}K", cif_name)
        run_body.append(
            f"launch run_one {cif_name} {gas} '{out_dir}' '{cif}' {mol_name} {mol_def} "
            f"{ff} {use_charges} {temperature} {p_pa:.6e} {n_cycles} {unit_cells}"
        )

    command = "\n".join([
        "#!/bin/bash",
        "set -euo pipefail",
        f"RASPA_DIR={raspa_dir}",
        "export LD_LIBRARY_PATH=$RASPA_DIR/lib:$LD_LIBRARY_PATH",
        "run_sim() {",
        "  if command -v stdbuf >/dev/null 2>&1; then",
        '    stdbuf -oL -eL "$RASPA_DIR/bin/simulate" "$@"',
        "  else",
        '    "$RASPA_DIR/bin/simulate" "$@"',
        "  fi",
        "}",
        f"mkdir -p {work_dir}",
        "run_one() {",
        "  CIF_NAME=\"$1\"; GAS_LABEL=\"$2\"; W=\"$3\"; CIF=\"$4\"",
        "  GAS=\"$5\"; GAS_DEF=\"$6\"; FORCEFIELD=\"$7\"; USE_CHARGES=\"$8\"",
        "  TEMP=\"$9\"; PRESSURE=\"${10}\"; NCYCLES=\"${11}\"; UC=\"${12}\"",
        "  mkdir -p \"$W/Output/System_0\"",
        "  cp \"$CIF\" \"$W/\" 2>/dev/null || true",
        # Local-mode: stage FF + molecule defs into the run dir (never the
        # external RASPA2 share — agent fixes only touch per-job copies).
        *stage_raspa_local_shell(work_var="W", ff_var="FORCEFIELD", gas_var="GAS", gasdef_var="GAS_DEF").split("\n"),
        "  cat > \"$W/simulation.input\" << SIMEOF",
        "SimulationType                MonteCarlo",
        "NumberOfCycles                $NCYCLES",
        "NumberOfInitializationCycles  $((NCYCLES * 2))",
        "PrintEvery                    1000",
        "RestartFile                   no",
        "Forcefield                    $FORCEFIELD",
        "UseChargesFromCIFFile         $USE_CHARGES",
        *_charge_method_lines(charge_method),
        "Framework 0",
        "FrameworkName $CIF_NAME",
        "CutOffVDW                     12.0",
        "UnitCells                     $UC",
        "ExternalTemperature           $TEMP",
        "ExternalPressure              $PRESSURE",
        "Component 0 MoleculeName             $GAS",
        "            MoleculeDefinition       $GAS_DEF",
        "            TranslationProbability   0.5",
        "            RotationProbability      0.5",
        "            ReinsertionProbability   0.5",
        "            SwapProbability          1.0",
        "            CreateNumberOfMolecules  0",
        "SIMEOF",
        "  ( cd \"$W\" && run_sim simulation.input > run.log 2>&1 ) || {",
        "    echo \"ERROR: RASPA failed for $CIF_NAME + $GAS_LABEL\" | tee -a \"$W/run.log\"",
        "    return 1",
        "  }",
        "}",
        "MAX_PARALLEL=${SLURM_CPUS_PER_TASK:-1}",
        "PIDS=()",
        "FAILURES=0",
        "launch() {",
        "  \"$@\" &",
        "  PIDS+=(\"$!\")",
        "  if (( ${#PIDS[@]} >= MAX_PARALLEL )); then",
        "    wait \"${PIDS[0]}\" || FAILURES=$((FAILURES + 1))",
        "    PIDS=(\"${PIDS[@]:1}\")",
        "  fi",
        "}",
        "",
        "\n".join(run_body),
        "for pid in \"${PIDS[@]}\"; do wait \"$pid\" || FAILURES=$((FAILURES + 1)); done",
        "if (( FAILURES > 0 )); then echo \"ERROR: $FAILURES GCMC branch(es) failed\"; exit 1; fi",
        "",
        f"echo 'GCMC batch job completed for {len(cifs)} CIFs x {len(gases)} gases'",
        # Success indicator (ai2-kit absorb): proves real completion for jobs
        # that later vanish from the scheduler (gpu2 disables accounting).
        f"echo \"$SLURM_JOB_ID\" > \"{work_dir}/job.done\"",
    ])

    safe_branch = "".join(c if c.isalnum() or c in "_-" else "_" for c in work_path.name)[:32]
    job_name = slurm_kwargs.pop("job_name", f"gcmc_{safe_branch or 'batch'}")
    result = submit_and_return(
        job_name=job_name,
        command=command,
        work_dir=work_dir,
        mode=mode,
        partition=slurm_kwargs.pop("partition", "compute"),
        cpus_per_task=slurm_kwargs.pop("cpus_per_task", 4),
        walltime=slurm_kwargs.pop("walltime", "08:00:00"),
        **slurm_kwargs,
    )
    if result.get("submitted"):
        result["status"] = "SUBMITTED"
        result["work_dir"] = work_dir
        result["pairs"] = [{"cif": c, "gas": g} for _, c, g, *_ in jobs]
        result["note"] = (
            f"GCMC batch job submitted (ONE job ID: {result.get('job_id', '?')}) for "
            f"{len(cifs)} CIFs × {len(gases)} gases at {temperature}K, {pressure} bar. "
            f"Results under {work_dir}. Use check_job(job_id='{result.get('job_id', '')}', work_dir='{work_dir}') to monitor."
        )
    return result
