"""Opt-in production Session: real GCMC submission, observation and native parsing.

No scheduler or scientific output is mocked. The model chooses registered
tools/arguments from a scientific user problem. The harness restricts scope and
budget, records receipts, and independently checks the result after execution.
"""
from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from test_sdk_production_collaboration import save


def hashes():
    paths = sorted((ROOT / "agents").glob("*.py")) + [Path(__file__).resolve()]
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-hpc", action="store_true", required=True)
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--model-calls", type=int, default=24)
    args = parser.parse_args()
    if not 300 <= args.timeout <= 3600 or not 8 <= args.model_calls <= 30:
        parser.error("timeout must be 300-3600 and model-calls 8-30")
    from agents.config import AgentConfig
    from agents.defns import ORCHESTRATOR
    from agents.job_watch import JobWatch
    from agents.parallel_workflow import READ_ONLY
    from agents.registry import _build_default_registry
    from agents.session import ExecutionBudgetExceeded, Session
    from agents.slurm import check_job_status
    from agents.task_line import TaskLineStore
    from agents.watch_context import set_context, clear_context
    from agents.workspace import PATH_ARGUMENTS, readonly_shell

    user = "sdk_real_science"
    cid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    root = ROOT / "runs" / user / cid
    root.mkdir(parents=True)
    os.chmod(root, 0o700)
    config = AgentConfig(api_key="test-unused-local-codex-provider", model_provider="codex_cli", max_tokens=4096)
    registry = _build_default_registry()
    watch = JobWatch(str(root / "job_watch.json"))
    lines = TaskLineStore(str(root / "task_lines.json"))
    lines.begin_line(cid, username=user, conv_id=cid)
    session = Session(config=config, registry=registry)
    session._current_line_id = cid
    session._evidence_root = root / "evidence"
    session.recovery_gate.path = root / "recovery.json"
    session._on_checkpoint = lambda state, reason: save(root / "checkpoint.json", state)
    started = time.monotonic()
    frozen = hashes()
    calls, submissions, observations, blocked, answers, scheduler_receipts = [], [], [], [], [], []
    session._on_interrupt_requested = lambda: time.monotonic() - started > args.timeout
    allowed = set(READ_ONLY) | {
        "handoff_to_adsorption", "handoff_to_analyst", "handoff_to_harness", "handoff_to_monitor",
        "run_gcmc_isotherm", "run_gcmc_batch", "resource_health", "get_tool_schema", "find_cif",
        "inspect_path", "inspect_run", "read_file", "grep_search", "check_job", "diagnose_job",
        "list_my_jobs", "query_literature", "task_line_query", "task_line_update",
        "recovery_state", "prepare_retry", "accept_recovered_result",
        "lifecycle_state", "run_bash",
        "stage_cif_subset", "write_file", "validate_gcmc_results",
        "validate_method", "verify_data_authenticity", "check_conclusion_reliability",
    }

    def observe(item):
        status = check_job_status(item["job_id"], work_dir=item["work_dir"])
        observations.append({"job_id": item["job_id"], "elapsed_s": time.monotonic() - started, "status": status})
        save(root / "scheduler_observations.json", observations)
        if status.get("terminal"):
            capture_scheduler(item["job_id"])
        return status

    def capture_scheduler(job_id):
        proc = subprocess.run(["scontrol", "show", "job", job_id, "-o"], capture_output=True, text=True, timeout=15)
        fields = dict(re.findall(r"(?:^|\s)(\w+)=(\S+)", proc.stdout))
        scheduler_receipts.append({"job_id": job_id, "returncode": proc.returncode,
                                   "stdout": proc.stdout, "stderr": proc.stderr,
                                   "node": fields.get("NodeList"), "state": fields.get("JobState"),
                                   "exit_code": fields.get("ExitCode"), "elapsed_s": time.monotonic() - started})
        save(root / "scheduler_receipts.json", scheduler_receipts)

    def guard(name, params):
        reason = ""
        if name not in allowed:
            reason = "This small real smoke permits registered single-gas GCMC, evidence reading and scoped coordination only."
        if name == "run_bash":
            command = str(params.get("command", ""))
            try:
                words = shlex.split(command)
            except ValueError:
                words = []
            executable = Path(words[0]).name if words else ""
            resource_command = executable == "sinfo" or (executable == "scontrol" and words[1:3] in
                                                          (["show", "nodes"], ["show", "node"], ["show", "partitions"]))
            safe_resource = resource_command and not any(x in command for x in ("$", "`", "\n", ";", ">", "<", "&&", "||")) and "|" not in words
            if not (readonly_shell(command) or safe_resource):
                reason = "Only simple read-only scheduler resource commands are allowed; simulations still use native registered tools."
        for key in PATH_ARGUMENTS:
            if params.get(key):
                p = Path(str(params[key])).expanduser()
                p = p.resolve() if p.is_absolute() else (ROOT / p).resolve()
                write_path = key in {"output_dir", "output_csv", "output_markdown", "output", "output_path"} or name == "write_file" and key == "path"
                if write_path and not p.is_relative_to(root):
                    reason = "All evidence writes and staged copies must stay in this private smoke workspace."
                read_roots = (root, ROOT.parent / "cifs", ROOT / "forcefields", ROOT / ".claude" / "skills",
                              ROOT / "agents", ROOT / "tools", ROOT / "scripts", ROOT / "docs")
                if not any(p.is_relative_to(base) for base in read_roots) and not (name == "inspect_path" and p == ROOT):
                    reason = "Use own-session artifacts, shared scientific inputs, or project source/skills for diagnosis."
        if name in {"run_gcmc_isotherm", "run_gcmc_batch"}:
            gas_ok = params.get("gas") == "CH4" if name == "run_gcmc_isotherm" else params.get("gases") == ["CH4"]
            pressure_ok = (params.get("pressure_start") == 1 and params.get("pressure_end") == 1 and params.get("n_pressure_points", 10) == 1
                           if name == "run_gcmc_isotherm" else params.get("pressure") == 1)
            if not gas_ok or params.get("temperature") != 298:
                reason = "The approved scientific smoke is methane at 298 K."
            elif not pressure_ok:
                reason = "The approved scientific smoke is exactly one pressure point at 1 bar."
            elif name == "run_gcmc_batch" and len(list(Path(params.get("cif_dir", "")).glob("*.cif"))) != 1:
                reason = "Stage exactly one existing MOF-5 CIF in a new own-session subset for a single-structure batch."
            elif not isinstance(params.get("cycles"), int) or not 1 <= params["cycles"] <= 1000:
                reason = "Explicitly choose at most 1000 production cycles for this non-convergence smoke."
            elif len(submissions) >= 2:
                reason = "One initial scheduler job plus one evidence-backed failure repair is the complete smoke budget."
            elif submissions:
                status = observe(submissions[-1])
                if not (status.get("terminal") and status.get("failed")):
                    reason = "Existing job is live or successful: observe and parse it; do not duplicate it."
        if name in {"check_job", "diagnose_job"} and params.get("job_id") and str(params["job_id"]) not in {s["job_id"] for s in submissions}:
            reason = "Only jobs submitted by this smoke belong to this task."
        if reason:
            blocked.append({"tool": name, "params": params, "reason": reason})
            save(root / "blocked_calls.json", blocked)
        return reason

    session._on_tool_guard = guard
    def recorder(name, native):
        def record_submission(params):
            result = native(params)
            if isinstance(result, dict) and result.get("submitted") and result.get("job_id"):
                submissions.append({"job_id": str(result["job_id"]), "work_dir": result["work_dir"],
                                    "tool": name, "params": params, "receipt": result, "submitted_s": time.monotonic() - started})
                save(root / "submissions.json", submissions)
            return result
        return record_submission

    for name in ("run_gcmc_isotherm", "run_gcmc_batch"):
        registry.get(name).execute = recorder(name, registry.get(name).execute)
    original_call = session._call_api

    def bounded(agent):
        if len(calls) >= args.model_calls:
            raise ExecutionBudgetExceeded("real scientific smoke model budget reached")
        item = {"agent": agent.name, "started_s": time.monotonic() - started}
        calls.append(item)
        save(root / "model_calls.json", calls)
        response = original_call(agent)
        item.update(response=session._json_safe(response.content), finished_s=time.monotonic() - started)
        save(root / "model_calls.json", calls)
        return response

    session._call_api = bounded
    prompt = (
        "直接自主完成一个真实科学计算冒烟，不需要再次确认：用项目已有 MOF-5 结构做 CH4 在 298 K、1 bar 的单点 GCMC 吸附检查。"
        "这是验证真实计算提交、运行和原生结果解析的短测试，不用于发表或收敛精度结论。仅一个压力点，生产循环数由你选择但不能超过1000；"
        "这是一个单步计算，查状态与读取原生结果属于同一作业验收，不新增计算分支。"
        "请选已有合适电荷的结构，自己核对需要的晶胞、力场和参数，使用现有领域工具在真实CPU计算节点提交，不能在登录节点运行模拟。"
        "先确认当前资源可用，不使用旧固定节点清单；不得运行训练、GPU、电荷生成或其他额外计算。最多一个成功作业；"
        "若首个真实作业失败，只允许在实际日志证据支持的修复后再提交一次。排队或等待不是重提理由。"
        f"唯一工作目录是本会话 {root}，共享CIF可读但不得修改。"
        "必须跟踪自己提交的真实 job_id，核查终态，并读取原生输出文件亲自提取平均绝对吸附量、单位、误差（若有）、温度和压力；"
        "同时说明执行节点、作业退出状态和读取的准确结果文件。不能把成功提交或工具返回当成计算完成，不要编造结果。"
        "完成后简洁报告实际值及低采样smoke局限。"
    )
    save(root / "input.json", {"prompt": prompt, "kind": "REAL_SCIENTIFIC_GCMC_NO_MOCKS", "source_hashes": frozen,
                               "production_instructions_sha256": hashlib.sha256(ORCHESTRATOR.instructions.encode()).hexdigest()})
    result = {"kind": "REAL_SCIENTIFIC_GCMC_NO_MOCKS", "report": str(root / "result.json")}
    try:
        with patch("agents.config._config", config), patch("agents.defns._registry", registry), \
             patch("agents.task_line.get_store", lambda: lines), patch("agents.job_watch.get_watch", lambda: watch), \
             (root / "private_console.log").open("w") as log, redirect_stdout(log), redirect_stderr(log):
            set_context(user, cid, "lead-orchestrator", cid)
            answers.append(session.run_until_complete(prompt, agent=ORCHESTRATOR, max_rounds=12, verbose=False))
            # Lifecycle observation only: this code never chooses submission
            # parameters or launches/relaunches a calculation itself.
            for _ in range(3):
                if not submissions or time.monotonic() - started > args.timeout:
                    break
                status = observe(submissions[-1])
                while not status.get("terminal") and time.monotonic() - started <= args.timeout:
                    time.sleep(15)
                    status = observe(submissions[-1])
                save(root / "session_state.json", session.export_state())
                if not status.get("terminal"):
                    result["pending_job"] = submissions[-1]["job_id"]
                    break
                own_calls = session.memory.tool_call_log
                has_native_read = any(c.get("tool") in {"read_file", "grep_search"} and
                                      "Average loading absolute" in c.get("result", "") for c in own_calls)
                if not status.get("failed") and has_native_read:
                    break
                event = json.dumps({"job_id": submissions[-1]["job_id"], "work_dir": submissions[-1]["work_dir"],
                                    "actual_scheduler_observation": status}, ensure_ascii=False)
                answers.append(session.reply("本会话真实作业状态事件：" + event +
                    "。继续完成原任务：若成功则读取原生输出并提取实际量与单位；若失败则依据日志诊断，保持原科学范围。", max_rounds=10, verbose=False))
            result["answers"] = answers
    except Exception as exc:
        result["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    except KeyboardInterrupt:
        result["exception"] = {"type": "HarnessInterrupted", "message": "Operator stopped this owned acceptance process; any jobs remain recorded."}
    finally:
        clear_context()
        save(root / "session_state.json", session.export_state())
        tool_calls = session.memory.tool_call_log
        save(root / "tool_calls.json", tool_calls)
        result.update(submissions=submissions, model_calls=len(calls), elapsed_s=round(time.monotonic() - started, 2))
        native_values = []
        for job in submissions:
            work = Path(job["work_dir"])
            for data in sorted(work.glob("**/Output/System_0/*.data")):
                text = data.read_text(errors="replace")
                match = re.search(r"Average loading absolute \[mol/kg framework\]\s+([-+\d.eE]+)\s+\+/-\s+([-+\d.eE]+)", text)
                if match:
                    input_path = data.parents[2] / "simulation.input"
                    simulation_input = input_path.read_text() if input_path.exists() else ""
                    pressure = re.search(r"^ExternalPressure\s+(\S+)", simulation_input, re.MULTILINE)
                    temperature = re.search(r"^ExternalTemperature\s+(\S+)", simulation_input, re.MULTILINE)
                    native_values.append({"job_id": job["job_id"], "file": str(data), "sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
                                          "loading": float(match[1]), "uncertainty": float(match[2]), "unit": "mol/kg framework",
                                          "native_line": match[0], "pressure_pa": float(pressure[1]) if pressure else None,
                                          "temperature_K": float(temperature[1]) if temperature else None})
            capture_scheduler(job["job_id"])
        result["native_values"] = native_values
        result["scheduler_receipts"] = scheduler_receipts
        terminal_success = {str(o["job_id"]) for o in observations if o["status"].get("terminal") and not o["status"].get("failed") and o["status"].get("status") == "COMPLETED"}
        final_text = answers[-1] if answers else ""
        numbers = [float(n) for n in re.findall(r"(?<![\w])[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", final_text)]
        result["checks"] = {
            "production_handoff": any(c.get("tool") == "handoff_to_adsorption" for c in tool_calls),
            "model_submitted_real_registered_gcmc": bool(submissions) and all(s["job_id"].isdigit() for s in submissions),
            "one_success_at_most_one_failed_attempt": len(terminal_success) == 1 and 1 <= len(submissions) <= 2,
            "real_scheduler_exit_zero_on_compute_node": any(r["job_id"] in terminal_success and r["state"] == "COMPLETED" and
                r["exit_code"] == "0:0" and r["node"] and r["node"] not in {"node01", "(null)"} for r in scheduler_receipts),
            "agent_checked_own_job": any(c.get("tool") == "check_job" and str(c.get("params", {}).get("job_id")) in terminal_success for c in tool_calls),
            "native_finite_loading_with_units": bool(native_values) and all(math.isfinite(v["loading"]) and math.isfinite(v["uncertainty"]) for v in native_values),
            "native_inputs_match_approved_temperature_pressure": bool(native_values) and all(v["pressure_pa"] == 100000 and v["temperature_K"] == 298 for v in native_values),
            "agent_read_native_loading": any(c.get("tool") in {"read_file", "grep_search"} and "Average loading absolute" in c.get("result", "") for c in tool_calls),
            "answer_matches_actual_loading": any(math.isclose(n, v["loading"], rel_tol=0.005, abs_tol=1e-7) for v in native_values for n in numbers),
            "fixed_implementation": frozen == hashes(),
            "bounded_without_exception": "exception" not in result and time.monotonic() - started <= args.timeout,
        }
        result["blocked_calls"] = blocked
        result["passed"] = all(result["checks"].values())
        save(root / "result.json", result)
        print(json.dumps({"passed": result["passed"], "checks": result["checks"], "job_ids": [s["job_id"] for s in submissions],
                          "model_calls": len(calls), "elapsed_s": result["elapsed_s"], "report": result["report"]}), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
