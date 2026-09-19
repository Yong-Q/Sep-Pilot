"""Real Agent SDK test for charge-assignment versus electrostatics semantics.

The model receives a natural-language scientific request and must compile the
production structured workflow.  No calculation or scheduler call is made:
the proposal stays pending for user approval.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.session import ExecutionBudgetExceeded, Session
from agents.state_io import write_checkpoint
from agents.task_line import TaskLineStore
from agents.watch_context import clear_context, set_context


def main() -> int:
    cid = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    # Match the production per-user/per-session workspace exactly.  The
    # workspace guard resolves sdk_charge/<conv_id> to this directory.
    scope = ROOT / "runs" / "sdk_charge" / cid
    scope.mkdir(parents=True, exist_ok=False)
    raw_cif = scope / "inputs" / "HKUST-1_raw.cif"
    raw_cif.parent.mkdir(parents=True)
    raw_cif.write_text(
        "data_HKUST-1\n_cell_length_a 26.3\n_cell_length_b 26.3\n"
        "_cell_length_c 26.3\n_cell_angle_alpha 90\n"
        "_cell_angle_beta 90\n_cell_angle_gamma 90\n",
        encoding="utf-8",
    )
    charge_dir = scope / "charged"
    result_dir = scope / "gcmc"
    lines = TaskLineStore(str(scope / "task_lines.json"))
    lines.begin_line(cid, username="sdk_charge", conv_id=cid)
    set_context("sdk_charge", cid, "lead-orchestrator", cid)

    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client = session.client.with_options(timeout=90, max_retries=0)
    session._current_line_id = cid
    session._evidence_root = scope / "evidence"
    session._on_checkpoint = lambda state, reason: write_checkpoint(
        scope / "session_checkpoint.json", state
    )
    agent = copy.copy(ORCHESTRATOR)
    agent.functions = [
        fn for fn in agent.functions
        if fn.__name__ in {"task_line_query", "get_tool_schema", "propose_workflow_patch"}
    ]
    real_call = session._call_api
    model_calls = []

    def bounded(current_agent):
        if len(model_calls) >= 8:
            raise ExecutionBudgetExceeded("charge orchestration model-call budget")
        response = real_call(current_agent)
        model_calls.append([
            {"name": block.name, "arguments": block.input}
            for block in response.content
            if getattr(block, "type", "") == "tool_use"
        ])
        return response

    session._call_api = bounded
    result = {
        "kind": "REAL_SDK_AUTONOMOUS_CHARGE_WORKFLOW",
        "calculation": "NONE; structured production-tool proposal only",
        "model_calls": model_calls,
    }
    prompt = (
        f"请为 HKUST-1 中 C2H4/C2H6 在 298K、0.1-10 bar 的 GCMC 等温线编排可执行链。"
        f"原始 CIF 是 {raw_cif}，已知没有 _atom_site_charge；带电结构输出到 {charge_dir}，"
        f"GCMC 结果放到 {result_dir}。框架的声明晶胞净电荷为 0 e。"
        "用项目内置工具自主确定快速默认电荷路线和静电求和方法；"
        "只编译结构化编排，不执行、不提交作业，等用户之后批准。"
        "不要给文本步骤，请自主查 schema 并调用 propose_workflow_patch。"
    )
    try:
        with patch("agents.task_line.get_store", lambda: lines):
            result["answer"] = session.run_until_complete(
                prompt, agent=agent, max_rounds=8, verbose=False
            )
        proposal = session._pending_workflow_patch or {}
        nodes = proposal.get("proposed_nodes", [])
        by_tool = {node.get("tool"): node for node in nodes}
        charge = by_tool.get("run_pacman_charge", {})
        validate = by_tool.get("validate_framework_charges", {})
        gcmc_nodes = [node for node in nodes if node.get("tool") == "run_gcmc_isotherm"]
        charge_args = charge.get("arguments", {})
        validate_args = validate.get("arguments", {})
        all_gcmc_args = [node.get("arguments", {}) for node in gcmc_nodes]
        result["proposal"] = proposal
        result["checks"] = {
            "model_called_structured_patch": any(
                call.get("tool") == "propose_workflow_patch"
                for call in session.memory.tool_call_log
            ),
            "pacmof_is_default_assignment": charge_args.get("method") == "pacmof",
            "charge_validation_is_explicit": bool(validate)
                and validate_args.get("expected_net_charge") == 0,
            "charge_precedes_validation": charge.get("step_id")
                in validate.get("depends_on", []),
            "validation_precedes_gcmc": len(gcmc_nodes) == 2
                and all(validate.get("step_id") in node.get("depends_on", []) for node in gcmc_nodes),
            "mixture_scope_complete": {args.get("gas") for args in all_gcmc_args}
                == {"C2H4", "C2H6"},
            "electrostatics_is_not_charge_assignment": all(
                args.get("electrostatics_method") in {"Ewald", "Wolf"}
                and "charge_method" not in args
                for args in all_gcmc_args
            ),
            "no_none_solver_workaround": all(
                args.get("electrostatics_method") != "None" for args in all_gcmc_args
            ),
            "not_executed_before_approval": proposal.get("new_version") == 1
                and session.goal_contract.approved_plan_version == 0,
        }
        result["passed"] = all(result["checks"].values())
    except Exception as error:
        result.update(passed=False, error=f"{type(error).__name__}: {error}")
    finally:
        write_checkpoint(scope / "test_result.json", result)
        clear_context()
        print(json.dumps({
            "passed": result.get("passed"),
            "report": str(scope / "test_result.json"),
            "checks": result.get("checks"),
            "error": result.get("error"),
        }, ensure_ascii=False), flush=True)
    return 0 if result.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
