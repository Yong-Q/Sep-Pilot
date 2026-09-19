#!/usr/bin/env python3
"""Repair a TaskLine ghost left by a proven pre-dispatch framework failure.

This command never starts, cancels or resubmits work.  It only acts when the
authoritative workflow runtime proves that the tool was never entered and no
scheduler receipt or resource lease exists for the node.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agents.orchestration_chain import sync_chain
from agents.parallel_workflow import WorkflowStore
from agents.state_io import json_transaction
from agents.task_line import get_store
from agents.watch_context import clear_context, set_context


def reconcile(project_root: Path, username: str, conv_id: str) -> dict:
    runtime_store = WorkflowStore(project_root / "data/state/parallel_workflows.json")
    workflow_id = runtime_store.identity(username, conv_id)
    all_runtime = runtime_store.snapshot()
    runtime = all_runtime.get("workflows", {}).get(workflow_id, {})
    if not runtime:
        raise ValueError("scoped workflow runtime does not exist")
    leased = {
        (lease.get("workflow_id"), lease.get("step_id"))
        for lease in all_runtime.get("leases", {}).values()
    }
    safe_nodes = {
        step_id: node
        for step_id, node in runtime.get("nodes", {}).items()
        if node.get("status") == "failed"
        and node.get("dispatch_phase") == "preflight"
        and not node.get("tool_returned")
        and not node.get("job_ids")
        and (workflow_id, step_id) not in leased
    }
    if not safe_nodes:
        raise ValueError("no failed lease-free pre-dispatch nodes are eligible")

    checkpoint_path = project_root / "runs" / username / conv_id / "session_checkpoint.json"
    if not checkpoint_path.exists():
        raise ValueError("scoped main session checkpoint does not exist")
    checkpoint = json.loads(checkpoint_path.read_text())
    if checkpoint.get("owner") and checkpoint["owner"].get("username") not in {None, username}:
        raise PermissionError("checkpoint owner differs from requested user")

    set_context(username, conv_id, "lead-orchestrator", conv_id)
    try:
        task_lines = get_store()
        line = task_lines.get_line(conv_id, username=username, conv_id=conv_id)
        if not line:
            raise ValueError("scoped TaskLine does not exist")
        repaired = []
        for step_id, node in safe_nodes.items():
            projected = next((item for item in line.get("steps", [])
                              if item.get("step_id") == step_id), None)
            if not projected or projected.get("status") not in {
                "running", "submitted", "unconfirmed", "failed"
            }:
                continue
            validation = dict(projected.get("validation", {}))
            validation.update({
                "runtime_token": None,
                "resource_leases": [],
                "lease_released": True,
                "dispatch_not_entered": True,
                "executor_status": "failed",
                "error": node.get("error", "pre-dispatch framework failure"),
            })
            task_lines.upsert_step(
                conv_id, step_id, username=username, conv_id=conv_id,
                status="failed", done=False, validation=validation,
                resource_allocation=node.get("resource_allocation", {}),
                effective_arguments=node.get("effective_arguments", {}),
                note="reconciled from authoritative lease-free pre-dispatch failure",
            )
            repaired.append(step_id)
        if not repaired:
            raise ValueError("no stale TaskLine projection required repair")

        with json_transaction(checkpoint_path) as saved:
            interaction = saved.get("pending_user_interaction") or {}
            related = set((interaction.get("params") or {}).get("related_nodes") or [])
            if interaction and related and related <= set(repaired) and not saved.get("pending_workflow_patch"):
                saved["pending_user_interaction"] = None
                saved["waiting_for_user_input"] = False
                saved["pending_param_question"] = ""
            saved.setdefault("context", {})["workflow_recompile_required"] = {
                "reason": "approved graph ended at a pre-dispatch failure and did not cover the requested final deliverable",
                "failed_nodes": repaired,
                "required_action": "persist one complete revised draft and compile its full DAG before resuming",
            }
            goal = saved.get("goal_contract", {})
            if not saved.get("pending_workflow_patch"):
                goal["pending_plan_version"] = goal.get("approved_plan_version", 0)
            saved["checkpoint_reason"] = "preflight_projection_reconciled"

        current_line = task_lines.get_line(conv_id, username=username, conv_id=conv_id) or {}
        checkpoint = json.loads(checkpoint_path.read_text())
        chain = sync_chain(
            project_root / "runs" / username / conv_id,
            lambda: current_line,
            lambda: runtime_store.snapshot(workflow_id),
            "preflight_projection_reconciled",
            state=checkpoint,
        )
        return {
            "ok": True,
            "username": username,
            "conv_id": conv_id,
            "repaired_nodes": repaired,
            "jobs_affected": 0,
            "resubmitted": False,
            "chain_revision": chain.get("revision"),
        }
    finally:
        clear_context()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--username", required=True)
    parser.add_argument("--conv-id", required=True)
    args = parser.parse_args()
    print(json.dumps(reconcile(args.project_root.resolve(), args.username, args.conv_id), ensure_ascii=False))


if __name__ == "__main__":
    main()
