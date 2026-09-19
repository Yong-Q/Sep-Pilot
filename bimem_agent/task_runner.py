from __future__ import annotations

from typing import Any, Dict

from .catalog import load_catalog, skill_index
from .legacy_bridge import (
    build_guest_forcefield_command,
    build_charge_command,
    build_external_potential_command,
    build_pore_analysis_command,
    build_string_tst_command,
    build_structure_gen_command,
    build_xtb_optimize_command,
    execute_charge_plan,
    execute_external_potential_plan,
    execute_pore_analysis_plan,
    execute_structure_gen_plan,
    execute_string_tst_plan,
    execute_xtb_optimize_plan,
    helper_call,
    run_literature_rag,
    run_adsorption_workflow,
    run_interaction,
    run_jobs_workflow,
    run_material_props,
    run_md_optimize,
    run_pore_analysis,
    run_subprocess_plan,
    run_vasp_workflow,
)
from .paths import LEGACY_ROOT
from .result_inspector import inspect_job_name, inspect_path, inspect_run


def _wrap_result(skill: str, action: str, status: str, payload: Any, executed: bool, automation_level: str) -> Dict[str, Any]:
    return {
        "skill": skill,
        "action": action,
        "status": status,
        "executed": executed,
        "automation_level": automation_level,
        "payload": payload,
    }


def _text_status(payload: Any) -> str:
    if not isinstance(payload, str):
        return "completed"
    lowered = payload.lower()
    if (
        "失败" in payload
        or payload.startswith("错误:")
        or "error:" in lowered
        or "submit_failed" in lowered
        or "qsub_not_found" in lowered
    ):
        return "failed"
    return "completed"


def _resolve_skill_name(raw_skill: str) -> str:
    catalog = load_catalog()
    for skill in catalog["skills"]:
        if raw_skill == skill["id"] or raw_skill in skill.get("aliases", []):
            return skill["id"]
    raise KeyError(f"Unknown skill: {raw_skill}")


def _resolve_refs(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith("$steps."):
        path = value[len("$steps."):].split(".")
        current: Any = context.get("steps", context)
        for token in path:
            if isinstance(current, dict):
                current = current[token]
            else:
                raise KeyError(f"Cannot resolve reference segment '{token}' in {value}")
        return current
    if isinstance(value, list):
        return [_resolve_refs(item, context) for item in value]
    if isinstance(value, dict):
        return {key: _resolve_refs(item, context) for key, item in value.items()}
    return value


def _maybe_execute_plan(skill: str, action: str, automation_level: str, plan: Dict[str, Any], execute: bool) -> Dict[str, Any]:
    if execute:
        payload = run_subprocess_plan(plan)
        status = "completed" if payload["returncode"] == 0 else "failed"
        return _wrap_result(skill, action, status, payload, True, automation_level)
    return _wrap_result(skill, action, "planned", plan, False, automation_level)


def run_task_spec(spec: Dict[str, Any], execute: bool = False, context: Dict[str, Any] | None = None) -> Dict[str, Any]:
    if "steps" in spec:
        return run_workflow_spec(spec, execute=execute)

    skill = _resolve_skill_name(spec["skill"])
    action = spec.get("action", "default")
    params = _resolve_refs(spec.get("params", {}), context or {})
    catalog = skill_index()

    automation_level = catalog[skill]["automation_level"]

    if skill == "cif":
        payload = helper_call("find_cif", **params)
        return _wrap_result(skill, action, _text_status(payload), payload, True, automation_level)
    if skill == "paper":
        payload = helper_call("fetch_paper", **params)
        return _wrap_result(skill, action, _text_status(payload), payload, True, automation_level)
    if skill == "literature-rag":
        payload = run_literature_rag(
            query=params["query"],
            n_results=params.get("n_results", 5),
            query_type=params.get("query_type", "auto"),
        )
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "adsorption":
        payload = run_adsorption_workflow(action, **params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "analysis":
        payload = helper_call("analyze", **params)
        return _wrap_result(skill, action, _text_status(payload), payload, True, automation_level)
    if skill == "jobs":
        normalized_action = "submit-single" if action == "submit" else action
        payload = run_jobs_workflow(normalized_action, **params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "cdft":
        func = {
            "pipeline": "run_cdft_pipeline",
            "submit": "run_cdft_submit",
            "collect": "collect_cdft_results",
            "inputs": "run_cdft_inputs",
        }[action]
        payload = helper_call(func, **params)
        return _wrap_result(skill, action, _text_status(payload), payload, True, automation_level)
    if skill == "pore-analysis":
        if params.get("scheduler") == "pbs":
            plan = build_pore_analysis_command(params)
            if execute:
                payload = execute_pore_analysis_plan(plan)
                status = "completed" if payload["returncode"] == 0 else "failed"
                return _wrap_result(skill, action, status, payload, True, automation_level)
            return _wrap_result(skill, action, "planned", plan, False, automation_level)
        payload = run_pore_analysis(**params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "material-props":
        payload = run_material_props(**params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "interaction":
        payload = run_interaction(**params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "md-optimize":
        payload = run_md_optimize(**params)
        return _wrap_result(skill, action, "completed", payload, True, automation_level)
    if skill == "charge":
        plan = build_charge_command(params)
        if execute:
            payload = execute_charge_plan(plan)
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "string-tst":
        plan = build_string_tst_command(params)
        if execute:
            payload = execute_string_tst_plan(plan)
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "structure-gen":
        plan = build_structure_gen_command(params)
        if execute:
            payload = execute_structure_gen_plan(plan)
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "xtb-optimize":
        plan = build_xtb_optimize_command(params)
        if execute:
            payload = execute_xtb_optimize_plan(plan)
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "external-potential":
        plan = build_external_potential_command(params)
        if execute:
            payload = execute_external_potential_plan(plan)
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "guest-forcefield":
        try:
            plan = build_guest_forcefield_command(action, params)
        except (ValueError, KeyError) as error:
            return _wrap_result(skill, action, 'failed', {'error':str(error), 'returncode':2,
                'next_action':'main chat obtains explicit scientific model parameters; no inputs modified'}, False, automation_level)
        if execute:
            payload = run_subprocess_plan(plan)
            if "output_dir" in plan:
                payload["output_dir"] = plan["output_dir"]
            if "manifest_path" in plan:
                payload["manifest_path"] = plan["manifest_path"]
            if "cif_path" in plan:
                payload["cif_path"] = plan["cif_path"]
            status = "completed" if payload["returncode"] == 0 else "failed"
            return _wrap_result(skill, action, status, payload, True, automation_level)
        return _wrap_result(skill, action, "planned", plan, False, automation_level)
    if skill == "vasp-dft":
        payload = run_vasp_workflow(action, **params)
        return _wrap_result(skill, action, "completed", payload, True, "full")
    if skill == "results":
        if action in {"default", "inspect"}:
            payload = inspect_path(params["path"], max_entries=params.get("max_entries", 20))
            return _wrap_result(skill, action, "completed", payload, True, automation_level)
        if action == "job":
            payload = inspect_job_name(
                job_name=params["job_name"],
                legacy_root=params.get("legacy_root", LEGACY_ROOT),
                max_lines=params.get("max_lines", 8),
            )
            return _wrap_result(skill, action, "completed", payload, True, automation_level)
        if action == "run":
            payload = inspect_run(run_id=params["run_id"], project=params.get("project"))
            return _wrap_result(skill, action, "completed", payload, True, automation_level)
        raise ValueError(f"Unsupported results action: {action}")
    if skill in {"overview", "harness"}:
        return _wrap_result(skill, action, "metadata_only", {"message": catalog[skill]["automation"]}, False, automation_level)

    raise NotImplementedError(f"Skill not wired in task runner: {skill}")


def run_workflow_spec(spec: Dict[str, Any], execute: bool = False) -> Dict[str, Any]:
    steps = spec.get("steps", [])
    if not steps:
        raise ValueError("Workflow spec must include non-empty 'steps'")

    workflow_name = spec.get("workflow_name", "unnamed-workflow")
    context: Dict[str, Any] = {"steps": {}}
    results = []

    for step in steps:
        step_id = step["id"]
        result = run_task_spec(step, execute=execute, context=context)
        context["steps"][step_id] = result
        results.append({"id": step_id, **result})

    return {
        "workflow_name": workflow_name,
        "status": "completed",
        "executed": execute,
        "steps": results,
    }
