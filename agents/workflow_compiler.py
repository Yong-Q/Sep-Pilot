"""Mechanical compilation of executable workflow node contracts."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SAFE_SCHEMA_DEFAULT_ARGUMENTS = frozenset({
    "offset",
    "pattern",
    "max_bytes",
    "timeout_seconds",
    "format",
    "recursive",
    "max_files",
})


@dataclass(frozen=True)
class WorkflowCompilation:
    changes: list[dict[str, Any]]
    defaults_applied: list[dict[str, Any]]


def materialize_safe_defaults(node, registry, step_id):
    """Copy explicitly safe JSON-schema defaults into an executable node."""
    from .workflow_patch import WorkflowContractError

    result = copy.deepcopy(node)
    arguments = result.setdefault("arguments", {})
    tool = registry.get(result.get("tool", ""))
    if tool is None:
        raise WorkflowContractError("workflow node tool is not registered", {
            "step_id": step_id,
            "tool": result.get("tool"),
            "next_action": "Select one concrete registered tool and compile its real schema.",
        })
    schema = tool.input_schema
    applied = []
    for name, field in schema.get("properties", {}).items():
        if (
            name in arguments
            or name not in SAFE_SCHEMA_DEFAULT_ARGUMENTS
            or "default" not in field
        ):
            continue
        arguments[name] = copy.deepcopy(field["default"])
        applied.append({
            "step_id": step_id,
            "tool": result["tool"],
            "parameter": name,
            "value": copy.deepcopy(field["default"]),
            "source": "tool_schema",
        })
    return result, applied


_DIRECTORY_ARGUMENTS = frozenset({
    "cif_dir", "input_dir", "work_dir", "job_work_dir", "output_dir",
    "model_dir", "cwd",
})
_FILE_ARGUMENTS = frozenset({
    "cif", "cif_path", "input_path", "data_csv", "trajectory_path",
    "output_csv", "output_markdown", "output", "output_path",
})
_BINDABLE_INPUT_ARGUMENTS = frozenset({
    "cif", "cif_path", "cif_dir", "input_path", "data_csv",
    "trajectory_path", "model_dir",
})
_READ_PATH_TOOLS = frozenset({"read_file", "inspect_path", "grep_search"})


def _is_bindable_input(tool_name, target, arguments):
    if target in _BINDABLE_INPUT_ARGUMENTS:
        return True
    if target == "path":
        return tool_name in _READ_PATH_TOOLS
    if target == "input_dir":
        return tool_name != "run_cdft" or arguments.get("action") in {"submit", "collect"}
    if target == "work_dir":
        return tool_name in {
            "analyze_gcmc_screening", "validate_gcmc_results",
            "generate_scientific_report",
        }
    return False


def _graph_context(changes, existing_steps=()):
    from .workflow_patch import WorkflowContractError

    nodes = {}
    for step in existing_steps or ():
        step_id = step.get("step_id")
        if (not isinstance(step_id, str) or not step_id.strip()
                or step.get("status") == "superseded" or step.get("branch_parent")):
            continue
        if step_id in nodes:
            raise WorkflowContractError("duplicate existing workflow step identity", {
                "step_id": step_id,
                "next_action": "Repair the persisted workflow before compiling a patch.",
            })
        nodes[step_id] = copy.deepcopy(step)
    edited, upserts = set(), set()
    for change in changes:
        step_id = change.get("step_id")
        if not isinstance(step_id, str) or not step_id.strip() or step_id != step_id.strip():
            raise WorkflowContractError("invalid workflow step identity", {
                "step_id": step_id,
                "next_action": "Use one non-empty, whitespace-trimmed step_id.",
            })
        if step_id in edited:
            raise WorkflowContractError("duplicate workflow step identity", {
                "step_id": step_id,
                "next_action": "Submit one final upsert per step_id.",
            })
        edited.add(step_id)
        if change.get("operation") == "remove":
            nodes.pop(step_id, None)
            continue
        if change.get("operation") != "upsert":
            raise WorkflowContractError("unknown workflow patch operation", {
                "step_id": step_id,
                "next_action": "Use upsert or remove.",
            })
        nodes[step_id] = change.get("node") or {}
        upserts.add(step_id)

    visiting, visited, order = set(), set(), []

    def visit(step_id):
        if step_id in visiting:
            raise WorkflowContractError("workflow dependency cycle", {
                "step_id": step_id,
                "next_action": "Remove the cyclic dependency before compiling arguments.",
            })
        if step_id in visited:
            return
        visiting.add(step_id)
        dependencies = nodes[step_id].get("depends_on", [])
        if not isinstance(dependencies, list):
            raise WorkflowContractError("workflow dependencies must be a list", {
                "step_id": step_id,
                "next_action": "Use depends_on as a list of existing step IDs.",
            })
        for dependency in dependencies:
            if dependency not in nodes:
                raise WorkflowContractError("workflow references a missing dependency", {
                    "step_id": step_id,
                    "dependency": dependency,
                    "next_action": "Add the producer node or correct depends_on.",
                })
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)
        order.append(step_id)

    for step_id in nodes:
        visit(step_id)

    ancestors = {}
    for step_id in order:
        inherited = set()
        for dependency in nodes[step_id].get("depends_on", []):
            inherited.add(dependency)
            inherited.update(ancestors[dependency])
        ancestors[step_id] = inherited
    return nodes, order, ancestors, upserts


def _artifact_candidate(producer_id, node, output_index, target, consumer_tool,
                        conversation_root):
    from .output_contract import normalize_output, output_path

    outputs = node.get("expected_outputs", [])
    if (isinstance(output_index, bool) or not isinstance(output_index, int)
            or output_index < 0 or output_index >= len(outputs)):
        return None
    output = outputs[output_index]
    contract = normalize_output(output)
    arguments = node.get("arguments", {})
    base = (arguments.get("output_dir") or arguments.get("job_work_dir")
            or arguments.get("work_dir") or conversation_root)
    path = output_path(output, base)
    wildcard = contract["kind"] == "file" and any(char in path.name for char in "*?[")
    if target in _DIRECTORY_ARGUMENTS:
        path = path if contract["kind"] == "directory" else path.parent
        kind = "directory"
    elif target in _FILE_ARGUMENTS or (
            target == "path" and consumer_tool in {"read_file", "write_file"}):
        if contract["kind"] != "file" or wildcard:
            return None
        kind = "file"
    else:
        path = path.parent if wildcard else path
        kind = "directory" if contract["kind"] == "directory" or wildcard else "file"
    return {
        "step_id": producer_id,
        "output_index": output_index,
        "path": str(path),
        "kind": kind,
    }


def compile_dataflow_bindings(changes, registry, project_root, conversation_root,
                              existing_steps=()):
    """Resolve approved ancestor artifacts into concrete consumer path arguments."""
    from .workflow_patch import WorkflowContractError
    from .workspace import PATH_ARGUMENTS

    result = copy.deepcopy(changes)
    nodes, order, ancestors, upserts = _graph_context(result, existing_steps)

    for step_id in order:
        if step_id not in upserts:
            continue
        node = nodes[step_id]
        tool = registry.get(node.get("tool", ""))
        if tool is None:
            continue
        arguments = node.setdefault("arguments", {})
        properties = tool.input_schema.get("properties", {})
        required = set(tool.input_schema.get("required", []))
        bindings = node.get("input_bindings", {})
        if not isinstance(bindings, dict):
            raise WorkflowContractError("input_bindings must be an object", {
                "step_id": step_id, "tool": node.get("tool"),
                "next_action": "Map each consumer argument to {step_id, output_index}.",
            })

        for target, binding in bindings.items():
            valid_shape = (isinstance(binding, dict)
                           and set(binding) == {"step_id", "output_index"})
            producer_id = binding.get("step_id") if isinstance(binding, dict) else None
            output_index = binding.get("output_index") if isinstance(binding, dict) else None
            if (not valid_shape or target not in properties or target not in PATH_ARGUMENTS
                    or not _is_bindable_input(node.get("tool", ""), target, arguments)
                    or producer_id not in ancestors[step_id]
                    or isinstance(output_index, bool) or not isinstance(output_index, int)
                    or output_index < 0):
                raise WorkflowContractError("invalid workflow input binding", {
                    "step_id": step_id, "tool": node.get("tool"),
                    "parameter": target, "binding": copy.deepcopy(binding),
                    "next_action": (
                        "Bind a real path argument to one declared output of a dependency ancestor."
                    ),
                })
            candidate = _artifact_candidate(
                producer_id, nodes[producer_id], output_index, target,
                node.get("tool", ""), conversation_root)
            if candidate is None:
                raise WorkflowContractError("workflow input binding selects an incompatible or missing output", {
                    "step_id": step_id, "tool": node.get("tool"),
                    "parameter": target, "binding": copy.deepcopy(binding),
                    "next_action": "Select an in-range compatible expected_outputs entry.",
                })
            existing = arguments.get(target)
            if existing is not None and str(Path(str(existing)).resolve()) != candidate["path"]:
                raise WorkflowContractError("workflow input binding conflicts with an explicit argument", {
                    "step_id": step_id, "tool": node.get("tool"),
                    "parameter": target, "argument": existing,
                    "binding_path": candidate["path"],
                    "next_action": "Remove the conflicting value or bind the matching producer output.",
                })
            arguments[target] = candidate["path"]

        for target in sorted(required & PATH_ARGUMENTS - set(arguments)):
            candidates = []
            if _is_bindable_input(node.get("tool", ""), target, arguments):
                for producer_id in sorted(ancestors[step_id]):
                    for output_index in range(len(nodes[producer_id].get("expected_outputs", []))):
                        candidate = _artifact_candidate(
                            producer_id, nodes[producer_id], output_index, target,
                            node.get("tool", ""), conversation_root)
                        if candidate is not None:
                            candidates.append(candidate)
            unique = {}
            for candidate in candidates:
                unique[(candidate["path"], candidate["kind"])] = candidate
            candidates = list(unique.values())
            if len(candidates) == 1:
                arguments[target] = candidates[0]["path"]
                continue
            raise WorkflowContractError("required path argument cannot be bound unambiguously", {
                "step_id": step_id, "tool": node.get("tool"),
                "missing_parameters": [target],
                "binding_candidates": candidates,
                "input_schema": copy.deepcopy(tool.input_schema),
                "next_action": (
                    "Add input_bindings selecting one dependency ancestor output; do not ask the user "
                    "for this internal path."
                ),
            })
    return result


def compile_workflow_changes(changes, registry, project_root, conversation_root,
                             existing_steps=()):
    """Run the complete mechanical DAG compilation pipeline once."""
    from .workflow_patch import (
        WorkflowContractError,
        canonical_arguments,
        canonical_expected_outputs,
        complete_compute_contract,
        resolve_workflow_placeholders,
    )

    compiled = resolve_workflow_placeholders(
        changes, conversation_root, existing_steps=existing_steps)
    for change in compiled:
        if change.get("operation") != "upsert" or not change.get("node"):
            continue
        node = change["node"]
        node["arguments"] = canonical_arguments(node.get("arguments", {}), project_root)
        node["expected_outputs"] = canonical_expected_outputs(
            node.get("expected_outputs", []), project_root)
        change["node"] = complete_compute_contract(
            node, change["step_id"], project_root, conversation_root)

    compiled = compile_dataflow_bindings(
        compiled, registry, project_root, conversation_root, existing_steps)
    defaults_applied = []
    for change in compiled:
        if change.get("operation") != "upsert" or not change.get("node"):
            continue
        node, applied = materialize_safe_defaults(
            change["node"], registry, change["step_id"])
        change["node"] = node
        defaults_applied.extend(applied)
        issues = registry.validate_params(node.get("tool", ""), node.get("arguments", {}))
        if issues:
            tool = registry.get(node.get("tool", ""))
            receipt = registry._validation_receipt(
                node.get("tool", ""), issues, tool.input_schema if tool else {})
            raise WorkflowContractError("workflow tool arguments failed compilation", {
                **receipt,
                "step_id": change["step_id"],
                "agent": node.get("agent"),
            })
    return WorkflowCompilation(compiled, defaults_applied)
