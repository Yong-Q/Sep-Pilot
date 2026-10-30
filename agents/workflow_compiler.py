"""Mechanical compilation of executable workflow node contracts."""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import json
import re
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
    "completion_policy",
})

@dataclass(frozen=True)
class WorkflowCompilation:
    changes: list[dict[str, Any]]
    defaults_applied: list[dict[str, Any]]


def _report_source_identity(runtime_node):
    """Hash only durable source identity, never model text or complete results."""
    reference = runtime_node.get('result_ref') or {}
    return {key: value for key, value in {
        'execution_fingerprint': runtime_node.get('execution_fingerprint'),
        'call_id': reference.get('call_id'),
        'evidence_path': reference.get('evidence_path'),
    }.items() if value is not None}


def _verified_fragment_reusable(prior, candidate, source, runtime_nodes, source_contract=None):
    old_args = prior.get('arguments', {})
    if ({k: v for k, v in old_args.items() if k != 'output_path'} !=
            {k: v for k, v in candidate['arguments'].items() if k != 'output_path'}):
        return False
    if (prior.get('agent') != candidate.get('agent')
            or prior.get('depends_on') != [source]
            or prior.get('expected_outputs') != [old_args.get('output_path')]):
        return False
    fragment = runtime_nodes.get(prior.get('step_id'), {})
    producer = runtime_nodes.get(source, {})
    from .workflow_patch import execution_contract
    if execution_contract(fragment.get('contract', {})) != execution_contract(prior):
        return False
    if (source_contract is not None and
            execution_contract(producer.get('contract', {})) != execution_contract(source_contract)):
        return False
    receipt = fragment.get('result') or {}
    call_id = (producer.get('result_ref') or {}).get('call_id')
    fingerprint = producer.get('execution_fingerprint')
    artifact = (fragment.get('artifacts') or {}).get(old_args.get('output_path')) or {}
    size = artifact.get('size')
    return bool(fragment.get('status') == 'succeeded' and fragment.get('phase') == 'finish'
        and isinstance(size, int) and not isinstance(size, bool) and size > 0
        and re.fullmatch(r'[0-9a-f]{64}', str(receipt.get('fragment_fingerprint', '')))
        and producer.get('status') == 'succeeded' and call_id and fingerprint
        and receipt.get('source_step') == source
        and receipt.get('source_evidence_call_id') == call_id
        and receipt.get('source_attempt_fingerprint') == fingerprint
        and receipt.get('report_file') == old_args.get('output_path'))


def _inherit_owned_fragment_metadata(changes, existing, runtime_nodes):
    """Restore omitted compiler fields only from the approved native lineage.

    Full-node upserts must not erase system fragment identity. Historical
    metadata is usable only when the current native contract matches the
    existing step and the original parent/source association remains exact.
    No source/dependency/output/agent/scientific argument is inferred or changed.
    """
    from .workflow_patch import WorkflowContractError, execution_contract
    def owned_assembly(parent_id, parent):
        if parent.get('report_role') == 'assembly':
            return True
        if parent.get('report_role') is not None:
            return False
        # Older schema-valid assembly upserts could erase the internal role.
        # Recover only the original parent association from its native history;
        # a familiar name or model-supplied report_mode never establishes it.
        native = runtime_nodes.get(parent_id, {})
        if execution_contract(native.get('contract', {})) != execution_contract(parent):
            return False
        args = parent.get('arguments') or {}
        sources = args.get('source_steps')
        if (not isinstance(sources, list) or len(sources) < 2
                or any(not isinstance(source, str) or not source for source in sources)
                or len(set(sources)) != len(sources) or sources != parent.get('depends_on')):
            return False
        for attempt in reversed((native.get('attempt_history') or [])[-64:]):
            prior = attempt.get('contract') if isinstance(attempt, dict) else None
            if not isinstance(prior, dict):
                continue
            old_args = prior.get('arguments') or {}
            if (prior.get('step_id') == parent_id == parent.get('step_id')
                    and prior.get('tool') == parent.get('tool') == 'generate_scientific_report'
                    and prior.get('agent') == parent.get('agent')
                    and prior.get('report_role') == 'assembly'
                    and old_args.get('report_mode') == args.get('report_mode') == 'assembly'
                    and old_args.get('source_steps') == prior.get('depends_on') == sources):
                return True
        return False
    for change in changes:
        if change.get('operation') != 'upsert' or not isinstance(change.get('node'), dict):
            continue
        key = change.get('step_id'); candidate = change['node']; prior = existing.get(key)
        if not isinstance(prior, dict):
            continue
        current_args = prior.get('arguments') or {}
        sources = current_args.get('source_steps')
        if not isinstance(sources, list) or len(sources) != 1:
            continue
        source = sources[0]
        def owned(record):
            args = record.get('arguments') or {}
            parent_id = record.get('report_parent'); parent = existing.get(parent_id, {})
            parent_args = parent.get('arguments') or {}
            return (isinstance(parent_id, str) and record.get('step_id') == key
                and key == parent_id + '__fragment__' + str(source)
                and record.get('tool') == 'generate_scientific_report'
                and record.get('report_role') == 'fragment'
                and record.get('report_source', source) == source
                and args.get('report_mode') == 'fragment' and args.get('source_steps') == [source]
                and record.get('depends_on') == [source]
                and parent.get('tool') == 'generate_scientific_report'
                and owned_assembly(parent_id, parent) and parent_args.get('report_mode') == 'assembly'
                and key in parent_args.get('source_steps', [])
                and parent_args.get('source_steps') == parent.get('depends_on'))
        anchor = prior if owned(prior) else None
        if anchor is None and all(prior.get(field) is None for field in ('report_role','report_parent','report_source')):
            native = runtime_nodes.get(key, {})
            if execution_contract(native.get('contract', {})) == execution_contract(prior):
                for attempt in reversed((native.get('attempt_history') or [])[-64:]):
                    historical = attempt.get('contract') if isinstance(attempt, dict) else None
                    if isinstance(historical, dict) and owned(historical):
                        anchor = historical
                        break
        if anchor is None:
            continue
        args = candidate.get('arguments') or {}
        valid = (candidate.get('step_id') == key and candidate.get('tool') == 'generate_scientific_report'
            and candidate.get('agent') == prior.get('agent')
            and prior.get('tool') == candidate['tool']
            and prior.get('depends_on') == [source] and current_args.get('report_mode') in (None,'fragment')
            and candidate.get('depends_on') == [source] and args.get('source_steps') == [source]
            and args.get('report_mode', 'fragment') == 'fragment'
            and isinstance(args.get('output_path'), str) and bool(args['output_path'])
            and candidate.get('expected_outputs') == [args['output_path']]
            and all(candidate.get(field, expected) == expected for field, expected in {
                'report_role':'fragment','report_parent':anchor['report_parent'],'report_source':source}.items()))
        if not valid:
            raise WorkflowContractError('owned report fragment upsert changes its source or role', {
                'step_id':key,'report_parent':anchor['report_parent'],'source_step':source,
                'next_action':'Keep the same fragment type, parent, agent and single source; use an unexpanded report for a different association.'})
        candidate.setdefault('report_role','fragment')
        candidate.setdefault('report_parent',anchor['report_parent'])
        candidate.setdefault('report_source',source)
        candidate['arguments'].setdefault('report_mode','fragment')
    return changes


def expand_parallel_report_nodes(changes, conversation_root, existing_steps=(), *,
                                 target_plan_version=None, source_evidence=None, expansion_receipts=None):
    """Split a multi-source final report into independent fragment branches."""
    from .workflow_patch import WorkflowContractError

    if (target_plan_version is not None and (isinstance(target_plan_version, bool)
            or not isinstance(target_plan_version, int) or target_plan_version < 1)):
        raise WorkflowContractError('report target plan version must be a positive integer', {})
    runtime_nodes = source_evidence or {}
    result = copy.deepcopy(changes)
    existing_by_id = {step.get('step_id'): step for step in existing_steps or ()}
    result = _inherit_owned_fragment_metadata(result, existing_by_id, runtime_nodes)
    proposed_by_id = {change.get('step_id'): change.get('node', {})
                      for change in result if change.get('operation') == 'upsert'}
    occupied = set(existing_by_id)
    occupied.update(change.get('step_id') for change in result)
    expanded = []
    for change in result:
        if change.get('operation') != 'upsert' or not change.get('node'):
            expanded.append(change)
            continue
        report_id = change.get('step_id')
        node = change['node']
        arguments = node.get('arguments', {})
        sources = list(arguments.get('source_steps') or [])
        if node.get('tool') != 'generate_scientific_report' or len(sources) < 2:
            expanded.append(change)
            continue
        if sources != list(node.get('depends_on', [])):
            raise WorkflowContractError('parallel report sources must be direct dependencies', {
                'step_id': report_id,
                'source_steps': sources,
                'depends_on': node.get('depends_on', []),
                'next_action': 'Use the same ordered direct dependencies for source_steps and depends_on.',
            })
        if arguments.get('report_mode') == 'assembly':
            # Replays and revisions may already contain compiler-owned
            # fragments. Re-expanding those creates fragments of fragments.
            for source in sources:
                fragment = proposed_by_id.get(source, existing_by_id.get(source, {}))
                if (fragment.get('report_parent') != report_id
                        or fragment.get('report_role') != 'fragment'
                        or fragment.get('tool') != 'generate_scientific_report'
                        or fragment.get('arguments', {}).get('report_mode') != 'fragment'):
                    raise WorkflowContractError('report assembly requires owned fragment dependencies', {
                        'step_id': report_id, 'source_step': source,
                        'next_action': 'Use compiler-owned fragments or propose an unexpanded report.',
                    })
            # These are compiler-owned fields omitted by the public patch
            # schema. Preserve the role after verifying every direct fragment.
            if node.get('report_role') not in (None, 'assembly'):
                raise WorkflowContractError('report assembly changes its compiler-owned role', {
                    'step_id': report_id})
            node['report_role'] = 'assembly'
            expanded.append(change)
            continue
        fragment_ids = []
        generated_ids = []
        original_dependencies = list(node.get('depends_on', []))
        for source in sources:
            fragment_id = f'{report_id}__fragment__{source}'
            if fragment_id in occupied:
                prior = proposed_by_id.get(fragment_id, existing_by_id.get(fragment_id, {}))
                prior_args = prior.get('arguments', {})
                owned_fragment = (
                    (prior.get('report_role') or prior_args.get('report_mode')) == 'fragment'
                    and prior.get('report_parent') == report_id
                    and prior_args.get('source_steps') == [source]
                )
                if not owned_fragment:
                    raise WorkflowContractError('parallel report fragment identity collides with an existing node', {
                        'step_id': report_id, 'fragment_step_id': fragment_id,
                        'next_action': 'Rename the report or conflicting workflow node.',
                    })
            occupied.add(fragment_id)
            fragment_ids.append(fragment_id)
            identity = {'report_id': report_id, 'source_step': source,
                        'target_plan_version': target_plan_version,
                        'source': _report_source_identity(runtime_nodes.get(source, {})),
                        'title': arguments.get('title', 'Scientific report'),
                        'instructions': arguments.get('instructions')}
            digest = hashlib.sha256(json.dumps(identity, sort_keys=True,
                separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()[:24]
            fragment_path = str(Path(conversation_root) / 'reports' / 'fragments' / f'{digest}.md')
            fragment_arguments = {
                'source_steps': [source],
                'output_path': fragment_path,
                'title': f"{arguments.get('title', 'Scientific report')} · {source}",
                'report_mode': 'fragment',
            }
            if arguments.get('instructions'):
                fragment_arguments['instructions'] = arguments['instructions']
            fragment = {
                'step_id': fragment_id,
                'agent': node.get('agent', 'communicator'),
                'tool': 'generate_scientific_report',
                'arguments': fragment_arguments,
                'depends_on': [source],
                'expected_outputs': [fragment_path],
                'resource_locks': [],
                'report_role': 'fragment',
                'report_parent': report_id,
                'report_source': source,
            }
            explicit = proposed_by_id.get(fragment_id)
            if explicit is not None:
                # The formally proposed recovery contract owns its path. Never
                # replace it with an automatically generated legacy location.
                if (explicit.get('tool') != 'generate_scientific_report'
                        or explicit.get('depends_on') != [source]
                        or explicit.get('expected_outputs') != [explicit.get('arguments', {}).get('output_path')]):
                    raise WorkflowContractError('explicit report fragment requires matching output contract', {
                        'step_id': fragment_id})
                # It is already included among input changes (in either order).
                continue
            prior = existing_by_id.get(fragment_id, {})
            if _verified_fragment_reusable(prior, fragment, source, runtime_nodes,
                    proposed_by_id.get(source, existing_by_id.get(source))):
                fragment = {key: copy.deepcopy(value) for key, value in prior.items()
                            if key in fragment}
            expanded.append({'operation': 'upsert', 'step_id': fragment_id, 'node': fragment})
            generated_ids.append(fragment_id)
        node['depends_on'] = fragment_ids
        node['arguments']['source_steps'] = fragment_ids
        node['arguments']['report_mode'] = 'assembly'
        node['report_role'] = 'assembly'
        if expansion_receipts is not None and generated_ids == fragment_ids:
            expansion_receipts.append({
                'source': 'workflow_compiler', 'parameter': 'report_pipeline_expansion',
                'step_id': report_id, 'fragment_step_ids': fragment_ids,
                'original_depends_on': original_dependencies,
            })
        expanded.append(change)
    return expanded


_CDFT_PREPARATION_ARGUMENTS = frozenset({
    'cif_dir', 'cif_path', 'gas', 'gases', 'temperature', 'bulk_densities',
    'bulk_density_unit', 'framework_charge', 'input_dir', 'job_work_dir', 'ngrid', 'dl',
})


def expand_cdft_pipeline_nodes(changes, conversation_root, existing_steps=(), *, source_evidence=None):
    """Split only never-dispatched pipeline contracts; preserve the submit ID."""
    from .workflow_patch import WorkflowContractError
    root = Path(conversation_root).resolve()
    existing = {step.get('step_id'): step for step in existing_steps or ()}
    runtime = source_evidence or {}
    occupied = set(existing) | {change.get('step_id') for change in changes}
    result, receipts = [], []
    def owned_path(value):
        path = Path(value)
        if not path.is_absolute() or not path.resolve().is_relative_to(root):
            raise WorkflowContractError('cDFT pipeline output must stay inside its owned conversation', {'path': str(path)})
        return path.resolve()
    for original in changes:
        change = copy.deepcopy(original)
        node = change.get('node') or {}
        args = node.get('arguments') or {}
        if change.get('operation') != 'upsert' or node.get('tool') != 'run_cdft' or args.get('action') != 'pipeline':
            result.append(change)
            continue
        step_id = change.get('step_id')
        if not isinstance(step_id, str) or not step_id.strip():
            raise WorkflowContractError('cDFT pipeline requires a concrete step identity', {})
        for prior in (existing.get(step_id) or {}, runtime.get(step_id) or {}):
            if (prior.get('status') not in (None, 'pending') or prior.get('done')
                    or any(prior.get(key) for key in ('token', 'attempt_id', 'job_ids', 'started_at',
                        'tool_returned', 'result', 'result_ref', 'artifacts', 'node_verification',
                        'attempt_history', 'execution_fingerprint', 'dispatch_phase'))):
                raise WorkflowContractError('cannot mechanically replace an attempted cDFT pipeline', {'step_id': step_id})
        prepare_id = step_id + '__prepare'
        if prepare_id in occupied:
            raise WorkflowContractError('cDFT preparation identity collides with an existing node', {'step_id': step_id, 'prepare_step_id': prepare_id})
        occupied.add(prepare_id)
        original_args = copy.deepcopy(args)
        original_outputs = copy.deepcopy(node.get('expected_outputs', []))
        dependencies = list(node.get('depends_on', []))
        directory = owned_path(args.get('input_dir') or root / 'cdft' / 'prepared' / hashlib.sha256(step_id.encode()).hexdigest()[:24])
        if directory.exists() or directory.is_symlink():
            raise WorkflowContractError('cDFT pipeline preparation cannot overwrite an existing input location; use a prepared submit or a fresh path', {'step_id': step_id, 'input_dir': str(directory)})
        if args.get('job_work_dir'):
            owned_path(args['job_work_dir'])
        if args.get('output'):
            owned_path(args['output'])
        args['input_dir'] = str(directory)
        bindings = copy.deepcopy(node.get('input_bindings') or {})
        if any(key not in {'cif_dir', 'cif_path'} for key in bindings):
            raise WorkflowContractError('pipeline bindings must refer to source CIFs, not an already prepared input directory', {'step_id': step_id})
        prepare_args = {key: copy.deepcopy(value) for key, value in args.items() if key in _CDFT_PREPARATION_ARGUMENTS}
        prepare_args['action'] = 'inputs'
        prepare = {'step_id': prepare_id, 'agent': node.get('agent'), 'tool': 'run_cdft',
            'arguments': prepare_args, 'depends_on': dependencies,
            'expected_outputs': [{'kind': 'directory', 'path': str(directory), 'pattern': '*.dat', 'min_count': 1}],
            'resource_locks': copy.deepcopy(node.get('resource_locks', [])),
            'compiler_role': 'cdft_prepare', 'compiler_parent': step_id}
        if bindings:
            prepare['input_bindings'] = copy.deepcopy(bindings)
        args['action'] = 'submit'
        args.pop('resource_review_id', None)  # New execution contract requires its own native review.
        node['depends_on'] = list(dict.fromkeys(dependencies + [prepare_id]))
        node['input_bindings'] = {**bindings, 'input_dir': {'step_id': prepare_id, 'output_index': 0}}
        node['compiler_role'] = 'cdft_submit'
        node['compiler_prepare'] = prepare_id
        result.extend([{'operation': 'upsert', 'step_id': prepare_id, 'node': prepare}, change])
        receipts.append({'source': 'workflow_compiler', 'parameter': 'cdft_pipeline_expansion',
            'step_id': step_id, 'tool': 'run_cdft', 'prepare_step_id': prepare_id,
            'original_depends_on': dependencies, 'original_arguments': original_args,
            'original_expected_outputs': original_outputs, 'value': str(directory)})
    return result, receipts


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
            or name in (result.get('input_bindings') or {})
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


def _expects_directory(target, consumer_tool, arguments):
    """Return whether a path argument consumes a directory artifact.

    ``validate_framework_charges`` keeps the historical ``cif_path`` name for
    both single-CIF and batch-directory validation.  An explicit report
    destination is the schema-level signal for its batch form.
    """
    return target in _DIRECTORY_ARGUMENTS or (
        consumer_tool == "validate_framework_charges"
        and target == "cif_path"
        and bool(arguments.get("output_path"))
    )


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
                        conversation_root, consumer_arguments=None):
    from .output_contract import normalize_output, output_path

    outputs = node.get("expected_outputs", [])
    if (isinstance(output_index, bool) or not isinstance(output_index, int)
            or output_index < 0 or output_index >= len(outputs)):
        return None
    output = outputs[output_index]
    contract = normalize_output(output)
    arguments = node.get("arguments", {})
    consumer_arguments = consumer_arguments or {}
    base = (arguments.get("output_dir") or arguments.get("job_work_dir")
            or arguments.get("work_dir") or conversation_root)
    path = output_path(output, base)
    wildcard = contract["kind"] == "file" and any(char in path.name for char in "*?[")
    if _expects_directory(target, consumer_tool, consumer_arguments):
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
        if (node.get('tool')=='convert_physical_units' and 'value' in arguments
                and any((nodes.get(dep) or {}).get('tool')=='run_analysis_sandbox' for dep in ancestors[step_id])
                and 'value' not in (node.get('input_bindings') or {})):
            raise WorkflowContractError('conversion depending on sandbox data has an unbound literal value',{
                'step_id':step_id,'parameter':'value','contract_code':'sandbox_literal_conversion','executed':False,
                'next_action':'If converting a derived result, bind value to its required typed JSON property using input_bindings. If the user explicitly supplied an independent literal, preserve it and do not make that conversion depend on unrelated sandbox output.'})
        if node.get('tool')=='run_analysis_sandbox':
            from .output_contract import normalize_output,output_path
            from .sandbox_planning_contract import obvious_sandbox_path_issues
            path_issues=obvious_sandbox_path_issues(arguments.get('script'))
            if path_issues:
                raise WorkflowContractError('sandbox script references a path unavailable in the isolated environment',{
                    'step_id':step_id,'contract_code':'sandbox_script_path','executed':False,
                    'issues':path_issues,'next_action':'Read selected files through their /inputs paths and write each declared data file directly under /output; preserve original files and use the real tool schema.'})
            declared=arguments.get('declared_outputs') or []
            directory=arguments.get('output_dir')
            if (not isinstance(directory,str) or not Path(directory).is_absolute()
                    or not isinstance(declared,list) or not 1<=len(declared)<=8
                    or any(not isinstance(x,dict) or not isinstance(x.get('name'),str)
                        or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_.-]{0,127}',x['name']) for x in declared)):
                raise WorkflowContractError('data sandbox requires literal owned output declarations',{'step_id':step_id})
            expected={str(Path(directory)/'artifacts'/item['name']) for item in declared}
            outputs=node.get('expected_outputs') or []
            try:
                actual={str(output_path(item,conversation_root)) for item in outputs
                        if normalize_output(item)['kind']=='file'}
            except (ValueError,TypeError,KeyError):actual=set()
            if len(expected)!=len(declared) or len(outputs)!=len(expected) or actual!=expected:
                raise WorkflowContractError('data sandbox expected_outputs must exactly name declared artifacts',{
                    'step_id':step_id,'contract_code':'sandbox_output_declaration','executed':False,
                    'required_outputs':sorted(expected),
                    'next_action':'Use these exact file paths as expected_outputs; each corresponds to an explicitly declared output in the fresh sandbox directory. Do not use recursive globs or create the files during planning.'})
        properties = tool.input_schema.get("properties", {})
        required = set(tool.input_schema.get("required", []))
        bindings = node.get("input_bindings", {})
        if not isinstance(bindings, dict):
            raise WorkflowContractError("input_bindings must be an object", {
                "step_id": step_id, "tool": node.get("tool"),
                "next_action": "Map each consumer argument to {step_id, output_index}.",
            })

        for target, binding in bindings.items():
            if isinstance(binding,dict) and 'json_pointer' in binding:
                from .analysis_artifact_binding import declared_scalar_schema
                from .output_contract import normalize_output,output_path
                producer_id=binding.get('step_id');index=binding.get('output_index');kind=binding.get('value_type')
                spec=properties.get(target) or {};producer=nodes.get(producer_id) or {};producer_args=producer.get('arguments') or {}
                if target in arguments:
                    raise WorkflowContractError('invalid or unapproved sandbox scalar binding: explicit parameter conflict',{
                        'step_id':step_id,'parameter':target,'contract_code':'sandbox_scalar_conflict',
                        'executed':False,'next_action':'An explicit value and a JSON Pointer cannot both supply this parameter. Preserve user-specified scientific constraints: keep the explicit value and remove the binding, or remove only a redundant model-supplied value when the user authorized data-derived binding.'})
                if (set(binding)!={'step_id','output_index','json_pointer','value_type'}
                        or producer_id not in ancestors[step_id] or type(index)is not int
                        or not 0<=index<len(producer.get('expected_outputs') or [])
                        or producer.get('tool')!='run_analysis_sandbox' or target in PATH_ARGUMENTS
                        or target in {'script','command','code','task','query','shell','environment','memory_mb','nodelist','partition','num_processes','cpus_per_task','resource_review_id'}
                        or spec.get('type') not in {'number','integer','string','boolean'}
                        or not (kind==spec.get('type') or kind=='integer' and spec.get('type')=='number')
                        or kind=='string' and (not isinstance(spec.get('enum'),list) or not spec['enum'])):
                    raise WorkflowContractError('invalid or unapproved sandbox scalar binding',{'step_id':step_id,'parameter':target})
                output=producer['expected_outputs'][index]
                if normalize_output(output)['kind']!='file':
                    raise WorkflowContractError('sandbox scalar binding requires a JSON artifact file',{'step_id':step_id})
                path=output_path(output,conversation_root)
                declared=[x for x in producer_args.get('declared_outputs',[]) if isinstance(x,dict) and x.get('name')==path.name and x.get('format')=='json']
                if len(declared)!=1 or not isinstance(declared[0].get('json_schema'),dict):
                    raise WorkflowContractError('sandbox scalar binding requires a declared JSON schema',{'step_id':step_id})
                try:declared_scalar_schema(declared[0]['json_schema'],binding['json_pointer'],kind)
                except (ValueError,TypeError,KeyError) as error:
                    raise WorkflowContractError('sandbox scalar pointer is not guaranteed by its output schema',{'step_id':step_id,'parameter':target}) from error
                continue
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
                node.get("tool", ""), conversation_root, arguments)
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
                            node.get("tool", ""), conversation_root, arguments)
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
                             existing_steps=(), *, target_plan_version=None, source_evidence=None):
    """Run the complete mechanical DAG compilation pipeline once."""
    from .workflow_patch import (
        WorkflowContractError,
        canonical_arguments,
        canonical_expected_outputs,
        complete_compute_contract,
        resolve_workflow_placeholders,
    )

    report_receipts = []
    compiled = expand_parallel_report_nodes(
        changes, conversation_root, existing_steps=existing_steps,
        target_plan_version=target_plan_version, source_evidence=source_evidence,
        expansion_receipts=report_receipts)
    compiled = resolve_workflow_placeholders(
        compiled, conversation_root, existing_steps=existing_steps)
    for change in compiled:
        if change.get("operation") != "upsert" or not change.get("node"):
            continue
        node = change["node"]
        node["arguments"] = canonical_arguments(
            node.get("arguments", {}), project_root, conversation_root)
        node["expected_outputs"] = canonical_expected_outputs(
            node.get("expected_outputs", []), project_root, conversation_root)
        if node.get('tool') == 'run_cdft' and node['arguments'].get('action') == 'pipeline':
            from .output_contract import normalize_output
            outputs = [normalize_output(value) for value in node['expected_outputs']]
            if outputs:
                if (len(outputs) != 1 or outputs[0]['kind'] != 'file'
                        or any(char in outputs[0]['path'] for char in '*?[')
                        or node['arguments'].get('output') not in (None, outputs[0]['path'])):
                    raise WorkflowContractError('cDFT pipeline expansion requires one consistent final output file', {'step_id': change['step_id']})
                node['arguments']['output'] = outputs[0]['path']
        change["node"] = complete_compute_contract(
            node, change["step_id"], project_root, conversation_root)

    compiled, expansion_receipts = expand_cdft_pipeline_nodes(
        compiled, conversation_root, existing_steps, source_evidence=source_evidence)
    compiled = compile_dataflow_bindings(
        compiled, registry, project_root, conversation_root, existing_steps)
    defaults_applied = list(expansion_receipts)
    for change in compiled:
        if change.get("operation") != "upsert" or not change.get("node"):
            continue
        node, applied = materialize_safe_defaults(
            change["node"], registry, change["step_id"])
        change["node"] = node
        defaults_applied.extend(applied)
        scalar={k for k,v in (node.get('input_bindings') or {}).items() if isinstance(v,dict) and 'json_pointer' in v}
        tool=registry.get(node.get('tool',''))
        deferred=scalar & set((tool.input_schema if tool else {}).get('required',[]))
        issues = registry.validate_plan_params(node.get('tool',''),node.get('arguments',{}),deferred) if deferred else registry.validate_params(node.get("tool", ""), node.get("arguments", {}))
        if issues:
            tool = registry.get(node.get("tool", ""))
            receipt = registry._validation_receipt(
                node.get("tool", ""), issues, tool.input_schema if tool else {})
            raise WorkflowContractError("workflow tool arguments failed compilation", {
                **receipt,
                "step_id": change["step_id"],
                "agent": node.get("agent"),
            })
    by_id = {change['step_id']: {'step_id': change['step_id'], **change['node']}
             for change in compiled if change.get('operation') == 'upsert'}
    for receipt in report_receipts:
        receipt['compiled_nodes'] = copy.deepcopy({
            key: by_id[key] for key in [receipt['step_id'], *receipt['fragment_step_ids']]})
    defaults_applied.extend(report_receipts)
    return WorkflowCompilation(compiled, defaults_applied)


def draft_report_coverage(nodes, outlined, compiler_receipts):
    """Project exact compiler-owned report expansions for draft coverage only.

    The executable graph is unchanged. Receipts must come directly from the
    current compilation, and every generated contract must still match exactly.
    Model role labels, stale receipts, or extra nodes cannot exempt draft coverage.
    """
    compiled = {node.get('step_id'): node for node in nodes}
    for receipt in compiler_receipts:
        if (receipt.get('source') != 'workflow_compiler'
                or receipt.get('parameter') != 'report_pipeline_expansion'):
            continue
        parent_id = receipt.get('step_id')
        fragments = receipt.get('fragment_step_ids') or []
        expected = receipt.get('compiled_nodes') or {}
        keys = [parent_id, *fragments]
        if (not fragments or parent_id not in outlined
                or len(set(keys)) != len(keys)
                or any(key in outlined for key in fragments)
                or set(expected) != set(keys)
                or any(compiled.get(key) != expected[key] for key in keys)):
            continue
        parent = compiled[parent_id]
        sources = receipt.get('original_depends_on') or []
        if (parent.get('tool') != 'generate_scientific_report'
                or parent.get('report_role') != 'assembly'
                or parent.get('arguments', {}).get('report_mode') != 'assembly'
                or parent.get('depends_on') != fragments
                or parent.get('arguments', {}).get('source_steps') != fragments
                or len(sources) != len(fragments)):
            continue
        if any(compiled[key].get('tool') != 'generate_scientific_report'
               or compiled[key].get('agent') != parent.get('agent')
               or compiled[key].get('report_role') != 'fragment'
               or compiled[key].get('report_parent') != parent_id
               or compiled[key].get('report_source') != source
               or compiled[key].get('depends_on') != [source]
               or compiled[key].get('arguments', {}).get('source_steps') != [source]
               or compiled[key].get('arguments', {}).get('report_mode') != 'fragment'
               for key, source in zip(fragments, sources)):
            continue
        for key in fragments:
            compiled.pop(key)
        compiled[parent_id] = {**parent, 'depends_on': list(sources)}
    return compiled
