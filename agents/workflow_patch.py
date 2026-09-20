"""Validation of versioned, executable DAG edits (not prose plans)."""
import copy
import json
import re
from pathlib import Path


class WorkflowContractError(ValueError):
    def __init__(self, message, details):
        super().__init__(message)
        self.details = {'error': message, 'error_kind': 'workflow_contract',
                        'executed': False, **details}


def canonical_arguments(arguments, project_root, relative_root=None):
    from .workspace import PATH_ARGUMENTS, resolve_project_path
    result = copy.deepcopy(arguments)
    base = Path(relative_root or project_root).resolve()
    for key in PATH_ARGUMENTS & set(result):
        value = result[key]
        if isinstance(value, str) and value:
            path = Path(value).expanduser()
            if path.is_absolute():
                result[key] = str(path.resolve())
            elif path.parts and path.parts[0] == 'runs':
                result[key] = str(resolve_project_path(value, project_root))
            else:
                result[key] = str((base / path).resolve())
    return result


_WORKFLOW_PLACEHOLDER = re.compile(r'\{([^{}]+)\}')


def resolve_workflow_placeholders(changes, conversation_root, existing_steps=()):
    """Resolve deterministic DAG path references before path validation.

    The planning prompt historically encouraged ``{session}`` and
    ``{step_id.output_dir}``, but the compiler passed those strings to
    ``Path.resolve`` literally.  Resolve both forms from the proposed DAG; an
    unknown reference is returned as a precise contract error instead of
    becoming a directory named ``{session}``.
    """
    result = copy.deepcopy(changes)

    def proposed_nodes():
        mapped = {step.get('step_id'): copy.deepcopy(step) for step in existing_steps or ()
                  if step.get('step_id') and step.get('status') != 'superseded'
                  and not step.get('branch_parent')}
        for change in result:
            if change.get('operation') == 'remove':
                mapped.pop(change.get('step_id'), None)
            elif change.get('operation') == 'upsert' and change.get('node'):
                mapped[change.get('step_id')] = change.get('node', {})
        return mapped

    nodes = proposed_nodes()

    def lookup(token, current):
        if token == 'session':
            return str(Path(conversation_root).resolve())
        if '.' in token:
            step_id, field = token.rsplit('.', 1)
            node = nodes.get(step_id) or {}
            value = node.get('arguments', {}).get(field)
            if isinstance(value, str) and value:
                return value
        value = current.get('arguments', {}).get(token)
        if isinstance(value, str) and value:
            return value
        return None

    def resolve(value, current):
        if isinstance(value, str):
            rendered = value
            for _ in range(12):
                unresolved = []
                def replace(match):
                    replacement = lookup(match.group(1), current)
                    if replacement is None:
                        unresolved.append(match.group(1))
                        return match.group(0)
                    return replacement
                updated = _WORKFLOW_PLACEHOLDER.sub(replace, rendered)
                if updated == rendered:
                    break
                rendered = updated
            return rendered
        if isinstance(value, list):
            return [resolve(item, current) for item in value]
        if isinstance(value, dict):
            return {key: resolve(item, current) for key, item in value.items()}
        return value

    # Multiple passes let an upstream argument itself contain {session}.
    for _ in range(12):
        before = json.dumps(result, sort_keys=True, default=str)
        for change in result:
            node = change.get('node') or {}
            if change.get('operation') == 'upsert':
                from .workspace import PATH_ARGUMENTS
                arguments = node.setdefault('arguments', {})
                for key in PATH_ARGUMENTS & set(arguments):
                    arguments[key] = resolve(arguments[key], node)
                node['expected_outputs'] = resolve(node.get('expected_outputs', []), node)
        nodes = proposed_nodes()
        if json.dumps(result, sort_keys=True, default=str) == before:
            break
    unresolved = []
    def collect_placeholders(value):
        if isinstance(value, str):
            unresolved.extend(_WORKFLOW_PLACEHOLDER.findall(value))
        elif isinstance(value, list):
            for item in value:
                collect_placeholders(item)
        elif isinstance(value, dict):
            for item in value.values():
                collect_placeholders(item)
    for change in result:
        if change.get('operation') != 'upsert':
            continue
        from .workspace import PATH_ARGUMENTS
        node = change.get('node', {})
        path_values = {key: value for key, value in node.get('arguments', {}).items()
                       if key in PATH_ARGUMENTS}
        collect_placeholders(path_values)
        collect_placeholders(node.get('expected_outputs', []))
    if unresolved:
        raise WorkflowContractError('workflow contains unresolved path placeholders', {
            'placeholders': sorted(set(unresolved)),
            'next_action': 'use {session}, {step_id.output_dir}, or {step_id.job_work_dir}; otherwise provide one concrete session-owned path',
        })
    return result


def canonical_expected_outputs(outputs, project_root, relative_root=None):
    """Normalize explicitly project-anchored output paths once.

    Bare names remain relative to the node artifact directory. Paths beginning
    with a known project workspace component (notably ``runs/...``) are
    project-relative and must not later be appended to ``output_dir`` again.
    """
    anchored = {'runs', 'data', 'tmp', 'output', 'outputs', 'reports', 'gcmc_output'}
    base = Path(relative_root or project_root).resolve()
    normalized = []
    for value in outputs or []:
        item = copy.deepcopy(value)
        raw = item if isinstance(item, str) else item.get('path')
        path = Path(str(raw)).expanduser()
        parts = path.parts
        if path.is_absolute() or relative_root is not None or (parts and parts[0] in anchored):
            if path.is_absolute():
                target = path
            elif parts and parts[0] == 'runs':
                target = Path(project_root) / path
            else:
                target = base / path
            resolved = str(target.resolve())
            if isinstance(item, str): item = resolved
            else: item['path'] = resolved
        normalized.append(item)
    return normalized


def operational_patch(existing_steps, changes, project_root=None):
    """Typed contract comparison, never classification of the user's words.

    Only node/partition moves with identical science and resource budgets are
    covered here. Changed methods, inputs, densities, budgets or dependencies
    remain explicit model-led user decisions.
    """
    old = {n['step_id']: n for n in existing_steps}
    # Paths, native artifact declarations, locks and scheduler allocations are
    # execution mechanics.  Correcting them inside the same agent/tool/data-flow
    # node does not alter method, gases, thermodynamic conditions or scope.
    # Input paths remain protected because changing cif_dir/input data changes
    # material scope.  Memory/CPU/walltime remain budgets.  Only derived output
    # locations and scheduler placement are mechanically repairable here.
    scheduling = {
        'partition', 'nodelist', 'resource_review_id', 'output_dir', 'work_dir',
        'job_work_dir', 'output_csv', 'output_markdown', 'output', 'output_path',
    }
    def output_semantics(values):
        from .output_contract import normalize_output
        shaped = []
        for value in values or []:
            contract = normalize_output(value)
            shaped.append({key: item for key, item in contract.items() if key != 'path'})
        return shaped
    for change in changes:
        before = old.get(change['step_id'])
        after = change.get('node', {})
        if change['operation'] != 'upsert' or not before or 'expected_outputs' not in before:
            return False
        # Agent ownership and dependency edges are orchestration metadata.  The
        # compiler separately verifies capabilities, cycles and producer/consumer
        # edges, so repairing them does not require another user confirmation.
        # A tool change remains protected because it may change the method.
        if before.get('tool') != after.get('tool'):
            return False
        if output_semantics(before.get('expected_outputs', [])) != output_semantics(after.get('expected_outputs', [])):
            return False
        science_before = {k: v for k, v in before.get('arguments', {}).items() if k not in scheduling}
        science_after = {k: v for k, v in after.get('arguments', {}).items() if k not in scheduling}
        if science_before != science_after:
            return False
    return bool(changes)


def execution_contract(node):
    """Executable fields only; a patch description cannot change node state."""
    return {key: copy.deepcopy(node.get(key, [] if key in
        {'depends_on', 'expected_outputs', 'resource_locks'} else {}
        if key in {'arguments', 'input_bindings'} else None))
        for key in ('agent', 'tool', 'arguments', 'input_bindings', 'depends_on',
                    'expected_outputs', 'resource_locks')}


def workflow_efficiency_issues(nodes, project_root=None):
    """Return structural DAG issues that require semantic recompilation.

    The model still decides the scientific plan.  These checks only reject
    objectively redundant dispatch contracts and missing artifact edges; they
    never infer a scientific method from words or turn resource contention into
    a fake dependency.
    """
    root = Path(project_root or '.').resolve()
    issues = []
    active = [node for node in nodes if node.get('status') != 'superseded' and not node.get('branch_parent')]
    by_id = {node.get('step_id'): node for node in active}

    # Exact execution duplicates are never useful DAG nodes. One owned node can
    # carry multiple internal tasks/job-array elements without another handoff.
    seen = {}
    for node in active:
        contract = execution_contract(node)
        contract.pop('depends_on', None)
        key = json.dumps(contract, sort_keys=True, separators=(',', ':'), default=str)
        if key in seen:
            issues.append({
                'kind': 'duplicate_dispatch', 'nodes': [seen[key], node.get('step_id')],
                'next_action': 'merge into one DAG node and one tool invocation; represent internal work as tasks or one scheduler array',
            })
        else:
            seen[key] = node.get('step_id')

    # run_gcmc_batch already accepts a gas array and submits one scheduler job.
    # Splitting only the gas list is therefore an objectively avoidable set of
    # delegations/submissions, not a matter of model preference.
    batch_groups = {}
    for node in active:
        if node.get('tool') != 'run_gcmc_batch':
            continue
        args = copy.deepcopy(node.get('arguments', {}))
        gases = tuple(sorted(str(gas) for gas in args.pop('gases', []) if str(gas)))
        # Output locations are execution bookkeeping, not a scientific reason
        # to split one CIF×gas batch into multiple submissions.
        for key in ('output_dir', 'output_csv', 'work_dir', 'job_work_dir', 'resource_review_id'):
            args.pop(key, None)
        signature = json.dumps(args, sort_keys=True, separators=(',', ':'), default=str)
        batch_groups.setdefault(signature, []).append((node.get('step_id'), gases))
    for group in batch_groups.values():
        if len(group) > 1:
            issues.append({
                'kind': 'fragmented_batch', 'tool': 'run_gcmc_batch',
                'nodes': [item[0] for item in group],
                'gases': sorted({gas for _, gases in group for gas in gases}),
                'next_action': 'use one run_gcmc_batch node with the combined gases array; it submits one scheduler job',
            })

    # GCMC scientific reduction has a typed evidence parser.  A free-form
    # shell descendant can emit placeholders, parse the wrong unit, or call an
    # uptake ratio IAST selectivity, so it is not an admissible replacement.
    for node in active:
        if node.get('tool') != 'run_bash':
            continue
        if any(by_id.get(parent, {}).get('tool') == 'run_gcmc_batch'
               for parent in node.get('depends_on', [])):
            issues.append({
                'kind': 'untyped_gcmc_analysis', 'nodes': [node.get('step_id')],
                'next_action': 'replace this node with analyze_gcmc_screening using the completed batch work_dir and session-owned CSV/Markdown outputs',
            })

    # An isotherm tool owns its complete pressure range. Multiple nodes with the
    # same scientific inputs except pressure partitioning multiply both handoff
    # and sbatch count and must be compiled as one range.
    isotherm_groups = {}
    for node in active:
        if node.get('tool') != 'run_gcmc_isotherm':
            continue
        args = copy.deepcopy(node.get('arguments', {}))
        pressure = {key: args.pop(key, None) for key in ('pressure_start', 'pressure_end', 'n_pressure_points')}
        for key in ('work_dir', 'job_work_dir', 'output', 'output_csv', 'resource_review_id'):
            args.pop(key, None)
        signature = json.dumps(args, sort_keys=True, separators=(',', ':'), default=str)
        isotherm_groups.setdefault(signature, []).append((node.get('step_id'), pressure))
    for group in isotherm_groups.values():
        if len(group) > 1:
            issues.append({
                'kind': 'fragmented_scan', 'tool': 'run_gcmc_isotherm',
                'nodes': [item[0] for item in group], 'pressure_segments': [item[1] for item in group],
                'next_action': 'use one isotherm node with pressure_start, pressure_end and n_pressure_points; never submit once per pressure point',
            })

    # Artifact flow is an objective data dependency. If a consumer explicitly
    # reads a path produced by another node, that producer must be an ancestor.
    def absolute(value):
        path = Path(str(value))
        return (path if path.is_absolute() else root / path).resolve()

    def produced_paths(node):
        result = []
        args = node.get('arguments', {})
        base = absolute(args.get('output_dir') or args.get('job_work_dir') or args.get('work_dir') or '.')
        from .output_contract import output_path
        for output in node.get('expected_outputs', []):
            try:
                result.append(output_path(output, base).resolve())
            except Exception:
                continue
        for key in ('output_dir', 'output_csv', 'output_markdown', 'output'):
            if args.get(key):
                result.append(absolute(args[key]))
        return result

    def consumed_paths(node):
        args = node.get('arguments', {})
        keys = ['cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'data_csv', 'model_dir']
        if node.get('tool') == 'analyze_gcmc_screening': keys.append('work_dir')
        return [absolute(args[key]) for key in
                keys
                if isinstance(args.get(key), str) and args[key]]

    def ancestors(step_id):
        result, pending = set(), list(by_id.get(step_id, {}).get('depends_on', []))
        while pending:
            parent = pending.pop()
            if parent in result:
                continue
            result.add(parent)
            pending.extend(by_id.get(parent, {}).get('depends_on', []))
        return result

    producers = {node.get('step_id'): produced_paths(node) for node in active}
    for consumer in active:
        upstream = ancestors(consumer.get('step_id'))
        for input_path in consumed_paths(consumer):
            for producer_id, outputs in producers.items():
                if producer_id == consumer.get('step_id') or producer_id in upstream:
                    continue
                if any(input_path == output or input_path.is_relative_to(output) or output.is_relative_to(input_path)
                       for output in outputs):
                    issues.append({
                        'kind': 'missing_data_dependency', 'producer': producer_id,
                        'consumer': consumer.get('step_id'), 'path': str(input_path),
                        'next_action': f'add {producer_id} to {consumer.get("step_id")}.depends_on; this is a real producer→consumer edge',
                    })
    return issues


def patched_graph(existing_steps, changes, project_root=None):
    def identity(value):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError('step_id/dependency must be a non-empty string without surrounding whitespace')
        return value

    nodes = {}
    for step in existing_steps:
        if step.get('status') == 'superseded' or step.get('branch_parent'):
            continue
        # Completed coordinator discovery is evidence, not an implicit worker
        # delegation. Retain it in the prior TaskLine history, not the new DAG.
        if ('expected_outputs' not in step and not step.get('job_ids') and not step.get('recovery_key')):
            from .parallel_workflow import READ_ONLY
            from .workspace import readonly_shell
            validation = step.get('validation', {})
            if (step.get('tool') in READ_ONLY or step.get('tool') in {'execute_workflow', 'propose_workflow_patch', 'task_line_query', 'request_user_decision'}
                    or validation.get('dispatch_not_entered') is True or validation.get('schema') == 'failed'
                    or step.get('tool') == 'run_bash' and readonly_shell(step.get('arguments', {}).get('command', ''))):
                continue
        key = identity(step.get('step_id'))
        if key in nodes:
            raise ValueError(f'duplicate existing step_id {key}')
        nodes[key] = copy.deepcopy(step)
    edited, changed = set(), set()
    for change in changes:
        key = identity(change.get('step_id'))
        if key in edited:
            raise ValueError(f'duplicate patch step_id {key}; submit one final edit per node')
        edited.add(key)
        if change['operation'] == 'remove':
            if key not in nodes:
                raise ValueError(f'cannot remove missing step_id {key}')
            del nodes[key]
            changed.add(key)
        elif change['operation'] == 'upsert':
            node = copy.deepcopy(change['node'])
            if node.get('step_id', key) != key:
                raise ValueError('node step_id disagrees with patch step_id')
            node['step_id'] = key
            previous = nodes.get(key)
            if previous is None or execution_contract(previous) != execution_contract(node):
                changed.add(key)
                nodes[key] = node
            else:
                # An identical upsert is not a retry or "mark complete".
                # Preserve its result/failed state and all descendant receipts.
                nodes[key] = {**previous, **node}
        else:
            raise ValueError('workflow patch operations must be upsert/remove')
    if not nodes:
        raise ValueError('workflow patch cannot remove the entire workflow')
    for key, node in nodes.items():
        from .output_contract import normalize_output
        outputs = node.get('expected_outputs', [])
        if not isinstance(outputs, list):
            raise ValueError(f'{key} expected_outputs must be a list of artifact contracts')
        for output in outputs:
            normalize_output(output)
        dependencies = node.get('depends_on', [])
        if not isinstance(dependencies, list):
            raise ValueError(f'{key} depends_on must be a list of step IDs')
        for dependency in dependencies:
            identity(dependency)
        if len(set(dependencies)) != len(dependencies):
            raise ValueError(f'{key} has duplicate dependencies')
    visiting, visited = set(), set()
    def visit(key):
        if key in visiting:
            raise ValueError('workflow dependency cycle')
        if key in visited:
            return
        visiting.add(key)
        for dependency in nodes[key].get('depends_on', []):
            if dependency not in nodes:
                raise ValueError(f'{key} references missing dependency {dependency}')
            visit(dependency)
        visiting.remove(key)
        visited.add(key)
    for key in nodes:
        visit(key)
    # A data edge must carry the actual producer directory and gas scope.
    # The evaluator5 graph approved inputs in one location but submit in a
    # nonexistent inputs_xe directory; textual dependency IDs did not catch it.
    from pathlib import Path
    def path(value):
        p = Path(value)
        return (p if p.is_absolute() else Path(project_root or '.') / p).resolve()
    def gases(arguments):
        return set(arguments.get('gases') or ([arguments['gas']] if arguments.get('gas') else []))
    for key, node in nodes.items():
        if node.get('tool') != 'run_cdft' or node.get('arguments', {}).get('action') != 'submit':
            continue
        ancestors = set()
        def collect(parent):
            if parent in ancestors: return
            ancestors.add(parent)
            for other in nodes[parent].get('depends_on', []): collect(other)
        for dependency in node.get('depends_on', []): collect(dependency)
        producers = [nodes[parent] for parent in ancestors if nodes[parent].get('tool') == 'run_cdft'
                     and nodes[parent].get('arguments', {}).get('action') == 'inputs']
        if not producers: continue  # explicit existing inputs are still supported
        arguments = node['arguments']
        candidates = [producer for producer in producers if gases(producer['arguments']) == gases(arguments)]
        if len(candidates) != 1:
            raise ValueError(f'{key}: submit needs exactly one upstream input producer for the same gas scope')
        source = candidates[0]['arguments']
        if not source.get('input_dir') or not arguments.get('input_dir') or path(source['input_dir']) != path(arguments['input_dir']):
            raise ValueError(f'{key}: inputs→submit must carry the same explicit input_dir, not a guessed directory')
        if source.get('temperature') is not None and arguments.get('temperature') is not None and source['temperature'] != arguments['temperature']:
            raise ValueError(f'{key}: inputs→submit temperature mismatch')
    # All descendants of a modified node must be recomputed, unless explicitly
    # removed. Their old outputs belong to the previous graph version.
    affected = set(changed)
    while True:
        additional = {k for k,n in nodes.items() if set(n.get('depends_on', [])) & affected}
        if additional <= affected:
            break
        affected |= additional
    return list(nodes.values()), affected


def complete_compute_contract(node, step_id, project_root, conversation_root):
    """Complete deterministic native output arguments without model retries."""
    from pathlib import Path
    import hashlib
    from .output_contract import output_path, normalize_output
    from fnmatch import fnmatchcase
    result = copy.deepcopy(node)
    if result.get('tool') == 'generate_scientific_report':
        args = result.setdefault('arguments', {})
        dependencies = list(result.get('depends_on', []))
        if args.get('source_steps') and dependencies:
            # The executor accepts direct dependencies only.  Evidence from
            # earlier ancestors is already represented by those dependency
            # receipts, so this normalization does not change the science.
            args['source_steps'] = dependencies
        if args.get('source_steps') and not args.get('output_path'):
            candidates = []
            for output in result.get('expected_outputs', []):
                contract = normalize_output(output)
                if contract['kind'] == 'file':
                    candidates.append(str(output_path(output, Path(conversation_root))))
            if len(candidates) == 1:
                args['output_path'] = candidates[0]
        return result
    if result.get('tool') != 'run_cdft' or result.get('arguments', {}).get('action', 'pipeline') not in {'pipeline', 'submit'}:
        return result
    args = result['arguments']
    if not args.get('job_work_dir'):
        args['job_work_dir'] = str(Path(conversation_root) / 'cdft' / 'nodes' / hashlib.sha256(step_id.encode()).hexdigest()[:24])
    base = Path(args['job_work_dir']).resolve()
    actual = Path(args.get('output') or base / 'results.csv')
    actual = (actual if actual.is_absolute() else Path(project_root) / actual).resolve()
    # pipeline/submit has one executor-defined final acceptance artifact.
    # Raw ``*.data`` files are solver internals in a timestamped child and must
    # never become an additional hard DAG condition guessed by the model.
    result['expected_outputs'] = [{'kind': 'file', 'path': str(actual)}]
    return result
