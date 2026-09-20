"""Cooperative, durable DAG executor. No free-form plan or automatic resubmit.

Node claims and global shared/exclusive resource leases commit together. Tool
threads own isolated Sessions; main chat and supervisor receive durable events.
Scheduler jobs keep leases after the submission function returns.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import uuid
import inspect
from concurrent.futures import ThreadPoolExecutor

from .goal_contract import GoalContract
from .recovery import RecoveryGate, failure_reason, input_fingerprint, is_submission, result_object
from .state_io import json_transaction, write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from .workflow_patch import patched_graph
from .output_contract import output_path, artifact_fact

READ_ONLY = frozenset({'discover_forcefield', 'inspect_forcefield', 'validate_framework_charges', 'convert_physical_units', 'get_tool_schema', 'read_file', 'grep_search', 'inspect_path', 'inspect_run',
    'query_literature', 'check_job', 'diagnose_job', 'list_my_jobs', 'resource_health', 'assess_job_resources', 'resource_review_decision'})
KINDS = {'generate_structure': 'structures', 'run_pacman_charge': 'charged',
    'stage_cif_subset': 'structures', 'analyze_gcmc_screening': 'gcmc_analysis',
    'run_henry': 'gcmc', 'run_gcmc_isotherm': 'gcmc', 'run_gcmc_batch': 'gcmc',
    'run_henry_chain': 'gcmc', 'run_isotherm_chain': 'gcmc', 'run_cdft': 'cdft',
    'run_pore_analysis': 'pore', 'run_md_optimize': 'md', 'run_xtb_optimize': 'xtb',
    'build_guest_forcefield': 'ff', 'run_vasp': 'vasp', 'run_string_tst': 'tst',
    'run_external_potential': 'vext', 'calc_binding_energy': 'binding'}
ACTIVE = {'running', 'waiting_jobs', 'waiting_prerequisite', 'uncertain','prefinish'}


def same_scientific_contract(before, after):
    """Derived lease paths are not a scientific request to rerun a node."""
    return ({k: v for k, v in before.items() if k != 'resources'} ==
            {k: v for k, v in after.items() if k != 'resources'})


def artifact_base(node, project_root, conversation_root, result=None):
    result, args = result or {}, node['arguments']
    value = (args.get('output_dir') or args.get('job_work_dir') or args.get('work_dir')
             or result.get('output_dir') or result.get('work_dir'))
    if node['tool'] == 'analyze_gcmc_screening' and args.get('output_csv'):
        value = str(Path(args['output_csv']).parent)
    # cDFT's preparation root is NOT its timestamped scheduler job folder.
    # Relative solver artifacts must follow the actual scheduler receipt.
    if node['tool'] == 'run_cdft' and args.get('action') in {'pipeline', 'submit'} and result.get('work_dir'):
        value = result['work_dir']
    if value is None:
        value = project_root if node['tool'] in READ_ONLY or node['tool'] == 'write_file' else Path(conversation_root) / KINDS.get(node['tool'], 'ml' if node['tool'].startswith('ml_') else node['tool'])
    base = Path(value)
    return (base if base.is_absolute() else Path(project_root) / base).resolve()


def execution_fingerprint(node, main):
    """Conservative cache identity: actual inputs plus executor/solver sources."""
    root = Path(main.config.project_root)
    for key in ('cif_dir', 'input_dir'):
        path = Path(node['arguments'].get(key) or '__no_input_directory__')
        path = path if path.is_absolute() else root / path
        if path.is_dir() and sum(1 for _, _entry in zip(range(513), path.rglob('*'))) > 512:
            return None  # Never assert an unchanged dataset beyond the bounded audit.
    tool = main.registry.get(node['tool'])
    try:
        executor_source = inspect.getsourcefile(tool.execute)
        executor_code = inspect.getsource(tool.execute)
    except (OSError, TypeError):
        return None
    sources = [Path(executor_source)] if executor_source else []
    sources += [root / name for name in ('agents/registry.py', 'agents/chain.py', 'agents/slurm.py', 'env/tools.json', 'env/tool_versions.json')]
    if node['tool'] == 'run_cdft':
        folder = root / 'tools/cdft/cDFT_Initialization'
        sources += list(folder.glob('*.py')) + [folder / 'cDFT/DM_cdft']
    hashes = {}
    for path in sources:
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b''): digest.update(chunk)
            hashes[str(path.resolve())] = digest.hexdigest()
    return hashlib.sha256(json.dumps({'inputs': input_fingerprint(node['arguments'], root),
        'code': executor_code, 'sources': hashes, 'tool_version': getattr(tool, 'version', None)}, sort_keys=True).encode()).hexdigest()


def node_path_manifest(node, project_root, conversation_root, result=None, result_ref=None):
    result, reference = result or {}, result_ref or {}
    args = node['arguments']
    def absolute(value):
        from .workspace import resolve_project_path
        return str(resolve_project_path(value, project_root))
    base = artifact_base(node, project_root, conversation_root, result)
    calculation = result.get('work_dir') or args.get('job_work_dir') or args.get('work_dir') or str(base)
    input_keys = ['cif', 'cif_path', 'cif_dir', 'input_dir', 'input_path', 'data_csv']
    if node['tool'] == 'analyze_gcmc_screening': input_keys.append('work_dir')
    inputs = [absolute(args[k]) for k in input_keys if args.get(k)]
    if result.get('input_dir'): inputs.append(absolute(result['input_dir']))
    outputs = [absolute(p) for p in (result.get('output_files') or result.get('charged_cifs') or [])]
    for key in ('output_csv', 'output_markdown', 'manifest'):
        if result.get(key): outputs.append(absolute(result[key]))
    return {'calculation_dir': absolute(calculation), 'result_dir': str(base), 'input_paths': sorted(set(inputs)),
        'expected_outputs': [str(output_path(p, base)) for p in node.get('expected_outputs', [])],
        'output_files': sorted(set(outputs)), 'checkpoint_path': reference.get('checkpoint_path'),
        'evidence_path': reference.get('evidence_path'), 'updated_at': time.time()}


def resources_for(node, project_root, conversation_root):
    tool, args = node['tool'], node['arguments']
    resources = {}
    def add(key, mode):
        if resources.get(key) != 'write':
            resources[key] = mode
    def path(value):
        p = Path(value)
        return 'path:' + str((p if p.is_absolute() else Path(project_root) / p).resolve())
    for field in ('cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'data_csv', 'trajectory_path'):
        if args.get(field):
            prepares_inputs = tool == 'run_cdft' and field == 'input_dir' and args.get('action', 'pipeline') in {'inputs', 'pipeline'}
            add(path(args[field]), 'write' if prepares_inputs else 'read')
    if tool in {'read_file', 'write_file', 'inspect_path', 'grep_search'} and args.get('path'):
        add(path(args['path']), 'write' if tool == 'write_file' else 'read')
    for field in ('output_dir', 'output_csv', 'output_markdown', 'output', 'output_path', 'work_dir', 'job_work_dir'):
        if args.get(field):
            analysis_input = tool == 'analyze_gcmc_screening' and field == 'work_dir'
            evidence_output = tool == 'validate_framework_charges' and field == 'output_path'
            add(path(args[field]), 'write' if evidence_output else 'read' if tool in READ_ONLY or analysis_input else 'write')
    base = artifact_base(node, project_root, conversation_root)
    for output in node.get('expected_outputs', []):
        target = output_path(output, base)
        if any(c in target.name for c in '*?['): target = target.parent
        source = args.get('path') if tool == 'read_file' else None
        is_read_source = source and path(source) == 'path:' + str(target)
        add('path:' + str(target), 'read' if is_read_source else 'write')
    explicit_scoped = (tool in {'run_cdft', 'run_xtb_optimize', 'generate_structure', 'build_guest_forcefield',
                                'run_gcmc_batch', 'stage_cif_subset', 'analyze_gcmc_screening', 'submit_job'}
                       and any(args.get(key) for key in ('job_work_dir', 'output_dir', 'work_dir')))
    if tool not in READ_ONLY and tool != 'write_file' and not explicit_scoped:
        add('path:' + str((Path(conversation_root) / KINDS.get(tool, 'ml' if tool.startswith('ml_') else tool)).resolve()), 'write')
    if tool in {'run_henry_chain', 'run_isotherm_chain'}:
        add('path:' + str((Path(conversation_root) / 'charged').resolve()), 'write')
    if tool == 'build_guest_forcefield':
        # Legacy LigParGen discovers new outputs in a global /tmp namespace.
        add('named:ligpargen-global-temporary-output', 'write')
    for name in node.get('resource_locks', []):
        if not isinstance(name, str) or not name.strip(): raise ValueError('resource locks require non-empty names')
        if name.startswith('global:'):
            add('named:' + name.removeprefix('global:'), 'write')
        elif name.startswith('path:'):
            add(path(name.removeprefix('path:')), 'write')
        else:
            scope = hashlib.sha256(str(Path(conversation_root).resolve()).encode()).hexdigest()[:24]
            add('named:' + scope + ':' + name, 'write')
    if tool == 'run_bash':
        # Known scoped shell needs only an own-session lease. Unknown contracts
        # retain the conservative whole-project lease.
        shell_root = Path(project_root).resolve()
        try:
            parts = Path(conversation_root).resolve().relative_to(shell_root / 'runs').parts
            if len(parts) == 2:
                from .workspace import scoped_shell_contract
                _, issues = scoped_shell_contract(args.get('command', ''), args.get('cwd', ''), *parts, shell_root)
                if not issues: shell_root = Path(conversation_root).resolve()
        except ValueError:
            pass
        add('path:' + str(shell_root), 'write')
    return [{'key': key, 'mode': mode} for key, mode in sorted(resources.items())]


def conflicts(a, b):
    if a['mode'] == b['mode'] == 'read': return False
    if a['key'].startswith('path:') and b['key'].startswith('path:'):
        pa, pb = Path(a['key'][5:]), Path(b['key'][5:])
        return pa == pb or pa in pb.parents or pb in pa.parents
    return a['key'] == b['key']


def runtime_summary(state, limit=24):
    if not state: return {}
    counts, nodes = {}, {}
    ordered = sorted(state['nodes'].items(), key=lambda pair: (pair[1]['status'] == 'succeeded', -pair[1].get('updated_at', 0)))
    for _, node in ordered: counts[node['status']] = counts.get(node['status'], 0) + 1
    for key, node in ordered[:limit]:
        nodes[key] = {'status': node['status'], 'phase':node.get('phase'), 'agent':node.get('contract',{}).get('agent'), 'tool':node.get('contract',{}).get('tool'), 'result_ref': node.get('result_ref'),
                     'job_ids': node.get('job_ids', []), 'path_manifest': node.get('path_manifest'),
                     'error': str(node.get('error', ''))[:500],
                     'message_ids': list(node.get('messages', {}))[-4:]}
    return {'status': state['status'], 'plan_version': state['plan_version'], 'user_paused': state.get('user_paused', False), 'counts': counts,
            'nodes': nodes, 'omitted_nodes': max(0, len(ordered)-limit), 'details': 'query conversation workflow API or authoritative files'}


class WorkflowStore:
    def __init__(self, path, max_running=None):
        self.path = Path(path)
        self.max_running = int(max_running if max_running is not None else os.environ.get('BIMEM_WORKFLOW_MAX_RUNNING', '8'))
        if not 1 <= self.max_running <= 64: raise ValueError('workflow running limit must be in [1, 64]')

    @staticmethod
    def identity(username, conv_id):
        return hashlib.sha256(json.dumps([username, conv_id]).encode()).hexdigest()[:24]

    def snapshot(self, workflow_id=None):
        data = json.loads(self.path.read_text()) if self.path.exists() else {}
        if not isinstance(data, dict): raise ValueError('workflow state is corrupt')
        return copy.deepcopy(data.get('workflows', {}).get(workflow_id, {})) if workflow_id else data

    def activate(self, workflow_id, nodes, goal, username, conv_id, reusable_steps=()):
        with json_transaction(self.path) as data:
            workflows = data.setdefault('workflows', {})
            old = workflows.get(workflow_id, {})
            if old and (old['username'] != username or old['conv_id'] != conv_id):
                raise PermissionError('workflow belongs to a different user/session')
            if old.get('plan_version') == goal['approved_plan_version'] and old.get('nodes'):
                if old.get('pause_kind') == 'user_change':
                    raise ValueError('pending node change requires a revised approved plan, not a resume of the old contracts')
                if old['goal_contract']['version'] != goal['version']:
                    scope = ('method', 'gases', 'parameters', 'execution_mode', 'approved_plan_version')
                    incoming = {n['step_id']: n for n in nodes}
                    if (any(old['goal_contract'].get(k) != goal.get(k) for k in scope)
                            or set(incoming) != set(old['nodes'])
                            or any(not same_scientific_contract(n['contract'], incoming[k]) for k, n in old['nodes'].items())):
                        raise ValueError('goal changed; approve a new plan version before execution')
                    old.setdefault('goal_reconciliation_history', []).append({'from_version': old['goal_contract']['version'],
                        'to_version': goal['version'], 'basis': 'main model resumed exact approved contracts and unchanged protected science', 'time': time.time()})
                    old['goal_contract'] = copy.deepcopy(goal)
                if not any(n['status'] in {'failed', 'uncertain', 'validation_failed', 'needs_resources'} for n in old['nodes'].values()):
                    old['status'] = 'completed' if all(n['status'] == 'succeeded' for n in old['nodes'].values()) else 'active'
                return copy.deepcopy(old)
            incoming={n['step_id']:n for n in nodes}
            for key, prior in old.get('nodes',{}).items():
                if prior.get('status') in ACTIVE and incoming.get(key)!=prior.get('contract'):
                    raise ValueError('resolve running/unknown owners before changing/removing their node; unchanged live branches may carry forward')
            states = {}
            for node in nodes:
                prior = old.get('nodes', {}).get(node['step_id'], {})
                same = prior.get('contract') == node
                cached = (node['step_id'] in reusable_steps and prior.get('status') == 'succeeded'
                          and same_scientific_contract(prior.get('contract', {}), node))
                unresolved = (same_scientific_contract(prior.get('contract', {}), node)
                    and prior.get('status') in {'failed', 'validation_failed', 'needs_resources'})
                states[node['step_id']] = copy.deepcopy(prior) if cached or unresolved or same and prior.get('status') in ACTIVE else {
                    'contract': copy.deepcopy(node), 'status': 'pending', 'messages': copy.deepcopy(prior.get('messages', {})),
                    'attempt_history': copy.deepcopy(prior.get('attempt_history', [])) +
                        ([{k: copy.deepcopy(v) for k, v in prior.items() if k != 'attempt_history'}] if prior else [])}
                if cached or unresolved: states[node['step_id']]['contract'] = copy.deepcopy(node)
            history = copy.deepcopy(old.get('plan_history', []))
            if old: history.append({'plan_version': old['plan_version'], 'goal_contract': old['goal_contract'],
                                    'nodes': copy.deepcopy(old['nodes'])})
            workflows[workflow_id] = {'username': username, 'conv_id': conv_id,
                'plan_version': goal['approved_plan_version'], 'goal_contract': copy.deepcopy(goal),
                'status': ('needs_user' if any(n['status'] in {'failed', 'validation_failed', 'needs_resources', 'uncertain'} for n in states.values())
                           else 'completed' if all(n['status'] == 'succeeded' for n in states.values()) else 'active'),
                'user_paused': old.get('user_paused', False),
                'nodes': states, 'created_at': time.time(), 'outbox': copy.deepcopy(old.get('outbox', {})), 'plan_history': history}
            return copy.deepcopy(workflows[workflow_id])

    def claim(self, workflow_id, step_id):
        with json_transaction(self.path) as data:
            workflow = data.get('workflows', {}).get(workflow_id, {})
            node = workflow.get('nodes', {}).get(step_id)
            if workflow.get('status') != 'active' or workflow.get('user_paused') or not node or node['status'] != 'pending': return None
            running = sum(n['status'] == 'running' for w in data.get('workflows', {}).values() for n in w['nodes'].values())
            if running >= self.max_running: return None
            if any(workflow['nodes'][dep]['status'] != 'succeeded' for dep in node['contract'].get('depends_on', [])): return None
            leases = data.setdefault('leases', {})
            resources = node['contract']['resources']
            if any(conflicts(a, b) for lease in leases.values() for a in resources for b in lease['resources']): return None
            token = uuid.uuid4().hex
            if node.get('token'):
                node.setdefault('attempt_history', []).append({key: copy.deepcopy(node.get(key)) for key in
                    ('token', 'status', 'job_ids', 'result_ref', 'error', 'origin_goal_version')})
            for key in ('tool_returned', 'jobs_confirmed_terminal', 'job_ids', 'result', 'result_ref', 'evidence_call',
                        'recovery_key', 'attempt_id', 'output_baseline', 'error', 'artifacts', 'resolution', 'checkpoint_path'):
                node.pop(key, None)
            node.update(status='running', token=token, owner_pid=os.getpid(), started_at=time.time(), heartbeat=time.time(),
                        origin_goal_version=workflow['goal_contract']['version'], execution_plan_version=workflow['plan_version'],
                        dispatch_phase='preflight', tool_returned=False, jobs_confirmed_terminal=False, job_ids=[])
            leases[token] = {'workflow_id': workflow_id, 'step_id': step_id,
                             'resources': copy.deepcopy(resources), 'owner_pid': os.getpid()}
            return copy.deepcopy(node)

    def claim_blockers(self, workflow_id, step_id):
        """Explain why a pending node cannot currently acquire a lease."""
        data = self.snapshot()
        workflow = data.get('workflows', {}).get(workflow_id, {})
        node = workflow.get('nodes', {}).get(step_id)
        if not node:
            return ['runtime node is missing']
        blockers = []
        if workflow.get('status') != 'active':
            blockers.append('workflow status is ' + str(workflow.get('status')))
        if workflow.get('user_paused'):
            blockers.append('workflow is paused by user control')
        if node.get('status') != 'pending':
            blockers.append('node status is ' + str(node.get('status')))
        running = sum(
            item.get('status') == 'running'
            for candidate in data.get('workflows', {}).values()
            for item in candidate.get('nodes', {}).values()
        )
        if running >= self.max_running:
            blockers.append(f'global worker capacity is full ({running}/{self.max_running})')
        waiting = [dep for dep in node.get('contract', {}).get('depends_on', [])
                   if workflow.get('nodes', {}).get(dep, {}).get('status') != 'succeeded']
        if waiting:
            blockers.append('dependencies not succeeded: ' + ', '.join(waiting))
        resources = node.get('contract', {}).get('resources', {})
        conflicts_found = sorted({
            f'{requested} conflicts with {held}'
            for lease in data.get('leases', {}).values()
            for requested in resources
            for held in lease.get('resources', {})
            if conflicts(requested, held)
        })
        if conflicts_found:
            blockers.append('resource lease conflict: ' + '; '.join(conflicts_found))
        return blockers or ['node was not claimed; retry executor tick']

    def update(self, workflow_id, step_id, token, fields, release=False):
        with json_transaction(self.path) as data:
            workflow = data.get('workflows', {}).get(workflow_id, {})
            node = workflow.get('nodes', {}).get(step_id)
            if not node or node.get('token') != token: return False
            node.update(copy.deepcopy(fields))
            if node.get('status') == 'succeeded':
                workflow.get('branch_blockers', {}).pop(step_id, None)
            node['updated_at'] = time.time()
            if release: data.get('leases', {}).pop(token, None)
            if workflow.get('status') == 'needs_user' and workflow.get('pause_kind') == 'reconciliation' and not any(
                    n['status'] in {'uncertain', 'failed', 'validation_failed'} for n in workflow['nodes'].values()):
                workflow['status'] = 'active'
                workflow.pop('pause_reason', None)
                workflow.pop('pause_kind', None)
            if workflow.get('status') == 'active' and all(n['status'] == 'succeeded' for n in workflow['nodes'].values()):
                workflow['status'] = 'completed'
            return True

    def pause(self, workflow_id, reason, kind='failure', step_id=None):
        with json_transaction(self.path) as data:
            workflow = data.get('workflows', {}).get(workflow_id)
            if workflow and workflow['status'] != 'completed':
                if step_id is not None:
                    workflow.setdefault('branch_blockers', {})[step_id] = {'reason': reason, 'kind': kind}
                    independent = any(k != step_id and (
                        n['status'] in ACTIVE or n['status'] == 'pending' and all(
                            workflow['nodes'][dep]['status'] == 'succeeded'
                            for dep in n['contract'].get('depends_on', [])))
                        for k, n in workflow['nodes'].items())
                    retrying = any(n.get('resource_retry_at') for n in workflow['nodes'].values())
                    if independent or retrying:
                        return
                # A later recovery notification cannot overwrite a user change.
                if workflow.get('pause_kind') == 'user_change' and kind != 'user_change': return
                workflow.update(status='needs_user', pause_reason=reason, pause_kind=kind)

    def reconcile_goal(self, workflow_id, goal):
        """Metadata revisions may advance only with identical scientific contracts."""
        with json_transaction(self.path) as data:
            workflow = data['workflows'][workflow_id]
            saved = workflow['goal_contract']
            protected = ('method', 'gases', 'parameters', 'execution_mode', 'execution_authorized', 'approved_plan_version')
            incoming = {n['step_id']: n for n in goal.get('approved_nodes', [])}
            if (any(saved.get(k) != goal.get(k) for k in protected)
                    or set(incoming) != set(workflow['nodes'])
                    or any(not same_scientific_contract(n['contract'], incoming[k]) for k, n in workflow['nodes'].items())):
                return False
            workflow.setdefault('goal_reconciliation_history', []).append({
                'from_version': saved['version'], 'to_version': goal['version'],
                'basis': 'identical protected scope and approved node contracts', 'time': time.time()})
            workflow['goal_contract'] = copy.deepcopy(goal)
            return True

    def user_pause(self, workflow_id, paused=True):
        with json_transaction(self.path) as data:
            workflow = data.get('workflows', {}).get(workflow_id)
            if workflow: workflow['user_paused'] = bool(paused)

    def retire(self, workflow_id):
        """Authenticated conversation deletion cannot orphan a live writer."""
        with json_transaction(self.path) as data:
            workflow = data.get('workflows', {}).get(workflow_id)
            if not workflow: return
            if any(node['status'] in ACTIVE for node in workflow['nodes'].values()):
                raise ValueError('resolve running/unknown workflow owners before deleting the main conversation')
            workflow.update(status='retired_by_user', user_paused=True, retired_at=time.time())

    def enqueue_event(self, workflow_id, event_id, kind, payload):
        with json_transaction(self.path) as data:
            data['workflows'][workflow_id].setdefault('outbox', {}).setdefault(event_id, {
                'event_id': event_id, 'kind': kind, 'payload': copy.deepcopy(payload), 'sent': False})

    def event_sent(self, workflow_id, event_id):
        with json_transaction(self.path) as data:
            data['workflows'][workflow_id]['outbox'][event_id]['sent'] = True

    def send(self, workflow_id, step_id, text, kind, message_id=None):
        if kind not in {'status', 'comment', 'change'} or not text.strip(): raise ValueError('message requires text and status/comment/change kind')
        message_id = message_id or uuid.uuid4().hex
        with json_transaction(self.path) as data:
            workflow = data['workflows'][workflow_id]
            node = workflow['nodes'][step_id]
            messages = node.setdefault('messages', {})
            if message_id in messages:
                if messages[message_id]['text'] != text or messages[message_id]['kind'] != kind:
                    raise ValueError('message ID already belongs to different content')
                return copy.deepcopy(messages[message_id])
            message = {'message_id': message_id, 'step_id': step_id, 'text': text,
                       'kind': kind, 'time': time.time(), 'receipt': 'queued'}
            if kind == 'status': message.update(receipt='status_returned', node_status=node['status'])
            if kind == 'comment' and node['status'] in {'succeeded', 'failed', 'validation_failed'}:
                message['receipt'] = 'returned_to_main_chat_node_not_running'
            if kind == 'change':
                workflow.update(status='needs_user', pause_reason='user requested node contract change', pause_kind='user_change')
                message['receipt'] = 'awaiting_main_chat_approval'
            messages[message_id] = message
            return copy.deepcopy(message)

    def acknowledge(self, workflow_id, step_id, message_ids):
        with json_transaction(self.path) as data:
            messages = data['workflows'][workflow_id]['nodes'][step_id].get('messages', {})
            for mid in message_ids:
                if mid in messages and messages[mid]['kind'] != 'change':
                    messages[mid].update(receipt='seen_at_tool_boundary', received_at=time.time())


class ParallelWorkflow:
    def __init__(self, main, root, store, lifecycle, task_lines, username, conv_id, max_workers=4, job_watch=None,
                 on_change=None):
        self.main, self.root, self.store, self.lifecycle = main, Path(root), store, lifecycle
        self.task_lines, self.username, self.conv_id = task_lines, username, conv_id
        self.workflow_id = store.identity(username, conv_id)
        self.job_watch = job_watch
        self.on_change = on_change
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix='workflow-node')
        self.max_workers, self.futures, self.lock = max_workers, {}, threading.Lock()

    def snapshot(self): return self.store.snapshot(self.workflow_id)

    def _line(self):
        return self.task_lines.get_line(self.main._current_line_id, username=self.username, conv_id=self.conv_id) or {}

    def _step(self, step_id, **fields):
        self.task_lines.upsert_step(self.main._current_line_id, step_id, username=self.username, conv_id=self.conv_id, **fields)
        self.sync_chain('node_transition:' + step_id)

    def _step_metadata(self, step_id, token=None, **fields):
        """Mirror non-contract runtime details without blocking computation.

        The WorkflowStore update happens first and is authoritative.  TaskLine
        and orchestration-chain copies make resource decisions visible in the
        UI/audit trail, but a projection/schema mismatch must not prevent an
        already approved scientific tool from entering its dispatch path.
        """
        try:
            self._step(step_id, **fields)
            return True
        except Exception as error:
            if token:
                self.store.update(self.workflow_id, step_id, token, {
                    'metadata_sync_error': str(error),
                    'metadata_sync_fields': sorted(fields),
                    'metadata_sync_at': time.time(),
                })
            try:
                self.sync_chain('runtime_metadata_fallback:' + step_id)
            except Exception:
                pass
            return False

    def sync_chain(self, reason):
        from .orchestration_chain import sync_chain
        jobs = [j for j in self.job_watch.list() if (j.get('username'), j.get('conv_id')) == (self.username, self.conv_id)] if self.job_watch else None
        result = sync_chain(self.root, self._line, self.snapshot, reason, jobs=jobs)
        if self.on_change:
            try:
                self.on_change(reason)
            except Exception:
                pass
        return result

    def start(self, plan_version):
        from .defns import resolve_agent
        goal = self.main.goal_contract.to_dict()
        approved_contracts = goal.get('approved_nodes') or []
        read_only_authorized = (goal.get('execution_mode') in {'read_only', 'prepare_only'}
            and approved_contracts
            and all(not is_submission(node.get('tool'), node.get('arguments', {})) for node in approved_contracts))
        if (self.main._pending_workflow_patch or self.main._awaiting_plan_approval
                or not (goal.get('execution_authorized') or read_only_authorized)):
            raise ValueError('main chat must obtain explicit plan execution approval first')
        if plan_version != goal.get('approved_plan_version') or not goal.get('approved_nodes'):
            raise ValueError('execute requires the exact approved structured plan version')
        line = self._line()
        if int(line.get('plan_version', 0)) != plan_version: raise ValueError('TaskLine plan version differs from approval')
        approved, _ = patched_graph(goal['approved_nodes'], [], self.main.config.project_root)
        nodes = [{key: copy.deepcopy(n[key]) for key in ('step_id', 'agent', 'tool', 'arguments',
                  'depends_on', 'expected_outputs', 'resource_locks') if key in n} for n in approved]
        existing = self.snapshot()
        if self.main._on_resource_review and not any(n['status'] in ACTIVE for n in existing.get('nodes',{}).values()):
            unreviewed=[n for n in nodes if n['tool']=='run_cdft' and n['arguments'].get('action','pipeline') in {'pipeline','submit'} and not n['arguments'].get('resource_review_id')
                        and all(existing.get('nodes',{}).get(dep,{}).get('status')=='succeeded' for dep in n.get('depends_on',[]))]
            if unreviewed:
                from .workflow_patch import WorkflowContractError
                receipt=self.main._on_resource_review({'kind':'before_submission','nodes':unreviewed,'goal_version':goal['version']})
                if receipt.get('status')!='ready':
                    raise WorkflowContractError('resource agent requires review before execution',{'resource_review':receipt,'next_action':'main chat negotiates finite RAM budget and recompiles; no serial bypass'})
        if not existing and any(n.get('job_ids') for n in line.get('steps', [])):
            raise ValueError('legacy jobs require explicit reconciliation; runtime will not redispatch them')
        if not existing and any(n['tool'] not in READ_ONLY for n in nodes) and any(
            entry.get('status') in {'reserved', 'uncertain', 'submitted', 'completed_unverified'}
            for entry in self.main.recovery_gate.snapshot().values()):
            raise ValueError('legacy dispatch ledger has unresolved intent/results; reconcile before migrating mutations')
        for node in nodes:
            owner = resolve_agent(node['agent'])
            if node['tool'] not in owner.function_map() or owner.name in {'supervisor', 'monitor'}:
                raise ValueError('node tool/agent capability mismatch or lifetime observer used as worker')
            if node['tool'].startswith('handoff_to_') or node['tool'] not in self.main.registry.all_names():
                raise ValueError('runtime nodes require a concrete registered tool, not a handoff')
            issues = self.main.registry.validate_params(node['tool'], node['arguments'])
            from .workspace import tool_scope_issues
            issues += tool_scope_issues(node['arguments'], self.username, self.conv_id, self.main.config.project_root)
            base = artifact_base(node, self.main.config.project_root, self.root)
            for output in node.get('expected_outputs', []):
                issues += tool_scope_issues({'output': str(output_path(output, base))},
                    self.username, self.conv_id, self.main.config.project_root)
            if issues: raise ValueError(f'invalid node contract: {issues}')
            if not self.main.goal_contract.guard_tool_call(node['tool'], node['arguments'])[0]: raise ValueError('node changes approved goal')
            if is_submission(node['tool'], node['arguments']) and not node.get('expected_outputs'):
                raise ValueError('asynchronous compute nodes require explicit output contracts')
            node['resources'] = resources_for(node, self.main.config.project_root, self.root)
        reusable = set()
        incoming = {n['step_id']: n for n in nodes}
        for s in line.get('steps', []):
            prior = existing.get('nodes', {}).get(s['step_id'], {})
            unchanged_artifacts = prior.get('artifacts', {}) == self._artifacts(prior['contract'], prior.get('result', {})) if prior else False
            # A verified scheduler result is an existing scientific result, not
            # a request to recompute it when unrelated SDK source changes.
            # Reuse requires an identical scientific contract, immutable output
            # facts and an owned scheduler completion receipt.
            verified_submission = (prior.get('status') == 'succeeded' and prior.get('phase') == 'finish'
                and prior.get('node_verification') and prior.get('job_ids')
                and incoming.get(s['step_id']) and same_scientific_contract(prior.get('contract', {}), incoming[s['step_id']])
                and is_submission(prior.get('contract', {}).get('tool'), prior.get('contract', {}).get('arguments', {}))
                and self.job_watch and prior.get('artifacts')
                and all((j.get('username'), j.get('conv_id')) == (self.username, self.conv_id)
                        and j.get('terminal') and not j.get('failed') and j.get('state') == 'COMPLETED'
                        for j in [(self.job_watch.get(job_id) or {}) for job_id in prior['job_ids']]))
            if (s.get('done') and s.get('status') == 'completed' and s.get('validation', {}).get('output_contract') == 'passed'
                    and (verified_submission or prior.get('execution_fingerprint') and prior['execution_fingerprint'] == execution_fingerprint(prior['contract'], self.main))
                    and unchanged_artifacts):
                reusable.add(s['step_id'])
                if verified_submission:
                    self.store.update(self.workflow_id, s['step_id'], prior['token'], {'reuse_basis': {
                        'kind': 'verified_completed_submission', 'no_resubmission': True,
                        'original_execution_fingerprint': prior.get('execution_fingerprint'), 'checked_at': time.time()}})
        if any(prior.get('status') == 'succeeded'
               and is_submission(prior.get('contract', {}).get('tool'), prior.get('contract', {}).get('arguments', {}))
               and incoming.get(key) and same_scientific_contract(incoming[key], prior['contract']) and key not in reusable
               for key, prior in existing.get('nodes', {}).items()):
            raise ValueError('Previously completed scheduler result cannot be implicitly recomputed: reconcile changed artifacts or missing completion proof first.')
        if existing.get('plan_version') == plan_version and any(
            value['status'] == 'succeeded' and key not in reusable for key, value in existing.get('nodes', {}).items()):
            self.store.pause(self.workflow_id, 'cached inputs/tool sources/artifacts changed; approve revised plan', kind='input_changed')
            raise ValueError('cached inputs/tool sources/artifacts changed; approve a revised plan version before reuse')
        activated = self.store.activate(self.workflow_id, nodes, goal, self.username, self.conv_id, reusable)
        blocked = [key for key, value in activated['nodes'].items()
                   if value['status'] in {'failed', 'validation_failed', 'needs_resources', 'uncertain'}]
        if blocked:
            return {'status': 'needs_recovery', 'scheduled': False, 'blocked_nodes': blocked,
                    'reason': 'Unchanged failed nodes remain failed. Restore verified results or provide a verified execution repair; identical upsert is not a retry.',
                    'workflow_id': self.workflow_id, 'plan_version': plan_version}
        self.store.user_pause(self.workflow_id, False)
        self._emit('workflow_started', {'plan_version': plan_version}, f'start:{plan_version}')
        return {'status': 'scheduled', 'plan_version': plan_version, 'workflow_id': self.workflow_id,
                'delivery_owner': 'lead-orchestrator', 'negotiation_owner': 'lead-orchestrator'}

    def start_and_tick(self, plan_version, jobs=(), paused=False):
        """Activate a plan and prove that at least one root entered dispatch."""
        receipt = self.start(plan_version)
        if receipt.get('scheduled') is False:
            return receipt
        self.tick(jobs=jobs, paused=paused)
        state = self.snapshot()
        roots = {
            step_id: node for step_id, node in state.get('nodes', {}).items()
            if not node.get('contract', {}).get('depends_on')
        }
        dispatched = {
            step_id: {'status': node.get('status'), 'token': bool(node.get('token'))}
            for step_id, node in roots.items() if node.get('token')
        }
        if dispatched:
            return {**receipt, 'scheduled': True, 'dispatch_started': True,
                    'root_nodes': dispatched}
        blocked = {
            step_id: self.store.claim_blockers(self.workflow_id, step_id)
            for step_id in roots
        }
        return {**receipt, 'status': 'waiting_dispatch', 'scheduled': False,
                'dispatch_started': False, 'blocked_nodes': blocked,
                'reason': 'No root node acquired an execution lease.'}

    def revalidate_outputs(self,step_id,expected_outputs,completed_job_id=None):
        """Repair a stopped generator's artifact contract from actual proof."""
        from .output_contract import normalize_output
        from .slurm import check_job_status
        if completed_job_id:
            current = self.snapshot()['nodes'][step_id]
            current_job_ids = {str(value) for value in current.get('job_ids', [])}
            repairing_current_generator = (
                current['contract']['tool'] == 'generate_structure'
                and current['status'] in {'prefinish', 'validation_failed'}
                and str(completed_job_id) in current_job_ids
            )
            if not repairing_current_generator:
                return self._restore_verified_dataset(step_id, completed_job_id)
        state=self.snapshot();node=state['nodes'][step_id]
        args, result = node['contract']['arguments'], result_object(node.get('result', {}))
        native_output = bool(node['contract']['tool'] != 'generate_structure'
                             and result.get('job_id') and any(args.get(key) and result.get(key)
                             for key in ('output_csv', 'output_dir')))
        if node['contract']['tool'] == 'run_cdft' or native_output:
            if node['status'] not in {'prefinish', 'validation_failed'} or not node.get('jobs_confirmed_terminal'):
                raise ValueError('native output repair requires an execution-ended owned node')
            self._bind_native_result_path(step_id, node['token'], node.get('result', {}))
            current = self.snapshot()['nodes'][step_id]
            base = artifact_base(current['contract'], self.main.config.project_root, self.root, current.get('result'))
            if [str(output_path(o, base)) for o in expected_outputs] != [str(output_path(o, base)) for o in current['contract']['expected_outputs']]:
                raise ValueError('native path repair may only use the actual scheduler receipt CSV, not an arbitrary output')
            self._settle(step_id, current['token'], current.get('result', {}))
            self.main._checkpoint('native_output_contract_revalidated')
            return {'ok': True, 'step_id': step_id, 'status': self.snapshot()['nodes'][step_id]['status'],
                    'no_resubmission': True, 'actual_outputs': current['contract']['expected_outputs']}
        if node['status'] not in {'validation_failed','prefinish'} or node['contract']['tool']!='generate_structure':
            raise ValueError('only a stopped generator with output-validation failure may use this repair')
        if state['plan_version']!=self.main.goal_contract.approved_plan_version:
            raise ValueError('runtime/approved plan changed; reconcile before output repair')
        jobs=node.get('job_ids',[])
        if not jobs:raise ValueError('generator needs an actual linked scheduler receipt')
        for job_id in jobs:
            record=self.job_watch.get(job_id) or {}
            if (record.get('username'),record.get('conv_id'))!=(self.username,self.conv_id):raise PermissionError('foreign job')
            actual=({'status':'COMPLETED','terminal':True,'failed':False}
                    if record.get('terminal') and not record.get('failed') and record.get('state')=='COMPLETED'
                    else check_job_status(job_id,work_dir=record.get('work_dir','')))
            if not actual.get('terminal') or actual.get('failed') or actual.get('status')!='COMPLETED':
                raise ValueError('actual scheduler completion must be confirmed before output repair')
        base=artifact_base(node['contract'],self.main.config.project_root,self.root,node.get('result'))
        outputs=[normalize_output(value) for value in expected_outputs]
        target_count=int(node['contract']['arguments']['n_structures'])
        if (not outputs or sum(o.get('min_count',1) for o in outputs)<target_count
                or any(o['kind']!='directory' or not output_path(o,base).is_relative_to(base)
                       or not o.get('pattern','').endswith('.cif') for o in outputs)):
            raise ValueError('expected CIF directories must stay within the original dataset and cover the full requested count')
        revised=copy.deepcopy(node['contract']);revised['expected_outputs']=expected_outputs
        facts=self._artifacts(revised,node.get('result',{}))
        if not all(fact and fact['size']>0 and fact['mtime_ns']>=int(node['started_at']*1e9) for fact in facts.values()):
            raise ValueError('fresh nonempty complete artifacts required; no status-only repair')
        if len({item[0] for fact in facts.values() for item in fact['files']})<target_count:
            raise ValueError('overlapping contracts cannot double-count the same CIF')
        proof={'artifacts':facts,'job_ids':jobs,'execution_call_id':node['result_ref']['call_id'],'no_resubmission':True}
        self.task_lines.repair_output_contract(self.main._current_line_id,step_id,expected_outputs,proof,state['plan_version'])
        ledger=RecoveryGate(path=self.root/'recovery_state.json')
        if ledger.snapshot().get(node.get('recovery_key'),{}).get('status')=='completed_unverified':ledger.accept_result(node['recovery_key'],proof)
        self.store.update(self.workflow_id,step_id,node['token'],{'contract':revised,'output_contract_repair':proof})
        for approved in self.main.goal_contract.approved_nodes:
            if approved['step_id']==step_id:approved['expected_outputs']=copy.deepcopy(expected_outputs)
        # Reuse the normal verified-success settlement; never set done by text.
        self._settle(step_id,node['token'],node.get('result',{}))
        with json_transaction(self.store.path) as data:
            workflow=data['workflows'][self.workflow_id]
            if not workflow.get('user_paused') and workflow.get('pause_kind')=='failure' and not any(n['status'] in {'failed','uncertain','validation_failed'} for n in workflow['nodes'].values()):
                workflow['status']='active';workflow.pop('pause_reason',None);workflow.pop('pause_kind',None)
        self.main._checkpoint('verified_output_contract_repair')
        return {'ok':True,'step_id':step_id,'status':self.snapshot()['nodes'][step_id]['status'],'no_resubmission':True,
                'verified_files':sum(len(f['files']) for f in facts.values()),'next_action':'normal readiness queue may continue'}

    def _restore_verified_dataset(self, step_id, completed_job_id):
        """Adopt an archived verified dataset after a stopped duplicate attempt.

        Only exact original file identities survive; extra files from the
        stopped attempt are retained as evidence, not added to the dataset.
        """
        from .slurm import check_job_status
        state = self.snapshot()
        node = state['nodes'][step_id]
        if node['contract']['tool'] != 'generate_structure':
            return self._restore_verified_submission(step_id, completed_job_id)
        if node['status'] in ACTIVE:
            raise ValueError('only a stopped generator can restore its archived completed result')
        for job_id in node.get('job_ids', []):
            job = self.job_watch.get(job_id) or {}
            if (job.get('username'), job.get('conv_id')) != (self.username, self.conv_id):
                raise PermissionError('foreign attempt')
            if not job.get('terminal') or job.get('state') == 'UNKNOWN':
                raise ValueError('all current attempts must be confirmed stopped before restoring a prior result')
        completed_job_id = str(completed_job_id)
        job = self.job_watch.get(completed_job_id) or {}
        if (job.get('username'), job.get('conv_id')) != (self.username, self.conv_id):
            raise PermissionError('foreign completed job')
        actual = ({'status': 'COMPLETED', 'terminal': True, 'failed': False}
                  if job.get('terminal') and not job.get('failed') and job.get('state') == 'COMPLETED'
                  else check_job_status(completed_job_id, work_dir=job.get('work_dir', '')))
        if actual.get('status') != 'COMPLETED' or not actual.get('terminal') or actual.get('failed'):
            raise ValueError('original job must have confirmed successful completion')
        candidates = [n for version in self._line().get('plan_history', []) for n in version.get('steps', [])
            if n.get('step_id') == step_id and n.get('done') and n.get('status') == 'completed'
            and completed_job_id in n.get('job_ids', []) and n.get('validation', {}).get('output_contract') == 'passed'
            and n.get('arguments') == node['contract']['arguments']]
        archived = next((n for n in reversed(candidates) if n.get('output_contract_history')), None)
        if not archived: raise ValueError('no archived verified dataset with identical generation arguments')
        proof = archived['output_contract_history'][-1]['verification']
        files = {name: (size, mtime) for fact in proof['artifacts'].values() for name, size, mtime in fact['files']}
        if len(files) != int(node['contract']['arguments']['n_structures']):
            raise ValueError('archived proof must cover exactly the requested original dataset')
        base = artifact_base(node['contract'], self.main.config.project_root, self.root)
        for name, expected in files.items():
            path = Path(name)
            if not path.resolve().is_relative_to(base) or not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != expected:
                raise ValueError('original dataset file changed or escaped its owned root; no restoration')
        execution_id = proof['execution_call_id']
        if not isinstance(execution_id, str) or not execution_id.isalnum(): raise ValueError('invalid archived evidence identity')
        call = json.loads((self.root / 'evidence' / f'{execution_id}.json').read_text())
        result = result_object(call.get('result'))
        if call.get('failed') or call.get('tool') != 'generate_structure' or call.get('params') != node['contract']['arguments'] or str(result.get('job_id')) != completed_job_id:
            raise ValueError('actual original execution evidence does not match the dataset')
        frozen = sorted(files)
        restoration = {'completed_job_id': completed_job_id, 'execution_call_id': execution_id,
            'preserved_original_files': frozen, 'excluded_attempts': node.get('job_ids', []), 'no_resubmission': True}
        self.task_lines.repair_output_contract(self.main._current_line_id, step_id, frozen, restoration,
            state['plan_version'], restoration=restoration)
        revised = copy.deepcopy(node['contract'])
        revised['expected_outputs'] = frozen
        revised['resources'] = resources_for(revised, self.main.config.project_root, self.root)
        with json_transaction(self.store.path) as data:
            saved = data['workflows'][self.workflow_id]['nodes'][step_id]
            if saved['token'] != node['token'] or saved['status'] in ACTIVE or node['token'] in data.get('leases', {}):
                raise ValueError('attempt became active during restoration; refuse to unlock')
            previous = {k: copy.deepcopy(v) for k, v in saved.items() if k != 'attempt_history'}
            saved.clear()
            saved.update(contract=revised, status='prefinish', phase='prefinish', token=uuid.uuid4().hex,
                started_at=call['time'], origin_goal_version=self.main.goal_contract.version,
                execution_plan_version=state['plan_version'], job_ids=[completed_job_id], jobs_confirmed_terminal=True,
                result=result, result_ref={'call_id': execution_id, 'evidence_path': str(self.root/'evidence'/f'{execution_id}.json')},
                output_baseline={}, restoration=restoration, attempt_history=node.get('attempt_history', [])+[previous])
        for approved in self.main.goal_contract.approved_nodes:
            if approved['step_id'] == step_id: approved['expected_outputs'] = frozen
        pending = self.main._pending_workflow_patch
        if pending and step_id in pending.get('affected_nodes', []):
            self.main.context['invalidated_workflow_patch'] = {'plan_version': pending['new_version'],
                'reason': 'Archived result restoration changed the base node contract; the old draft cannot replace it.'}
            self.main._pending_workflow_patch = None
            if (self.main._pending_user_interaction or {}).get('tool') == 'workflow_patch_decision':
                self.main._pending_user_interaction = None
                self.main._waiting_for_user_input = False
        restored = self.snapshot()['nodes'][step_id]
        self._emit('worker_prefinish', {'step_id': step_id, 'restoration': restoration,
            'verification_owner': revised['agent'], 'main_already_handling': True}, restored['token'] + ':restored_prefinish')
        self.main._checkpoint('restored_original_verified_dataset')
        return {'ok': True, 'step_id': step_id, 'status': 'prefinish', 'verified_files': len(files),
                'accepted_completed_job_id': completed_job_id, 'no_resubmission': True,
                'next_action': 'Main model inspects the original dataset and finishes with actual fresh evidence; user pause remains in force.'}

    def _restore_verified_submission(self, step_id, completed_job_id):
        """Restore an unchanged, previously verified scheduler attempt."""
        state, completed_job_id = self.snapshot(), str(completed_job_id)
        node = state['nodes'][step_id]
        if node['status'] in ACTIVE:
            raise ValueError('current attempt must be confirmed stopped before restoring a prior result')
        if not is_submission(node['contract']['tool'], node['contract']['arguments']):
            raise ValueError('only scheduler submissions have restorable completed receipts')
        for job_id in node.get('job_ids', []):
            current_job = self.job_watch.get(job_id) or {}
            if (current_job.get('username'), current_job.get('conv_id')) != (self.username, self.conv_id):
                raise PermissionError('foreign current attempt')
            if not current_job.get('terminal') or current_job.get('state') == 'UNKNOWN':
                raise ValueError('all current attempts must be confirmed stopped before restoration')
        job = self.job_watch.get(completed_job_id) or {}
        if ((job.get('username'), job.get('conv_id')) != (self.username, self.conv_id)
                or not job.get('terminal') or job.get('failed') or job.get('state') != 'COMPLETED'):
            raise ValueError('original owned job must have confirmed successful completion')
        candidates = [attempt for attempt in reversed(node.get('attempt_history', []))
            if attempt.get('status') == 'succeeded' and attempt.get('phase') == 'finish'
            and completed_job_id in attempt.get('job_ids', []) and attempt.get('node_verification')
            and same_scientific_contract(attempt.get('contract', {}), node['contract'])]
        if not candidates:
            raise ValueError('no archived verified attempt with the unchanged scientific contract')
        archived = candidates[0]
        if archived.get('artifacts') != self._artifacts(archived['contract'], archived.get('result', {})):
            raise ValueError('archived result artifacts changed; restoration refused')
        call_id = archived.get('result_ref', {}).get('call_id')
        if not isinstance(call_id, str) or not call_id.isalnum():
            raise ValueError('archived execution evidence identity is invalid')
        call = json.loads((self.root / 'evidence' / f'{call_id}.json').read_text())
        receipt = result_object(call.get('result'))
        if (call.get('failed') or call.get('tool') != node['contract']['tool']
                or call.get('params') != node['contract']['arguments']
                or str(receipt.get('job_id')) != completed_job_id):
            raise ValueError('archived execution evidence does not match the current contract')
        restoration = {'completed_job_id': completed_job_id, 'execution_call_id': call_id,
                       'excluded_attempts': node.get('job_ids', []), 'no_resubmission': True}
        with json_transaction(self.store.path) as data:
            workflow = data['workflows'][self.workflow_id]
            saved = workflow['nodes'][step_id]
            if saved.get('status') in ACTIVE or saved.get('token') in data.get('leases', {}):
                raise ValueError('attempt became active during restoration')
            history = saved.get('attempt_history', []) + [{k: copy.deepcopy(v) for k, v in saved.items() if k != 'attempt_history'}]
            restored = copy.deepcopy(archived)
            restored.update(contract=copy.deepcopy(node['contract']), token=uuid.uuid4().hex,
                            origin_goal_version=self.main.goal_contract.version,
                            execution_plan_version=state['plan_version'], restoration=restoration,
                            attempt_history=history, updated_at=time.time())
            workflow['nodes'][step_id] = restored
            if not any(n.get('status') in {'failed', 'validation_failed', 'needs_resources', 'uncertain'}
                       for n in workflow['nodes'].values()):
                workflow['status'] = 'active'
                workflow.pop('pause_reason', None); workflow.pop('pause_kind', None)
        self._step(step_id, status='completed', done=True, job_ids=[completed_job_id],
                   validation={'output_contract': 'passed', 'restored_completed_job': completed_job_id},
                   output_files=archived.get('path_manifest', {}).get('output_files', []),
                   path_manifest=archived.get('path_manifest'))
        self.sync_chain('verified_submission_restored:' + step_id)
        return {'ok': True, 'step_id': step_id, 'status': 'succeeded',
                'accepted_completed_job_id': completed_job_id, 'no_resubmission': True,
                'next_action': 'dependency readiness queue may continue'}

    def finish_node(self,step_id,evidence_call_ids,conclusion):
        node=self.snapshot()['nodes'][step_id]
        if node['status'] == 'succeeded' and node.get('node_verification'):
            return {'ok': True, 'step_id': step_id, 'phase': 'finish', 'already_finished': True,
                    'verification': copy.deepcopy(node['node_verification']),
                    'next_action': 'Use the stored verification receipt; no new validation or dispatch was performed.'}
        if node['status']!='prefinish':raise ValueError('finish requires an execution-ended node awaiting verification')
        self._bind_native_result_path(step_id, node['token'], node.get('result', {}))
        node = self.snapshot()['nodes'][step_id]
        if not evidence_call_ids or len(conclusion.strip())<10:raise ValueError('real validation evidence and a factual conclusion required')
        base=artifact_base(node['contract'],self.main.config.project_root,self.root,node.get('result'))
        targets=[str(base)]+[str(output_path(value,base)) for value in node['contract'].get('expected_outputs',[])]
        linked_output_evidence=False
        validation_call_ids=[]
        execution_call_id=node.get('result_ref',{}).get('call_id')
        for call_id in evidence_call_ids:
            if not isinstance(call_id,str) or not call_id.isalnum():raise ValueError('invalid evidence identity')
            call=json.loads((self.root/'evidence'/f'{call_id}.json').read_text())
            # Models commonly include the node's own scheduler receipt next to
            # the real inspection call. It proves provenance but is not result
            # validation: tolerate only this exact persisted execution call and
            # require a separate fresh output-linked validator below.
            if call_id == execution_call_id:
                if call.get('tool') != node['contract']['tool'] or call.get('failed'):
                    raise ValueError('execution receipt identity does not match the node')
                continue
            if (call.get('time',0)<node['started_at'] or call.get('failed')
                    or call.get('tool') not in READ_ONLY|{'run_bash','run_cdft','validate_gcmc_results','revalidate_workflow_node_outputs'}):
                raise ValueError('verification must use fresh successful inspection/validation tools')
            validation_call_ids.append(call_id)
            linked_output_evidence |= any(target in str(value) for target in targets for value in call.get('params',{}).values())
            if call['tool']=='run_cdft' and call['params'].get('action')!='collect':raise ValueError('submission is not result validation')
            if call['tool']=='run_bash' and failure_reason(call.get('result')):raise ValueError('shell verification failed')
        if not linked_output_evidence:raise ValueError('at least one verification call must inspect the actual result root')
        facts=self._artifacts(node['contract'],node.get('result',{}))
        if not all(fact and fact['size']>0 for fact in facts.values()):raise ValueError('native output contract not met; repair/validate without resubmission')
        self.store.update(self.workflow_id,step_id,node['token'],{'node_verification':{'evidence_call_ids':validation_call_ids,
            'conclusion':conclusion,'verified_at':time.time(),'owner':node['contract']['agent']}})
        self._settle(step_id,node['token'],node.get('result',{}))
        self.main._checkpoint('model_verified_node_finish')
        next_state = 'user_paused' if self.snapshot().get('user_paused') else 'triggered'
        resume_error = None
        if self.snapshot().get('pause_kind') == 'user_change':
            next_state, resume_error = 'blocked', 'A requested node change is still awaiting a revised approved plan.'
        if (next_state == 'triggered' and (self.snapshot()['goal_contract']['version'] != self.main.goal_contract.version
                                            or self.snapshot()['status'] == 'needs_user')):
            try:
                # The model's verified finish is a readiness operation. Refresh
                # only an exact approved DAG with unchanged protected science;
                # a dialogue metadata version must not strand its dependents.
                self.start(self.main.goal_contract.approved_plan_version)
                if self.snapshot()['status'] == 'needs_user':
                    next_state, resume_error = 'blocked', self.snapshot().get('pause_reason', 'other unresolved nodes remain')
            except Exception as error:
                next_state, resume_error = 'blocked', str(error)
        self.tick()  # finish is the readiness trigger, not a demand for a new chat turn
        return {'ok':True,'step_id':step_id,'phase':self.snapshot()['nodes'][step_id].get('phase'),
                'next_step_state': next_state, 'resume_error': resume_error,
                'next_action':'dependency readiness queue triggered' if next_state == 'triggered' else 'preserve pause or reconcile before dispatch'}

    def _emit(self, kind, payload, suffix):
        state = self.snapshot()
        transitions = {'succeeded': 'worker_completed','prefinish':'worker_prefinish', 'failed': 'worker_failed',
                       'uncertain': 'worker_uncertain', 'validation_failed': 'worker_validation_failed'}
        node = state.get('nodes', {}).get(payload.get('step_id'), {})
        token = suffix.split(':', 1)[0]
        if kind in transitions.values() and node.get('token') == token and node.get('status') in transitions:
            kind = transitions[node['status']]
            suffix = token + ':' + node['status']
            payload = {'error': node.get('error'), 'result_ref': node.get('result_ref'),
                       'evidence_call': node.get('evidence_call'),
                       'origin_goal_version': node.get('origin_goal_version'), **payload}
        event_id = f'runtime:{self.workflow_id}:{suffix}'
        self.store.enqueue_event(self.workflow_id, event_id, kind, {**payload, 'workflow_id': self.main._current_line_id,
            'runtime_id': self.workflow_id, 'origin_goal_version': payload.get('origin_goal_version', state.get('goal_contract', {}).get('version')),
            'plan_version': state.get('plan_version')})
        self.flush_events()
        self.sync_chain('event:' + kind)
        return event_id

    def flush_events(self):
        for event in self.snapshot().get('outbox', {}).values():
            if not event['sent']:
                self.lifecycle.emit(event['kind'], event['payload'], event['event_id'])
                self.store.event_sent(self.workflow_id, event['event_id'])

    def recover_notifications(self):
        """Rebuild the tiny state→outbox crash gap from durable node/message facts."""
        state = self.snapshot()
        outbox = state.get('outbox', {})
        # Reconcile projection-only ghost ownership.  The executor is the
        # authority for dispatch state: a failed preflight with no tool return,
        # no scheduler receipt and no global lease never entered the scientific
        # tool, even if an older TaskLine writer crashed while it still said
        # ``running``.
        all_state = self.store.snapshot()
        leased = {(lease.get('workflow_id'), lease.get('step_id'))
                  for lease in all_state.get('leases', {}).values()}
        line_by_id = {item.get('step_id'): item for item in self._line().get('steps', [])}
        projection_reconciled = False
        for step_id, node in state.get('nodes', {}).items():
            projected = line_by_id.get(step_id, {})
            safe_preflight_failure = (
                node.get('status') == 'failed'
                and node.get('dispatch_phase') == 'preflight'
                and not node.get('tool_returned')
                and not node.get('job_ids')
                and (self.workflow_id, step_id) not in leased
            )
            if safe_preflight_failure and projected.get('status') in {'running', 'submitted', 'unconfirmed'}:
                validation = copy.deepcopy(projected.get('validation', {}))
                validation.update({
                    'runtime_token': None, 'resource_leases': [], 'lease_released': True,
                    'dispatch_not_entered': True, 'executor_status': 'failed',
                    'error': node.get('error', 'pre-dispatch framework failure'),
                })
                self.task_lines.upsert_step(
                    self.main._current_line_id, step_id, username=self.username,
                    conv_id=self.conv_id, status='failed', done=False,
                    validation=validation,
                )
                projection_reconciled = True
        if projection_reconciled:
            self.sync_chain('preflight_projection_reconciled')
        kinds = {'succeeded': 'worker_completed','prefinish':'worker_prefinish', 'failed': 'worker_failed',
                 'uncertain': 'worker_uncertain', 'validation_failed': 'worker_validation_failed'}
        for step_id, node in state.get('nodes', {}).items():
            if node['status'] in kinds and node.get('token'):
                suffix = node['token'] + ':' + node['status']
                event_id = f'runtime:{self.workflow_id}:{suffix}'
                if event_id not in outbox:
                    self._emit(kinds[node['status']], {'step_id': step_id,
                        'requires_user': node['status'] in {'uncertain', 'validation_failed'}}, suffix)
            node_key = hashlib.sha256(step_id.encode()).hexdigest()[:24]
            for message in node.get('messages', {}).values():
                suffix = 'message:' + node_key + ':' + message['message_id']
                if f'runtime:{self.workflow_id}:{suffix}' not in outbox:
                    self._emit('worker_user_message', {'message': message,
                        'requires_user': message['kind'] == 'change'}, suffix)

    def send(self, step_id, text, kind='comment', message_id=None):
        message = self.store.send(self.workflow_id, step_id, text, kind, message_id)
        node_key = hashlib.sha256(step_id.encode()).hexdigest()[:24]
        self._emit('worker_user_message', {'message': message, 'requires_user': kind == 'change'}, 'message:' + node_key + ':' + message['message_id'])
        return message

    def resolve_local_write(self, step_id, decision_id, verification_call_id):
        """Resolve a stopped, non-scheduler writer as FAILED, never as success."""
        node = self.snapshot()['nodes'][step_id]
        if node['status'] != 'uncertain' or node['contract']['tool'] != 'write_file':
            raise ValueError('local resolution only applies to an uncertain write_file node')
        try:
            os.kill(node['owner_pid'], 0)
            stopped = bool(node.get('tool_returned'))
        except ProcessLookupError:
            stopped = True
        if not stopped: raise ValueError('original writer still may be executing; lease cannot be released')
        decision = self.main.memory.user_preferences.get('decision:' + decision_id, {})
        answer = str(decision.get('answer', ''))
        if (decision.get('source') != 'authenticated_user' or step_id not in decision.get('related_nodes', [])
                or not any(w in answer.lower() for w in ('确认', '同意', 'approve', 'confirmed'))
                or any(w in answer.lower() for w in ('不确认', '不同意', '不要', 'not approve'))):
            raise ValueError('main chat needs a genuine user decision for this node')
        call = self.main._load_evidence_call(next((c for c in self.main.memory.tool_call_log if c.get('call_id') == verification_call_id), None))
        if not call or call.get('tool') != 'read_file' or call.get('time', 0) <= node['started_at']:
            raise ValueError('fresh actual file inspection is required')
        def canonical(path):
            p = Path(path)
            return (p if p.is_absolute() else self.main.config.project_root / p).resolve()
        if canonical(call.get('params', {}).get('path', '')) != canonical(node['contract']['arguments']['path']):
            raise ValueError('inspection must target the original write path')
        self.store.update(self.workflow_id, step_id, node['token'], {'status': 'failed', 'resolution': {
            'decision_id': decision_id, 'verification_call_id': verification_call_id,
            'source': 'authenticated_user', 'next_action': 'main chat proposes and negotiates repair'}}, release=True)
        self._step(step_id, status='failed')
        self._emit('worker_resolved', {'step_id': step_id, 'main_already_handling': True}, node['token'] + ':resolved')
        return {'status': 'failed_confirmed', 'lease_released': True, 'resubmitted': False}

    def _worker(self, step_id, ticket):
        try:
            self._worker_body(step_id, ticket)
        except Exception as error:
            preflight = self.snapshot()['nodes'][step_id].get('dispatch_phase') == 'preflight'
            self.store.update(self.workflow_id, step_id, ticket['token'], {'status': 'failed' if preflight else 'uncertain', 'error': str(error)}, release=preflight)
            line_step = next((item for item in self._line().get('steps', [])
                              if item.get('step_id') == step_id), {})
            validation = copy.deepcopy(line_step.get('validation', {}))
            validation.update({'error': str(error),
                               'executor_status': 'failed' if preflight else 'uncertain'})
            if preflight:
                validation.update({'runtime_token': None, 'resource_leases': [],
                                   'lease_released': True, 'dispatch_not_entered': True})
            self._step(step_id, status='failed' if preflight else 'uncertain',
                       done=False, validation=validation)
            self.store.pause(self.workflow_id, 'worker initialization/checkpoint failed; inspect dispatch state', kind='failure' if preflight else 'reconciliation', step_id=step_id)
            self._emit('worker_failed', {'step_id': step_id, 'error': str(error),
                'requires_user': not preflight}, ticket['token'] + ':initialization')

    def _worker_body(self, step_id, ticket):
        from .session import Session
        from .defns import resolve_agent
        from .watch_context import set_context, clear_context
        node, token = ticket['contract'], ticket['token']
        worker = Session(config=self.main.config, registry=self.main.registry)
        path = self.root / 'workers' / hashlib.sha256(step_id.encode()).hexdigest()[:24] / 'session_checkpoint.json'
        owner = {'username': self.username, 'conv_id': self.conv_id, 'step_id': step_id}
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('owner') and saved['owner'] != owner:
                raise PermissionError('worker checkpoint belongs to a different user/session/node')
            worker.import_state(saved)
        worker.current_agent = resolve_agent(node['agent'])
        state = self.snapshot()
        worker.goal_contract = GoalContract.from_dict(state['goal_contract'])
        worker._current_line_id = self.main._current_line_id
        worker.recovery_gate.path = self.root / 'recovery_state.json'
        worker._evidence_root = self.root / 'evidence'
        worker.context['assigned_node'] = copy.deepcopy(node)
        worker.context['dependency_results'] = {dep: {**(state['nodes'][dep].get('result_ref') or {}),
            'path_manifest': state['nodes'][dep].get('path_manifest'), 'job_ids': state['nodes'][dep].get('job_ids', []),
            'status': state['nodes'][dep]['status'], 'tool': state['nodes'][dep]['contract']['tool'],
            'agent': state['nodes'][dep]['contract']['agent']} for dep in node.get('depends_on', [])}
        def checkpoint(saved, reason):
            saved['owner'] = owner
            references = compact_checkpoint_evidence(saved, self.root / 'evidence')
            write_checkpoint(path, saved)
            apply_evidence_references(worker.memory, references)
        worker._on_checkpoint = checkpoint
        def inbox():
            messages = self.snapshot()['nodes'][step_id].get('messages', {})
            seen = worker.context.setdefault('received_messages', {})
            for mid, message in messages.items():
                if mid not in seen:
                    seen[mid] = copy.deepcopy(message)
                    worker.messages.append({'role': 'user', 'content': '[节点定向消息] ' + json.dumps(message, ensure_ascii=False)})
            worker._checkpoint('worker_inbox')
            self.store.acknowledge(self.workflow_id, step_id, list(seen))
        recovery_key = attempt = None
        try:
            inbox()
            if self.snapshot()['status'] != 'active' or self.snapshot().get('user_paused'):
                self.store.update(self.workflow_id, step_id, token, {'status': 'pending'}, release=True)
                return
            if step_id in (self.main._pending_workflow_patch or {}).get('affected_nodes', []):
                self.store.update(self.workflow_id, step_id, token, {'status': 'pending'}, release=True)
                return
            line = self._line()
            if self.main.goal_contract.version != worker.goal_contract.version:
                if self.store.reconcile_goal(self.workflow_id, self.main.goal_contract.to_dict()):
                    worker.goal_contract = GoalContract.from_dict(self.main.goal_contract.to_dict())
            if int(line.get('plan_version', 0)) != state['plan_version'] or self.main.goal_contract.version != worker.goal_contract.version:
                raise ValueError('approved goal/plan changed before dispatch; request main chat review')
            manifest = node_path_manifest(node, self.main.config.project_root, self.root, result_ref={'checkpoint_path': str(path)})
            self.store.update(self.workflow_id, step_id, token, {'path_manifest': manifest})
            self._step(step_id, status='running', path_manifest=manifest, output_dir=manifest['result_dir'], validation={
                'runtime_token': token, 'execution_session': str(path), 'resource_leases': node['resources']})
            args = copy.deepcopy(node['arguments'])
            allocation = {}
            if is_submission(node['tool'], args) and self.main._on_resource_review:
                # Review a successor only when its dependencies are real and
                # it is about to dispatch, not while preparing another branch.
                receipt=self.main._on_resource_review({'kind':'before_submission','nodes':[node],'goal_version':worker.goal_contract.version})
                if receipt.get('status')!='ready':
                    retries = self.snapshot()['nodes'][step_id].get('resource_review_attempts', 0) + 1
                    retryable = receipt.get('status') == 'waiting_resources' or receipt.get('status') == 'review_retry' and retries < 3
                    self.store.update(self.workflow_id,step_id,token,{'status':'needs_resources','resource_review':receipt,
                        'resource_review_attempts': retries, 'resource_retry_at': time.time() + min(300, 15 * retries) if retryable else None},release=True)
                    self._step(step_id,status='needs_resources',validation={'resource_review':receipt})
                    self.store.pause(self.workflow_id,'ready successor requires verified resources',kind='resource_review', step_id=step_id)
                    self._emit('worker_resource_review_required',{'step_id':step_id,'resource_review':receipt,
                        'requires_user': False, 'automatic_retry': retryable,
                        'main_already_handling': retryable},token+':resources')
                    return
                properties = self.main.registry.get(node['tool']).input_schema.get('properties', {})
                if 'resource_review_id' in properties:
                    args['resource_review_id']=receipt['resource_review_id']
                allocation = receipt.get('resource_allocations', {}).get(step_id, {})
                for key in ('memory_mb', 'nodelist', 'partition'):
                    if allocation.get(key) is not None and key in properties:
                        args.setdefault(key, allocation[key])
                if allocation.get('cpus') is not None:
                    if 'cpus_per_task' in properties:
                        args.setdefault('cpus_per_task', allocation['cpus'])
                    elif 'num_processes' in properties:
                        args.setdefault('num_processes', allocation['cpus'])
                self.store.update(self.workflow_id, step_id, token, {'resource_allocation': allocation,
                    'effective_arguments': copy.deepcopy(args)})
                self._step_metadata(step_id, token=token, resource_allocation=allocation,
                                    effective_arguments=copy.deepcopy(args))
            before = self._artifacts(node, {})
            if is_submission(node['tool'], args):
                recovery_key = worker.recovery_gate.key(node['tool'], args, worker.goal_contract.version, step_id, worker.config.project_root)
                self.store.update(self.workflow_id, step_id, token, {'recovery_key': recovery_key})
                attempt, reason = worker.recovery_gate.claim(recovery_key, node['tool'], args,
                    input_fingerprint(args, worker.config.project_root), worker.goal_contract.version, worker.config.project_root)
                if not attempt: raise ValueError(reason)
            self.store.update(self.workflow_id, step_id, token, {'checkpoint_path': str(path), 'recovery_key': recovery_key,
                'attempt_id': attempt, 'output_baseline': before})
            worker._checkpoint('before_assigned_tool')
            set_context(self.username, self.conv_id, worker.current_agent.name, worker._current_line_id,
                tool_name=node['tool'], recovery_key=recovery_key or '', attempt_id=attempt or '',
                recovery_path=str(worker.recovery_gate.path), step_id=step_id, resource_allocation=allocation)
            self.store.update(self.workflow_id, step_id, token, {'dispatch_phase': 'entered_tool'})
            raw = self.main.registry.execute_dict(node['tool'], args)
            self.store.update(self.workflow_id, step_id, token, {'tool_returned': True})
            if recovery_key: worker.recovery_gate.outcome(recovery_key, attempt, raw)
            worker.memory.record_tool_call(worker.current_agent.name, node['tool'], args,
                json.dumps(raw, ensure_ascii=False, default=str), failed=bool(failure_reason(raw)))
            worker._checkpoint('after_assigned_tool')
            call = worker.memory.tool_call_log[-1]
            self.store.update(self.workflow_id, step_id, token, {'result': result_object(raw), 'result_ref': {
                'call_id': call['call_id'], 'checkpoint_path': str(path), 'evidence_path': call.get('evidence_path')},
                'evidence_call': call})
            inbox()
            self._settle(step_id, token, raw)
        except Exception as error:
            # Any exception after claim/dispatch is conservatively uncertain.
            status = 'uncertain' if recovery_key else 'failed'
            self.store.update(self.workflow_id, step_id, token, {'status': status, 'error': str(error)}, release=status == 'failed')
            # Keep the durable TaskLine aligned with the executor.  In
            # particular, a pre-dispatch framework exception must not leave an
            # obsolete runtime token/resource lease that later makes a failed
            # node look live and forces the user to choose a cleanup path.
            line_step = next((item for item in self._line().get('steps', [])
                              if item.get('step_id') == step_id), {})
            validation = copy.deepcopy(line_step.get('validation', {}))
            validation.update({'error': str(error), 'executor_status': status})
            if status == 'failed':
                validation.update({'runtime_token': None, 'resource_leases': [],
                                   'lease_released': True})
            self._step(step_id, status=status, done=False, validation=validation)
            self.store.pause(self.workflow_id, 'node failed or evidence/dispatch state is uncertain', kind='reconciliation' if status == 'uncertain' else 'failure', step_id=step_id)
            self._emit('worker_failed', {'step_id': step_id, 'error': str(error), 'requires_user': status == 'uncertain'}, token + ':exception')
        finally:
            clear_context()

    def _artifacts(self, node, result):
        args = node['arguments']
        base = artifact_base(node, self.main.config.project_root, self.root, result)
        facts = {}
        for name in node.get('expected_outputs', []):
            facts[str(output_path(name, base))] = artifact_fact(name, base)
        return facts

    def _bind_native_result_path(self, step_id, token, result):
        """Native scheduler receipt fixes managed CSV location, not science.

        The native batch creates a timestamped child of the approved work root.
        Explicit custom outputs, unrelated files and paths outside the leased
        parent cannot be silently substituted.
        """
        node = self.snapshot()['nodes'][step_id]
        contract, obj = node['contract'], result_object(result)
        args = contract['arguments']
        from .output_contract import normalize_output
        receipt_key = next((key for key in ('output_csv', 'output_dir')
                            if args.get(key) and obj.get(key)), None)
        generic_output = bool(receipt_key and obj.get('job_id'))
        cdft_csv = (contract['tool'] == 'run_cdft' and args.get('action') in {'submit', 'pipeline'}
                    and not args.get('output') and args.get('job_work_dir') and obj.get('work_dir') and obj.get('job_id'))
        if not (generic_output or cdft_csv):
            return False
        if generic_output:
            approved = Path(args[receipt_key]).resolve()
            actual_file = Path(obj[receipt_key]).resolve()
            if actual_file != approved or not actual_file.is_relative_to(self.root.resolve()):
                raise ValueError('native output receipt differs from the approved session path')
            parent = artifact_base(contract, self.main.config.project_root, self.root, obj)
            after = str(actual_file)
            before = ''
        else:
            parent = Path(args['job_work_dir']).resolve()
            actual = Path(obj['work_dir']).resolve()
            # cDFT may use an explicitly approved project calculation root
            # outside the conversation result folder. Its existing safety
            # boundary is the approved job_work_dir; generic CSV receipts are
            # stricter and remain confined to this conversation above.
            if not actual.is_relative_to(parent):
                raise ValueError('native result path escapes the approved leased calculation root')
            before = str(parent / 'results.csv')
            after = str(actual / 'results.csv')
        revised = copy.deepcopy(contract)
        changed = False
        for index, output in enumerate(revised.get('expected_outputs', [])):
            spec = normalize_output(output)
            resolved_before = str(output_path(spec, parent))
            project_path = Path(spec['path'])
            project_path = (project_path if project_path.is_absolute()
                            else Path(self.main.config.project_root) / project_path).resolve()
            expected_kind = 'directory' if generic_output and receipt_key == 'output_dir' else 'file'
            matches = resolved_before == before if cdft_csv else project_path == Path(after)
            if spec['kind'] == expected_kind and matches and resolved_before != after:
                if not before: before = resolved_before
                revised['expected_outputs'][index] = {**spec, 'path': after}
                changed = True
        if not changed: return False
        execution = node.get('evidence_call', {})
        if (str(result_object(execution.get('result') or execution.get('result_preview')).get('job_id', obj['job_id'])) != str(obj['job_id'])
                or str(obj['job_id']) not in node.get('job_ids', []) and node.get('job_ids')):
            raise ValueError('native output binding must match the actual scheduler receipt')
        proof = {'kind': 'native_receipt_path', 'job_id': str(obj['job_id']),
            'execution_call_id': node.get('result_ref', {}).get('call_id'), 'before': before, 'after': after,
            'science_arguments_unchanged': True, 'no_resubmission': True}
        self.task_lines.repair_output_contract(self.main._current_line_id, step_id, revised['expected_outputs'], proof, self.snapshot()['plan_version'])
        history = copy.deepcopy(node.get('output_contract_history', []))
        history.append(proof)
        if not self.store.update(self.workflow_id, step_id, token, {'contract': revised, 'output_contract_history': history}):
            raise ValueError('node ownership changed during native output binding')
        for approved in self.main.goal_contract.approved_nodes:
            if approved['step_id'] == step_id: approved['expected_outputs'] = copy.deepcopy(revised['expected_outputs'])
        self.sync_chain('native_output_path_bound:' + step_id)
        return True

    def _settle(self, step_id, token, result):
        self._bind_native_result_path(step_id, token, result)
        current = self.snapshot()['nodes'][step_id]
        obj = result_object(result)
        manifest = node_path_manifest(current['contract'], self.main.config.project_root, self.root, obj, current.get('result_ref'))
        from .workspace import tool_scope_issues
        for value in [manifest['calculation_dir'], manifest['result_dir'], *manifest['input_paths'], *manifest['output_files'], *manifest['expected_outputs']]:
            if tool_scope_issues({'path': value}, self.username, self.conv_id, self.main.config.project_root):
                raise ValueError('tool result path belongs to another private session')
        self.store.update(self.workflow_id, step_id, token, {'path_manifest': manifest})
        self._step(step_id, path_manifest=manifest, output_dir=manifest['result_dir'], output_files=manifest['output_files'])
        reason = failure_reason(result)
        entry = RecoveryGate(path=self.root / 'recovery_state.json').snapshot().get(current.get('recovery_key'), {})
        if reason or entry.get('status') == 'failed':
            reason = reason or entry.get('error')
            unknown_jobs = bool(entry.get('job_ids')) and not current.get('jobs_confirmed_terminal')
            self.store.update(self.workflow_id, step_id, token, {'status': 'uncertain' if unknown_jobs else 'failed', 'error': reason}, release=not unknown_jobs)
            line_step = next((item for item in self._line().get('steps', [])
                              if item.get('step_id') == step_id), {})
            validation = copy.deepcopy(line_step.get('validation', {}))
            validation.update({'error': reason,
                               'executor_status': 'uncertain' if unknown_jobs else 'failed'})
            if not unknown_jobs:
                validation.update({'runtime_token': None, 'resource_leases': [],
                                   'lease_released': True})
            self._step(step_id, status='uncertain' if unknown_jobs else 'failed',
                       done=False, validation=validation)
            self.store.pause(self.workflow_id, 'failed node requires diagnosis and plan review', step_id=step_id)
            self._emit('worker_failed', {'step_id': step_id, 'error': reason, 'evidence_call': current.get('evidence_call')}, token + ':failed')
            return
        if entry.get('status') in {'submitted', 'reserved', 'uncertain'}:
            if not entry.get('job_ids'):
                self.store.update(self.workflow_id, step_id, token, {'status': 'uncertain', 'error': 'submission helper returned without a scheduler job receipt'})
                self.store.pause(self.workflow_id, 'main chat must reconcile missing scheduler receipt', kind='reconciliation', step_id=step_id)
                self._emit('worker_uncertain', {'step_id': step_id, 'requires_user': True,
                    'evidence_call': current.get('evidence_call')}, token + ':missing-receipt')
                return
            if self.job_watch:
                for jid in entry['job_ids']:
                    previous = self.job_watch.get(jid) or {}
                    if previous.get('username') and (previous.get('username') != self.username or previous.get('conv_id') != self.conv_id):
                        raise ValueError('scheduler receipt conflicts with another conversation owner')
                    if not previous.get('username') or not previous.get('conv_id'):
                        self.job_watch.register(jid, work_dir=obj.get('work_dir') or current['contract']['arguments'].get('job_work_dir', ''),
                            username=self.username, conv_id=self.conv_id, agent_name=current['contract']['agent'], tool=current['contract']['tool'])
            status = 'waiting_prerequisite' if obj.get('chain_status') == 'waiting' else 'waiting_jobs'
            self.store.update(self.workflow_id, step_id, token, {'status': status, 'job_ids': entry.get('job_ids', [])})
            self._step(step_id, status='submitted', job_ids=entry.get('job_ids', []))
            return
        if entry.get('status') == 'completed_unverified' and obj.get('chain_status') == 'waiting':
            self.store.update(self.workflow_id, step_id, token, {'status': 'pending'}, release=True)
            return
        artifacts = self._artifacts(current['contract'], obj)
        baseline = current.get('output_baseline', {})
        read_source = current['contract']['arguments'].get('path') if current['contract']['tool'] == 'read_file' else None
        if read_source:
            p = Path(read_source)
            read_source = str((p if p.is_absolute() else self.main.config.project_root / p).resolve())
        idempotent_verified_reuse = (
            current['contract']['tool'] == 'stage_cif_subset'
            and obj.get('ok') is True and obj.get('status') == 'reused'
            and obj.get('count') == current['contract']['arguments'].get('limit')
            and len(obj.get('files') or []) == current['contract']['arguments'].get('limit')
        )
        valid = all(fact and fact['size'] > 0 and (
                    idempotent_verified_reuse or path == read_source or
                    fact != baseline.get(path) and fact['mtime_ns'] >= int(current['started_at'] * 1e9)
                    ) for path, fact in artifacts.items())
        if is_submission(current['contract']['tool'],current['contract']['arguments']) and not current.get('node_verification'):
            self.store.update(self.workflow_id,step_id,token,{'status':'prefinish','phase':'prefinish','artifacts':artifacts,
                'machine_output_check':valid,'path_manifest':manifest,'verification_owner':current['contract']['agent']})
            self._step(step_id,status='prefinish',done=False,path_manifest=manifest)
            self._emit('worker_prefinish',{'step_id':step_id,'contract':current['contract'],'result_ref':current.get('result_ref'),
                'job_ids':current.get('job_ids',[]),'machine_output_check':valid,'verification_owner':current['contract']['agent'],
                'instruction':'Execution ended, not finished. Validate native outputs and scientific conditions using real tools; repair output contracts if needed, then finish_workflow_node. Never resubmit a completed job.'},token+':prefinish')
            return
        if not valid:
            # The function/jobs have stopped writing, so repair may acquire the
            # resource. Failed output validation still blocks all descendants.
            self.store.update(self.workflow_id, step_id, token, {'status': 'validation_failed', 'error': 'declared outputs missing, empty or unchanged', 'artifacts': artifacts}, release=True)
            self._step(step_id, status='failed',
                validation={'output_contract': 'failed', 'artifacts': artifacts})
            self.store.pause(self.workflow_id, 'output verification requires main chat repair', step_id=step_id)
            self._emit('worker_validation_failed', {'step_id': step_id, 'requires_user': True, 'artifacts': artifacts,
                'evidence_call': current.get('evidence_call')}, token + ':validation')
            return
        if entry.get('status') == 'completed_unverified':
            RecoveryGate(path=self.root / 'recovery_state.json').accept_result(current['recovery_key'], {'artifacts': artifacts, 'execution_call_id': current.get('result_ref', {}).get('call_id')})
        files = []
        for path, fact in artifacts.items():
            files.extend([item[0] for item in fact.get('files', [])] if fact and fact.get('files') else [path])
        manifest['output_files'] = sorted(set(manifest['output_files'] + files))
        self._step(step_id, status='completed', done=True, output_files=manifest['output_files'], path_manifest=manifest,
            validation={'output_contract': 'passed', 'runtime_token': token, 'execution_session': current.get('checkpoint_path')})
        self.store.update(self.workflow_id, step_id, token, {'status': 'succeeded', 'phase':'finish','artifacts': artifacts, 'path_manifest': manifest,
            'execution_fingerprint': execution_fingerprint(current['contract'], self.main)}, release=True)
        self._emit('worker_completed', {'step_id': step_id, 'result_ref': current.get('result_ref'),
            'evidence_call': current.get('evidence_call'), 'origin_goal_version': current['origin_goal_version'],
            'runtime_completed': self.snapshot()['status'] == 'completed'}, token + ':completed')

    def tick(self, jobs=(), paused=False, blocked_nodes=()):
        with self.lock:
            self.futures = {key: future for key, future in self.futures.items() if not future.done()}
            state = self.snapshot()
            if not state: return
            if paused: self.store.user_pause(self.workflow_id)
            self.recover_notifications()
            self.flush_events()
            gate = RecoveryGate(path=self.root / 'recovery_state.json')
            gate.sync_jobs(jobs)
            for step_id, node in state['nodes'].items():
                if (node['status'] == 'needs_resources' and node.get('resource_retry_at')
                        and time.time() >= node['resource_retry_at'] and not paused
                        and not state.get('user_paused')):
                    self.store.update(self.workflow_id, step_id, node['token'],
                                      {'status': 'pending', 'resource_retry_at': None}, release=True)
                    self._step(step_id, status='pending', done=False)
                if node['status'] in {'waiting_jobs', 'waiting_prerequisite', 'uncertain'}:
                    ids = gate.snapshot().get(node.get('recovery_key'), {}).get('job_ids', [])
                    by_id = {str(j['job_id']): j for j in jobs if j.get('job_id')}
                    terminal = ids and all(by_id.get(jid, {}).get('terminal') and str(by_id[jid].get('state', '')).upper() != 'UNKNOWN' for jid in ids)
                    if terminal:
                        self.store.update(self.workflow_id, step_id, node['token'], {'jobs_confirmed_terminal': True})
                    if node['status'] != 'uncertain' or terminal:
                        self._settle(step_id, node['token'], node.get('result', {}))
                elif node['status'] == 'running':
                    try: os.kill(node['owner_pid'], 0)
                    except ProcessLookupError:
                        safe = node['contract']['tool'] in READ_ONLY
                        preflight = node.get('dispatch_phase') == 'preflight'
                        self.store.update(self.workflow_id, step_id, node['token'], {
                            'status': 'pending' if safe else 'failed' if preflight else 'uncertain',
                            'error': 'worker disappeared before dispatch' if preflight else 'worker disappeared; unknown mutations are never redispatched'}, release=safe or preflight)
                        if preflight and node.get('recovery_key'):
                            entry = gate.snapshot().get(node['recovery_key'], {})
                            if entry.get('status') == 'reserved' and not entry.get('job_ids'):
                                gate.outcome(node['recovery_key'], entry['attempt_id'], {'error':'process died before persisted entered_tool boundary'})
                        if not safe: self.store.pause(self.workflow_id, 'dead worker requires diagnosis/plan review' if preflight else 'dead worker requires dispatch reconciliation',
                            kind='failure' if preflight else 'reconciliation', step_id=step_id)
                        self._emit('worker_orphaned', {'step_id': step_id, 'requires_user': not safe and not preflight}, node['token'] + ':orphan')
                    else:
                        if step_id in self.futures:
                            self.store.update(self.workflow_id, step_id, node['token'], {'heartbeat': time.time()})
                        if time.time() - node['started_at'] > 360:
                            self._emit('worker_stalled', {'step_id': step_id, 'requires_user': True}, node['token'] + ':stalled')
            state = self.snapshot()
            if paused or state.get('user_paused') or state['status'] != 'active': return
            if state['plan_version'] != self.main.goal_contract.approved_plan_version or state['goal_contract']['version'] != self.main.goal_contract.version:
                if not self.store.reconcile_goal(self.workflow_id, self.main.goal_contract.to_dict()):
                    self.store.pause(self.workflow_id, 'scientific contract changed; approved plan reconciliation required')
                    return
            blocked_nodes = set(blocked_nodes) | set((self.main._pending_workflow_patch or {}).get('affected_nodes', []))
            for step_id in state['nodes']:
                if step_id in blocked_nodes: continue
                if len(self.futures) >= self.max_workers: break
                ticket = self.store.claim(self.workflow_id, step_id)
                if ticket: self.futures[step_id] = self.pool.submit(self._worker, step_id, ticket)

    def shutdown(self):
        self.pool.shutdown(wait=False, cancel_futures=False)
