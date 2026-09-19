"""Conversation-scoped durable chain, including retired nodes and revisions.

The executable DAG is not the complete scientific task history. This record
joins approved contracts, actual execution, job paths and error branches.
"""
import copy
import hashlib
import json
from pathlib import Path
import time

from .state_io import json_transaction
from .workflow_view import authoritative_line


def durable(value):
    if isinstance(value, dict):
        return {k: durable(v) for k, v in value.items() if k not in
                {'updated_at', 'heartbeat', 'service_heartbeat', 'checkpoint_at', 'transcript_message_count'}}
    if isinstance(value, (list, tuple)):
        return [durable(v) for v in value]
    return value


def sync_chain(root, line, runtime, reason, *, state=None, jobs=None, turn_id=None):
    root = Path(root).resolve()
    with json_transaction(root / 'orchestration_chain.json') as data:
        # Gather authoritative snapshots under the chain lock. A slower worker
        # cannot publish a stale pre-lock snapshot over a newer transition.
        actual_line = line() if callable(line) else line
        actual_runtime = runtime() if callable(runtime) else runtime
        return _sync_locked(data, root, actual_line, actual_runtime, reason, state=state, jobs=jobs, turn_id=turn_id)


def _sync_locked(data, root, line, runtime, reason, *, state=None, jobs=None, turn_id=None):
    path = root / 'orchestration_chain.json'
    root = Path(root).resolve()
    owner = {k: (runtime or line or {}).get(k) for k in ('username', 'conv_id')}
    view = authoritative_line(line, runtime)
    active = {n['step_id']: n for n in view.get('steps', []) if 'expected_outputs' in n}
    retired = {}
    for version in (line or {}).get('plan_history', []):
        for node in version.get('steps', []):
            if 'expected_outputs' in node and node['step_id'] not in active:
                retired.setdefault(node['step_id'], []).append({'plan_version': version.get('version'), 'node': node})
    snapshot = {'approved_plan_version': view.get('plan_version'), 'execution_plan_version': runtime.get('plan_version'),
        'execution_status': runtime.get('status'), 'user_paused': runtime.get('user_paused', False),
        'nodes': active, 'retired_nodes': retired,
        'execution_history': runtime.get('plan_history', []), 'contract_history': (line or {}).get('plan_history', [])}
    for key, node in snapshot['nodes'].items():
        actual = runtime.get('nodes', {}).get(key, {})
        if node.get('contract_matches_execution') is False:
            actual = {}
        node['attempt_history'] = copy.deepcopy(actual.get('attempt_history', node.get('attempt_history', [])))
        node['verification'] = copy.deepcopy(actual.get('node_verification'))
        node['artifacts'] = copy.deepcopy(actual.get('artifacts', {}))
        node['output_contract_history'] = copy.deepcopy(actual.get('output_contract_history', node.get('output_contract_history', [])))
    if jobs is not None:
        if any((j.get('username'), j.get('conv_id')) != (owner['username'], owner['conv_id']) for j in jobs):
            raise PermissionError('chain cannot contain foreign job records')
        snapshot['jobs'] = {str(j['job_id']): j for j in jobs}
    if state is not None:
        snapshot.update(goal=state.get('goal_contract'), branches=state.get('error_branches', {}),
            pending_patch=state.get('pending_workflow_patch'), pending_user_interaction=state.get('pending_user_interaction'),
            withdrawn_patches=state.get('context', {}).get('withdrawn_workflow_patches', []))
    if data.get('owner') and data['owner'] != owner:
        raise PermissionError('chain belongs to another user/session')
    previous = data.get('current', {})
    if previous.get('nodes') and (not active or
            int(snapshot.get('approved_plan_version') or 0) < int(previous.get('approved_plan_version') or 0)):
        for key in ('nodes', 'approved_plan_version', 'retired_nodes', 'contract_history'):
            snapshot[key] = copy.deepcopy(previous.get(key))
        snapshot['graph_sync_pending'] = True
    else:
        snapshot['graph_sync_pending'] = False
    # Empty/older snapshots during restoration must not destroy the last
    # committed graph. Retain it for display without granting execution rights.
    if active and int(snapshot.get('approved_plan_version') or 0) >= int(data.get('last_complete_graph', {}).get('version') or 0):
        data['last_complete_graph'] = {
            'version': snapshot.get('approved_plan_version'),
            'source': 'persisted_chain', 'nodes': copy.deepcopy(list(active.values())),
        }
    elif not data.get('last_complete_graph') and previous.get('nodes'):
        data['last_complete_graph'] = {
            'version': previous.get('approved_plan_version'), 'source': 'persisted_chain',
            'nodes': copy.deepcopy(list(previous['nodes'].values())),
        }
    if 'jobs' in snapshot:
        merged = copy.deepcopy(previous.get('jobs', {}))
        for job_id, job in snapshot['jobs'].items():
            prior = merged.get(job_id, {})
            observed = str(job.get('updated_at', ''))
            if observed >= str(prior.get('record_updated_at', '')):
                merged[job_id] = {**job, 'record_updated_at': observed}
        snapshot['jobs'] = merged
    # Background transitions cannot erase main-chat branches or user intent.
    current = {**previous, **durable(snapshot)}
    changes = {k: {'before': previous.get(k), 'after': v} for k, v in current.items() if previous.get(k) != v}
    data.setdefault('schema_version', 1)
    data['owner'] = owner
    if changes:
        revisions = data.setdefault('revisions', [])
        parent_hash = revisions[-1]['hash'] if revisions else None
        revision = {'sequence': len(revisions) + 1, 'time': time.time(), 'reason': reason,
                    'parent_hash': parent_hash, 'changes': changes}
        revision['hash'] = hashlib.sha256(json.dumps(revision, sort_keys=True, default=str).encode()).hexdigest()
        revisions.append(revision)
    data['current'] = current
    if turn_id:
        turn = data.setdefault('turns', {}).setdefault(str(turn_id), {'started_at': time.time()})
        turn.update(last_boundary=reason, chain_revision=len(data.get('revisions', [])), updated_at=time.time())
    data['updated_at'] = time.time()
    return {'path': str(path), 'revision': len(data.get('revisions', [])), 'updated': bool(changes)}


def read_last_graph(root, owner):
    """Read only the exact session's durable graph; never restore authority."""
    try:
        data = json.loads((Path(root) / 'orchestration_chain.json').read_text())
    except (OSError, ValueError):
        return {}
    if data.get('owner') != owner:
        return {}
    current = data.get('current') or {}
    if current.get('nodes'):
        return {'version': current.get('approved_plan_version'), 'source': 'persisted_chain',
                'revision': len(data.get('revisions', [])),
                'retained': bool(current.get('graph_sync_pending')),
                'nodes': list(current['nodes'].values())}
    graph = data.get('last_complete_graph') or {}
    if graph.get('nodes'):
        return {**graph, 'retained': True}
    return {}
