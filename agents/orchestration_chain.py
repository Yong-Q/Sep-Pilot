"""Conversation-scoped durable chain, including retired nodes and revisions.

The executable DAG is not the complete scientific task history. This record
joins approved contracts, actual execution, job paths and error branches.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat
import time

from .state_io import ConversationLock, json_transaction, write_checkpoint
from .chain_evidence import publish, read_reference as read_chain_reference
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
    # Covers both manifest commit and sidecar publication, across processes.
    # The sidecar is only a display projection, never execution authority.
    lease = ConversationLock(root / 'orchestration_projection.lock')
    lease.acquire()
    try:
        path = root / 'orchestration_chain.json'
        with json_transaction(path) as data:
            actual_line = line() if callable(line) else line
            actual_runtime = runtime() if callable(runtime) else runtime
            owner = {k: (actual_runtime or actual_line or {}).get(k) for k in ('username', 'conv_id')}
            if data.get('owner') and data['owner'] != owner:
                raise PermissionError('chain belongs to another user/session')
            if data and data.get('schema_version', 1) < 2:
                data['legacy_archive_ref'] = publish(root, owner, 'legacy_chain', raw=path.read_bytes())
                for revision in data.get('revisions', []):
                    if 'changes' in revision:
                        revision['changes_ref'] = publish(root, owner, 'revision_changes', revision.pop('changes'))
                        revision['hash_format'] = 'legacy_inline_changes'
            result = _sync_locked(data, root, actual_line, actual_runtime, reason,
                                  state=state, jobs=jobs, turn_id=turn_id)
            graph = _graph(data)
            projection = {'schema_version': 1, 'owner': owner, 'graph': graph}
        projection['chain_signature'] = _signature(path)
        write_checkpoint(root / 'orchestration_current.json', projection)
        return result
    finally:
        lease.release()


_HISTORY_FIELDS = ('execution_history', 'contract_history', 'retired_nodes', 'withdrawn_patches',
                   'attempt_history', 'output_contract_history', 'previous_execution')
_INLINE_HISTORY_BYTES = 16384


def _compact_history(root, owner, value):
    """Only named historical payloads spill; current contracts remain intact."""
    for key in _HISTORY_FIELDS:
        if key not in value:
            continue
        item = value[key]
        # A retained, already compacted graph must keep its existing reference.
        if not item and value.get(key + '_ref'):
            continue
        value.pop(key + '_ref', None)
        value.pop(key + '_count', None)
        if len(json.dumps(item, ensure_ascii=False, default=str).encode('utf-8')) > _INLINE_HISTORY_BYTES:
            value[key + '_ref'] = publish(root, owner, 'history:' + key, item)
            value[key + '_count'] = len(item) if isinstance(item, (list, dict)) else None
            value[key] = {} if isinstance(item, dict) else []
    return value


def _signature(path):
    info = path.stat()
    return {'inode': info.st_ino, 'size': info.st_size, 'mtime_ns': info.st_mtime_ns}


def _read_projection(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 8 * 1024 * 1024:
            raise ValueError('invalid chain projection file')
        raw = stream.read(info.st_size + 1)
    projection = json.loads(raw)
    graph = projection.get('graph') if isinstance(projection, dict) else None
    if not isinstance(graph, dict) or not isinstance(graph.get('nodes', []), list):
        raise ValueError('invalid chain projection graph')
    if any(not isinstance(node, dict) or not isinstance(node.get('step_id'), str)
           for node in graph.get('nodes', [])):
        raise ValueError('invalid chain projection nodes')
    return projection


def _sync_locked(data, root, line, runtime, reason, *, state=None, jobs=None, turn_id=None):
    path = root / 'orchestration_chain.json'
    root = Path(root).resolve()
    owner = {k: (runtime or line or {}).get(k) for k in ('username', 'conv_id')}
    view = authoritative_line(line, runtime,(state or {}).get('goal_contract'))
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
        for key in ('nodes', 'approved_plan_version', 'retired_nodes', 'contract_history',
                    'retired_nodes_ref', 'retired_nodes_count', 'contract_history_ref', 'contract_history_count'):
            snapshot[key] = copy.deepcopy(previous.get(key))
        snapshot['graph_sync_pending'] = True
    else:
        snapshot['graph_sync_pending'] = False
    _compact_history(root, owner, snapshot)
    for node in snapshot['nodes'].values():
        _compact_history(root, owner, node)
    # Empty/older snapshots during restoration must not destroy the last
    # committed graph. Retain it for display without granting execution rights.
    if active and int(snapshot.get('approved_plan_version') or 0) >= int(data.get('last_complete_graph', {}).get('version') or 0):
        data['last_complete_graph'] = {
            'version': snapshot.get('approved_plan_version'),
            'source': 'persisted_chain', 'nodes': copy.deepcopy(list(snapshot['nodes'].values())),
        }
    elif not data.get('last_complete_graph') and previous.get('nodes'):
        data['last_complete_graph'] = {
            'version': previous.get('approved_plan_version'), 'source': 'persisted_chain',
            'nodes': copy.deepcopy(list(snapshot['nodes'].values())),
        }
    for node in data.get('last_complete_graph', {}).get('nodes', []):
        _compact_history(root, owner, node)
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
    for key in _HISTORY_FIELDS:
        if key in snapshot:
            for suffix in ('_ref', '_count'):
                if key + suffix not in snapshot:
                    current.pop(key + suffix, None)
    _compact_history(root, owner, current)
    changes = {k: {'before': previous.get(k), 'after': current.get(k),
                   'before_present': k in previous, 'after_present': k in current}
               for k in sorted(previous.keys() | current.keys())
               if previous.get(k) != current.get(k) or (k in previous) != (k in current)}
    data['schema_version'] = 2
    data['owner'] = owner
    if changes:
        revisions = data.setdefault('revisions', [])
        parent_hash = revisions[-1]['hash'] if revisions else None
        revision = {'sequence': len(revisions) + 1, 'time': time.time(), 'reason': reason,
                    'parent_hash': parent_hash, 'hash_format': 'changes_reference_v2',
                    'changes_ref': publish(root, owner, 'revision_changes', changes)}
        revision['hash'] = hashlib.sha256(json.dumps(revision, sort_keys=True, default=str).encode()).hexdigest()
        revisions.append(revision)
    data['current'] = current
    if turn_id:
        turn = data.setdefault('turns', {}).setdefault(str(turn_id), {'started_at': time.time()})
        if (turn.get('last_boundary'), turn.get('chain_revision')) != (reason, len(data.get('revisions', []))):
            turn.update(last_boundary=reason, chain_revision=len(data.get('revisions', [])), updated_at=time.time())
    if changes:
        data['updated_at'] = time.time()
    return {'path': str(path), 'revision': len(data.get('revisions', [])), 'updated': bool(changes)}


def read_last_graph(root, owner, *, strict=False):
    """Read only the exact session's durable graph; never restore authority."""
    try:
        root = Path(root)
        path = root / 'orchestration_chain.json'
        signature = _signature(path)
        try:
            projection = _read_projection(root / 'orchestration_current.json')
            if (isinstance(projection, dict) and projection.get('owner') == owner
                    and projection.get('chain_signature') == signature
                    and _signature(path) == signature):
                return projection.get('graph') or {}
        except (OSError, ValueError, TypeError):
            pass
        data = json.loads(path.read_text())
    except FileNotFoundError:
        return {}
    except (OSError, ValueError, TypeError):
        if strict:
            raise
        return {}
    if not isinstance(data, dict):
        if strict:
            raise ValueError('chain manifest must be a JSON object')
        return {}
    if data.get('owner') != owner:
        if strict:
            raise PermissionError('chain belongs to another user/session')
        return {}
    return _graph(data)


def _graph(data):
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


def read_chain_revision(root, owner, sequence):
    """Explicit audit read; UI and execution paths never hydrate revisions."""
    data = json.loads((Path(root) / 'orchestration_chain.json').read_text())
    if data.get('owner') != owner:
        raise PermissionError('chain belongs to another user/session')
    revision = copy.deepcopy(next(r for r in data.get('revisions', []) if r['sequence'] == sequence))
    if 'changes_ref' in revision:
        revision['changes'] = read_chain_reference(root, owner, revision['changes_ref'])
        if revision.get('hash_format') == 'legacy_inline_changes':
            revision.pop('changes_ref')
            revision.pop('hash_format')
    return revision
