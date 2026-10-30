"""Fresh read-only orientation after executor deployment; never retry authority."""
import copy
import json
import os
from pathlib import Path
import time

from .executor_identity import executor_identity

_PROCESS_STARTED_NS = str(time.time_ns())


def executor_epoch(executor):
    body = executor_identity(executor)
    return {'pid': os.getpid(), 'process_epoch': str(os.getpid()) + ':' + _PROCESS_STARTED_NS,
            'loaded_executor': body}


def _binding(session, event):
    """Run the event's native inspection once, only for the current owned failure.

    Source fingerprint changes authorize observation, not retry or computation.
    The original failure, user decision, goal, plan and attempt counts remain.
    """
    recovery = (event.get('payload') or {}).get('recovery') or {}
    if (event.get('kind') != 'worker_recovery_ready'
            or recovery.get('action') != 'verify_changed_executor'
            or recovery.get('tool') != 'inspect_runtime'
            or recovery.get('retry_granted') is not False):
        return None
    from .watch_context import get_context
    from .installations import executor_fingerprint
    from .recovery import failure_reason
    from .workflow_patch import execution_contract
    owner = get_context()
    runtime = session._runtime_snapshot() if session._runtime_snapshot else {}
    goal = session.goal_contract.to_dict()
    key = recovery.get('recovery_key')
    entry = session.recovery_gate.snapshot().get(key, {})
    step = (event.get('payload') or {}).get('step_id')
    node = runtime.get('nodes', {}).get(step, {})
    from .workflow_execution_identity import denied_retry_origin, bound_retry_identity
    origin = denied_retry_origin(node)
    proof_node = origin or node
    if origin and not bound_retry_identity(session, entry.get('tool'), entry.get('params', {}), step,
            owner=(owner.get('username'), owner.get('conv_id'))):
        return None
    params = recovery.get('arguments') or {}
    tool = params.get('tool_name')
    arguments = params.get('arguments')
    root = Path(session.config.project_root).resolve()
    approved = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step]
    from .workflow_user_control import workflow_control_guard
    live_control = getattr(session, '_native_workflow_control', None)
    interrupted = (not workflow_control_guard(session, owner.get('username'), owner.get('conv_id'))()
                   if callable(live_control) else session._interrupt_requested())
    if (not key or not tool or not isinstance(arguments, dict)
            or (owner.get('username'), owner.get('conv_id')) != (runtime.get('username'), runtime.get('conv_id'))
            or not owner.get('username') or not owner.get('conv_id')
            or runtime.get('user_paused') or interrupted
            or goal.get('execution_authorized') is not True
            or runtime.get('plan_version') != goal.get('approved_plan_version')
            or runtime.get('goal_contract', {}).get('version') != goal.get('version')
            or len(approved) != 1 or approved[0].get('tool') != tool
            or execution_contract(approved[0]) != execution_contract(node.get('contract', {}))
            or proof_node.get('status') != 'failed' or proof_node.get('tool_returned') is not True
            or node.get('job_ids') or proof_node.get('job_ids') or entry.get('status') != 'failed' or entry.get('job_ids')
            or entry.get('attempts', 3) >= 3 or entry.get('tool') != tool
            or entry.get('attempt_id') != proof_node.get('attempt_id')
            or entry.get('attempt_id') != recovery.get('failed_attempt_id')
            or entry.get('params') != arguments
            or Path(entry.get('project_root') or '').resolve() != root):
        return None
    current = executor_fingerprint(tool, root)
    if (not current or current != recovery.get('runtime_fingerprint')
            or current == entry.get('runtime_fingerprint')):
        return None
    definition = session.registry.get(tool)
    if definition is None:
        return None
    epoch = executor_epoch(definition.execute)
    identity = json.dumps({'goal': goal, 'owner': {key: owner.get(key) for key in ('username', 'conv_id')},
        'plan_version': runtime.get('plan_version'), 'epoch': epoch, 'entry': entry,
        'node': {key: node.get(key) for key in ('status', 'contract', 'token', 'attempt_id',
            'recovery_key', 'job_ids', 'effective_arguments', 'tool_returned', 'result_ref')},
        'proof_node': {key: proof_node.get(key) for key in ('status', 'contract', 'token', 'attempt_id',
            'recovery_key', 'job_ids', 'effective_arguments', 'tool_returned', 'result_ref')}},
        sort_keys=True, ensure_ascii=False, default=str)
    return {'entry': copy.deepcopy(entry), 'epoch': epoch, 'current': current,
            'identity': identity, 'owner': owner, 'tool': tool, 'key': key}


def refresh_executor_event(session, event):
    """Persist one fresh native observation, with no retry or control changes."""
    binding = _binding(session, event)
    if binding is None:
        return event
    from .workflow_user_control import workflow_control_guard
    from .recovery import failure_reason
    entry, epoch = binding['entry'], binding['epoch']
    current, key, tool = binding['current'], binding['key'], binding['tool']
    control = workflow_control_guard(session, binding['owner']['username'], binding['owner']['conv_id'])
    if not control():
        return event
    cache = session.__dict__.setdefault('_executor_recovery_observations', {})
    cache_key = (event.get('event_id'), key, entry['attempt_id'], epoch['process_epoch'], current)
    previous = cache.get(cache_key)
    if previous and previous['identity'] == binding['identity'] and previous['control']():
        result = copy.deepcopy(event)
        result['payload']['recovery']['current_executor_orientation'] = copy.deepcopy(previous['orientation'])
        return result
    result = copy.deepcopy(event)
    orientation = {'schema': 'executor-recovery-orientation-v1', 'recovery_key': key,
        'failed_attempt_id': entry['attempt_id'], 'failed_owner_pid': entry.get('owner_pid'),
        'failed_at': entry.get('failed_at'), 'failed_runtime_fingerprint': entry.get('runtime_fingerprint'),
        'current_runtime_fingerprint': current, **epoch,
        'old_failure_scope': 'historical attempt; not a verification of the current executor',
        'retry_granted': False, 'scientific_execution': False}
    result['payload']['recovery']['current_executor_orientation'] = orientation
    body = epoch['loaded_executor']
    if body.get('verified') is not True or body.get('matches') is not True:
        orientation.update(status='backend_reload_required', inspection_executed=False, inspection_current=False)
        return result
    safe_params = {'tool_name': tool, 'arguments': copy.deepcopy(entry['params'])}
    raw = session.registry.execute_dict('inspect_runtime', safe_params)
    failed = bool(failure_reason(raw))
    encoded = json.dumps(raw, ensure_ascii=False, default=str)
    session.memory.record_tool_call('lead-orchestrator', 'inspect_runtime', safe_params, encoded, failed=failed)
    call = session.memory.tool_call_log[-1]
    after = _binding(session, event)
    fresh = bool(control() and after is not None and after['identity'] == binding['identity'])
    orientation.update(status=('inspection_failed' if failed else 'inspection_succeeded') if fresh else 'superseded',
        inspection_executed=True, inspection_current=fresh,
        inspection={'call_id': call['call_id'], 'time': call['time'], 'tool': 'inspect_runtime',
                    'result': raw, 'failed': failed})
    session._checkpoint('executor_recovery_native_inspection')
    if fresh:
        if len(cache) >= 16:
            cache.pop(next(iter(cache)))
        cache[cache_key] = {'identity': binding['identity'], 'control': control,
                            'orientation': copy.deepcopy(orientation)}
    return result
