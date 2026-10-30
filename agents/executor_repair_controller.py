"""Deterministic recovery of verified executor-import faults, never physics.

Only the original authorized task, retained failed attempt and a changed,
loaded executor can enter. Two actual native observations feed the existing
Session recovery gate. The normal runtime still owns every subsequent claim.
"""
import copy
import hashlib
import json
from pathlib import Path
import re
import time
import uuid


def _capture(runtime, step, *, control):
    from .executor_identity import executor_identity
    from .installations import executor_fingerprint
    from .recovery import input_fingerprint, is_submission
    from .workflow_execution_identity import denied_retry_origin, bound_retry_identity
    state = runtime.snapshot(); actor = runtime.main
    from .watch_context import get_context
    context = get_context()
    if (context.get('username'), context.get('conv_id')) != (runtime.username, runtime.conv_id):
        return None
    if (not control() or state.get('user_paused') or actor._awaiting_plan_approval
            or actor._pending_workflow_patch
            or (actor._pending_user_interaction and actor._pending_user_interaction.get('tool') != 'lifecycle_wait')
            or actor.goal_contract.execution_authorized is not True
            or (state.get('username'), state.get('conv_id')) != (runtime.username, runtime.conv_id)):
        return None
    node = state.get('nodes', {}).get(step, {})
    origin = denied_retry_origin(node) or node
    if (node.get('status') != 'failed' or node.get('job_ids')
            or origin.get('status') != 'failed' or origin.get('job_ids')
            or origin.get('tool_returned') is not True or not origin.get('attempt_id')
            or not node.get('token') or not origin.get('recovery_key')):
        return None
    gate = actor.recovery_gate
    entry = gate.snapshot().get(origin['recovery_key'], {})
    args = entry.get('params') or {}; tool = entry.get('tool')
    if (entry.get('status') != 'failed' or entry.get('job_ids') or entry.get('attempts', 3) >= 3
            or entry.get('attempt_id') != origin['attempt_id'] or not is_submission(tool, args)):
        return None
    from .runtime_health_probe import PROFILES
    profile = PROFILES.get(tool)
    if not profile or profile.get('kind') != 'project_python_child':
        return None
    # Only native Python import failures belong to this repair class. Numeric,
    # convergence, unknown submission and arbitrary error text are excluded.
    failure = entry.get('last_result') or {}
    trace = '\n'.join(str(failure.get(key) or '') for key in ('error', 'stderr'))[:32768]
    modules = profile.get('modules') or ()
    if (not re.search(r'(?m)^(?:ModuleNotFoundError|ImportError):', trace)
            or 'Traceback (most recent call last)' not in trace
            or not any(module in trace or module.replace('.', '/') in trace for module in modules)):
        return None
    bound = bound_retry_identity(actor, tool, args, step,
        owner=(runtime.username, runtime.conv_id), ledger=gate)
    if not bound or bound['key'] != origin['recovery_key']:
        return None
    current = executor_fingerprint(tool, actor.config.project_root)
    if not current or current == entry.get('runtime_fingerprint'):
        return None
    loaded = executor_identity(actor.registry.get(tool).execute)
    if loaded.get('verified') is not True or loaded.get('matches') is not True:
        return None
    fingerprint = input_fingerprint(args, actor.config.project_root)
    if fingerprint != entry.get('fingerprint'):
        return None
    if (entry.get('approved_runtime_fingerprint') == current
            and entry.get('approved_fingerprint') == fingerprint
            and entry.get('approved_attempt_id') == entry['attempt_id']):
        return None
    ref = origin.get('result_ref') or {}
    call_id = ref.get('call_id')
    evidence = runtime.root / 'evidence' / (str(call_id) + '.json')
    if (not isinstance(call_id, str) or not re.fullmatch('[A-Za-z0-9_-]{1,128}', call_id)
            or Path(ref.get('evidence_path') or '') != evidence):
        return None
    identity = {'owner': (runtime.username, runtime.conv_id),
        'goal': actor.goal_contract.to_dict(), 'plan': state.get('plan_version'),
        'step': step, 'token': node['token'], 'node': {key: node.get(key) for key in
            ('status', 'contract', 'attempt_id', 'recovery_key', 'verified_retry', 'job_ids', 'dispatch_phase')},
        'key': bound['key'], 'attempt': entry['attempt_id'], 'source': current,
        'inputs': fingerprint, 'loaded': loaded['loaded_code_sha256']}
    epoch = hashlib.sha256(json.dumps(identity, sort_keys=True, default=str).encode()).hexdigest()
    return {'identity': identity, 'epoch': epoch, 'entry': entry, 'tool': tool,
        'args': args, 'token': node['token'], 'evidence': evidence, 'state': state}


def repair_verified_executors(runtime, *, paused=False, max_checks=1):
    """One bounded controller pass; no graph edits or direct tool submissions."""
    if paused:
        return []
    from .state_io import json_transaction
    from .workflow_user_control import workflow_control_guard
    from .recovery import failure_reason, result_object
    actor = runtime.main
    control = workflow_control_guard(actor, runtime.username, runtime.conv_id)
    results = []
    for step, node in runtime.snapshot().get('nodes', {}).items():
        if len(results) >= max_checks: break
        if node.get('status') != 'failed': continue
        captured = _capture(runtime, step, control=control)
        if captured is None: continue
        marker = {'epoch': captured['epoch'], 'claim_id': uuid.uuid4().hex,
                  'status': 'checking', 'started_at': time.time(), 'retry_granted': False}
        with json_transaction(runtime.store.path) as data:
            state = data.get('workflows', {}).get(runtime.workflow_id, {})
            current = state.get('nodes', {}).get(step, {})
            prior = current.get('executor_repair_controller') or {}
            if (not control() or state.get('user_paused')
                    or (state.get('username'), state.get('conv_id')) != (runtime.username, runtime.conv_id)
                    or current.get('token') != captured['token'] or current.get('status') != 'failed'
                    or state.get('goal_contract') != captured['state'].get('goal_contract')
                    or prior.get('epoch') == captured['epoch']):
                continue
            current['executor_repair_controller'] = copy.deepcopy(marker)
        def valid():
            fresh = _capture(runtime, step, control=control)
            return fresh is not None and fresh['identity'] == captured['identity']
        def commit_valid():
            """No ledger reads: safe inside RecoveryGate's own transaction."""
            from .executor_identity import executor_identity
            from .installations import executor_fingerprint
            from .recovery import input_fingerprint
            if (not control() or runtime.main is not actor
                    or actor.goal_contract.to_dict() != captured['identity']['goal']
                    or actor._pending_workflow_patch or actor._awaiting_plan_approval
                    or (actor._pending_user_interaction
                        and actor._pending_user_interaction.get('tool') != 'lifecycle_wait')):
                return False
            state = runtime.snapshot(); current = state.get('nodes', {}).get(step, {})
            node_identity = {key: current.get(key) for key in
                ('status', 'contract', 'attempt_id', 'recovery_key', 'verified_retry', 'job_ids', 'dispatch_phase')}
            if ((state.get('username'), state.get('conv_id')) != captured['identity']['owner']
                    or state.get('user_paused') or state.get('plan_version') != captured['identity']['plan']
                    or state.get('goal_contract') != captured['state'].get('goal_contract')
                    or current.get('token') != captured['token']
                    or node_identity != captured['identity']['node']):
                return False
            body = executor_identity(actor.registry.get(captured['tool']).execute)
            return (body.get('verified') is True and body.get('matches') is True
                and body.get('loaded_code_sha256') == captured['identity']['loaded']
                and executor_fingerprint(captured['tool'], actor.config.project_root) == captured['identity']['source']
                and input_fingerprint(captured['args'], actor.config.project_root) == captured['identity']['inputs'])
        def record(tool, params):
            raw = actor.registry.execute_dict(tool, params)
            actor.memory.record_tool_call('operations', tool, params,
                json.dumps(raw, ensure_ascii=False, default=str), failed=bool(failure_reason(raw)))
            actor._checkpoint('native_executor_repair_observation')
            return raw, actor.memory.tool_call_log[-1]['call_id']
        try:
            if not valid(): raise ValueError('native repair scope changed before diagnosis')
            diagnostic, diagnosis_id = record('read_file', {'path': str(captured['evidence']), 'max_chars': 16384})
            marker['diagnosis_call_id'] = diagnosis_id
            if failure_reason(diagnostic): raise ValueError('native failure receipt could not be inspected')
            if not valid(): raise ValueError('native repair scope changed during diagnosis')
            verification, verification_id = record('inspect_runtime',
                {'tool_name': captured['tool'], 'arguments': copy.deepcopy(captured['args'])})
            verification = result_object(verification)
            health = verification.get('runtime_health_probe') or {}
            marker.update(verification_call_id=verification_id,
                health_status=health.get('status'), health_reason=health.get('reason'),
                unavailable_checks=[item.get('name') for item in verification.get('checks', [])
                                    if item.get('available') is not True][:32])
            if (failure_reason(verification) or verification.get('ok') is not True
                    or health.get('ok') is not True or health.get('status') != 'passed'
                    or health.get('profile') != 'project_python_child' or not valid()):
                raise ValueError('configured trusted-child import health or current scope was not verified')
            permit = actor._prepare_retry(captured['entry']['key'], captured['tool'],
                copy.deepcopy(captured['args']),
                'The retained native failure is a confirmed Python import error before any scheduler submission.',
                diagnosis_id,
                'The changed loaded executor and configured scientific interpreter passed native import health checks.',
                verification_id, control_valid=commit_valid)
            marker.update(status='permit_granted', retry_granted=True, permit=permit,
                          diagnosis_call_id=diagnosis_id, verification_call_id=verification_id)
        except Exception as error:
            marker.update(status='verification_failed' if control() else 'superseded',
                          error_type=type(error).__name__, error=str(error)[:1000])
        marker['finished_at'] = time.time()
        runtime.store.update(runtime.workflow_id, step, captured['token'], {'executor_repair_controller': marker})
        results.append(copy.deepcopy(marker))
    return results
