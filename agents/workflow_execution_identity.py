"""Bind a reviewed retry to its original owned approved-node attempt.

Chat metadata revisions do not create a second scientific task. This helper
never reconciles a goal, grants retry, mutates a ledger or submits work.
"""
from pathlib import Path
import re

RETRY_IDENTITY_DENIAL = 'reviewed retry identity no longer binds the owned approved attempt; no new dispatch'


def denied_retry_origin(node):
    """Only this host-generated preclaim denial may reuse its prior permit."""
    retry = node.get('verified_retry') or {}
    if (node.get('status') != 'failed' or node.get('dispatch_phase') != 'preflight'
            or node.get('attempt_id') or node.get('recovery_key')
            or node.get('tool_returned') or node.get('job_ids')
            or node.get('error') != RETRY_IDENTITY_DENIAL or not retry.get('attempt_id')):
        return None
    from .parallel_workflow import same_scientific_contract
    matches = [old for old in node.get('attempt_history', [])[-16:]
        if old.get('attempt_id') == retry['attempt_id'] and old.get('recovery_key')
        and old.get('token') and old.get('status') == 'failed'
        and same_scientific_contract(old.get('contract', {}), node.get('contract', {}))]
    return matches[-1] if matches else None


def archived_attempt_goal_approval(runtime, goal, step_id, original, entry):
    """Bind the retained attempt to an actual owned archived approved plan.

    This does not manufacture a missing reconciliation record. The archive
    must already carry this exact failed attempt under the current Goal scope;
    a new scientific Goal or a different attempt cannot borrow that approval.
    """
    from .parallel_workflow import same_scientific_contract
    protected = ('version', 'method', 'gases', 'parameters', 'execution_mode', 'execution_authorized')
    for archive in runtime.get('plan_history', [])[-32:]:
        approved_goal = archive.get('goal_contract') or {}
        if (any(approved_goal.get(key) != goal.get(key) for key in protected)
                or not isinstance(archive.get('plan_version'), int)
                or isinstance(archive.get('plan_version'), bool)
                or approved_goal.get('approved_plan_version') != archive['plan_version']
                or archive['plan_version'] >= runtime.get('plan_version', 0)):
            continue
        historical = (archive.get('nodes') or {}).get(step_id, {})
        approvals = [node for node in approved_goal.get('approved_nodes', []) if node.get('step_id') == step_id]
        current_approvals = [node for node in goal.get('approved_nodes', []) if node.get('step_id') == step_id]
        if (len(approvals) != 1 or len(current_approvals) != 1
                or historical.get('attempt_id') != entry.get('attempt_id')
                or historical.get('recovery_key') != entry.get('key')
                or historical.get('origin_goal_version') != entry.get('goal_version')
                or historical.get('token') != original.get('token')
                or historical.get('status') != 'failed'
                or not same_scientific_contract(historical.get('contract', {}), original.get('contract', {}))
                or not same_scientific_contract(approvals[0], current_approvals[0])):
            continue
        return {'kind': 'archived_native_attempt_approval', 'plan_version': archive['plan_version'],
                'goal_version': approved_goal['version'], 'attempt_id': entry['attempt_id']}
    return None


def metadata_only_goal_revision(before, after):
    """Prove an old parser's version bump was a plain resume, not new scope.

    Typed equality alone cannot prove this: force fields, new files or material
    scope may be expressed in prose that older extractors did not recognize.
    """
    from .goal_contract import scope_preserving_continuation
    protected = ('method','gases','parameters','execution_mode','execution_authorized','approved_plan_version')
    if any(before.get(key) != after.get(key) for key in protected):
        return False
    old = before.get('directives') or []
    new = after.get('directives') or []
    if new[:len(old)] != old:
        return False
    approved = after.get('approved_nodes') or []
    def resume(text):
        return (scope_preserving_continuation(str(text),approved)
                or bool(re.fullmatch(r'\s*(?:继续|接着|继续执行|resume|continue)[。.!！\s]*',str(text),re.I)))
    return (all(resume(text) for text in new[len(old):])
            and (before.get('active_goal') == after.get('active_goal') or resume(after.get('active_goal',''))))


def reconciled_origin_version(runtime, origin, goal):
    """Follow only native, already-proved metadata revision history.

    Keeps the execution origin unchanged. It supplies no reconciliation,
    authorization, retry or state mutation; consumers still verify the current
    owner, goal/plan, exact approved contract and source bytes themselves.
    """
    target=goal.get('version')
    if (type(origin) is not int or origin<1 or type(target) is not int or target<origin):
        return None
    version=origin
    for proof in runtime.get('goal_reconciliation_history', []):
        if (proof.get('from_version')==version
                and proof.get('basis')=='identical protected scope and approved node contracts'
                and proof.get('metadata_only_verified') is True
                and type(proof.get('to_version')) is int and version<proof['to_version']<=target):
            version=proof['to_version']
    return version if version==target else None


def bound_retry_identity(session, tool, arguments, step_id, *, owner=None, ledger=None):
    from .parallel_workflow import RESOURCE_ARGUMENTS, _approval_digest, same_scientific_contract
    from .watch_context import get_context
    runtime = session._runtime_snapshot() if session._runtime_snapshot else {}
    goal = session.goal_contract.to_dict()
    if owner is None:
        context = get_context()
        owner = (context.get('username'), context.get('conv_id'))
    from .workflow_user_control import workflow_control_guard
    live_control = getattr(session, '_native_workflow_control', None)
    interrupted = (not workflow_control_guard(session, *owner)() if callable(live_control)
                   else session._interrupt_requested())
    if (not all(owner) or tuple(owner) != (runtime.get('username'), runtime.get('conv_id'))
            or runtime.get('user_paused') or interrupted
            or session._awaiting_plan_approval or session._pending_workflow_patch
            or goal.get('execution_authorized') is not True
            or _approval_digest(goal) != _approval_digest(runtime.get('goal_contract', {}))
            or goal.get('approved_plan_version') != runtime.get('plan_version')):
        return None
    node = runtime.get('nodes', {}).get(step_id, {})
    approved = [n for n in goal.get('approved_nodes', []) if n.get('step_id') == step_id]
    if (len(approved) != 1 or approved[0].get('tool') != tool
            or not same_scientific_contract(node.get('contract', {}), approved[0])):
        return None
    original = node
    if node.get('status') in {'pending', 'running'}:
        retry = node.get('verified_retry') or {}
        matches = [old for old in node.get('attempt_history', [])[-16:]
                   if old.get('attempt_id') == retry.get('attempt_id')
                   and old.get('recovery_key') and old.get('token')]
        if not retry.get('attempt_id') or not matches:
            return None
        original = matches[-1]
        if not same_scientific_contract(original.get('contract', {}), node.get('contract', {})):
            return None
    elif node.get('status') == 'failed' and denied_retry_origin(node):
        original = denied_retry_origin(node)
    elif node.get('status') != 'failed':
        return None
    key = original.get('recovery_key')
    gate = ledger or session.recovery_gate
    entry = gate.snapshot().get(key, {})
    root = Path(session.config.project_root).resolve()
    origin = original.get('origin_goal_version')
    if (not key or not original.get('token') or not original.get('attempt_id')
            or entry.get('key') != key or entry.get('tool') != tool
            or entry.get('status') != 'failed'
            or entry.get('attempt_id') != original['attempt_id']
            or entry.get('goal_version') != origin
            or not isinstance(origin, int) or isinstance(origin, bool)
            or Path(entry.get('project_root') or '').resolve() != root
            or gate.key(tool, entry.get('params', {}), entry.get('identity_goal_version',origin), step_id, root) != key):
        return None
    science = lambda value: {k: v for k, v in value.items() if k not in RESOURCE_ARGUMENTS}
    declared = science(approved[0].get('arguments', {}))
    if (science(arguments) != declared or science(entry.get('params', {})) != declared
            or science(original.get('effective_arguments') or original.get('contract', {}).get('arguments', {})) != declared):
        return None
    version = origin
    for proof in runtime.get('goal_reconciliation_history', []):
        if (proof.get('from_version') == version
                and proof.get('basis') == 'identical protected scope and approved node contracts'
                and proof.get('metadata_only_verified') is True
                and isinstance(proof.get('to_version'), int) and not isinstance(proof.get('to_version'), bool)
                and proof['to_version'] > version):
            version = proof['to_version']
    archived = (archived_attempt_goal_approval(runtime, goal, step_id, original, entry)
                if version != goal.get('version') else None)
    if version != goal.get('version') and archived is None:
        return None
    return {'key': key, 'goal_version': goal['version'],
            'identity_goal_version': entry.get('identity_goal_version',origin), 'attempt_id': entry['attempt_id'],
            'step_id': step_id, 'metadata_reconciled_goal_version': version,
            'archived_native_approval': archived}
