"""Role capabilities for the control plane, independent of data-path leases.

Roles are supplied by the executing Session, never by model tool arguments.
Elevated control does not grant shell/file writes or cross-session ownership.
"""

JOB_CONTROL = frozenset({'retarget_queued_job', 'cancel_watched_job'})
OPERATIONS_CONTROL = JOB_CONTROL | frozenset({
    'propose_workflow_patch','apply_workflow_patch','discard_workflow_patch',
    'execute_workflow','message_workflow_node','repair_workflow_runtime_inputs',
    'revalidate_workflow_node_outputs','reconcile_watched_job','prepare_retry','requeue_local_preparation','prepare_cdft_partial_recovery',
    'recovery_state','lifecycle_state','inspect_runtime','task_line_query',
    'resource_health','assess_job_resources','restart_backend',
})


def issue_operations_authority(source, owner, goal_version):
    """Issue a native delegation; caller must be the main Session controller."""
    import copy
    require_user_control_source(source, owner)
    if not source.get('turn_id'):
        raise PermissionError('operations delegation requires an authenticated turn identity')
    return {'source':copy.deepcopy(source),'owner':dict(source['owner']),
            'goal_version':goal_version,'tools':sorted(OPERATIONS_CONTROL)}


def controller_authorized(role, source, owner, goal_version, authority=None):
    """One policy for every Session control entry; model arguments grant nothing."""
    if role == 'lead-orchestrator':
        return True
    if role != 'operations' or not isinstance(authority,dict):
        return False
    expected={'username':owner.get('username'),'conv_id':owner.get('conv_id')}
    issued=authority.get('source') or {}
    return bool(expected['username'] and expected['conv_id']
        and authority.get('owner')==expected and issued.get('owner')==expected
        and (source or {}).get('owner')==expected
        and issued.get('origin')=='user' and issued.get('turn_id')
        and authority.get('goal_version')==goal_version
        and set(authority.get('tools') or ())==set(OPERATIONS_CONTROL)
        and (source or {}).get('origin') in {'user','system'}
        and ((source or {}).get('origin')=='system' or issued.get('turn_id')==source.get('turn_id')))


def operations_authority_required(tool):
    from .parallel_workflow import READ_ONLY
    return tool not in READ_ONLY | {'handoff_to_lead-orchestrator'}


def require_user_control_source(source, owner):
    """Check provenance, not the meaning of a user's natural-language request.

    Main chat chooses an actual operation. User intent/ambiguity/withdrawal
    are understood by that model, never by a keyword list or classifier bool.
    Background events cannot manufacture a new human-directed operation.
    """
    if not source or source.get('origin') != 'user':
        raise PermissionError('scheduler control requires a real user turn; background event may only propose an operation')
    if source.get('owner') != {'username': owner.get('username'), 'conv_id': owner.get('conv_id')}:
        raise PermissionError('control turn belongs to another user/session')


def role_denial(role, tool):
    from .parallel_workflow import READ_ONLY
    if role=='operations' and tool not in OPERATIONS_CONTROL | READ_ONLY | {
            'handoff_to_lead-orchestrator','handoff_to_patcher','get_tool_schema'}:
        return 'operations controls infrastructure; scientific execution and acceptance remain with approved workers and main chat'
    if tool in JOB_CONTROL | {'apply_workflow_patch','discard_workflow_patch','revalidate_workflow_node_outputs','finish_workflow_node','repair_workflow_runtime_inputs','requeue_local_preparation','prepare_cdft_partial_recovery'} and role not in {'lead-orchestrator','operations'}:
        return 'scheduler mutation belongs to main chat; observer/worker must send an evidence-backed proposal'
    if role in {'monitor', 'supervisor'} and tool not in READ_ONLY | {
            'task_line_query', 'recovery_state', 'lifecycle_state', 'supervisor_decision'}:
        return 'observer role is read-only for computation and files; report to main chat instead'
    return ''


def bypasses_data_lease(role, tool):
    from .parallel_workflow import READ_ONLY
    return not role_denial(role, tool) and (tool in READ_ONLY or tool in JOB_CONTROL)
