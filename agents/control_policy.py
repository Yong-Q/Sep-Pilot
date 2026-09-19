"""Role capabilities for the control plane, independent of data-path leases.

Roles are supplied by the executing Session, never by model tool arguments.
Elevated control does not grant shell/file writes or cross-session ownership.
"""

JOB_CONTROL = frozenset({'retarget_queued_job', 'cancel_watched_job'})


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
    if tool in JOB_CONTROL | {'apply_workflow_patch','discard_workflow_patch','revalidate_workflow_node_outputs','finish_workflow_node'} and role != 'lead-orchestrator':
        return 'scheduler mutation belongs to main chat; observer/worker must send an evidence-backed proposal'
    if role in {'monitor', 'supervisor'} and tool not in READ_ONLY | {
            'task_line_query', 'recovery_state', 'lifecycle_state', 'supervisor_decision'}:
        return 'observer role is read-only for computation and files; report to main chat instead'
    return ''


def bypasses_data_lease(role, tool):
    from .parallel_workflow import READ_ONLY
    return not role_denial(role, tool) and (tool in READ_ONLY or tool in JOB_CONTROL)
