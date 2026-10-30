"""Model effort for coordination, separate from execution authority and science.

The coordinator routes work and communicates; domain specialists keep their
existing model settings. Consecutive native planning errors can request more
reasoning. No user text, material names or particular test case is classified.
"""


def coordinator_effort(session, role):
    if role != 'lead-orchestrator':
        return None
    configured = getattr(session.config, 'coordinator_reasoning_effort', 'low')
    if configured not in {'low', 'high', 'max'}:
        return None
    plan_tools = {'record_workflow_draft', 'propose_workflow_patch', 'apply_workflow_patch'}
    log = getattr(session.memory, 'tool_call_log', [])
    boundary = getattr(session, '_reply_tool_log_start', None)
    # Old planning failures from another user turn are history, not a reason
    # to keep the current coordinator on an expensive recovery profile.
    current = log[boundary:] if type(boundary) is int and 0 <= boundary <= len(log) else []
    recent = [call for call in current[-12:]
              if call.get('tool') in plan_tools]
    if (configured == 'low' and len(recent) >= 2
            and all(call.get('failed') is True for call in recent[-2:])):
        return 'high'
    return configured
