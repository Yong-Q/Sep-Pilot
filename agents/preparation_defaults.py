"""Evidence for native prerequisite defaults, never scientific execution approval.

Routes explicitly name a preparation tool, its defaultable selector, the
validation tool and its consumers. Physical conditions are never defaulted.
"""
import copy
import hashlib
import json
import re


ROUTES = {
    'run_pacman_charge': {
        'selector': 'method', 'validator': 'validate_framework_charges',
        'consumers': {'run_gcmc_isotherm', 'run_gcmc_batch', 'run_cdft'},
    },
}


def preparation_default_evidence(session, tool_name, parameter, related_nodes):
    route = ROUTES.get(tool_name)
    goal = session.goal_contract
    draft = session.context.get('workflow_draft') or {}
    if (not route or parameter != route['selector']
            or not goal.execution_authorized or goal.execution_mode != 'workflow'
            or goal.approved_nodes or draft.get('goal_version') != goal.version
            or not draft.get('completion_criteria') or not related_nodes):
        return None
    # Existing Goal constraints remain authoritative. This also protects a
    # natural prohibition that predates typed charge-source extraction.
    directives = '\n'.join(goal.directives or [goal.active_goal])
    if re.search(r'(?:不要|不用|不得|禁止|do\s+not|don.t|never).{0,16}'
                 r'(?:赋|重算|assign|predict).{0,12}(?:电荷|charges?)', directives, re.I):
        return None
    from .task_line import get_store
    from .watch_context import get_context
    owner = get_context()
    line = get_store().get_line(session._current_line_id) if session._current_line_id else None
    if (not line or not owner.get('username') or not owner.get('conv_id')
            or (line.get('username'), line.get('conv_id')) != (owner['username'], owner['conv_id'])
            or line.get('plan_version', 0) != 0):
        return None
    nodes = {n['step_id']: n for n in draft.get('nodes', [])}
    if (not set(related_nodes) <= set(nodes) or any(n.get('missing_parameters') for n in nodes.values())):
        return None
    tool = session.registry.get(tool_name)
    prop = ((tool.input_schema if tool else {}).get('properties') or {}).get(parameter) or {}
    if 'default' not in prop:
        return None
    default = prop['default']
    if 'enum' in prop and default not in prop['enum']:
        return None

    def ancestors(step, seen=None):
        seen = set() if seen is None else seen
        for dep in nodes[step].get('depends_on', []):
            if dep not in nodes:
                return None
            if dep not in seen:
                seen.add(dep)
                if ancestors(dep, seen) is None:
                    return None
        return seen

    selected = []
    for key in related_nodes:
        node = nodes[key]
        if node.get('tool') != tool_name:
            continue
        args = node.get('arguments') or {}
        # This tool records evidence; it never rewrites a proposed method.
        # The planner must already have selected the actual schema default.
        if args.get(parameter) != default or not goal.guard_tool_call(tool_name, args, enforce_approved_node=False)[0]:
            return None
        validators = [vid for vid, candidate in nodes.items()
            if candidate.get('tool') == route['validator'] and key in (ancestors(vid) or set())]
        consumers = [cid for cid, candidate in nodes.items()
            if candidate.get('tool') in route['consumers']
            and any(vid in (ancestors(cid) or set()) for vid in validators)]
        if not validators or not consumers:
            return None
        if any(not goal.guard_tool_call(nodes[cid]['tool'], nodes[cid].get('arguments') or {},
                enforce_approved_node=False)[0] for cid in consumers):
            return None
        selected.append({'step_id': key, 'tool_name': tool_name, 'parameter': parameter,
            'value': copy.deepcopy(default), 'source': 'native_registry.input_schema.properties.default',
            'validation_steps': validators, 'consumer_steps': consumers})
    if not selected:
        return None
    encoded = json.dumps(draft, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    return {'goal_version': goal.version, 'draft_version': draft['version'],
        'draft_sha256': hashlib.sha256(encoded).hexdigest(), 'selections': selected,
        'execution_authority_unchanged': True, 'science_arguments_unchanged': True,
        'validation_required': True}


def pending_default_review(session):
    """Wake once for a native review; never clear an untyped legacy question."""
    pending = session._pending_user_interaction or {}
    params = pending.get('params') or {}
    if (pending.get('tool') != 'user_decision' or not params.get('decision_id')
            or params.get('candidate_job_id') or params.get('withdrawn_reconsideration')):
        return None
    for tool_name, route in ROUTES.items():
        evidence = preparation_default_evidence(session, tool_name, route['selector'], params.get('related_nodes') or [])
        if evidence:
            key = str(evidence['goal_version']) + ':' + tool_name + ':' + route['selector']
            seen = session.context.setdefault('native_default_review_events', [])
            if key in seen:
                return None
            seen.append(key)
            session.context['native_default_review_events'] = seen[-16:]
            return {'ok': True, 'status': 'preparation_default_review_required',
                'next_action': 'review_preparation_default', 'decision_id': params['decision_id'],
                'default_evidence': evidence,
                'instruction': 'Review the pending question against this native schema-default evidence. '
                    'If the only unresolved issue is selecting this prerequisite default, call '
                    'resolve_preparation_defaults with this decision_id/tool_name/parameter and then '
                    'compile the full draft. Keep any real missing scientific input as a user question. '
                    'Never skip the prerequisite or its validation; do not claim a job was submitted.'}
    return None


def preparation_review_binding(session, event):
    """Read-only permission to review a matching internal default question.

    This does not resolve the question, grant execution, or permit replay of
    unrelated old events. The native resolver remains the decision owner.
    """
    payload=event.get('payload')or{}
    pending=session._pending_user_interaction or{};params=pending.get('params')or{}
    evidence=payload.get('default_evidence')or{};selections=evidence.get('selections')or[]
    if (event.get('kind')!='preflight_recovered'
            or payload.get('status')!='preparation_default_review_required'
            or payload.get('requires_user',False)is not False
            or session._awaiting_plan_approval or session._pending_workflow_patch
            or pending.get('tool')!='user_decision'
            or not params.get('decision_id')or params['decision_id']!=payload.get('decision_id')
            or params.get('candidate_job_id')or params.get('withdrawn_reconsideration')or not selections):
        return None
    selected=selections[0]
    validation_session=copy.copy(session)
    validation_session.goal_contract=copy.deepcopy(session.goal_contract)
    current=preparation_default_evidence(validation_session,selected.get('tool_name'),selected.get('parameter'),
        params.get('related_nodes')or[])
    if not current or current!=evidence:
        return None
    return {'event_id':event.get('event_id'),'decision_id':params['decision_id'],
        'goal_version':current['goal_version'],'draft_sha256':current['draft_sha256'],
        'permission':'review_native_preparation_default_only',
        'execution_authority_unchanged':True,'pending_question_preserved':True}
