"""Read-only proof of one failed report retained across an approved plan rebase."""
import copy,hashlib,json,re
from .workflow_patch import execution_contract
from .goal_contract import scope_preserving_continuation,prohibits_submission

def _digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def prove_failure_plan_bridge(state,step_id,goal,line,approval,owner):
    if ({k:state.get(k) for k in ('username','conv_id')}!=owner or state.get('user_paused')
            or state.get('plan_version')!=goal.get('approved_plan_version')):raise ValueError('report retry owner/approval changed')
    node=(state.get('nodes') or {}).get(step_id) or {};old_version=node.get('execution_plan_version')
    if (type(old_version)is not int or old_version>=state['plan_version'] or node.get('status')!='failed'
            or (node.get('contract') or {}).get('tool')!='generate_scientific_report'
            or not isinstance(node.get('token'),str) or not re.fullmatch(r'[0-9a-f]{32}',node['token'])
            or node.get('job_ids') or node.get('attempt_id')):
        raise ValueError('only original failed report can bridge old plans')
    histories=[h for h in state.get('plan_history') or [] if h.get('plan_version')==old_version]
    if len(histories)!=1:raise ValueError('old plan archive absent or ambiguous')
    history=histories[0];prior=(history.get('nodes') or {}).get(step_id) or {}
    fields=('status','token','origin_goal_version','execution_plan_version','result_ref','result','tool_returned','dispatch_phase','attempt_id','job_ids')
    if any(prior.get(k)!=node.get(k) for k in fields) or execution_contract(prior.get('contract') or {})!=execution_contract(node['contract']):
        raise ValueError('old report attempt identity differs from current failed node')
    from .parallel_workflow import _valid_execution_approval,_approval_digest,_ACTIVATION_SCOPE
    activation=state.get('activation_recovery') or {};saved=history.get('goal_contract') or {}
    if (not _valid_execution_approval(approval,goal,line,owner['username'],owner['conv_id'])
            or _approval_digest(state.get('goal_contract') or {})!=_approval_digest(goal)
            or activation.get('formal_approval')!=approval or activation.get('owner')!=[owner['username'],owner['conv_id']]
            or activation.get('from_plan_version')!=old_version or activation.get('approved_plan_version')!=state['plan_version']
            or activation.get('approval_sha256')!=_approval_digest(goal) or activation.get('from_approval_sha256')!=_approval_digest(saved)
            or activation.get('status')!='blocked' or (activation.get('details') or {}).get('no_serial_fallback') is not True
            or (activation.get('details') or {}).get('error')!='Unchanged failed nodes remain failed. Restore verified results or provide a verified execution repair; identical upsert is not a retry.'
            or any(saved.get(k)!=goal.get(k) for k in (*_ACTIVATION_SCOPE,'original_goal','active_goal'))):
        raise ValueError('native activation approval does not prove the original plan rebase')
    old_directives=saved.get('directives') or [];new_directives=goal.get('directives') or []
    if (new_directives[:len(old_directives)]!=old_directives or not new_directives[len(old_directives):]
            or any(not scope_preserving_continuation(v,saved.get('approved_nodes') or []) or prohibits_submission(v)
                   or re.search(r'(?:不要|停止|不再|禁止|暂停).{0,8}(?:继续|接着|恢复|推进|执行)|\b(?:do not|don\x27t|stop|never)\s+(?:resume|continue|execute)\b',v,re.I)
                   for v in new_directives[len(old_directives):])):
        raise ValueError('report plan rebase contains an unproved user goal change')
    old_nodes=saved.get('approved_nodes') or [];new_nodes=goal.get('approved_nodes') or []
    before={n['step_id']:execution_contract(n) for n in old_nodes};after={n['step_id']:execution_contract(n) for n in new_nodes}
    if (not before or len(before)!=len(old_nodes) or len(after)!=len(new_nodes) or before!=after
            or set(before)!=set(history.get('nodes') or {}) or set(before)!=set(state.get('nodes') or {})):
        raise ValueError('report rebase changed approved scientific graph')
    if any(execution_contract((history['nodes'][k] or {}).get('contract') or {})!=v or
           execution_contract((state['nodes'][k] or {}).get('contract') or {})!=v for k,v in before.items()):
        raise ValueError('report rebase changed a native contract')
    base=approval.get('base_plan_version')
    if type(base)is not int or not old_version<base<state['plan_version'] or state['plan_version']-old_version>8:
        raise ValueError('unproved intermediate report plan range')
    for version in range(old_version,base+1):
        entries=[h for h in (line.get('plan_history') or []) if h.get('version')==version]
        if len(entries)!=1:raise ValueError('intermediate report plan history absent or ambiguous')
        steps=entries[0].get('steps') or [];items={n.get('step_id'):n for n in steps}
        if len(items)!=len(steps) or not set(before).issubset(items):raise ValueError('intermediate graph membership changed')
        for key,contract in before.items():
            old=items[key];intermediate=execution_contract(old)
            if intermediate==contract:continue
            old_node=history['nodes'][key];attempts=[old_node]+list(old_node.get('attempt_history') or [])
            if (version<=old_version or key!=step_id or contract.get('tool')!='generate_scientific_report'
                    or old.get('status')!='pending' or old.get('done') or old.get('token') or old.get('result_ref')
                    or old.get('job_ids') or old.get('execution_plan_version') or old.get('node_verification') or old.get('output_files')
                    or {k:v for k,v in contract.items() if k!='arguments'}!={k:v for k,v in intermediate.items() if k!='arguments'}
                    or any(a.get('execution_plan_version')==version for a in attempts)):
                raise ValueError('intermediate report change was not proven unexecuted')
    transition={'approval_sha256':_digest(approval),'prior_graph_sha256':_digest(before),'from':old_version,'to':state['plan_version']}
    events=[v for v in (state.get('outbox') or {}).values() if v.get('kind')=='worker_failed'
            and (v.get('payload') or {}).get('step_id')==step_id
            and (v.get('payload') or {}).get('plan_version')==old_version
            and (v.get('payload') or {}).get('correlation_token')==node['token']
            and (v.get('payload') or {}).get('result_ref')==node.get('result_ref')]
    if len(events)!=1:raise ValueError('old report native failed event is absent or ambiguous')
    return {'owner':dict(owner),'step_id':step_id,'token':node['token'],'from_plan_version':old_version,
            'to_plan_version':state['plan_version'],'archive_sha256':_digest(history),'transition_sha256':_digest(transition),
            'failed_event_sha256':_digest(events[0]),'no_retry_authority':True}
