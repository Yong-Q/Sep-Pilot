"""Current candidate/withdrawal identities. Never execution or retry authority."""
import copy
import hashlib
import json
import time
import uuid
from .workflow_patch import execution_contract


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def _without_receipts(value):
    if isinstance(value,dict):return {k:_without_receipts(v) for k,v in value.items() if k!='resource_review_id'}
    if isinstance(value,list):return [_without_receipts(v) for v in value]
    return value


def semantic_fingerprint(patch):
    changes=[]
    for change in patch.get('changes',[]):
        row={'operation':change.get('operation'),'step_id':change.get('step_id')}
        if change.get('operation')=='upsert':row['node']=_without_receipts(execution_contract(change.get('node') or {}))
        changes.append(row)
    return digest(sorted(changes,key=lambda x:(str(x['step_id']),str(x['operation']))))


def authority_fingerprint(scope):
    # Text-only goal/plan version changes are not new scientific permission.
    scope=copy.deepcopy(scope or {});scope.pop('approved_plan_version',None)
    scope.pop('version',None)
    if 'approved_contracts' in scope:
        scope['approved_contracts']=sorted(_without_receipts(scope['approved_contracts']),key=digest)
    return digest(_without_receipts(scope))


def evidence_fingerprint(runtime,patch):
    nodes=(runtime or {}).get('nodes',{});selected={change.get('step_id') for change in patch.get('changes',[])}
    pending=list(selected)
    for change in patch.get('changes',[]):pending.extend((change.get('node')or{}).get('depends_on',[]))
    while pending:
        key=pending.pop()
        if key not in selected:selected.add(key)
        for dep in nodes.get(key,{}).get('contract',{}).get('depends_on',[]):
            if dep not in selected:selected.add(dep);pending.append(dep)
    rows={}
    for key in sorted(str(k) for k in selected if k is not None):
        node=nodes.get(key,{})
        rows[key]={'status':node.get('status'),'token':node.get('token'),'attempt_id':node.get('attempt_id'),
            'job_ids':node.get('job_ids') or [],'producer_call_id':(node.get('result_ref')or{}).get('call_id'),
            'artifacts':node.get('artifacts') or {},
            'native_fact_sha256':digest((node.get('validation_dossier')or{}).get('method_facts')or{})}
    return digest({'owner':[(runtime or{}).get('username'),(runtime or{}).get('conv_id')],'nodes':rows})


def withdrawal_record(session,patch,reason,runtime,source):
    from .requested_parameters import proposal_goal_scope
    return {'patch':copy.deepcopy(patch),'reason':reason,'schema_version':1,'withdrawal_id':uuid.uuid4().hex,
        'withdrawn_at':time.time(),'control_source':copy.deepcopy(source),
        'authority_scope_sha256':authority_fingerprint(proposal_goal_scope(session.goal_contract)),
        'evidence_fingerprint':evidence_fingerprint(runtime,patch),'semantic_sha256':semantic_fingerprint(patch)}


def _withdrawal_id(entry):
    return entry.get('withdrawal_id') or digest({'patch':entry.get('patch'),'reason':entry.get('reason')})


def _matching_entry(session,patch,runtime):
    from .requested_parameters import proposal_goal_scope
    semantic=semantic_fingerprint(patch);authority=authority_fingerprint(proposal_goal_scope(session.goal_contract))
    for entry in reversed(session.context.get('withdrawn_workflow_patches',[])[-64:]):
        prior=entry.get('patch') or {}
        source=entry.get('control_source')
        if source and source.get('origin')!='user':continue
        owner={key:(runtime or{}).get(key) for key in ('username','conv_id')}
        if source and all(owner.values()) and source.get('owner')!=owner:continue
        if semantic_fingerprint(prior)!=semantic:continue
        old_scope=entry.get('authority_scope_sha256')
        if old_scope is None:
            if not prior.get('goal_scope'):continue  # Truly unscoped legacy history grants no new guard authority.
            old_scope=authority_fingerprint(prior['goal_scope'])
        if old_scope!=authority:continue
        baseline=entry.get('evidence_fingerprint')
        if baseline is not None and baseline!=evidence_fingerprint(runtime,patch):continue
        return entry
    return None


def explicit_owner_command(session,requested_changes,runtime):
    """Only the owner API's exact structured request can reopen unchanged scope.

    A main-agent claim, a fresh chat turn or a natural-language answer is not a
    typed consent event. Ordinary scientific approval/execution guards remain.
    """
    if requested_changes is None or not (runtime or{}).get('user_paused'):return False
    from .watch_context import get_context
    from .control_policy import controller_authorized
    from .answer_delivery import digest as request_digest
    source=session._operation_control_source() or {}
    role=getattr(session.current_agent,'name','')
    return (role=='operations' and source.get('api_operation')=='stage_paused_plan'
        and source.get('origin')=='user' and source.get('turn_id')
        and source.get('requested_changes_sha256')==request_digest(requested_changes)
        and controller_authorized(role,source,get_context(),session.goal_contract.version,
                                  getattr(session,'_operations_authority',None)))


def withdrawn_conflict(session,patch,runtime,requested_changes=None):
    entry=_matching_entry(session,patch,runtime)
    if entry is None:return None
    if explicit_owner_command(session,requested_changes,runtime):return None
    return {'ok':False,'status':'withdrawn_duplicate','error_kind':'withdrawn_workflow_proposal',
        'error':'The same withdrawn candidate has unchanged typed authority and native evidence.',
        'applied':False,'executed':False,'retryable':False,
        'withdrawn_plan_version':entry['patch'].get('new_version'),'withdrawal_reason':str(entry.get('reason',''))[:600],
        'semantic_sha256':semantic_fingerprint(patch),'evidence_baseline_available':entry.get('evidence_fingerprint') is not None,
        'next_action':'Continue diagnosis under the approved workflow. Do not renumber this rejected candidate or automatically ask the same question. A changed typed user goal, changed native evidence or an authenticated owner structured command may reopen negotiation. Plain discussion answers, a new turn ID and generic continuation are not typed consent; if natural-language consent cannot be reliably represented, explain that limit without repeatedly asking the rejected question.'}


def reconsideration_binding(session,version,runtime,owner):
    from .requested_parameters import proposal_goal_scope
    choices=[entry for entry in session.context.get('withdrawn_workflow_patches',[])[-64:]
             if (entry.get('patch')or{}).get('new_version')==version]
    if len(choices)!=1:raise ValueError('one exact withdrawn candidate is required for reconsideration')
    entry=choices[0]
    source=entry.get('control_source')
    if source and (source.get('origin')!='user' or source.get('owner')!=owner):
        raise PermissionError('withdrawn candidate belongs to another user/session decision')
    patch=entry['patch'];latest=_matching_entry(session,patch,runtime)
    if latest is not None:entry=latest
    binding={'withdrawal_id':_withdrawal_id(entry),'semantic_sha256':semantic_fingerprint(patch),
        'authority_scope_sha256':authority_fingerprint(proposal_goal_scope(session.goal_contract)),
        'owner':copy.deepcopy(owner),'asked_at':time.time(),'withdrawn_plan_version':entry['patch'].get('new_version')}
    summary=[{'operation':change.get('operation'),'step_id':change.get('step_id')} for change in patch.get('changes',[])][:16]
    return binding,summary,str(entry.get('reason',''))[:500]


def proposal_state(session,requested_version=None):
    pending=session._pending_workflow_patch
    withdrawn=session.context.get('withdrawn_workflow_patches',[])[-8:]
    status='not_current'
    if pending and pending.get('new_version')==requested_version:status='pending'
    elif any((entry.get('patch')or{}).get('new_version')==requested_version for entry in withdrawn):status='withdrawn'
    elif requested_version==session.goal_contract.approved_plan_version:status='approved_not_a_submission_receipt'
    return {'diagnostic_only':True,'approved_plan_version':session.goal_contract.approved_plan_version,
        'pending':({k:pending.get(k) for k in ['base_version','new_version','reason']} if pending else None),
        'withdrawn':[{'plan_version':(entry.get('patch')or{}).get('new_version'),'reason':str(entry.get('reason',''))[:500]}
                     for entry in withdrawn],'requested_plan_version':requested_version,'requested_status':status}
