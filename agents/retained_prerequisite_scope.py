"""Native per-node notification scope across an approved downstream condition edit.

This proof never changes origin, ledger, metadata history, a job, or acceptance.
Only explicitly declared temperature/pressure/cycle-independent prerequisites
are eligible. Every other tool and undeclared changed goal parameter fail closed.
"""
import copy
import hashlib
import json
from pathlib import Path

TOOL_GOAL_DEPENDENCIES={
    'run_pacman_charge':{
        'input_parameters':('cif_path','cif_dir'),
        'required_parameters':('method',),
        'independent_goal_parameters':frozenset({'temperature_K','pressure_range','pressure_unit','cycles'}),
        'parameter_dependencies':{'framework_charge_source':'method'},
        'scheduler_job_required':True},
    'validate_framework_charges':{
        'input_parameters':('cif_path',),
        'required_parameters':('expected_net_charge','charge_source'),
        'independent_goal_parameters':frozenset({'temperature_K','pressure_range','pressure_unit','cycles'}),
        'parameter_dependencies':{'expected_cell_net_charge':'expected_net_charge','framework_charge_source':'charge_source'},
        'scheduler_job_required':False},
}
_IDENTITY=('token','attempt_id','job_ids','recovery_key','result_ref','result','effective_arguments',
           'resolved_arguments','runtime_input_bindings','runtime_binding_fingerprint','tool_returned')
_EVENT_KINDS=frozenset({'worker_prefinish','worker_validation_dossier_ready','worker_completed',
    'worker_failed','worker_uncertain','worker_validation_failed'})

def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()

def _same(a,b):return _digest(a)==_digest(b)

def defer_pending_prerequisite_event(session,event,runtime):
    """Keep a still-owned prerequisite notice while a scope edit is negotiated.

    Deferral is not approval, acceptance or a retry. Once the new graph is
    current, the ordinary full source/job proof decides whether to deliver it.
    """
    try:
        payload=event.get('payload') or {}
        origin=payload.get('origin_goal_version')
        if (event.get('kind') not in _EVENT_KINDS or type(origin) is not int
                or origin==session.goal_contract.version):return False
        from .parallel_workflow import WorkflowStore
        goal=session.goal_contract.to_dict()
        old=runtime.get('goal_contract') or {};node=(runtime.get('nodes') or {}).get(payload.get('step_id')) or {}
        policy=TOOL_GOAL_DEPENDENCIES.get((node.get('contract') or {}).get('tool'))
        root=Path(session._evidence_root).absolute().parent;owner=(root.parent.name,root.name)
        prior=old.get('parameters') or {};current=goal.get('parameters') or {}
        changed={k for k in set(prior)|set(current) if prior.get(k)!=current.get(k)}
        token=node.get('token')
        return bool(policy and event.get('kind') in _EVENT_KINDS and changed
            and changed<=policy['independent_goal_parameters']
            and type(origin) is int and origin==old.get('version')==node.get('origin_goal_version')
            and type(goal.get('version')) is int and goal['version']>origin
            and any(old.get(k)!=goal.get(k) for k in ('version','approved_plan_version'))
            and all(old.get(k)==goal.get(k) for k in ('method','gases','execution_mode','execution_authorized','original_goal'))
            and goal.get('execution_authorized') is True
            and (runtime.get('username'),runtime.get('conv_id'))==owner and not runtime.get('user_paused')
            and runtime.get('plan_version')==old.get('approved_plan_version')
            and payload.get('plan_version')==runtime.get('plan_version')
            and payload.get('runtime_id')==WorkflowStore.identity(*owner)
            and payload.get('workflow_id')==session._current_line_id
            and token and payload.get('correlation_token')==token
            and payload.get('attempt_id')==node.get('attempt_id')
            and str(event.get('event_id','')).startswith('runtime:'+payload['runtime_id']+':'+token+':')
            and node.get('status') in {'waiting_jobs','waiting_prerequisite','prefinish','succeeded','failed','uncertain','validation_failed'})
    except (ValueError,KeyError,TypeError,AttributeError,OSError):return False

def _input_facts(files,arguments,policy):
    paths=[]
    for key in policy['input_parameters']:
        if not arguments.get(key):continue
        path=files.owned(arguments[key])
        if path.is_dir():
            paths.extend(sorted(path.glob('*.cif')))
        else:paths.append(path)
    if not 1<=len(paths)<=128 or len(set(paths))!=len(paths):raise ValueError('bounded explicit CIF source required')
    facts=[];total=0
    for path in paths:
        _,fact=files.read(path);total+=fact['size']
        if not 0<fact['size']<=16*1024**2 or total>64*1024**2:raise ValueError('bounded full CIF SHA required')
        facts.append(fact)
    return facts

def retained_node_event_proof(session,event,runtime,*,jobs=None):
    """Prove only this original approved prerequisite belongs to the new graph."""
    files=None
    try:
        from .goal_contract import GoalContract
        from .parallel_workflow import _valid_execution_approval,_approval_digest,WorkflowStore
        from .workflow_patch import execution_contract
        from .workflow_user_control import workflow_control_current
        from .report_recovery import _OwnedFiles
        from .local_preparation_recovery import checkpoint_call_witness
        from .recovery import RecoveryGate,input_fingerprint,result_object,failure_reason
        from .task_line import get_store
        payload=event.get('payload')or{};step=payload.get('step_id');node=(runtime.get('nodes')or{}).get(step)or{}
        contract=node.get('contract')or{};policy=TOOL_GOAL_DEPENDENCIES.get(contract.get('tool'))
        root=Path(session._evidence_root).absolute().parent;owner={'username':root.parent.name,'conv_id':root.name}
        goal=session.goal_contract.to_dict();origin=node.get('origin_goal_version');old_plan=node.get('execution_plan_version')
        if (not policy or event.get('kind') not in _EVENT_KINDS
                or any(runtime.get(k)!=v for k,v in owner.items())
                or runtime.get('user_paused') or runtime.get('pause_kind') not in {None,'failure','reconciliation','resource_review'}
                or runtime.get('status') not in {'active','needs_user','completed'}
                or session._pending_workflow_patch or session._awaiting_plan_approval
                or not workflow_control_current(session,owner['username'],owner['conv_id'])
                or goal.get('execution_authorized') is not True or goal.get('execution_mode')!='workflow'
                or runtime.get('plan_version')!=goal.get('approved_plan_version')
                or _approval_digest(runtime.get('goal_contract')or{})!=_approval_digest(goal)
                or type(origin)is not int or not 0<origin<goal.get('version',0)
                or type(old_plan)is not int or not old_plan<runtime['plan_version']
                or payload.get('origin_goal_version')!=origin
                or payload.get('plan_version') not in {old_plan,runtime['plan_version']}
                or node.get('tool_returned') is not True or not node.get('token')
                or node.get('status') not in {'waiting_jobs','waiting_prerequisite','prefinish','succeeded','failed','uncertain','validation_failed'}):return None
        wid=WorkflowStore.identity(owner['username'],owner['conv_id'])
        if (payload.get('runtime_id')!=wid or payload.get('workflow_id')!=session._current_line_id
                or payload.get('correlation_token')!=node['token'] or payload.get('attempt_id')!=node.get('attempt_id')
                or not str(event.get('event_id','')).startswith('runtime:'+wid+':'+node['token']+':')):return None
        approved=[n for n in goal.get('approved_nodes',[])if n.get('step_id')==step]
        if len(approved)!=1 or execution_contract(approved[0])!=execution_contract(contract):return None
        arguments=node.get('effective_arguments')or contract.get('arguments')or{}
        if any(key not in arguments for key in policy['required_parameters']):return None
        if not GoalContract.from_dict(copy.deepcopy(goal)).guard_tool_call(contract['tool'],arguments)[0]:return None
        line=get_store().get_line(session._current_line_id,username=owner['username'],conv_id=owner['conv_id'])
        approval=(session.context or{}).get('approved_execution_proof')or{}
        if (approval.get('kind') not in {'main_model_applied_plan','explicit_user_parameter_patch'}
                or not _valid_execution_approval(approval,goal,line,owner['username'],owner['conv_id'])):return None
        historical=[]
        for archive in runtime.get('plan_history',[])[-32:]:
            old_goal=archive.get('goal_contract')or{};prior=(archive.get('nodes')or{}).get(step)or{}
            old_approved=[n for n in old_goal.get('approved_nodes',[])if n.get('step_id')==step]
            from .workflow_execution_identity import reconciled_origin_version
            if (archive.get('plan_version')!=old_plan or old_goal.get('approved_plan_version')!=old_plan
                    or reconciled_origin_version(archive,origin,old_goal) is None
                    or old_goal.get('execution_authorized') is not True
                    or len(old_approved)!=1 or execution_contract(old_approved[0])!=execution_contract(contract)
                    or execution_contract(prior.get('contract')or{})!=execution_contract(contract)
                    or not _same({k:prior.get(k)for k in _IDENTITY},{k:node.get(k)for k in _IDENTITY})):continue
            if any(old_goal.get(k)!=goal.get(k)for k in('method','gases','execution_mode','execution_authorized','original_goal')):continue
            oldp=old_goal.get('parameters')or{};newp=goal.get('parameters')or{}
            changed={k for k in set(oldp)|set(newp)if oldp.get(k)!=newp.get(k)}
            if not changed or not changed<=policy['independent_goal_parameters']:continue
            historical.append((archive,old_goal,changed))
        if len(historical)!=1:return None
        archive,old_goal,changed=historical[0]
        old_lines=[h for h in line.get('plan_history',[])if h.get('version')==old_plan]
        if len(old_lines)!=1:return None
        old_steps=[n for n in old_lines[0].get('steps',[])if n.get('step_id')==step]
        if len(old_steps)!=1 or execution_contract(old_steps[0])!=execution_contract(contract):return None
        for field,parameter in policy['parameter_dependencies'].items():
            declared=(goal.get('parameters')or{}).get(field)
            if declared is not None and str(declared).casefold()!=str(arguments.get(parameter)).casefold():return None
        files=_OwnedFiles(root,owner);inputs=_input_facts(files,arguments,policy)
        reference=node.get('result_ref')or{};cid=reference.get('call_id')
        if not isinstance(cid,str)or not cid.isalnum()or reference.get('evidence_path')!=str(root/'evidence'/(cid+'.json')):return None
        call,producer=files.read(reference['evidence_path'],as_json=True);result=result_object(call.get('result'))
        if (call.get('call_id')!=cid or call.get('tool')!=contract['tool']or call.get('failed')is not False
                or not _same(call.get('params'),arguments)or not _same(result,result_object(node.get('result')))
                or failure_reason(result)):return None
        witness=checkpoint_call_witness(files,reference.get('checkpoint_path'),call,step_id=step,require_result_hash=True)
        checkpoint,_=files.read(reference['checkpoint_path'],as_json=True)
        if execution_contract((checkpoint.get('context')or{}).get('assigned_node')or{})!=execution_contract(contract):return None
        ids=node.get('job_ids')or[];job_proofs=[];ledger_identity=None
        if policy['scheduler_job_required']:
            if not ids or len(ids)>64 or not node.get('attempt_id')or not node.get('recovery_key'):return None
            ledger,_=files.read(root/'recovery_state.json',as_json=True);entry=ledger.get(node['recovery_key'])or{}
            key=RecoveryGate().key(contract['tool'],arguments,entry.get('identity_goal_version',origin),step,session.config.project_root)
            if (entry.get('key')!=node['recovery_key']or key!=node['recovery_key']or entry.get('goal_version')!=origin
                    or entry.get('attempt_id')!=node['attempt_id']or entry.get('tool')!=contract['tool']
                    or not _same(entry.get('params'),arguments)or entry.get('fingerprint')!=input_fingerprint(arguments,session.config.project_root)
                    or set(map(str,entry.get('job_ids')or[]))!=set(map(str,ids))
                    or entry.get('status')not in {'submitted','completed_unverified','completed','failed'}):return None
            native_ids=result.get('job_ids')or([str(result['job_id'])]if result.get('job_id')else[])
            if set(map(str,native_ids))!=set(map(str,ids)):return None
            if jobs is None:
                from .job_watch import get_watch
                watch=get_watch();jobs=[watch.get(str(jid))or{}for jid in ids]
            for jid in ids:
                matches=[j for j in jobs if str(j.get('job_id'))==str(jid)]
                if len(matches)!=1:return None
                job=matches[0]
                if (any(job.get(k)!=v for k,v in owner.items())or job.get('tool')!=contract['tool']
                        or job.get('attempt_id')!=node['attempt_id']or job.get('recovery_key')!=node['recovery_key']
                        or job.get('state')in {None,'UNKNOWN'}):return None
                if node.get('status')in {'prefinish','succeeded'}and(job.get('terminal')is not True or job.get('failed')is not False or job.get('state')!='COMPLETED'):return None
                job_proofs.append({k:job.get(k)for k in('job_id','username','conv_id','tool','attempt_id','recovery_key','state','terminal','failed')})
            ledger_identity={k:entry.get(k)for k in('key','tool','params','attempt_id','job_ids','goal_version','fingerprint')}
        else:
            if ids or node.get('recovery_key'):return None
            rows=result.get('results')or[result]
            if result.get('ok')is not True or result.get('input_modified')is not False:return None
            observed={str(Path(f['path'])):f['sha256']for f in inputs}
            declared={str(Path(row.get('cif_path',''))):row.get('sha256')for row in rows}
            if observed!=declared:return None
        proof={'kind':'unchanged_prerequisite_in_current_approved_goal','owner':owner,'step_id':step,
            'origin_goal_version':origin,'current_goal_version':goal['version'],'original_plan_version':old_plan,
            'current_plan_version':runtime['plan_version'],'token':node['token'],'attempt_id':node.get('attempt_id'),
            'job_ids':ids,'contract_sha256':_digest(execution_contract(contract)),
            'old_approval_sha256':_approval_digest(old_goal),'new_approval_sha256':_approval_digest(goal),
            'independent_changed_goal_parameters':sorted(changed),'input_facts':inputs,
            'producer_sha256':producer['sha256'],'producer_checkpoint_sha256':witness['sha256'],
            'ledger_identity':ledger_identity,'job_witnesses':job_proofs,
            'no_origin_or_ledger_rewrite':True,'no_acceptance_or_retry_authority':True}
        proof['proof_sha256']=_digest(proof);return proof
    except (ValueError,KeyError,TypeError,AttributeError,OSError,OverflowError):return None
    finally:
        if files is not None:files.close()

def event_goal_is_current(session,event,runtime,*,jobs=None):
    payload=event.get('payload')or{}
    origin=payload.get('origin_goal_version',session.goal_contract.version)
    if origin==session.goal_contract.version:return True
    # Keep the existing, separately proved metadata-only resume route. A real
    # temperature/scope revision cannot create this native history record.
    try:
        from .workflow_execution_identity import reconciled_origin_version
        from .parallel_workflow import _approval_digest,WorkflowStore
        from .workflow_patch import execution_contract
        goal=session.goal_contract.to_dict();node=(runtime.get('nodes')or{}).get(payload.get('step_id'))or{}
        root=Path(session._evidence_root).absolute().parent;owner=(root.parent.name,root.name)
        approved=[n for n in goal.get('approved_nodes',[])if n.get('step_id')==payload.get('step_id')]
        if ((runtime.get('username'),runtime.get('conv_id'))==owner and not runtime.get('user_paused')
                and goal.get('execution_authorized')is True
                and _approval_digest(runtime.get('goal_contract')or{})==_approval_digest(goal)
                and payload.get('plan_version')==runtime.get('plan_version')==goal.get('approved_plan_version')
                and len(approved)==1 and execution_contract(approved[0])==execution_contract(node.get('contract')or{})
                and payload.get('runtime_id')==WorkflowStore.identity(*owner)
                and payload.get('correlation_token')==node.get('token') and node.get('token')
                and reconciled_origin_version(runtime,origin,goal)==goal['version']):return True
    except (ValueError,KeyError,TypeError,AttributeError,OSError):pass
    return retained_node_event_proof(session,event,runtime,jobs=jobs)is not None
