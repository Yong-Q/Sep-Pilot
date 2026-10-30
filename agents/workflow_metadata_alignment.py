"""Proof-only forward alignment of metadata to a terminal native allocation."""
import copy


def _protected_steps(state, proposed_nodes, affected):
    from .workflow_patch import execution_contract
    changed={item['step_id'] for item in proposed_nodes
        if item.get('step_id') in state.get('nodes',{})
        and execution_contract(item)!=execution_contract(state['nodes'][item['step_id']].get('contract',{}))}
    return set(affected)|changed


def prove_forward_transition(runtime, proposed_nodes, plan_version, affected):
    """A Session-private proof; never dispatch, approve, edit state or a ledger."""
    from .native_resource_carry import prove_native_resource_carry,proof_matches,_metadata_only,_identity
    from .parallel_workflow import ACTIVE
    actor=runtime.main;state=runtime.snapshot();goal=copy.deepcopy(actor.goal_contract.to_dict())
    owner=(runtime.username,runtime.conv_id);turn=getattr(actor,'_api_turn_id',None)
    # A first complete proposal has no persisted executor state to carry.
    # The ordinary draft, owner and approval gates still bind its eventual
    # application. Do not treat a malformed or foreign existing state as new.
    if (state == {} and all(isinstance(v,str) and v for v in owner)
            and goal.get('approved_plan_version') == 0
            and type(plan_version) is int and plan_version > 0
            and isinstance(proposed_nodes,list) and isinstance(affected,(list,set,tuple))):
        return {'no_dispatch':True,'covered_terminal_nodes':[],'proofs':{}}
    if ((state.get('username'),state.get('conv_id'))!=owner
            or not isinstance(proposed_nodes,list) or not isinstance(affected,(list,set,tuple))):
        raise ValueError('native metadata proposal requires its current unpaused owned forward plan')
    incoming={item['step_id']:item for item in proposed_nodes}
    if len(incoming)!=len(proposed_nodes):raise ValueError('native metadata proposal has duplicate steps')
    protected=_protected_steps(state,proposed_nodes,affected)
    active={key for key,node in state.get('nodes',{}).items() if key in protected and node.get('status') in ACTIVE}
    candidates={key for key,node in state.get('nodes',{}).items() if key in incoming
        and node.get('status') in {'prefinish','succeeded'}
        and _metadata_only(node.get('contract',{}),incoming[key])}
    if not active and not candidates:return {'no_dispatch':True,'covered_terminal_nodes':[],'proofs':{}}
    if state.get('user_paused') or type(plan_version) is not int or plan_version<=goal.get('approved_plan_version',0):
        raise ValueError('native metadata proposal requires its current unpaused owned forward plan')
    if active-candidates:raise ValueError('affected runtime node still owns running/unknown resources; native metadata proof is inapplicable')
    forward=copy.deepcopy(goal);forward.update(approved_plan_version=plan_version,approved_nodes=copy.deepcopy(proposed_nodes))
    proofs={}
    for key in sorted(candidates):
        node=state['nodes'][key];future=runtime.futures.get(key)
        if (node.get('jobs_confirmed_terminal') is not True or not node.get('job_ids')
                or future is not None and not future.done()):
            raise ValueError('native metadata alignment requires an ended worker and confirmed terminal jobs')
        jobs={str(job):runtime.job_watch.get(str(job)) or {} for job in node['job_ids']} if runtime.job_watch else {}
        proof=prove_native_resource_carry(state,key,incoming[key],runtime.root,forward,jobs)
        if not proof_matches(state,key,incoming[key],runtime.root,forward,proof):
            raise ValueError('native metadata proof changed before Session transition')
        proofs[key]=proof
    fresh=runtime.snapshot()
    if (runtime.main is not actor or actor.goal_contract.to_dict()!=goal
            or getattr(actor,'_api_turn_id',None)!=turn or actor._interrupt_requested()
            or fresh.get('user_paused') or fresh.get('plan_version')!=state.get('plan_version')
            or fresh.get('goal_contract')!=state.get('goal_contract')
            or (fresh.get('username'),fresh.get('conv_id'))!=owner
            or any(_identity(fresh.get('nodes',{}).get(key,{}))!=_identity(state['nodes'][key])
                   or fresh['nodes'][key].get('status')!=state['nodes'][key].get('status') for key in proofs)):
        raise ValueError('native metadata proposal scope changed during proof')
    return {'no_dispatch':True,'covered_terminal_nodes':sorted(proofs),'proofs':proofs,
            'proposed_plan_version':plan_version,'authority':'proof_only_no_approval'}


def session_transition_proof(session, proposed_nodes, affected, plan_version):
    """Preserve the ordinary ACTIVE guard unless the owned runtime proves it."""
    from .parallel_workflow import ACTIVE
    snapshot=getattr(session,'_runtime_snapshot',None)
    if not callable(snapshot):return {'covered_terminal_nodes':[],'no_dispatch':True}
    state=snapshot();protected=_protected_steps(state,proposed_nodes,affected)
    active={key for key,n in state.get('nodes',{}).items() if key in protected and n.get('status') in ACTIVE}
    callback=getattr(session,'_on_workflow_metadata_transition',None)
    if not callable(callback):
        if active:raise ValueError('affected runtime node still owns running/unknown resources; reconcile before patching')
        return {'covered_terminal_nodes':[],'no_dispatch':True}
    result=callback(proposed_nodes,plan_version,affected)
    if (not isinstance(result,dict) or result.get('no_dispatch') is not True
            or not active<=set(result.get('covered_terminal_nodes',[]))):
        raise ValueError('native metadata proof does not cover every affected active node')
    return result


def resource_mismatch_feedback(runtime, state, step_id, incoming_nodes, cause):
    """Give concrete repair metadata while retaining the rejected exact guard."""
    from .workflow_patch import execution_contract
    node=state['nodes'][step_id];current=runtime.main.goal_contract.to_dict();incoming={x['step_id']:x for x in incoming_nodes};requested=incoming[step_id]
    actual=copy.deepcopy(node.get('effective_arguments') or {})
    keys=set(actual)|set(requested.get('arguments') or {})
    changed=sorted(key for key in keys if actual.get(key)!=requested.get('arguments',{}).get(key))
    receipt={key:copy.deepcopy(node.get(key)) for key in ['attempt_id','job_ids','result_ref','resource_allocation']}
    details={'error_kind':'native_resource_metadata_mismatch','error':'existing terminal allocation metadata has no valid native carry proof',
        'step_id':step_id,'executed':False,'applied':False,'retry_granted':False,'science_failure':False,
        'no_resubmission_required':True,'runtime_plan_version':state.get('plan_version'),
        'approved_plan_version':current.get('approved_plan_version'),'requested_changed_fields':changed,
        'requested_arguments':copy.deepcopy(requested.get('arguments') or {}),'actual_executed_arguments':actual,
        'original_native_attempt':receipt,'guard_reason':str(cause),'next_action':'Preserve the original terminal computation. Propose a forward metadata-only plan using the actual executed allocation; obtain ordinary approval, activate, then revalidate the current native outputs. Do not request a new resource review, submit again or change scientific parameters.'}
    aligned=copy.deepcopy(requested);aligned['arguments']=actual
    proposed=[aligned if x['step_id']==step_id else copy.deepcopy(x) for x in incoming_nodes]
    try:
        proof=prove_forward_transition(runtime,proposed,current['approved_plan_version']+1,[step_id])
        details.update(alignment_candidate_verified=True,
            alignment_candidate={'operation':'upsert','step_id':step_id,'node':execution_contract(aligned)},
            alignment_proof=proof,candidate_authority='proposal_only_not_approved_or_applied')
    except (ValueError,PermissionError,OSError,TypeError,KeyError) as error:
        details.update(alignment_candidate_verified=False,alignment_verification_error=str(error))
    return details


def output_metadata_goal(state, step_id, outputs):
    """Revise only kernel-verified output paths in the same approved plan."""
    goal=copy.deepcopy(state.get('goal_contract') or {})
    found=[item for item in goal.get('approved_nodes',[]) if item.get('step_id')==step_id]
    if len(found)!=1:raise ValueError('native output repair requires one current approved source node')
    found[0]['expected_outputs']=copy.deepcopy(outputs)
    return goal


def refreshed_output_carry(runtime, state, step_id, contract, goal):
    """Re-attest existing carry after a separately proved native output repair.

    The old proof must still be valid. Scientific arguments, allocation,
    producer, attempt and jobs remain fixed; only verified output metadata is
    revised, then the original native proof builder signs the new identity.
    """
    from .native_resource_carry import prove_native_resource_carry,proof_matches
    node=state['nodes'][step_id];old=node.get('approved_resource_carry')
    if old is None:return None
    original=[item for item in state.get('goal_contract',{}).get('approved_nodes',[]) if item.get('step_id')==step_id]
    if len(original)!=1 or not proof_matches(state,step_id,original[0],runtime.root,state['goal_contract'],old):
        raise ValueError('native output repair cannot reuse an invalid original allocation proof')
    revised=copy.deepcopy(state);revised['goal_contract']=copy.deepcopy(goal)
    revised['nodes'][step_id]['contract']=copy.deepcopy(contract)
    incoming=next(item for item in goal['approved_nodes'] if item.get('step_id')==step_id)
    jobs={str(job):runtime.job_watch.get(str(job)) or {} for job in node.get('job_ids',[])} if runtime.job_watch else {}
    proof=prove_native_resource_carry(revised,step_id,incoming,runtime.root,goal,jobs)
    if not proof_matches(revised,step_id,incoming,runtime.root,goal,proof):
        raise ValueError('native output allocation proof changed during re-attestation')
    return proof


def metadata_history_context(runtime, state, goal, line, approval, resource_proofs, *, check_ledger=True):
    """Keep accepted preparation only for its already-terminal native successor.

    This is explicit approved history retention, never a replacement current
    cache digest and never permission to feed stale inputs to a new calculation.
    """
    from .parallel_workflow import _valid_execution_approval
    from .native_resource_carry import proof_matches
    from .workflow_patch import execution_contract
    from .recovery import is_submission
    if not resource_proofs:return None
    owner={'username':runtime.username,'conv_id':runtime.conv_id}
    if ((state.get('username'),state.get('conv_id'))!=(runtime.username,runtime.conv_id)
            or state.get('user_paused') or goal.get('execution_authorized') is not True
            or goal.get('approved_plan_version',0)<state.get('plan_version',0)
            or line.get('plan_version')!=goal.get('approved_plan_version')):
        return None
    if any(state.get('goal_contract',{}).get(k)!=goal.get(k) for k in ['version','method','gases','parameters','execution_mode','execution_authorized']):return None
    if goal['approved_plan_version']>state['plan_version'] and not _valid_execution_approval(approval,goal,line,runtime.username,runtime.conv_id):return None
    incoming={x['step_id']:x for x in goal.get('approved_nodes',[])}
    if set(incoming)!=set(state.get('nodes',{})):return None
    prepared=set();retained_presentations=set()
    for key,node in state['nodes'].items():
        before=node['contract'];after=incoming[key]
        if execution_contract(before)!=execution_contract(after):
            # A stopped or not-yet-run report/figure may be revised under the
            # new approval; this grants no right to reuse its failed output.
            presentation={'generate_result_figures','generate_scientific_report'}
            editable=(before.get('tool') in presentation and after.get('tool') in presentation
                and node.get('status') in {'failed','validation_failed','pending'}
                and not node.get('job_ids') and (node.get('dispatch_phase') not in {'preflight','entered_tool'}
                    or node.get('tool_returned') is True))
            if editable:continue
            proof=resource_proofs.get(key)
            if not proof or not proof_matches(state,key,incoming[key],runtime.root,goal,proof,check_ledger=check_ledger):return None
        elif (before.get('tool') in {'generate_result_figures','generate_scientific_report'}
                and node.get('status')=='succeeded' and node.get('phase')=='finish'):
            retained_presentations.add(key)
        if is_submission(node['contract']['tool'],node['contract'].get('arguments',{})):
            if key not in resource_proofs or node.get('status') not in {'prefinish','succeeded'} or node.get('jobs_confirmed_terminal') is not True:return None
            for dep in node['contract'].get('depends_on',[]):
                prior=state['nodes'].get(dep,{})
                if prior.get('contract',{}).get('tool')=='run_cdft' and prior['contract'].get('arguments',{}).get('action')=='inputs':prepared.add(dep)
    return {'kind':'approved_native_metadata_history','owner':owner,'from_plan_version':state['plan_version'],
        'to_plan_version':goal['approved_plan_version'],'native_resource_proofs':copy.deepcopy(resource_proofs),
        'prepared_steps':sorted(prepared),'retained_presentations':sorted(retained_presentations),
        'no_execution_authority':True,'no_scientific_change':True}


def prepared_terminal_history_proof(runtime,state,step_id,transition):
    """Reuse the original native recipe/manifest proof, bound to its token/event."""
    from .presentation_transition import completed_transition_proof
    from .report_recovery import _OwnedFiles
    from .cdft_result_facts import inspect_cdft_batch_quality
    from .recovery import result_object
    node=state['nodes'][step_id]
    if step_id not in transition.get('prepared_steps',[]):raise ValueError('only the original terminal successor preparation may be retained')
    events=[event for event in state.get('outbox',{}).values() if event.get('kind')=='worker_completed'
        and event.get('payload',{}).get('step_id')==step_id
        and event.get('payload',{}).get('correlation_token')==node.get('token')
        and event.get('payload',{}).get('result_ref')==node.get('result_ref')]
    if len(events)!=1:raise ValueError('completed preparation token lacks its original native completion event')
    jobs={str(j):runtime.job_watch.get(str(j)) or {} for current in state['nodes'].values() for j in current.get('job_ids',[])} if runtime.job_watch else {}
    proof=completed_transition_proof(state,step_id,runtime.root,upstream_jobs=jobs)
    result=result_object(node.get('result'));items=result.get('input_manifest',{}).get('items') or []
    prepared={item['path']:(item['sha256'],item['bytes']) for item in items}
    successors=[current for key,current in state['nodes'].items() if key in transition['native_resource_proofs'] and step_id in current['contract'].get('depends_on',[])]
    if not successors:raise ValueError('completed preparation has no attested original terminal successor')
    equivalent_preparations=[]
    for successor in successors:
        args=successor.get('effective_arguments') or successor['contract']['arguments'];source=node['contract']['arguments']['input_dir']
        if args.get('input_dir')!=source:raise ValueError('terminal calculation did not use the retained preparation path')
        actual=result_object(successor.get('result'));quality=inspect_cdft_batch_quality(actual['work_dir'],owned_root=runtime.root,
            gases=args.get('gases') or [args['gas']],source_input_dir=source)
        sealed={item['path']:(item['sha256'],item['bytes']) for item in quality.get('original_input_facts',{}).get('inputs',[])}
        if quality.get('expected_materials_verified') is not True or sealed!=prepared:
            raise ValueError('terminal calculation source inputs differ from the completed native preparation manifest')
        files=_OwnedFiles(runtime.root,transition['owner'])
        try:checkpoint,_=files.read(successor['result_ref']['checkpoint_path'],as_json=True)
        finally:files.close()
        dependency=checkpoint.get('context',{}).get('dependency_results',{}).get(step_id,{})
        if dependency.get('call_id')!=node.get('result_ref',{}).get('call_id'):
            from .preparation_identity_bridge import prove_historical_preparation
            equivalent_preparations.append(prove_historical_preparation(state,step_id,runtime.root,
                transition['owner'],dependency))
    if equivalent_preparations:
        import hashlib
        from .report_recovery import _canonical
        proof['source_proof_sha256']=hashlib.sha256(_canonical({
            'current_native_proof':proof['source_proof_sha256'],
            'original_preparation_proofs':equivalent_preparations})).hexdigest()
    proof['transition']=copy.deepcopy(transition)
    return proof
