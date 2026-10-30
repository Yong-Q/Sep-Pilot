"""Proof-bound carry of unchanged local results across an approved DAG revision.

A retained result is not an execution permit. Any unknown tool, changed ancestry,
foreign job, or absent original evidence fails closed.
"""
import copy
from .workflow_patch import execution_contract

_LOCAL_TOOLS={'run_cdft','analyze_gcmc_screening','analyze_cdft_screening',
              'inspect_path','validate_framework_charges','query_literature','download_scientific_file'}


def preparation_transition(state, goal, line, approval, step_id, *, owner):
    from .parallel_workflow import _valid_execution_approval
    old=state.get('goal_contract') or {}; nodes=state.get('nodes') or {}
    if (state.get('plan_version') == goal.get('approved_plan_version')
        or not _valid_execution_approval(approval,goal,line,owner['username'],owner['conv_id'])
        or approval.get('base_plan_version')!=state.get('plan_version')
        or (state.get('username'),state.get('conv_id'))!=(owner['username'],owner['conv_id'])
        or state.get('user_paused') or state.get('pause_kind') not in {None,'failure','reconciliation'}
        or old.get('approved_plan_version')!=state.get('plan_version')
        or any(old.get(k)!=goal.get(k) for k in ('method','gases','parameters','execution_mode','execution_authorized','original_goal'))):
        return None
    old_approved=old.get('approved_nodes') or [];new_approved=goal.get('approved_nodes') or []
    before={n.get('step_id'):n for n in old_approved};after={n.get('step_id'):n for n in new_approved}
    if len(before)!=len(old_approved) or len(after)!=len(new_approved) or None in before or None in after:return None
    node=nodes.get(step_id) or {};contract=node.get('contract') or {}; tool=contract.get('tool')
    if tool not in _LOCAL_TOOLS or tool=='run_cdft' and contract.get('arguments',{}).get('action')!='inputs':return None
    pending=[step_id];seen=set()
    while pending:
        key=pending.pop()
        if key in seen:continue
        seen.add(key)
        if len(seen)>128:return None
        item=nodes.get(key) or {};declared=item.get('contract') or {}
        if (key not in before or key not in after or item.get('status')!='succeeded'
            or item.get('phase')!='finish' or not item.get('token')
            or execution_contract(before[key])!=execution_contract(declared)
            or execution_contract(after[key])!=execution_contract(declared)):
            return None
        pending.extend(declared.get('depends_on') or [])
    return {'kind':'approved_local_preparation_history','owner':dict(owner),
            'from_plan_version':state['plan_version'],'to_plan_version':goal['approved_plan_version'],
            'step_id':step_id,'unchanged_ancestry':sorted(seen),'no_execution_authority':True}


def completed_preparation_proof(runtime,state,step_id,transition):
    """Re-prove producer and source bytes; cDFT also binds its original job inputs."""
    from .presentation_transition import completed_transition_proof
    from .workflow_metadata_alignment import prepared_terminal_history_proof
    from .recovery import result_object
    node=state['nodes'][step_id];tool=node['contract']['tool']
    jobs={str(j):runtime.job_watch.get(str(j)) or {} for n in state['nodes'].values()
          for j in n.get('job_ids',[])} if runtime.job_watch else {}
    if tool=='run_cdft':
        approved={n['step_id']:n for n in runtime.main.goal_contract.to_dict().get('approved_nodes',[])}
        successors={}
        for key,n in state['nodes'].items():
            if (step_id not in n.get('contract',{}).get('depends_on',[]) or n.get('status')!='succeeded'
                or n.get('phase')!='finish' or not n.get('node_verification') or not n.get('job_ids')
                or key not in approved):continue
            if execution_contract(approved[key])!=execution_contract(n['contract']):
                from .native_resource_carry import prove_native_resource_carry
                try:
                    prove_native_resource_carry(state,key,approved[key],runtime.root,
                        runtime.main.goal_contract.to_dict(),jobs)
                except (ValueError,PermissionError,OSError,TypeError,KeyError):
                    continue
            if any((j.get('username'),j.get('conv_id'))!=(runtime.username,runtime.conv_id)
                or str(j.get('job_id'))!=str(jid) or j.get('tool')!=n['contract']['tool']
                or j.get('attempt_id')!=n.get('attempt_id') or j.get('recovery_key')!=n.get('recovery_key')
                or j.get('terminal') is not True or j.get('failed') is not False or j.get('state')!='COMPLETED'
                for jid in n['job_ids'] for j in [jobs.get(str(jid)) or {}]):continue
            successors[key]={}
        if not successors:raise ValueError('completed preparation requires its unchanged, verified original terminal calculation')
        native={**transition,'prepared_steps':[step_id],'native_resource_proofs':successors}
        proof=prepared_terminal_history_proof(runtime,state,step_id,native)
    else:
        if tool in {'analyze_gcmc_screening','analyze_cdft_screening'} and not result_object(node.get('result')).get('source_facts'):
            raise ValueError('analysis result lacks a source hash manifest')
        proof=completed_transition_proof(state,step_id,runtime.root,upstream_jobs=jobs)
    proof['transition']=copy.deepcopy(transition)
    return proof
