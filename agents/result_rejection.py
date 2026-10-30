"""Reject a proven terminal native contract violation without retry or acceptance.

This is an owner control operation, not a model opinion about scientific quality.
Only complete, unambiguous GCMC replication evidence is supported initially.
"""
import copy
import hashlib
import json
import time

from .state_io import json_transaction


def replication_violation(native):
    """Missing/ambiguous data, warnings, or unknown geometry are not a verdict."""
    cases=native.get('cases') or []
    allowed={'unit_cells_input_invalid','unit_cells_input_mismatch','unit_cells_native_mismatch'}
    violations=[]
    if native.get('errors') or not cases:
        raise ValueError('native rejection requires complete bounded case evidence')
    for case in cases:
        check=case.get('unit_cell_check') or {}
        expected=check.get('expected');actual=check.get('native');observed=check.get('input')
        positive=lambda value,n: isinstance(value,list) and len(value)==n and all(type(v)is int and v>0 for v in value)
        issues=check.get('issues') or []
        if (not case.get('data') or not case.get('native_input')
                or not positive(expected,3) or not positive(actual,3)
                or not isinstance(observed,list) or not 1<=len(observed)<=3
                or not all(type(v)is int and v>0 for v in observed)
                or any(item.get('code') not in allowed for item in issues)):
            raise ValueError('native rejection cannot infer a violation from missing or ambiguous evidence')
        if actual!=expected or observed!=expected:
            violations.append({'case_dir':case['case_dir'],'expected':expected,
                               'input':observed,'native':actual,'issues':copy.deepcopy(issues)})
    if not violations:
        raise ValueError('native evidence does not prove a supported result contract violation')
    return violations


def reject_workflow_result(runtime,step_id,*,expected_token,control_valid):
    """Keep an invalid attempt and pause intact; release only its terminal lease."""
    if not isinstance(step_id,str) or not step_id or not isinstance(expected_token,str) or not expected_token:
        raise ValueError('native rejection requires the exact current step and attempt token')
    if not callable(control_valid):raise ValueError('native rejection requires an owner control callback')
    main=runtime.main;state=runtime.snapshot();node=state.get('nodes',{}).get(step_id,{})
    goal=copy.deepcopy(main.goal_contract.to_dict());turn=getattr(main,'_api_turn_id',None)
    fields=('token','attempt_id','job_ids','contract','result_ref','result','status','tool_returned',
            'jobs_confirmed_terminal','dispatch_phase','started_at')
    identity={key:copy.deepcopy(node.get(key))for key in fields}
    def current(visible):
        from .parallel_workflow import _approval_digest
        now=visible.get('nodes',{}).get(step_id,{})
        future=runtime.futures.get(step_id)
        if (runtime.main is not main or main.goal_contract.to_dict()!=goal
                or getattr(main,'_api_turn_id',None)!=turn or control_valid() is not True
                or main._pending_workflow_patch or main._awaiting_plan_approval
                or (visible.get('username'),visible.get('conv_id'))!=(runtime.username,runtime.conv_id)
                or visible.get('goal_contract')!=state.get('goal_contract')
                or _approval_digest(visible.get('goal_contract',{}))!=_approval_digest(goal)
                or visible.get('plan_version')!=goal.get('approved_plan_version')
                or visible.get('plan_version')!=state.get('plan_version')
                or visible.get('user_paused')!=state.get('user_paused')
                or visible.get('user_pause_revision',0)!=state.get('user_pause_revision',0)
                or now.get('token')!=expected_token
                or any(now.get(key)!=value for key,value in identity.items())
                or future is not None and not future.done()):
            raise ValueError('native result rejection was superseded or its owned attempt changed')
    current(state)
    if node.get('status')=='validation_failed' and node.get('result_rejection'):
        proof=node['result_rejection']
        if (proof.get('token')!=expected_token or proof.get('contract')!=node.get('contract')
                or proof.get('owner')!=[runtime.username,runtime.conv_id]):
            raise ValueError('stored rejection does not belong to the current attempt')
        leases=runtime.store.snapshot().get('leases',{})
        if expected_token in leases or any((lease.get('workflow_id'),lease.get('step_id'))==
                (runtime.workflow_id,step_id)for lease in leases.values()):
            raise ValueError('previously rejected attempt has acquired another lease')
        current(runtime.snapshot())
        return _receipt(step_id,proof,already=True)
    if node.get('status')!='prefinish':raise ValueError('native rejection requires an execution-ended prefinish attempt')
    proof=runtime._gcmc_result_repair_proof(step_id,expected_token,inspection=True,rejection=True)
    current(runtime.snapshot())
    proof['violations']=replication_violation(proof['native'])
    proof['kind']='owned_native_result_rejection'
    digest=lambda value:hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
                                                allow_nan=False,separators=(',',':')).encode()).hexdigest()
    with json_transaction(runtime.store.path)as data:
        visible=data.get('workflows',{}).get(runtime.workflow_id,{})
        current(visible)
        resources=node['contract'].get('resources',{})
        from .parallel_workflow import conflicts
        leases=data.get('leases',{});own=leases.get(expected_token)
        if (not isinstance(own,dict) or (own.get('workflow_id'),own.get('step_id'))!=(runtime.workflow_id,step_id)
                or own.get('resources')!=resources):
            raise ValueError('native rejection requires this exact terminal attempt lease')
        for key,lease in leases.items():
            if key!=expected_token and ((lease.get('workflow_id'),lease.get('step_id'))==(runtime.workflow_id,step_id)
                    or any(conflicts(a,b)for a in resources for b in lease.get('resources',{}))):
                raise ValueError('native rejection may not release or cross a foreign owner lease')
        # Re-read and hash actual native bytes, producer/checkpoint and scheduler
        # authority under the state transaction; stat equality alone is not proof.
        final=runtime._gcmc_result_repair_proof(step_id,expected_token,inspection=True,rejection=True)
        final['violations']=replication_violation(final['native']);final['kind']=proof['kind']
        if final!=proof:raise ValueError('native result rejection evidence changed before commit')
        current(runtime.snapshot())
        current(visible)
        proof['proof_sha256']=digest(proof)
        proof['rejected_at']=time.time()
        target=visible['nodes'][step_id]
        target.update(status='validation_failed',phase='validation_failed',machine_output_check=False,
                      result_rejection=copy.deepcopy(proof),updated_at=time.time())
        # Preserve all original result/dossier/attempt/contract records verbatim.
        leases.pop(expected_token)
        visible.setdefault('branch_blockers',{})[step_id]={
            'kind':'native_result_rejected','reason':'Native executed replication violates the approved result contract.'}
    return _receipt(step_id,proof)


def _receipt(step_id,proof,already=False):
    return {'ok':True,'step_id':step_id,'status':'validation_failed','rejected':True,
            'already_rejected':already,'lease_released':True,'no_resubmission':True,
            'retry_granted':False,'scientific_acceptance':False,'proof_sha256':proof['proof_sha256'],
            'job_ids':copy.deepcopy(proof['job_ids']),
            'next_action':'Approve a separate replacement node and new directory; no execution was scheduled.'}
