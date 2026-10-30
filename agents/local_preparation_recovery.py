"""Read-only proof for bounded reexecution of a known-returned preparation.

This is not result acceptance or a scheduler retry permit. Only the declared
native preparation capability can return to normal pending/claim execution.
"""
import copy
import hashlib
import math
import re
import time

from .recovery import result_object, failure_reason
from .report_recovery import _OwnedFiles, _same, _canonical
from .workflow_patch import execution_contract


def preparation_capability(contract):
    return (contract.get('tool') == 'run_cdft'
            and contract.get('arguments', {}).get('action') == 'inputs')


def checkpoint_call_witness(files, checkpoint_path, call, *, step_id=None, require_result_hash=False):
    """Verify a real compact producer log, without demanding erased results."""
    checkpoint,fact=files.read(checkpoint_path,as_json=True)
    owner=checkpoint.get('owner',{})
    if (owner.get('username')!=files.root.parent.name or owner.get('conv_id')!=files.root.name
            or step_id is not None and owner.get('step_id')!=step_id
            or checkpoint.get('checkpoint_at',0)<call['time']):
        raise ValueError('native evidence producer checkpoint owner/time does not match')
    matches=[item for item in checkpoint.get('memory',{}).get('tool_call_log',[])
             if isinstance(item,dict) and item.get('call_id')==call['call_id']]
    if len(matches)!=1:raise ValueError('native tool call is absent or ambiguous in its producer checkpoint')
    recorded=matches[0]
    fields=('call_id','time','agent','tool','params','failed')
    reference=recorded.get('evidence_path') or (recorded.get('result_ref') or {}).get('evidence_path')
    if (not _same({key:recorded.get(key) for key in fields},{key:call.get(key) for key in fields})
            or reference!=str(files.root/'evidence'/(call['call_id']+'.json'))
            or recorded.get('result_preview')!=str(call.get('result',''))[:500]
            or 'result' in recorded and not _same(recorded['result'],call.get('result'))):
        raise ValueError('native tool call metadata/result reference differs from its producer checkpoint')
    digest=hashlib.sha256(str(call.get('result','')).encode('utf-8')).hexdigest()
    if require_result_hash and recorded.get('result_sha256')!=digest:
        raise ValueError('verification checkpoint lacks the trusted native result digest; inspect again')
    return fact


def prove_returned_preparation(state, step_id, session_root, goal, expected_token,
                               failed_call_id, verification_call_id, verification_record):
    """Bind the stopped native process, current approval and fresh inspection."""
    from .installations import executor_fingerprint, installation_profile
    owner={key:state.get(key) for key in ('username','conv_id')}
    current=state.get('nodes',{}).get(step_id,{})
    contract=current.get('contract',{})
    saved=state.get('goal_contract',{})
    protected=('version','method','gases','parameters','execution_mode','execution_authorized','approved_plan_version')
    approvals=[node for node in goal.get('approved_nodes',[]) if node.get('step_id')==step_id]
    if (state.get('user_paused') or state.get('pause_kind') not in {None,'failure','reconciliation'}
            or state.get('status') not in {'active','needs_user'}
            or goal.get('execution_authorized') is not True
            or not _same({key:saved.get(key) for key in protected},{key:goal.get(key) for key in protected})
            or state.get('plan_version')!=goal.get('approved_plan_version')
            or len(approvals)!=1 or not _same(execution_contract(approvals[0]),execution_contract(contract))
            or not expected_token or current.get('token')!=expected_token
            or current.get('status')!='failed' or current.get('dispatch_phase')!='entered_tool'
            or current.get('tool_returned') is not True or not preparation_capability(contract)
            or current.get('job_ids') or current.get('attempt_id') or current.get('recovery_key')
            or current.get('verified_retry') or current.get('configuration_recovery')
            or current.get('execution_plan_version')!=state.get('plan_version')
            or current.get('origin_goal_version')!=saved.get('version')):
        raise ValueError('local preparation recovery requires exact approved, unpaused, known-returned failed ownership')
    files=_OwnedFiles(session_root,owner)
    try:
        def read_call(call_id):
            if not isinstance(call_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',call_id):
                raise ValueError('native evidence call identity required')
            path=files.root/'evidence'/(call_id+'.json')
            call,fact=files.read(path,as_json=True)
            timestamp=call.get('time') if isinstance(call,dict) else None
            if (not isinstance(call,dict) or call.get('call_id')!=call_id
                    or not isinstance(timestamp,(int,float)) or isinstance(timestamp,bool)
                    or not math.isfinite(timestamp) or timestamp<=0 or timestamp>time.time()+1):
                raise ValueError('native evidence identity/time is invalid')
            return call,fact
        reference=current.get('result_ref') or {}
        if (reference.get('call_id')!=failed_call_id
                or reference.get('evidence_path')!=str(files.root/'evidence'/(failed_call_id+'.json'))):
            raise ValueError('failure evidence does not identify the current returned attempt')
        failed,failed_fact=read_call(failed_call_id)
        if failed['time']<current.get('started_at',float('inf')):
            raise ValueError('native failed call predates this dispatch')
        failed_checkpoint=checkpoint_call_witness(files,reference.get('checkpoint_path',''),failed,step_id=step_id)
        effective=current.get('effective_arguments') or contract['arguments']
        # Preparation has no scheduler allocation to justify changing science.
        if not _same(effective,contract['arguments']):
            raise ValueError('local preparation effective arguments differ from its approved contract')
        native=result_object(failed.get('result'))
        code=native.get('exit_code')
        if (failed.get('tool')!=contract['tool'] or failed.get('failed') is not True
                or not _same(failed.get('params'),effective)
                or not _same(native,result_object(current.get('result')))
                or not failure_reason(native) or native.get('submitted') is not False
                or not isinstance(code,int) or isinstance(code,bool) or not 0<code<=255
                or native.get('job_id') or native.get('job_ids')):
            raise ValueError('native failed child must have a known normal exit without any scheduler job')
        verified,verified_fact=read_call(verification_call_id)
        fields=('call_id','time','agent','tool','params','failed')
        digest=hashlib.sha256(str(verified.get('result','')).encode('utf-8')).hexdigest()
        if (not isinstance(verification_record,dict)
                or not _same({key:verification_record.get(key) for key in fields},{key:verified.get(key) for key in fields})
                or verification_record.get('evidence_path')!=str(files.root/'evidence'/(verification_call_id+'.json'))
                or verification_record.get('result_preview')!=str(verified.get('result',''))[:500]
                or verification_record.get('result_sha256')!=digest
                or 'result' in verification_record and not _same(verification_record['result'],verified.get('result'))):
            raise ValueError('fresh verification must bind the actual Session record_tool_call result digest')
        verified_checkpoint=checkpoint_call_witness(files,files.root/'session_checkpoint.json',verified,require_result_hash=True)
        observation=result_object(verified.get('result'))
        actual=executor_fingerprint(contract['tool'],files.root.parents[2])
        if (failed_call_id==verification_call_id or verified.get('time',0)<=failed['time']
                or verified.get('tool')!='inspect_runtime' or verified.get('failed') is not False
                or not _same(verified.get('params'),{'tool_name':contract['tool'],'arguments':effective})
                or observation.get('ok') is not True or observation.get('read_only') is not True
                or observation.get('verified_tool')!=contract['tool']
                or not _same(observation.get('checked_arguments'),effective)
                or failure_reason(observation) or observation.get('runtime_fingerprint')!=actual
                or installation_profile(contract['tool'],observation.get('checked_arguments') or {})
                    !=installation_profile(contract['tool'],effective)):
            raise ValueError('fresh owned native installation inspection must verify the exact preparation and current sources')
        return {'schema_version':1,'kind':'verified_returned_local_preparation','owner':owner,
            'step_id':step_id,'expected_token':expected_token,'goal_version':goal['version'],
            'plan_version':state['plan_version'],'approved_contract':execution_contract(contract),
            'failed_call_id':failed_call_id,'failure_evidence_sha256':failed_fact['sha256'],
            'failure_checkpoint_sha256':failed_checkpoint['sha256'],
            'verification_call_id':verification_call_id,'verification_evidence_sha256':verified_fact['sha256'],
            'verification_checkpoint_sha256':verified_checkpoint['sha256'],
            'verification_record_sha256':hashlib.sha256(_canonical({key:verification_record.get(key) for key in (*fields,'result_sha256')})).hexdigest(),
            'executor_fingerprint':actual,'known_native_child_exited':True,
            'approved_goal_sha256':hashlib.sha256(_canonical(goal)).hexdigest(),
            'no_result_acceptance':True,'no_scientific_dispatch':True}
    finally:files.close()
