"""Revalidate a sandbox-derived scalar against its actual execution receipt."""
from pathlib import Path

def proven_derived_arguments(state, step_id, current_goal):
    from .config import get_config
    from .registry import get_registry
    from .analysis_artifact_binding import resolve_analysis_input_bindings
    from .goal_contract import GoalContract
    from .parallel_workflow import _approval_digest
    from .workflow_execution_identity import reconciled_origin_version
    from .report_recovery import _OwnedFiles, _same
    from .local_preparation_recovery import checkpoint_call_witness
    from .recovery import result_object
    owner={key:state.get(key) for key in ('username','conv_id')}
    if any(not isinstance(v,str) or not v or Path(v).name!=v or v in {'.','..'} for v in owner.values()):
        raise ValueError('invalid owner')
    root=Path(get_config().project_root).absolute()/'runs'/owner['username']/owner['conv_id']
    node=(state.get('nodes') or {}).get(step_id) or {};contract=node.get('contract') or {}
    if (node.get('status')!='succeeded' or node.get('phase')!='finish' or not node.get('token')
            or node.get('tool_returned') is not True or node.get('job_ids')
            or not node.get('analysis_input_proofs') or not contract.get('input_bindings')):
        raise ValueError('derived execution identity incomplete')
    files=_OwnedFiles(root,owner)
    try:
        goal=current_goal if isinstance(current_goal,dict) else {}
        if (goal.get('execution_authorized') is not True or goal.get('approved_plan_version')!=state.get('plan_version')
                or _approval_digest(goal)!=_approval_digest(state.get('goal_contract') or {})
                or reconciled_origin_version(state,node.get('origin_goal_version'),goal)!=goal.get('version')):
            raise ValueError('current approved goal changed')
        resolved,proofs=resolve_analysis_input_bindings(root,state,contract,contract.get('arguments') or {},get_registry(),GoalContract.from_dict(goal))
        if not proofs or not _same(proofs,node['analysis_input_proofs']) or not _same(resolved,node.get('effective_arguments')):
            raise ValueError('derived proof or effective arguments changed')
        ref=node.get('result_ref') or {};cid=ref.get('call_id')
        if not isinstance(cid,str) or not cid.isalnum() or ref.get('evidence_path')!=str(root/'evidence'/(cid+'.json')):
            raise ValueError('consumer evidence identity changed')
        completed=[event for event in (state.get('outbox') or {}).values()
                   if event.get('kind')=='worker_completed'
                   and (event.get('payload') or {}).get('step_id')==step_id
                   and (event.get('payload') or {}).get('correlation_token')==node['token']
                   and _same((event.get('payload') or {}).get('result_ref'),ref)]
        if len(completed)!=1:
            raise ValueError('consumer token is not tied to its native completed receipt')
        call,_=files.read(ref['evidence_path'],as_json=True)
        if (call.get('call_id')!=cid or call.get('tool')!=contract.get('tool') or call.get('failed') is not False
                or not _same(call.get('params'),resolved) or not _same(result_object(call.get('result')),result_object(node.get('result')))):
            raise ValueError('consumer result receipt changed')
        checkpoint_call_witness(files,ref.get('checkpoint_path'),call,step_id=step_id,require_result_hash=True)
        return resolved
    finally:
        files.close()
