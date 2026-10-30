"""Attest identical native preparations across a recorded historical recheck."""
import hashlib
from pathlib import Path
from .report_recovery import _OwnedFiles,_same,_canonical
from .recovery import result_object
from .local_preparation_recovery import checkpoint_call_witness
from .workflow_patch import execution_contract


def prove_historical_preparation(state,step_id,root,owner,dependency):
    node=(state.get('nodes')or{}).get(step_id)or{};contract=node.get('contract')or{};current=result_object(node.get('result'))
    if ((state.get('username'),state.get('conv_id'))!=(owner.get('username'),owner.get('conv_id'))
        or node.get('status')!='succeeded' or node.get('phase')!='finish' or not node.get('token')
        or contract.get('tool')!='run_cdft' or (contract.get('arguments')or{}).get('action')!='inputs'
        or not isinstance(dependency,dict) or not isinstance(dependency.get('call_id'),str)):
        raise ValueError('current owned native preparation or dependency identity missing')
    old=[x for x in node.get('attempt_history',[]) if (x.get('result_ref')or{}).get('call_id')==dependency['call_id']]
    if len(old)!=1:raise ValueError('unique original preparation attempt required')
    old=old[0];ref=old.get('result_ref')or{}
    if (old.get('status')!='succeeded' or old.get('phase')!='finish' or not old.get('token')
        or execution_contract(old.get('contract')or{})!=execution_contract(contract)
        or any(dependency.get(k)!=ref.get(k)for k in ('call_id','evidence_path','checkpoint_path'))):
        raise ValueError('original preparation contract or dependency reference changed')
    archives=[a for a in state.get('plan_history',[])if a.get('plan_version')==old.get('execution_plan_version')]
    if len(archives)!=1:raise ValueError('unique original approved plan archive required')
    archive=archives[0];source=(archive.get('nodes')or{}).get(step_id)or{};goal=archive.get('goal_contract')or{}
    approved=[x for x in goal.get('approved_nodes',[])if x.get('step_id')==step_id]
    identity=('status','phase','token','contract','result','result_ref','execution_plan_version','origin_goal_version')
    if (len(approved)!=1 or execution_contract(approved[0])!=execution_contract(contract)
        or not _same({k:source.get(k)for k in identity},{k:old.get(k)for k in identity})):
        raise ValueError('original approved preparation archive differs')
    old_result=result_object(old.get('result'))
    for result in (old_result,current):
        manifest=result.get('input_manifest')or{}
        if (result.get('submitted') is not False or result.get('reused_existing_inputs') is not True
            or manifest.get('verification')!='fresh_recipe_byte_comparison'
            or result.get('input_dir')!=(contract.get('arguments')or{}).get('input_dir')):
            raise ValueError('preparation receipt lacks native byte comparison')
    if (not _same(old_result.get('input_manifest'),current.get('input_manifest'))
        or not _same(old_result.get('input_generation_arguments'),current.get('input_generation_arguments'))):
        raise ValueError('native preparation recipe or input manifest differs')
    files=_OwnedFiles(root,owner);facts=[]
    try:
        for attempt in (old,node):
            reference=attempt.get('result_ref')or{};cid=reference.get('call_id')
            if not isinstance(cid,str)or not cid.isalnum()or reference.get('evidence_path')!=str(files.root/'evidence'/(cid+'.json')):raise ValueError('native evidence path differs')
            call,fact=files.read(reference['evidence_path'],as_json=True);facts.append(fact)
            if (call.get('call_id')!=cid or call.get('tool')!='run_cdft' or call.get('failed') is not False
                or not _same(call.get('params'),attempt.get('effective_arguments')or contract['arguments'])
                or not _same(result_object(call.get('result')),result_object(attempt.get('result')))):
                raise ValueError('native preparation call changed')
            facts.append(checkpoint_call_witness(files,reference.get('checkpoint_path'),call,step_id=step_id,require_result_hash=True))
        manifest=current['input_manifest'];items=manifest.get('items')
        if not isinstance(items,list) or not 1<=len(items)<=512 or len(items)!=manifest.get('n_inputs'):
            raise ValueError('invalid preparation input inventory')
        seen=set()
        for item in items:
            name=item.get('path') if isinstance(item,dict) else None
            if (not isinstance(name,str) or not name or Path(name).is_absolute()
                    or '..' in Path(name).parts or name in seen):raise ValueError('unsafe preparation input identity')
            seen.add(name);_,fact=files.read(Path(current['input_dir'])/name);facts.append(fact)
            if fact['sha256']!=item.get('sha256') or fact['size']!=item.get('bytes'):
                raise ValueError('preparation input bytes changed')
        proof={'owner':owner,'step_id':step_id,'old_call':ref['call_id'],'old_token':old['token'],
            'current_call':node['result_ref']['call_id'],'current_token':node['token'],
            'input_manifest_sha256':hashlib.sha256(_canonical(current['input_manifest'])).hexdigest(),
            'facts':facts,'no_resubmission':True,'scientific_acceptance':False}
        proof['proof_sha256']=hashlib.sha256(_canonical(proof)).hexdigest()
        return proof
    finally:files.close()
