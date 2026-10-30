"""Proof-only recovery of one unchanged, previously accepted downloaded file."""
import copy,hashlib
from pathlib import Path
from .report_recovery import _OwnedFiles,_canonical,_same
from .local_preparation_recovery import checkpoint_call_witness
from .recovery import result_object
from .workflow_patch import execution_contract


def prove_archived_download(state,step_id,root,goal,*,owner):
    if ((state.get('username'),state.get('conv_id'))!=(owner.get('username'),owner.get('conv_id'))
        or state.get('user_paused') or state.get('status') not in {'active','needs_user'}
        or goal.get('execution_authorized') is not True or state.get('plan_version')!=goal.get('approved_plan_version')):
        raise ValueError('current owner, plan or authorization changed')
    node=(state.get('nodes')or{}).get(step_id)or{};contract=node.get('contract')or{};args=contract.get('arguments')or{}
    approved=[n for n in goal.get('approved_nodes',[])if n.get('step_id')==step_id]
    result=result_object(node.get('result'))
    if (len(approved)!=1 or execution_contract(approved[0])!=execution_contract(contract)
        or contract.get('tool')!='download_scientific_file' or node.get('status')!='failed'
        or node.get('dispatch_phase')!='entered_tool' or node.get('tool_returned') is not True
        or not node.get('token') or node.get('job_ids') or node.get('attempt_id') or node.get('recovery_key')
        or result.get('error_kind')!='scientific_download_failed' or result.get('output_published') is not False
        or result.get('existing_results_overwritten') is not False):
        raise ValueError('current failed attempt is not a stopped no-overwrite download')
    candidates=[x for x in node.get('attempt_history',[])if x.get('status')=='succeeded' and x.get('phase')=='finish'
        and (x.get('contract')or{}).get('tool')=='download_scientific_file'
        and execution_contract(x['contract'])==execution_contract(contract)]
    if len(candidates)!=1:raise ValueError('unique original accepted download is required')
    old=candidates[0];archives=[a for a in state.get('plan_history',[])if a.get('plan_version')==old.get('execution_plan_version')]
    if len(archives)!=1:raise ValueError('unique original approved plan archive is required')
    archived=archives[0];saved=(archived.get('nodes')or{}).get(step_id)or{};og=archived.get('goal_contract')or{}
    original=[n for n in og.get('approved_nodes',[])if n.get('step_id')==step_id]
    identity=('status','phase','token','contract','result','result_ref','artifacts','execution_plan_version','origin_goal_version')
    if (len(original)!=1 or execution_contract(original[0])!=execution_contract(contract)
        or not _same({k:saved.get(k)for k in identity},{k:old.get(k)for k in identity})):
        raise ValueError('original approved node archive identity differs')
    files=_OwnedFiles(root,owner);facts=[]
    try:
        for attempt in (old,node):
            ref=attempt.get('result_ref')or{};cid=ref.get('call_id')
            if not isinstance(cid,str)or not cid.isalnum()or ref.get('evidence_path')!=str(files.root/'evidence'/(cid+'.json')):raise ValueError('evidence identity differs')
            call,fact=files.read(ref['evidence_path'],as_json=True);facts.append(fact)
            if (call.get('call_id')!=cid or call.get('tool')!='download_scientific_file'
                or call.get('failed') is not (attempt is node)
                or not _same(call.get('params'),attempt.get('effective_arguments')or args)
                or not _same(result_object(call.get('result')),result_object(attempt.get('result')))):
                raise ValueError('native call or parameters differ')
            facts.append(checkpoint_call_witness(files,ref.get('checkpoint_path'),call,step_id=step_id,require_result_hash=True))
        original_result=result_object(old.get('result'));path=args.get('output_path')
        if (original_result.get('status')!='success' or original_result.get('output_path')!=path
            or original_result.get('output_files')!=[path] or original_result.get('existing_results_overwritten') is not False
            or type(original_result.get('bytes')) is not int or original_result['bytes']<=0):raise ValueError('original download receipt is incomplete')
        _,fact=files.read(path);facts.append(fact)
        if fact['sha256']!=original_result.get('sha256') or fact['size']!=original_result['bytes']:
            raise ValueError('original downloaded file bytes changed')
        identity={'owner':owner,'step_id':step_id,'current_token':node['token'],'original_token':old['token'],
            'current_plan':state['plan_version'],'original_plan':old['execution_plan_version'],
            'original_call':old['result_ref']['call_id'],'failed_call':node['result_ref']['call_id'],
            'facts':facts,'no_download':True,'no_file_write':True,'scientific_acceptance':False}
        identity['proof_sha256']=hashlib.sha256(_canonical(identity)).hexdigest()
        return identity
    finally:files.close()
