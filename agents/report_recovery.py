"""Read-only proofs for stopped local results; never execute or authorize tools.

The caller must independently prove that the original writer/lease stopped and
recheck this proof inside its ownership transaction. A proof is neither retry
permission nor result acceptance: normal dossier/finish validation still applies.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .workflow_patch import execution_contract

_RECEIPT_FIELDS = ('status','report_file','output_path','output_files','source_steps','source_step',
    'source_evidence_call_id','source_attempt_fingerprint','evidence_fingerprint','fragment_fingerprint','evidence_count')
_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
_SHA = re.compile(r'[0-9a-f]{64}')
_MAX_JSON = 8 * 1024 * 1024
_MAX_FILE = 64 * 1024 * 1024


def _canonical(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()


def _same(left,right):
    return _canonical(left)==_canonical(right)


def _decoded(value):
    if isinstance(value,str):
        try:return json.loads(value)
        except json.JSONDecodeError:return value
    return value


class _OwnedFiles:
    """Open every path component with NOFOLLOW, hashing bounded read chunks."""
    def __init__(self,root,owner):
        self.root=Path(os.path.abspath(root))
        if (self.root.resolve(strict=True)!=self.root or self.root.name!=owner['conv_id']
                or self.root.parent.name!=owner['username']):
            raise PermissionError('recovery root does not match its owned session scope')
        self.fd=os.open(self.root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)

    def close(self):os.close(self.fd)

    def owned(self,path):
        candidate=Path(path)
        if not candidate.is_absolute():candidate=self.root/candidate
        candidate=Path(os.path.abspath(candidate))
        if not candidate.is_relative_to(self.root) or candidate==self.root:
            raise PermissionError('recovery evidence path escapes its owned session scope')
        return candidate

    def read(self,path,*,as_json=False):
        path=self.owned(path);parts=path.relative_to(self.root).parts
        parent=os.dup(self.fd);fd=None
        try:
            for part in parts[:-1]:
                following=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
                os.close(parent);parent=following
            fd=os.open(parts[-1],os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
            before=os.fstat(fd)
            if not stat.S_ISREG(before.st_mode):raise ValueError('recovery evidence must be a regular file')
            if before.st_size>_MAX_FILE:raise ValueError('recovery evidence exceeds bounded file size')
            if as_json and before.st_size>_MAX_JSON:raise ValueError('native recovery receipt exceeds bounded JSON size')
            digest=hashlib.sha256();chunks=[];size=0
            while True:
                block=os.read(fd,65536)
                if not block:break
                size+=len(block)
                if size>_MAX_FILE:raise ValueError('recovery evidence grew beyond bounded file size')
                digest.update(block)
                if as_json:
                    if size>_MAX_JSON:raise ValueError('native recovery receipt grew beyond bounded JSON size')
                    chunks.append(block)
            after=os.fstat(fd);linked=os.stat(parts[-1],dir_fd=parent,follow_symlinks=False)
            identity=lambda s:(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
            if size!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(linked):
                raise ValueError('recovery evidence changed while being read')
            facts={'path':str(path),'size':size,'mtime_ns':after.st_mtime_ns,'ctime_ns':after.st_ctime_ns,
                   'inode':after.st_ino,'sha256':digest.hexdigest()}
            return (json.loads(b''.join(chunks)) if as_json else None),facts
        finally:
            if fd is not None:os.close(fd)
            os.close(parent)


def _context(state,step_id,session_root,approved_goal):
    owner={k:state.get(k) for k in ('username','conv_id')}
    if not all(isinstance(v,str) and v for v in owner.values()):raise PermissionError('missing recovery owner scope')
    node=state.get('nodes',{}).get(step_id)
    if not isinstance(node,dict) or not node.get('token'):raise ValueError('recovery requires the current owned attempt token')
    goal=approved_goal if approved_goal is not None else state.get('goal_contract',{})
    saved=state.get('goal_contract',{})
    protected=('version','approved_plan_version','execution_authorized','execution_mode','gases','method','parameters')
    if (not isinstance(goal,dict) or not _same({k:goal.get(k) for k in protected},{k:saved.get(k) for k in protected})
            or goal.get('execution_authorized') is not True or goal.get('execution_mode')!='workflow'
            or goal.get('approved_plan_version')!=state.get('plan_version')):
        raise ValueError('current approved goal/plan changed or lacks execution authority')
    from .workflow_execution_identity import reconciled_origin_version
    if (reconciled_origin_version(state,node.get('origin_goal_version'),goal) is None
            or node.get('execution_plan_version')!=state.get('plan_version')):
        raise ValueError('current dispatch goal/plan versions do not match approval')
    approvals=[n for n in goal.get('approved_nodes',[]) if n.get('step_id')==step_id]
    contract=node.get('contract',{})
    if len(approvals)!=1 or not _same(execution_contract(approvals[0]),execution_contract(contract)):
        raise ValueError('current contract does not match the approved node')
    if node.get('job_ids'):raise ValueError('local result recovery cannot reconcile scheduler jobs')
    files=_OwnedFiles(session_root,owner)
    return node,contract,goal,files,{'username':owner['username'],'conv_id':owner['conv_id'],'step_id':step_id,
        'goal_version':goal['version'],'plan_version':state['plan_version'],'current_token':node['token'],
        'current_contract_sha256':hashlib.sha256(_canonical(execution_contract(contract))).hexdigest(),
        'no_resubmission':True,'no_output_write':True}


def _call(files,reference,contract,expected_result):
    call_id=reference.get('call_id')
    if not isinstance(call_id,str) or not _ID.fullmatch(call_id):raise ValueError('invalid native evidence call identity')
    path=files.owned(reference.get('evidence_path',''))
    if path!=files.root/'evidence'/(call_id+'.json'):raise PermissionError('native evidence call path does not match owned identity')
    call,facts=files.read(path,as_json=True)
    if (not isinstance(call,dict) or call.get('call_id')!=call_id or call.get('tool')!=contract.get('tool')
            or call.get('failed') is not False or not _same(call.get('params'),contract.get('arguments'))
            or 'result' not in call or not _same(_decoded(call['result']),_decoded(expected_result))):
        raise ValueError('native call tool/params/result identity does not match the recorded attempt')
    return call,facts,{'call_id':call_id,'evidence_path':str(path)}


def _finish_proof(proof):
    proof['result_sha256']=hashlib.sha256(_canonical(proof['result'])).hexdigest()
    # Full current native results are returned to the ordinary dossier builder,
    # but evidence SHA + result digest bind them without embedding text twice.
    identity={k:v for k,v in proof.items() if k not in {'result','original_result'}}
    proof['proof_sha256']=hashlib.sha256(_canonical(identity)).hexdigest()
    return copy.deepcopy(proof)


def verify_archived_report_fragment(state,step_id,session_root,approved_goal=None):
    node,contract,goal,files,base=_context(state,step_id,session_root,approved_goal)
    try:
        args=contract.get('arguments',{});sources=args.get('source_steps',[])
        if (node.get('status')!='uncertain' or contract.get('tool')!='generate_scientific_report'
                or args.get('report_mode')!='fragment' or len(sources)!=1 or contract.get('depends_on')!=sources):
            raise ValueError('only an uncertain single-source native report fragment can reuse history')
        output=files.owned(args.get('output_path',''))
        if len(contract.get('expected_outputs',[]))!=1 or files.owned(contract['expected_outputs'][0])!=output:
            raise ValueError('fragment output contract must name its one immutable file')
        source=state.get('nodes',{}).get(sources[0],{})
        if source.get('status')!='succeeded':raise ValueError('current report source is not succeeded')
        for historical in reversed(node.get('attempt_history',[])):
            if historical.get('status')!='succeeded' or historical.get('phase')!='finish':continue
            if not _same(execution_contract(historical.get('contract',{})),execution_contract(contract)):continue
            receipt=_decoded(historical.get('result'))
            if not isinstance(receipt,dict):continue
            try:
                source_ref=source.get('result_ref') or {};source_call=source_ref.get('call_id')
                if (receipt.get('status')!='success' or receipt.get('source_step')!=sources[0]
                        or receipt.get('source_steps')!=sources or receipt.get('source_evidence_call_id')!=source_call
                        or not _SHA.fullmatch(str(source.get('execution_fingerprint','')))
                        or receipt.get('source_attempt_fingerprint')!=source.get('execution_fingerprint')
                        or not _SHA.fullmatch(str(receipt.get('fragment_fingerprint','')))
                        or not _SHA.fullmatch(str(receipt.get('evidence_fingerprint','')))
                        or files.owned(receipt.get('report_file',''))!=output
                        or files.owned(receipt.get('output_path',''))!=output):
                    raise ValueError('historical report receipt refers to a different current source/output identity')
                _,artifact=files.read(output)
                old_artifact=(historical.get('artifacts') or {}).get(str(output)) or {}
                if (artifact['size']<=0 or artifact['size']!=old_artifact.get('size')
                        or artifact['sha256']!=receipt['fragment_fingerprint']):
                    raise ValueError('immutable fragment hash/size differs from its historical receipt')
                if not isinstance(source_call,str) or not _ID.fullmatch(source_call):raise ValueError('invalid source call identity')
                source_path=files.owned(source_ref.get('evidence_path',''))
                if source_path!=files.root/'evidence'/(source_call+'.json'):raise PermissionError('source evidence escaped its owned call identity')
                _,source_fact=files.read(source_path)
                if source_fact['sha256']!=receipt['evidence_fingerprint']:raise ValueError('current source evidence hash changed')
                call,call_fact,reference=_call(files,historical.get('result_ref') or {},contract,historical['result'])
                result={k:receipt[k] for k in _RECEIPT_FIELDS if k in receipt}
                baseline=historical.get('output_baseline') or {}
                baseline={str(output):baseline[str(output)]} if str(output) in baseline else {}
                proof={**base,'kind':'archived_verified_report_fragment','historical_token':historical.get('token'),
                    'historical_goal_version':historical.get('origin_goal_version'),'historical_plan_version':historical.get('execution_plan_version'),
                    'historical_execution_fingerprint':historical.get('execution_fingerprint'),
                    'execution_call_id':reference['call_id'],'result':result,'result_ref':reference,
                    'started_at':historical.get('started_at',call.get('time')),'output_baseline':baseline,
                    'artifacts':{str(output):{'size':artifact['size'],'mtime_ns':artifact['mtime_ns']}},
                    'artifact_facts':{str(output):artifact},'execution_evidence':call_fact,
                    'source_identity':{'step_id':sources[0],'token':source.get('token'),'execution_fingerprint':source['execution_fingerprint'],
                        'call_id':source_call,'evidence_path':str(source_path),'evidence_sha256':source_fact['sha256']}}
                proof.update(original_result=result,original_result_ref=reference,
                    historical_output_baseline=baseline,verified_artifacts=proof['artifacts'])
                return _finish_proof(proof)
            except (ValueError,PermissionError,OSError,TypeError,KeyError):
                continue
        raise ValueError('no complete owned historical fragment proof matches the current source and approved contract')
    finally:files.close()


def verify_returned_readonly(state,step_id,session_root,approved_goal=None):
    node,contract,goal,files,base=_context(state,step_id,session_root,approved_goal)
    try:
        # Import at call time so the executor and this verifier share exactly
        # one capability classification, without a module-initialization cycle.
        from .parallel_workflow import READ_ONLY
        if (contract.get('tool') not in READ_ONLY or node.get('tool_returned') is not True
                or node.get('status') not in {'running','pending','uncertain','validation_failed'} or node.get('result') is None):
            raise ValueError('only a returned current READ_ONLY attempt can be reconciled')
        call,facts,reference=_call(files,node.get('result_ref') or {},contract,node['result'])
        result=_decoded(node['result'])
        from .recovery import failure_reason
        if failure_reason(result):raise ValueError('failed native evidence is not a successful returned result')
        observation={}
        if node.get('status')=='validation_failed':
            # Only this strong, hash-bound native observation may reopen a
            # stopped freshness failure. Generic text reads carry no full-byte
            # source/parser proof and retain the original recovery policy.
            if contract.get('tool')!='read_scientific_document':
                raise ValueError('failed readonly freshness requires a native source-document observation')
            from .local_preparation_recovery import checkpoint_call_witness
            from .parallel_workflow import _readonly_observed_artifacts
            checkpoint=node.get('result_ref',{}).get('checkpoint_path') or node.get('checkpoint_path')
            producer=checkpoint_call_witness(files,checkpoint,call,step_id=step_id,require_result_hash=True)
            reference['checkpoint_path']=str(files.owned(checkpoint))
            paths=_readonly_observed_artifacts(contract,result,files.root.parents[2],native_call=reference)
            outputs=contract.get('expected_outputs') or []
            from .output_contract import normalize_output
            expected=[normalize_output(value)for value in outputs]
            if (not expected or any(item['kind']!='file' or str(files.owned(item['path']))not in paths for item in expected)):
                raise ValueError('unchanged declared outputs lack exact native observation proof')
            observed_facts=[]
            for item in expected:
                _,fact=files.read(item['path'])
                if not fact['size']:raise ValueError('native observed source is empty')
                observed_facts.append(fact)
            observation={'native_observation_verified':True,'observed_source_facts':observed_facts,
                'producer_checkpoint_fact':producer,'no_document_reextraction':True,
                'scientific_claims_verified':False}
        return _finish_proof({**base,'kind':'current_returned_evidence','result':result,'result_ref':reference,
            'execution_call_id':reference['call_id'],'execution_evidence':facts,'started_at':node.get('started_at',call.get('time')),
            'output_baseline':node.get('output_baseline') or {},'artifacts':node.get('artifacts') or {},**observation})
    finally:files.close()
