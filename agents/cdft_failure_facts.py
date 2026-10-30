"""Bounded native failure evidence. Diagnosis is never a retry permit or a result."""
import copy
import hashlib
import json
import math
import os
from pathlib import Path

from .report_recovery import _OwnedFiles,_same,_canonical
from .recovery import RecoveryGate,input_fingerprint,result_object
from .local_preparation_recovery import checkpoint_call_witness


def inspect_failed_cdft_attempt(state,step_id,session_root,goal,jobs):
    """Authenticate one failed scheduler attempt and independently parse its files.

    No solver runs, output is modified, node is finished or retry is approved.
    Other branches need not finish before an owned failure can be diagnosed.
    """
    from .cdft_result_facts import inspect_cdft_batch_quality
    from .result_validation import cdft_collected_rows,cdft_completion_assessment
    from .workload_budget import _owned_fd
    root=Path(session_root);owner={k:state.get(k) for k in ('username','conv_id')}
    node=state.get('nodes',{}).get(step_id,{})
    contract=node.get('contract',{});args=node.get('effective_arguments') or contract.get('arguments',{})
    ids=node.get('job_ids') or []
    if (contract.get('tool')!='run_cdft' or args.get('action')!='submit'
            or node.get('status')!='failed' or not node.get('token') or not node.get('attempt_id')
            or node.get('tool_returned') is not True or node.get('dispatch_phase')!='entered_tool'
            or len(ids)!=1 or not str(ids[0]).isdigit()
            or goal.get('version')!=state.get('goal_contract',{}).get('version')):
        raise ValueError('failed cDFT facts require an owned known-returned scheduler attempt')
    job_id=str(ids[0]);job=jobs.get(job_id,{})
    if ((job.get('username'),job.get('conv_id'))!=(owner['username'],owner['conv_id'])
            or str(job.get('job_id'))!=job_id or job.get('attempt_id')!=node['attempt_id']
            or job.get('recovery_key')!=node.get('recovery_key') or job.get('tool')!='run_cdft'
            or job.get('terminal') is not True or job.get('failed') is not True
            or job.get('state') not in {'FAILED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','BOOT_FAIL','CANCELLED','PREEMPTED','DEADLINE','REVOKED'}):
        raise ValueError('failed cDFT diagnosis requires this exact owned terminal failure')
    files=_OwnedFiles(root,owner);observed=[];total=0
    def read(path,*,json_value=False):
        nonlocal total
        if len(observed)>=1024:raise ValueError('failure evidence inventory exceeds bound')
        value,fact=files.read(path,as_json=json_value);total+=fact['size']
        if total>128*1024*1024:raise ValueError('failure evidence exceeds total byte bound')
        observed.append(fact);return value,fact
    try:
        ref=node.get('result_ref') or {};call_id=ref.get('call_id')
        if not isinstance(call_id,str) or not call_id.isalnum():raise ValueError('failed native producer is absent')
        path=root/'evidence'/(call_id+'.json')
        if ref.get('evidence_path')!=str(path):raise ValueError('failed producer evidence is not owned')
        call,call_fact=read(path,json_value=True);result=result_object(call.get('result'))
        if (call.get('call_id')!=call_id or call.get('tool')!='run_cdft' or call.get('failed') is not False
                or not _same(call.get('params'),args) or not _same(result,result_object(node.get('result')))
                or str(result.get('job_id'))!=job_id):raise ValueError('failed job producer receipt differs from its actual invocation')
        witness=checkpoint_call_witness(files,ref.get('checkpoint_path'),call,step_id=step_id);observed.append(witness)
        ledger,ledger_fact=read(root/'recovery_state.json',json_value=True);entry=ledger.get(node.get('recovery_key'),{})
        if (entry.get('status')!='failed' or entry.get('tool')!='run_cdft' or entry.get('attempt_id')!=node['attempt_id']
                or set(map(str,entry.get('job_ids',[])))!={job_id} or not _same(entry.get('params'),args)
                or entry.get('fingerprint')!=input_fingerprint(args,root.parents[2])):
            raise ValueError('failed attempt ledger/input fingerprint does not match')
        work=files.owned(result.get('work_dir',''));source=files.owned(args.get('input_dir',''))
        approved=files.owned(contract.get('arguments',{}).get('job_work_dir',''))
        if not work.is_relative_to(approved) or str(work)!=job.get('work_dir'):
            raise ValueError('failed native output escaped its original owned scheduler root')
        script=files.owned(result.get('script_path') or result.get('script') or '')
        if script.parent!=work:raise ValueError('failed job script belongs to a different attempt')
        _,script_fact=read(script)
        quality=inspect_cdft_batch_quality(work,owned_root=root,gases=args.get('gases') or [args.get('gas')],source_input_dir=source)
        if quality.get('expected_materials_verified') is not True:raise ValueError('failed scientific input membership is unverified')
        names=quality['expected_materials'];inputs=quality['original_input_facts']['inputs']
        if not 0<len(names)<=512 or len(names)!=len(set(names)):raise ValueError('failed material membership is ambiguous')
        for item in inputs:
            _,fact=read(source/item['path'])
            if fact['sha256']!=item['sha256'] or fact['size']!=item['bytes']:raise ValueError('failed input changed during diagnosis')
        manifest_fact=None
        if args.get('partial_recovery_manifest'):
            manifest,manifest_fact=read(args['partial_recovery_manifest'],json_value=True)
            original=manifest.get('original',{});expected={x['path']:x for x in original.get('original_input_facts',{}).get('inputs',[])
                if Path(x['path']).stem in set(original.get('retry_materials',[]))}
            core={k:v for k,v in manifest.items() if k not in {'input_dir','recovery_id'}}
            identity=hashlib.sha256(_canonical(core)).hexdigest()
            if (manifest.get('kind')!='cdft_timeout_subset_recovery' or manifest.get('recovery_id')!=identity
                    or manifest.get('input_dir')!=str(source) or manifest.get('timeout_seconds')!=args.get('timeout')
                    or set(expected)!={x['path'] for x in inputs} or any(not _same(x,expected[x['path']]) for x in inputs)):
                raise ValueError('failed retry input does not equal its immutable manifest subset')
        batch,batch_fact=read(work/'batch_execution.json',json_value=True)
        outcomes=batch.get('outcomes',{})
        if (str(batch.get('job_id'))!=job_id or type(batch.get('exit_code')) is not int or batch['exit_code']==0
                or sorted(batch.get('expected_materials',[]))!=sorted(names)
                or set(outcomes)!={n+'.dat' for n in names}):raise ValueError('native failure batch receipt is inconsistent')
        rows=cdft_collected_rows(work,quality.get('collected_csv_sha256'))
        assessment=cdft_completion_assessment({'contract':{'arguments':{'completion_policy':'strict'}}},quality,collected_rows=rows)
        if assessment['integrity_errors']:raise ValueError('failed native CSV contains inconsistent or fabricated values')
        _,csv_fact=read(work/'results.csv')
        for material in quality.get('materials',{}).values():
            for path,expected in material.get('artifacts',{}).items():
                _,actual=read(path)
                if any(actual[key]!=expected.get(key) for key in ('sha256','size')):
                    raise ValueError('native quality input/output/log changed during diagnosis')
        cases=[]
        for name in names:
            run=work/'input_dir'/name;native,native_fact=read(run/'execution.json',json_value=True)
            if any(native.get(k)!=v for k,v in outcomes[name+'.dat'].items() if k!='manifest'):
                raise ValueError('case and batch native execution receipts disagree')
            if (str(native.get('job_id'))!=job_id or native.get('material')!=name
                    or type(native.get('exit_code')) is not int
                    or type(native.get('timed_out')) is not bool or type(native.get('cancelled')) is not bool
                    or native.get('timeout_seconds')!=args.get('timeout')
                    or any(type(native.get(k)) not in (int,float) or not math.isfinite(native[k])
                        for k in ('started_at','finished_at','wall_seconds'))
                    or native['finished_at']<native['started_at'] or native['wall_seconds']<0
                    or native['timed_out'] and native['wall_seconds']<native['timeout_seconds']):
                raise ValueError('native case exit/timeout identity is invalid')
            source_item=next(x for x in inputs if Path(x['path']).stem==name)
            for path,checksum in [(run/'input.dat',native.get('input_sha256')),
                    (run/'solver.log',native.get('stdout_sha256')),(run/'stderr.log',native.get('stderr_sha256'))]:
                _,fact=read(path)
                if fact['sha256']!=checksum or path.name=='input.dat' and checksum!=source_item['sha256']:
                    raise ValueError('native case input/log hash changed')
            for filename,fact in native.get('native_outputs',{}).items():
                if filename not in {'output.dat','hr.dat'}:raise ValueError('unexpected claimed native output')
                _,actual=read(run/filename)
                if any(actual[k]!=fact.get(k) for k in ('sha256','size')):raise ValueError('native output hash changed')
            material=quality['materials'][name]
            cases.append({'material':name,'exit_code':native['exit_code'],'timed_out':native['timed_out'],
                'cancelled':native['cancelled'],'timeout_seconds':native['timeout_seconds'],
                'output_valid':material.get('output_valid') is True,'technical_completion':material.get('technical_completion'),
                'legacy_file_presence_flag':native.get('native_outputs_complete'),
                'errors':material.get('errors',[]),'execution_receipt':native_fact,'scientific_convergence':None})
        # Files may be added after the first parser pass. Bind material sets
        # again without ever adopting unrelated new results into this attempt.
        for directory,expected,suffix in ((source,[x['path'] for x in inputs],'.dat'),
                (work/'input',[x+'.dat' for x in quality.get('input_materials',[])],'.dat'),
                (work/'input_dir',quality.get('run_materials',[]),None)):
            fd=_owned_fd(root,directory,directory=True)
            try:
                members=os.listdir(fd)
                if len(members)>8192:raise ValueError('failure directory membership exceeds bound')
                selected=[name for name in members if suffix is None or name.endswith(suffix)]
                if sorted(selected)!=sorted(expected):raise ValueError('failed material membership changed during diagnosis')
            finally:os.close(fd)
        for fact in observed:
            fd=_owned_fd(root,Path(fact['path']))
            try:
                info=os.fstat(fd);visible=os.stat(fact['path'],follow_symlinks=False)
                identity=lambda s:(s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                if identity(info)!=identity(visible) or identity(info)!=tuple(fact[k] for k in ('inode','size','mtime_ns','ctime_ns')):
                    raise ValueError('failure evidence changed after bounded inspection')
            finally:os.close(fd)
        timeouts=sum(case['timed_out'] for case in cases)
        core={'schema_version':1,'kind':'cdft_failed_attempt_facts','read_only':True,'execution_authority':False,
            'owner':owner,'step_id':step_id,'token':node['token'],'attempt_id':node['attempt_id'],'job_id':job_id,
            'work_dir':str(work),'input_fingerprint':entry['fingerprint'],'source_execution_call_id':call_id,
            'files':observed,'manifest':manifest_fact,'cases':cases,
            'counts':{'expected':len(names),'valid_final_density':sum(x['output_valid'] for x in cases),'timed_out':timeouts},
            'scientific_convergence':None,'advisor':{'automatic_retry':False,'installation_health_is_remediation':False,
                'next_action':'diagnose_solver' if timeouts else 'diagnose_native_failure',
                'requires_verified_remediation':True,'preserve_prior_jobs_and_results':True,
                'blocked_downstream':'strict completion remains required; missing density cannot be merged or ranked',
                'instruction':'Use fresh diagnosis and an independently verified concrete repair before the existing retry gate. File presence, installation success and elapsed time do not prove scientific convergence.'}}
        core['proof_sha256']=hashlib.sha256(json.dumps(core,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        return core
    finally:files.close()
