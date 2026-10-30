"""Fresh native result facts without workflow, contract, or ledger mutations."""
import copy
import hashlib
import json


def native_inspection_kind(contract):
    """One supported native-method table for inspection, provenance and finish.

    This selects a fact type only. Owner, execution, bytes and scientific
    acceptance still require their original independent native proofs.
    """
    from .gcmc_results import TOOLS
    if not isinstance(contract, dict):
        return None
    tool = contract.get('tool')
    if not isinstance(tool,str):
        return None
    if tool in TOOLS:
        return 'gcmc_native'
    if (tool == 'run_cdft' and isinstance(contract.get('arguments'),dict)
            and contract['arguments'].get('action') == 'submit'):
        return 'cdft_native'
    return None


def inspect_workflow_result(runtime, step_id, *, expected_token=None, control_valid=None):
    """The caller binds this to its owned Session and records the returned evidence.

    Native GCMC results and existing typed cDFT submit dossiers are supported.
    Other capabilities fail explicitly, without inspecting arbitrary paths.
    """
    from .gcmc_results import TOOLS, proof_files_unchanged
    from .recovery import RecoveryGate

    if not isinstance(step_id, str) or not step_id or len(step_id) > 256:
        raise ValueError('result inspection requires one owned workflow step')
    main = runtime.main
    state = runtime.snapshot()
    node = state.get('nodes', {}).get(step_id)
    if not isinstance(node, dict):
        raise ValueError('result inspection step does not exist in this workflow')
    if (state.get('username'), state.get('conv_id')) != (runtime.username, runtime.conv_id):
        raise PermissionError('result inspection belongs to a different Session owner')
    native_kind = native_inspection_kind(node.get('contract'))
    is_cdft = native_kind == 'cdft_native'
    if native_kind is None:
        return {'status': 'unsupported', 'read_only': True, 'executed': False,
                'step_id': step_id, 'error': 'native result inspection is not implemented for this tool/action',
                'supported_capabilities': {'run_gcmc_isotherm':['native_results'],'run_gcmc_batch':['native_results'],
                                           'run_cdft':['submit:existing_typed_dossier']},
                'available_read_only_tools':['read_file','get_tool_schema']}
    token = node.get('token')
    goal = copy.deepcopy(main.goal_contract.to_dict())
    turn = getattr(main, '_api_turn_id', None)
    identity_fields = ('token', 'attempt_id', 'job_ids', 'contract', 'result_ref', 'result',
                       'status', 'jobs_confirmed_terminal', 'tool_returned', 'dispatch_phase', 'started_at')
    identity = {key: copy.deepcopy(node.get(key)) for key in identity_fields}
    ledger = RecoveryGate(path=runtime.root / 'recovery_state.json')
    ledger_fields = ('attempt_id', 'tool', 'params', 'fingerprint', 'job_ids', 'status', 'last_result')
    watch_fields = ('username', 'conv_id', 'tool', 'attempt_id', 'recovery_key',
                    'terminal', 'failed', 'state', 'work_dir')
    def native_authority():
        entry = ledger.snapshot().get(node.get('recovery_key'), {})
        jobs = {str(job): runtime.job_watch.get(str(job)) or {} for job in node.get('job_ids', [])} if runtime.job_watch else {}
        return {'ledger': {key: copy.deepcopy(entry.get(key)) for key in ledger_fields},
                'jobs': {job: {key: copy.deepcopy(item.get(key)) for key in watch_fields}
                         for job, item in jobs.items()}}
    native_binding = native_authority()

    def current():
        # Scheduler/ledger reads precede the final current-turn and runtime CAS.
        # No counters or unrelated observer timestamps are part of this identity.
        native_current = native_authority()
        visible = runtime.snapshot()
        current_node = visible.get('nodes', {}).get(step_id, {})
        if (runtime.main is not main or main.goal_contract.to_dict() != goal
                or getattr(main, '_api_turn_id', None) != turn
                or main._awaiting_plan_approval or main._pending_workflow_patch
                or (expected_token is not None and expected_token != token)
                or (control_valid is not None and (not callable(control_valid) or control_valid() is not True))
                or (visible.get('username'), visible.get('conv_id')) != (runtime.username, runtime.conv_id)
                or visible.get('plan_version') != state.get('plan_version')
                or visible.get('goal_contract') != state.get('goal_contract')
                or native_current != native_binding
                or any(current_node.get(key) != value for key, value in identity.items())):
            raise ValueError('owned result inspection was superseded or its native attempt changed')

    current()
    if is_cdft:
        return _inspect_cdft_dossier(runtime,state,node,goal,step_id,current)
    proof = runtime._gcmc_result_repair_proof(step_id, token, inspection=True)
    current()
    if not proof_files_unchanged(proof, runtime.root):
        raise ValueError('native result evidence changed after inspection')
    current()  # stat/read operations may release the GIL; recheck after them.
    digest = hashlib.sha256(json.dumps(proof, sort_keys=True, ensure_ascii=False,
                                      allow_nan=False).encode()).hexdigest()
    return {'status': 'inspected', 'read_only': True, 'no_resubmission': True,
            'workflow_id': runtime.workflow_id, 'owner': [runtime.username, runtime.conv_id],
            'step_id': step_id, 'token': token, 'attempt_id': node.get('attempt_id'),
            'job_ids': copy.deepcopy(proof['job_ids']), 'plan_version': state['plan_version'],
            'goal_version': goal['version'], 'contract': copy.deepcopy(node['contract']),
            'source_execution_call_id': proof['execution_call_id'],
            'source_execution_sha256': proof['execution_evidence_sha256'],
            'provenance_sha256': digest, 'method_facts': {'gcmc_native': proof['native']},
            'scientific_acceptance_unchanged': True,
            'statistical_convergence_established': False}


def _inspect_cdft_dossier(runtime,state,node,goal,step_id,current):
    """Reuse typed native facts only while their actual bounded bytes still match."""
    import os
    from pathlib import Path
    from .cdft_partial_recovery import prove_completed_attempt
    from .report_recovery import _OwnedFiles
    from .workload_budget import _owned_fd
    jobs={str(job):runtime.job_watch.get(str(job)) or {} for job in node.get('job_ids',[])} if runtime.job_watch else {}
    attempt=prove_completed_attempt(state,step_id,runtime.root,goal,jobs,inspection=True)
    compact=node.get('validation_dossier') or {};call_id=compact.get('evidence_call_id')
    if not isinstance(call_id,str) or not call_id.isalnum():
        return {'status':'unavailable','read_only':True,'executed':False,'step_id':step_id,
                'error':'This attempt has no existing typed native dossier; no result was inferred.',
                'available_read_only_tools':['read_file']}
    owner={'username':runtime.username,'conv_id':runtime.conv_id}
    files=_OwnedFiles(runtime.root,owner)
    try:
        path=runtime.root/'evidence'/(call_id+'.json')
        if compact.get('evidence_path')!=str(path):raise ValueError('unowned native cDFT dossier')
        record,dossier_fact=files.read(path,as_json=True);full=record.get('result')
        if (record.get('call_id')!=call_id or record.get('failed') is not False
                or record.get('tool')!='inspect_workflow_result' or not isinstance(full,dict)
                or any(full.get(key)!=value for key,value in {'workflow_id':runtime.workflow_id,
                    'step_id':step_id,'contract':node['contract'],'job_ids':node['job_ids'],
                    'attempt_id':node['attempt_id']}.items())):
            raise ValueError('native cDFT dossier does not belong to this attempt')
        for key,value in compact.items():
            if key=='inventory' and isinstance(value,dict):
                if any(full.get(key,{}).get(k)!=v for k,v in value.items()):raise ValueError('cDFT dossier compact inventory changed')
            elif full.get(key)!=value:raise ValueError('cDFT dossier compact facts changed')
        quality=full.get('method_facts',{}).get('cdft_native')
        if not isinstance(quality,dict) or quality.get('expected_materials_verified') is not True:
            return {'status':'unavailable','read_only':True,'executed':False,'step_id':step_id,
                    'error':'No verified typed cDFT material inventory is available; convergence remains unknown.',
                    'native_dossier_path':str(path),'available_read_only_tools':['read_file']}
        work=Path(attempt['work_dir']);source=Path(attempt['input_dir']);observed=[dossier_fact];total=dossier_fact['size']
        def read_expected(name,expected):
            nonlocal total
            if len(observed)>=1024:raise ValueError('cDFT inspection file inventory exceeds bound')
            if type(expected.get('size')) is not int or not 0<=expected['size']<=128*1024*1024-total:
                raise ValueError('cDFT inspection exceeds bounded evidence budget')
            _,fact=files.read(name)
            if any(fact.get(k)!=expected.get(k) for k in ('sha256','size')):raise ValueError('native cDFT source bytes changed')
            total+=fact['size'];observed.append(fact)
        sources=quality.get('original_input_facts',{}).get('inputs') or []
        names=quality.get('expected_materials') or []
        if not sources or not 0<len(names)<=512 or len(names)!=len(set(names)):
            raise ValueError('native cDFT source membership is unavailable or ambiguous')
        absent=[]
        for item in sources:
            name=item.get('path')
            if not isinstance(name,str) or Path(name).name!=name:raise ValueError('invalid native input member')
            read_expected(source/name,{'sha256':item.get('sha256'),'size':item.get('bytes')})
        for material in names:
            if not isinstance(material,str) or material in {'.','..'} or Path(material).name!=material:raise ValueError('invalid material identity')
            facts=quality.get('materials',{}).get(material,{}).get('artifacts',{})
            for name,fact in facts.items():read_expected(name,fact)
            for filename in ('input.dat','output.dat','hr.dat','solver.log','stderr.log','execution.json'):
                missing=work/'input_dir'/material/filename
                if str(missing) not in facts:
                    try:
                        fd=_owned_fd(runtime.root,missing);os.close(fd)
                    except FileNotFoundError:absent.append(str(missing))
                    else:raise ValueError('previously absent native cDFT evidence appeared')
        if quality.get('collected_csv_sha256'):
            csvpath=work/'results.csv';_,fact=files.read(csvpath)
            if fact['sha256']!=quality['collected_csv_sha256']:raise ValueError('collected cDFT CSV changed')
            total+=fact['size'];observed.append(fact)
            if total>128*1024*1024:raise ValueError('cDFT inspection exceeds bounded evidence budget')
        else:raise ValueError('typed cDFT dossier has no collected CSV proof')
        def membership():
            values={}
            for directory in (source,work/'input',work/'input_dir'):
                fd=_owned_fd(runtime.root,directory,directory=True)
                try:
                    entries=os.listdir(fd)
                    if len(entries)>8192:raise ValueError('cDFT source membership exceeds bound')
                    values[str(directory)]=sorted(entries)
                finally:os.close(fd)
            return values
        members=membership()
        if (sorted(n[:-4] for n in members[str(source)] if n.endswith('.dat'))!=sorted(names)
                or sorted(n[:-4] for n in members[str(work/'input')] if n.endswith('.dat'))!=sorted(quality.get('input_materials',[]))
                or members[str(work/'input_dir')]!=sorted(quality.get('run_materials',[]))):
            raise ValueError('native cDFT material set changed')
        current()
        for fact in observed:
            fd=_owned_fd(runtime.root,Path(fact['path']))
            try:
                stat=os.fstat(fd);linked=os.stat(fact['path'],follow_symlinks=False)
                keys=('st_ino','st_size','st_mtime_ns','st_ctime_ns')
                actual=tuple(getattr(stat,k) for k in keys)
                if actual!=tuple(getattr(linked,k) for k in keys) or actual!=tuple(fact[k] for k in ('inode','size','mtime_ns','ctime_ns')):
                    raise ValueError('native cDFT evidence changed after verification')
            finally:os.close(fd)
        for name in absent:
            try:fd=_owned_fd(runtime.root,Path(name))
            except FileNotFoundError:continue
            else:os.close(fd);raise ValueError('native cDFT missing-evidence identity changed')
        if membership()!=members:raise ValueError('native cDFT input membership changed after inspection')
        current()
        provenance={'attempt':attempt,'dossier':dossier_fact,'native_files':observed,'absent':absent,'membership':members}
        return {'status':'inspected','read_only':True,'no_resubmission':True,
            'workflow_id':runtime.workflow_id,'owner':[runtime.username,runtime.conv_id],
            'step_id':step_id,'token':node['token'],'attempt_id':node['attempt_id'],'job_ids':node['job_ids'],
            'plan_version':state['plan_version'],'goal_version':goal['version'],'contract':copy.deepcopy(node['contract']),
            'source_execution_call_id':attempt['execution_call_id'],
            'source_execution_sha256':attempt['execution_evidence']['sha256'],
            'provenance_sha256':hashlib.sha256(json.dumps(provenance,sort_keys=True,allow_nan=False).encode()).hexdigest(),
            'method_facts':copy.deepcopy(full['method_facts']),'scientific_acceptance_unchanged':True,
            'scientific_convergence_established':quality.get('scientific_convergence') is True}
    finally:files.close()
