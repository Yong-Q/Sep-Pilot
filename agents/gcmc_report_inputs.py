"""Compact native GCMC observations with original-byte custody across stages.

RASPA .data stays outside model context and plot mounts. Every inherited view
reopens its succeeded producer, immutable validation dossier and original native
inputs/results. Numeric report/plot data comes only from the public subtree.
"""
import hashlib
import json
import os
from pathlib import Path
import stat

SCHEMA='native_gcmc_report_view_v1'
ROLE='native_gcmc_result'
STREAM_VERIFICATION='native_gcmc_source_stream_sha256'
DIRECTORY_VERIFICATION='native_gcmc_directory_identity_v1'
MAX_FILE=128*1024**2
MAX_TOTAL=2*1024**3


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode()).hexdigest()


def implementation_sha256():
    from . import gcmc_results
    return digest({name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
        for name,module in [('projection',__import__(__name__,fromlist=['*'])),('parser',gcmc_results)]})


def stream_fact(root,path,budget):
    from .native_source_stream import stream_owned_source
    stream={'source_files':budget.get('gcmc_files',0),'source_bytes':budget.get('gcmc_bytes',0)}
    _,fact=stream_owned_source(root,path,stream,max_file_bytes=MAX_FILE,max_total_bytes=MAX_TOTAL,
        verification=STREAM_VERIFICATION)
    budget['gcmc_files']=stream['source_files'];budget['gcmc_bytes']=stream['source_bytes']
    return fact


def directory_fact(root,path):
    from .report_authoring import _relative
    from .workload_budget import _owned_fd
    relative=_relative(root,path);fd=_owned_fd(Path(root),Path(root)/relative,directory=True)
    try:
        info=os.fstat(fd)
        with os.scandir(fd)as iterator:
            entries=[]
            for index,item in enumerate(iterator):
                if index>=4096:raise ValueError('native GCMC output membership exceeds bound')
                entries.append(item.name)
        after=os.fstat(fd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if identity(info)!=identity(after):raise ValueError('native GCMC output membership changed')
        return {'path':str(relative),'size':info.st_size,'sha256':digest(sorted(entries)),
            'device':info.st_dev,'inode':info.st_ino,'mtime_ns':info.st_mtime_ns,'ctime_ns':info.st_ctime_ns,
            'verification':DIRECTORY_VERIFICATION}
    finally:os.close(fd)


def _file_map(native):
    result={}
    for case in native.get('cases',[]):
        for name in ('data','native_input'):
            if isinstance(case.get(name),dict):result[case[name]['path']]=case[name]
        geometry=(case.get('unit_cell_check')or{}).get('geometry_proof')or{}
        for name in ('source','staged_source','case_source'):
            if isinstance(geometry.get(name),dict):result[geometry[name]['path']]=geometry[name]
    return result


def _public(native,work):
    cases=[]
    for case in native['cases']:
        row={'case':str(Path(case['case_dir']).relative_to(work)),'status':'valid'if case['ok']else'failed',
            'gas':case.get('gas'),'temperature_K':case.get('temperature_K'),'pressure_Pa':case.get('pressure_Pa'),
            'production_cycles':case.get('production_cycles'),'initialization_cycles':case.get('initialization_cycles'),
            'absolute_loading':case.get('loadings')or{},'loading_uncertainty':None,
            'uncertainty_scope':'Native parser does not attest a standard error; no uncertainty or independent sample count inferred.',
            'scientific_convergence':'unknown','statistical_convergence_established':False}
        blocks=case.get('statistical_blocks')or{}
        row['statistical_blocks']={'distinct_index_count':blocks.get('unique_index_count'),
            'scan_complete':blocks.get('scan_complete')is True,'is_independent_sample_count':False,
            'establishes_statistical_convergence':False}
        cells=case.get('unit_cell_check')or{}
        row['unit_cells']={'verified':cells.get('verified')is True,'input':cells.get('input'),
            'native':cells.get('native'),'geometry_proof_verified':cells.get('geometry_proof_verified')is True,
            'cutoff_warning_count':(cells.get('native_geometry')or{}).get('cutoff_warning_count'),
            'cutoff_safety_scope':'Lattice/replication checks do not establish sampling convergence or force-field validity.'}
        if not case['ok']:row['failure']={'error_kind':case.get('error_kind'),'original_diagnostic_retained':True}
        cases.append(row)
    return {'method':'GCMC','quantity':'absolute loading','native_loading_units_preserved':True,
        'cases':cases,'counts':native['counts'],'scientific_convergence':'unknown',
        'statistical_convergence_established':False,
        'limitations':['Native statistical block index count is not an independent sample count or convergence proof.',
            'Short-run finite loading is an engineering observation, not a converged isotherm or validated force field.',
            'Native units are retained; no conversion between mol/kg-framework and cDFT mol/L is performed.']}


def build_view(root,owner,state,source,*,read,budget,file_facts,attest):
    from .report_authoring import _result,_relative
    from .gcmc_results import TOOLS,inspect_gcmc_results
    from .workflow_patch import execution_contract
    root=Path(root);contract=source.get('contract')or{};ref=source.get('result_ref')or{}
    if ((state.get('username'),state.get('conv_id'))!=(owner['username'],owner['conv_id'])
            or source.get('status')!='succeeded' or contract.get('tool')not in TOOLS
            or not source.get('token') or not callable(attest)):
        raise ValueError('native GCMC view requires an owned succeeded producer')
    cid=ref.get('call_id');path=root/'evidence'/(str(cid)+'.json')
    if ref.get('evidence_path')!=str(path):raise ValueError('native GCMC producer evidence identity differs')
    raw,call_fact=read(root,path,budget);call=json.loads(raw)
    args=source.get('effective_arguments')or contract.get('arguments')or{}
    from .workspace import tool_scope_issues
    if any(tool_scope_issues({'path':args[key]},owner['username'],owner['conv_id'],root.parents[2])for key in ('cif','cif_dir','output_dir')if args.get(key)):
        raise ValueError('native GCMC input belongs to a foreign or protected scope')
    result=_result(source.get('result'))
    if (call.get('call_id')!=cid or call.get('tool')!=contract['tool']or call.get('failed')is not False
            or call.get('params')!=args or _result(call.get('result'))!=result):
        raise ValueError('native GCMC producer call differs from current attempt')
    witness=attest(source,call,raw)
    if not isinstance(witness,dict)or witness.get('checkpoint_verified')is not True:
        raise ValueError('native GCMC producer checkpoint missing')
    cp=witness.get('checkpoint_path')or ref.get('checkpoint_path')
    _,cp_fact=read(root,cp,budget)
    if cp_fact['sha256']!=witness.get('checkpoint_sha256'):raise ValueError('native GCMC checkpoint changed')
    file_facts.extend([call_fact,cp_fact])
    work=Path(result.get('work_dir')or'')
    _relative(root,work)
    saved=source.get('validation_dossier')or{};did=saved.get('evidence_call_id')
    if saved.get('evidence_path')!=str(root/'evidence'/(str(did)+'.json')):
        raise ValueError('native GCMC accepted result lacks original byte-bound validation dossier')
    draw,dfact=read(root,saved['evidence_path'],budget);dcall=json.loads(draw);dossier=dcall.get('result')or{}
    expected_attempt=source.get('attempt_id')or source['token']
    if (dcall.get('call_id')!=did or dcall.get('tool')!='inspect_workflow_result' or dcall.get('failed')is not False
            or not isinstance(dossier,dict) or dossier.get('step_id')!=contract['step_id']
            or dossier.get('attempt_id')!=expected_attempt or dossier.get('job_ids')!=source.get('job_ids')
            or dossier.get('contract')!=contract):
        raise ValueError('native GCMC accepted dossier differs from current attempt')
    original=(dossier.get('method_facts')or{}).get('gcmc_native')or{}
    if original!=saved.get('method_facts',{}).get('gcmc_native'):
        raise ValueError('native GCMC compact dossier differs from immutable original')
    native=inspect_gcmc_results(work,contract['tool'],args,max_bytes=MAX_TOTAL,native_receipt=result)
    if (native['errors']or native['integrity_errors']or not native['counts']['succeeded']
            or native['counts']!=original.get('counts')):
        raise ValueError('native GCMC current result coverage/integrity differs from accepted output')
    current_files=_file_map(native);accepted_files=_file_map(original)
    if set(current_files)!=set(accepted_files):raise ValueError('native GCMC original file membership changed')
    for name,fact in current_files.items():
        accepted=accepted_files[name]
        if any(fact.get(key)!=accepted.get(key)for key in ('sha256','size')):
            raise ValueError('native GCMC original input/result bytes changed after acceptance')
        observed=stream_fact(root,name,budget)
        if observed['sha256']!=fact['sha256']or observed['size']!=fact['size']:
            raise ValueError('native GCMC result changed between parser and source hash')
        file_facts.append(observed)
    for case in native['cases']:
        if case.get('output_directory'):file_facts.append(directory_fact(root,case['output_directory']['path']))
    file_facts.append(dfact)
    binding={'owner':dict(owner),'step_id':contract['step_id'],'token':source['token'],
        'attempt_id':source.get('attempt_id'),'producer_call_id':cid,'producer_sha256':call_fact['sha256'],
        'checkpoint_sha256':cp_fact['sha256'],'contract_sha256':digest(execution_contract(contract)),
        'validation_dossier_sha256':dfact['sha256'],'parser_sha256':implementation_sha256(),
        'job_ids':source.get('job_ids')or[],
        'native_failures':[{'case_dir':c['case_dir'],'error_kind':c.get('error_kind'),'reason':c.get('reason')}for c in native['cases']if not c['ok']],
        'original_sources':[{key:fact[key]for key in ('path','sha256','size')}for fact in file_facts
            if fact.get('verification')==STREAM_VERIFICATION]}
    return {'schema':SCHEMA,'role':ROLE,'public':_public(native,work),'binding':binding,
        'execution_authorized':False,'scientific_acceptance':False}


def attach_gcmc_inputs(root,owner,state,source,candidates,sealed_inputs,*,read,budget,file_facts,attest,publish=True):
    from .gcmc_results import TOOLS
    if source.get('contract',{}).get('tool')not in TOOLS:return None
    value=build_view(root,owner,state,source,read=read,budget=budget,file_facts=file_facts,attest=attest)
    if not publish:return value
    from .ml_dataset import immutable_json
    path=Path(root)/'evidence'/('gcmc-view-'+digest(value)+'.json')
    immutable_json(root,path,value)
    raw,fact=read(root,path,budget);file_facts.append(fact)
    candidates.append(str(path));sealed_inputs[str(path)]=fact['sha256']
    # Raw .data is audited by streaming facts, never passed into the 2MiB packet reader.
    candidates[:]=[name for name in candidates if Path(name).suffix.lower()!='.data']
    return {'path':fact['path'],'sha256':fact['sha256'],'document':_document(value)}


def _document(value):
    return {'role':ROLE,'metadata_schema':SCHEMA,'json_root_pointer':'/public',
        'untrusted_content':value['public'],'native_projection':'native GCMC stream-derived result view; original bytes retained',
        'parser_sha256':value['binding']['parser_sha256']}


def restore_gcmc_view(root,owner,state,path,raw,*,read,budget,file_facts,attest):
    path=Path(path)
    if path.parent!=Path(root)/'evidence'or not path.name.startswith('gcmc-view-'):return None
    value=json.loads(raw)
    if (not isinstance(value,dict)or value.get('schema')!=SCHEMA
            or path.name!='gcmc-view-'+digest(value)+'.json'):
        raise ValueError('native GCMC view differs from immutable content identity')
    binding=value.get('binding')or{};source=state.get('nodes',{}).get(binding.get('step_id'),{})
    expected=build_view(root,owner,state,source,read=read,budget=budget,file_facts=file_facts,attest=attest)
    if value!=expected:raise ValueError('native GCMC view lost original producer/input/result binding')
    return _document(value)


def result_tables(document,source_id,tables):
    if document.get('metadata_schema')!=SCHEMA:return
    rows=[]
    for case in document['untrusted_content']['cases']:
        if case['status']!='valid':continue
        for unit,value in case['absolute_loading'].items():
            rows.append({'case':case['case'],'gas':case['gas'],'temperature_K':case['temperature_K'],
                'pressure_Pa':case['pressure_Pa'],'absolute_loading':value,'loading_unit':unit,
                'production_cycles':case['production_cycles'],'initialization_cycles':case['initialization_cycles'],
                'native_statistical_block_indices':case['statistical_blocks']['distinct_index_count']})
    if not rows:return
    tables.append({'id':'T'+str(len(tables)+1),'source_id':source_id,'source_ids':[source_id],
        'format':'native_gcmc_json','path':document['path'],'sha256':document['sha256'],
        'columns':list(rows[0]),'rows':rows,'row_count':len(rows),'row_identity_columns':['case','gas','loading_unit'],
        'column_units':{'temperature_K':'K','pressure_Pa':'Pa','absolute_loading':'see loading_unit',
            'production_cycles':'cycles','initialization_cycles':'cycles','native_statistical_block_indices':'distinct native labels, not independent samples'},
        'units':'native row loading_unit; no conversion or convergence inference'})
