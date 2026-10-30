"""LLM-authored plotting code, executed only in a verified Linux OS namespace.

Input/units/method audits are observations, not scientific acceptance. Generated
code still needs scientific/visual review; isolation cannot prove honest plots.
"""
import csv
import codecs
import ctypes
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import stat
import subprocess
import sys
import time
import uuid

from .workload_budget import _owned_fd
from .adsorption_summary import _safe_parent
from .plot_sandbox_worker import native_png_output_contract,native_png_output_contract_sha256,native_data_output_contract

MAX_STREAM_TEXT=64*1024**2
MAX_SANDBOX_INPUTS=128*1024**2


def _stream_text(root,path,limit,*,sink=None):
    """Hash/copy bounded owned UTF-8 text with constant memory and path CAS."""
    fd=_owned_fd(root,path)
    try:
        before=os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid!=os.getuid() or before.st_nlink!=1
                or not 0<before.st_size<=limit):
            raise ValueError('bounded nonempty regular input required')
        digest=hashlib.sha256();decoder=codecs.getincrementaldecoder('utf-8-sig')('strict')
        preview='';size=0
        while True:
            block=os.read(fd,65536)
            if not block:break
            size+=len(block)
            if size>before.st_size or size>limit or b'\x00' in block:raise ValueError('invalid or changed text input')
            digest.update(block);text=decoder.decode(block)
            if len(preview)<6000:preview+=(text[:6000-len(preview)])
            if sink is not None:
                view=memoryview(block)
                while view:
                    written=os.write(sink,view)
                    if written<=0:raise OSError('short sandbox input write')
                    view=view[written:]
        decoder.decode(b'',final=True)
        after=os.fstat(fd);visiblefd=_owned_fd(root,path)
        try:visible=os.fstat(visiblefd)
        finally:os.close(visiblefd)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if size!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(visible):
            raise ValueError('input changed while streaming')
        return {'sha256':digest.hexdigest(),'bytes':size,'preview':preview}
    finally:os.close(fd)


def _read(root,path,limit):
    fd=_owned_fd(root,path)
    try:
        a=os.fstat(fd)
        if not stat.S_ISREG(a.st_mode) or not 0<a.st_size<=limit:raise ValueError('bounded nonempty regular input required')
        raw=bytearray()
        while len(raw)<=a.st_size:
            data=os.read(fd,min(65536,a.st_size+1-len(raw)))
            if not data:break
            raw.extend(data)
        b=os.fstat(fd)
        if (a.st_ino,a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(b.st_ino,b.st_size,b.st_mtime_ns,b.st_ctime_ns) or len(raw)!=a.st_size:raise ValueError('input changed during reading')
        return bytes(raw)
    finally:os.close(fd)


def _audit(raw,table,index):
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    fields=reader.fieldnames
    if not fields or len(fields)>128 or len(set(fields))!=len(fields) or any(not f for f in fields):raise ValueError('CSV columns must be nonempty and unique')
    units=table.get('units',{})
    if not isinstance(units,dict) or any(k not in fields or not isinstance(v,str) or not 0<len(v)<=80 for k,v in units.items()):raise ValueError('units must name actual columns')
    method=table.get('method_column')
    if method is not None and method not in fields:raise ValueError('method column missing')
    missing={k:0 for k in fields};nonfinite={k:0 for k in fields};finite={k:0 for k in fields};methods=set();count=0
    for row in reader:
        count+=1
        if count>10000 or None in row or any(v is None for v in row.values()):raise ValueError('CSV row shape/count invalid')
        for key,value in row.items():
            if not value.strip():missing[key]+=1;continue
            try:number=float(value)
            except ValueError:continue
            if math.isfinite(number):finite[key]+=1
            else:nonfinite[key]+=1
        if method:methods.add(row[method].strip() or 'unspecified')
    if not count or not any(finite.values()):raise ValueError('table has no finite numeric observations; do not draw invented values')
    return {'source_path':table['path'],'sandbox_path':f'/inputs/table{index}.csv','sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'rows':count,'columns':fields,'units':{k:units.get(k,'unspecified') for k in fields},'units_origin':'caller_declared','missing_counts':missing,'nonfinite_counts':nonfinite,'finite_counts':finite,'methods':sorted(methods),'method_column':method,'scientific_verification':False}


def _write(fd,name,raw):
    child=os.open(name,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
    try:
        with os.fdopen(child,'wb',closefd=False)as out:out.write(raw);out.flush();os.fsync(out.fileno())
    finally:os.close(child)


def _validate_data_schema(schema,depth=0):
    """Accept only bounded local structural schemas, without refs or regexes."""
    allowed={'type','properties','required','items','additionalProperties','minimum','maximum',
             'minItems','maxItems','minLength','maxLength','enum','description'}
    if depth>8 or not isinstance(schema,dict) or set(schema)-allowed:
        raise ValueError('bounded local data schema required')
    properties=schema.get('properties',{})
    if not isinstance(properties,dict) or len(properties)>128:raise ValueError('schema property bound exceeded')
    for key,value in properties.items():
        if not isinstance(key,str) or len(key)>128:raise ValueError('invalid schema property name')
        _validate_data_schema(value,depth+1)
    if 'items' in schema:_validate_data_schema(schema['items'],depth+1)
    extra=schema.get('additionalProperties')
    if isinstance(extra,dict):_validate_data_schema(extra,depth+1)
    elif extra is not None and type(extra) is not bool:raise ValueError('invalid additionalProperties schema')
    enum=schema.get('enum')
    if enum is not None and (not isinstance(enum,list) or len(enum)>128):raise ValueError('schema enum bound exceeded')
    from jsonschema import Draft7Validator
    Draft7Validator.check_schema(schema)


def _link_commit_data_outputs(fd,temporary,checked):
    """NFS fallback: no-replace links followed by one atomic commit marker.

    A directory can be visible before commit, but no consumer is authorized
    until the immutable marker AND successful native tool receipt are verified.
    Failed stages are deliberately retained for diagnosis.
    """
    source=os.open(temporary,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    target=None
    try:
        os.mkdir('artifacts',mode=0o700,dir_fd=fd)
        target=os.open('artifacts',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        records=[{'name':item['name'],'sha256':item['sha256'],'bytes':item['bytes']}for item,_ in checked]
        for item,_ in checked:
            os.link(item['name'],item['name'],src_dir_fd=source,dst_dir_fd=target,follow_symlinks=False)
        payload=json.dumps({'schema_version':1,'protocol':'nfs-link-commit-v1','artifacts':records},sort_keys=True,separators=(',',':')).encode()
        _write(source,'.publication-pending',payload)
        os.link('.publication-pending','.publication.json',src_dir_fd=source,dst_dir_fd=target,follow_symlinks=False)
        os.fsync(target)
        for item,_ in checked:os.unlink(item['name'],dir_fd=source)
        os.unlink('.publication-pending',dir_fd=source)
        os.fsync(source)
        os.close(source);source=None
        os.rmdir(temporary,dir_fd=fd);os.fsync(fd)
        return {'protocol':'nfs-link-commit-v1','commit_sha256':hashlib.sha256(payload).hexdigest()}
    finally:
        if target is not None:os.close(target)
        if source is not None:os.close(source)


def _publish_data_outputs(fd,out,observed,declared,audits):
    """Recheck every artifact and atomically publish the complete data set."""
    from jsonschema import Draft7Validator
    expected={item['name']:item for item in declared}
    if (not isinstance(observed,list) or len(observed)!=len(expected)
            or {item.get('name') for item in observed if isinstance(item,dict)}!=set(expected)):
        raise ValueError('data output membership differs from approved declarations')
    source=os.open('rendered',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    checked=[]
    try:
        for item in observed:
            name=item['name'];spec=expected[name];child=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=source)
            try:
                info=os.fstat(child)
                if not stat.S_ISREG(info.st_mode) or not 0<=info.st_size<=8*1024**2:raise ValueError('invalid sandbox data artifact')
                data=bytearray()
                while len(data)<info.st_size:
                    chunk=os.read(child,min(65536,info.st_size-len(data)))
                    if not chunk:raise ValueError('truncated sandbox data artifact')
                    data.extend(chunk)
                if os.fstat(child).st_size!=info.st_size:raise ValueError('sandbox artifact changed')
            finally:os.close(child)
            raw=bytes(data);sha=hashlib.sha256(raw).hexdigest()
            if sha!=item.get('sha256') or len(raw)!=item.get('bytes'):
                raise ValueError('sandbox data artifact identity differs from worker receipt')
            item=dict(item,format=spec['format'])
            # CSV/JSON validation remains for callers explicitly asking for
            # structured data. The generic file output has no format gate.
            if spec['format']=='json':
                def unique(pairs):
                    out={}
                    for key,value in pairs:
                        if key in out:raise ValueError('duplicate JSON key')
                        out[key]=value
                    return out
                def invalid(value):raise ValueError('nonfinite JSON number')
                value=json.loads(raw.decode('utf-8-sig'),object_pairs_hook=unique,parse_constant=invalid)
                if not isinstance(value,(dict,list)):raise ValueError('JSON output must be an object or array')
                item['json_root_type']='object' if isinstance(value,dict) else 'array'
            elif spec['format']=='csv':
                import csv,io
                reader=csv.reader(io.StringIO(raw.decode('utf-8-sig'),newline=''),strict=True);header=next(reader,None)
                if not header or any(not x for x in header) or len(set(header))!=len(header):raise ValueError('invalid CSV columns')
                count=0
                for row in reader:
                    count+=1
                    if len(row)!=len(header):raise ValueError('invalid CSV row width')
                item.update(columns=header,rows=count)
            elif spec['format']=='txt':
                if '\x00' in raw.decode('utf-8-sig'):raise ValueError('invalid text output bytes')
            if spec.get('json_schema') is not None:
                value=json.loads(raw.decode('utf-8-sig'))
                error=next(Draft7Validator(spec['json_schema']).iter_errors(value),None)
                if error is not None:
                    pointer='/'+'/'.join(str(x).replace('~','~0').replace('/','~1') for x in error.absolute_path)
                    raise ValueError(f'JSON artifact {name} violates schema rule {error.validator} at {pointer}; inspect the declared schema and source, not an invented replacement')
            if spec.get('required_columns') is not None and not set(spec['required_columns'])<=set(item.get('columns') or []):
                missing=sorted(set(spec['required_columns'])-set(item.get('columns') or []))
                raise ValueError('CSV artifact '+name+' is missing required columns: '+json.dumps(missing,ensure_ascii=False))
            if spec.get('min_rows') is not None and item.get('rows',-1)<spec['min_rows']:
                raise ValueError(f'CSV artifact {name} has {item.get("rows",-1)} rows; declared minimum is {spec["min_rows"]}. Check actual source membership before revising the contract.')
            checked.append((item,raw))
    finally:os.close(source)
    temporary='.data-stage-'+uuid.uuid4().hex
    os.mkdir(temporary,mode=0o700,dir_fd=fd)
    target=os.open(temporary,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
    try:
        for item,raw in checked:_write(target,item['name'],raw)
        os.fsync(target)
    finally:os.close(target)
    libc=ctypes.CDLL(None,use_errno=True)
    rename=getattr(libc,'renameat2',None)
    publication={'protocol':'atomic-rename-noreplace-v1'}
    if rename is None:
        publication=_link_commit_data_outputs(fd,temporary,checked)
    else:
        rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
        if rename(fd,os.fsencode(temporary),fd,b'artifacts',1)!=0:
            code=ctypes.get_errno()
            if code in {errno.EINVAL,errno.ENOSYS,errno.EOPNOTSUPP}:
                publication=_link_commit_data_outputs(fd,temporary,checked)
            else:raise OSError(code,'atomic no-replace data publication failed')
    os.fsync(fd)
    return [{**item,'path':str(out/'artifacts'/item['name']),
             'source_refs':[a['sha256'] for a in audits],
             'evidence_class':'sandbox_derived_unverified','scientific_acceptance':False}
            for item,_ in checked],publication


def plot_result_code(script,input_tables=None,output_dir=None,*,owned_root,timeout_seconds=30,guidance_files=None,input_files=None,_mode='plot',_declared_outputs=None,_owner=None,_producer=None,_online=False,_proxy_port=None,_direct=False,_allow_local=False):
    """A new immutable attempt per call; repair code by selecting a NEW output_dir.

    Script reads /inputs/tableN.csv and /inputs/audit.json, writes 1..8 PNG files
    under /output. No shell/network/process creation. Source table bytes copied
    read-only; exact script, snapshots, logs, hashes retained even on failure.
    """
    root=Path(os.path.abspath(owned_root));out=Path(os.path.abspath(output_dir))
    if _mode not in {'plot','data'}:raise ValueError('unsupported sandbox mode')
    if (type(_online)is not bool or type(_direct)is not bool or type(_allow_local)is not bool
            or (_online and _mode!='data') or (_direct and not _online)):
        raise ValueError('invalid sandbox online mode')
    if _online and (not isinstance(_producer,dict) or not _producer.get('step_id') or not _producer.get('runtime_token')):
        raise PermissionError('online sandbox requires an owned active native task')
    if _online and not _direct and (type(_proxy_port)is not int or not 1<=_proxy_port<=65535):
        raise ValueError('online sandbox requires a configured local proxy')
    if _mode=='data':
        if (not isinstance(_owner,dict) or set(_owner)!={'username','conv_id'}
                or (root.parent.name,root.name)!=(_owner.get('username'),_owner.get('conv_id'))):
            raise PermissionError('data sandbox requires the authenticated owner scope')
        if not isinstance(_declared_outputs,list) or not _declared_outputs:raise ValueError('declare the task output files')
        names_expected=[x.get('name') for x in _declared_outputs if isinstance(x,dict)]
        if len(names_expected)!=len(_declared_outputs) or len(set(names_expected))!=len(names_expected):raise ValueError('unique declared data outputs required')
        import re
        for item in _declared_outputs:
            name=item.get('name');kind=item.get('format')
            if (not isinstance(name,str) or not name or name in {'.','..','.publication.json','.publication-pending'} or '/' in name or '\x00' in name
                    or kind not in {'json','csv','txt','file'} or (kind!='file' and not name.lower().endswith('.'+kind))
                    or set(item)-{'name','format','json_schema','required_columns','min_rows'}):raise ValueError('invalid declared data output')
            schema=item.get('json_schema')
            if schema is not None:
                encoded=json.dumps(schema,allow_nan=False) if isinstance(schema,dict) else ''
                if kind!='json' or not isinstance(schema,dict) or len(encoded)>16384 or '$ref' in encoded:raise ValueError('bounded reference-free JSON schema required')
                _validate_data_schema(schema)
            columns=item.get('required_columns')
            if columns is not None and (kind!='csv' or not isinstance(columns,list) or not columns or len(columns)>128 or any(not isinstance(c,str) or not c for c in columns) or len(set(columns))!=len(columns)):raise ValueError('invalid required CSV columns')
            minimum=item.get('min_rows')
            if minimum is not None and (kind!='csv' or type(minimum) is not int or not 0<=minimum<=50000):raise ValueError('invalid minimum CSV rows')
    if not out.is_relative_to(root) or out==root:raise ValueError('plot directory must be new and session-owned')
    if not isinstance(script,str) or not 0<len(script.encode())<=65536:raise ValueError('bounded Python script required')
    compile(script,'script.py','exec')
    if type(timeout_seconds)is not int or not 1<=timeout_seconds<=60:raise ValueError('timeout must be integer 1..60 seconds')
    if bool(input_tables)==bool(input_files):raise ValueError('choose exactly one input_tables or input_files source set')
    descriptors=input_files if input_files is not None else input_tables
    if not isinstance(descriptors,list) or not descriptors or (_mode=='plot' and len(descriptors)>8):raise ValueError('require owned source inputs within the sandbox resource budget')
    from .result_guidance import read_result_guidance
    guidance=read_result_guidance(guidance_files)['files'] if guidance_files else []
    sources=[];audits=[];names=[];derived=[];total=0
    for i,table in enumerate(descriptors):
        allowed={'path','units','method_column'}|({'input_id','format','expected_sha256'}if input_files is not None else set())
        if not isinstance(table,dict) or set(table)-allowed:raise ValueError('invalid source input descriptor')
        path=Path(table['path'])
        if not path.is_absolute():raise ValueError('owned absolute table path required')
        sourcefd=_owned_fd(root,path)
        try:
            info=os.fstat(sourcefd);size=info.st_size
            if _mode=='data' and (not stat.S_ISREG(info.st_mode) or info.st_uid!=os.getuid() or info.st_nlink!=1):
                raise ValueError('data sandbox input must be an owned singly linked regular file')
        finally:os.close(sourcefd)
        large=size>8*1024**2
        if large and (input_files is None or table.get('format')!='text'):
            raise ValueError('large raw files require explicit UTF-8 text format')
        streamed=_stream_text(root,path,MAX_STREAM_TEXT) if large else None
        raw=streamed['preview'].encode('utf-8') if streamed else _read(root,path,8*1024**2)
        total+=streamed['bytes'] if streamed else len(raw)
        if total>MAX_SANDBOX_INPUTS:raise ValueError('source byte budget exceeded')
        sources.append(None if streamed else raw)
        if input_files is None:
            names.append(f'table{i}.csv');audits.append(_audit(raw,table,i))
        else:
            from .plot_input_adapter import describe
            filename,audit=describe(raw,table,i)
            if streamed:
                audit.update(sha256=streamed['sha256'],bytes=streamed['bytes'],
                    preview={'text':streamed['preview'],'truncated':True},text_lines=None,
                    inspection_scope='bounded preview; complete original is sandbox input')
                audit.pop('native_structure',None);audit.pop('native_structure_gap',None)
            if filename in names:raise ValueError('plot input IDs must be unique')
            if audit['format']=='csv':
                parsed=_audit(raw,table,i);parsed.update(sandbox_path=audit['sandbox_path'],input_id=audit['input_id'],format='csv')
                audit=parsed
            expected=table.get('expected_sha256')
            if expected is not None and (not isinstance(expected,str) or len(expected)!=64 or expected!=audit['sha256']):
                raise ValueError('sandbox source differs from the authenticated input SHA')
            names.append(filename);audits.append(audit)
            if audit.get('native_structure'):
                from .plot_input_adapter import native_thermo_input
                sidecar=native_thermo_input(raw,audit['input_id'])
                if sidecar is None or sidecar[2]!=audit['native_structure']:raise ValueError('native thermo parser identity changed during input preparation')
                derived.append((sidecar[0],sidecar[1]));total+=len(sidecar[1])
                if total>MAX_SANDBOX_INPUTS:raise ValueError('source and native parsed input byte budget exceeded')
    parent=_safe_parent(root,out.parent)
    try:
        os.mkdir(out.name,mode=0o700,dir_fd=parent)
        fd=os.open(out.name,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
    finally:os.close(parent)
    implementations=_implementation_facts(Path(__file__).resolve().parents[1],mode=_mode,online=_online)
    receipt={'schema_version':1,'execution_id':uuid.uuid4().hex,
        'evidence_class':'sandbox_derived_unverified','retry_authorized':False,
        'visual_review':{'status':'pending','reason':'no_native_image_to_provider_bridge'},
        'parameters':{'backend':'python','timeout_seconds':timeout_seconds,'input_tables':input_tables,
            'input_files':input_files,'guidance_files':guidance_files or []},
        'output_schema_sha256':native_png_output_contract_sha256(),'output_contract':native_png_output_contract(),
        'guidance':[{k:v for k,v in f.items() if k!='text'} for f in guidance],'cacheable':False,'implementation_files':implementations,'ok':False,'status':'sandbox_unavailable','scientific_acceptance':False,'script_sha256':hashlib.sha256(script.encode()).hexdigest(),'output_dir':str(out),'inputs':audits,'artifacts':[],
        'error':'The required OS sandbox is unavailable; no code was executed outside isolation.',
        'error_kind':'plot_sandbox_unavailable',
        'warnings':['Generated code and figures require visual/scientific review; table presence is not convergence evidence.','Missing/nonfinite data must remain explicit, never silently zero-filled.','Different methods/units must be labelled and separated; mixed-method rankings are exploratory.'],
        'sandbox':{'os_isolated':False,'backend':'linux-unshare-chroot-seccomp','network':False,'host_ipc':False}}
    if _mode=='data':
        receipt.update(output_contract=native_data_output_contract(),declared_outputs=_declared_outputs,
            output_schema_sha256=hashlib.sha256(json.dumps(_declared_outputs,sort_keys=True,separators=(',',':')).encode()).hexdigest(),
            owner=dict(_owner),producer=dict(_producer or {}),visual_review=None,error_kind='data_sandbox_unavailable')
    pending_data=None
    def write(name,raw):_write(fd,name,raw)
    try:
        write('script.py',script.encode())
        os.mkdir('code',dir_fd=fd);os.mkdir('inputs',dir_fd=fd)
        codefd=os.open('code',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        inputsfd=os.open('inputs',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
        try:
            _write(codefd,'script.py',script.encode())
            helper=Path(__file__).with_name('plot_render_audit.py')
            _write(codefd,'native_plot_audit.py',_read(helper.parent,helper,256*1024))
            for filename,raw,descriptor,audit in zip(names,sources,descriptors,audits):
                if raw is not None:
                    _write(inputsfd,filename,raw)
                else:
                    child=os.open(filename,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=inputsfd)
                    try:
                        copied=_stream_text(root,Path(descriptor['path']),MAX_STREAM_TEXT,sink=child)
                        if copied['sha256']!=audit['sha256'] or copied['bytes']!=audit['bytes']:
                            raise ValueError('source changed before sandbox staging')
                        os.fsync(child)
                    finally:os.close(child)
            for filename,raw in derived:_write(inputsfd,filename,raw)
            # Host paths remain in the private execution receipt only. The
            # isolated script needs the mapped input identity, not host layout.
            sandbox_audits=[{key:value for key,value in item.items() if key!='source_path'} for item in audits]
            _write(inputsfd,'audit.json',json.dumps(sandbox_audits,ensure_ascii=False).encode())
            _write(inputsfd,'output_contract.json',json.dumps(native_data_output_contract() if _mode=='data' else native_png_output_contract(),sort_keys=True).encode())
            guidance_index=[]
            for i,item in enumerate(guidance):
                name=f'guidance{i}.md';_write(inputsfd,name,item['text'].encode())
                guidance_index.append({'path':item['path'],'sandbox_path':'/inputs/'+name,'sha256':item['sha256']})
            _write(inputsfd,'guidance_index.json',json.dumps(guidance_index).encode())
        finally:os.close(codefd);os.close(inputsfd)
        unshare=shutil.which('unshare')
        if unshare and (os.environ.get('SLURM_JOB_ID') or _allow_local) and platform.system()=='Linux' and platform.machine()=='x86_64':
            worker=Path(__file__).with_name('plot_sandbox_worker.py')
            env={'PATH':'/usr/bin:/bin','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LANG':'C.UTF-8'}
            command=[unshare,'--user','--map-root-user','--mount','--pid','--fork','--net','--ipc','--kill-child=KILL',sys.executable,'-I',str(worker),f'/proc/self/fd/{fd}',str(timeout_seconds),'data-online' if _online else _mode]
            if _online:
                # A separate user scope bounds the entire process tree, unlike
                # per-process rlimits. Fail closed if the scope is unavailable.
                systemd_run=shutil.which('systemd-run')
                if not systemd_run:
                    raise RuntimeError('Online sandbox requires a task resource scope')
                runtime_dir=os.environ.get('XDG_RUNTIME_DIR',f'/run/user/{os.getuid()}')
                if runtime_dir != f'/run/user/{os.getuid()}':
                    raise RuntimeError('Unexpected user runtime directory')
                env['XDG_RUNTIME_DIR']=runtime_dir
                command=[systemd_run,'--user','--scope','--quiet','--unit=bimem-sandbox-'+uuid.uuid4().hex,
                    '-p','TasksMax=32','-p','MemoryMax=2G','-p','CPUQuota=100%',*command]
            start=time.monotonic()
            if _online:
                from .sandbox_task_proxy import host_proxy_socket
                key=hashlib.sha256(json.dumps({'owner':_owner,'producer':_producer},sort_keys=True).encode()).hexdigest()
                work_path=root/'.sandbox-work'/key
                workfd=_safe_parent(root,work_path)
                try:
                    info=os.fstat(workfd)
                    if info.st_uid!=os.getuid() or stat.S_IMODE(info.st_mode)&0o077:
                        raise PermissionError('task workspace is not private and owned')
                    # Ensure Python's import machinery sees a real directory
                    # before a child installs a dependency during this run.
                    try:os.mkdir('dependencies',mode=0o700,dir_fd=workfd)
                    except FileExistsError:pass
                    depsfd=os.open('dependencies',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=workfd)
                    try:
                        deps=os.fstat(depsfd)
                        if deps.st_uid!=os.getuid():raise PermissionError('task dependency directory is not owned')
                    finally:os.close(depsfd)
                    try:os.mkdir('tmp',mode=0o700,dir_fd=workfd)
                    except FileExistsError:pass
                    tmpfd=os.open('tmp',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=workfd)
                    try:
                        temp_info=os.fstat(tmpfd)
                        if temp_info.st_uid!=os.getuid():raise PermissionError('task temporary directory is not owned')
                    finally:os.close(tmpfd)
                    socket_name='network-proxy-'+uuid.uuid4().hex+'.sock'
                    with host_proxy_socket(workfd,_proxy_port,socket_name=socket_name,direct=_direct):
                        env.update(TASK_SANDBOX_WORK_FD=str(workfd),TASK_SANDBOX_PROXY_SOCKET=socket_name)
                        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,pass_fds=(fd,workfd),start_new_session=True)
                        try:stdout,stderr=process.communicate(timeout=timeout_seconds+15)
                        except BaseException:
                            os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
                finally:os.close(workfd)
            else:
                process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,env=env,pass_fds=(fd,),start_new_session=True)
                try:stdout,stderr=process.communicate(timeout=timeout_seconds+15)
                except BaseException:
                    os.killpg(process.pid,signal.SIGKILL);process.communicate();raise
            write('launcher.stdout',stdout[:65536]);write('launcher.stderr',stderr[:65536])
            receipt['duration_seconds']=round(time.monotonic()-start,6)
            try:observed=json.loads(stdout.decode().strip().splitlines()[-1])
            except (ValueError,IndexError,UnicodeError):observed={}
            receipt['exit_code']=observed.get('exit_code');receipt['timed_out']=observed.get('timed_out',False)
            if isinstance(observed.get('output_validation'),dict):receipt['output_validation']=observed['output_validation']
            receipt['render_diagnostics']={'evidence_class':'in_process_runtime_advisory',
                'independently_verified':False,'scientific_acceptance':False,
                'reports':observed.get('render_diagnostics',[])}
            if process.returncode==0 and observed.get('os_isolated') is True:
                receipt['sandbox']['os_isolated']=True;receipt['status']='execution_failed'
                if _online:
                    receipt['sandbox'].update(network=True,network_scope='task_private_proxy',
                        network_connectivity_verified=False)
                receipt['error']=('The isolated data script did not produce its declared outputs; inspect its native stderr and exit status.'
                    if _mode=='data' else 'The isolated plot script did not produce accepted PNG output; inspect its native stderr and exit status.')
                receipt['error_kind']='data_execution_failed' if _mode=='data' else 'plot_execution_failed'
                if observed.get('output_error'):
                    receipt['error']='Generated output violates the native contract: '+str(observed['output_error'])[:1000]
                    receipt['error_kind']='data_output_contract' if _mode=='data' else 'plot_output_contract'
                if _mode=='data' and observed.get('exit_code')==0 and observed.get('data_artifacts'):
                    pending_data=observed['data_artifacts']
                if observed['exit_code']==0 and observed.get('images'):
                    # The untrusted script cannot write host rendered/. Launcher
                    # copies only checked images after the one child has stopped.
                    renderfd=os.open('rendered',os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
                    try:
                        for name in observed['images']:
                            imagefd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=renderfd)
                            try:
                                info=os.fstat(imagefd)
                                if not stat.S_ISREG(info.st_mode) or not native_png_output_contract()['image_bytes']['min']<=info.st_size<=native_png_output_contract()['image_bytes']['max']:raise ValueError('invalid rendered PNG')
                                raw=bytearray()
                                while len(raw)<info.st_size:
                                    chunk=os.read(imagefd,min(65536,info.st_size-len(raw)))
                                    if not chunk:raise ValueError('truncated rendered PNG')
                                    raw.extend(chunk)
                                raw=bytes(raw)
                            finally:os.close(imagefd)
                            write(name,raw);receipt['artifacts'].append({'path':str(out/name),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'kind':'PNG','source_refs':[a['sha256'] for a in audits],
                                'schema_validation':{'passed':True,'validator':'Pillow.Image.verify','format':'PNG','max_dimensions':[native_png_output_contract()['width_px']['max'],native_png_output_contract()['height_px']['max']]}})
                    finally:os.close(renderfd)
                    receipt['ok']=True;receipt['status']='rendered_needs_review'
                    receipt.pop('error',None);receipt.pop('error_kind',None)
        # Rehash actual sources before saying these plots reflect current tables.
        for table,audit,raw in zip(descriptors,audits,sources):
            observed=(_stream_text(root,Path(table['path']),MAX_STREAM_TEXT)['sha256'] if raw is None
                      else hashlib.sha256(_read(root,Path(table['path']),8*1024**2)).hexdigest())
            if observed!=audit['sha256']:
                receipt['ok']=False;receipt['status']='source_changed'
                receipt['error']='Plot source bytes changed during rendering.';receipt['error_kind']='plot_source_changed';break
        if _implementation_facts(Path(__file__).resolve().parents[1],mode=_mode,online=_online)!=implementations:
            receipt['ok']=False;receipt['status']='implementation_changed'
            receipt['error']='Plot implementation changed during rendering.';receipt['error_kind']='plot_implementation_changed'
        if (_mode=='data' and pending_data is not None and receipt.get('sandbox',{}).get('os_isolated') is True
                and receipt.get('error_kind') not in {'plot_source_changed','plot_implementation_changed'}):
            try:
                checked,publication=_publish_data_outputs(fd,out,pending_data,_declared_outputs,audits)
                receipt.update(ok=True,status='data_ready_needs_review',artifacts=checked,publication=publication)
                receipt.pop('error',None);receipt.pop('error_kind',None)
            except (OSError,ValueError,TypeError,KeyError) as error:
                receipt.update(error='Data output rejected: '+str(error)[:800],error_kind='data_output_contract')
        write('execution.json',json.dumps(receipt,ensure_ascii=False,sort_keys=True).encode())
        write('manifest.json',json.dumps(receipt,ensure_ascii=False,sort_keys=True).encode());os.fsync(fd)
        return receipt
    finally:os.close(fd)


from .plot_dependencies import PLOT_IMPLEMENTATIONS


def _implementation_facts(project_root,mode='plot',online=False):
    root=Path(project_root)
    files={}
    names=('agents/registry.py','agents/installations.py',*PLOT_IMPLEMENTATIONS)
    if mode=='data':names=(*names,'agents/analysis_artifact_binding.py','agents/workflow_compiler.py','agents/watch_context.py')
    if online:names=(*names,'agents/sandbox_task_proxy.py')
    for name in names:
        raw=_read(root,root/name,16*1024**2)
        files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    return files


def plot_dependency_identity(arguments,project_root):
    """Auditable script/data/source identity, NEVER permission to reuse images.

    Arbitrary code can import any installed library, read system runtime/font
    assets, use time or randomness. That environment is not operator-locked;
    claiming a complete reusable computation identity would be false. Scripts
    remain reusable as source, with new attempts and fresh input observations.
    """
    try:
        root=Path(os.path.abspath(project_root));out=Path(arguments['output_dir'])
        parts=out.relative_to(root/'runs').parts
        if not out.is_absolute() or len(parts)<3 or any(p in {'.','..'} for p in parts):return None
        owned=root/'runs'/parts[0]/parts[1]
        script=arguments['script']
        if not isinstance(script,str) or not 0<len(script.encode())<=65536:return None
        compile(script,'script.py','exec')
        tables=arguments['input_tables']
        if not isinstance(tables,list) or not 1<=len(tables)<=8:return None
        observed=[];total=0
        for i,table in enumerate(tables):
            raw=_read(owned,Path(table['path']),8*1024**2);total+=len(raw)
            if total>16*1024**2:return None
            observed.append(_audit(raw,table,i))
        payload={'schema':'plot-code-provenance-v1','arguments':arguments,'inputs':observed,
                 'implementation_files':_implementation_facts(root)}
        raw=json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        return {**payload,'identity_sha256':hashlib.sha256(raw).hexdigest(),'cacheable':False,
                'cache_reason':'arbitrary plotting code has no operator-locked complete runtime dependency identity'}
    except (OSError,ValueError,TypeError,KeyError,UnicodeError,SyntaxError):return None
