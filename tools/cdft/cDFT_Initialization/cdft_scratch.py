"""Opt-in future-job local execution. No imports from the agent control plane.

Only preflight may fall back to shared execution. Once a process starts, an
error preserves its evidence and NEVER starts another copy of the calculation.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import ctypes
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

MB=1024*1024


class ScratchUnavailable(ValueError):
    """A verified resource prerequisite is unavailable before execution."""


def normalized_policy(policy):
    if policy is None:return None
    if not isinstance(policy,dict) or type(policy.get('enabled',False)) is not bool:
        raise ValueError('operator scratch policy must contain a boolean enabled')
    if not policy.get('enabled'):return None
    allowed={'enabled','root','capacity_mb','reserve_mb','sync_seconds','on_unavailable','tmpfs'}
    if set(policy)-allowed:raise ValueError('unknown operator scratch fields')
    root=policy.get('root')
    if not isinstance(root,str) or not Path(root).is_absolute() or '..' in Path(root).parts or root=='/':
        raise ValueError('explicit absolute non-root scratch location required')
    result={**policy,'reserve_mb':policy.get('reserve_mb',256),
            'sync_seconds':policy.get('sync_seconds',60),'on_unavailable':policy.get('on_unavailable','fail')}
    for key,minimum in [('capacity_mb',1),('reserve_mb',0),('sync_seconds',1)]:
        if type(result.get(key)) is not int or not minimum<=result[key]<=2**31-1:
            raise ValueError('finite integer operator scratch bound required: '+key)
    if result['on_unavailable'] not in {'fail','shared'}:raise ValueError('invalid scratch fallback policy')
    if 'tmpfs' in result:
        memory=result['tmpfs']
        if not isinstance(memory,dict) or set(memory)!={'per_worker_mb','overhead_mb','executable_sha256'}:
            raise ValueError('tmpfs requires an executable-bound operator memory estimate')
        if any(type(memory.get(k)) is not int or memory[k]<1 for k in ('per_worker_mb','overhead_mb')):
            raise ValueError('tmpfs process memory bounds must be positive integers')
        if not isinstance(memory['executable_sha256'],str) or not re.fullmatch('[0-9a-f]{64}',memory['executable_sha256']):
            raise ValueError('tmpfs executable SHA256 required')
    return result


def _directory(path):
    path=Path(path)
    if not path.is_absolute() or '..' in path.parts:raise ValueError('absolute non-traversing directory required')
    fd=os.open('/',os.O_RDONLY|os.O_DIRECTORY)
    try:
        for part in path.parts[1:]:
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        return fd
    except BaseException:os.close(fd);raise


def _open_regular(path):
    path=Path(path);parent=_directory(path.parent)
    try:fd=os.open(path.name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
    finally:os.close(parent)
    if not stat.S_ISREG(os.fstat(fd).st_mode):os.close(fd);raise ValueError('regular native file required')
    return fd


def file_fact(path,limit=256*MB):
    fd=_open_regular(path)
    try:
        before=os.fstat(fd)
        if before.st_size>limit:raise ValueError('native file exceeds byte bound')
        digest=hashlib.sha256();size=0
        while True:
            block=os.read(fd,65536)
            if not block:break
            size+=len(block)
            if size>limit:raise ValueError('native file grew past byte bound')
            digest.update(block)
        after=os.fstat(fd)
        linked=_open_regular(path)
        try:visible=os.fstat(linked)
        finally:os.close(linked)
        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if size!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(visible):
            raise ValueError('native file changed during observation')
        return {'sha256':digest.hexdigest(),'size':size}
    finally:os.close(fd)


def immutable_bytes(path,data,mode=0o600):
    path=Path(path);directory=_directory(path.parent);temporary='.scratch-publish-'+uuid.uuid4().hex
    try:
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,mode,dir_fd=directory)
        with os.fdopen(fd,'wb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path.name,src_dir_fd=directory,dst_dir_fd=directory,follow_symlinks=False)
        os.fsync(directory)
    finally:
        try:os.unlink(temporary,dir_fd=directory)
        except FileNotFoundError:pass
        os.close(directory)


def _json_bytes(value):return json.dumps(value,sort_keys=True,indent=2,allow_nan=False).encode()


def _regular_bytes(source,limit):
    expected=file_fact(source,limit);fd=_open_regular(source)
    try:
        data=b''
        while len(data)<=limit:
            block=os.read(fd,min(65536,limit+1-len(data)))
            if not block:break
            data+=block
    finally:os.close(fd)
    if len(data)!=expected['size'] or hashlib.sha256(data).hexdigest()!=expected['sha256']:
        raise ValueError('source changed while staging')
    return data,expected


def _copy_regular(source,target,limit):
    data,expected=_regular_bytes(source,limit)
    immutable_bytes(target,data,0o700 if os.stat(source,follow_symlinks=False).st_mode&0o111 else 0o600)
    if file_fact(target,limit)!=expected:raise ValueError('staged bytes differ')
    return expected


def filesystem_type(path):
    """Ask the kernel filesystem magic; rootfs names can hide tmpfs backing."""
    command=shutil.which('stat')
    if not command:return 'unknown'
    try:
        result=subprocess.run([command,'-f','-c','%t','--',str(path)],check=True,
            stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=10)
        magic=int(result.stdout.strip(),16)
    except (OSError,ValueError,subprocess.SubprocessError):return 'unknown'
    return {0x01021994:'tmpfs',0xEF53:'ext4',0x58465342:'xfs',0x9123683E:'btrfs',
            0x2FC12FC1:'zfs',0x6969:'nfs',0x0BD00BD0:'lustre',0x47504653:'gpfs',
            0xFF534D42:'cifs'}.get(magic,'unknown')


def rename_noreplace(source,target):
    """Atomic directory publication, with no unsafe overwrite fallback."""
    library=ctypes.CDLL(None,use_errno=True)
    operation=getattr(library,'renameat2',None)
    if operation is None:raise OSError(errno.ENOTSUP,'atomic no-replace rename is unavailable')
    operation.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint]
    source=Path(source);target=Path(target)
    source_fd=_directory(source.parent);target_fd=_directory(target.parent)
    try:
        if operation(source_fd,os.fsencode(source.name),target_fd,os.fsencode(target.name),1)!=0:
            code=ctypes.get_errno();raise OSError(code,os.strerror(code),str(target))
    finally:os.close(source_fd);os.close(target_fd)


def _publication_probe(work):
    source=work/('.scratch-probe-'+uuid.uuid4().hex);target=work/('.scratch-probe-'+uuid.uuid4().hex)
    source.mkdir();target.mkdir()
    try:
        try:rename_noreplace(source,target)
        except FileExistsError:pass
        else:raise ScratchUnavailable('publication filesystem did not preserve an existing destination')
        target.rmdir();rename_noreplace(source,target)
    except OSError as error:raise ScratchUnavailable('atomic no-replace publication unavailable: '+str(error)) from error
    finally:
        for path in (source,target):
            if path.exists():path.rmdir()


def _allocated_workers(workers):
    if type(workers) is not int or not 1<=workers<=4096:raise ValueError('positive bounded worker count required')
    value=os.environ.get('SLURM_CPUS_PER_TASK')
    if value is not None:
        if not value.isdigit() or int(value)<1:raise ValueError('invalid allocated CPU count')
        workers=min(workers,int(value))
    return workers


def prepare_job(*,work_dir,input_dat_dir,executable,policy,job_id,workers,memory_mb,
                expected_inputs=None,expected_executable_sha256=None):
    policy=normalized_policy(policy)
    if policy is None:return {'mode':'shared','reason':'operator_disabled'}
    if not isinstance(job_id,str) or not re.fullmatch('[0-9]+',job_id):raise ValueError('actual Slurm job ID required')
    work=Path(work_dir);inputs=Path(input_dat_dir);exe=Path(executable);root=Path(policy['root'])
    for path in (work,inputs,root):fd=_directory(path);os.close(fd)
    # A fallback must never overwrite a previous calculation either.
    if any(path.exists() or path.is_symlink() for path in (work/'results.csv', work/'input_dir', work/('.scratch_claim_'+job_id+'.json'))):
        raise FileExistsError('native result destination already exists; preserve the previous attempt')
    entries=list(inputs.iterdir())
    if len(entries)>1024:raise ValueError('input inventory exceeds bound')
    names=sorted(p.name for p in entries if p.name.endswith('.dat'))
    if not names or len(names)>512:raise ValueError('bounded nonempty native input inventory required')
    native={name:file_fact(inputs/name,16*MB) for name in names}
    if any(not fact['size'] for fact in native.values()):raise ValueError('empty native input')
    if expected_inputs is not None and expected_inputs!=native:raise ValueError('submitted input manifest changed')
    exe_fact=file_fact(exe,256*MB)
    if not exe_fact['size'] or not os.stat(exe,follow_symlinks=False).st_mode&0o111:raise ValueError('configured executable required')
    if expected_executable_sha256 is not None and exe_fact['sha256']!=expected_executable_sha256:
        raise ValueError('submitted executable changed')
    workers=_allocated_workers(workers)
    kind=filesystem_type(root);capacity=policy['capacity_mb']*MB
    if kind not in {'ext2','ext3','ext4','xfs','btrfs','zfs','tmpfs'}:
        raise ScratchUnavailable('configured scratch is not a verified node-local filesystem: '+kind)
    if sum(item['size'] for item in native.values())+exe_fact['size']>=capacity:
        raise ScratchUnavailable('inputs and executable exceed the operator scratch budget')
    if shutil.disk_usage(root).free<capacity+policy['reserve_mb']*MB:
        raise ScratchUnavailable('node-local free capacity does not cover the configured scratch budget')
    if kind=='tmpfs':
        bound=policy.get('tmpfs') or {}
        allocated=memory_mb
        if type(allocated) is not int or allocated<1:raise ScratchUnavailable('finite allocated memory required for tmpfs')
        environment=os.environ.get('SLURM_MEM_PER_NODE')
        if environment:
            if not environment.isdigit() or int(environment)<1:raise ScratchUnavailable('invalid scheduler memory evidence')
            allocated=min(allocated,int(environment))
        if (bound.get('executable_sha256')!=exe_fact['sha256']
                or bound.get('per_worker_mb',0)<1 or bound.get('overhead_mb',0)<1
                or bound['per_worker_mb']*workers+bound['overhead_mb']+policy['capacity_mb']>allocated):
            raise ScratchUnavailable('tmpfs consumes job memory; executable-bound process and scratch budgets do not fit')
    if not shutil.which('rsync'):raise ScratchUnavailable('rsync is unavailable on the selected node')
    _publication_probe(work)
    token=uuid.uuid4().hex
    immutable_bytes(work/('.scratch_claim_'+job_id+'.json'),_json_bytes({'job_id':job_id,'token':token,'kind':'exclusive_local_job_attempt'}))
    scratch=Path(tempfile.mkdtemp(prefix='cdft-'+job_id+'-',dir=root))
    plan={'schema_version':1,'mode':'local','job_id':job_id,'token':token,'scratch':str(scratch),
        'root':str(root),'work_dir':str(work),'workers':workers,'memory_mb':memory_mb,'policy':policy,
        'filesystem':kind,'inputs':native,'executable_sha256':exe_fact['sha256']}
    work_fd=_directory(work)
    try:plan['work_identity']=[os.fstat(work_fd).st_dev,os.fstat(work_fd).st_ino]
    finally:os.close(work_fd)
    immutable_bytes(scratch/'.owner.json',_json_bytes({key:plan[key] for key in ('job_id','token','scratch','root','work_dir')}))
    (scratch/'input').mkdir();(scratch/'input_dir').mkdir()
    for name in names:_copy_regular(inputs/name,scratch/'input'/name,16*MB)
    _copy_regular(exe,scratch/'solver',256*MB)
    immutable_bytes(scratch/'plan.json',_json_bytes(plan))
    return plan


def _validate_plan(plan):
    root=Path(plan['root']);scratch=Path(plan['scratch'])
    if (not scratch.is_absolute() or scratch in {Path('/'),Path('/tmp'),root}
            or scratch.parent!=root or not scratch.name.startswith('cdft-'+plan['job_id']+'-')):
        raise ValueError('refusing unowned scratch path')
    for path in (root,scratch):fd=_directory(path);os.close(fd)
    work=Path(plan['work_dir']);work_fd=_directory(work)
    try:
        observed=os.fstat(work_fd)
        if [observed.st_dev,observed.st_ino]!=plan.get('work_identity'):raise ValueError('shared result root was replaced')
    finally:os.close(work_fd)
    marker=json.loads(_regular_bytes(scratch/'.owner.json',4096)[0])
    if marker!={key:plan[key] for key in ('job_id','token','scratch','root','work_dir')}:
        raise ValueError('scratch ownership marker differs')
    return scratch,Path(plan['work_dir'])


def rsync_tree(source,target,*,partial=False):
    target=Path(target);target.mkdir(parents=True,exist_ok=True);fd=_directory(target);os.close(fd)
    command=['rsync','-a','--no-owner','--no-group','--no-links','--no-specials','--no-devices','--protect-args']
    if partial:command+=['--include=*/','--include=*.log','--include=input.dat','--include=output.dat','--include=hr.dat','--include=execution.json','--exclude=*']
    subprocess.run(command+['--',str(source)+'/',str(target)+'/'],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=120)


def sync_partial(plan):
    scratch,work=_validate_plan(plan);target=work/('.partial_'+plan['job_id']+'_'+plan['token'])
    rsync_tree(scratch/'input_dir',target,partial=True)
    return {'snapshot_only':True,'path':str(target),'not_a_scientific_checkpoint':True}


def observed_tree_bytes(root,limit):
    """Capacity is a bounded metadata observation, not a mutable-file proof.

    Other workers are allowed to keep writing. Only each stopped worker's
    publication is hashed. This observation is not an enforced disk quota.
    """
    total=0;count=0;pending=[Path(root)]
    while pending:
        directory=pending.pop();fd=_directory(directory)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    count+=1
                    if count>4096:raise ValueError('native output inventory exceeds bound')
                    info=entry.stat(follow_symlinks=False)
                    if stat.S_ISDIR(info.st_mode):pending.append(directory/entry.name)
                    elif stat.S_ISREG(info.st_mode):total+=info.st_size
                    else:raise ValueError('native output contains a symlink or nonregular file')
                    if total>limit:raise ValueError('aggregate native output exceeds scratch byte bound')
        finally:os.close(fd)
    return total


def tree_manifest(root,limit):
    result={};total=0;pending=[Path(root)]
    while pending:
        directory=pending.pop();fd=_directory(directory)
        try:
            with os.scandir(fd) as entries:
                for entry in entries:
                    if len(result)+len(pending)>=4096:raise ValueError('native output inventory exceeds bound')
                    path=directory/entry.name
                    if entry.is_symlink():raise ValueError('native output symlinks cannot be published')
                    if entry.is_dir(follow_symlinks=False):pending.append(path)
                    elif entry.is_file(follow_symlinks=False):
                        fact=file_fact(path,limit-total);total+=fact['size'];result[str(path.relative_to(root))]=fact
                    else:raise ValueError('native output contains a nonregular file')
        finally:os.close(fd)
    return result


def _publish_case(plan,run):
    scratch,work=_validate_plan(plan);limit=plan['policy']['capacity_mb']*MB
    manifest=tree_manifest(run,limit);parent=work/'input_dir';parent.mkdir(exist_ok=True)
    staging=work/('.publish_'+plan['token']+'_'+run.name);target=parent/run.name
    staging.mkdir();rsync_tree(run,staging)
    if tree_manifest(staging,limit)!=manifest:raise ValueError('published copy differs from closed native outputs')
    rename_noreplace(staging,target)
    directory=_directory(parent)
    try:os.fsync(directory)
    finally:os.close(directory)
    return manifest


def guarded_cleanup(plan):
    scratch,_work=_validate_plan(plan)
    if _regular_bytes(scratch/'.publication_verified',256)[0].decode()!=plan['token']:
        raise ValueError('publication marker differs')
    shutil.rmtree(scratch)



def _numeric_output_completeness(input_data, output_data, henry_data):
    """Finite final values for every native component, never convergence."""
    def finite(token):
        value=float(token)
        if not math.isfinite(value):raise ValueError('nonfinite native numeric result')
        return value
    lines=input_data.decode('utf-8').splitlines()
    gas_rows=[i for i,line in enumerate(lines)if line.strip()=='Kind_of_gas']
    temperature_rows=[i for i,line in enumerate(lines)if line.strip().startswith('Temperature(K)')]
    if len(gas_rows)!=1 or len(temperature_rows)!=1:raise ValueError('unbound native gas count or temperature')
    count=int(lines[gas_rows[0]+1].strip());temperature=finite(lines[temperature_rows[0]+1].split()[0])
    if not 1<=count<=4096 or temperature<=0:raise ValueError('invalid declared native components or temperature')
    densities={};headers=0;total=[]
    for line in output_data.decode('utf-8').splitlines():
        if 'Molecule'in line and 'Average_density'in line:
            if 'Average_density(mol/L)'not in line:raise ValueError('unknown native density unit')
            headers+=1;continue
        if not headers:continue
        match=re.fullmatch(r'\s*(\d+)\s+(\S+)\s+\((\S+)%\)\s+(\S+)\s*',line)
        if match:
            index=int(match[1]);density,percentage,maximum=map(finite,match.group(2,3,4))
            if index in densities or density<0 or maximum<0 or not 0<=percentage<=100:
                raise ValueError('duplicate or invalid final native density')
            densities[index]=density
        elif line.strip().startswith('total density:'):
            value=finite(line.strip().split(':',1)[1].strip())
            if value<0:raise ValueError('negative total native density')
            total.append(value)
    if headers!=1 or len(total)!=1 or set(densities)not in (set(range(count)),set(range(1,count+1))):
        raise ValueError('missing final native component densities or total density')
    henry={};seen_temperature=None;henry_headers=0;section_components=0
    for line in henry_data.decode('utf-8').splitlines():
        if not line.strip():continue
        parts=line.split()
        if 'unit:'in line:
            if len(parts)!=2 or parts[1]!='unit:mol/(L*atm)':
                raise ValueError('ambiguous or unknown native Henry unit')
            # Native mixtures repeat T/unit before each component. Also keep
            # the legacy one-header/all-components form; mixed or empty
            # sections cannot establish an unambiguous component binding.
            if henry_headers and section_components!=1:
                raise ValueError('ambiguous native Henry component headers')
            seen_temperature=finite(parts[0])
            if not math.isclose(seen_temperature,temperature,rel_tol=0,abs_tol=1e-8):
                raise ValueError('native Henry temperature differs from input')
            henry_headers+=1;section_components=0
        else:
            if seen_temperature is None or len(parts)!=2:raise ValueError('unbound native Henry component')
            index=int(parts[0]);value=finite(parts[1])
            if index in henry or value<0:raise ValueError('duplicate or invalid native Henry component')
            henry[index]=value
            section_components+=1
    if henry_headers>1 and section_components!=1:
        raise ValueError('ambiguous native Henry component headers')
    if set(henry)!=set(range(count)):raise ValueError('missing native Henry components')
    return {'version':2,'criterion':'finite_final_component_results',
        'components':count,'temperature_K':temperature,'density_unit':'mol/L',
        'henry_unit':'mol/(L*atm)','convergence_verified':False}


def execute_native_case(*,source,run,executable,job_id,timeout,stop,processes,lock):
    """One real child, one immutable outcome. A zero exit is not convergence."""
    if type(timeout) is not int or timeout<1:raise ValueError('positive per-case timeout required')
    if stop.is_set():return {'executed':False,'exit_code':None,'cancelled':True}
    source=Path(source);run=Path(run);executable=Path(executable)
    input_fact=file_fact(source,16*MB);solver_fact=file_fact(executable,256*MB)
    run.mkdir();_copy_regular(source,run/'input.dat',16*MB)
    started=time.monotonic();started_at=time.time();timed_out=False
    with (run/'solver.log').open('xb') as stdout,(run/'stderr.log').open('xb') as stderr:
        env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        with lock:
            if stop.is_set():return {'executed':False,'exit_code':None,'cancelled':True}
            process=subprocess.Popen([str(executable)],cwd=run,stdout=stdout,stderr=stderr,env=env,start_new_session=True)
            processes[source.name]=process
        try:
            termination_at=None
            while True:
                elapsed=time.monotonic()-started
                if termination_at is None and (elapsed>=timeout or stop.is_set()):
                    timed_out=elapsed>=timeout;termination_at=time.monotonic()
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                if termination_at is not None and time.monotonic()-termination_at>=10:
                    try:os.killpg(process.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                try:code=process.wait(timeout=.2);break
                except subprocess.TimeoutExpired:continue
        finally:
            with lock:processes.pop(source.name,None)
    receipt={'schema_version':1,'material':source.stem,'job_id':str(job_id),'exit_code':code,
        'timed_out':timed_out,'cancelled':stop.is_set(),'timeout_seconds':timeout,
        'started_at':started_at,'finished_at':time.time(),'wall_seconds':time.monotonic()-started,
        'input_sha256':input_fact['sha256'],'stdout_sha256':file_fact(run/'solver.log')['sha256'],
        'stderr_sha256':file_fact(run/'stderr.log')['sha256'],'solver_sha256':solver_fact['sha256'],
        'scientific_convergence':None}
    native_outputs={};output_errors=[];output_data={};validation=None
    for name in ('output.dat','hr.dat'):
        try:
            data,fact=_regular_bytes(run/name,MB)
            if not fact['size']:raise ValueError('required native output is empty')
            native_outputs[name]=fact;output_data[name]=data
        except (OSError,ValueError) as error:output_errors.append(name+': '+str(error))
    if len(output_data)==2:
        try:
            staged,staged_fact=_regular_bytes(run/'input.dat',16*MB)
            if staged_fact['sha256']!=input_fact['sha256']:raise ValueError('native staged input changed')
            validation=_numeric_output_completeness(staged,output_data['output.dat'],output_data['hr.dat'])
        except (ValueError,IndexError,OverflowError,UnicodeError,OSError)as error:
            output_errors.append('native numeric completeness: '+str(error))
    receipt.update(native_outputs=native_outputs,native_outputs_complete=not output_errors,
        native_output_errors=output_errors,native_output_validation=validation)
    immutable_bytes(run/'execution.json',_json_bytes(receipt))
    return receipt


def run_shared_batch(*,work_dir,input_dat_dir,executable,job_id,workers,timeout,collect_script,gases):
    """Shared storage remains default; failures cannot disappear behind xargs/echo."""
    work=Path(work_dir);inputs=Path(input_dat_dir);executable=Path(executable)
    for directory in (work,inputs):
        fd=_directory(directory);os.close(fd)
    if type(timeout) is not int or timeout<1:raise ValueError('positive per-case timeout required')
    workers=_allocated_workers(workers);file_fact(executable,256*MB)
    if not re.fullmatch(r'[1-9][0-9]*',str(job_id)):raise ValueError('numeric job ID required')
    names=sorted(p.name for p in inputs.iterdir() if p.suffix=='.dat')
    if not names or len(names)>512:raise ValueError('bounded nonempty native input inventory required')
    for name in names:file_fact(inputs/name,16*MB)
    if os.path.lexists(work/'results.csv') or os.path.lexists(work/'input_dir'):
        raise FileExistsError('existing native results are never replaced')
    immutable_bytes(work/('.native_execution_'+str(job_id)+'.json'),_json_bytes({'job_id':str(job_id),'inputs':names}))
    (work/'input_dir').mkdir()
    stop=threading.Event();processes={};lock=threading.RLock();outcomes={};errors=[]
    def terminated(signum,frame):
        stop.set()
        with lock:
            for process in processes.values():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
    handlers={}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):handlers[signum]=signal.signal(signum,terminated)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures={pool.submit(execute_native_case,source=inputs/name,run=work/'input_dir'/Path(name).stem,
                executable=executable,job_id=job_id,timeout=timeout,stop=stop,processes=processes,lock=lock):name for name in names}
            for future in as_completed(futures):
                try:outcomes[futures[future]]=future.result()
                except Exception as error:errors.append(futures[future]+': '+str(error))
        temporary=work/('.results_'+uuid.uuid4().hex+'.csv')
        with (work/'collection.stdout.log').open('xb') as out,(work/'collection.stderr.log').open('xb') as err:
            collection=subprocess.run([sys.executable,str(collect_script),'--work_dir',str(work),'--gases',*gases,
                '--output',str(temporary)],stdout=out,stderr=err,timeout=120)
        if collection.returncode:errors.append('native collector exited '+str(collection.returncode))
        if temporary.exists():
            immutable_bytes(work/'results.csv',_regular_bytes(temporary,16*MB)[0])
            temporary.unlink()  # exact task-owned temporary, published bytes already fsynced
        else:errors.append('collector produced no CSV')
        success=(not stop.is_set() and not errors and len(outcomes)==len(names) and
            all(item.get('exit_code')==0 and item.get('native_outputs_complete') and not item.get('timed_out')
                and not item.get('cancelled') for item in outcomes.values()))
        report={'schema_version':1,'mode':'shared','job_id':str(job_id),'exit_code':0 if success else 1,
            'outcomes':outcomes,'errors':errors,'expected_materials':[Path(name).stem for name in names],
            'scientific_convergence':None,'partial_results_retained':not success}
        immutable_bytes(work/'batch_execution.json',_json_bytes(report));return report
    finally:
        for signum,handler in handlers.items():signal.signal(signum,handler)


def run_prepared(plan,*,gases,collect_script,timeout):
    scratch,work=_validate_plan(plan)
    if type(timeout) is not int or timeout<1:raise ValueError('positive per-case timeout required')
    immutable_bytes(scratch/'.running',plan['token'].encode())
    if file_fact(scratch/'solver')['sha256']!=plan['executable_sha256']:raise ValueError('staged executable changed')
    _copy_regular(collect_script,scratch/'collect.py',4*MB)
    stop=threading.Event();cancelled=threading.Event();processes={};lock=threading.RLock();publisher=threading.Lock();errors=[];warnings=[];outcomes={}
    def periodic():
        while not stop.wait(plan['policy']['sync_seconds']):
            try:
                with publisher:sync_partial(plan)
            except Exception as error:warnings.append('periodic snapshot: '+str(error))
    background=threading.Thread(target=periodic,daemon=True);background.start()
    def terminated(signum,frame):
        cancelled.set()
        stop.set()
        with lock:
            for process in processes.values():
                if process.poll() is None:
                    try:os.killpg(process.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
    old_handlers={}
    if threading.current_thread() is threading.main_thread():
        for signum in (signal.SIGTERM,signal.SIGINT):old_handlers[signum]=signal.signal(signum,terminated)
    def execute(name):
        if stop.is_set():return {'executed':False,'exit_code':None,'cancelled':True}
        source=scratch/'input'/name;run=scratch/'input_dir'/Path(name).stem
        if file_fact(source,16*MB)!=plan['inputs'][name]:raise ValueError('staged native input changed')
        receipt=execute_native_case(source=source,run=run,executable=scratch/'solver',job_id=plan['job_id'],
            timeout=timeout,stop=stop,processes=processes,lock=lock)
        if receipt.get('executed') is False:return receipt
        with publisher:
            observed_tree_bytes(scratch,plan['policy']['capacity_mb']*MB)
            manifest=_publish_case(plan,run)
        return {**receipt,'manifest':manifest}
    try:
        with ThreadPoolExecutor(max_workers=plan['workers']) as pool:
            futures={pool.submit(execute,name):name for name in plan['inputs']}
            for future in as_completed(futures):
                try:outcomes[futures[future]]=future.result()
                except Exception as error:errors.append(futures[future]+': '+str(error))
        stop.set();background.join(timeout=125)
        if background.is_alive():raise RuntimeError('background sync did not stop; preserving scratch')
        collect_log=scratch/'collection.stdout.log';collect_err=scratch/'collection.stderr.log'
        with collect_log.open('xb') as out,collect_err.open('xb') as err:
            result=subprocess.run([sys.executable,str(scratch/'collect.py'),'--work_dir',str(scratch),'--gases',*gases,'--output',str(scratch/'results.csv')],stdout=out,stderr=err,timeout=120)
        if result.returncode:errors.append('native collector exited '+str(result.returncode))
        for name in ('results.csv','collection.stdout.log','collection.stderr.log'):
            source=scratch/name
            if source.exists():_copy_regular(source,work/name,16*MB)
        for name,item in outcomes.items():
            if item.get('manifest') and tree_manifest(work/'input_dir'/Path(name).stem,plan['policy']['capacity_mb']*MB)!=item['manifest']:
                errors.append(name+': shared publication changed; preserving local evidence')
        success=(not cancelled.is_set() and not errors and len(outcomes)==len(plan['inputs']) and all(item.get('exit_code')==0
            and item.get('native_outputs_complete') and not item.get('timed_out') and not item.get('cancelled')
            for item in outcomes.values()))
        report={'schema_version':1,'mode':'local','job_id':plan['job_id'],'scratch':str(scratch),'work_dir':str(work),
            'exit_code':0 if success else 1,'cancelled':cancelled.is_set(),'outcomes':outcomes,'errors':errors,'warnings':warnings,'cleanup_done':False,
            'scientific_convergence':None,'input_manifest':plan['inputs'],'filesystem':plan['filesystem']}
        immutable_bytes(work/('scratch_publication_'+plan['job_id']+'.json'),_json_bytes(report))
        if success and not cancelled.is_set():
            immutable_bytes(scratch/'.publication_verified',plan['token'].encode())
            if not cancelled.is_set():guarded_cleanup(plan);report['cleanup_done']=True
        if cancelled.is_set():
            report.update(exit_code=1,cancelled=True)
            immutable_bytes(work/('scratch_cancellation_'+plan['job_id']+'.json'),
                _json_bytes({'job_id':plan['job_id'],'cancelled':True,'exit_code':1,'cleanup_done':report['cleanup_done']}))
        return report
    except Exception as error:
        return {'mode':'local','job_id':plan['job_id'],'scratch':str(scratch),'work_dir':str(work),
            'exit_code':1,'cleanup_done':False,'errors':errors+[str(error)],'scientific_convergence':None}
    finally:
        stop.set();background.join(timeout=125)
        for signum,handler in old_handlers.items():signal.signal(signum,handler)


def dispatch_job(*,policy,collect_script,gases,timeout,**request):
    normalized=normalized_policy(policy)
    try:plan=prepare_job(policy=policy,**request)
    except ScratchUnavailable as error:
        if normalized and normalized['on_unavailable']=='shared':
            return {'mode':'shared','reason':str(error),'exit_code':78,'scientific_processes_started':0}
        raise
    if plan['mode']=='shared':return {**plan,'exit_code':78,'scientific_processes_started':0}
    # No fallback exists below this point, regardless of process exit status.
    return run_prepared(plan,gases=gases,collect_script=collect_script,timeout=timeout)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['dispatch','shared'])
    for name in ('work-dir','input-dat-dir','executable','policy','job-id','workers','memory-mb','collect-script','gases','timeout'):
        parser.add_argument('--'+name,required=name not in ('policy','memory-mb'))
    parser.add_argument('--expected-inputs');parser.add_argument('--expected-executable-sha256')
    args=parser.parse_args()
    try:
        common=dict(work_dir=Path(args.work_dir),input_dat_dir=Path(args.input_dat_dir),executable=Path(args.executable),
            job_id=args.job_id,workers=int(args.workers),collect_script=Path(args.collect_script),
            gases=json.loads(args.gases),timeout=int(args.timeout))
        if args.action=='shared':report=run_shared_batch(**common)
        else:report=dispatch_job(**common,
            policy=json.loads(args.policy),memory_mb=int(args.memory_mb),
            expected_inputs=json.loads(args.expected_inputs) if args.expected_inputs else None,
            expected_executable_sha256=args.expected_executable_sha256)
        print(json.dumps(report,sort_keys=True));return report['exit_code']
    except Exception as error:
        print(json.dumps({'mode':'not_executed_or_preserved_failure','error':str(error),'no_automatic_reexecution':True}),file=sys.stderr)
        return 2


if __name__=='__main__':raise SystemExit(main())
