"""Content-bound final review source guard. No runtime calls occur under a lock.

Capture is lock-free and independently rechecks native inspection. The sealed
list is not execution/acceptance authority: the caller still owns goal, attempt,
lease and final Store CAS. No protection against an arbitrary hostile same-UID
writer's multi-file atomicity is claimed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import stat
from .workload_budget import _owned_fd

MAX_FILES=2048
MAX_BYTES=384*1024*1024
MAX_FILE_BYTES=256*1024*1024
MAX_ENTRIES=8192


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def _path(root,value):
    if not isinstance(value,str):raise ValueError('review source requires an owned absolute path')
    p=Path(value)
    if not p.is_absolute()or'..'in p.parts or p==root or not p.is_relative_to(root):
        raise ValueError('review source escapes owned root')
    return p


def _identity(info):
    return (info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns,info.st_mode)


def _file(root,path,budget,*,identities=None):
    fd=_owned_fd(root,path)
    try:
        before=os.fstat(fd)
        if not stat.S_ISREG(before.st_mode)or not 0<=before.st_size<=MAX_FILE_BYTES or budget[0]+before.st_size>MAX_BYTES:
            raise ValueError('review source exceeds bounded regular-file closure')
        count=0;digest=hashlib.sha256()
        while True:
            chunk=os.read(fd,256*1024)
            if not chunk:break
            count+=len(chunk)
            if count>before.st_size or budget[0]+count>MAX_BYTES:raise ValueError('review source grew')
            digest.update(chunk)
        after=os.fstat(fd)
    finally:os.close(fd)
    visible=_owned_fd(root,path)
    try:linked=os.fstat(visible)
    finally:os.close(visible)
    if count!=before.st_size or _identity(before)!=_identity(after)or _identity(after)!=_identity(linked):
        raise ValueError('review source changed during content hashing')
    budget[0]+=count
    if identities is not None:identities.append((path,False,_identity(after)))
    return {'path':str(path),'sha256':digest.hexdigest(),'size':count}


def _members(root,path,*,identities=None):
    fd=_owned_fd(root,path,directory=True)
    try:
        before=os.fstat(fd);names=[]
        with os.scandir(fd)as entries:
            for entry in entries:
                if len(names)>=MAX_ENTRIES:raise ValueError('review directory membership exceeds bound')
                names.append(entry.name)
        after=os.fstat(fd)
    finally:os.close(fd)
    visible=_owned_fd(root,path,directory=True)
    try:linked=os.fstat(visible)
    finally:os.close(visible)
    if _identity(before)!=_identity(after)or _identity(after)!=_identity(linked):raise ValueError('review directory changed')
    if identities is not None:identities.append((path,True,_identity(after)))
    return sorted(names)


def capture_review_sources(runtime,binding,native_receipts,evidence_facts):
    """Capture the exact previously reviewed native bytes, outside Store locks."""
    root=Path(runtime.root).absolute()
    if root.resolve(strict=True)!=root or(root.parent.name,root.name)!=(runtime.username,runtime.conv_id):
        raise ValueError('review root is not the owned conversation')
    state=runtime.snapshot();node=state['nodes'][binding['step_id']]
    if (node.get('token')!=binding['token']or node.get('attempt_id')!=binding['attempt_id']
            or _digest(node['contract'])!=binding['contract_sha256']or node.get('job_ids')!=binding['job_ids']):
        raise ValueError('review source attempt changed before capture')
    expected={};directories=set();absent=set()
    def add(value,fact):
        p=_path(root,str(value));size=fact.get('size',fact.get('bytes'));sha=fact.get('sha256')
        if type(size)is not int or size<0 or not isinstance(sha,str)or not re.fullmatch('[a-f0-9]{64}',sha):
            raise ValueError('native review source has no content binding')
        value={'path':str(p),'sha256':sha,'size':size}
        if str(p)in expected and expected[str(p)]!=value:raise ValueError('conflicting reviewed source bytes')
        expected[str(p)]=value
        if len(expected)>MAX_FILES:raise ValueError('review source file inventory exceeds bound')
    def walk(value,depth=0):
        if depth>32:raise ValueError('review source shape exceeds bound')
        if isinstance(value,dict):
            if 'path'in value and 'sha256'in value and('size'in value or'bytes'in value):
                # Native input inventories use relative filenames; their anchor
                # is supplied explicitly by the cDFT branch below.
                if isinstance(value['path'],str)and Path(value['path']).is_absolute():add(value['path'],value)
            if value.get('directory')is True and isinstance(value.get('path'),str):directories.add(str(_path(root,value['path'])))
            for key,item in value.items():
                if isinstance(key,str)and Path(key).is_absolute()and isinstance(item,dict)and'sha256'in item:
                    add(key,item)
                walk(item,depth+1)
        elif isinstance(value,list):
            for item in value:walk(item,depth+1)
    for fact in evidence_facts:add(fact['path'],fact)
    tool=node['contract']['tool'];args=node.get('effective_arguments')or node['contract'].get('arguments',{})
    if not isinstance(native_receipts,list)or not 1<=len(native_receipts)<=8:raise ValueError('native review receipt required')
    for raw in native_receipts:
        fresh=runtime.inspect_workflow_result(binding['step_id'],expected_token=binding['token'])
        if _digest(fresh)!=_digest(raw):raise ValueError('reviewed native facts changed before source capture')
        walk(raw.get('method_facts')or{})
        if tool in {'run_gcmc_isotherm','run_gcmc_batch'}:
            proof=runtime._gcmc_result_repair_proof(binding['step_id'],binding['token'],inspection=True)
            if hashlib.sha256(json.dumps(proof,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()!=raw.get('provenance_sha256'):raise ValueError('native GCMC proof changed during source capture')
            walk(proof)
            if tool=='run_gcmc_batch':directories.add(str(_path(root,args['cif_dir'])))
        elif tool=='run_cdft'and args.get('action')=='submit':
            from .cdft_partial_recovery import prove_completed_attempt
            jobs={str(job):runtime.job_watch.get(str(job))or{}for job in node['job_ids']}if runtime.job_watch else{}
            attempt=prove_completed_attempt(state,binding['step_id'],root,runtime.main.goal_contract.to_dict(),jobs,inspection=True);walk(attempt)
            quality=raw['method_facts']['cdft_native'];work=_path(root,attempt['work_dir']);source=_path(root,attempt['input_dir'])
            for item in quality.get('original_input_facts',{}).get('inputs',[]):
                name=item.get('path')
                if not isinstance(name,str)or Path(name).name!=name:raise ValueError('invalid original input identity')
                add(str(source/name),item)
            for material in quality['expected_materials']:
                if not isinstance(material,str)or Path(material).name!=material or material in {'.','..'}:raise ValueError('invalid material identity')
                facts=quality['materials'][material]['artifacts']
                run=work/'input_dir'/material
                try:run_fd=_owned_fd(root,run,directory=True)
                except FileNotFoundError:absent.add(str(run))
                else:os.close(run_fd);directories.add(str(run))
                for name in ('input.dat','output.dat','hr.dat','solver.log','stderr.log','execution.json'):
                    p=work/'input_dir'/material/name
                    if str(p)not in facts:absent.add(str(p))
            csvpath=work/'results.csv';fact=_file(root,csvpath,[0])
            if fact['sha256']!=quality['collected_csv_sha256']:raise ValueError('reviewed CSV changed')
            add(str(csvpath),fact)
            directories.update(map(str,(source,work/'input',work/'input_dir')))
        else:raise ValueError('unsupported native review source closure')
    budget=[0];captured=[]
    for name,fact in sorted(expected.items()):
        observed=_file(root,Path(name),budget)
        if observed!=fact:raise ValueError('reviewed source changed during capture')
        captured.append(fact)
    if len(directories)>MAX_FILES or len(absent)>MAX_FILES:raise ValueError('review source membership exceeds bound')
    memberships={name:_members(root,Path(name))for name in sorted(directories)}
    root_info=root.stat()
    sealed={'schema_version':1,'kind':'owner_review_source_closure','root':str(root),
        'owner':[runtime.username,runtime.conv_id],'root_identity':[root_info.st_dev,root_info.st_ino],
        'binding':copy.deepcopy(binding),'files':captured,'absent':sorted(absent),'memberships':memberships}
    sealed['sha256']=_digest(sealed)
    if not review_sources_unchanged(root,sealed):raise ValueError('native sources changed before sealing')
    # Membership and metadata discovery are lock-free; compare the complete
    # original native digest after capture rather than adopting a new baseline.
    for raw in native_receipts:
        if _digest(runtime.inspect_workflow_result(binding['step_id'],expected_token=binding['token']))!=_digest(raw):
            raise ValueError('native result changed while capturing review closure')
    return sealed


def review_sources_unchanged(root,sealed):
    """Pure bounded file SHA/absence/membership checks; no runtime or file locks."""
    try:
        root=Path(root).absolute();proof=copy.deepcopy(sealed);sha=proof.pop('sha256')
        if sha!=_digest(proof)or proof.get('kind')!='owner_review_source_closure'or proof.get('schema_version')!=1:
            return False
        if proof['root']!=str(root)or root.resolve(strict=True)!=root or [root.parent.name,root.name]!=proof['owner']:return False
        identity=lambda:[root.stat().st_dev,root.stat().st_ino]
        if identity()!=proof['root_identity']:return False
        facts=proof['files'];absent=proof['absent'];members=proof['memberships']
        if not 1<=len(facts)<=MAX_FILES or len(absent)>MAX_FILES or len(members)>MAX_FILES:return False
        budget=[0];identities=[]
        for fact in facts:
            if _file(root,_path(root,fact['path']),budget,identities=identities)!=fact:return False
        for name in absent:
            try:fd=_owned_fd(root,_path(root,name))
            except FileNotFoundError:continue
            else:os.close(fd);return False
        for name,names in members.items():
            if _members(root,_path(root,name),identities=identities)!=names:return False
        # Content hashes authorize the bytes. This final linked identity pass
        # catches ordinary writes/renames during later files' hash/membership I/O.
        for path,directory,expected in identities:
            fd=_owned_fd(root,path,directory=directory)
            try:
                if _identity(os.fstat(fd))!=expected:return False
            finally:os.close(fd)
        return identity()==proof['root_identity']
    except (OSError,ValueError,TypeError,KeyError,AttributeError):return False
