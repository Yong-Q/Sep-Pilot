"""Trusted namespace launcher. Never imported/executed by generated plotting code.

Linux x86-64 only. A temporary directory without these OS protections is NOT a
sandbox. The privileged namespace parent alone copies checked PNGs to the host.
"""
import ctypes
import csv
import errno
import hashlib
import json
import io
import os
from pathlib import Path
import resource
import re
import signal
import stat
import subprocess
import sys
import time


def native_png_output_contract():
    """The trusted worker's actual enforced output bounds; no model override."""
    return {'schema_version':1,'format':'PNG','output_directory':'/output',
        'image_count':{'min':1,'max':8},'width_px':{'min':1,'max':4096},
        'height_px':{'min':1,'max':4096},'image_bytes':{'min':1,'max':16*1024**2},
        'regular_files_only':True,'allowed_suffixes':['.png'],'auxiliary_files_allowed':False}


def native_png_output_contract_sha256():
    return hashlib.sha256(json.dumps(native_png_output_contract(),sort_keys=True,
        separators=(',',':')).encode()).hexdigest()


def native_data_output_contract():
    return {'schema_version':1,'output_directory':'/output','file_count':{'min':1},
        'file_bytes':{'min':0,'max':8*1024**2},'total_bytes_max':16*1024**2,
        'allowed_suffixes':'any','regular_files_only':True,
        'csv_max_rows':50000,'csv_max_columns':128,'scientific_acceptance':False}


def _data_outputs(root,stage,result):
    """Validate bytes after the untrusted child exits; no child verdict is trusted."""
    contract=native_data_output_contract();names=list((root/'output').iterdir());total=0;accepted=[]
    if len(names)<contract['file_count']['min']:
        raise ValueError('sandbox output requires at least one file')
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError('duplicate JSON key')
            result[key]=value
        return result
    def invalid(value):raise ValueError('nonfinite JSON number')
    for path in sorted(names):
        st=path.lstat();suffix=path.suffix.lower()
        if (not path.name or path.name in {'.','..'} or '/' in path.name or '\x00' in path.name
                or not stat.S_ISREG(st.st_mode) or st.st_nlink!=1
                or not 0<=st.st_size<=contract['file_bytes']['max']):raise ValueError('invalid data output file')
        total+=st.st_size
        if total>contract['total_bytes_max']:raise ValueError('data outputs exceed total byte limit')
        raw=path.read_bytes()
        if len(raw)!=st.st_size:raise ValueError('invalid data output bytes')
        observed={'name':path.name,'format':'file','bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        accepted.append((path,raw,observed))
    target=stage/'rendered';target.mkdir(exist_ok=True)
    for path,raw,_ in accepted:
        with (target/path.name).open('xb') as stream:stream.write(raw)
    result['data_artifacts']=[item for _,_,item in accepted]
    result['output_validation']={'contract':contract,'scientific_acceptance':False}


def validate_png_observation(observed):
    """Validate only parent-observed format/size facts, never scientific claims."""
    contract=native_png_output_contract();issues=[]
    if observed.get('regular')is not True or str(observed.get('suffix','')).lower()not in contract['allowed_suffixes']:
        issues.append('bounded_regular_png_required')
    size=observed.get('bytes')
    if type(size)is not int or not contract['image_bytes']['min']<=size<=contract['image_bytes']['max']:
        issues.append('image_byte_limit')
    if observed.get('format')!='PNG':issues.append('png_format_required')
    for field in ('width_px','height_px'):
        value=observed.get(field)
        if type(value)is not int or not contract[field]['min']<=value<=contract[field]['max']:
            issues.append(field+'_limit')
    return {'passed':not issues,'observed':dict(observed),'allowed':contract,'issues':issues,
        'evidence_class':'native_output_format_observation','scientific_acceptance':False}


def mount(*args):
    # Direct syscall keeps the secure /proc/self/fd anchor in this process;
    # libmount path canonicalization in a fork can lose that anchor.
    libc=ctypes.CDLL(None,use_errno=True)
    libc.mount.argtypes=[ctypes.c_char_p,ctypes.c_char_p,ctypes.c_char_p,ctypes.c_ulong,ctypes.c_void_p]
    libc.mount.restype=ctypes.c_int
    if args[0]=='--make-rprivate':source=None;target=args[1];kind=None;flags=16384|262144;data=None
    elif args[0]=='--bind':source=args[1];target=args[2];kind=None;flags=4096;data=None
    elif args[0]=='-o':source=None;target=args[2];kind=None;flags=32|4096|1|2|4;data=None
    else:
        source=args[-2];target=args[-1];kind=args[1];flags=2|4|(8 if 'noexec' in args[3] else 0);data=','.join(x for x in args[3].split(',') if x.startswith('size='))
    enc=lambda x:None if x is None else os.fsencode(x)
    if libc.mount(enc(source),enc(target),enc(kind),ctypes.c_ulong(flags),enc(data))!=0:
        raise OSError(ctypes.get_errno(),f'namespace mount failed: {source!s} {target!s} {kind!s} {flags!s} {data!s}')


def bind(source,target):
    target.mkdir(parents=True,exist_ok=True)
    mount('--bind',source,target)
    mount('-o','remount,bind,ro,nosuid,nodev',target)


def secure_child(root,timeout,ready,mode='plot'):
    os.chroot(root);os.chdir('/output')
    resource.setrlimit(resource.RLIMIT_AS,(2*1024**3,2*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(timeout+1,timeout+2))
    image_limit=native_png_output_contract()['image_bytes']['max']
    resource.setrlimit(resource.RLIMIT_FSIZE,(image_limit,image_limit))
    resource.setrlimit(resource.RLIMIT_NOFILE,(128,128))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    libc=ctypes.CDLL(None,use_errno=True)
    for cap in range(64):libc.prctl(24,cap,0,0,0) # PR_CAPBSET_DROP
    class Header(ctypes.Structure):_fields_=[('version',ctypes.c_uint32),('pid',ctypes.c_int)]
    class Data(ctypes.Structure):_fields_=[('effective',ctypes.c_uint32),('permitted',ctypes.c_uint32),('inheritable',ctypes.c_uint32)]
    if libc.capset(ctypes.byref(Header(0x20080522,0)),(Data*2)())!=0:raise OSError('capability removal failed')
    if libc.prctl(38,1,0,0,0)!=0:raise OSError('no_new_privs unavailable')
    # No fork/exec/network/ptrace: one bounded Python process, not an OS shell.
    class Filter(ctypes.Structure):_fields_=[('code',ctypes.c_ushort),('jt',ctypes.c_ubyte),('jf',ctypes.c_ubyte),('k',ctypes.c_uint32)]
    class Program(ctypes.Structure):_fields_=[('len',ctypes.c_ushort),('filter',ctypes.POINTER(Filter))]
    ops=[Filter(0x20,0,0,4),Filter(0x15,1,0,0xc000003e),Filter(0x06,0,0,0x80000000),Filter(0x20,0,0,0),Filter(0x35,0,1,0x40000000),Filter(0x06,0,0,0x80000000)]
    ops.extend([Filter(0x15,0,4,56),Filter(0x20,0,0,16),Filter(0x45,1,0,0x10000),Filter(0x06,0,0,0x50000|errno.EPERM),Filter(0x06,0,0,0x7fff0000)])
    ops.extend([Filter(0x15,0,1,435),Filter(0x06,0,0,0x50000|errno.ENOSYS)])
    for number in [29,30,31,64,65,66,67,68,69,70,71,220,240,241,242,243,244,245,41,53,57,58,59,62,101,155,161,165,166,200,234,272,298,304,308,310,311,321,322,425,426,427]:
        ops.extend([Filter(0x15,0,1,number),Filter(0x06,0,0,0x50000|errno.EPERM)])
    ops.append(Filter(0x06,0,0,0x7fff0000));array=(Filter*len(ops))(*ops)
    if libc.prctl(22,2,ctypes.byref(Program(len(ops),array)),0,0)!=0:raise OSError('seccomp unavailable')
    # Host Python already started; remove host paths and close every extra fd.
    sys.path[:]=[p for p in sys.path if p.startswith(sys.prefix+'/')]
    os.write(ready,b'1')
    os.closerange(3,1048576)
    import runpy
    if mode=='plot':
        instrumentation=runpy.run_path('/code/native_plot_audit.py',run_name='__native_plot_audit__')
        instrumentation['install']()
    runpy.run_path('/code/script.py',run_name='__main__')


def main(stage,timeout,mode='plot'):
    if mode not in {'plot','data'}:raise ValueError('unsupported sandbox output mode')
    global STAGE_FD
    inherited=int(stage.rsplit('/',1)[-1])
    before=os.fstat(inherited)
    STAGE_FD=os.open(os.readlink(stage),os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    after=os.fstat(STAGE_FD)
    if (before.st_dev,before.st_ino)!=(after.st_dev,after.st_ino):raise ValueError('plot stage identity changed')
    os.close(inherited)
    stage=Path(f'/proc/self/fd/{STAGE_FD}');root=stage/'rootfs';root.mkdir()
    mount('--make-rprivate','/')
    os.chdir(root)
    mount('-t','tmpfs','-o','size=96m,nosuid,nodev','tmpfs','.')
    os.chdir(root)
    root=Path('.')
    for path in ['/usr','/lib','/lib64',sys.prefix]:
        if Path(path).exists():bind(path,root/path.lstrip('/'))
    bind(stage/'inputs',root/'inputs');bind(stage/'code',root/'code')
    for name,size in [('output','32m'),('tmp','32m')]:
        (root/name).mkdir();mount('-t','tmpfs','-o',f'size={size},nosuid,nodev,noexec','tmpfs',root/name)
    (root/'dev').mkdir()
    for name in ['null','urandom','zero']:
        (root/'dev'/name).touch();mount('--bind','/dev/'+name,root/'dev'/name)
    # Hide the host and the namespace supervisor even if generated code probes.
    libc=ctypes.CDLL(None);libc.prctl(4,0,0,0,0) # PR_SET_DUMPABLE
    ready_read,ready_write=os.pipe()
    pid=os.fork()
    if pid==0:
        try:
            os.close(ready_read)
            for fd,name in [(1,'stdout.log'),(2,'stderr.log')]:
                target=os.open(stage/name,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600);os.dup2(target,fd);os.close(target)
            os.environ.clear();os.environ.update({'HOME':'/tmp','MPLCONFIGDIR':'/tmp/matplotlib','MPLBACKEND':'Agg','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1','LANG':'C.UTF-8','PATH':'/usr/bin'})
            secure_child(str(root),timeout,ready_write,mode)
            sys.stdout.flush();sys.stderr.flush();os._exit(0)
        except BaseException:
            import traceback;traceback.print_exc();sys.stderr.flush();os._exit(1)
    os.close(ready_write)
    end=time.monotonic()+timeout;timed_out=False
    while True:
        done,status=os.waitpid(pid,os.WNOHANG)
        if done:break
        if time.monotonic()>end:
            timed_out=True;os.kill(pid,signal.SIGKILL);_,status=os.waitpid(pid,0);break
        time.sleep(.05)
    code=os.waitstatus_to_exitcode(status)
    ready=os.read(ready_read,2)==b'1';os.close(ready_read)
    result={'exit_code':code,'timed_out':timed_out,'images':[],'os_isolated':ready}
    # Runtime observations can be manipulated by generated Python. Retain
    # bounded diagnostics, never consume a child-supplied pass/ok/authority.
    diagnostics=[]
    try:
        with(stage/'stderr.log').open('rb')as stream:
            stream.seek(0,2);size=stream.tell();stream.seek(max(0,size-65536));tail=stream.read(65536).decode('utf-8',errors='replace')
        for line in tail.splitlines():
            if not line.startswith('SEPPILOT_RENDER_DIAGNOSTIC '):continue
            try:
                data=json.loads(line[len('SEPPILOT_RENDER_DIAGNOSTIC '):],parse_constant=lambda _:None)
                if not isinstance(data,dict):continue
                sanitized={k:data[k]for k in('axes','blocking_diagnostics','notes')if k in data}
                sanitized.update(evidence_class='in_process_runtime_advisory',independently_verified=False,scientific_acceptance=False)
                diagnostics.append(sanitized)
            except (ValueError,TypeError):continue
        result['render_diagnostics']=diagnostics[-8:]
    except OSError:pass
    if code==0 and not timed_out:
        if mode=='data':
            try:_data_outputs(root,stage,result)
            except (OSError,ValueError,UnicodeError,csv.Error,RecursionError) as error:
                result['output_error']=str(error)[:1000]
            print(json.dumps(result),flush=True);return
        from PIL import Image
        names=list((root/'output').iterdir());contract=native_png_output_contract()
        result['output_validation']={'contract':contract,'contract_sha256':native_png_output_contract_sha256(),
            'observed_image_count':len(names),'checked_images':[],'scientific_acceptance':False}
        if not contract['image_count']['min']<=len(names)<=contract['image_count']['max']:
            result['output_error']=f"require {contract['image_count']['min']}..{contract['image_count']['max']} PNG outputs only; observed {len(names)}"
            print(json.dumps(result),flush=True);return
        validated=[]
        try:
            for path in sorted(names):
                s=path.lstat();observed={'name':path.name,'regular':stat.S_ISREG(s.st_mode),
                    'suffix':path.suffix.lower(),'bytes':s.st_size}
                if (observed['regular']is not True or observed['suffix']not in contract['allowed_suffixes']
                        or not contract['image_bytes']['min']<=s.st_size<=contract['image_bytes']['max']):
                    result['output_validation']['checked_images'].append(validate_png_observation(observed))
                    raise ValueError('invalid plot output: only bounded regular PNG permitted; observed '+json.dumps(observed))
                with Image.open(path)as image:
                    observed.update(format=image.format,width_px=image.width,height_px=image.height)
                    checked=validate_png_observation(observed)
                    result['output_validation']['checked_images'].append(checked)
                    if not checked['passed']:
                        raise ValueError(f"PNG dimensions invalid: observed {image.width}x{image.height} px; allowed width {contract['width_px']['min']}..{contract['width_px']['max']}, height {contract['height_px']['min']}..{contract['height_px']['max']}; format {contract['format']}")
                    image.verify()
                validated.append(path)
        except (OSError,ValueError)as error:
            # Isolation succeeded and the child returned. This is a generated
            # output-contract failure, not a missing/unsafe sandbox backend.
            result['output_error']=str(error)[:1000]
            print(json.dumps(result),flush=True);return
        target=stage/'rendered';target.mkdir(exist_ok=True)
        for path in validated:
            raw=path.read_bytes()
            with (target/path.name).open('xb')as out:out.write(raw)
            result['images'].append(path.name)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    try:main(sys.argv[1],int(sys.argv[2]),sys.argv[3] if len(sys.argv)>3 else 'plot')
    except BaseException as exc:
        import traceback;traceback.print_exc()
        if isinstance(exc,subprocess.CalledProcessError):print(exc.stderr.decode(errors='replace'),file=sys.stderr)
        print(json.dumps({'sandbox_error':type(exc).__name__}),flush=True);sys.exit(70)
