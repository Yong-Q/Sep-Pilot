"""Owned immutable scientific downloads and bounded document text reading."""
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid
from urllib.parse import urljoin

from .report_recovery import _OwnedFiles


def _owner_root(config,create=False):
    from .watch_context import get_context
    from .workspace import validate_scope_component
    context=get_context();owner={key:validate_scope_component(context.get(key)or'')for key in ('username','conv_id')}
    project=Path(config.project_root).absolute()
    if project.resolve()!=project:raise PermissionError('document project root must not be linked')
    fd=os.open(project,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:
        for part in ('runs',owner['username'],owner['conv_id']):
            if create:
                try:os.mkdir(part,0o700,dir_fd=fd)
                except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
    finally:os.close(fd)
    return project/'runs'/owner['username']/owner['conv_id'],owner


def _parent(files,path,create=False):
    path=files.owned(path);parts=path.relative_to(files.root).parts;fd=os.dup(files.fd)
    try:
        for part in parts[:-1]:
            if create:
                try:os.mkdir(part,0o700,dir_fd=fd)
                except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=fd)
            os.close(fd);fd=child
        return fd,parts[-1]
    except BaseException:os.close(fd);raise


def _read_bytes(files,path,limit):
    parent,name=_parent(files,path);fd=None
    try:
        fd=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=parent)
        first=os.fstat(fd)
        if not stat.S_ISREG(first.st_mode)or not 0<first.st_size<=limit:
            raise ValueError('bounded nonempty regular document required')
        blocks=[];size=0;digest=hashlib.sha256()
        while True:
            block=os.read(fd,65536)
            if not block:break
            size+=len(block)
            if size>limit:raise ValueError('document grew beyond byte limit')
            blocks.append(block);digest.update(block)
        last=os.fstat(fd);linked=os.stat(name,dir_fd=parent,follow_symlinks=False)
        identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns,value.st_mode)
        if size!=first.st_size or identity(first)!=identity(last)or identity(last)!=identity(linked):
            raise ValueError('document changed during read')
        return b''.join(blocks),{'path':str(files.owned(path)),'bytes':size,'sha256':digest.hexdigest()}
    finally:
        if fd is not None:os.close(fd)
        os.close(parent)


def download_scientific_file(arguments,config):
    from .scientific_http import operator_network,open_public,close_response
    root,owner=_owner_root(config,create=True)
    url=str(arguments['url']).strip();current=url;max_bytes=int(arguments.get('max_bytes',20*1024**2))
    timeout=float(arguments.get('timeout_seconds',30));expected=arguments.get('expected_sha256','').lower()
    if not 1<=max_bytes<=100*1024**2 or not 1<=timeout<=120:raise ValueError('bounded download size/time required')
    if expected and not re.fullmatch('[0-9a-f]{64}',expected):raise ValueError('valid expected SHA256 required')
    target=Path(arguments['output_path']).expanduser()
    if not target.is_absolute():target=Path(config.project_root)/target
    target=Path(os.path.abspath(target))
    if not target.is_relative_to(root)or target==root:raise PermissionError('download output escapes its owned session')
    files=_OwnedFiles(root,owner);parent=None;temporary=None;response=manager=None;redirects=[];hops=[]
    deadline=time.monotonic()+timeout
    try:
        network=operator_network(config.project_root)
        parent,name=_parent(files,target,create=True)
        try:os.stat(name,dir_fd=parent,follow_symlinks=False)
        except FileNotFoundError:pass
        else:raise FileExistsError('download destination already exists; select a fresh owned path')
        for _ in range(6):
            response,manager,hop=open_public(current,network,deadline);hops.append(hop)
            if response.status in {301,302,303,307,308}:
                location=response.headers.get('Location');close_response(response,manager);response=manager=None
                if not location:raise ValueError('redirect omitted Location')
                current=urljoin(current,location);redirects.append(current);continue
            break
        else:raise ValueError('too many scientific download redirects')
        if response.status!=200:raise ValueError('scientific download HTTP '+str(response.status))
        declared=response.headers.get('Content-Length')
        if declared and int(declared)>max_bytes:raise ValueError('declared download exceeds max_bytes')
        temporary='.download-'+uuid.uuid4().hex
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent)
        size=0;digest=hashlib.sha256()
        with os.fdopen(fd,'wb')as output:
            for block in response.stream(65536,decode_content=True):
                if time.monotonic()>deadline:raise TimeoutError('scientific download time budget exhausted')
                size+=len(block)
                if size>max_bytes:raise ValueError('download exceeds max_bytes')
                digest.update(block);output.write(block)
            output.flush();os.fsync(output.fileno())
        actual=digest.hexdigest()
        if not size:raise ValueError('scientific download was empty')
        if expected and actual!=expected:raise ValueError('download SHA256 differs from expected source')
        # A new hard link is an immutable publication: an existing result is
        # never overwritten, even if it appeared after the earlier check.
        os.link(temporary,name,src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
        os.fsync(parent);os.unlink(temporary,dir_fd=parent);temporary=None
        return {'status':'success','source_url':url,'final_url':current,'redirects':redirects,
            'http_status':response.status,'content_type':response.headers.get('Content-Type',''),
            'bytes':size,'sha256':actual,'output_path':str(target),'output_files':[str(target)],
            'transport_hops':hops,'retrieved_at':time.time(),'existing_results_overwritten':False,'fulltext_or_paper_identity_verified':False}
    except Exception as error:
        receipt={'error':str(error)[:1000],'error_kind':'scientific_download_failed',
            'source_url':url,'final_url':current,'redirects':redirects,'transport_hops':hops,
            'output_published':False,'existing_results_overwritten':False}
        if temporary and parent is not None:
            # Keep partial bytes as explicit failure evidence, not a final PDF.
            receipt['partial_evidence_path']=str(target.parent/temporary)
        return receipt
    finally:
        if response is not None:close_response(response,manager)
        if parent is not None:os.close(parent)
        files.close()


def observed_document_source_paths(contract, result, project_root, native_call):
    """Attest an unchanged input observed by the actual native reader.

    An input PDF/HTML is not a newly produced artifact. Its native read receipt,
    bounded extraction, current bytes and exact owned input contract must agree.
    This does not excuse a missing derived output or approve document claims.
    """
    from .watch_context import get_context
    from .report_recovery import _OwnedFiles, _same
    from .recovery import result_object
    from .literature_report_bridge import attach_native_document
    from .output_contract import normalize_output
    files = None
    try:
        if contract.get('tool') != 'read_scientific_document' or not isinstance(native_call, dict):
            return set()
        arguments = contract.get('arguments') or {}
        owner = get_context()
        if not owner.get('username') or not owner.get('conv_id'):
            return set()
        root = Path(project_root).resolve() / 'runs' / owner['username'] / owner['conv_id']
        files = _OwnedFiles(root, owner)
        requested = arguments.get('path')
        if (not isinstance(requested, str) or not Path(requested).is_absolute()
                or '..' in Path(requested).parts):
            return set()
        source = files.owned(requested)
        if str(source) != requested:
            return set()
        call_id = native_call.get('call_id')
        if (not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', call_id)
                or Path(native_call.get('evidence_path', '')) != root / 'evidence' / (call_id + '.json')):
            return set()
        call, _ = files.read(native_call['evidence_path'], as_json=True)
        obj = result_object(result)
        original = obj.get('source_document')
        if (not isinstance(call, dict) or not isinstance(original, dict)
                or call.get('call_id') != call_id or call.get('tool') != contract['tool']
                or call.get('failed') is not False or not _same(call.get('params'), arguments)
                or not _same(result_object(call.get('result')), obj)
                or original.get('path') != requested):
            return set()
        parser = Path(__file__).with_name('scientific_document_worker.py')
        if (parser.is_symlink() or parser.stat().st_size > 1024 * 1024
                or obj.get('parser_implementation_sha256') != hashlib.sha256(parser.read_bytes()).hexdigest()):
            return set()
        summary = {}; facts = []
        attach_native_document(root, obj, summary, {}, facts, None)
        if len(facts) != 1 or root / summary['native_document']['path'] != source:
            return set()
        for output in contract.get('expected_outputs') or []:
            spec = normalize_output(output)
            if spec['kind'] != 'file' or files.owned(spec['path']) != source:
                return set()
        return {str(source)}
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return set()
    finally:
        if files is not None:
            files.close()


def read_scientific_document(arguments,config):
    root,owner=_owner_root(config);files=_OwnedFiles(root,owner)
    try:
        path=Path(arguments['path']);path=path if path.is_absolute()else Path(config.project_root)/path
        raw,fact=_read_bytes(files,path,32*1024**2)
        expected=arguments.get('expected_sha256')
        if expected and expected.lower()!=fact['sha256']:raise ValueError('document SHA256 differs from expected source')
        start=int(arguments.get('start_page',1));pages=int(arguments.get('max_pages',20));chars=int(arguments.get('max_chars',60000))
        if not(1<=start<=10000 and 1<=pages<=100 and 100<=chars<=200000):raise ValueError('bounded document pages/characters required')
        worker=Path(__file__).with_name('scientific_document_worker.py')
        import importlib.util
        parser=importlib.util.find_spec('pypdf')
        parser_directory=str(Path(parser.origin).resolve().parent.parent)if parser and parser.origin else''
        # Fixed trusted parser only. No shell, user code, embedded PDF actions,
        # OCR/browser or network imports. Limits are installed inside the child.
        environment={'PATH':'/usr/bin:/bin','LANG':'C.UTF-8','OMP_NUM_THREADS':'1',
            'OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1'}
        process=subprocess.run([sys.executable,'-I','-B',str(worker),str(start),str(pages),str(chars),parser_directory],
            input=raw,capture_output=True,timeout=15,env=environment)
        if process.returncode!=0:
            raise ValueError('bounded document parser failed: '+process.stderr.decode(errors='replace')[-800:])
        if len(process.stdout)>1024*1024:raise ValueError('document parser output exceeds bound')
        result=json.loads(process.stdout)
        expected_format=arguments.get('format','auto')
        if expected_format!='auto'and result.get('document_type')not in{None,expected_format}:
            raise ValueError('actual document format differs from requested format')
        if result.get('error'):return {**result,'source_document':fact,'scientific_claims_verified':False}
        _,after=_read_bytes(files,path,32*1024**2)
        if after['sha256']!=fact['sha256']:raise ValueError('document source changed during text extraction')
        if arguments.get('source_url'):
            from urllib.parse import urlsplit,urlunsplit
            for link in result.get('links',[]):
                resolved=urlsplit(urljoin(arguments['source_url'],link['href']))
                if resolved.scheme in {'http','https'}and resolved.hostname and not resolved.username and not resolved.password:
                    link['resolved_url']=urlunsplit((resolved.scheme,resolved.netloc,resolved.path,resolved.query,''))
        result.update(source_document=fact,source_url_declared=arguments.get('source_url',''),
            source_origin_verified=False,scientific_claims_verified=False,
            parser_implementation_sha256=hashlib.sha256(worker.read_bytes()).hexdigest())
        return result
    except (OSError,ValueError,KeyError,TypeError,subprocess.TimeoutExpired)as error:
        return {'error':str(error)[:1200],'error_kind':'scientific_document_read_failed','scientific_claims_verified':False}
    finally:files.close()
