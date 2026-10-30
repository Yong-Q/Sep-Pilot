"""Shared production/SDK scientific-audit policy, separate from resource monitoring."""
import hashlib
import json
from pathlib import Path

SCIENCE_TOOLS = frozenset({
    'discover_forcefield', 'inspect_forcefield', 'validate_framework_charges',
    'convert_physical_units', 'query_literature', 'read_scientific_document',
})


def is_source_result(tool, result):
    """A measured mismatch is valid negative evidence, not a failed read."""
    if not isinstance(result, dict): return False
    if tool == 'read_scientific_document':
        source=result.get('source_document')or{}
        blocks=result.get('pages')if result.get('document_type')=='pdf'else result.get('blocks')
        return (result.get('ok')is True and result.get('status')=='text_extracted'
            and result.get('embedded_actions_executed')is False
            and isinstance(source.get('path'),str)and bool(source.get('sha256'))
            and isinstance(blocks,list)and any(isinstance(block,dict)and block.get('text')for block in blocks))
    if tool == 'query_literature':
        if result.get('bibliographic_only') is True or result.get('evidence_complete') is False:
            return False
        # Some native web fallbacks mark metadata 'complete'. Inspect actual
        # content and its source identity; titles/DOIs alone verify no claim.
        hits = []
        for key in ('results', 'local', 'web', 'hits', 'matches'):
            values = result.get(key)
            if isinstance(values, list): hits.extend(values[:128])
        def source_content(hit):
            return isinstance(hit, dict) and any(isinstance(hit.get(key), str) and hit[key].strip()
                for key in ('claim', 'text', 'content', 'abstract')) and any(
                isinstance(hit.get(key), str) and hit[key].strip()
                for key in ('source_url', 'url', 'doi', 'citation', 'source', 'document_id', 'file_path'))
        if not any(source_content(hit) for hit in hits):
            return False
    if tool == 'validate_framework_charges':
        return (result.get('read_only') is True and result.get('status') in {'validated','charge_mismatch'}
                and bool(result.get('sha256')) and 'cell_net_charge' in result)
    return result.get('ok') is True and not result.get('error')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def canonical_source_inputs(items):
    """Repeated reads/order do not change inputs; distinct hashes remain distinct."""
    return [{'path': path, 'sha256': checksum} for path, checksum in
            sorted({(item['path'], item['sha256']) for item in items})]


def knowledge_fingerprint(root):
    root = Path(root).resolve()
    files = {}
    for relative in ('forcefields/towhee/catalog.json', 'forcefields/towhee/knowledge.json', 'env/forcefield_sources.json', 'env/physical_units.json'):
        path = root / relative
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None
    return digest(files)


def review_action(state, receipt):
    """No resubmission grant. A reviewer outage is not a main-task failure."""
    state['last_receipt'] = receipt
    if receipt.get('review_status') == 'verified':
        state['status'] = 'verified'
        return 'verified'
    if receipt.get('review_status') == 'insufficient_evidence':
        if state.get('verification_retries', 0) < 1:
            state['verification_retries'] = state.get('verification_retries', 0) + 1
            state['status'] = 'retrying_supervisor'
            return 'retry_supervisor'
    elif receipt.get('review_status') == 'rejected_content' and state.get('corrections', 0) < 1:
        state['corrections'] = state.get('corrections', 0) + 1
        state['status'] = 'correcting_main'
        return 'correct_main'
    state['status'] = 'needs_user'
    return 'ask_user'


EVIDENCE_FAMILIES = {
    'validate_framework_charges': 'charges', 'inspect_forcefield': 'forcefield',
    'discover_forcefield': 'forcefield', 'convert_physical_units': 'units',
    'query_literature': 'literature', 'read_scientific_document': 'document_text',
}
FAMILY_TOOLS = {
    'charges': 'validate_framework_charges', 'forcefield': 'inspect_forcefield',
    'units': 'convert_physical_units', 'literature': 'query_literature',
    'native_results': 'read_file', 'document_text': 'read_scientific_document',
}


def _owned_native_json(path, evidence_root, max_bytes=2 * 1024 * 1024):
    """Bounded, regular, no-symlink native dossier read; never a tool execution."""
    import os
    import stat
    root = Path(evidence_root).absolute()
    path = Path(path).absolute()
    relative = path.relative_to(root)
    if len(relative.parts) != 1 or '..' in relative.parts:
        raise ValueError('unowned native evidence')
    current = root
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError('symlink native evidence')
    directory = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        fd = os.open(relative.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
    except BaseException:
        os.close(directory)
        raise
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= max_bytes:
            raise ValueError('native dossier exceeds bounded regular-file limit')
        raw = b''
        while len(raw) <= before.st_size:
            chunk = os.read(fd, min(65536, before.st_size + 1 - len(raw)))
            if not chunk: break
            raw += chunk
        after = os.fstat(fd)
        visible = os.stat(relative.name, dir_fd=directory, follow_symlinks=False)
        if (after.st_ino, after.st_size, after.st_mtime_ns) != (visible.st_ino, visible.st_size, visible.st_mtime_ns):
            raise ValueError('native evidence path replaced')
        if len(raw) != before.st_size or (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('native evidence changed')
        return json.loads(raw), hashlib.sha256(raw).hexdigest()
    finally:
        os.close(fd)
        os.close(directory)


def _artifact_reference_aliases(node, dossier, scope):
    """Exact owned artifact identities, including displayed scope-relative paths."""
    candidates=list((node.get('artifacts') or {}).keys())+list(dossier.get('successful_artifacts') or [])
    for spec in node.get('contract',{}).get('expected_outputs',[]):
        candidates.append(spec.get('path') if isinstance(spec,dict) else spec)
    identities=[]
    for value in candidates:
        if not isinstance(value,str):continue
        path=Path(value)
        if not path.is_absolute() or not path.is_relative_to(scope):continue
        identities.extend([str(path),str(Path(scope.name)/path.relative_to(scope))])
    return identities


def _owned_artifact_observations(node, evidence_root, remaining):
    """No typed physics is inferred from a regular file or a model verdict."""
    import os
    from .recovery import result_object
    scope=Path(evidence_root).parent
    owner={'username':scope.parent.name,'conv_id':scope.name}
    if not node.get('job_ids'):return []
    ref=node.get('result_ref') or {};call_id=ref.get('call_id')
    if not isinstance(call_id,str) or not call_id.isalnum():return []
    producer_path=Path(evidence_root)/(call_id+'.json')
    if ref.get('evidence_path') != str(producer_path):return []
    producer,_=_owned_native_json(producer_path,evidence_root,min(2*1024*1024,remaining))
    if not isinstance(producer,dict):return []
    result=result_object(producer.get('result'))
    if not isinstance(result,dict):return []
    job_ids=result.get('job_ids') or ([result['job_id']] if result.get('job_id') else [])
    if (producer.get('call_id')!=call_id or producer.get('failed') is not False
            or producer.get('tool')!=node.get('contract',{}).get('tool')
            or set(map(str,job_ids))!=set(map(str,node.get('job_ids',[])))):return []
    from .workload_budget import _owned_fd
    observations=[]
    for name,fact in list((node.get('artifacts') or {}).items())[:32]:
        if not isinstance(fact,dict) or not 0<fact.get('size',0)<=min(remaining,2*1024*1024):continue
        try:
            path=Path(name);fd=_owned_fd(scope,path)
            try:
                before=os.fstat(fd)
                if before.st_size>min(remaining,2*1024*1024):continue
                raw=b''
                while len(raw)<=before.st_size:
                    chunk=os.read(fd,min(65536,before.st_size+1-len(raw)))
                    if not chunk:break
                    raw+=chunk
                after=os.fstat(fd)
            finally:os.close(fd)
            linked=_owned_fd(scope,path)
            try:visible=os.fstat(linked)
            finally:os.close(linked)
            identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
            if (len(raw)!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(visible)
                    or fact['size']!=len(raw) or fact.get('mtime_ns',after.st_mtime_ns)!=after.st_mtime_ns):continue
            checksum=hashlib.sha256(raw).hexdigest();witness=None
            if fact.get('sha256'):
                if fact['sha256']!=checksum:continue
            else:
                for reference in node.get('node_verification',{}).get('evidence_call_ids',[])[:8]:
                    if not isinstance(reference,str) or not reference.isalnum():continue
                    witness_path=Path(evidence_root)/(reference+'.json')
                    record,witness_sha=_owned_native_json(witness_path,evidence_root,min(remaining,2*1024*1024))
                    observed=result_object(record.get('result'));params=record.get('params') or {}
                    if (record.get('call_id')!=reference or record.get('failed') is not False or record.get('tool')!='read_file'
                            or record.get('time',0)<node.get('started_at',0) or not isinstance(observed,dict)
                            or params.get('path')!=str(path) or observed.get('path')!=str(path)
                            or observed.get('truncated') is not False or observed.get('offset',0)!=0
                            or not isinstance(observed.get('content'),str)
                            or observed.get('limit',0)<observed.get('total_lines',1)):
                        continue
                    if raw.decode('utf-8').splitlines()==observed['content'].splitlines():
                        witness={'path':str(witness_path),'sha256':witness_sha};break
                if not witness:continue
            observation={'path':str(path),'sha256':checksum,'size':len(raw),'inode':after.st_ino,
                         'mtime_ns':after.st_mtime_ns,'ctime_ns':after.st_ctime_ns}
            if witness:observation['verification_witness']=witness
            observations.append(observation);remaining-=len(raw)
        except (OSError,ValueError,PermissionError,TypeError,KeyError):continue

    return observations


def referenced_native_steps(state, evidence_root, answer):
    """Cheap identity match only; this grants no source or scientific authority."""
    import re
    if not evidence_root or not isinstance(state,dict) or not isinstance(answer,str):return []
    scope=Path(evidence_root).absolute().parent
    if (state.get('username'),state.get('conv_id'))!=(scope.parent.name,scope.name):return []
    found=[]
    for step,node in list((state.get('nodes') or {}).items())[:128]:
        if node.get('status') not in {'succeeded','prefinish'}:continue
        dossier=node.get('validation_dossier') or {}
        identities=[dossier.get('evidence_call_id'),dossier.get('evidence_path')]
        identities.extend(str(job) for job in node.get('job_ids',[]))
        identities.extend(_artifact_reference_aliases(node,dossier,scope))
        if any(isinstance(value,str) and len(value)>=3 and re.search(
                r'(?<![A-Za-z0-9])'+re.escape(value)+r'(?![A-Za-z0-9])',answer)
                for value in identities if value):found.append(step)
    return found


def native_result_sources(state, evidence_root, answer=None):
    """Only current, owned executor dossiers qualify; historical prose does not."""
    if not evidence_root or not isinstance(state, dict): return []
    root = Path(evidence_root).absolute()
    if (state.get('username'), state.get('conv_id')) != (root.parent.parent.name, root.parent.name):
        return []
    from .parallel_workflow import WorkflowStore
    workflow_id = WorkflowStore.identity(state['username'], state['conv_id'])
    if state.get('workflow_id', workflow_id) != workflow_id:
        return []
    sources = []
    remaining = 4 * 1024 * 1024
    for step, node in list((state.get('nodes') or {}).items())[:128]:
        dossier = node.get('validation_dossier') or {}
        if node.get('status') not in {'succeeded', 'prefinish'}:
            continue
        if answer is not None:
            import re
            identities = [dossier.get('evidence_call_id'), dossier.get('evidence_path')]
            identities.extend(str(job) for job in node.get('job_ids', []))
            identities.extend(_artifact_reference_aliases(node,dossier,root.parent))
            if not any(isinstance(value, str) and len(value) >= 3 and re.search(r'(?<![A-Za-z0-9])' + re.escape(value) + r'(?![A-Za-z0-9])', answer) for value in identities if value):
                continue
        if remaining <= 0 or len(sources) >= 16: break
        try:
            path = Path(dossier['evidence_path'])
            record, checksum = _owned_native_json(path, root, min(2 * 1024 * 1024, remaining))
            remaining -= path.stat().st_size
            if not isinstance(record, dict) or record.get('tool') != 'inspect_workflow_result' or record.get('failed') is not False:
                continue
            if record.get('call_id') != dossier.get('evidence_call_id'):
                continue
            full = record.get('result')
            if not isinstance(full, dict):
                continue
            # Runtime deliberately omits the full inventory and verification owner.
            # Every field it DOES carry must still equal the immutable native value.
            matches = True
            for key, value in dossier.items():
                if key == 'inventory' and isinstance(value, dict):
                    if set(value) - {'root', 'file_count', 'total_bytes', 'escaped_paths', 'truncated', 'reasons', 'files'}:
                        matches = False
                    if any(full.get('inventory', {}).get(name) != fact for name, fact in value.items()):
                        matches = False
                elif full.get(key) != value:
                    matches = False
            if not matches:
                continue
            expected = {'workflow_id': workflow_id, 'step_id': step,
                        'attempt_id': node.get('attempt_id'), 'job_ids': node.get('job_ids') or [],
                        'tool': node.get('contract', {}).get('tool'), 'contract': node.get('contract')}
            if any(dossier.get(key) != value or full.get(key) != value for key, value in expected.items()):
                continue
            observations=[]
            if not dossier.get('method_facts'):
                observations=_owned_artifact_observations(node,root,remaining)
                if not observations:continue
                remaining-=sum(item['size'] for item in observations)
            sources.append({'path': str(path.absolute()), 'sha256': checksum,
                            'call_id': record['call_id'], **{k: expected[k] for k in ('step_id','attempt_id','job_ids')},
                            'method_facts': dossier.get('method_facts') or {},
                            'artifact_observations':observations,'typed_scientific_facts_available':bool(dossier.get('method_facts'))})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return sources


def verified_evidence_family(tool, params, result, native_sources=(), evidence_root=None):
    """A source covers its own family only; a DOI/title is not a measured claim."""
    if tool == 'read_file':
        if not isinstance(result, dict) or result.get('error') or not evidence_root:
            return None
        for source in native_sources:
            if str(Path(params.get('path', '')).absolute()) != source['path'] or result.get('path') != source['path']:
                continue
            try:
                _, checksum = _owned_native_json(source['path'], evidence_root)
                if checksum == source['sha256']:
                    return 'native_results'
            except (ValueError, OSError, TypeError):
                pass
        return None
    if tool == 'read_scientific_document':
        source=result.get('source_document')or{} if isinstance(result,dict)else{}
        if (not is_source_result(tool,result)or not evidence_root
                or params.get('expected_sha256')!=source.get('sha256')
                or str(Path(params.get('path','')).absolute())!=source.get('path')):return None
        try:
            from .report_recovery import _OwnedFiles
            from .scientific_documents import _read_bytes
            scope=Path(evidence_root).absolute().parent
            files=_OwnedFiles(scope,{'username':scope.parent.name,'conv_id':scope.name})
            try:_,fact=_read_bytes(files,source['path'],32*1024**2)
            finally:files.close()
            if fact!=source:return None
        except (OSError,ValueError,TypeError,KeyError):return None
        return 'document_text'
    if tool == 'discover_forcefield':
        return None  # Discovery is an index, not inspected scientific parameters.
    return EVIDENCE_FAMILIES.get(tool) if is_source_result(tool, result) else None


def native_inspection_sources(calls, state, evidence_root, answer=None):
    """Bind a recorded fresh inspection to its current native producer attempt.

    Tool names, self-declared JSON, and an ``ok`` flag are insufficient. The
    Session's record-time SHA survives compaction and must match owned evidence.
    ``answer=None`` is for this turn's real calls; historical reuse first needs
    an explicit job/call/path identity in the answer, avoiding ordinary-chat IO.
    """
    import re
    from .parallel_workflow import WorkflowStore, RESOURCE_ARGUMENTS, same_scientific_contract
    from .recovery import result_object
    if not isinstance(state, dict) or not evidence_root or not isinstance(calls, list):
        return []
    root = Path(evidence_root).absolute()
    owner = [state.get('username'), state.get('conv_id')]
    if owner != [root.parent.parent.name, root.parent.name]: return []
    workflow_id = WorkflowStore.identity(*owner)
    if state.get('workflow_id', workflow_id) != workflow_id: return []
    goal = state.get('goal_contract') or {}
    if goal.get('approved_plan_version') != state.get('plan_version'): return []
    found, remaining = [], 4 * 1024 * 1024
    recent = calls[-128:]
    for call in reversed(recent):
        if not isinstance(call, dict) or call.get('tool') != 'inspect_workflow_result' or call.get('failed') is not False:
            continue
        call_id = call.get('call_id')
        if (not isinstance(call_id, str) or not call_id.isalnum() or len(call_id) > 128
                or sum(item.get('call_id') == call_id for item in recent if isinstance(item, dict)) != 1):
            continue
        params = call.get('params') or {};step = params.get('step_id')
        node = state.get('nodes', {}).get(step, {})
        if node.get('status') not in {'succeeded', 'prefinish'} or params != {'step_id': step}: continue
        path = root / (call_id + '.json')
        if call.get('evidence_path') != str(path): continue
        if answer is not None:
            identities = [call_id, str(path), *map(str, node.get('job_ids') or [])]
            if not any(re.search(r'(?<![A-Za-z0-9])' + re.escape(value) + r'(?![A-Za-z0-9])', answer)
                       for value in identities): continue
        if remaining <= 0 or len(found) >= 16: break
        try:
            record, checksum = _owned_native_json(path, root, min(2 * 1024 * 1024, remaining))
            raw = record.get('result')
            if (not isinstance(raw, str) or any(record.get(key) != call.get(key)
                    for key in ('call_id', 'tool', 'params', 'time', 'failed'))
                    or hashlib.sha256(raw.encode()).hexdigest() != call.get('result_sha256')):
                continue
            remaining -= path.stat().st_size
            result = result_object(raw)
            expected = {'workflow_id': workflow_id, 'owner': owner, 'step_id': step,
                'token': node.get('token'), 'attempt_id': node.get('attempt_id'),
                'job_ids': node.get('job_ids') or [], 'contract': node.get('contract'),
                'goal_version': goal.get('version'), 'plan_version': state.get('plan_version')}
            if (result.get('status') != 'inspected' or result.get('read_only') is not True
                    or result.get('no_resubmission') is not True
                    or any(result.get(key) != value for key, value in expected.items())): continue
            approvals = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step]
            if len(approvals) != 1: continue
            if not same_scientific_contract(node['contract'], approvals[0]):
                from .native_resource_carry import view_contract_matches
                if not view_contract_matches(state,step,approvals[0]): continue
            ref = node.get('result_ref') or {};producer_id = ref.get('call_id')
            if (not isinstance(producer_id, str) or not producer_id.isalnum()
                    or ref.get('evidence_path') != str(root / (producer_id + '.json'))
                    or result.get('source_execution_call_id') != producer_id): continue
            if remaining <= 0: continue
            producer_path = root / (producer_id + '.json')
            producer, producer_sha = _owned_native_json(producer_path, root,
                                                       min(2 * 1024 * 1024, remaining))
            remaining -= producer_path.stat().st_size
            if (producer_sha != result.get('source_execution_sha256')
                    or producer.get('call_id') != producer_id or producer.get('failed') is not False
                    or producer.get('tool') != node['contract'].get('tool')): continue
            expected_args = node.get('effective_arguments') or node['contract'].get('arguments', {})
            science = lambda arguments: {key: value for key, value in arguments.items() if key not in RESOURCE_ARGUMENTS}
            if (science(producer.get('params') or {}) != science(expected_args)
                    or result_object(producer.get('result')) != result_object(node.get('result'))): continue
            facts = result.get('method_facts', {})
            from .workflow_result_inspection import native_inspection_kind
            native_kind = native_inspection_kind(node.get('contract'))
            if native_kind is None or not isinstance(facts,dict): continue
            if native_kind == 'cdft_native':
                native = facts.get(native_kind)
                if (node['contract'].get('arguments',{}).get('action') != 'submit'
                        or not isinstance(native,dict) or native.get('expected_materials_verified') is not True
                        or not native.get('original_input_facts',{}).get('inputs')
                        or native.get('work_dir') != result_object(producer.get('result')).get('work_dir')): continue
            else:
                native = facts.get(native_kind)
                if (not isinstance(native, dict) or native.get('read_only') is not True
                        or native.get('tool') != node['contract']['tool']
                        or native.get('root') != result_object(producer.get('result')).get('work_dir')): continue
            found.append({'path': str(path), 'sha256': checksum, 'call_id': call_id,
                          'step_id': step, 'attempt_id': node['attempt_id'], 'job_ids': node['job_ids'],
                          'method_facts': result['method_facts']})
        except (OSError, ValueError, TypeError, KeyError):
            continue
    return found
