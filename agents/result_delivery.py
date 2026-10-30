"""Owned, immutable user deliverables; never a general file browser.

Only current accepted native report/figure receipts grant access. An opaque
identifier selects a receipt entry, never an arbitrary request path. Bytes are
opened under the session directory with NOFOLLOW and verified before sending.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

from .report_recovery import _OwnedFiles, _same, _decoded
from .workflow_patch import execution_contract
from .local_preparation_recovery import checkpoint_call_witness

_SHA = re.compile(r'[0-9a-f]{64}')
_ID = re.compile(r'[A-Za-z0-9_-]{1,128}')
_TYPES = {'.pdf': ('application/pdf', 32 * 1024 * 1024), '.md': ('text/markdown; charset=utf-8', 2 * 1024 * 1024),
          '.csv': ('text/csv; charset=utf-8', 8 * 1024 * 1024),
          '.json': ('application/json', 8 * 1024 * 1024),
          '.png': ('image/png', 16 * 1024 * 1024)}
_TOOLS = {'generate_scientific_report', 'generate_result_figures', 'download_scientific_file', 'run_analysis_sandbox'}


def _scope(root, state, owner):
    root = Path(os.path.abspath(root))
    if (set(owner) != {'username', 'conv_id'} or not all(
            isinstance(value, str) and _ID.fullmatch(value) for value in owner.values())
            or any(state.get(key) != value for key, value in owner.items())
            or root.name != owner['conv_id'] or root.parent.name != owner['username']):
        raise PermissionError('deliverable owner scope differs from current session')
    goal = state.get('goal_contract') or {}
    if state.get('plan_version') != goal.get('approved_plan_version'):
        raise ValueError('deliverable plan is not current')
    return root, goal


def _entries(root, state, owner):
    root, goal = _scope(root, state, owner)
    files = _OwnedFiles(root, owner)
    entries = []
    try:
        approved = {node['step_id']: node for node in goal.get('approved_nodes', [])}
        for step, node in state.get('nodes', {}).items():
            contract = node.get('contract') or {}
            if (node.get('status') != 'succeeded' or node.get('phase') != 'finish'
                    or not node.get('token') or contract.get('tool') not in _TOOLS
                    or step not in approved
                    or not _same(execution_contract(approved[step]), execution_contract(contract))):
                continue
            # Historical evidence indexes are intentionally not authored
            # deliverables. Ignore them before reading old producer CPs, which
            # may predate result digests; they must not block newer verified
            # reports and figures in the same accepted workflow.
            published = _decoded(node.get('result'))
            if not isinstance(published, dict):
                continue
            if (contract['tool'] == 'generate_scientific_report'
                    and published.get('status') != 'authored_verified'):
                continue
            if (contract['tool'] == 'generate_result_figures'
                    and (published.get('ok') is not True
                         or published.get('status') != 'rendered_needs_review')):
                continue
            ref = node.get('result_ref') or {}
            call_id = ref.get('call_id')
            if not isinstance(call_id, str) or not _ID.fullmatch(call_id):
                raise ValueError('accepted report lacks native producer evidence')
            expected = root / 'evidence' / (call_id + '.json')
            if str(expected) != ref.get('evidence_path'):
                raise PermissionError('deliverable producer evidence escaped its owner')
            call, _ = files.read(expected, as_json=True)
            result = _decoded(call.get('result'))
            if (call.get('call_id') != call_id or call.get('tool') != contract.get('tool')
                    or call.get('failed') is not False
                    or not _same(call.get('params'), node.get('effective_arguments') or contract.get('arguments'))
                    or not _same(result, _decoded(node.get('result'))) or not isinstance(result, dict)):
                raise ValueError('accepted deliverable does not match its native producer')
            checkpoint_call_witness(files, ref.get('checkpoint_path'), call,
                                    step_id=step, require_result_hash=True)
            candidates = []
            if contract['tool'] == 'generate_scientific_report':
                from .report_validation_receipt import report_validation_satisfied
                requirements=(contract.get('arguments')or{}).get('report_requirements')or{}
                from .report_review_intent import explicit_strict_review
                if not report_validation_satisfied(result,strict_required=(requirements.get('independent_review') is True or explicit_strict_review(goal))):
                    continue
                candidates.append({'path': result.get('report_file'),
                                   'sha256': result.get('report_sha256'), 'kind': 'report'})
                candidates += [dict(item, kind='source_data') for item in result.get('data_sources', [])]
            elif contract['tool']=='run_analysis_sandbox':
                args=node.get('effective_arguments') or contract.get('arguments') or {}
                if (result.get('ok') is not True or result.get('status')!='data_ready_needs_review'
                        or result.get('owner')!=owner or result.get('producer')!={'step_id':step,'runtime_token':node['token']}
                        or (result.get('sandbox') or {}).get('os_isolated') is not True):
                    continue
                declared={x.get('name') for x in args.get('declared_outputs',[]) if isinstance(x,dict)}
                expected_paths=set()
                for entry in contract.get('expected_outputs') or []:
                    if isinstance(entry,str):expected_paths.add(os.path.abspath(entry))
                    elif isinstance(entry,dict) and entry.get('kind')=='file' and isinstance(entry.get('path'),str):expected_paths.add(os.path.abspath(entry['path']))
                for artifact in result.get('artifacts') or []:
                    if (isinstance(artifact,dict)
                            and artifact.get('name') in declared and os.path.abspath(str(artifact.get('path') or '')) in expected_paths
                            and artifact.get('scientific_acceptance') is False):
                        candidates.append(dict(artifact,kind='sandbox_file'))
            elif contract['tool']=='download_scientific_file':
                if result.get('status')!='success':continue
                declared=(node.get('effective_arguments')or contract.get('arguments')or{}).get('output_path')
                if not isinstance(declared,str):raise ValueError('document lacks approved output path')
                declared=Path(declared).expanduser()
                if not declared.is_absolute():declared=root.parent.parent.parent/declared
                if result.get('output_path')!=os.path.abspath(declared):
                    raise ValueError('document download differs from approved output path')
                candidates=[{'path':result.get('output_path'),'sha256':result.get('sha256'),'kind':'source_document'}]
            elif result.get('ok') is True and result.get('status') == 'rendered_needs_review':
                candidates = [dict(item, kind='figure') for item in result.get('artifacts', [])]
            for candidate in candidates:
                raw_path = candidate.get('path')
                if not isinstance(raw_path, str) or not _SHA.fullmatch(str(candidate.get('sha256', ''))):
                    raise ValueError('native deliverable lacks a valid immutable content hash')
                path = files.owned(raw_path)
                suffix = path.suffix.lower()
                if candidate['kind']=='source_document' and suffix!='.pdf':continue
                if suffix not in _TYPES and candidate['kind']!='sandbox_file':
                    continue
                # Source tables may be session-relative. Report/figure bytes
                # must come from the exact producer output declaration.
                if candidate['kind'] not in {'source_data','sandbox_file'} and str(path) not in result.get('output_files', []):
                    raise ValueError('deliverable is not a declared native output')
                relative = path.relative_to(root).as_posix()
                identity = {'owner': owner, 'step_id': step, 'token': node['token'],
                            'path': relative, 'sha256': candidate['sha256']}
                artifact_id = hashlib.sha256(json.dumps(identity, sort_keys=True,
                    separators=(',', ':')).encode()).hexdigest()
                entries.append({'artifact_id': artifact_id, 'name': path.name, 'step_id': step,
                    'kind': candidate['kind'], 'sha256': candidate['sha256'],
                    'report_validation_mode':result.get('review_mode') if candidate['kind']=='report' else None,
                    'independent_review_performed':result.get('independent_review_performed') if candidate['kind']=='report' else None,
                    'visual_review': (result.get('visual_review') or {}).get('status')
                        if candidate['kind'] == 'figure' else None,
                    '_path': str(path), '_identity': identity})
                if len(entries) > 128:
                    raise ValueError('deliverable inventory exceeds bounded limit')
        return entries
    finally:
        files.close()


def list_deliverables(root, state, owner):
    return [{key: value for key, value in entry.items() if not key.startswith('_')}
            for entry in _entries(root, state, owner)]


def read_deliverable(root, state, owner, artifact_id, *, current_state):
    if not isinstance(artifact_id, str) or not _SHA.fullmatch(artifact_id):
        raise KeyError('unknown deliverable identifier')
    selected = next((item for item in _entries(root, state, owner)
                     if item['artifact_id'] == artifact_id), None)
    if selected is None:
        raise KeyError('deliverable is absent from current accepted outputs')
    root = Path(os.path.abspath(root)); path = Path(selected['_path'])
    media_type, limit = _TYPES.get(path.suffix.lower(),('application/octet-stream', 8 * 1024 * 1024))
    files = _OwnedFiles(root, owner)
    parent = os.dup(files.fd); fd = None
    try:
        parts = path.relative_to(root).parts
        for part in parts[:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent); parent = following
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= limit:
            raise ValueError('deliverable is not a bounded regular file')
        chunks = []; length = 0; digest = hashlib.sha256()
        while True:
            block = os.read(fd, min(65536, limit + 1 - length))
            if not block:
                break
            length += len(block)
            if length > limit:
                raise ValueError('deliverable exceeded its read bound')
            chunks.append(block); digest.update(block)
        after = os.fstat(fd); linked = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns,
                                  item.st_ctime_ns, item.st_mode)
        if (length != before.st_size or identity(before) != identity(after)
                or identity(after) != identity(linked) or digest.hexdigest() != selected['sha256']):
            raise ValueError('deliverable changed after native publication')
        raw = b''.join(chunks)
        if selected['kind']=='sandbox_file':
            pass  # The owned sandbox artifact is delivered byte-for-byte.
        elif path.suffix.lower() == '.png':
            if not raw.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('figure is not a PNG')
        elif path.suffix.lower()=='.pdf':
            if b'%PDF-'not in raw[:1024]:raise ValueError('document is not PDF')
        else:
            raw.decode('utf-8')
        latest = current_state()
        if (latest.get('plan_version') != state.get('plan_version')
                or not _same(latest.get('goal_contract'), state.get('goal_contract'))
                or not any(item['artifact_id'] == artifact_id
                           for item in _entries(root, latest, owner))):
            raise ValueError('deliverable control changed during download')
        return raw, media_type, selected['name']
    finally:
        if fd is not None:
            os.close(fd)
        os.close(parent); files.close()
