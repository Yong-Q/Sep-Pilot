"""Read-only proof for keeping an accepted report on the same approved DAG.

This is immutable-result reuse, not an execution fingerprint or an LLM cache.
It never creates a result, repairs a source, grants execution, or accepts new
science. Unknown/legacy receipts remain ineligible except an independently
verified native restoration that has already passed normal finish validation.
"""
import copy
import hashlib
import os
import stat
from pathlib import Path
import re

from .report_recovery import _OwnedFiles, _same, _canonical
from .workflow_patch import execution_contract
from .local_preparation_recovery import checkpoint_call_witness
from .recovery import result_object

TOOLS = frozenset({'generate_scientific_report', 'generate_result_figures'})
_SHA = re.compile(r'[0-9a-f]{64}')


def verify_legacy_index(state, step_id, session_root, goal):
    """Keep a native historical index as an index, never as authored delivery."""
    owner = {key: state.get(key) for key in ('username', 'conv_id')}
    files = _OwnedFiles(session_root, owner); facts = []; visited = set()
    def native(key, *, digest_required=True):
        node = state.get('nodes', {}).get(key, {}); contract = node.get('contract', {})
        ref = node.get('result_ref') or {}; call_id = ref.get('call_id')
        if (node.get('status') != 'succeeded' or not node.get('token')
                or not isinstance(call_id, str) or not call_id.isalnum()
                or ref.get('evidence_path') != str(files.root / 'evidence' / (call_id + '.json'))):
            raise ValueError('historical index producer is not a current owned accepted node')
        call, fact = files.read(ref['evidence_path'], as_json=True); facts.append(fact)
        result = result_object(call.get('result'))
        if (call.get('call_id') != call_id or call.get('tool') != contract.get('tool')
                or call.get('failed') is not False
                or not _same(call.get('params'), node.get('effective_arguments') or contract.get('arguments'))
                or not _same(result, result_object(node.get('result')))):
            raise ValueError('historical index native receipt differs')
        facts.append(checkpoint_call_witness(files, ref.get('checkpoint_path'), call,
            step_id=key, require_result_hash=digest_required))
        return node, result, fact
    def visit(key):
        if key in visited or len(visited) >= 64:
            raise ValueError('historical index dependency cycle or excessive depth')
        visited.add(key)
        node, result, _ = native(key); args = node['contract']['arguments']
        if (node['contract']['tool'] != 'generate_scientific_report' or result.get('status') != 'success'
                or result.get('source_steps') != args.get('source_steps')):
            raise ValueError('historical index schema or source contract differs')
        report = result.get('report_file')
        if (report != result.get('output_path') or report != args.get('output_path')
                or result.get('output_files') != [report]):
            raise ValueError('historical index output path differs from its native declaration')
        _, output = files.read(report); facts.append(output)
        mode = args.get('report_mode') or node['contract'].get('report_role') or 'fragment'
        if mode == 'fragment':
            source_id = result.get('source_step'); source, _, source_fact = native(source_id, digest_required=False)
            if (args.get('source_steps') != [source_id]
                    or result.get('source_evidence_call_id') != source['result_ref']['call_id']
                    or result.get('source_attempt_fingerprint') != source.get('execution_fingerprint')
                    or result.get('evidence_fingerprint') != source_fact['sha256']
                    or result.get('fragment_fingerprint') != output['sha256']):
                raise ValueError('historical index contains a stale source attempt or changed output')
        elif mode == 'assembly':
            rows = result.get('fragment_manifest')
            if not isinstance(rows, list) or [row.get('step_id') for row in rows] != args.get('source_steps'):
                raise ValueError('historical assembly fragment manifest differs')
            sections = ['# ' + args['title'].strip(), '', args.get('instructions',
                'This live report is assembled from completed, evidence-scoped workflow branches.'), '']
            for row in rows:
                fragment = visit(row['step_id']); current = result_object(state['nodes'][row['step_id']]['result'])
                if any(row.get(name) != current.get(name) for name in (
                        'source_step', 'source_attempt_fingerprint', 'fragment_fingerprint', 'report_file')):
                    raise ValueError('historical assembly references a stale fragment')
                sections.extend([fragment.strip(), ''])
            expected = '\n'.join(sections).rstrip() + '\n'
            if hashlib.sha256(expected.encode()).hexdigest() != output['sha256']:
                raise ValueError('historical assembly bytes no longer match immutable native fragments')
        else:
            raise ValueError('unsupported historical index mode')
        # Read only bounded report text; original calls/tables stay evidence.
        path = files.owned(report)
        if output['size'] > 2 * 1024 * 1024:
            raise ValueError('historical index exceeds bounded report size')
        parent = os.dup(files.fd); fd = None
        try:
            parts = path.relative_to(files.root).parts
            for part in parts[:-1]:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
                os.close(parent); parent = child
            fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size != output['size']:
                raise ValueError('historical index file identity changed')
            blocks = []; count = 0
            while count <= output['size']:
                block = os.read(fd, min(65536, output['size'] + 1 - count))
                if not block: break
                blocks.append(block); count += len(block)
            raw = b''.join(blocks)
            after = os.fstat(fd); linked = os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
            identity = lambda item: (item.st_dev, item.st_ino, item.st_size, item.st_mtime_ns, item.st_ctime_ns, item.st_mode)
            if identity(before) != identity(after) or identity(after) != identity(linked):
                raise ValueError('historical index file changed during read')
        finally:
            if fd is not None: os.close(fd)
            os.close(parent)
        if hashlib.sha256(raw).hexdigest() != output['sha256']:
            raise ValueError('historical index changed while reading')
        visited.remove(key)
        return raw.decode('utf-8')
    try:
        visit(step_id)
        node = state['nodes'][step_id]
        proof = {'kind': 'verified_historical_evidence_index', **owner, 'step_id': step_id,
            'current_token': node['token'], 'goal_version': goal['version'],
            'plan_version': state['plan_version'], 'facts': facts,
            'delivery_class': 'legacy_evidence_index', 'authored_report': False,
            'no_reexecution': True, 'no_resubmission': True}
        proof['proof_sha256'] = hashlib.sha256(_canonical(proof)).hexdigest()
        return proof
    finally:
        files.close()


def verify_completed_presentation(state, step_id, session_root, goal):
    owner = {key: state.get(key) for key in ('username', 'conv_id')}
    node = state.get('nodes', {}).get(step_id, {})
    contract = node.get('contract') or {}
    approved = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step_id]
    if (state.get('user_paused') or goal.get('execution_authorized') is not True
            or state.get('plan_version') != goal.get('approved_plan_version')
            or not _same(state.get('goal_contract'), goal)
            or node.get('status') != 'succeeded' or node.get('phase') != 'finish'
            or not node.get('token') or node.get('job_ids') or contract.get('tool') not in TOOLS
            or len(approved) != 1
            or not _same(execution_contract(approved[0]), execution_contract(contract))):
        raise ValueError('presentation reuse requires exact current accepted node and approved plan')
    if node.get('restoration'):
        # Legacy recovery has its own stronger source/history proof, not a
        # fabricated new-format author/reviewer receipt or new LLM invocation.
        from .parallel_workflow import _verify_local_restoration
        observed = copy.deepcopy(state)
        observed['nodes'][step_id]['status'] = 'prefinish'
        restored = _verify_local_restoration(observed, step_id, Path(session_root), goal)
        proof = {'kind': 'verified_completed_restored_presentation', **owner,
            'step_id': step_id, 'current_token': node['token'],
            'goal_version': goal['version'], 'plan_version': state['plan_version'],
            'restoration_proof_sha256': restored['proof_sha256'],
            'verification': node.get('node_verification'), 'no_reexecution': True,
            'no_resubmission': True}
        proof['proof_sha256'] = hashlib.sha256(_canonical(proof)).hexdigest()
        return proof
    if (contract.get('tool') == 'generate_scientific_report'
            and result_object(node.get('result')).get('status') == 'success'):
        return verify_legacy_index(state, step_id, session_root, goal)
    files = _OwnedFiles(session_root, owner)
    facts = []; identities = []; seen = set(); budget = 32 * 1024 * 1024
    visited = 0

    def read(path, as_json=False):
        nonlocal budget
        path = files.owned(path)
        if path.lstat().st_size > budget:
            raise ValueError('presentation reuse exceeds its bounded evidence budget')
        data, fact = files.read(path, as_json=as_json)
        budget -= fact['size']; facts.append(fact)
        return data, fact

    def native(key, source):
        nonlocal budget
        reference = source.get('result_ref') or {}; call_id = reference.get('call_id')
        if not isinstance(call_id, str) or not call_id.isalnum():
            raise ValueError('presentation producer requires a native call identity')
        path = files.root / 'evidence' / (call_id + '.json')
        if reference.get('evidence_path') != str(path):
            raise PermissionError('presentation native evidence escaped its owner')
        call, fact = read(path, True)
        result = result_object(call.get('result'))
        if (call.get('call_id') != call_id or call.get('tool') != source['contract']['tool']
                or call.get('failed') is not False
                or not _same(call.get('params'), source.get('effective_arguments') or source['contract']['arguments'])
                or not _same(result, result_object(source.get('result')))):
            raise ValueError('presentation native producer differs from accepted evidence')
        checkpoint = reference.get('checkpoint_path')
        if not isinstance(checkpoint, str) or not checkpoint:
            raise ValueError('presentation producer checkpoint is absent')
        if files.owned(checkpoint).lstat().st_size > budget:
            raise ValueError('presentation producer checkpoint exceeds remaining evidence budget')
        witness = checkpoint_call_witness(files, checkpoint, call, step_id=key, require_result_hash=True)
        budget -= witness['size']; facts.append(witness)
        return result, fact

    def visit(key, expected=None):
        nonlocal visited
        visited += 1
        if key in seen or visited > 128:
            raise ValueError('presentation source cycle or excessive dependency count')
        seen.add(key)
        current = state.get('nodes', {}).get(key, {})
        if current.get('status') != 'succeeded' or not current.get('token'):
            raise ValueError('presentation source is no longer accepted')
        result, fact = native(key, current)
        identity = {'step_id': key, 'token': current.get('token'), 'attempt_id': current.get('attempt_id'),
            'execution_fingerprint': current.get('execution_fingerprint'),
            'call_id': (current.get('result_ref') or {}).get('call_id'),
            'source_sha256': fact['sha256'],
            'contract_sha256': hashlib.sha256(_canonical(execution_contract(current['contract']))).hexdigest()}
        if expected is not None and (set(expected) not in (set(identity), set(identity) - {'contract_sha256'})
                or not _same({field: identity[field] for field in expected}, expected)):
            raise ValueError('stale source attempt in accepted presentation')
        identities.append(identity)
        if current.get('contract', {}).get('tool') not in TOOLS:
            seen.remove(key)
            return
        refs = result.get('source_receipts')
        sources = current['contract'].get('arguments', {}).get('source_steps') or []
        if (not isinstance(refs, list) or len(refs) != len(sources)
                or any(not isinstance(ref, dict) for ref in refs)
                or {ref.get('step_id') for ref in refs} != set(sources)):
            raise ValueError('accepted presentation lacks complete native source receipts')
        if current['contract']['tool'] == 'generate_scientific_report':
            from .report_validation_receipt import report_validation_satisfied
            requirements=(current['contract'].get('arguments')or{}).get('report_requirements')or{}
            from .report_review_intent import explicit_strict_review
            if not report_validation_satisfied(result,strict_required=(requirements.get('independent_review') is True or explicit_strict_review(goal))):
                raise ValueError('evidence indexes are not model-authored reports')
            outputs = [{'path': result.get('report_file'), 'sha256': result.get('report_sha256')}]
            outputs += result.get('data_sources') or []
        else:
            if result.get('ok') is not True or result.get('status') != 'rendered_needs_review':
                raise ValueError('figure receipt does not prove rendered native outputs')
            outputs = list(result.get('artifacts') or [])
            outputs += result.get('source_tables') or []
        if not outputs:
            raise ValueError('accepted presentation has no immutable output')
        for output in outputs:
            if not isinstance(output, dict) or not _SHA.fullmatch(str(output.get('sha256', ''))):
                raise ValueError('presentation output requires its native content digest')
            _, observed = read(output['path'])
            if observed['sha256'] != output['sha256']:
                raise ValueError('accepted presentation output or source data changed')
        for ref in refs:
            visit(ref['step_id'], ref)
        seen.remove(key)
    try:
        visit(step_id)
        proof = {'kind': 'verified_completed_presentation', **owner, 'step_id': step_id,
            'goal_version': goal['version'], 'plan_version': state['plan_version'],
            'current_token': node['token'], 'identities': identities, 'facts': facts,
            'approved_goal_sha256': hashlib.sha256(_canonical(goal)).hexdigest(),
            'no_reexecution': True, 'no_resubmission': True}
        proof['proof_sha256'] = hashlib.sha256(_canonical(proof)).hexdigest()
        return proof
    finally:
        files.close()
