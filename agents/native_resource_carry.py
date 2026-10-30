"""Attest already executed allocation metadata; never grant a new allocation."""
import copy
import hashlib
import json
import os
import stat
from pathlib import Path

from .report_recovery import _OwnedFiles, _same
from .workflow_patch import execution_contract

_METADATA = frozenset({'memory_mb', 'resource_review_id'})
_SCOPE = ('version', 'method', 'gases', 'parameters', 'execution_mode', 'execution_authorized')
_IDENTITY = ('contract', 'token', 'attempt_id', 'recovery_key', 'job_ids', 'result_ref',
             'result', 'effective_arguments', 'resource_allocation', 'tool_returned')
_LEDGER_IDENTITY = ('tool', 'params', 'attempt_id', 'job_ids', 'fingerprint', 'goal_version')
MAX_CREDENTIAL_BYTES = 1024 * 1024


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def _identity(node):
    return {key: node.get(key) for key in _IDENTITY}


def _goal_scope(goal):
    return {key: goal.get(key) for key in _SCOPE + ('approved_plan_version',)}


def _credential_unchanged(root, fact):
    """Hash current bytes even when all stat fields collide; at most 1 MiB."""
    from .workload_budget import _owned_fd
    fd = _owned_fd(Path(root), Path(fact['path']))
    try:
        before = os.fstat(fd)
        identity = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                  value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
        if (not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_CREDENTIAL_BYTES
                or before.st_ino != fact['inode'] or any(getattr(before, 'st_' + key) != fact[key]
                    for key in ('size', 'mtime_ns', 'ctime_ns'))): return False
        checksum = hashlib.sha256(); count = 0
        while True:
            block = os.read(fd, min(65536, MAX_CREDENTIAL_BYTES + 1 - count))
            if not block: break
            count += len(block)
            if count > MAX_CREDENTIAL_BYTES: return False
            checksum.update(block)
        linked = _owned_fd(Path(root), Path(fact['path']))
        try: visible = os.fstat(linked)
        finally: os.close(linked)
        return (count == before.st_size and identity(before) == identity(os.fstat(fd)) == identity(visible)
                and checksum.hexdigest() == fact.get('sha256'))
    finally: os.close(fd)


def _metadata_only(before, incoming):
    left, right = execution_contract(before), execution_contract(incoming)
    a, b = left.pop('arguments'), right.pop('arguments')
    return (left == right and a != b
        and {k: v for k, v in a.items() if k not in _METADATA}
        == {k: v for k, v in b.items() if k not in _METADATA})


def prove_native_resource_carry(state, step_id, incoming, owned_root, goal, jobs):
    """Hash bounded native evidence outside WorkflowStore's transaction."""
    from .local_preparation_recovery import checkpoint_call_witness
    from .recovery import failure_reason, result_object
    node = state.get('nodes', {}).get(step_id, {})
    owner = {key: state.get(key) for key in ('username', 'conv_id')}
    saved = state.get('goal_contract', {})
    effective = node.get('effective_arguments')
    allocation = node.get('resource_allocation') or {}
    ids = node.get('job_ids')
    approved = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step_id]
    if ((node.get('status'), node.get('phase')) not in {('prefinish', 'prefinish'), ('succeeded', 'finish')}
            or node.get('tool_returned') is not True or not node.get('token') or not node.get('attempt_id')
            or not isinstance(ids, list) or not ids or len(ids) > 64 or len(set(ids)) != len(ids)
            or goal.get('execution_authorized') is not True
            # A later authorized plan may retain the exact already executed
            # science. Re-attest it under the current revision; never reuse an
            # old proof across revisions or permit a rollback/new execution.
            or any(type(source.get(key)) is not int or source[key] < 1
                for source in (saved, goal) for key in ('version', 'approved_plan_version'))
            or saved['version'] > goal['version']
            or saved['approved_plan_version'] > goal['approved_plan_version']
            or any(saved.get(key) != goal.get(key) for key in _SCOPE if key != 'version')
            or len(approved) != 1 or execution_contract(approved[0]) != execution_contract(incoming)
            or not _metadata_only(node.get('contract', {}), incoming)
            or not _same(effective, incoming.get('arguments'))
            or allocation.get('budget_final') is not True
            or allocation.get('memory_mb') != effective.get('memory_mb')
            or allocation.get('resource_review_id') != effective.get('resource_review_id')):
        raise ValueError('native carry requires unchanged completed science and exact executed allocation metadata')
    for job_id in ids:
        job = jobs.get(str(job_id), {})
        if ((job.get('username'), job.get('conv_id')) != (owner['username'], owner['conv_id'])
                or str(job.get('job_id')) != str(job_id) or job.get('attempt_id') != node['attempt_id']
                or job.get('tool', node['contract']['tool']) != node['contract']['tool']
                or job.get('terminal') is not True or job.get('failed') is not False
                or job.get('state') != 'COMPLETED'):
            raise ValueError('native allocation carry requires original owned successful terminal jobs')
    files = _OwnedFiles(owned_root, owner)
    try:
        ref = node.get('result_ref') or {}
        path = files.root / 'evidence' / (str(ref.get('call_id')) + '.json')
        if ref.get('evidence_path') != str(path): raise ValueError('native carry evidence identity mismatch')
        call, call_fact = files.read(path, as_json=True)
        result = result_object(call.get('result'))
        if (call.get('call_id') != ref.get('call_id') or call.get('tool') != node['contract']['tool']
                or call.get('failed') is not False or failure_reason(result)
                or not _same(call.get('params'), effective) or not _same(result, result_object(node.get('result')))):
            raise ValueError('native carry producer does not attest exact actual parameters and receipt')
        native_ids = set(map(str, result.get('job_ids') or []))
        if result.get('job_id'): native_ids.add(str(result['job_id']))
        if native_ids != set(map(str, ids)): raise ValueError('native carry receipt job identities mismatch')
        checkpoint_fact = checkpoint_call_witness(files, ref.get('checkpoint_path'), call, step_id=step_id)
        ledger, ledger_fact = files.read(files.root / 'recovery_state.json', as_json=True)
        entry = ledger.get(node.get('recovery_key'), {})
        if (entry.get('attempt_id') != node['attempt_id'] or entry.get('tool') != node['contract']['tool']
                or set(map(str, entry.get('job_ids') or [])) != set(map(str, ids))
                or entry.get('status') not in {'completed_unverified', 'completed'}
                or not _same(entry.get('params'), effective)):
            raise ValueError('native allocation carry ledger does not attest original completed attempt')
        proof = {'schema_version': 1, 'kind': 'executed_resource_metadata', 'owner': owner,
            'step_id': step_id, 'token': node['token'], 'job_ids': copy.deepcopy(ids),
            'plan_version': goal['approved_plan_version'], 'goal_sha256': _hash(_goal_scope(goal)),
            'node_identity_sha256': _hash(_identity(node)), 'incoming_contract': execution_contract(incoming),
            'ledger_entry_sha256': _hash({key: entry.get(key) for key in _LEDGER_IDENTITY}),
            'ledger_fact': ledger_fact,
            'files': [call_fact, checkpoint_fact], 'no_dispatch': True}
        proof['proof_sha256'] = _hash(proof)
        return proof
    finally: files.close()


def proof_matches(state, step_id, incoming, owned_root, goal, proof, *, check_ledger=True):
    """Fresh bounded credential SHA checks, including the final store transaction.

    Linked identities detect common concurrent replacements; stat is never a
    substitute for SHA. This is not an atomic snapshot against arbitrary
    non-cooperating external writers across two different files.
    """
    from .gcmc_results import _owned_fd
    node = state.get('nodes', {}).get(step_id, {})
    if (not isinstance(proof, dict) or proof.get('schema_version') != 1
            or proof.get('kind') != 'executed_resource_metadata'
            or proof.get('proof_sha256') != _hash({k: v for k, v in proof.items() if k != 'proof_sha256'})
            or proof.get('owner') != {key: state.get(key) for key in ('username', 'conv_id')}
            or proof.get('step_id') != step_id or proof.get('token') != node.get('token')
            or proof.get('goal_sha256') != _hash(_goal_scope(goal)) or proof.get('plan_version') != goal.get('approved_plan_version')
            or proof.get('node_identity_sha256') != _hash(_identity(node))
            or not _same(proof.get('incoming_contract'), execution_contract(incoming))
            or node.get('status') not in {'prefinish', 'succeeded'}):
        return False
    try:
        expected_paths = {(node.get('result_ref') or {}).get(key) for key in ('evidence_path', 'checkpoint_path')}
        if (len(proof.get('files', [])) != 2 or len(expected_paths) != 2
                or {fact.get('path') for fact in proof['files']} != expected_paths): return False
        for fact in proof['files']:
            if not _credential_unchanged(owned_root, fact): return False
        if check_ledger:
            files = _OwnedFiles(owned_root, proof['owner'])
            try: ledger, _ = files.read(files.root / 'recovery_state.json', as_json=True)
            finally: files.close()
            entry = ledger.get(node.get('recovery_key'), {})
            if (entry.get('status') not in {'completed_unverified', 'completed'}
                    or _hash({key: entry.get(key) for key in _LEDGER_IDENTITY}) != proof.get('ledger_entry_sha256')):
                return False
        facts = proof['files'] if check_ledger else proof['files'] + [proof['ledger_fact']]
        for fact in facts:
            fd = _owned_fd(Path(owned_root), Path(fact['path']))
            try:
                current = os.fstat(fd)
                if (not stat.S_ISREG(current.st_mode) or current.st_ino != fact['inode']
                        or any(getattr(current, 'st_' + key) != fact[key] for key in ('size', 'mtime_ns', 'ctime_ns'))):
                    return False
            finally: os.close(fd)
    except (OSError, ValueError, KeyError, TypeError): return False
    return True


def approved_graph_matches(state, approved_nodes, owned_root, goal):
    approved = {node['step_id']: node for node in approved_nodes}
    if len(approved) != len(approved_nodes) or set(approved) != set(state.get('nodes', {})):
        return False
    if state.get('plan_version') != goal.get('approved_plan_version'): return False
    for step_id, node in state['nodes'].items():
        if execution_contract(node['contract']) == execution_contract(approved[step_id]): continue
        if not proof_matches(state, step_id, approved[step_id], owned_root, goal,
                             node.get('approved_resource_carry')): return False
    return True


def view_contract_matches(state, step_id, incoming):
    """Read-only presentation uses the same proof, not a resource-field ignore list."""
    node = state.get('nodes', {}).get(step_id, {})
    if execution_contract(node.get('contract', {})) == execution_contract(incoming): return True
    ref = node.get('result_ref') or {}
    try:
        root = Path(ref['evidence_path']).parent.parent
        if (root.name, root.parent.name) != (state.get('conv_id'), state.get('username')): return False
        return proof_matches(state, step_id, incoming, root, state.get('goal_contract', {}),
                             node.get('approved_resource_carry'))
    except (KeyError, TypeError, ValueError, OSError): return False


# Presentation classifications never authorize dispatch, carry, or completion.
# Resource quantities remain compared and visible; only the receipt identifier
# is classified separately from allocation values and scientific arguments.
_VIEW_RESOURCES = frozenset({'memory_mb', 'nodelist', 'partition', 'num_processes',
                           'cpus_per_task', 'resource_review_id', 'walltime'})


def view_contract_status(state, step_id, incoming, current_goal=None, advisory_cache=False):
    node = state.get('nodes', {}).get(step_id, {})
    declared = execution_contract(incoming)
    canonical = execution_contract(node.get('contract', {}))
    approved = declared['arguments']
    baseline = canonical['arguments']
    effective = node.get('effective_arguments')
    resolved = node.get('resolved_arguments') or baseline

    def science(contract):
        value = copy.deepcopy(contract)
        value['arguments'] = {k: v for k, v in value['arguments'].items()
                              if k not in _VIEW_RESOURCES}
        return value

    def differences(left, right, keys):
        return sorted(k for k in keys if (k in left) != (k in right) or left.get(k) != right.get(k))

    canonical_changes = differences(approved, baseline, _VIEW_RESOURCES)
    effective_changes = differences(baseline, effective, _VIEW_RESOURCES) if isinstance(effective, dict) else []
    effective_science = (not isinstance(effective, dict) or
        {k: v for k, v in effective.items() if k not in _VIEW_RESOURCES}
        == {k: v for k, v in resolved.items() if k not in _VIEW_RESOURCES})
    if not effective_science and node.get('analysis_input_proofs'):
        try:
            if advisory_cache:
                from .derived_view_cache import view_proven_derived_arguments as prove
            else:
                from .derived_completion_proof import proven_derived_arguments as prove
            verified=prove(state,step_id,current_goal)
            effective_science=({k:v for k,v in effective.items() if k not in _VIEW_RESOURCES}
                               == {k:v for k,v in verified.items() if k not in _VIEW_RESOURCES})
        except (ValueError,PermissionError,OSError,KeyError,TypeError):
            # Provenance failure must remain a completion blocker.
            effective_science=False
    science_matches = science(declared) == science(canonical) and effective_science
    matches = view_contract_matches(state, step_id, incoming)
    allocation = node.get('resource_allocation') or {}
    mapping = {'memory_mb': 'memory_mb', 'nodelist': 'nodelist', 'partition': 'partition',
               'num_processes': 'cpus', 'cpus_per_task': 'cpus',
               'resource_review_id': 'resource_review_id'}
    claimed_resources = sorted(set(effective).intersection(mapping)) if isinstance(effective, dict) else []
    receipt_status = 'not_recorded'
    if claimed_resources:
        receipt_status = 'unreviewed'
        if allocation.get('budget_final') is True and allocation.get('evidence_call_ids'):
            checks = [effective[k] == allocation.get(mapping[k]) for k in claimed_resources]
            receipt_status = ('recorded_review' if effective.get('resource_review_id')
                              and all(checks) else 'inconsistent')
    # Walltime is a declared limit, not an observed CPU/RAM allocation. A
    # completed legacy adapter may retain it without copying context-injected
    # allocation arguments into effective_arguments. Report the missing record
    # explicitly; an empty allocation comparison proves no inconsistency.
    walltime_status = 'not_recorded'
    walltime_changed = False
    if isinstance(effective, dict) and 'walltime' in effective:
        walltime_changed = effective['walltime'] != resolved.get('walltime')
        if walltime_changed:
            walltime_status = 'differs_from_approved_contract'
        elif allocation.get('walltime') is None:
            walltime_status = 'execution_record_unknown'
        else:
            walltime_status = ('matches_recorded_allocation' if effective['walltime'] == allocation['walltime']
                               else 'inconsistent')
            walltime_changed = walltime_status == 'inconsistent'
    if not canonical_changes:
        resource_status = 'matches_canonical_contract'
    elif canonical_changes == ['resource_review_id']:
        resource_status = 'receipt_identity_changed'
    else:
        resource_status = 'allocation_changed'
    acceptance = 'contract_accepted' if matches else 'contract_not_accepted'
    if not effective_science:
        acceptance = 'effective_science_changed'
    elif (claimed_resources and receipt_status != 'recorded_review') or walltime_changed:
        acceptance = 'effective_resources_unverified'
    return {
        'contract_matches_execution': matches,
        'science_contract_status': 'matches' if science_matches else 'changed',
        'resource_contract_status': resource_status,
        'resource_contract_changed_fields': canonical_changes,
        'effective_resource_changed_fields': effective_changes,
        'effective_resource_receipt_status': receipt_status,
        'effective_resource_claimed_fields': claimed_resources,
        'effective_walltime_status': walltime_status,
        'approved_arguments': copy.deepcopy(approved),
        'approved_arguments_source': 'task_line.arguments',
        'canonical_arguments': copy.deepcopy(baseline),
        'canonical_arguments_source': 'workflow_executor.nodes.contract.arguments',
        'effective_arguments': copy.deepcopy(effective),
        'effective_arguments_source': ('workflow_executor.nodes.effective_arguments'
                                       if isinstance(effective, dict) else 'not_recorded'),
        'execution_acceptance_status': acceptance,
        'execution_acceptance_advisory': bool(advisory_cache),
    }
