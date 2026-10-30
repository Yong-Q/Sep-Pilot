"""Native carry of accepted history during an approved presentation delta.

The typed receipt never grants execution. Main-chat approval remains mandatory;
this proof merely prevents unchanged accepted science/history from being run a
second time while adding or replacing presentation nodes.
"""
import copy
import hashlib
import re
from pathlib import Path

from .report_recovery import _OwnedFiles, _same, _canonical
from .recovery import result_object
from .workflow_patch import execution_contract, patched_graph
from .presentation_continuation import classify_presentation_continuation, PRESENTATION_TOOLS
from .local_preparation_recovery import checkpoint_call_witness


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _self_recorded_reuse_receipt(state, receipt, approval):
    """Reconstruct only the timestamp changed by our own reuse bookkeeping.

    The old receipt hash must match exactly, all other source identities must
    match a native plan archive, and the later reuse record must be explicit.
    Nothing here mutates history or accepts scientific output.
    """
    from .presentation_continuation import _canonical as canonical_receipt
    keys = receipt.get('scientific_source_steps') or []
    approved_at = approval.get('approved_at')
    if not keys or len(keys)>128 or type(approved_at) not in (int,float):return False
    fields=('status','token','updated_at','result_ref','execution_fingerprint','artifacts',
            'node_verification','origin_goal_version','execution_plan_version')
    nodes=state.get('nodes') or {}
    for archive in reversed((state.get('plan_history') or [])[-32:]):
        old=archive.get('nodes') or {}; candidate={}; changed=False
        for key in keys:
            current=nodes.get(key) or {}; previous=old.get(key) or {}
            observed={field:current.get(field) for field in fields}
            if current.get('updated_at')!=previous.get('updated_at'):
                basis=current.get('reuse_basis') or {}; checked=basis.get('checked_at');stamp=previous.get('updated_at')
                if (basis.get('kind')!='verified_completed_submission' or basis.get('no_resubmission') is not True
                    or basis.get('original_execution_fingerprint')!=current.get('execution_fingerprint')
                    or type(checked) not in (int,float) or type(stamp) not in (int,float)
                    or not stamp<=approved_at<=checked
                    or any(current.get(field)!=previous.get(field) for field in fields if field!='updated_at')):
                    break
                observed['updated_at']=stamp;changed=True
            candidate[key]=observed
        else:
            if changed and hashlib.sha256(canonical_receipt(candidate).encode()).hexdigest()==receipt.get('source_receipts_sha256'):
                return True
    return False


def canonical_presentation_runtime(state, goal, session_root, jobs):
    """Resolve only previously executed budget metadata by its native proof.

    The original declaration stays in the store. The pure classifier receives
    an in-memory view of the exact allocation already recorded in the current
    approved goal; this neither changes a budget nor creates an execution.
    """
    from .native_resource_carry import prove_native_resource_carry, _metadata_only
    canonical = None
    approved = {item['step_id']: item for item in goal.get('approved_nodes', [])}
    for key, node in state.get('nodes', {}).items():
        incoming = approved.get(key)
        if (incoming is None or node.get('contract', {}).get('arguments') == incoming.get('arguments')
                or not _metadata_only(node.get('contract', {}), incoming)):
            continue
        try:
            prove_native_resource_carry(state, key, incoming, session_root, goal, jobs)
        except (ValueError, PermissionError, OSError, TypeError, KeyError):
            continue  # The ordinary pure classifier still rejects the mismatch.
        if canonical is None: canonical = copy.deepcopy(state)
        canonical['nodes'][key]['contract'] = copy.deepcopy(incoming)
    return state if canonical is None else canonical


def transition_context(state, goal, line, approval, *, project_root, owner, jobs):
    """Recompute the already-approved typed delta against the old native DAG."""
    transition = (approval or {}).get('presentation_transition')
    if not transition or state.get('plan_version') == goal.get('approved_plan_version'):
        return None
    from .parallel_workflow import _valid_execution_approval
    if (not _valid_execution_approval(approval, goal, line, owner['username'], owner['conv_id'])
            or approval.get('base_plan_version') != state.get('plan_version')
            or goal.get('approved_plan_version', 0) <= state.get('plan_version', 0)
            or state.get('user_paused') or state.get('pause_kind') not in {None, 'failure', 'reconciliation'}):
        raise ValueError('presentation history carry lacks current native plan approval')
    receipt = transition.get('receipt') or {}; changes = transition.get('changes')
    source = approval.get('control_source') or {}
    if (source.get('turn_id') != receipt.get('control_turn_id')
            or source.get('goal_version') != goal.get('version')
            or receipt.get('goal_version') != goal.get('version')):
        raise ValueError('presentation history carry control turn is stale')
    old_goal = state.get('goal_contract') or {}
    if any(old_goal.get(key) != goal.get(key) for key in (
            'method', 'gases', 'parameters', 'execution_mode', 'execution_authorized')):
        raise ValueError('presentation history carry cannot change scientific scope')
    view = canonical_presentation_runtime(state, old_goal,
        Path(project_root) / 'runs' / owner['username'] / owner['conv_id'], jobs)
    old = [copy.deepcopy(node['contract']) for node in view.get('nodes', {}).values()]
    fresh = classify_presentation_continuation(old, changes, project_root=project_root,
        owner=owner, goal=old_goal, runtime=view)
    expected_receipt={key:value for key,value in receipt.items() if key!='control_turn_id'}
    if (fresh.get('source_receipts_sha256')!=expected_receipt.get('source_receipts_sha256')
            and _self_recorded_reuse_receipt(state,receipt,approval)):
        fresh={**fresh,'source_receipts_sha256':expected_receipt['source_receipts_sha256']}
    if (not fresh['technical_delivery_only'] or not _same(fresh,expected_receipt)):
        raise ValueError('presentation history carry typed source or graph delta changed')
    expected, _ = patched_graph(old, changes, project_root)
    expected = {node['step_id']: execution_contract(node) for node in expected}
    approved = {node['step_id']: execution_contract(node) for node in goal.get('approved_nodes', [])}
    if not _same(expected, approved):
        raise ValueError('presentation history carry differs from the approved complete DAG')
    return {'kind': 'approved_presentation_transition', 'owner': dict(owner),
        'from_plan_version': state['plan_version'], 'to_plan_version': goal['approved_plan_version'],
        'approval_sha256': _digest(approval), 'transition_sha256': _digest(transition),
        'no_execution_authority': True, 'no_scientific_change': True}


def completed_transition_proof(state, step_id, session_root, *, upstream_jobs):
    """Prove exact accepted data, using its original approval and receipt."""
    owner = {key: state.get(key) for key in ('username', 'conv_id')}
    node = state.get('nodes', {}).get(step_id, {}); contract = node.get('contract') or {}
    if node.get('status') != 'succeeded' or node.get('phase') != 'finish' or not node.get('token'):
        raise ValueError('only accepted stopped history can be retained')
    tool = contract.get('tool'); goal = state['goal_contract']
    if tool in PRESENTATION_TOOLS:
        from .completed_presentation import verify_completed_presentation
        proof = verify_completed_presentation(state, step_id, session_root, goal)
    elif tool in {'inspect_path', 'validate_framework_charges', 'query_literature'}:
        from .completed_readonly import verify_completed_readonly
        proof = verify_completed_readonly(state, step_id, session_root, goal, upstream_jobs=upstream_jobs)
    elif tool in {'analyze_gcmc_screening', 'analyze_cdft_screening', 'download_scientific_file'} or (
            tool == 'run_cdft' and contract.get('arguments', {}).get('action') == 'inputs'):
        files = _OwnedFiles(session_root, owner); facts = []
        try:
            ref = node.get('result_ref') or {}; call_id = ref.get('call_id')
            if not isinstance(call_id, str) or not call_id.isalnum() or ref.get('evidence_path') != str(
                    files.root / 'evidence' / (call_id + '.json')):
                raise ValueError('completed preparation native call identity differs')
            call, fact = files.read(ref['evidence_path'], as_json=True); facts.append(fact)
            result = result_object(call.get('result'))
            if (call.get('call_id') != call_id or call.get('tool') != tool or call.get('failed') is not False
                    or not _same(call.get('params'), node.get('effective_arguments') or contract.get('arguments'))
                    or not _same(result, result_object(node.get('result')))):
                raise ValueError('completed preparation native producer changed')
            facts.append(checkpoint_call_witness(files, ref.get('checkpoint_path'), call,
                step_id=step_id, require_result_hash=True))
            if tool == 'run_cdft':
                manifest = result.get('input_manifest') or {}; directory = files.owned(contract['arguments']['input_dir'])
                if (manifest.get('schema_version') != 1 or result.get('submitted') is not False
                        or manifest.get('n_inputs') != result.get('n_inputs')
                        or not _same(result.get('input_generation_arguments'), contract['arguments'])):
                    raise ValueError('completed cDFT inputs lack their original recipe')
                items = manifest.get('items') or []
                if len(items) != result.get('n_inputs') or not 1 <= len(items) <= 512:
                    raise ValueError('completed cDFT input membership differs')
                expected = {str(files.owned(directory / item['path'])) for item in items}
                actual = {str(path) for path in directory.glob('*.dat')}
                if expected != actual:
                    raise ValueError('completed cDFT input membership changed')
                outputs = [(directory / item['path'], item['sha256']) for item in items]
            elif tool == 'download_scientific_file':
                target = contract['arguments'].get('output_path')
                if (not isinstance(target,str) or result.get('status') != 'success'
                        or result.get('output_path') != target or result.get('output_files') != [target]
                        or result.get('existing_results_overwritten') is not False
                        or not isinstance(result.get('sha256'),str) or len(result['sha256']) != 64
                        or type(result.get('bytes')) is not int or result['bytes'] <= 0):
                    raise ValueError('completed download lacks exact immutable source receipt')
                _, output_fact = files.read(target)
                if output_fact['size'] != result['bytes'] or output_fact['sha256'] != result['sha256']:
                    raise ValueError('completed downloaded document bytes changed')
                facts.append(output_fact)
                outputs = []
            else:
                records = result.get('output_facts')
                if not isinstance(records, dict) or not 1 <= len(records) <= 64:
                    raise ValueError('completed analysis lacks immutable output hashes')
                outputs = [(path, fact['sha256']) for path, fact in records.items()]
                # Analysis input manifests also remain bound to exact source bytes.
                source_facts = result.get('source_facts') or []
                if isinstance(source_facts, dict):
                    source_facts = [{'path': key, **value} for key, value in source_facts.items()]
                for item in source_facts:
                    if isinstance(item, dict) and item.get('path') and item.get('sha256'):
                        outputs.append((item['path'], item['sha256']))
            if len(outputs) > 512:
                raise ValueError('completed preparation evidence exceeds bounded membership')
            for path, expected in outputs:
                _, fact = files.read(path); facts.append(fact)
                if fact['sha256'] != expected:
                    raise ValueError('completed preparation source or output bytes changed')
            proof = {'kind': 'verified_completed_preparation', 'facts': facts,
                'current_token': node['token'], 'no_reexecution': True}
            proof['proof_sha256'] = _digest(proof)
        finally:
            files.close()
    else:
        raise ValueError('completed local tool has no native carry proof')
    return {'kind': 'presentation_transition_history', **owner, 'step_id': step_id,
        'current_token': node['token'], 'node_identity_sha256': _digest({key: node.get(key) for key in (
            'contract', 'token', 'attempt_id', 'result', 'result_ref', 'artifacts', 'execution_fingerprint')}),
        'source_proof_sha256': proof['proof_sha256'],
        'delivery_class': proof.get('delivery_class', 'existing_native_result'),
        'no_execution_authority': True, 'no_reexecution': True}


def retire_unproved_presentation_history(existing_steps, changes, *, runtime, session_root):
    """Propose archival of unusable old presentation components; never write."""
    if (not changes or any(change.get('operation') != 'upsert' or
            (change.get('node') or {}).get('tool') not in PRESENTATION_TOOLS for change in changes)):
        return changes, []
    edited = {change['step_id'] for change in changes}; old = {node['step_id']: node for node in existing_steps}
    candidates = {key for key, node in old.items() if key not in edited and node.get('tool') in PRESENTATION_TOOLS}
    bad = set()
    from .completed_presentation import verify_completed_presentation
    for key in candidates:
        if runtime.get('nodes', {}).get(key, {}).get('status') != 'succeeded':
            continue
        try:
            verify_completed_presentation(runtime, key, session_root, runtime['goal_contract'])
        except (ValueError, PermissionError, OSError, KeyError, TypeError):
            bad.add(key)
    # Retire complete old presentation components; never leave orphan compiler
    # fragment/assembly metadata or silently remove a scientific consumer.
    retired = set(bad)
    while True:
        expanded = retired | {key for key in candidates if (
            any(dep in retired for dep in old[key].get('depends_on', [])) or
            any(key in old[other].get('depends_on', []) for other in retired))}
        if expanded == retired: break
        retired = expanded
    if any(any(dep in retired for dep in (change.get('node') or {}).get('depends_on', [])) for change in changes):
        raise ValueError('new presentation references an unproved historical report; reauthor from its original native data')
    if any(node.get('tool') not in PRESENTATION_TOOLS and any(dep in retired for dep in node.get('depends_on', []))
           for node in old.values()):
        raise ValueError('unproved presentation has a scientific consumer; explicit plan repair is required')
    if any(runtime['nodes'][key].get('status') in {'running', 'prefinish', 'uncertain', 'waiting_jobs', 'waiting_prerequisite'}
           for key in retired):
        raise ValueError('historical presentation still owns an active attempt')
    return (copy.deepcopy(changes) + [{'operation': 'remove', 'step_id': key} for key in sorted(retired)], sorted(retired))


def _approved_noop_rebase(state, goal, line, approval):
    """Rejoin an approved, unchanged graph after unactivated plan revisions.

    This only recognizes bookkeeping revisions: it neither accepts any
    artifact nor authorizes execution. The normal native carry proofs still
    validate every original job, source, figure and visual receipt.
    """
    from .goal_contract import prohibits_submission, scope_preserving_continuation
    from .parallel_workflow import is_submission
    old = state.get('goal_contract') or {}
    start, base, target = state.get('plan_version'), approval.get('base_plan_version'), goal.get('approved_plan_version')
    recognized_pause=(state.get('pause_kind'),state.get('pause_reason')) in {
        ('input_changed','cached inputs/tool sources/artifacts changed; approve revised plan'),
        ('failure','failed node requires diagnosis and plan review')}
    if (any(type(v) is not int for v in (start, base, target)) or not start < base < target
            or target - start > 8 or not recognized_pause):
        return False
    previous = old.get('directives') or []; current = goal.get('directives') or []
    if (not isinstance(previous, list) or not isinstance(current, list) or current[:len(previous)] != previous
            or not current[len(previous):] or not all(isinstance(v, str) and not prohibits_submission(v)
                                                     and not re.search(r'(?:不要|停止|不再|禁止|暂停).{0,8}(?:继续|接着|恢复|推进|执行)|\b(?:do not|don\x27t|stop|never)\s+(?:resume|continue|execute)\b', v, re.I)
                                                     and scope_preserving_continuation(v, old.get('approved_nodes') or [])
                                                     for v in current[len(previous):])):
        return False
    old_nodes = old.get('approved_nodes') or []; new_nodes = goal.get('approved_nodes') or []
    before = {node.get('step_id'): execution_contract(node) for node in old_nodes}
    after = {node.get('step_id'): execution_contract(node) for node in new_nodes}
    if (not before or None in before or len(before) != len(old_nodes) or len(after) != len(new_nodes)
            or not _same(before, after)):
        return False
    supported = PRESENTATION_TOOLS | {'inspect_path', 'validate_framework_charges', 'query_literature',
        'analyze_gcmc_screening', 'analyze_cdft_screening', 'download_scientific_file'}
    for node in (state.get('nodes') or {}).values():
        contract = node.get('contract') or {}; tool = contract.get('tool'); args = contract.get('arguments') or {}
        if (node.get('status') == 'succeeded' and not is_submission(tool, args)
                and tool not in supported and not (tool == 'run_cdft' and args.get('action') == 'inputs')):
            return False  # Never re-execute an unsupported completed local step.
    histories = line.get('plan_history') or []
    for version in range(start, base + 1):
        matches = [entry for entry in histories if entry.get('version') == version]
        if len(matches) != 1 or not isinstance(matches[0].get('steps'), list):
            return False
        steps = matches[0]['steps']; by_id = {entry.get('step_id'): entry for entry in steps}
        if len(by_id) != len(steps):return False
        for key,contract in before.items():
            if key not in by_id:return False
            historical=by_id[key]; prior=execution_contract(historical)
            if _same(contract,prior):continue
            # A reverted, never-entered failed report edit has no accepted
            # artifact to carry. All completed science/presentation contracts,
            # dependencies and output paths remain exactly unchanged.
            current=(state.get('nodes') or {}).get(key) or {}
            attempts=[current]+list(current.get('attempt_history') or [])
            if (version<=start or contract.get('tool')!='generate_scientific_report'
                    or prior.get('tool')!='generate_scientific_report'
                    or current.get('status')!='failed' or historical.get('status')!='pending'
                    or historical.get('done') or historical.get('job_ids') or historical.get('result_ref')
                    or historical.get('execution_plan_version') or historical.get('token')
                    or historical.get('node_verification') or historical.get('output_files')
                    or {k:v for k,v in contract.items() if k!='arguments'}!={k:v for k,v in prior.items() if k!='arguments'}
                    or any(a.get('execution_plan_version')==version for a in attempts)):
                return False
    return True


def report_only_history_context(state, goal, line, approval, *, owner):
    """Bind unchanged accepted presentation to an authorized report-only delta.

    Operational report repairs need not satisfy the fresh-output presentation
    classifier. They still require the exact native approval and unchanged
    scientific graph; this context grants no execution or cache authority.
    """
    from .parallel_workflow import _valid_execution_approval, _ACTIVATION_SCOPE
    old_goal = state.get('goal_contract') or {}
    rebase = isinstance(approval, dict) and _approved_noop_rebase(state, goal, line, approval)
    if (state.get('plan_version') == goal.get('approved_plan_version')
            or not isinstance(approval, dict) or approval.get('kind') != 'main_model_applied_plan'
            or not _valid_execution_approval(approval, goal, line, owner['username'], owner['conv_id'])
            or {key: state.get(key) for key in owner} != owner
            or state.get('user_paused') or (state.get('pause_kind') not in {None, 'failure', 'reconciliation'} and not rebase)
            or old_goal.get('approved_plan_version') != state.get('plan_version')
            or (approval.get('base_plan_version') != state.get('plan_version') and not rebase)
            or goal.get('execution_authorized') is not True
            or any(old_goal.get(key) != goal.get(key) for key in
                   (*_ACTIVATION_SCOPE, 'active_goal', 'original_goal'))
            or (old_goal.get('directives') != goal.get('directives') and not rebase)):
        return None
    nodes = state.get('nodes') or {}
    old_nodes = old_goal.get('approved_nodes') or []
    old_approved = {node.get('step_id'): node for node in old_nodes}
    if (len(old_approved) != len(old_nodes) or set(old_approved) != set(nodes)
            or any(not _same(execution_contract(old_approved[key]), execution_contract(node.get('contract', {})))
                   for key, node in nodes.items())):
        return None
    approved = goal.get('approved_nodes') or []
    incoming = {node.get('step_id'): node for node in approved}
    if not nodes or len(incoming) != len(approved) or None in incoming:
        return None
    changed = []
    retained = []
    for key in set(nodes) | set(incoming):
        before = nodes.get(key, {}).get('contract')
        after = incoming.get(key)
        if before is not None and after is not None and _same(execution_contract(before), execution_contract(after)):
            carry_tools = PRESENTATION_TOOLS | {'inspect_path', 'validate_framework_charges', 'query_literature',
                'analyze_gcmc_screening', 'analyze_cdft_screening', 'download_scientific_file'} if rebase else PRESENTATION_TOOLS
            cdft_inputs = rebase and before.get('tool') == 'run_cdft' and (before.get('arguments') or {}).get('action') == 'inputs'
            if ((before.get('tool') in carry_tools or cdft_inputs)
                    and nodes[key].get('status') == 'succeeded' and nodes[key].get('phase') == 'finish'):
                retained.append(key)
            continue
        # Removing, introducing, reassigning or changing any non-report node
        # (including a figure) is outside this narrow report-only history path.
        if any(node is not None and node.get('tool') != 'generate_scientific_report'
               for node in (before, after)):
            return None
        changed.append(key)
    if (not changed and not rebase) or not retained:
        return None
    return {'kind': 'approved_report_only_history', 'owner': dict(owner),
        'from_plan_version': state['plan_version'], 'to_plan_version': goal['approved_plan_version'],
        'approval_sha256': _digest(approval), 'changed_report_steps': sorted(changed),
        'retained_steps': sorted(retained), 'no_execution_authority': True,
        'no_scientific_change': True}


def completed_report_only_history_proof(state, step_id, session_root, *, upstream_jobs):
    """Reuse original native presentation, inputs and private visual evidence."""
    base = completed_transition_proof(state, step_id, session_root, upstream_jobs=upstream_jobs)
    from .report_authoring import _read
    from .report_native_inputs import verified_visual_evidence
    root = Path(session_root); budget = {'bytes': 0}; facts = []; visuals = []; cache = {}; jobs = []
    pending = [step_id]; seen = set()
    while pending:
        key = pending.pop()
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 128:
            raise ValueError('completed presentation history exceeds bounded ancestry')
        node = state.get('nodes', {}).get(key, {})
        if node.get('status') != 'succeeded':
            raise ValueError('completed presentation history source is no longer accepted')
        contract = node.get('contract') or {}
        for job_id in node.get('job_ids') or []:
            job = upstream_jobs.get(str(job_id)) or {}
            if ((job.get('username'), job.get('conv_id')) != (state.get('username'), state.get('conv_id'))
                    or str(job.get('job_id')) != str(job_id) or job.get('attempt_id') != node.get('attempt_id')
                    or job.get('tool') != contract.get('tool') or job.get('terminal') is not True
                    or job.get('failed') is not False or job.get('state') != 'COMPLETED'):
                raise ValueError('accepted presentation source job is no longer the original owned successful attempt')
            jobs.append({key: job.get(key) for key in ('job_id', 'username', 'conv_id', 'attempt_id',
                'tool', 'terminal', 'failed', 'state')})
        if contract.get('tool') not in PRESENTATION_TOOLS:
            continue
        result = result_object(node.get('result'))
        inputs = result.get('source_inputs') or []
        if not isinstance(inputs, list) or len(inputs) > 64:
            raise ValueError('completed presentation has invalid original input membership')
        for item in inputs:
            if not isinstance(item, dict) or not item.get('path') or not item.get('sha256'):
                raise ValueError('completed presentation original input identity is absent')
            _, fact = _read(root, item['path'], budget)
            if fact['sha256'] != item['sha256']:
                raise ValueError('completed presentation original source bytes changed')
            facts.append(fact)
        if contract.get('tool') == 'generate_result_figures':
            for artifact in result.get('artifacts') or []:
                _, fact = _read(root, artifact['path'], budget)
                if fact['sha256'] != artifact['sha256']:
                    raise ValueError('accepted figure output changed')
                facts.append(fact)
                visual = verified_visual_evidence(root, state, key, fact, read=_read,
                    budget=budget, file_facts=facts, cache=cache)
                if visual.get('status') != 'verified':
                    raise ValueError('accepted figure lacks its original independently verified visual evidence')
                visuals.append(visual)
        pending.extend(contract.get('arguments', {}).get('source_steps') or [])
    result = {**base, 'source_proof_sha256': _digest({'native': base['source_proof_sha256'],
        'original_input_facts': facts, 'visual_evidence': visuals, 'source_jobs': jobs}),
        'verified_visual_count': len(visuals)}
    return result
