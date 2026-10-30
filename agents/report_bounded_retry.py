"""One bounded native report retry after a changed verified capability epoch.

This does not assert that a source-code change fixed the previous failure.
The old attempt is retained, its old implementation identity remains unknown,
and ordinary report authoring, independent review and publication gates remain
the only way to produce a result.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import stat


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def bounded_retry_history(node):
    """Preserve a node's automatic retry budget across native plan archives."""
    pending = [node]; seen = 0; records = {}
    while pending:
        current = pending.pop(); seen += 1
        if seen > 128 or not isinstance(current, dict): raise ValueError('report retry history is malformed or too large')
        entries = current.get('report_retry_epochs') or []
        older = current.get('attempt_history') or []
        if not isinstance(entries, list) or not isinstance(older, list): raise ValueError('report retry history is malformed')
        for item in entries:
            if not isinstance(item, dict) or not isinstance(item.get('capability_revision'), str):
                raise ValueError('report retry history entry is malformed')
            key = item['capability_revision']
            if key in records and records[key] != item: raise ValueError('conflicting retry history for capability')
            records[key] = copy.deepcopy(item)
        pending.extend(older)
    return [records[key] for key in sorted(records)]


def _missing_owned_output(files, target):
    """Prove absence without following links in any existing path component."""
    path = files.owned(target)
    parts = path.relative_to(files.root).parts
    parent = os.dup(files.fd)
    try:
        for name in parts[:-1]:
            try:
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            except FileNotFoundError:
                return True
            os.close(parent); parent = child
        try:
            os.stat(parts[-1], dir_fd=parent, follow_symlinks=False)
        except FileNotFoundError:
            return True
        return False
    finally:
        os.close(parent)


def _source_state_and_jobs(runtime, state, step_id):
    """Bind complete ancestor nodes and terminal job facts without job locks."""
    nodes = state.get('nodes') or {}; pending = list((nodes.get(step_id, {}).get('contract') or {}).get('depends_on') or [])
    seen = set(); jobs = {}
    while pending:
        key = pending.pop()
        if key in seen: continue
        if len(seen) >= 128: raise ValueError('report ancestry exceeds bound')
        node = nodes.get(key)
        if not isinstance(node, dict) or node.get('status') != 'succeeded': raise ValueError('report ancestor is not completed')
        seen.add(key); pending.extend((node.get('contract') or {}).get('depends_on') or [])
    ids = {str(job) for key in seen for job in nodes[key].get('job_ids') or []}
    if ids:
        watch = getattr(runtime, 'job_watch', None); path = Path(getattr(watch, '_watch_file', ''))
        if not path.is_absolute(): raise ValueError('native job watch file is unavailable')
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > 64 * 1024**2:
                raise ValueError('job watch file is not bounded regular data')
            with os.fdopen(fd, 'rb', closefd=False) as stream: raw = stream.read(64 * 1024**2 + 1)
            after = os.fstat(fd); linked = os.stat(path, follow_symlinks=False)
            identity = lambda v: (v.st_dev, v.st_ino, v.st_size, v.st_mtime_ns, v.st_ctime_ns, v.st_mode)
            if len(raw) != before.st_size or identity(before) != identity(after) or identity(after) != identity(linked):
                raise ValueError('job watch changed during source proof')
            observed = json.loads(raw)
        finally: os.close(fd)
        for key in seen:
            node = nodes[key]
            for job_id in node.get('job_ids') or []:
                job = observed.get(str(job_id)) or {}
                if ((job.get('username'), job.get('conv_id')) != (runtime.username, runtime.conv_id)
                        or str(job.get('job_id')) != str(job_id) or job.get('attempt_id') != node.get('attempt_id')
                        or job.get('recovery_key') != node.get('recovery_key')
                        or job.get('tool') != node.get('contract', {}).get('tool')
                        or job.get('terminal') is not True or job.get('failed') is not False
                        or job.get('state') != 'COMPLETED'):
                    raise ValueError('report ancestor job is not the original owned successful attempt')
                jobs[str(job_id)] = {k: job.get(k) for k in ('job_id', 'username', 'conv_id', 'attempt_id',
                    'recovery_key', 'tool', 'terminal', 'failed', 'state', 'work_dir')}
    fields = ('contract', 'status', 'token', 'attempt_id', 'job_ids', 'recovery_key', 'result_ref',
              'result', 'execution_fingerprint', 'artifacts', 'effective_arguments')
    return {'nodes': {key: {field: copy.deepcopy(nodes[key].get(field)) for field in fields} for key in sorted(seen)},
            'jobs': jobs}


def capture_report_retry(runtime, state, step_id, *, control_valid, prepared_packet=None, return_packet=False, user_authorization=None):
    """Prove a failed report. Initial packet may publish immutable metadata.

    A prepared packet is only re-read and rehashed; it never publishes metadata
    and is suitable for the final WorkflowStore transaction revalidation.
    """
    from .failure_recovery import FailureRecovery, diagnostic_revision
    from .installations import inspect_installation
    from .executor_identity import executor_identity
    from .report_authoring import build_result_packet, sources_unchanged
    from .report_recovery import _context, _same
    from .local_preparation_recovery import checkpoint_call_witness
    from .recovery import result_object
    from .workflow_patch import execution_contract
    from .watch_context import get_context
    from . import report_authoring, report_source_metadata
    from .goal_contract import prohibits_submission
    import re
    actor = runtime.main
    if not callable(control_valid) or not control_valid(): raise ValueError('user control changed')
    context = get_context()
    if (context.get('username'), context.get('conv_id')) != (runtime.username, runtime.conv_id):
        raise PermissionError('report retry owner context differs')
    if (actor._pending_workflow_patch or actor._awaiting_plan_approval
            or (actor._pending_user_interaction and actor._pending_user_interaction.get('tool') != 'lifecycle_wait')):
        raise ValueError('pending user control blocks report retry')
    owner = {'username': runtime.username, 'conv_id': runtime.conv_id}
    goal = actor.goal_contract.to_dict()
    saved = state.get('goal_contract') or {}; old = saved.get('directives') or []; new = goal.get('directives') or []
    if (not isinstance(old, list) or not isinstance(new, list) or new[:len(old)] != old
            or any(not isinstance(v, str) or prohibits_submission(v)
                   or re.search(r'(?:不要|停止|不再|禁止|暂停).{0,8}(?:继续|接着|恢复|推进|执行)|\b(?:do not|don\x27t|stop|never)\s+(?:resume|continue|execute)\b', v, re.I)
                   for v in new[len(old):])):
        raise ValueError('new user control does not permit report execution')
    observed=(state.get('nodes') or {}).get(step_id) or {};plan_bridge=None
    if user_authorization is not None and observed.get('execution_plan_version')!=state.get('plan_version'):
        from .report_failure_plan_bridge import prove_failure_plan_bridge
        from .report_recovery import _OwnedFiles
        from .workflow_execution_identity import reconciled_origin_version
        from .parallel_workflow import _approval_digest
        plan_bridge=prove_failure_plan_bridge(state,step_id,goal,runtime._line(),
            (getattr(actor,'context',{}) or {}).get('approved_execution_proof'),owner)
        node=observed;contract=node.get('contract') or {}
        approved=[n for n in goal.get('approved_nodes',[]) if n.get('step_id')==step_id]
        if (goal.get('execution_authorized') is not True or goal.get('execution_mode')!='workflow'
                or _approval_digest(goal)!=_approval_digest(saved)
                or reconciled_origin_version(state,node.get('origin_goal_version'),goal) is None
                or len(approved)!=1 or execution_contract(approved[0])!=execution_contract(contract)
                or node.get('job_ids')):
            raise ValueError('current report approval or goal changed')
        files=_OwnedFiles(runtime.root,owner)
    else:
        node, contract, _, files, _ = _context(state, step_id, runtime.root, goal)
    try:
        args = contract.get('arguments') or {}; output = args.get('output_path'); result = result_object(node.get('result'))
        legacy_connection=(user_authorization is not None and plan_bridge is not None
            and result.get('error')=='LLM API 连接失败（Connection error.）'
            and isinstance(result.get('traceback'),str)
            and bool(re.search(r'(?m)^agents\.session\.LLMUnavailableError: LLM API 连接失败（Connection error\.）\s*$',result['traceback'])))
        structured_failure=(result.get('error_kind')=='report_authoring_failed' and result.get('executed') is False
            and result.get('report_published') is False and result.get('scientific_acceptance') is False)
        if (node.get('status') != 'failed' or contract.get('tool') != 'generate_scientific_report'
                or contract.get('agent') != 'communicator' or node.get('tool_returned') is not True
                or node.get('dispatch_phase') != 'entered_tool' or node.get('job_ids') or node.get('attempt_id')
                or not isinstance(output, str) or len(contract.get('expected_outputs') or []) != 1
                or files.owned(contract['expected_outputs'][0]) != files.owned(output)
                or not (structured_failure or legacy_connection)
                or not _missing_owned_output(files, output)):
            raise ValueError('failed report is not a stopped unpublished owned attempt')
        future = runtime.futures.get(step_id)
        if future is not None and not future.done(): raise ValueError('original report worker still active')
        prior_pid = node.get('owner_pid')
        if type(prior_pid) is not int or prior_pid <= 0: raise ValueError('original report process identity is unknown')
        if user_authorization is None:
            if prior_pid != os.getpid():
                try: os.kill(prior_pid, 0)
                except ProcessLookupError: pass
                except (PermissionError, OSError): raise ValueError('original report process cannot be checked') from None
                else: raise ValueError('original report owner process is still alive')
            elif future is None:
                raise ValueError('same-process completed worker ownership is not independently established')
        # A live backend PID may own many unrelated requests. Explicit-user
        # retry still requires the exact settled worker_failed event below and
        # the cross-process lease check under WorkflowStore commit.
        machine = FailureRecovery(runtime)
        if not machine._authority(state, step_id): raise ValueError('report failure authority changed')
        source = machine._source(state, step_id)
        ref = node.get('result_ref') or {}; call, _ = files.read(ref['evidence_path'], as_json=True)
        if call.get('failed') is not True or not _same(result_object(call.get('result')), result):
            raise ValueError('native failure receipt differs')
        witness = checkpoint_call_witness(files, ref['checkpoint_path'], call, step_id=step_id, require_result_hash=True)
        # A returned tool call alone is insufficient across hosts. The native
        # failed event is emitted after _settle records failed and releases the
        # lease; bind its exact workflow/step/token/attempt/call identity.
        stopped = []
        for event in (state.get('outbox') or {}).values():
            payload = event.get('payload') or {}
            if (event.get('kind') == 'worker_failed' and payload.get('runtime_id') == runtime.workflow_id
                    and payload.get('workflow_id') == actor._current_line_id and payload.get('step_id') == step_id
                    and payload.get('correlation_token') == node['token'] and payload.get('attempt_id') == node.get('attempt_id')
                    and payload.get('plan_version') == (plan_bridge['from_plan_version'] if plan_bridge else state['plan_version'])
                    and payload.get('origin_goal_version') == node.get('origin_goal_version')
                    and _same(payload.get('result_ref'), ref)
                    and (payload.get('evidence_call') or {}).get('call_id') == source['call_id']):
                stopped.append(event)
        if len(stopped) != 1: raise ValueError('native returned failure settlement is absent or ambiguous')
        revision = diagnostic_revision('generate_scientific_report')
        incidents = [v for v in (state.get('failure_recovery') or {}).get('incidents', {}).values()
                     if v.get('step_id') == step_id and v.get('token') == node.get('token')
                     and v.get('phase') == 'blocked' and v.get('diagnostic_revision') == revision
                     and machine._incident_current(state, v)]
        if len(incidents) != 1: raise ValueError('current blocked native report incident is not unique')
        older = [v.get('diagnostic_revision') for v in incidents[0].get('history', [])
                 if isinstance(v, dict) and v.get('diagnostic_revision') and v.get('diagnostic_revision') != revision]
        history = bounded_retry_history(node)
        if user_authorization is None:
            if not older: raise ValueError('no earlier different native diagnostic capability epoch')
            if len(history) >= 2 or any(v.get('capability_revision') == revision for v in history):
                raise ValueError('bounded retry already consumed for current capability')
        else:
            required={'kind','owner','step_id','plan_version','failure_call','failure_token','request_id','confirmed'}
            if (not isinstance(user_authorization,dict) or set(user_authorization)!=required
                    or user_authorization.get('kind')!='authenticated_user_report_retry'
                    or user_authorization.get('owner')!=owner or user_authorization.get('step_id')!=step_id
                    or user_authorization.get('plan_version')!=state['plan_version']
                    or user_authorization.get('failure_call')!=source['call_id']
                    or user_authorization.get('failure_token')!=node['token']
                    or user_authorization.get('confirmed') is not True
                    or not isinstance(user_authorization.get('request_id'),str)
                    or not re.fullmatch(r'[A-Za-z0-9_-]{16,128}',user_authorization['request_id'])):
                raise ValueError('user report retry authorization does not match the native failure')
            # The durable cross-workflow authorization ledger is checked
            # atomically by WorkflowStore together with lease and proof CAS.
        check = inspect_installation('generate_scientific_report', copy.deepcopy(args), actor.config)
        if check.get('ok') is not True or check.get('kind') != 'internal_python_source_identity':
            raise ValueError('current loaded report capability is not source-identical')
        functions = (report_authoring.build_result_packet, report_authoring.render_validated_draft,
                     report_authoring._scientific_claim_clauses, report_authoring._positive_mixture_selectivity_claim,
                     report_authoring._nonquantitative_literature_denial, report_source_metadata.metadata_tables,
                     report_source_metadata.cdft_document)
        code = [executor_identity(function) for function in functions]
        if any(v.get('verified') is not True or v.get('matches') is not True for v in code):
            raise ValueError('loaded report helper differs from current source')
        def attest(producer, original_call, raw):
            source_path = producer['result_ref']['checkpoint_path']
            proof = checkpoint_call_witness(files, source_path, original_call,
                step_id=producer['contract']['step_id'], require_result_hash=True)
            return {'checkpoint_verified': True, 'checkpoint_sha256': proof['sha256'],
                    'checkpoint_path': source_path, 'producer_result_digest_verified': True,
                    'quantity_contract': {}}
        source_state = _source_state_and_jobs(runtime, state, step_id)
        lookup = runtime.job_watch.get if runtime.job_watch else None
        packet = prepared_packet
        if packet is None:
            # This native builder may create content-addressed immutable
            # metadata sidecars; it is called only before taking the store lock.
            packet = build_result_packet(runtime.root, owner, state, step_id, attest=attest,
                                         control_valid=control_valid, job_lookup=lookup)
        if (packet.get('owner') != owner or packet.get('report_step') != step_id
                or packet.get('report_token') != node['token']):
            raise ValueError('prepared report packet identity changed')
        if not sources_unchanged(runtime.root, packet) or not control_valid():
            raise ValueError('report sources or user control changed')
        identity = {'kind': 'bounded_report_reauthor', 'owner': owner, 'step_id': step_id,
            'token': node['token'], 'goal_sha256': _digest(goal), 'plan_version': state['plan_version'],
            'user_pause_revision': state.get('user_pause_revision',0),
            'contract_sha256': _digest(execution_contract(contract)), 'failure_call': source['call_id'],
            'failure_call_sha256': source['call_sha256'], 'failure_checkpoint_sha256': witness['sha256'],
            'native_settlement_event_sha256': _digest(stopped[0]), 'prior_owner_host': 'unknown',
            'source_packet_sha256': packet['packet_sha256'], 'source_state_sha256': _digest(source_state),
            'retry_history_sha256': _digest(history),
            'capability_revision': revision,
            'earlier_diagnostic_revision': older[-1] if older else None, 'implementation': [
                {k:v[k] for k in ('qualified_name','source_sha256','loaded_code_sha256','source_code_sha256')} for v in code],
            'output': str(files.owned(output)), 'prior_implementation_identity': 'unknown',
            'fix_effectiveness': 'unknown', 'no_scientific_resubmission': True,
            'report_not_published': True, 'scientific_acceptance': False,
            'metadata_sidecars_may_have_been_created_before_commit': True}
        if user_authorization is not None:
            identity['kind']='authenticated_user_report_reauthor'
            identity['user_authorization']=copy.deepcopy(user_authorization)
        if plan_bridge is not None:
            identity['legacy_plan_bridge']=copy.deepcopy(plan_bridge)
            identity['legacy_connection_failure']=bool(legacy_connection)
        if user_authorization is not None:
            native_sources=packet.get('sources') or []
            lineage=sorted((item.get('call_id'),item.get('source_sha256')) for item in native_sources if isinstance(item,dict))
            if not lineage or any(not isinstance(a,str) or not isinstance(b,str) for a,b in lineage):
                raise ValueError('report retry source lineage is incomplete')
            identity['source_lineage_sha256']=_digest({'owner':owner,'sources':lineage,'mode':args.get('report_mode')})
            identity['output_identity_sha256']=_digest({'owner':owner,'output':identity['output']})
            # Distinct same-source reports may use their own allowance only
            # when their exact execution contracts coexisted in this already
            # approved plan. Later title/output churn cannot create a sibling.
            siblings=[]
            for approved in goal.get('approved_nodes') or []:
                approved_args=approved.get('arguments') or {}
                if (approved.get('tool')=='generate_scientific_report'
                        and approved.get('step_id')!=step_id
                        and approved_args.get('source_steps')==args.get('source_steps')
                        and approved_args.get('report_mode')==args.get('report_mode')):
                    siblings.append(_digest(execution_contract(approved)))
            if len(siblings)>128:raise ValueError('approved report sibling set exceeds bound')
            identity['approved_sibling_contract_sha256']=sorted(set(siblings))
        identity['proof_sha256'] = _digest(identity)
        return (identity, packet) if return_packet else identity
    finally:
        files.close()


def recover_report_retries(runtime, *, paused=False, max_checks=1):
    """Requeue at most one proved report; normal DAG dispatch executes later."""
    from .workflow_user_control import workflow_control_guard
    if paused: return []
    actor = runtime.main; control = workflow_control_guard(actor, runtime.username, runtime.conv_id)
    results = []
    for step, node in runtime.snapshot().get('nodes', {}).items():
        if len(results) >= max_checks: break
        if node.get('status') != 'failed' or node.get('contract', {}).get('tool') != 'generate_scientific_report': continue
        try:
            proof, packet = capture_report_retry(runtime, runtime.snapshot(), step, control_valid=control, return_packet=True)
            def revalidate(state):
                if runtime.main is not actor or not control(): raise ValueError('user control changed')
                return capture_report_retry(runtime, state, step, control_valid=control, prepared_packet=packet)
            receipt = runtime.store.requeue_bounded_report(runtime.workflow_id, step, node['token'],
                actor.goal_contract.to_dict(), {'username': runtime.username, 'conv_id': runtime.conv_id},
                proof, revalidate)
            if receipt: results.append(receipt)
        except (ValueError, PermissionError, OSError, TypeError, KeyError, AttributeError):
            continue
    return results
