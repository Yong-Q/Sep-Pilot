"""Read-only scheduler assessment and evidence-backed resource-agent review."""
import json
import re
import subprocess
import time
import uuid


def _operator_workload_profile(session, node):
    """Bind a private conservative estimate to owned scope, binary and inputs.

    This is an operator-reviewed estimate, never a measured RSS claim. Request
    workload_facts/executable_sha256 must be produced by the trusted caller.
    """
    from pathlib import Path
    from bimem_agent.paths import get_runtime_section
    facts = node.get('workload_facts') or {}
    try:
        site = get_runtime_section('resource_policy')
        if Path(site['scope_root']).resolve() != Path(session.config.project_root).resolve():
            return None
        profile = site['workload_profiles'][node['tool']]
        executable = node.get('executable_sha256')
        if (not isinstance(executable, str) or not re.fullmatch('[a-f0-9]{64}', executable)
                or executable != profile.get('executable_sha256')
                or not isinstance(profile.get('basis'), str) or not profile['basis'].strip()):
            return None
        bounds = facts.get('native_bounds') or {}
        bound_keys = ['max_grid_points', 'max_gases', 'max_sites_per_gas']
        bound_keys.extend(key for key in ('max_framework_atoms', 'max_rotations') if key in profile)
        for key in bound_keys:
            value, maximum = bounds.get(key), profile.get(key)
            if (not isinstance(value, int) or isinstance(value, bool) or value < 1
                    or not isinstance(maximum, int) or isinstance(maximum, bool) or maximum < value):
                return None
        per_worker = profile['per_worker_mb']; overhead = profile['overhead_mb']
        if (not isinstance(per_worker, int) or isinstance(per_worker, bool) or per_worker < 1
                or not isinstance(overhead, int) or isinstance(overhead, bool) or overhead < 0):
            return None
        from .workload_budget import _digest
        return {'memory_evidence': {'source': 'operator', 'basis': profile['basis'],
                    'per_worker_mb': per_worker, 'unit': 'per_worker',
                    'workload_fingerprint': facts.get('input_fingerprint')},
                'overhead_mb': overhead, 'profile': profile,
                'profile_sha256': _digest(profile), 'executable_sha256': executable,
                'estimate_kind': 'operator_conservative_estimate'}
    except (KeyError, TypeError, ValueError, AttributeError):
        return None


def _workload_allocation(session, request, node, snapshot, proposed):
    """Return final batch sizing, a planning-only preview, or a typed blocker."""
    from .workload_budget import size_workload_budget
    facts = node.get('workload_facts') or {}
    planning = request.get('review_phase') == 'planning'
    if facts.get('status') != 'ready':
        return {'status': 'planning_only' if planning else 'needs_prepared_inputs', 'candidates': []}
    bound = _operator_workload_profile(session, node)
    if bound is None:
        return {'status': 'needs_estimate' if proposed.get('memory_mb') else 'needs_memory_evidence',
                'candidates': [], 'needs_estimate': ['binary_bound_operator_memory_profile', 'verified_native_input_bounds']}
    constraints = [dict(c) for c in request.get('resource_constraints', [])
                   if c.get('step_id') in (None, node.get('step_id'))]
    for field, key in (('workers', 'worker_cap'), ('cpus', 'cpu_cap')):
        if key in bound['profile']:
            constraints.append({'source': 'operator', 'field': field, 'kind': 'max',
                                'value': bound['profile'][key], 'basis': bound['memory_evidence']['basis']})
    arguments = node['arguments']
    # A model's total-RAM guess must not suppress useful concurrency once an
    # operator-bound per-worker estimate exists. Only trusted constraints cap it.
    suggestion = {key: value for key, value in proposed.items() if key != 'memory_mb'}
    def names(value): return value.split(',') if isinstance(value, str) else list(value or [])
    # Only the authenticated server/operator may supply hard placement pins.
    # Declared plan/model placement remains a preference among eligible choices.
    try:
        all_pins = request.get('resource_pins', {})
        pins = all_pins.get(node['step_id'], {})
        if not isinstance(pins, dict) or any(key not in {'nodelist', 'partition'} for key in pins):
            raise ValueError('invalid pin keys')
        for values in pins.values():
            if (not isinstance(values, list) or not values or any(
                    not isinstance(value, str) or not re.fullmatch('[A-Za-z0-9_.-]+', value) for value in values)):
                raise ValueError('invalid pin values')
    except (AttributeError, TypeError, ValueError):
        return {'status': 'invalid_constraints', 'candidates': []}
    # This envelope is loaded from the actual native resource_health evidence,
    # never request/model partition_limits. cDFT's native adapter defaults to 72h.
    import math
    from .job_control import _wall_seconds
    partition_facts = (snapshot or {}).get('partition_facts') or {}
    observed = partition_facts.get('observed_at')
    now = time.time()
    fresh = (partition_facts.get('verified') is True
        and partition_facts.get('source') == 'scontrol_show_partition'
        and isinstance(observed, (int, float)) and not isinstance(observed, bool)
        and math.isfinite(observed) and 0 <= now-observed <= 120)
    def time_compatible(candidate):
        if not fresh: return False
        limits = (partition_facts.get('limits') or {}).get(candidate['partition']) or {}
        try:
            wanted = _wall_seconds(arguments.get('walltime', '72:00:00'))
            maximum = _wall_seconds(limits.get('max_time'))
            return limits.get('verified') is True and limits.get('state') == 'UP' and wanted > 0 and maximum >= wanted
        except (TypeError, ValueError): return False
    for for_queue in (False, True):
        sized = size_workload_budget(facts, snapshot or {}, suggested_resources=suggestion,
            constraints=constraints, memory_evidence=bound['memory_evidence'],
            overhead_mb=bound['overhead_mb'], for_queue=for_queue)
        sized['candidates'] = [candidate for candidate in sized['candidates']
            if (not pins.get('nodelist') or candidate['nodelist'] in pins['nodelist'])
            and (not pins.get('partition') or candidate['partition'] in pins['partition'])
            and time_compatible(candidate)]
        if sized['candidates'] or sized['status'] not in {'ready', 'waiting_resources'}:
            break
    # Prefer useful concurrency; model placement is a tie-breaker, not a cap.
    sized['candidates'].sort(key=lambda candidate: (
        -candidate['cpus'],
        candidate['nodelist'] not in names(arguments.get('nodelist') or proposed.get('nodelist')),
        candidate['partition'] not in names(arguments.get('partition') or proposed.get('partition')),
        candidate['nodelist'], candidate['partition']))
    sized['partition_evidence_verified'] = fresh
    if sized['status'] == 'ready' and not sized['candidates']:
        sized['status'] = 'waiting_resources'
    for candidate in sized['candidates']:
        candidate.update(budget_final=True, estimate_kind=bound['estimate_kind'],
            memory_profile_sha256=bound['profile_sha256'], executable_sha256=bound['executable_sha256'])
    return sized


def _generic_resource_allocation(request, node, snapshot, proposed):
    """Apply trusted numeric/pin controls to existing generic resource sizing."""
    import copy
    from .node_inventory import profiled_resources
    from .resource_policy import native_requirements, tool_profile
    if not snapshot: return None
    def positive(value): return isinstance(value, int) and not isinstance(value, bool) and value > 0
    hard = {field: {'exact': None, 'max': None} for field in ('cpus', 'memory_mb')}
    try:
        for control in request.get('resource_constraints', []):
            if control.get('step_id') not in (None, node['step_id']): continue
            source, field, kind = (control.get(key) for key in ('source', 'field', 'kind'))
            if (source not in {'user', 'operator', 'hint', 'default'} or field not in hard
                    or kind not in {'exact', 'max', 'preferred'}
                    or not positive(control.get('value')) or not control.get('basis')):
                return None
            if source in {'user', 'operator'} and kind != 'preferred':
                before, value = hard[field][kind], control['value']
                if kind == 'exact' and before not in (None, value): return None
                hard[field][kind] = min(before, value) if kind == 'max' and before else value
        if any(c['exact'] and c['max'] and c['exact'] > c['max'] for c in hard.values()): return None
        pins = request.get('resource_pins', {}).get(node['step_id'], {})
        if not isinstance(pins, dict) or any(key not in {'nodelist', 'partition'} for key in pins): return None
        for values in pins.values():
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or not re.fullmatch('[A-Za-z0-9_.-]+', v) for v in values): return None
        view = copy.deepcopy(snapshot)
        view['nodes'] = {name: item for name, item in view.get('nodes', {}).items()
                         if not pins.get('nodelist') or name in pins['nodelist']}
        if pins.get('partition'):
            for item in view['nodes'].values():
                item['partitions'] = [part for part in item.get('partitions', []) if part in pins['partition']]
        declared = node['arguments']
        profile = tool_profile(view, node['tool'], declared)
        default = profile.get('default_cpus') or native_requirements(node['tool'], declared)['default_cpus']
        hint = proposed.get('num_processes', declared.get('num_processes', declared.get('cpus_per_task')))
        if not positive(default): return None
        cpus = max(default, hint) if positive(hint) else default
        cpus = hard['cpus']['exact'] or min(cpus, hard['cpus']['max'] or cpus)
        budget = hard['memory_mb']['exact'] or proposed.get('memory_mb') or declared.get('memory_mb')
        if budget is not None and not positive(budget): return None
        if hard['memory_mb']['max']:
            budget = min(budget, hard['memory_mb']['max']) if budget else hard['memory_mb']['max']
        minimum = profile.get('minimum_memory_mb', 64)
        if not positive(minimum) or budget is not None and budget < max(64, minimum):
            return None  # Do not inflate a real user cap or claim invalid native RAM is ready.
        options = dict(cpus=cpus, memory_mb=budget, arguments=declared,
            preferred_nodelist=declared.get('nodelist') or proposed.get('nodelist', ''),
            preferred_partition=declared.get('partition') or proposed.get('partition', ''))
        selected = profiled_resources(view, node['tool'], **options)
        if not selected and budget:
            selected = profiled_resources(view, node['tool'], for_queue=True, **options)
        if selected:
            selected['budget_final'] = True
        return selected
    except (AttributeError, KeyError, TypeError, ValueError):
        return None


def allocate_reviewed_resources(session, request, receipt):
    """Resolve operational budgets from fresh delegated evidence and operator policy."""
    from .recovery import result_object
    from .node_inventory import profiled_resources
    from .resource_policy import tool_profile, allocation_hardware
    from .recovery import is_submission
    if receipt.get('native_preview') is True:
        nodes = request.get('nodes') or []
        allocations = receipt.get('resource_allocations') or {}
        valid = (receipt.get('reviewer') == 'native_workload_policy'
            and request.get('review_phase') == 'planning' and bool(nodes)
            and set(allocations) == {node.get('step_id') for node in nodes}
            and all(node.get('tool') == 'run_cdft'
                and (node.get('workload_facts') or {}).get('status') != 'ready'
                and allocations[node['step_id']].get('budget_final') is False
                and allocations[node['step_id']].get('needs_workload_review') is True
                and not any(key in allocations[node['step_id']] for key in ('cpus','memory_mb','nodelist','partition'))
                for node in nodes))
        if valid:
            return receipt
        return {**receipt, 'status': 'waiting_resources', 'retryable': True,
                'resource_allocations': {}, 'reason': 'A planning preview cannot authorize scientific dispatch; prepared native evidence is required.'}
    if receipt.get('status') == 'review_retry':
        return receipt
    if not receipt.get('resource_review_id'):
        return {**receipt, 'status': 'review_retry', 'retryable': True,
                'reason': 'resource review has no evidence-backed receipt; resource agent must complete its assessment'}
    snapshots = [result_object(session._load_evidence_call(c).get('result')).get('node_inventory')
                 for c in session.memory.tool_call_log if c['tool'] == 'resource_health'
                 and c.get('call_id') in receipt.get('evidence_call_ids', [])]
    snapshot = next((s for s in reversed(snapshots) if s and s.get('verified') and not s.get('stale')), None)
    allocations = {}
    for node in request.get('nodes', []):
        if not is_submission(node['tool'], node['arguments']):
            continue
        proposed = receipt.get('suggested_resources', {})
        declared = node['arguments']
        if request.get('kind') == 'before_submission' and node['tool'] == 'run_gcmc_batch':
            from .gcmc_workload import allocate_gcmc_workload
            sizing = allocate_gcmc_workload(request, node, snapshot, proposed,
                owned_root=session._evidence_root.parent)
            if sizing['status'] == 'planning_only':
                allocations[node['step_id']] = {'budget_final': False, 'needs_workload_review': True,
                    'workload_budget_status': 'needs_prepared_inputs', 'resource_review_id': receipt['resource_review_id']}
                continue
            if sizing['status'] != 'ready' or not sizing['candidates']:
                return {**receipt, 'status': 'waiting_resources', 'retryable': True,
                    'workload_budget': sizing, 'resource_allocations': {},
                    'reason': 'Native GCMC case inventory, compatible capacity and a labelled per-worker RAM estimate are required; no guessed serial fallback.'}
            allocations[node['step_id']] = {**sizing['candidates'][0], 'resource_review_id': receipt['resource_review_id'],
                'evidence_call_ids': receipt.get('evidence_call_ids', [])}
            continue
        workload_review = (request.get('kind') == 'before_submission' and node['tool'] == 'run_cdft')
        provisional = False
        if workload_review:
            sizing = _workload_allocation(session, request, node, snapshot, proposed)
            if sizing['status'] == 'ready':
                allocations[node['step_id']] = {**sizing['candidates'][0],
                    'resource_review_id': receipt['resource_review_id'],
                    'evidence_call_ids': receipt.get('evidence_call_ids', [])}
                continue
            provisional = sizing['status'] == 'planning_only'
            if not provisional:
                return {**receipt, 'status': 'waiting_resources', 'retryable': True,
                        'workload_budget': sizing, 'resource_allocations': {},
                        'reason': 'Resource agent must verify prepared workload and source-bound per-worker memory before submission; no serial fallback or scientific dispatch.'}
        budget = declared.get('memory_mb') or proposed.get('memory_mb')
        requested_cpus = declared.get('num_processes') or declared.get('cpus_per_task') or proposed.get('num_processes')
        selected = None
        generic_review = request.get('kind') == 'before_submission' and not workload_review
        if generic_review:
            selected = _generic_resource_allocation(request, node, snapshot, proposed)
        elif snapshot:
            # A model hint may rank compatible nodes, never force a busy target
            # while immediate capacity satisfies all user/operator constraints.
            options = dict(cpus=requested_cpus, memory_mb=budget, arguments=declared,
                           nodelist=declared.get('nodelist', ''), partition=declared.get('partition', ''),
                           preferred_nodelist=proposed.get('nodelist', ''),
                           preferred_partition=proposed.get('partition', ''))
            selected = profiled_resources(snapshot, node['tool'], **options)
            if not selected and budget:
                selected = profiled_resources(snapshot, node['tool'], for_queue=True, **options)
        if not selected:
            return {**receipt, 'status': 'waiting_resources', 'retryable': True,
                    'reason': '资源智能体继续核验兼容容量与申请预算；普通资源占用应提交Slurm排队，无需用户补充内存。'}
        selected = allocation_hardware(snapshot, selected, tool_profile(snapshot, node['tool'], declared))
        if provisional:
            selected.pop('cpus', None)
            selected.update(budget_final=False, needs_workload_review=True,
                            workload_budget_status='needs_prepared_inputs')
        allocations[node['step_id']] = {**selected, 'resource_review_id': receipt['resource_review_id'],
            'estimate_basis': receipt.get('reason'), 'evidence_call_ids': receipt.get('evidence_call_ids', [])}
    return {**receipt, 'status': 'ready', 'resource_allocations': allocations,
            'reason': '资源智能体已核验节点；缺省内存按配置的可用内存策略分配，已有显式申请保持不变。'}


def assess_job_resources(job_id):
    from .job_watch import get_watch
    from .watch_context import get_context
    from .config import get_config
    from .node_inventory import node_inventory
    from .slurm import normalize_job_id, check_job_status
    job_id = normalize_job_id(job_id)
    job = get_watch().get(job_id) or {}
    owner = get_context()
    if owner.get('username') and (job.get('username'),job.get('conv_id')) != (owner['username'],owner.get('conv_id')):
        return {'error':'resource assessment requires a job owned by this user/session', 'read_only':True}
    if not re.fullmatch(r'\d+', job_id): return {'error':'scheduler job ID must be numeric', 'read_only':True}
    result = subprocess.run(['scontrol','show','job',job_id,'-o'],capture_output=True,text=True,timeout=8)
    if result.returncode: return {'error':'scheduler job constraints unavailable', 'read_only':True}
    fields = dict(re.findall(r'(\w+)=(\S+)',result.stdout))
    constraints = {key:fields.get(key) for key in ('JobState','Reason','ReqNodeList','NumCPUs','TimeLimit','ReqTRES','StartTime','Partition')}
    from datetime import datetime
    now=datetime.now().astimezone()
    try:
        start=datetime.fromisoformat(fields.get('StartTime','')).replace(tzinfo=now.tzinfo)
        constraints['estimated_wait_seconds']=max(0,(start-now).total_seconds())
    except ValueError: constraints['estimated_wait_seconds']=None
    constraints['observed_server_time']=now.isoformat()
    from pathlib import Path
    script=Path(fields.get('Command',''))
    root=Path(job.get('work_dir') or get_config().project_root).resolve()
    if script.is_file() and script.resolve().is_relative_to(root) and script.stat().st_size<=1024*1024:
        constraints['script_has_memory_directive']=bool(re.search(r'^#SBATCH\s+(?:--mem(?:-per-cpu)?(?:=|\s))',script.read_text(errors='replace'),re.M))
    inventory = node_inventory(get_config().project_root, refresh=True, persist=False)
    # Partition limits are separate scheduler facts, not inferred from node names.
    partition_facts = inventory.get('partition_facts')
    if partition_facts is None:
        from .node_inventory import read_partition_limits
        partition_facts = read_partition_limits()
    partition_facts = partition_facts or {}
    partition_limits = partition_facts.get('limits', {}) if partition_facts.get('verified') is True else {}
    from .job_control import retarget_compatibility
    compatibility = retarget_compatibility({'effective_constraints': constraints,
        'node_inventory': inventory, 'partition_limits': partition_limits}, job)
    return {'ok':True,'read_only':True,'job_id':job_id,'observed_at':time.time(),
        'scheduler_status':check_job_status(job_id,work_dir=job.get('work_dir','')),
        'effective_constraints':constraints,'node_inventory':inventory,
        'partition_limits':partition_limits,'compatible_targets':compatibility['compatible_targets'],
        'target_rejections':compatibility['target_rejections'],
        'compatibility_reason_codes':compatibility['reason_codes'],
        'execution_started':fields.get('JobState')=='RUNNING',
        'note':'Do not shrink RAM without workload evidence/budget. Snapshot is not a reservation. Pending is not failure/completion. No cancel, update or resubmit was performed.'}


def native_workload_review(session, request):
    """Native policy evaluation using actual observations, never a model decision.

    Trusted request facts come from the authenticated server. All scientific
    resource guards remain in the shared allocator; unknowns retain the model
    estimation path. A preview contains no guessed capacity or dispatch permit.
    """
    import copy
    nodes = request.get('nodes') or []
    if (request.get('kind') != 'before_submission' or not nodes
            or any(node.get('tool') != 'run_cdft'
                or node.get('arguments', {}).get('action') not in {'pipeline', 'submit'} for node in nodes)):
        return None
    preview = (request.get('review_phase') == 'planning'
        and all((node.get('workload_facts') or {}).get('status') != 'ready' for node in nodes))
    if not preview and any(node['arguments'].get('action') != 'submit'
            or (node.get('workload_facts') or {}).get('status') != 'ready'
            or _operator_workload_profile(session, node) is None for node in nodes):
        return None
    receipt = {'status': 'ready', 'resource_review_id': uuid.uuid4().hex,
        'reviewer': 'native_workload_policy', 'reviewed_at': time.time(),
        'read_only': True, 'readonly': True, 'not_resubmitted': True,
        'model_decision_delivered': False, 'evidence_call_ids': [], 'evidence_calls': []}
    if preview:
        receipt.update(native_preview=True, reason='Planning preview only; prepared inputs and native final sizing are required at dispatch.',
            resource_allocations={node['step_id']: {'budget_final': False,
                'needs_workload_review': True, 'workload_budget_status': 'needs_prepared_inputs',
                'resource_review_id': receipt['resource_review_id'], 'evidence_call_ids': []} for node in nodes})
    else:
        try:
            raw = session.registry.execute_dict('resource_health', {})
        except Exception as error:
            raw = {'read_only': True, 'error': 'native resource observation failed',
                   'exception_type': type(error).__name__}
        if not isinstance(raw, dict):
            raw = {'read_only': True, 'error': 'native resource observation returned a non-object'}
        failed = bool(raw.get('error')) or raw.get('read_only') is not True
        session.memory.record_tool_call('monitor', 'resource_health', {},
            json.dumps(raw, ensure_ascii=False), failed=failed)
        call = session.memory.tool_call_log[-1]
        session._checkpoint('resource_review_native_observation')
        if failed:
            return None
        receipt['evidence_call_ids'] = [call['call_id']]
        receipt['evidence_calls'] = [copy.deepcopy(call)]
        receipt['reason'] = 'Native observation and binary/input-bound operator workload estimate; allocation is not a scheduler reservation or a scientific acceptance.'
        receipt = allocate_reviewed_resources(session, request, receipt)
        if (receipt.get('status') != 'ready'
                or set(receipt.get('resource_allocations', {})) != {node['step_id'] for node in nodes}
                or any(item.get('budget_final') is not True for item in receipt['resource_allocations'].values())):
            return None
    session.context['native_workload_review'] = {'resource_review_id': receipt['resource_review_id'],
        'reviewer': receipt['reviewer'], 'native_preview': bool(preview),
        'step_ids': [node['step_id'] for node in nodes], 'evidence_call_ids': receipt['evidence_call_ids']}
    session._checkpoint('resource_review_native_policy')
    return receipt


def run_resource_review(session, request):
    from .defns import MONITOR
    import copy
    from .resource_policy import native_requirements
    request = copy.deepcopy(request)
    for node in request.get('nodes', []):
        node['hardware_requirements'] = native_requirements(node.get('tool'), node.get('arguments'))
        if request.get('kind') == 'before_submission' and node.get('tool') == 'run_gcmc_batch':
            from .gcmc_workload import collect_gcmc_workload
            node['workload_facts'] = collect_gcmc_workload(node['arguments'], owned_root=session._evidence_root.parent)
        if request.get('kind') == 'before_submission' and node.get('tool') == 'run_cdft':
            bound = _operator_workload_profile(session, node)
            node['workload_budget_evidence'] = ({
                'estimate_kind': bound['estimate_kind'], 'per_worker_mb': bound['memory_evidence']['per_worker_mb'],
                'overhead_mb': bound['overhead_mb'], 'basis': bound['memory_evidence']['basis'],
                'profile_sha256': bound['profile_sha256']} if bound else {
                'status': 'needs_memory_evidence', 'note': 'No verified binary/input-bound estimate; total RAM alone is not per-worker memory evidence.'})
    agent = copy.copy(MONITOR)
    names = {'resource_health','assess_job_resources','resource_review_decision','read_file','grep_search'}
    agent.functions = [f for f in agent.functions if f.__name__ in names]
    session.current_agent = agent
    native_receipt = native_workload_review(session, request)
    if native_receipt is not None:
        return native_receipt
    session.messages.append({'role':'user','content':'[RESOURCE REVIEW]\n'+json.dumps(request,ensure_ascii=False)+
        '\n自主查正式资源工具并调用resource_review_decision。填写实际工具返回的call_id作为evidence_call_ids。未给内存时根据工作量、工具特点、已有日志估算，输出suggested_resources.memory_mb并用reason说明依据和余量；GCMC独立case按实际CIF×gas估算每worker内存，填写per_worker_memory_mb和per_worker_memory_basis，明确是保守估算而非实测峰值，CPU1提示不限制可行并发；已有显式预算保持不变。核对CPU/内存/兼容性/分区，暂时无空闲但总容量兼容时提交Slurm排队，不要求用户提供RAM，不替换科研条件。等待时长只引用工具证据。'})
    evidence = []
    receipt = {'status':'review_retry','retryable':True,'reason':'resource reviewer has not delivered a valid receipt; repair the tool/schema evidence exchange, not the user budget'}
    evidence_start=len(session.memory.tool_call_log)
    try:
        for round_id in range(5):
            session._resource_review_tools = names-{'resource_review_decision'} if not evidence and round_id<2 else names
            session._resource_review_final = round_id>=2
            from .codex_provider import CodexToolArgumentError
            try:
                response = session._call_api(agent)
            except CodexToolArgumentError as error:
                session._record_tool_protocol_error(agent, error)
                continue
            session.messages.append({'role':'assistant','content':response.content})
            results=[]
            for call in [b for b in response.content if getattr(b,'type','')=='tool_use']:
                if call.name not in names:
                    raw={'blocked':True,'executed':False,'reason':'resource agent is read-only'}
                elif call.name=='resource_review_decision':
                    issues=session.registry.validate_params(call.name,call.input)
                    cited=call.input.get('evidence_call_ids',[])
                    if issues or not evidence or not cited or not set(cited).issubset(set(evidence)):
                        raw={'error':'decision requires valid schema and actual successful resource call IDs',
                             'evidence_call_ids':evidence,'input_schema':session.registry.get(call.name).input_schema}
                    else:
                        receipt={**call.input,'resource_review_id':uuid.uuid4().hex,'reviewed_at':time.time(),
                                 'evidence_call_ids':cited,'readonly':True,'not_resubmitted':True}
                        raw={'ok':True,'delivered_to':'main_chat'}
                else:
                    raw=session.registry.execute_dict(call.name,call.input)
                record=session.memory.record_tool_call('monitor',call.name,call.input,json.dumps(raw,ensure_ascii=False),failed=bool(raw.get('error')))
                if call.name in names-{'resource_review_decision'} and not raw.get('error') and raw.get('read_only'):
                    evidence.append(session.memory.tool_call_log[-1]['call_id'])
                results.append({'type':'tool_result','tool_use_id':call.id,'content':json.dumps({**raw,'evidence_ref':{'call_id':session.memory.tool_call_log[-1]['call_id']}},ensure_ascii=False)})
                if raw.get('delivered_to'): break
            session.messages.append({'role':'user','content':results or '请实际调用来源工具并交付结构化资源结论，不要只输出文字。'})
            session._checkpoint('resource_review')
            if receipt.get('resource_review_id'):break
    finally:
        session._resource_review_tools=None;session._resource_review_final=False
    legacy_memory_proposal = False
    receipt['evidence_calls']=[c for c in session.memory.tool_call_log[evidence_start:] if c.get('call_id') in evidence]
    if request.get('kind') in {'pending_warning', 'job_pending_warning'}:
        from .recovery import result_object
        assessments=[result_object(session._load_evidence_call(c).get('result'))
                     for c in receipt['evidence_calls'] if c['tool']=='assess_job_resources']
        for assessment in assessments:
            constraints=assessment.get('effective_constraints',{})
            if constraints.get('JobState')!='PENDING' or constraints.get('script_has_memory_directive') is not False:
                continue
            from .job_watch import get_watch
            from .node_inventory import candidates
            job=get_watch().get(str(request.get('job_id',''))) or {}
            inventory=assessment.get('node_inventory') or {}
            profile=inventory.get('policy',{}).get('tool_profiles',{}).get(job.get('tool'),{})
            minimum=int(profile.get('minimum_memory_mb') or 64)
            base_fraction=float(profile.get('memory_fraction_of_available') or 0)
            cpus=int(constraints.get('NumCPUs') or profile.get('default_cpus') or 1)
            choices=(candidates(inventory,cpus=cpus,memory_mb=minimum,min_glibc=profile.get('min_glibc'))
                     if 0 < base_fraction <= 1 else [])
            targets=[]
            for partition in profile.get('partition_preference',[]):
                targets.extend((partition,node) for node in choices if partition in node.get('partitions',[]))
            if targets:
                legacy_memory_proposal = True
                partition,node=targets[0]
                available=int(node.get('memory',{}).get('available_for_scheduling_mib') or 0)
                fraction=float(profile.get('memory_fraction_by_partition',{}).get(partition,base_fraction) or 0)
                budget=max(minimum,int(available*fraction))
                receipt.update(status='propose_resource_patch',model_status=receipt.get('status'),
                    reason=('已查证脚本缺少显式内存，调度器因而按整节点内存申请；工具资源画像和当前节点事实支持'
                            f'将同一作业就地调整为 {budget} MiB、{partition}/{node["name"]}。不取消、不重提、不改变科研参数。'),
                    suggested_resources={'memory_mb':budget,'num_processes':cpus,
                                         'nodelist':node['name'],'partition':partition},
                    effective_constraints=constraints)
            else:
                receipt.update(status='waiting_resources',model_status=receipt.get('status'),
                    reason=('已查证脚本无显式内存预算，但没有工具资源画像或当前没有满足该画像的节点。'
                            '保留原作业，不能猜测更小内存或盲目迁移。'),
                    effective_constraints=constraints)
    if request.get('kind') in {'pending_warning', 'job_pending_warning'}:
        # A model node preference is optional. The proposal itself must follow
        # this turn's successful, fresh native assessment of the same owned job.
        from .job_control import retarget_compatibility
        from .job_watch import get_watch
        from .watch_context import get_context
        from .resource_policy import tool_profile
        from .recovery import result_object
        owner = get_context()
        job_id = str(request.get('job_id', ''))
        job = get_watch().get(job_id) or {}
        owned = (bool(owner.get('username') and owner.get('conv_id')) and
                 (job.get('username'), job.get('conv_id')) == (owner['username'], owner['conv_id']))
        proposed = False
        for call in reversed(receipt['evidence_calls']):
            if call.get('tool') != 'assess_job_resources' or call.get('failed'):
                continue
            saved = session._load_evidence_call(call)
            assessment = result_object(saved.get('result'))
            constraints = assessment.get('effective_constraints') or {}
            try:
                age = time.time() - float(assessment.get('observed_at', 0))
                fresh = 0 <= age <= 120
            except (TypeError, ValueError):
                fresh = False
            if (not owned or not fresh or assessment.get('ok') is not True
                    or assessment.get('read_only') is not True or assessment.get('error')
                    or str(saved.get('params', {}).get('job_id', '')) != job_id
                    or str(assessment.get('job_id', '')) != job_id
                    or constraints.get('JobState') != 'PENDING'
                    or constraints.get('script_has_memory_directive') is not True):
                continue  # Preserve the separate missing-memory repair path above.
            verified = retarget_compatibility(assessment, job)['compatible_targets']
            published = assessment.get('compatible_targets') or []
            keys = ('nodelist', 'partition', 'memory_mb', 'num_processes', 'walltime')
            options = [target for target in verified if any(
                all(candidate.get(key) == target.get(key) for key in keys)
                for candidate in published if isinstance(candidate, dict))]
            if not options:
                continue
            profile = tool_profile(assessment['node_inventory'], job.get('tool'), job.get('arguments'))
            def names(value):
                return value.split(',') if isinstance(value, str) else list(value or [])
            partitions = names(profile.get('partition_preference'))
            nodes = names(assessment['node_inventory'].get('policy', {}).get('preferred_nodes'))
            hint = receipt.get('suggested_resources') or {}
            def rank(target):
                valid_hint = (hint.get('nodelist') == target['nodelist']
                              and hint.get('partition') == target['partition']
                              and all(key not in hint or hint[key] == target[key]
                                      for key in ('memory_mb', 'num_processes', 'walltime')))
                return (not valid_hint,
                        partitions.index(target['partition']) if target['partition'] in partitions else len(partitions),
                        nodes.index(target['nodelist']) if target['nodelist'] in nodes else len(nodes),
                        target['partition'], target['nodelist'])
            target = min(options, key=rank)
            receipt.update(status='propose_resource_patch', model_status=receipt.get('status'),
                requires_user=False, retryable=True,
                resource_review_id=receipt.get('resource_review_id') or uuid.uuid4().hex,
                reviewed_at=time.time(), readonly=True, not_resubmitted=True,
                suggested_resources={key: target[key] for key in keys},
                effective_constraints=constraints, compatible_targets=options,
                evidence_call_ids=list(dict.fromkeys(receipt.get('evidence_call_ids', []) + [call['call_id']])),
                reason='Fresh native assessment supports this same-job target with the original CPU, RAM and walltime. This is a proposal, not a reservation or permission to cancel/resubmit; retarget must recheck the pending job.')
            proposed = True
            break
        if not proposed and not legacy_memory_proposal and receipt.get('status') not in {'waiting_resources', 'review_retry'}:
            receipt.update(status='waiting_resources', model_status=receipt.get('status'),
                retryable=True, requires_user=False,
                reason='Resource agent must obtain fresh compatible same-job evidence; preserve the original job and budget, without asking the user to select a node.')
    return receipt
