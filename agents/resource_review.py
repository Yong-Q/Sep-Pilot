"""Read-only scheduler assessment and evidence-backed resource-agent review."""
import json
import re
import subprocess
import time
import uuid


def allocate_reviewed_resources(session, request, receipt):
    """Resolve operational budgets from fresh delegated evidence and operator policy."""
    from .recovery import result_object
    from .node_inventory import profiled_resources, candidates
    from .recovery import is_submission
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
        args = {**{k: v for k, v in proposed.items() if k in {'memory_mb','nodelist','partition','num_processes'}}, **node['arguments']}
        budget = args.get('memory_mb') or proposed.get('memory_mb')
        requested_cpus = args.get('num_processes') or args.get('cpus_per_task') or proposed.get('num_processes')
        selected = profiled_resources(snapshot or {}, node['tool'], cpus=requested_cpus,
            memory_mb=budget, nodelist=args.get('nodelist', ''), partition=args.get('partition', '')) if snapshot else None
        if not selected and snapshot and budget:
            profile = snapshot.get('policy', {}).get('tool_profiles', {}).get(node['tool'], {})
            cpus = requested_cpus or profile.get('default_cpus', 1)
            eligible = candidates(snapshot, cpus=cpus, memory_mb=budget,
                                 min_glibc=profile.get('min_glibc'), for_queue=True)
            if args.get('nodelist'):
                eligible = [n for n in eligible if n['name'] in args['nodelist'].split(',')]
            partitions = [args['partition']] if args.get('partition') else profile.get('partition_preference') or sorted({p for n in eligible for p in n.get('partitions', [])})
            targets = [(part, n) for part in partitions for n in eligible if part in n['partitions']]
            if targets:
                part, target = targets[0]
                selected = {'nodelist': target['name'], 'partition': part, 'memory_mb': budget,
                            'cpus': cpus, 'scheduling_mode': 'queue', 'observed_at': snapshot.get('observed_at')}
        if not selected:
            return {**receipt, 'status': 'waiting_resources', 'retryable': True,
                    'reason': '资源智能体继续核验兼容容量与申请预算；普通资源占用应提交Slurm排队，无需用户补充内存。'}
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
    return {'ok':True,'read_only':True,'job_id':job_id,'observed_at':time.time(),
        'scheduler_status':check_job_status(job_id,work_dir=job.get('work_dir','')),
        'effective_constraints':constraints,'node_inventory':inventory,
        'execution_started':fields.get('JobState')=='RUNNING',
        'note':'Do not shrink RAM without workload evidence/budget. Snapshot is not a reservation. Pending is not failure/completion. No cancel, update or resubmit was performed.'}


def run_resource_review(session, request):
    from .defns import MONITOR
    import copy
    agent = copy.copy(MONITOR)
    names = {'resource_health','assess_job_resources','resource_review_decision','read_file','grep_search'}
    agent.functions = [f for f in agent.functions if f.__name__ in names]
    session.current_agent = agent
    session.messages.append({'role':'user','content':'[RESOURCE REVIEW]\n'+json.dumps(request,ensure_ascii=False)+
        '\n自主查正式资源工具并调用resource_review_decision。填写实际工具返回的call_id作为evidence_call_ids。未给内存时根据工作量、工具特点、已有日志估算，输出suggested_resources.memory_mb并用reason说明依据和余量；已有显式预算保持不变。核对CPU/内存/兼容性/分区，暂时无空闲但总容量兼容时提交Slurm排队，不要求用户提供RAM，不替换科研条件。等待时长只引用工具证据。'})
    evidence = []
    receipt = {'status':'review_retry','retryable':True,'reason':'resource reviewer has not delivered a valid receipt; repair the tool/schema evidence exchange, not the user budget'}
    evidence_start=len(session.memory.tool_call_log)
    try:
        for round_id in range(5):
            session._resource_review_tools = names-{'resource_review_decision'} if not evidence and round_id<2 else names
            session._resource_review_final = round_id>=2
            response = session._call_api(agent)
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
    return receipt
