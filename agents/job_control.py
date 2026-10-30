"""Scoped scheduling control. Retarget the existing pending job, never resubmit."""
import json
from pathlib import Path
import re
import subprocess
import time
import uuid


def pause_owned_workflow(project_root, username, conv_id):
    """Cancellation intent freezes future dispatch, not the live job lease."""
    from .parallel_workflow import WorkflowStore
    path = Path(project_root) / 'data/state/parallel_workflows.json'
    if not path.exists(): return
    store = WorkflowStore(path)
    identity = store.identity(username, conv_id)
    if store.snapshot(identity):
        store.user_pause(identity, True)


def _memory_mib(constraints):
    match = re.search(r'(?:^|,)mem=(\d+)([KMGT]?)(?:,|$)', constraints.get('ReqTRES') or '')
    if not match:
        raise ValueError('effective requested RAM is unknown')
    value = int(match[1]) * {'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2, '': 1}[match[2]]
    if value <= 0 or int(value) != value:
        raise ValueError('effective RAM is not a positive integer MiB')
    return int(value)


def _wall_seconds(value):
    if value == 'UNLIMITED':
        return float('inf')
    match = re.fullmatch(r'(?:(\d+)-)?(\d+):(\d{2}):(\d{2})', str(value))
    if not match or int(match[3]) >= 60 or int(match[4]) >= 60:
        raise ValueError('unknown scheduler walltime')
    return int(match[1] or 0)*86400 + int(match[2])*3600 + int(match[3])*60 + int(match[4])


def retarget_compatibility(assessment, job, nodelist='', partition='', memory_mb=None):
    """Read-only fixed-budget candidates; never relax native/operator hard pins.

    Current scheduler placement is the value being changed, not an operator pin.
    Both the requested target and alternatives use the same eligibility predicate.
    """
    from .node_inventory import profiled_resources
    from .resource_policy import tool_profile
    import copy
    inventory = copy.deepcopy(assessment.get('node_inventory') or {})
    constraints = assessment.get('effective_constraints') or {}
    targets, rejected = [], {}
    base = {'retryable': True, 'requires_user': False, 'compatible_targets': targets,
            'target_rejections': rejected, 'target_compatible': False,
            'effective_constraints': constraints}
    try:
        cpus = int(constraints['NumCPUs'])
        original_memory = _memory_mib(constraints)
        memory = original_memory if memory_mb is None else memory_mb
        walltime = constraints['TimeLimit']; seconds = _wall_seconds(walltime)
        if cpus < 1 or isinstance(memory, bool) or not isinstance(memory, int) or memory < 1 or seconds <= 0:
            raise ValueError('invalid effective budget')
        tool = job.get('tool'); arguments = job.get('arguments') or {}
        profile = tool_profile(inventory, tool, arguments)
        # Existing requested GPUs cannot be silently removed on a CPU-native tool.
        gpu_counts = re.findall(r'(?:^|,)gres/gpu(?:\:[^=,]+)?=(\d+)(?=,|$)', constraints.get('ReqTRES') or '')
        existing_gpus = max(map(int, gpu_counts), default=0)
        if existing_gpus < profile['gpus']:
            return {**base, 'reason_codes': ['existing_gpu_request_below_native_or_operator_requirement']}
        profile['gpus'] = max(profile['gpus'], existing_gpus)
        inventory.setdefault('policy', {}).setdefault('tool_profiles', {})[tool] = profile
    except (KeyError, TypeError, ValueError):
        return {**base, 'reason_codes': ['effective_budget_or_profile_unverified']}
    def names(value):
        return value.split(',') if isinstance(value, str) else list(value or [])
    def reasons(name, part):
        issues = []
        policy = inventory.get('policy') or {}; node = inventory.get('nodes', {}).get(name)
        if not inventory.get('verified') or inventory.get('stale'): issues.append('inventory_unverified_or_stale')
        if not node: return issues + ['node_unknown']
        if not node.get('states') or any(s not in policy.get('allowed_states', []) for s in node['states']): issues.append('state_not_allowed')
        if node.get('cpus', {}).get('idle', 0) < cpus: issues.append('insufficient_idle_cpus')
        ram = node.get('memory', {})
        if not ram.get('verified') or ram.get('available_for_scheduling_mib', 0) < memory: issues.append('ram_unverified_or_insufficient')
        if memory != original_memory:
            try:
                minimum = int(profile.get('minimum_memory_mb') or 64)
                fraction = float(profile.get('memory_fraction_by_partition', {}).get(
                    part, profile.get('memory_fraction_of_available')) or 0)
                available = int(ram.get('available_for_scheduling_mib') or 0)
                if not 0 < fraction <= 1 or available < minimum or memory != max(minimum, int(available*fraction)):
                    issues.append('memory_change_not_operator_profiled')
            except (TypeError, ValueError): issues.append('memory_change_profile_unverified')
        if part not in node.get('partitions', []): issues.append('node_partition_mismatch')
        if name in policy.get('excluded_nodes', []) or name in profile.get('excluded_nodes', []): issues.append('policy_node_excluded')
        if any(pin and name not in names(pin) for pin in (profile.get('nodelist'), profile.get('allowed_nodes'))): issues.append('policy_node_constraint')
        if any(pin and part not in names(pin) for pin in (profile.get('partition'), profile.get('allowed_partitions'))): issues.append('policy_partition_constraint')
        if profile.get('gpus') and (not node.get('gpus', {}).get('verified') or node['gpus'].get('available', 0) < profile['gpus']): issues.append('gpu_unverified_or_insufficient')
        try:
            if profile.get('min_glibc') and (not node.get('glibc') or tuple(map(int,node['glibc'].split('.'))) < tuple(map(int,profile['min_glibc'].split('.')))): issues.append('glibc_unverified_or_incompatible')
        except (TypeError,ValueError): issues.append('glibc_unverified_or_incompatible')
        limits = assessment.get('partition_limits', {}).get(part, {})
        try:
            if not limits.get('verified') or limits.get('state') != 'UP': issues.append('partition_unverified_or_not_up')
            if _wall_seconds(limits.get('max_time')) < seconds: issues.append('partition_walltime_too_short')
        except ValueError: issues.append('partition_walltime_unverified')
        if not issues:
            try:
                if not profiled_resources(inventory, tool, cpus=cpus, memory_mb=memory, nodelist=name, partition=part, arguments=arguments): issues.append('native_or_operator_profile_rejected')
            except (KeyError,TypeError,ValueError): issues.append('native_or_operator_profile_unverified')
        return issues
    if nodelist:
        base['reason_codes'] = reasons(nodelist, partition)
        base['target_compatible'] = not base['reason_codes']
    else:
        base['reason_codes'] = []
    for name, node in sorted(inventory.get('nodes', {}).items()):
        for part in sorted(node.get('partitions', [])):
            issues = reasons(name, part)
            if issues:
                rejected[f'{part}/{name}'] = issues
            elif len(targets) < 16:
                targets.append({'nodelist': name, 'partition': part, 'memory_mb': memory,
                                'cpus': cpus, 'num_processes': cpus, 'walltime': walltime,
                                'observed_at': inventory.get('observed_at')})
    return base


def _verified_rejection(journal, current, job_id):
    """Prove a completed EAGAIN rejection left observed controls unchanged.

    An exit code alone, transport failure, partial update or running job is not
    a rejection proof. Volatile scheduling estimates are not control fields.
    """
    import hashlib
    code = journal.get('exit_code')
    if (str(journal.get('job_id')) != str(job_id) or not isinstance(code, int)
            or isinstance(code, bool) or not 0 < code < 256
            or not re.fullmatch(r'Resource temporarily unavailable for job ' + re.escape(str(job_id)) + r'\s*',
                                journal.get('scheduler_error') or '')):
        return None
    before, after = journal.get('old_constraints') or {}, journal.get('after') or {}
    required = {'JobState', 'ReqNodeList', 'Partition', 'NumCPUs', 'TimeLimit', 'ReqTRES'}
    volatile = {'Reason', 'StartTime', 'estimated_wait_seconds', 'observed_server_time'}
    fields = (set(before) | set(after) | set(current)) - volatile
    if (not required.issubset(fields) or any(key not in obj for obj in (before, after, current) for key in fields)
            or any(obj.get('JobState') != 'PENDING' for obj in (before, after, current))
            or any(before[key] != after[key] or before[key] != current[key] for key in fields)):
        return None
    try:
        if int(before['NumCPUs']) < 1 or _memory_mib(before) < 1 or _wall_seconds(before['TimeLimit']) <= 0:
            return None
        if not before['ReqNodeList'] or not before['Partition']:
            return None
    except (KeyError, TypeError, ValueError):
        return None
    proof = {'kind': 'verified_scheduler_rejection', 'job_id': str(job_id),
             'unchanged_fields': {key: before[key] for key in sorted(fields)},
             'exit_code': code, 'scheduler_error': journal['scheduler_error']}
    proof['proof_sha256'] = hashlib.sha256(json.dumps(proof, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    return proof


def retarget_queued_job(job_id, nodelist, partition, username, conv_id, project_root, memory_mb=None):
    from .job_watch import get_watch
    from .node_inventory import node_inventory
    from .state_io import ConversationLock, write_checkpoint
    from .resource_review import assess_job_resources
    if not re.fullmatch(r'\d+',str(job_id)) or not re.fullmatch(r'[A-Za-z0-9_.-]+',nodelist) or not re.fullmatch(r'[A-Za-z0-9_.-]+',partition):
        raise ValueError('retarget requires one literal node/partition and numeric job ID')
    watch=get_watch();job=watch.get(str(job_id)) or {}
    if (job.get('username'),job.get('conv_id'))!=(username,conv_id):raise PermissionError('job belongs to another user/session')
    root=Path(project_root)/'runs'/username/conv_id
    lock=ConversationLock(root/'job_control'/f'{job_id}.lock')
    if not lock.acquire(blocking=False):return {'blocked':True,'executed':False,'reason':'job scheduling control already in progress'}
    control_basis = None
    try:
        for control_attempt in range(3):
            current_job = watch.get(str(job_id)) or {}
            if (current_job.get('username'), current_job.get('conv_id')) != (username, conv_id):
                raise PermissionError('job ownership changed during scheduling control')
            if (watch.get(str(job_id)) or {}).get('cancel_request', {}).get('status') in {'intent', 'sent', 'uncertain'}:
                return {'blocked': True, 'executed': False, 'reason': 'cancellation outcome is unresolved; reconcile the original job before retargeting'}
            assessment=assess_job_resources(str(job_id));constraints=assessment.get('effective_constraints',{})
            stable = {key: value for key, value in constraints.items()
                      if key not in {'Reason', 'StartTime', 'estimated_wait_seconds', 'observed_server_time'}}
            if control_basis is None:
                control_basis = stable
            elif stable != control_basis:
                return {'blocked':True,'executed':False,'status':'uncertain','job_id':str(job_id),
                        'requires_user':True,'retryable':False,'not_cancelled':True,'not_resubmitted':True,
                        'reason':'Observed job controls changed between attempts; stop and reconcile before further updates, without adopting a new budget.'}
            rejected_history = []
            for index,path in enumerate((root/'job_control').glob(f'{job_id}_*.json')):
                if index>=256:raise ValueError('bounded job-control audit exceeded; reconcile history before updating')
                prior = json.loads(path.read_text())
                if prior.get('status') == 'uncertain':
                    proof = _verified_rejection(prior, constraints, job_id)
                    if proof:
                        prior.update(status='verified_rejected', reconciled_from_status='uncertain',
                                     rejection_proof=proof, reconciled_at=time.time())
                        write_checkpoint(path, prior)
                if prior.get('status') in {'intent', 'uncertain'}:
                    return {'blocked':True,'executed':False,'requires_user':True,'reason':'previous scheduling update is unresolved; inspect original journal and same job before any new update'}
                if prior.get('status') == 'verified_rejected' and _verified_rejection(prior, constraints, job_id):
                    rejected_history.append((path, prior))
            if constraints.get('JobState')!='PENDING':
                return {'blocked':True,'executed':False,'requires_user':True,'reason':'Only a verified PENDING job may be retargeted; running/unknown/terminal jobs are not touched'}
            old_memory = _memory_mib(constraints)
            inventory=assessment.get('node_inventory') or {}
            profile=inventory.get('policy',{}).get('tool_profiles',{}).get(job.get('tool'),{})
            memory = old_memory if memory_mb is None else memory_mb
            if not isinstance(memory, int) or isinstance(memory, bool) or memory < 64:
                raise ValueError('memory_mb must be a finite integer >=64 MiB')
            feedback = retarget_compatibility(assessment, job, nodelist, partition, memory)
            if memory != old_memory:
                node_fact=inventory.get('nodes',{}).get(nodelist,{})
                available=int(node_fact.get('memory',{}).get('available_for_scheduling_mib') or 0)
                minimum=int(profile.get('minimum_memory_mb') or 64)
                fraction=float(profile.get('memory_fraction_by_partition',{}).get(
                    partition,profile.get('memory_fraction_of_available')) or 0)
                profiled=max(minimum,int(available*fraction)) if 0 < fraction <= 1 and available >= minimum else None
                if profiled is None or memory != profiled:
                    feedback = retarget_compatibility(assessment, job, nodelist, partition)
                    return {**feedback, 'blocked':True,'executed':False,'job_id':str(job_id),
                            'target_compatible':False,'status':'infrastructure_retry',
                            'reason_codes':['memory_change_not_operator_profiled'],
                            'reason':'Preserve the original RAM budget or use the operator-owned memory profile.',
                            'not_cancelled':True,'not_resubmitted':True}
            if not feedback['target_compatible']:
                return {**feedback,'blocked':True,'executed':False,'job_id':str(job_id),
                        'status':'infrastructure_retry','not_cancelled':True,'not_resubmitted':True,
                        'reason':'Requested target is incompatible; choose a fresh compatible target for this same job, preserving CPU/RAM/walltime. No cancellation or resubmission.'}
            requested = {'nodelist':nodelist,'partition':partition,'memory_mb':int(memory)}
            previous = [(path, prior) for path, prior in rejected_history if prior.get('requested') == requested]
            if len(previous) >= 3:
                path, prior = previous[-1]
                return {'ok':False,'status':'verified_rejected','blocked':True,'executed':False,
                        'retryable':True,'requires_user':False,'job_id':str(job_id),
                        'journal_path':str(path),'attempts_exhausted':True,
                        'compatible_targets':feedback['compatible_targets'],
                        'not_cancelled':True,'not_resubmitted':True,
                        'reason':'Three verified transient rejections for the same unchanged job/target/budget. Reassess scheduler limitations; do not repeat this request or claim another target will succeed.'}
            update_id=uuid.uuid4().hex
            journal=root/'job_control'/f'{job_id}_{update_id}.json'
            intent={'update_id':update_id,'job_id':str(job_id),'status':'intent','old_constraints':constraints,
                    'requested':{'nodelist':nodelist,'partition':partition,'memory_mb':int(memory)},'created_at':time.time(),
                    'resource_profile':job.get('tool') if memory != old_memory else '',
                    'not_cancelled':True,'not_resubmitted':True}
            write_checkpoint(journal,intent)
            result=subprocess.run(['scontrol','update',f'JobId={job_id}',f'ReqNodeList={nodelist}',f'Partition={partition}',f'MinMemoryNode={int(memory)}'],capture_output=True,text=True,timeout=10)
            after=assess_job_resources(str(job_id))
            actual=after.get('effective_constraints',{})
            new_memory_match=re.search(r'(?:^|,)mem=(\d+)([KMGT]?)',actual.get('ReqTRES',''))
            actual_memory=(int(new_memory_match.group(1))*{'K':1/1024,'M':1,'G':1024,'T':1024**2,'':1}[new_memory_match.group(2)]
                           if new_memory_match else None)
            confirmed=(actual.get('ReqNodeList')==nodelist and actual.get('Partition')==partition
                       and actual.get('JobState') in {'PENDING','RUNNING'}
                       and actual.get('NumCPUs')==constraints.get('NumCPUs')
                       and actual_memory == memory
                       and actual.get('TimeLimit') == constraints.get('TimeLimit'))
            intent.update(status='verified' if confirmed else 'uncertain',exit_code=result.returncode,
                          scheduler_error=result.stderr[:2000],after=actual,verified=confirmed)
            rejection = _verified_rejection(intent, actual, job_id) if not confirmed else None
            if rejection:
                intent.update(status='verified_rejected', rejection_proof=rejection)
            write_checkpoint(journal,intent)
            if rejection and len(previous) < 2:
                time.sleep(0.1 * (len(previous) + 1))
                continue
            return {'ok':confirmed,'status':intent['status'],'job_id':str(job_id),'update_id':update_id,
                    'requested':intent['requested'],'effective_constraints':actual,'journal_path':str(journal),
                    'not_cancelled':True,'not_resubmitted':True,'scientific_parameters_unchanged':True,
                    'memory_changed': old_memory != memory if confirmed else False,
                    'retryable':bool(rejection),'requires_user':False if rejection else not confirmed,
                    'attempts_exhausted':bool(rejection and len(previous) >= 2),
                    'next_action':('continue monitoring original job; scheduler allocation is not guaranteed' if confirmed else
                        'Bounded transient retries exhausted; reassess controller/cluster limitations. No assurance that another node solves the rejection; never cancel/resubmit as fallback.' if rejection else
                        'reconcile this existing job before further updates; never cancel/resubmit as fallback')}
    finally:lock.release()
