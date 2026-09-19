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


def retarget_queued_job(job_id, nodelist, partition, username, conv_id, project_root, memory_mb=None):
    from .job_watch import get_watch
    from .node_inventory import node_inventory, candidates
    from .state_io import ConversationLock, write_checkpoint
    from .resource_review import assess_job_resources
    if not re.fullmatch(r'\d+',str(job_id)) or not re.fullmatch(r'[A-Za-z0-9_.-]+',nodelist) or not re.fullmatch(r'[A-Za-z0-9_.-]+',partition):
        raise ValueError('retarget requires one literal node/partition and numeric job ID')
    watch=get_watch();job=watch.get(str(job_id)) or {}
    if (job.get('username'),job.get('conv_id'))!=(username,conv_id):raise PermissionError('job belongs to another user/session')
    root=Path(project_root)/'runs'/username/conv_id
    lock=ConversationLock(root/'job_control'/f'{job_id}.lock')
    if not lock.acquire(blocking=False):return {'blocked':True,'executed':False,'reason':'job scheduling control already in progress'}
    try:
        if (watch.get(str(job_id)) or {}).get('cancel_request', {}).get('status') in {'intent', 'sent', 'uncertain'}:
            return {'blocked': True, 'executed': False, 'reason': 'cancellation outcome is unresolved; reconcile the original job before retargeting'}
        for index,path in enumerate((root/'job_control').glob(f'{job_id}_*.json')):
            if index>=256:raise ValueError('bounded job-control audit exceeded; reconcile history before updating')
            if json.loads(path.read_text()).get('status') in {'intent','uncertain'}:
                return {'blocked':True,'executed':False,'requires_user':True,'reason':'previous scheduling update is unresolved; inspect original journal and same job before any new update'}
        assessment=assess_job_resources(str(job_id));constraints=assessment.get('effective_constraints',{})
        if constraints.get('JobState')!='PENDING':
            return {'blocked':True,'executed':False,'requires_user':True,'reason':'Only a verified PENDING job may be retargeted; running/unknown/terminal jobs are not touched'}
        match=re.search(r'(?:^|,)mem=(\d+)([KMGT]?)',constraints.get('ReqTRES') or '')
        if not match:raise ValueError('effective requested RAM is unknown; cannot retarget safely')
        old_memory=int(match[1])*{'K':1/1024,'M':1,'G':1024,'T':1024**2,'':1}[match[2]]
        inventory=assessment['node_inventory']
        profile=inventory.get('policy',{}).get('tool_profiles',{}).get(job.get('tool'),{})
        if memory_mb is None:
            memory = old_memory
        else:
            if not isinstance(memory_mb, int) or isinstance(memory_mb, bool) or memory_mb < 64:
                raise ValueError('memory_mb must be a finite integer >=64 MiB')
            node_fact=inventory.get('nodes',{}).get(nodelist,{})
            available=int(node_fact.get('memory',{}).get('available_for_scheduling_mib') or 0)
            minimum=int(profile.get('minimum_memory_mb') or 64)
            fraction=float(profile.get('memory_fraction_by_partition',{}).get(
                partition,profile.get('memory_fraction_of_available')) or 0)
            profiled=max(minimum,int(available*fraction)) if 0 < fraction <= 1 and available >= minimum else None
            if profiled is None or memory_mb != profiled:
                return {'blocked':True,'executed':False,'requires_user':True,
                        'reason':'memory change must match the fresh scheduler-available fraction in the operator-owned tool profile; preserve the original job'}
            memory = memory_mb
        eligible=[node for node in candidates(inventory,cpus=int(constraints['NumCPUs']),memory_mb=memory,min_glibc=profile.get('min_glibc'))
                  if node['name']==nodelist and partition in node['partitions']]
        if not eligible:return {'blocked':True,'executed':False,'requires_user':True,'reason':'Fresh CPU/RAM/partition facts do not support the requested node; preserve original job'}
        update_id=uuid.uuid4().hex
        journal=root/'job_control'/f'{job_id}_{update_id}.json'
        intent={'update_id':update_id,'job_id':str(job_id),'status':'intent','old_constraints':constraints,
                'requested':{'nodelist':nodelist,'partition':partition,'memory_mb':int(memory)},'created_at':time.time(),
                'resource_profile':job.get('tool') if memory_mb is not None else '',
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
                   and actual_memory == memory)
        intent.update(status='verified' if confirmed else 'uncertain',exit_code=result.returncode,
                      scheduler_error=result.stderr[:2000],after=actual,verified=confirmed)
        write_checkpoint(journal,intent)
        return {'ok':confirmed,'status':intent['status'],'job_id':str(job_id),'update_id':update_id,
                'requested':intent['requested'],'effective_constraints':actual,'journal_path':str(journal),
                'not_cancelled':True,'not_resubmitted':True,'scientific_parameters_unchanged':True,
                'memory_changed': old_memory != memory,
                'next_action':'continue monitoring original job; scheduler allocation is not guaranteed' if confirmed else 'reconcile this existing job before further updates; never cancel/resubmit as fallback'}
    finally:lock.release()
