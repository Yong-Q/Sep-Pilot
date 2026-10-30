"""Owned RASPA case observations and conservative independent-worker budgets.

No simulator, scheduler or writer is invoked. Memory quantities are labelled
estimates, never measured RSS or a claim of scientific convergence.
"""
import copy
import math
import os
from pathlib import Path
import re
import time

from .workload_budget import _owned_fd,_fact,_digest,size_workload_budget


def collect_gcmc_workload(arguments,*,owned_root):
    result={'tool':'run_gcmc_batch','status':'invalid_workload'};fd=None
    try:
        gases=arguments.get('gases')
        if(not isinstance(gases,list)or not 1<=len(gases)<=16 or len(set(gases))!=len(gases)
                or any(not isinstance(g,str)or not re.fullmatch('[A-Za-z0-9_+-]{1,32}',g)for g in gases)):
            return result
        directory=Path(arguments['cif_dir']);fd=_owned_fd(Path(owned_root),directory,directory=True)
        before=os.fstat(fd);names=[]
        with os.scandir(fd)as scan:
            for index,entry in enumerate(scan):
                if index>=8192:raise ValueError('CIF directory exceeds entry bound')
                if entry.name.endswith('.cif'):
                    if not re.fullmatch('[A-Za-z0-9_.+-]{1,256}\\.cif',entry.name):raise ValueError('unsafe material identity')
                    names.append(entry.name)
                    if len(names)>512:raise ValueError('CIF batch exceeds case bound')
        if not names:return {**result,'status':'needs_prepared_inputs'}
        if len(names)*len(gases)>512:raise ValueError('native result case inventory exceeds supported bound')
        facts=[_fact(fd,name)for name in sorted(names)]
        if sum(f['bytes']for f in facts)>64*1024*1024:raise ValueError('CIF aggregate exceeds bound')
        after=os.fstat(fd)
        if(before.st_dev,before.st_ino,before.st_mtime_ns)!=(after.st_dev,after.st_ino,after.st_mtime_ns):raise ValueError('CIF inventory changed')
        science={key:arguments.get(key)for key in ('gases','temperature','pressure','cycles','force_field','unit_cells','electrostatics_method','charge_method')}
        return {'status':'ready','tool':'run_gcmc_batch','execution_model':'independent_processes',
            'work_items':len(facts)*len(gases),'threads_per_worker':1,'inputs':facts,
            'input_fingerprint':_digest({'cif_dir':str(directory),'files':facts,'science':science})}
    except (OSError,ValueError,KeyError,TypeError):return result
    finally:
        if fd is not None:os.close(fd)


def _positive(value):return type(value)is int and value>0


def allocate_gcmc_workload(request,node,snapshot,proposed,*,owned_root):
    facts=collect_gcmc_workload(node['arguments'],owned_root=owned_root)
    empty={'status':facts['status'],'candidates':[]}
    if facts['status']!='ready':
        if request.get('review_phase')=='planning':empty['status']='planning_only'
        return empty
    if not snapshot:return {'status':'needs_resource_evidence','candidates':[]}
    from .resource_policy import tool_profile
    profile=tool_profile(snapshot,'run_gcmc_batch',node['arguments'])
    per_worker=profile.get('per_worker_memory_mb');basis=profile.get('memory_basis')
    if per_worker is not None:
        if not _positive(per_worker)or not isinstance(basis,str)or not basis.strip():
            return {'status':'invalid_memory_evidence','candidates':[]}
        source='operator';kind='operator_conservative_estimate'
    elif proposed.get('per_worker_memory_mb')is not None:
        per_worker=proposed.get('per_worker_memory_mb');basis=proposed.get('per_worker_memory_basis')
        if not _positive(per_worker)or not isinstance(basis,str)or not basis.strip():
            return {'status':'needs_memory_estimate','candidates':[]}
        source=kind='resource_agent_estimate'
    else:
        per_worker=proposed.get('memory_mb')or node['arguments'].get('memory_mb')
        if not _positive(per_worker):return {'status':'needs_memory_estimate','candidates':[]}
        source=kind='single_worker_budget_extrapolation'
        basis='Conservative per-worker extrapolation of the delegated single-worker RAM budget; estimate, not measured peak RSS.'
    overhead=profile.get('coordinator_memory_mb',0)
    if type(overhead)is not int or overhead<0:return {'status':'invalid_memory_evidence','candidates':[]}
    memory={'source':source,'basis':basis,'per_worker_mb':per_worker,'unit':'per_worker','workload_fingerprint':facts['input_fingerprint']}
    constraints=[dict(c)for c in request.get('resource_constraints',[])if c.get('step_id')in(None,node['step_id'])]
    if profile.get('worker_cap')is not None:
        constraints.append({'source':'operator','field':'workers','kind':'max','value':profile['worker_cap'],'basis':basis})
    if profile.get('cpu_cap')is not None:
        constraints.append({'source':'operator','field':'cpus','kind':'max','value':profile['cpu_cap'],'basis':basis})
    try:
        pins=request.get('resource_pins',{}).get(node['step_id'],{})
        if not isinstance(pins,dict)or any(k not in {'nodelist','partition'}for k in pins):raise ValueError('invalid pins')
        for values in pins.values():
            if not isinstance(values,list)or not values or any(not isinstance(v,str)or not re.fullmatch('[A-Za-z0-9_.-]+',v)for v in values):raise ValueError('invalid pins')
        view=copy.deepcopy(snapshot)
        view['nodes']={k:v for k,v in view.get('nodes',{}).items()if not pins.get('nodelist')or k in pins['nodelist']}
        if pins.get('partition'):
            for value in view['nodes'].values():value['partitions']=[p for p in value.get('partitions',[])if p in pins['partition']]
    except (TypeError,AttributeError,ValueError):return {'status':'invalid_constraints','candidates':[]}
    partition_facts=view.get('partition_facts')or{};stamp=partition_facts.get('observed_at')
    fresh=(partition_facts.get('verified')is True and partition_facts.get('source')=='scontrol_show_partition'
        and type(stamp)in(int,float)and math.isfinite(stamp)and 0<=time.time()-stamp<=120)
    from .job_control import _wall_seconds
    def compatible(candidate):
        if not fresh:return False
        limit=(partition_facts.get('limits')or{}).get(candidate['partition'])or{}
        try:
            wanted=_wall_seconds(node['arguments'].get('walltime','08:00:00'))
            return limit.get('verified')is True and limit.get('state')=='UP'and wanted>0 and _wall_seconds(limit.get('max_time'))>=wanted
        except (TypeError,ValueError):return False
    for queued in(False,True):
        sized=size_workload_budget(facts,view,constraints=constraints,memory_evidence=memory,
            overhead_mb=overhead,for_queue=queued,allow_estimates=True)
        sized['candidates']=[c for c in sized['candidates']if compatible(c)]
        if sized['candidates']or sized['status']not in{'ready','waiting_resources'}:break
    if not sized['candidates']and sized['status']=='ready':sized['status']='waiting_resources'
    for candidate in sized['candidates']:
        candidate.update(budget_final=True,workload_tool='run_gcmc_batch',per_worker_memory_mb=per_worker,
            coordinator_memory_mb=overhead,estimate_kind=kind,memory_estimate_source=source,
            memory_profile_sha256=_digest({'source':source,'per_worker_mb':per_worker,'basis':basis,'overhead_mb':overhead}),
            scientific_acceptance=False)
    return sized


def allocation_matches_inputs(arguments,allocation,*,owned_root):
    """Recheck a final allocation against the actual immutable case inventory."""
    facts=collect_gcmc_workload(arguments,owned_root=owned_root)
    if(facts.get('status')!='ready'or allocation.get('budget_final')is not True
            or allocation.get('workload_tool')!='run_gcmc_batch'
            or allocation.get('input_fingerprint')!=facts['input_fingerprint']
            or allocation.get('work_items')!=facts['work_items']or allocation.get('threads_per_worker')!=1):return False
    cpus,workers,ram,per_worker=(allocation.get(k)for k in('cpus','workers','memory_mb','per_worker_memory_mb'))
    overhead=allocation.get('coordinator_memory_mb')
    if(not all(_positive(v)for v in(cpus,workers,ram,per_worker))or type(overhead)is not int or overhead<0
            or not cpus==workers<=facts['work_items']or ram<workers*per_worker+overhead):return False
    source=allocation.get('memory_estimate_source');basis=allocation.get('memory_basis')
    if source not in{'operator','resource_agent_estimate','single_worker_budget_extrapolation'}or not isinstance(basis,str)or not basis.strip():return False
    return allocation.get('memory_profile_sha256')==_digest({'source':source,'per_worker_mb':per_worker,'basis':basis,'overhead_mb':overhead})
