"""Bounded proofs for reuse of completed read-only observations, never dispatch."""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
from urllib.parse import urlsplit
from .report_recovery import _OwnedFiles, _same, _canonical
from .workflow_patch import execution_contract
from .recovery import result_object, failure_reason

SUPPORTED = frozenset({'inspect_path','validate_framework_charges','query_literature'})
_RESOURCE = {'memory_mb','nodelist','partition','num_processes','cpus_per_task','resource_review_id'}


def verify_completed_readonly(state, step_id, session_root, approved_goal, *, upstream_jobs):
    owner={k:state.get(k)for k in ('username','conv_id')};node=state.get('nodes',{}).get(step_id,{})
    contract=node.get('contract',{});goal=approved_goal;saved=state.get('goal_contract',{})
    protected=('method','gases','parameters','execution_mode','approved_plan_version')
    approved=[n for n in goal.get('approved_nodes',[])if n.get('step_id')==step_id]
    if (state.get('user_paused') or node.get('status')!='succeeded' or node.get('phase')!='finish'
            or not node.get('token') or node.get('job_ids') or contract.get('tool')not in SUPPORTED
            or goal.get('execution_authorized')is not True or goal.get('approved_plan_version')!=state.get('plan_version')
            or not isinstance(goal.get('version'),int)or isinstance(goal.get('version'),bool)
            or not isinstance(saved.get('version'),int)or isinstance(saved.get('version'),bool)
            or goal['version']<saved['version']
            or not _same({k:saved.get(k)for k in protected},{k:goal.get(k)for k in protected})
            or len(approved)!=1 or not _same(execution_contract(contract),execution_contract(approved[0]))):
        raise ValueError('completed read-only reuse requires exact current approval, stopped result and unpaused scope')
    files=_OwnedFiles(session_root,owner);facts=[];budget=16*1024*1024
    def read(path,as_json=False):
        nonlocal budget
        candidate=files.owned(path)
        if candidate.lstat().st_size>budget:raise ValueError('read-only reuse exceeds bounded evidence budget')
        value,fact=files.read(candidate,as_json=as_json);budget-=fact['size'];facts.append(fact)
        return value,fact
    def native(prior,key):
        ref=prior.get('result_ref')or{};identifier=ref.get('call_id');evidence=files.owned(ref.get('evidence_path',''))
        if not isinstance(identifier,str)or evidence!=files.root/'evidence'/(identifier+'.json'):
            raise ValueError('native call identity is not owned')
        call,_=read(evidence,True);obj=result_object(call.get('result'));expected=prior.get('effective_arguments')or prior['contract']['arguments']
        science=lambda args:{k:v for k,v in args.items()if k not in _RESOURCE}
        if (call.get('call_id')!=identifier or call.get('tool')!=prior['contract']['tool'] or call.get('failed')is not False
                or failure_reason(obj) or not _same(science(call.get('params')or{}),science(expected))
                or not _same(obj,result_object(prior.get('result')))):
            raise ValueError('native call tool, input or result identity changed')
        checkpoint=files.owned(ref.get('checkpoint_path',''))
        if not checkpoint.is_relative_to(files.root/'workers')or checkpoint.name!='session_checkpoint.json':
            raise ValueError('producer checkpoint is not a worker checkpoint')
        producer,_=read(checkpoint,True)
        if producer.get('owner')!={**owner,'step_id':key}:raise PermissionError('producer checkpoint owner differs')
        matching=[c for c in producer.get('memory',{}).get('tool_call_log',[])if c.get('call_id')==identifier]
        if (len(matching)!=1 or matching[0].get('failed')is not False
                or matching[0].get('tool')!=call.get('tool') or not _same(matching[0].get('params'),call.get('params'))
                or matching[0].get('evidence_path')!=str(evidence)):
            raise ValueError('producer checkpoint does not prove the original native call')
        return obj
    def artifacts(prior):
        from .parallel_workflow import artifact_base
        from .output_contract import output_path,artifact_fact
        result=result_object(prior.get('result'));base=artifact_base(prior['contract'],files.root.parents[2],files.root,result)
        observed={str(output_path(spec,base)):artifact_fact(spec,base)for spec in prior['contract'].get('expected_outputs',[])}
        if not _same(observed,prior.get('artifacts',{})):raise ValueError('completed native artifact changed')
        return observed
    try:
        result=native(node,step_id);artifacts(node)
        tool,args=contract['tool'],contract['arguments']
        historical=False
        if tool=='inspect_path':
            from bimem_agent.result_inspector import inspect_path
            path=files.owned(args['path'])
            if path.resolve(strict=True)!=path or not path.is_dir():raise ValueError('only literal owned directory observations can be reused')
            entries=list(path.iterdir())
            if len(entries)>512 or len(entries)>args.get('max_entries',20):raise ValueError('directory observation was truncated or exceeds bound')
            for entry in entries:
                if entry.is_symlink():raise ValueError('inspection input contains a symlink')
                if entry.is_file():read(entry)
            if not _same(inspect_path(str(path),args.get('max_entries',20)),result):raise ValueError('directory observation changed')
            producers=[]
            for dependency in contract.get('depends_on',[]):
                source=state['nodes'].get(dependency,{})
                if source.get('status')!='succeeded'or source.get('phase')!='finish'or not source.get('node_verification')or not source.get('job_ids'):continue
                source_result=native(source,dependency);source_artifacts=artifacts(source)
                native_ids={str(x)for x in source_result.get('job_ids',[])}
                if source_result.get('job_id'):native_ids.add(str(source_result['job_id']))
                if not set(map(str,source['job_ids'])).issubset(native_ids):raise ValueError('upstream native job IDs differ')
                for jid in source['job_ids']:
                    observed=upstream_jobs.get(str(jid),{})
                    if ((observed.get('username'),observed.get('conv_id'))!=(owner['username'],owner['conv_id'])
                            or observed.get('state')!='COMPLETED'or observed.get('terminal')is not True or observed.get('failed')is not False):
                        raise ValueError('upstream job completion is not owned and verified')
                if str(path)in source_artifacts and source_artifacts[str(path)]:producers.append(dependency)
            if not producers:raise ValueError('empty-artifact inspection lacks verified native producer outputs')
        elif tool=='validate_framework_charges':
            rows=result.get('results');directory=files.owned(args['cif_path'])
            if (result.get('ok')is not True or result.get('read_only_inputs')is not True or result.get('input_modified')is not False
                    or not isinstance(rows,list)or not rows or result.get('n_cifs')!=len(rows)or result.get('n_valid')!=len(rows)
                    or result.get('charge_source')!=args.get('charge_source')or result.get('expected_net_charge')!=float(args['expected_net_charge'])
                    or result.get('tolerance_e')!=float(args.get('tolerance',1e-6))):raise ValueError('charge report does not prove approved charge conditions')
            expected=sorted(directory.glob('*_pacmof.cif'))if directory.is_dir()else[directory]
            if len(expected)>512 or {str(p)for p in expected}!={row.get('cif_path')for row in rows}:raise ValueError('charge input membership changed')
            for row in rows:
                _,fact=read(row['cif_path'])
                if (row.get('ok')is not True or row.get('sha256')!=fact['sha256']
                        or not math.isfinite(float(row['cell_net_charge']))or not math.isfinite(float(row['residual_e']))
                        or abs(float(row['residual_e']))>float(args.get('tolerance',1e-6))):raise ValueError('charge input bytes or validated cell charge changed')
            report,report_fact=read(args['output_path'],True)
            metadata={'output_path','output_files','reused_existing_report','report_sha256'}
            if not _same(report,{k:v for k,v in result.items()if k not in metadata}):raise ValueError('immutable charge report differs from native receipt')
            if result.get('report_sha256')and result['report_sha256']!=report_fact['sha256']:raise ValueError('charge report hash changed')
        else:
            rows=result.get('results')
            if result.get('ok')is not True or not isinstance(rows,list)or not rows:raise ValueError('historical literature has no actual source records')
            for row in rows:
                address=row.get('source_url')or row.get('url');identifier=row.get('doi')or row.get('arxiv_id')or row.get('citation')
                if not row.get('title')or not(identifier or isinstance(address,str)and urlsplit(address).scheme=='https'and urlsplit(address).hostname):
                    raise ValueError('historical literature lacks verifiable bibliographic source identity')
            if args.get('require_both')and(not result.get('local')or not result.get('web')or result.get('evidence_complete')is not True):
                raise ValueError('required literature source coverage was not established')
            historical=True
        proof={'kind':'verified_completed_readonly',**owner,'step_id':step_id,'goal_version':goal['version'],
            'plan_version':state['plan_version'],'current_token':node['token'],
            'approved_goal_sha256':hashlib.sha256(_canonical(goal)).hexdigest(),
            'contract_sha256':hashlib.sha256(_canonical(execution_contract(contract))).hexdigest(),
            'result_sha256':hashlib.sha256(_canonical(result)).hexdigest(),'facts':facts,
            'historical_remote_observation':historical,'no_reexecution':True,'no_resubmission':True}
        proof['proof_sha256']=hashlib.sha256(_canonical(proof)).hexdigest()
        return proof
    finally:files.close()
