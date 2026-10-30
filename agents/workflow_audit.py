"""Authenticate job references in coordinator audit rows, never grant execution."""
import hashlib
import re
from .local_preparation_recovery import checkpoint_call_witness
from .report_recovery import _OwnedFiles,_same
from .recovery import result_object


def verified_job_observation(actor,runtime,state,row):
    """Only an actual owned inspection may explain a non-DAG job reference.

    Current or archived attempts may be referenced. Their scientific validity
    is deliberately not inferred: a rejected result can still have been read.
    Unknown jobs, typed executable rows and dispatch intents remain blockers.
    """
    if (row.get('tool')!='inspect_workflow_result' or 'expected_outputs'in row
            or row.get('status')!='completed' or row.get('done')is not True
            or any(row.get(key)for key in ('recovery_key','attempt_id','token','dispatch_phase',
                'input_dir','output_dir','output_files','path_manifest','resource_locks','input_bindings','branch_parent'))):
        return False
    args=row.get('arguments');target=args.get('step_id')if isinstance(args,dict)else None
    ids=row.get('job_ids');owner={'username':runtime.username,'conv_id':runtime.conv_id}
    if (not isinstance(target,str) or args!={'step_id':target}
            or not isinstance(ids,list) or not 0<len(ids)<=64 or len(ids)!=len(set(ids))
            or any(not isinstance(j,str) or not re.fullmatch(r'[1-9][0-9]*',j)for j in ids)
            or (state.get('username'),state.get('conv_id'))!=(runtime.username,runtime.conv_id)):
        return False
    calls=getattr(actor.memory,'tool_call_log',[])
    history=state.get('plan_history')or []
    if not isinstance(calls,list) or len(calls)>4096 or len(history)>64:return False
    candidates=[state.get('nodes',{}).get(target,{})]
    candidates.extend(item.get('nodes',{}).get(target,{})for item in history if isinstance(item,dict))
    from .parallel_workflow import WorkflowStore
    workflow_id=WorkflowStore.identity(runtime.username,runtime.conv_id)
    covered=set();remaining=64*1024*1024
    try:
        files=_OwnedFiles(runtime.root,owner)
        try:
            def read(path):
                nonlocal remaining
                value,fact=files.read(path,as_json=True);remaining-=fact['size']
                if remaining<0:raise ValueError('audit evidence exceeds bounded read budget')
                return value,fact
            for record in calls:
                if (not isinstance(record,dict) or record.get('tool')!='inspect_workflow_result'
                        or record.get('params')!=args or record.get('failed')is not False):continue
                cid=record.get('call_id')
                if (not isinstance(cid,str)or not re.fullmatch(r'[A-Za-z0-9]{1,128}',cid)
                        or sum(c.get('call_id')==cid for c in calls if isinstance(c,dict))!=1):continue
                path=files.root/'evidence'/(cid+'.json')
                if record.get('evidence_path')!=str(path):continue
                call,_fact=read(path);raw=call.get('result')
                fields=('call_id','time','agent','tool','params','failed')
                if (not isinstance(raw,str) or not _same({k:record.get(k)for k in fields},{k:call.get(k)for k in fields})
                        or record.get('result_sha256')!=hashlib.sha256(raw.encode()).hexdigest()):continue
                witness=checkpoint_call_witness(files,files.root/'session_checkpoint.json',call,require_result_hash=True)
                remaining-=witness['size']
                if remaining<0:return False
                result=result_object(raw)
                if (result.get('status')!='inspected' or result.get('read_only')is not True
                        or result.get('no_resubmission')is not True or result.get('workflow_id')!=workflow_id
                        or result.get('owner')!=[runtime.username,runtime.conv_id] or result.get('step_id')!=target
                        or type(result.get('plan_version'))is not int
                        or not 0<result['plan_version']<=state.get('plan_version',0)):continue
                for node in candidates:
                    if (not node or not node.get('token') or not node.get('attempt_id')
                            or node.get('tool_returned')is not True or node.get('jobs_confirmed_terminal')is not True
                            or result.get('token')!=node['token']or result.get('attempt_id')!=node['attempt_id']
                            or result.get('job_ids')!=node.get('job_ids')or result.get('contract')!=node.get('contract')):continue
                    ref=node.get('result_ref')or {};pid=ref.get('call_id');native_ids=node.get('job_ids')or []
                    if (not isinstance(pid,str)or not re.fullmatch(r'[A-Za-z0-9]{1,128}',pid)
                            or ref.get('evidence_path')!=str(files.root/'evidence'/(pid+'.json'))
                            or result.get('source_execution_call_id')!=pid):continue
                    producer,fact=read(ref['evidence_path'])
                    effective=node.get('effective_arguments')or node['contract'].get('arguments',{})
                    if (result.get('source_execution_sha256')!=fact['sha256']or producer.get('call_id')!=pid
                            or producer.get('tool')!=node['contract'].get('tool')or producer.get('failed')is not False
                            or not _same(producer.get('params'),effective)
                            or not _same(result_object(producer.get('result')),result_object(node.get('result')))):continue
                    produced=result_object(producer.get('result'));produced_ids=set(map(str,produced.get('job_ids')or []))
                    if produced.get('job_id'):produced_ids.add(str(produced['job_id']))
                    if produced_ids!=set(native_ids):continue
                    witness=checkpoint_call_witness(files,ref.get('checkpoint_path',''),producer,step_id=target)
                    remaining-=witness['size']
                    if remaining<0:return False
                    ledger,_ledger_fact=read(files.root/'recovery_state.json');entry=ledger.get(node.get('recovery_key'),{})
                    if (entry.get('attempt_id')!=node['attempt_id']or entry.get('tool')!=node['contract'].get('tool')
                            or entry.get('status')not in {'completed_unverified','completed'}
                            or set(entry.get('job_ids')or [])!=set(native_ids)or not _same(entry.get('params'),effective)):continue
                    watched=[runtime.job_watch.get(j)or {}for j in native_ids]if runtime.job_watch else []
                    if (len(watched)!=len(native_ids)or not watched or any(
                            str(job.get('job_id'))!=str(jid)
                            or (job.get('username'),job.get('conv_id'))!=(runtime.username,runtime.conv_id)
                            or job.get('attempt_id')!=node['attempt_id']or job.get('recovery_key')!=node.get('recovery_key')
                            or job.get('tool')!=node['contract'].get('tool')or job.get('terminal')is not True
                            or job.get('state')!='COMPLETED'or job.get('failed')is not False for jid,job in zip(native_ids,watched))):continue
                    covered.update(set(native_ids)&set(ids));break
                if covered==set(ids):return True
        finally:files.close()
    except (OSError,ValueError,KeyError,TypeError,AttributeError):return False
    return False
