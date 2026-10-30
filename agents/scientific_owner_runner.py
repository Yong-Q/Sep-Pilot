"""Isolated, bounded owner-review dialogue; native controller retains all writes."""
import copy
import hashlib
import json
import os
from pathlib import Path
from .owner_review_diagnostics import ReviewFailure


def run_owner_review(actor, event, *, control_valid):
    from .agent import Agent
    from .defns import resolve_agent, _tool
    from .watch_context import get_context
    from .workload_budget import _owned_fd
    if not callable(control_valid):
        raise ReviewFailure('identity_mismatch')
    binding=copy.deepcopy((event.get('payload') or {}).get('review_binding') or {})
    owner=get_context()
    required=('username','conv_id','workflow_id','step_id','token','attempt_id','contract_sha256','producer_call_id','dossier_call_id')
    if (event.get('kind')!='worker_scientific_owner_review' or any(not isinstance(binding.get(k),str) or not binding[k] for k in required)
            or any(owner.get(k)!=binding[k] for k in ('username','conv_id'))
            or not isinstance(binding.get('allowed_read_paths'),list)):
        raise ReviewFailure('identity_mismatch')
    contract=copy.deepcopy((event.get('payload')or{}).get('review_contract'))
    if (not isinstance(contract,dict) or hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()!=binding['contract_sha256']):
        raise ReviewFailure('identity_mismatch')
    original=resolve_agent(contract.get('agent'))
    scope=Path(actor._evidence_root).absolute().parent
    if scope.name!=binding['conv_id'] or scope.parent.name!=binding['username']:
        raise ReviewFailure('identity_mismatch')
    paths=set()
    for value in binding['allowed_read_paths']:
        if not isinstance(value,str):raise ReviewFailure('evidence_invalid')
        path=Path(value)
        if not path.is_absolute() or '..' in path.parts or not path.is_relative_to(scope):
            raise ReviewFailure('evidence_invalid')
        paths.add(str(path))
    def guard():
        # The controller distinguishes goal/turn/attempt supersession before
        # the cooperative interrupt callback collapses these to one boolean.
        if control_valid() is not True or actor._interrupt_requested():
            raise ReviewFailure('controls_changed')
        current=get_context()
        if any(current.get(k)!=binding[k] for k in ('username','conv_id')):
            raise ReviewFailure('controls_changed')
    guard()
    role=Agent(name=original.name,model=original.model,max_turns=3,
        instructions=('Review only this exact native workflow node and attempt against its approved scientific contract. '
            'Read the supplied dossier first. Inspect fresh native results where needed. Unknown convergence remains unknown. '
            'Do not assume numeric presence proves convergence, completeness or scientific suitability. '
            'Use finish_workflow_node only to REQUEST the original controller finish gate with actual inspection call IDs. '
            'This dialogue cannot finish, submit, retry, revalidate, write, cancel, change scope or hand off. '
            'If evidence is unsupported or insufficient, explain the missing evidence and stop. Never invent success.'),
        functions=[_tool(n)for n in ('inspect_workflow_result','read_file','finish_workflow_node')])
    restriction=role.instructions
    role.instructions=lambda context:original.get_instructions(context)+'\n[NATIVE RESTRICTED SCIENTIFIC REVIEW]\n'+restriction
    previous=actor.current_agent
    actor.current_agent=role
    start=len(actor.memory.tool_call_log)
    actor.messages.append({'role':'user','content':'[NATIVE SCIENTIFIC OWNER REVIEW]\n'+json.dumps(event,ensure_ascii=False,default=str)})
    evidence_ids=set();finish=None;summary='';used=0;remaining=8*1024*1024
    try:
        for _ in range(3):
            guard()
            response=actor._call_api(role)
            guard()
            actor.messages.append({'role':'assistant','content':response.content})
            calls=[b for b in response.content if getattr(b,'type','')=='tool_use']
            if not calls:
                summary='\n'.join(getattr(b,'text','')for b in response.content if getattr(b,'type','')=='text')[:16000]
                break
            results=[]
            for call in calls:
                guard();used+=1
                params=call.input
                raw={'blocked':True,'executed':False,'reason':'restricted scientific owner-review capability'}
                eligible=False
                if used<=8 and isinstance(params,dict):
                    if call.name=='inspect_workflow_result' and params=={'step_id':binding['step_id']}:
                        callback=getattr(actor,'_on_workflow_result_inspect',None)
                        if callable(callback):
                            raw=callback(binding['step_id'])
                            guard()
                            eligible=(isinstance(raw,dict) and raw.get('status')=='inspected' and raw.get('read_only')is True
                                and not any(raw.get(k)for k in ('error','blocked','unsupported'))
                                and all(raw.get(k)==binding[k]for k in ('step_id','token','attempt_id','workflow_id','plan_version','goal_version'))
                                and raw.get('owner')==[binding['username'],binding['conv_id']]
                                and raw.get('job_ids')==binding['job_ids']
                                and raw.get('source_execution_call_id')==binding['producer_call_id']
                                and isinstance(raw.get('contract'),dict)
                                and hashlib.sha256(json.dumps(raw['contract'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==binding['contract_sha256'])
                            if not isinstance(raw,dict) or (raw.get('status')=='inspected' and not eligible):
                                raw={'blocked':True,'executed':False,'reason':'native inspection identity does not match this review binding'}
                    elif call.name=='read_file' and set(params)=={'path'} and isinstance(params.get('path'),str) and params['path']in paths:
                        fd=_owned_fd(scope,Path(params['path']))
                        try:
                            before=os.fstat(fd)
                            if not 0<before.st_size<=min(2*1024*1024,remaining):raise ReviewFailure('evidence_invalid')
                            chunks=[];count=0
                            while count<=before.st_size:
                                chunk=os.read(fd,min(65536,before.st_size+1-count))
                                if not chunk:break
                                chunks.append(chunk);count+=len(chunk)
                            data=b''.join(chunks);after=os.fstat(fd)
                        finally:os.close(fd)
                        linked=_owned_fd(scope,Path(params['path']))
                        try:visible=os.fstat(linked)
                        finally:os.close(linked)
                        identity=lambda s:(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
                        if len(data)!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(visible):
                            raise ReviewFailure('sources_changed')
                        remaining-=len(data)
                        raw={'ok':True,'read_only':True,'path':params['path'],'sha256':hashlib.sha256(data).hexdigest(),'content':data.decode('utf-8',errors='replace')}
                        guard()
                    elif call.name=='finish_workflow_node' and set(params)=={'step_id','evidence_call_ids','conclusion'}:
                        ids=params.get('evidence_call_ids');conclusion=params.get('conclusion')
                        if (params.get('step_id')==binding['step_id'] and isinstance(ids,list) and 0<len(ids)<=8
                                and all(isinstance(cid,str) and cid in evidence_ids for cid in ids)
                                and isinstance(conclusion,str) and 0<len(conclusion.strip())<=16000):
                            finish=copy.deepcopy(params)
                            raw={'ok':True,'finish_requested':True,'applied':False,'scientific_acceptance':False}
                guard()
                encoded=json.dumps(raw,ensure_ascii=False,default=str)
                failed=not isinstance(raw,dict) or bool(raw.get('error') or raw.get('blocked') or raw.get('isError') or raw.get('ok')is False)
                actor.memory.record_tool_call(role.name,call.name,params,encoded,failed=failed)
                if eligible:evidence_ids.add(actor.memory.tool_call_log[-1]['call_id'])
                results.append({'type':'tool_result','tool_use_id':call.id,'content':encoded})
            actor.messages.append({'role':'user','content':results})
            actor._checkpoint('scientific_owner_review_native_calls')
            guard()
            if finish or used>=8:break
        guard()
        result={'status':'finish_requested'if finish else'incomplete','finish_request':finish,
            'reason':summary or ('Native controller must validate evidence and apply the original finish gate.'if finish else'No evidence-backed finish request obtained.'),
            'evidence_calls':copy.deepcopy(actor.memory.tool_call_log[start:]),'review_binding':binding,
            'tool_calls_used':used,'applied':False,'retry_granted':False,'scientific_acceptance':False}
        actor._checkpoint('scientific_owner_review_stage_receipt')
        guard()
        return result
    finally:
        actor.current_agent=previous
