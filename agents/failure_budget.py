"""Bound repeated identical native failures across an individual user turn.

This is an advisory stop; it cannot approve, change a DAG or retry a job.
Only explicit error receipts count. No receipt, argument or error text is
retained in the compact state or exposed in the stop reply.
"""
import hashlib
import json
import re
from .recovery import failure_reason, result_object


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str,separators=(',',':')).encode()).hexdigest()


def _error_signature(tool, arguments, result):
    obj=result_object(result)
    if not isinstance(obj,dict) or not failure_reason(result):return None
    message=obj.get('error') or obj.get('reason')
    if not isinstance(message,str) or not message.strip():return None
    # Strip only volatile identities, not numbers or scientific parameters.
    message=re.sub(r'(?<![\w])/(?:[^\s/]+/)*[^\s/]+','<path>',message.strip())
    message=re.sub(r'\b[0-9a-f]{32,64}\b','<id>',message,flags=re.I)
    # Different scientific arguments or paths are distinct attempts, while a
    # model changing only its explanatory reason is not execution progress.
    targets={key:value for key,value in arguments.items()
             if key not in {'reason','explanation','comment','observed_at','time','call_id','tool_use_id'}}
    return _hash([tool,obj.get('error_kind'),obj.get('exception_type'),message[:1000],targets])


def progress_epoch(goal, runtime, draft):
    """Hash scientific/plan state and durable execution progress, not timestamps."""
    goal=goal or {};runtime=runtime or {};draft=draft or {}
    state={'goal':{k:goal.get(k) for k in ('version','approved_plan_version','execution_mode','method','gases','parameters')},
      'runtime':{k:runtime.get(k) for k in ('plan_version','status','user_paused')},
      'nodes':{key:{k:value.get(k) for k in ('status','phase','job_ids')} for key,value in (runtime.get('nodes') or {}).items()},
      'draft':{k:draft.get(k) for k in ('nodes','completion_criteria')}}
    return _hash(state)


class CrossDelegationFailureBudget:
    def __init__(self,limit=3):
        if type(limit) is not int or limit<2:raise ValueError('invalid failure limit')
        self.limit=limit;self.epoch=None;self.counts={}

    def observe(self,tool,arguments,result,epoch):
        if self.epoch != epoch:
            self.epoch=epoch;self.counts={}
        sig=_error_signature(tool,arguments or {},result)
        if sig is None:
            # Success of this same operation proves its former failure changed.
            obj=result_object(result)
            if not failure_reason(result) and not (isinstance(obj,dict) and (obj.get('blocked') or obj.get('executed') is False)):
                self.counts={k:v for k,v in self.counts.items() if v['tool']!=tool}
            return None
        entry=self.counts.setdefault(sig,{'tool':tool,'count':0})
        entry['count']+=1
        if entry['count']>=self.limit:
            return {'tool':tool,'attempts':entry['count'],'error_signature':sig,'status':'unresolved_repeated_failure','automatic_retry':False}
        return None
