"""A specialist returning is an exchange, not proof that its task executed.

Legacy framework patchers have inspection capabilities, not a trusted source
publication/reload executor. Their prose and generic write_file/shell receipts
cannot authenticate installed code. Scientific acceptance remains in the DAG.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import stat


def _observed_call(metadata, evidence_root):
    cid=metadata.get('call_id')
    if not evidence_root or not isinstance(cid,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',cid):
        return None
    path=Path(evidence_root)/(cid+'.json')
    try:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size>2*1024*1024:return None
            raw=b''
            while len(raw)<=2*1024*1024:
                part=os.read(fd,65536)
                if not part:break
                raw+=part
        finally:os.close(fd)
        if len(raw)>2*1024*1024:return None
        call=json.loads(raw)
        if (call.get('call_id')!=cid or call.get('tool')!=metadata.get('tool') or call.get('agent')!=metadata.get('agent')
                or call.get('params')!=metadata.get('params')
                or call.get('time')!=metadata.get('time')
                or hashlib.sha256(str(call.get('result','')).encode()).hexdigest()!=metadata.get('result_sha256')):
            return None
        result=call.get('result')
        if isinstance(result,str):result=json.loads(result)
        return result if isinstance(result,dict) else {}
    except (OSError,ValueError,TypeError):return None


def delegation_outcome(agent_name, calls, returned_text, *, evidence_root=None):
    """Describe only this segment's authenticated native exchange.

    No model text, generic write receipt or successful read verifies a framework
    repair. Source application requires a separate controlled patch executor,
    before/after source hashes, independent checks and loaded process identity.
    That capability is absent from the legacy handoff and cannot be invented by
    a field named repair_applied in a tool result.
    """
    calls=list(calls or [])[-128:]
    observed=[];blocked=[];failed=[];unverified=[]
    for metadata in calls:
        if metadata.get('agent')!=agent_name:continue
        cid=metadata.get('call_id')
        result=_observed_call(metadata,evidence_root)
        if result is None:
            unverified.append(cid);continue
        observed.append(cid)
        if result.get('blocked') or result.get('skipped') or result.get('executed') is False:
            blocked.append(cid)
        elif metadata.get('failed') or result.get('isError') or result.get('error'):
            failed.append(cid)
    has_text=isinstance(returned_text,str) and bool(returned_text.strip())
    repair=agent_name=='patcher'
    status=('repair_execution_required' if repair else 'returned' if has_text or set(observed)-set(blocked)-set(failed)
            else 'blocked' if blocked else 'failed' if failed else 'no_delivery')
    label=('修复建议已返回，尚未实施' if repair and has_text else '修复尚未实施'
           if repair else '说明已返回，完成情况待核验' if status=='returned'
           else '委派未执行' if status=='blocked' else '委派调用失败' if status=='failed' else '尚无交付')
    receipt={'schema_version':1,'agent':agent_name,'status':status,'label':label,
        'response_returned':True,'text_returned':has_text,'task_success_verified':False,
        'evidence_call_ids':observed,'not_executed_call_ids':blocked,
        'failed_call_ids':failed,'unverified_call_ids':unverified,
        'scientific_acceptance':False,'retry_granted':False}
    if repair:
        receipt.update(repair_applied=False,repair_verified=False,loaded_executor_verified=False,
            required_capability='controlled_framework_patch_executor',
            capability_available=False,
            next_action='Preserve this repair proposal and native evidence; request a controlled framework repair capability. '
                'Do not repeat the same unavailable handoff, claim applied source changes, relax scientific inputs or grant retry.')
    return receipt


def delegation_progress(delegations):
    lines=['[系统·委派返回清单] 返回说明不等于执行或验收成功。']
    for item in delegations:
        label=item.get('outcome',{}).get('label') or '待返回'
        lines.append(f"  {item['agent']} · {label}：{item.get('task','')}")
    lines.append('按真实DAG、执行回执与验收证据继续；修复建议未实施时不得标为修复完成或重复同一无能力委派。')
    return '\n'.join(lines)
