"""Bounded current workflow status for delivery, never scientific authority.

No raw error, tool result, log, history or configuration is copied. A failed
node and an associated job ID are not a diagnosis or proof of convergence.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


FAILURES = frozenset({'failed', 'validation_failed'})
STATUSES = frozenset({'pending','running','waiting_jobs','waiting_prerequisite','prefinish',
    'succeeded','failed','validation_failed','uncertain','needs_resources','not_started'})
SCOPE_FIELDS = ('method','gases','parameters','execution_mode','execution_authorized')


def _identity(value, limit=128):
    return value if isinstance(value,str) and re.fullmatch(r'[A-Za-z0-9_.+-]{1,'+str(limit)+'}',value) else None


def current_task_status(state, *, username, conv_id, goal, evidence_root):
    """Project current approved scope only; caller supplies its authorized owner."""
    if (not isinstance(state,dict) or not isinstance(goal,dict) or not evidence_root
            or goal.get('execution_authorized')is not True):
        return None
    root=Path(evidence_root).absolute()
    if (state.get('username'),state.get('conv_id'))!=(username,conv_id):return None
    if (root.parent.parent.name,root.parent.name)!=(username,conv_id) or root.name!='evidence':return None
    plan=goal.get('approved_plan_version');saved=state.get('goal_contract') or {}
    if not isinstance(saved,dict):return None
    if (type(plan)is not int or plan<1 or state.get('plan_version')!=plan
            or saved.get('approved_plan_version')!=plan
            or type(saved.get('version'))is not int or type(goal.get('version'))is not int
            or saved['version']>goal['version']
            or any(saved.get(key)!=goal.get(key) for key in SCOPE_FIELDS)):
        return None
    approved=goal.get('approved_nodes') or []
    if not isinstance(approved,list) or not 0<len(approved)<=256:return None
    keys=[item.get('step_id') for item in approved if isinstance(item,dict)]
    if (len(keys)!=len(approved) or any(not isinstance(k,str) or not 0<len(k)<=256
            or any(ord(c)<32 for c in k) for k in keys) or len(set(keys))!=len(keys)):
        return None
    workflow_id=hashlib.sha256(json.dumps([username,conv_id]).encode()).hexdigest()[:24]
    if state.get('workflow_id',workflow_id)!=workflow_id:return None
    nodes=state.get('nodes') or {}
    if not isinstance(nodes,dict):return None
    incomplete=[];failed=[];counts=Counter();identities=[]
    for key in keys:
        node=nodes.get(key)or {}
        if not isinstance(node,dict):return None
        status=node.get('status','not_started')
        status=status if isinstance(status,str) and status in STATUSES else 'unknown'
        counts[status]+=1
        jobs=[]
        values=node.get('job_ids')or []
        if not isinstance(values,(list,tuple)) or len(values)>512:return None
        for value in values:
            value=str(value)
            if re.fullmatch(r'[0-9][A-Za-z0-9_.+\[\]-]{0,127}',value) and value not in jobs:jobs.append(value)
        ref=node.get('result_ref')or {}
        ref=ref if isinstance(ref,dict) else {}
        identities.append([key,status,_identity(node.get('token')),_identity(node.get('attempt_id')),
                           sorted(jobs),_identity(ref.get('call_id'))])
        if status=='succeeded':continue
        item={'step_id':key,'status':status,'job_ids':jobs[:64],
              'attempt_id':_identity(node.get('attempt_id')),'token':_identity(node.get('token'))}
        call_id=ref.get('call_id')
        if isinstance(call_id,str) and call_id.isalnum() and len(call_id)<=128:
            expected=root/(call_id+'.json')
            if ref.get('evidence_path')==str(expected):
                item['result_ref']={'call_id':call_id,'evidence_path':str(expected)}
        incomplete.append(item)
        if status in FAILURES:failed.append(item)
    # Failures always precede other incompleteness; old successes/history never
    # occupy this bounded detail budget or hide the latest failed attempt.
    incomplete.sort(key=lambda item:(item['status'] not in FAILURES,item['step_id']))
    detail=incomplete[:32]
    facts={'schema_version':1,'scope':{'username':username,'conv_id':conv_id},
        'workflow_id':workflow_id,'goal_version':goal['version'],'runtime_goal_version':saved['version'],
        'plan_version':plan,'workflow_status':state.get('status') if state.get('status')in
            {'active','needs_user','completed','retired_by_user'} else 'unknown',
        'user_paused':state.get('user_paused')is True,
        'complete':state.get('status')=='completed' and not incomplete,
        'counts':dict(counts),'failed_nodes':[dict(item,job_ids=list(item['job_ids']))for item in failed[:32]],
        'incomplete_nodes':detail,'omitted_nodes':max(0,len(incomplete)-len(detail)),
        'node_identity_sha256':hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()}
    facts['status_sha256']=hashlib.sha256(json.dumps(facts,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
    return facts


def attach_task_status(text, facts):
    """A factual status envelope cannot be replaced by rejected draft prose."""
    if not facts or facts.get('complete')is True or not facts.get('incomplete_nodes'):
        return text
    lines=[f"当前任务尚未完成（批准计划 {facts['plan_version']}）。"]
    for item in facts['incomplete_nodes']:
        label='失败/未通过节点' if item['status']in FAILURES else '未完成节点'
        safe_step=re.sub(r'([\\`*_{}\[\]()<>!#])',r'\\\1',item['step_id'])
        row=f"{label}：{safe_step}（{item['status']}）"
        if item.get('job_ids'):row+='；关联作业 '+', '.join(item['job_ids'])
        if item.get('attempt_id'):row+='；当前尝试 '+item['attempt_id']
        lines.append(row)
        if item.get('result_ref'):
            lines.append('关联工具回执：'+item['result_ref']['evidence_path'])
    if facts.get('omitted_nodes'):lines.append(f"另有 {facts['omitted_nodes']} 个未完成节点，详见当前任务状态。")
    if facts.get('failed_nodes'):
        lines.append('这是当前执行状态记录；失败原因与科学收敛须以原生诊断证据核验，不能据此推断。')
    summary='\n'.join(lines)
    return text if isinstance(text,str) and text.startswith(summary+'\n') else summary+'\n\n'+str(text or '')
