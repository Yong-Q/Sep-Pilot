"""Answer delivery identity and bounded source checks; no execution authority."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

POLICY_VERSION = 1


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def source_references(event):
    payload=event.get('payload',{})
    values=payload.get('source_inputs',[])+payload.get('native_result_sources',[])
    references=set();stack=list(values);visited=0
    while stack:
        value=stack.pop();visited+=1
        if visited>8192 or len(references)>144:
            return [('', 'source summary exceeds bounded verification limit')]
        if isinstance(value,dict):
            if isinstance(value.get('path'),str) and isinstance(value.get('sha256'),str):
                references.add((value['path'],value['sha256']))
            stack.extend(item for item in value.values() if isinstance(item,(dict,list)))
        elif isinstance(value,list):
            stack.extend(value)
    return sorted(references)


def sources_unchanged(references, max_bytes=64*1024*1024):
    """Hash exact cited source bytes, bounded across all files, never follow links."""
    remaining=max_bytes
    if len(references)>144: return False
    for name,expected in references:
        path=Path(name)
        try:
            if not path.is_absolute() or path.resolve(strict=True)!=path: return False
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                before=os.fstat(fd)
                if not stat.S_ISREG(before.st_mode) or not 0<before.st_size<=remaining: return False
                checksum=hashlib.sha256();size=0
                while size<=before.st_size:
                    part=os.read(fd,min(65536,before.st_size+1-size))
                    if not part: break
                    checksum.update(part);size+=len(part)
                after=os.fstat(fd);visible=os.stat(path,follow_symlinks=False)
                identity=lambda v:(v.st_dev,v.st_ino,v.st_size,v.st_mtime_ns,v.st_ctime_ns)
                if size!=before.st_size or identity(before)!=identity(after) or identity(after)!=identity(visible): return False
                if checksum.hexdigest()!=expected: return False
                remaining-=size
            finally: os.close(fd)
        except (OSError,ValueError,TypeError): return False
    return True


def source_link_identities(references, max_bytes=64*1024*1024):
    """Supplement an actual SHA check across later bounded status I/O.

    This is neither a hash proof nor a cache/permission for any mutation.
    Callers must still perform sources_unchanged inside the identity interval.
    """
    if len(references)>144:return None
    identities=[];remaining=max_bytes
    try:
        for name,expected in references:
            path=Path(name)
            if not path.is_absolute() or path.resolve(strict=True)!=path:return None
            fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            try:
                info=os.fstat(fd);visible=os.stat(path,follow_symlinks=False)
                identity=lambda value:(value.st_dev,value.st_ino,value.st_size,value.st_mtime_ns,value.st_ctime_ns)
                if (not stat.S_ISREG(info.st_mode) or not 0<info.st_size<=remaining
                        or identity(info)!=identity(visible)):return None
                identities.append((name,expected,identity(info)));remaining-=info.st_size
            finally:os.close(fd)
    except (OSError,ValueError,TypeError):return None
    return tuple(identities)


def draft_identity(username,conv_id,uid,goal,text,event):
    core={'policy_version':POLICY_VERSION,'username':username,'conv_id':conv_id,'turn_uid':uid,
          'goal_sha256':digest(goal),'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
          'sources':source_references(event),'source_gaps':event.get('payload',{}).get('source_gaps',[]),'native_bindings':event.get('payload',{}).get('native_bindings',[]),
          'workflow_status_sha256':(event.get('payload',{}).get('workflow_status')or {}).get('status_sha256'),
          'runtime_completed':event.get('payload',{}).get('runtime_completed') is True,'required_families':event.get('payload',{}).get('required_evidence_families',[])}
    return {**core,'draft_id':digest(core)}


def persist_draft(evidence_root,draft,text,event):
    """Immutable recovery evidence; caller persists its small context pointer."""
    if evidence_root is None: return None
    root=Path(evidence_root)
    if root.resolve(strict=True)!=root.absolute(): raise ValueError('unowned answer evidence root')
    path=root/('answer-draft-'+draft['draft_id']+'.json')
    raw=json.dumps({'draft':draft,'text':text,'review_event':event},sort_keys=True,ensure_ascii=False).encode()
    if len(raw)>4*1024*1024: raise ValueError('answer draft exceeds bounded evidence size')
    try: fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    except FileExistsError:
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            info=os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 0 < info.st_size <= 4*1024*1024:
                raise ValueError('invalid immutable answer draft')
            old=json.loads(os.read(fd,info.st_size+1))
            if old.get('draft') != json.loads(json.dumps(draft)) or old.get('text') != text:
                raise ValueError('immutable answer draft identity mismatch')
        finally: os.close(fd)
    else:
        with os.fdopen(fd,'wb') as stream:
            stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return str(path)


def limited_answer(event,reason):
    """Do not leak rejected prose or invent a replacement scientific conclusion."""
    lines=['已有证据与产物已保留；本次答复的科学解释未完成证据核验，不能作为已验证结论。',
           '核验限制：'+str(reason)[:800]]
    for item in event.get('payload',{}).get('native_result_sources',[])[:16]:
        jobs=', '.join(str(v) for v in item.get('job_ids',[]))
        lines.append(f"已有结果证据：{item['path']}"+(f'；作业 {jobs}' if jobs else ''))
    lines.append('未重新计算或提交作业；可继续核对这些已有产物。')
    return '\n'.join(lines)


def completed_notification_binding(event, runtime, delivery, *, goal_sha256):
    """An already delivered exact native attempt needs no new model turn."""
    if (event.get('kind') not in {'worker_completed','job_completed'} or runtime.get('status')!='completed'
            or delivery.get('phase') not in {'published','limited'}
            or delivery.get('review_status') not in {'verified','insufficient_evidence'}
            or delivery.get('runtime_completed') is not True or not delivery.get('sources')
            or delivery.get('goal_sha256')!=goal_sha256
            or (delivery.get('username'),delivery.get('conv_id'))!=(runtime.get('username'),runtime.get('conv_id'))):return None
    payload=event.get('payload') or {};receipt=event.get('supervisor_receipt') or {}
    if payload.get('requires_user') or receipt.get('scientific_review',{}).get('passed') is False:return None
    for binding in delivery.get('native_bindings') or []:
        node=runtime.get('nodes',{}).get(binding.get('step_id'),{})
        if (node.get('status')!='succeeded' or not node.get('node_verification')
                or not binding.get('token') or binding.get('token')!=node.get('token')
                or not binding.get('attempt_id') or binding['attempt_id']!=node.get('attempt_id')
                or sorted(map(str,binding.get('job_ids',[])))!=sorted(map(str,node.get('job_ids',[])))
                or binding.get('dossier_call_id')!=node.get('validation_dossier',{}).get('evidence_call_id')):continue
        if event['kind']=='worker_completed':
            if (payload.get('step_id')!=binding['step_id'] or payload.get('correlation_token')!=binding['token']
                    or payload.get('attempt_id')!=binding['attempt_id']):continue
        else:
            job=payload.get('job') or {}
            if (str(job.get('job_id')) not in set(map(str,binding['job_ids']))
                    or job.get('attempt_id')!=binding['attempt_id'] or job.get('terminal') is not True
                    or job.get('failed') is not False or (job.get('username'),job.get('conv_id'))!=(runtime['username'],runtime['conv_id'])):continue
        return binding
    return None
