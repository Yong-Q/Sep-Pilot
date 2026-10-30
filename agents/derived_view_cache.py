"""Short-lived advisory cache; never used for completion or authorization."""
import copy,hashlib,json,os,stat,threading,time
from pathlib import Path
from .recovery import result_object

_lock=threading.Lock();_cache={};_TTL=30.0;_MAX=128
_FIELDS=('contract','status','phase','token','tool_returned','job_ids','execution_plan_version',
         'origin_goal_version','effective_arguments','analysis_input_proofs','result','result_ref')

def _key(state,step_id,goal):
    from .config import get_config
    node=(state.get('nodes') or {}).get(step_id) or {};proofs=node.get('analysis_input_proofs') or []
    if not proofs or not isinstance(goal,dict):raise ValueError('missing view proof')
    owner=[state.get('username'),state.get('conv_id')]
    if any(not isinstance(v,str) or not v or Path(v).name!=v or v in {'.','..'} for v in owner):raise ValueError('invalid view owner')
    root=Path(get_config().project_root).absolute()/'runs'/owner[0]/owner[1]
    producers={p.get('source_step') for p in proofs};nodes=state.get('nodes') or {}
    selected={key:{k:copy.deepcopy(nodes.get(key,{}).get(k)) for k in _FIELDS} for key in producers|{step_id}}
    events=[e for e in (state.get('outbox') or {}).values() if e.get('kind')=='worker_completed'
            and (e.get('payload') or {}).get('step_id')==step_id]
    metadata={'owner':owner,'plan':state.get('plan_version'),
              'goal':goal,'stored_goal':state.get('goal_contract'),'nodes':selected,'events':events}
    files=set()
    def add_ref(ref):
        for key in ('evidence_path','checkpoint_path'):
            if isinstance(ref.get(key),str):files.add(ref[key])
    add_ref(node.get('result_ref') or {})
    for proof in proofs:
        for item in proof.get('original_inputs') or []:files.add(item['path'])
        producer=nodes.get(proof.get('source_step')) or {};add_ref(producer.get('result_ref') or {})
        result=result_object(producer.get('result'))
        for item in result.get('artifacts') or []:files.add(item['path'])
        if (result.get('publication') or {}).get('protocol')=='nfs-link-commit-v1':
            files.add(str(Path(producer['contract']['arguments']['output_dir'])/'artifacts'/'.publication.json'))
    if not files:raise ValueError('empty view proof files')
    stats=[]
    for path in sorted(files):
        candidate=Path(path)
        if not candidate.is_absolute() or not candidate.is_relative_to(root) or candidate.resolve(strict=True)!=candidate:
            raise PermissionError('view evidence escapes its owned session')
        s=os.stat(path,follow_symlinks=False)
        if not stat.S_ISREG(s.st_mode):raise ValueError('view proof path is not a regular file')
        stats.append((path,s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode))
    raw=json.dumps([metadata,stats],ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode()
    return hashlib.sha256(raw).hexdigest()

def view_proven_derived_arguments(state,step_id,current_goal):
    from .derived_completion_proof import proven_derived_arguments
    key=_key(state,step_id,current_goal);now=time.monotonic()
    with _lock:
        item=_cache.get(key)
        if item and now-item[0]<_TTL:return copy.deepcopy(item[1])
    value=proven_derived_arguments(state,step_id,current_goal)
    # A change during the proof means the result must not be cached.
    if _key(state,step_id,current_goal)!=key:raise ValueError('view evidence changed during verification')
    with _lock:
        if len(_cache)>=_MAX:_cache.pop(next(iter(_cache)))
        _cache[key]=(now,copy.deepcopy(value))
    return value
