"""Bounded read-only frontend DTOs and coalesced projection construction.

No value here grants execution or modifies durable scientific state. Full native
facts and history remain in the evidence files referenced by this view.
"""
import asyncio
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import json
import time
from dataclasses import dataclass


CONTRACT_FIELDS = ('step_id','agent','tool','arguments','depends_on','expected_outputs',
                   'resources','resource_locks','input_bindings','description','label',
                   'report_role','report_parent','tasks','sub_tasks')
NODE_FIELDS = ('step_id','agent','tool','arguments','depends_on','expected_outputs','status',
    'done','phase','dispatch_phase','description','label','job_ids','output_files','result_ref',
    'evidence_refs','error','started_at','updated_at','finished_at','elapsed_seconds',
    'execution_plan_version','contract_matches_execution','state_authority','tasks','sub_tasks',
    'execution_acceptance_status','execution_acceptance_advisory','science_contract_status',
    'resolved_arguments','runtime_input_bindings','runtime_binding_fingerprint','node_verification')
# The approved and runtime forms must compare the same complete contract.
NODE_FIELDS = tuple(dict.fromkeys(CONTRACT_FIELDS + NODE_FIELDS))
PATH_FIELDS = ('calculation_dir','result_dir','input_paths','output_files','expected_outputs',
               'checkpoint_path','evidence_path','updated_at')
GOAL_FIELDS = ('schema_version','original_goal','active_goal','version','gases','method','parameters',
    'execution_mode','execution_authorized','requires_plan_approval','approved_plan_version','pending_plan_version')


def _pick(value, keys):
    return {k:value[k] for k in keys if k in value} if isinstance(value,dict) else {}


def thin_contract(value):
    return _pick(value, CONTRACT_FIELDS)


def thin_node(value):
    node=_pick(value,NODE_FIELDS)
    if isinstance(value,dict):
        if 'contract' in value:node['contract']=thin_contract(value['contract'])
        if 'execution_contract' in value:node['execution_contract']=thin_contract(value['execution_contract'])
        if 'path_manifest' in value:node['path_manifest']=_pick(value['path_manifest'],PATH_FIELDS)
    return node


def thin_line(line):
    result=_pick(line,('line_id','username','conv_id','plan_version','status','execution_plan_version',
                           'execution_status','user_paused','state_authority'))
    result['steps']=[thin_node(n) for n in line.get('steps',[]) if isinstance(n,dict)]
    return result


def thin_runtime(runtime):
    if not runtime:return {}
    result=_pick(runtime,('username','conv_id','status','plan_version','user_paused','pause_kind','pause_reason'))
    result['nodes']={k:thin_node(v) for k,v in runtime.get('nodes',{}).items() if isinstance(v,dict)}
    reports=[]
    for step_id,node in runtime.get('nodes',{}).items():
        contract=node.get('contract') or {}
        if contract.get('tool')!='generate_scientific_report':continue
        receipt=node.get('result') or {}
        if not isinstance(receipt,dict):continue
        content=receipt.get('report_preview')
        if isinstance(content,str) and content:
            reports.append({'step_id':step_id,'role':contract.get('report_role') or contract.get('arguments',{}).get('report_mode','report'),
                'title':contract.get('arguments',{}).get('title',step_id),'status':node.get('status'),
                'report_file':receipt.get('report_file',''),'content':content[:24000]})
    # Existing native summaries may already provide presentation previews.
    if not reports:
        for report in runtime.get('live_reports',[])[:24]:
            if not isinstance(report,dict):continue
            preview=_pick(report,('step_id','role','parent','title','status','report_file','updated_at'))
            if isinstance(report.get('content'),str):preview['content']=report['content'][:24000]
            reports.append(preview)
    result['live_reports']=reports[:24]
    return result


def thin_patch(patch):
    if not isinstance(patch,dict):return None
    result=_pick(patch,('base_version','new_version','reason','affected_nodes','requires_user_confirmation'))
    result['changes']=[{**_pick(c,('operation','step_id','reason')),**({'node':thin_contract(c['node'])} if isinstance(c.get('node'),dict) else {})}
                       for c in patch.get('changes',[]) if isinstance(c,dict)]
    return result


def thin_workflow(raw):
    result=_pick(raw,('scope','snapshot_at','awaiting_plan_approval','current_activity','memory_files'))
    goal=raw.get('goal_contract') or {}
    result['goal_contract']=_pick(goal,GOAL_FIELDS)
    result['goal_contract']['approved_nodes']=[thin_contract(n) for n in goal.get('approved_nodes',[])]
    result['parallel_runtime']=thin_runtime(raw.get('parallel_runtime') or {})
    result['lines']=[thin_line(line) for line in raw.get('lines',[]) if isinstance(line,dict)]
    graph=raw.get('graph_projection') or {}
    result['graph_projection']=_pick(graph,('source','status','version','revision','plan_version','executable','retained'))
    result['graph_projection']['nodes']=[thin_node(n) for n in graph.get('nodes',[]) if isinstance(n,dict)]
    lifecycle=raw.get('lifecycle') or {}
    result['lifecycle']={'agents':{k:_pick(v,('agent','status','read_only','last_seen','last_active_at','service_heartbeat','last_action','busy_event_id'))
        for k,v in (lifecycle.get('agents') or {}).items()},
        'recent_events':[_pick(e,('event_id','kind','created_at','main_chat','supervisor'))
                         for e in lifecycle.get('recent_events',[])[-8:]]}
    result['recent_operations']=[_pick(op,('call_id','time','agent','tool','failed','evidence_path','evidence_refs'))
                                  for op in raw.get('recent_operations',[])[-16:]]
    result['error_branches']=[_pick(e,('branch_id','failed_step','step_id','node_id','status','error','diagnosis','resolution','evidence_refs'))
                               for e in raw.get('error_branches',[])[-24:]]
    recovery=raw.get('recovery') or {}
    result['recovery']={k:_pick(v,('step_id','node_id','task_id','tool','status','attempt','attempts','max_attempts','last_error','evidence_refs'))
                        for k,v in recovery.items() if isinstance(v,dict)}
    result['pending_workflow_patch']=thin_patch(raw.get('pending_workflow_patch'))
    return result


@dataclass(frozen=True)
class ProjectionSignature:
    # Only server-created UI DTO keys opt into bounded data staleness. Native
    # authorization/claims never consume this type or the cached payload.
    control: tuple
    data: tuple
    progress: tuple


def _same_control(left, right):
    if isinstance(left, ProjectionSignature) and isinstance(right, ProjectionSignature):
        return left.control == right.control
    return left == right


def _same_cache_identity(left, right):
    if isinstance(left, ProjectionSignature) and isinstance(right, ProjectionSignature):
        return left.control == right.control and left.data == right.data
    return left == right


class _StaleProjection(Exception):
    pass


class ProjectionCache:
    """At most one real builder thread; waiting requests consume no I/O pool.

    Shield the shared producer, so cancellation of any HTTP waiter cannot release
    its permit or create a second producer. Cache bytes only; callers decode their
    own independent DTO. Failed/control-stale/oversized builds never enter the cache.
    Only typed advisory DTOs may tolerate bounded data/progress churn.
    """
    def __init__(self, max_bytes=16*1024*1024, max_item_bytes=1024*1024, max_entries=128, advisory_max_age=2.0, on_ready=None, identity_max_age=30.0):
        self.max_bytes=max_bytes;self.max_item_bytes=max_item_bytes;self.max_entries=max_entries
        self.advisory_max_age=advisory_max_age
        self.identity_max_age=identity_max_age
        self._on_ready=on_ready
        self._cache=OrderedDict();self.cached_bytes=0
        self._executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='workflow-projection')
        self._loop=None;self._flights={};self._semaphore=None

    def _loop_state(self):
        loop=asyncio.get_running_loop()
        if loop is not self._loop:
            self._loop=loop;self._flights={};self._semaphore=asyncio.Semaphore(1)
        return loop

    async def _produce(self, scope, signature, key_fn, builder, semaphore):
        async with semaphore:
            current=key_fn()
            if isinstance(current,ProjectionSignature) and tuple(current.control[:2])!=tuple(scope):
                raise PermissionError('projection owner/session changed during construction')
            if not _same_control(signature,current):raise _StaleProjection()
            if isinstance(signature,ProjectionSignature):signature=current
            def encode():
                value=builder()
                if isinstance(signature,ProjectionSignature):
                    if not isinstance(value,dict):raise ValueError('UI projection must be an object')
                    value['_projection_view']={'built_at':time.time(),
                        'snapshot_revision':signature.progress[0], 'advisory':True}
                return json.dumps(value,ensure_ascii=False,allow_nan=False,separators=(',',':')).encode('utf-8')
            future=asyncio.get_running_loop().run_in_executor(self._executor,encode)
            try:
                payload=await asyncio.shield(future)
            except asyncio.CancelledError:
                # Even loop shutdown/repeated cancellation must not release the
                # producer permit while an uncancellable native reader is alive.
                while not future.done():
                    try:await asyncio.shield(future)
                    except asyncio.CancelledError:continue
                    except BaseException:break
                raise
            if not _same_control(signature,key_fn()):raise _StaleProjection()
            if len(payload)<=min(self.max_item_bytes,self.max_bytes):
                previous=self._cache.pop(scope,None)
                if previous:self.cached_bytes-=len(previous[1])
                self._cache[scope]=(signature,payload);self.cached_bytes+=len(payload)
                while self.cached_bytes>self.max_bytes or len(self._cache)>self.max_entries:
                    _,(_,evicted)=self._cache.popitem(last=False);self.cached_bytes-=len(evicted)
                if self._on_ready is not None:
                    # The builder thread never publishes. Queue on the event
                    # loop after this complete payload is cached; a stale-view
                    # caller finishes its no-await SSE revision assignment
                    # before a later background producer can reach this point.
                    asyncio.get_running_loop().call_soon(
                        self._notify_ready,scope,signature,payload,key_fn)
            return signature,payload

    def _flight(self, scope, signature, key_fn, builder):
        task=self._flights.get(scope)
        if task is not None and task.done():
            # A queued done callback cannot keep an aged advisory payload as
            # the selected flight; retire it before choosing a fresh producer.
            self._flights.pop(scope,None);task=None
        if task is None:
            task=asyncio.create_task(self._produce(scope,signature,key_fn,builder,self._semaphore))
            self._flights[scope]=task
            def finished(done, owner=scope):
                if self._flights.get(owner) is done:self._flights.pop(owner,None)
                if not done.cancelled():done.exception()
            task.add_done_callback(finished)
        return task

    def _view(self, payload, signature, current, *, refreshing=False):
        value=json.loads(payload)
        if isinstance(signature,ProjectionSignature):
            meta=value.setdefault('_projection_view',{})
            meta['age_seconds']=max(0.,time.time()-meta.get('built_at',time.time()))
            meta['observed_revision']=current.progress[0]
            meta['data_refresh_pending']=refreshing or signature.data!=current.data
        return value

    async def get(self, scope, key_fn, builder):
        self._loop_state()
        while True:
            signature=key_fn()
            if isinstance(signature,ProjectionSignature) and tuple(signature.control[:2])!=tuple(scope):
                raise PermissionError('projection key belongs to another owner/session')
            cached=self._cache.get(scope)
            if cached and _same_cache_identity(cached[0],signature):
                metadata=json.loads(cached[1]).get('_projection_view',{})
                fresh=(not isinstance(signature,ProjectionSignature) or
                       time.time()-metadata.get('built_at',0)<=self.identity_max_age)
                if fresh:
                    self._cache.move_to_end(scope)
                    return self._view(cached[1],cached[0],signature)
            if cached and isinstance(signature,ProjectionSignature) and _same_control(cached[0],signature):
                # Only an advisory DTO may be returned during data churn, and
                # only for a bounded age. Real controls never use this lane.
                metadata=json.loads(cached[1]).get('_projection_view',{})
                age=time.time()-metadata.get('built_at',0)
                if age<=self.advisory_max_age:
                    self._flight(scope,signature,key_fn,builder)
                    return self._view(cached[1],cached[0],signature,refreshing=True)
            if cached:
                self.cached_bytes-=len(cached[1]);del self._cache[scope]
            task=self._flight(scope,signature,key_fn,builder)
            try:produced_signature,payload=await asyncio.shield(task)
            except _StaleProjection:
                if self._flights.get(scope) is task:self._flights.pop(scope,None)
                await asyncio.sleep(0)
                continue
            current=key_fn()
            if not _same_control(produced_signature,current):
                if self._flights.get(scope) is task:self._flights.pop(scope,None)
                await asyncio.sleep(0)
                continue
            if (isinstance(produced_signature,ProjectionSignature)
                    and produced_signature.data!=current.data):
                # A native data change may have landed during a cold builder.
                # Return its complete bounded snapshot now and start a fresh
                # background producer whose ready event cannot be consumed by
                # this old DTO's same-loop SSE revision assignment.
                self._flight(scope,current,key_fn,builder)
            return self._view(payload,produced_signature,current)

    def close(self):
        self._executor.shutdown(wait=True)

    def _notify_ready(self, scope, signature, payload, key_fn):
        cached=self._cache.get(scope)
        if cached is None or cached[0] is not signature or cached[1] is not payload:return
        try:
            current=key_fn()
            if not _same_control(signature,current):return
            if isinstance(current,ProjectionSignature) and tuple(current.control[:2])!=tuple(scope):return
            self._on_ready(scope)
        except Exception:
            # Observer transport cannot invalidate an existing complete DTO or
            # grant/deny scientific work. The next request can still read it.
            return


def projection_signature(conv, revision):
    """Cheap revision/control/stat key; never parse persisted history for a hit."""
    import hashlib
    from pathlib import Path
    from .config import get_config
    from .workspace import session_root
    from . import task_line
    session=getattr(conv,'session',None)
    goal=getattr(session,'goal_contract',None)
    root=session_root(conv.conv_id,conv.username)
    project_root=Path(getattr(getattr(session,'config',None),'project_root',get_config().project_root))
    files=[root/name for name in ('orchestration_current.json','orchestration_chain.json',
        'session_checkpoint.json','supervisor_checkpoint.json','lifecycle.json','recovery_state.json')]
    files.extend((Path(task_line.LINE_FILE),project_root/'data/state/parallel_workflows.json'))
    for obj in (getattr(session,'_lifecycle_store',None),getattr(session,'recovery_gate',None)):
        if getattr(obj,'path',None):files.append(Path(obj.path))
    lifecycle_path=getattr(getattr(session,'_lifecycle_store',None),'path',None)
    if lifecycle_path:files.append(Path(str(lifecycle_path)+'.heartbeat.json'))
    signatures=[]
    for path in dict.fromkeys(files):
        try:
            stat=path.stat();signatures.append((str(path),stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns))
        except FileNotFoundError:signatures.append((str(path),None))
    control={key:getattr(goal,key,None) for key in GOAL_FIELDS}
    control['approved_nodes']=[thin_contract(n) for n in getattr(goal,'approved_nodes',[])]
    control.update(pending=_pick(getattr(session,'_pending_user_interaction',None),
        ('tool','request_id','question','prompt','status','parameters','choices')),
        pending_identity=id(getattr(session,'_pending_user_interaction',None)),
        patch=thin_patch(getattr(session,'_pending_workflow_patch',None)))
    digest=hashlib.sha256(json.dumps(control,sort_keys=True,ensure_ascii=False,default=str).encode()).digest()
    controls=(conv.username,conv.conv_id,str(project_root),id(session),id(goal),digest,
        getattr(session,'_current_line_id',''),getattr(session,'_api_turn_id',None),
        getattr(session,'_awaiting_plan_approval',False),getattr(session,'_waiting_for_user_input',False),
        getattr(conv,'current_uid',None),getattr(conv,'is_processing',False),
        getattr(conv,'interrupt_requested',False),getattr(conv,'interrupt_message',''))
    # Broker revisions and streamed agent/status text are live header signals,
    # not scientific/control revisions that invalidate an expensive DAG DTO.
    return ProjectionSignature(controls,tuple(signatures),
        (revision,getattr(conv,'current_agent',None),getattr(conv,'current_status','')))
