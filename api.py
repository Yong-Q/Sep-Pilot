"""
BiMemAgent API - Multi-user, Multi-conversation Backend
"""
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field, StrictBool
from typing import Optional, List, Dict, Any
import json
import copy
import asyncio
import concurrent.futures
import os
import hashlib
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

from auth import (
    session_manager, register_user, login_user,
    load_users, registered_user_count, USER_DB, CONVERSATION_LOG, public_model_connection,
    save_model_connection, config_for_user, migrate_auth_storage, load_tokens,
)

app = FastAPI(title="BiMemAgent API", description="多用户会话隔离、持久化混合串并行编排与全生命周期协调", version="3.4.77")

# Durable lifecycle mailboxes survive the API process.  That is necessary for
# long scheduler jobs, but a pre-restart notification from a conversation with
# no executable DAG must not manufacture a brand-new model turn after startup.
# Events created after this boundary are genuinely new.  Older events are only
# recoverable when the authoritative workflow runtime still has unfinished
# nodes; the runtime, not prose/status strings, is the source of truth.
_PROCESS_STARTED_AT = time.time()

_allowed_origins = [item.strip() for item in os.environ.get(
    "BIMEM_ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:3001,http://localhost:5001"
).split(",") if item.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.middleware("http")
async def _security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(), geolocation=()"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data: blob:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
    )
    if request.url.path.startswith("/api/"):
        is_event_stream = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower() == "text/event-stream"
        response.headers["Cache-Control"] = (
            "no-cache, no-store, no-transform" if is_event_stream else "no-store")
    return response

# ── Concurrency control ──
# The gate lives at the individual provider-call boundary, not around a whole
# agent turn.  This lets independent sessions interleave tools/model calls.
from agents.model_gate import MODEL_CALL_CONCURRENCY, provider_state
MAX_THREAD_WORKERS = 8
_conv_locks: Dict[str, Any] = {}
_conv_locks_lock = threading.Lock()
_conversation_log_lock = threading.Lock()


class _ScopedEventBroker:
    """Thread-safe revision signal; payloads remain session-scoped at read time."""
    def __init__(self):
        self._condition = threading.Condition()
        self._revisions: Dict[tuple, int] = {}

    def publish(self, username: str, conv_id: str) -> int:
        key = (username, conv_id)
        with self._condition:
            revision = self._revisions.get(key, 0) + 1
            self._revisions[key] = revision
            self._condition.notify_all()
            return revision

    def revision(self, username: str, conv_id: str) -> int:
        with self._condition:
            return self._revisions.get((username, conv_id), 0)

    def wait(self, username: str, conv_id: str, previous: int, timeout: float = 20.0) -> int:
        key = (username, conv_id)
        with self._condition:
            self._condition.wait_for(lambda: self._revisions.get(key, 0) != previous, timeout=timeout)
            return self._revisions.get(key, 0)


_workflow_events = _ScopedEventBroker()

def _get_conv_lock(username: str, conv_id: str):
    """Per-conversation lock. NOTE: intentionally keyed by (username, conv_id),
    NOT username alone — different conversations of the same user must be able
    to run concurrently (e.g. the harness conversation drives a smoke-test
    conversation via the API while both belong to one user). A per-user lock
    serialized every conversation of the user, so a long harness turn (agent
    loop with run_bash polling) starved the test conversation for 180s and
    returned an empty "系统繁忙" answer — the smoke-test step1 symptom."""
    key = (username, conv_id)
    with _conv_locks_lock:
        if key not in _conv_locks:
            from agents.state_io import ConversationLock
            from agents.config import get_config
            filename = hashlib.sha256(json.dumps(key).encode()).hexdigest() + '.lock'
            _conv_locks[key] = ConversationLock(get_config().project_root / 'data' / 'state' / 'conversation_locks' / filename)
        return _conv_locks[key]


def _append_conversation_log(entries: List[Dict[str, Any]]) -> None:
    """Append complete JSONL records under a thread/process lock."""
    if not entries:
        return
    with _conversation_log_lock:
        with open(CONVERSATION_LOG, "a", encoding="utf-8") as f:
            os.chmod(CONVERSATION_LOG, 0o600)
            try:
                import fcntl
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            except Exception:
                pass
            try:
                for entry in entries:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
            finally:
                try:
                    import fcntl
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass


# ── Auth helpers ──
def _get_user(authorization: Optional[str] = Header(None)) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="请先登录")
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="登录凭据格式无效")
    token = parts[1]
    username = session_manager.get_username(token)
    if not username:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return username


def _get_admin(username: str = Depends(_get_user)):
    from auth import is_admin
    if not is_admin(username): raise HTTPException(403, 'administrator scope required')
    return username


def _get_conv(username: str, conv_id: str = ""):
    """Get the active conversation for a user."""
    user = session_manager.get_user(username)
    if conv_id:
        if conv_id not in user.conversations: raise HTTPException(404, 'conversation not found in this user scope')
        conv = user.conversations[conv_id]
        user.current_conv_id = conv_id
        return conv
    return user.get_current_conversation()


def _execution_status(session):
    pending = session._pending_user_interaction or {}
    if pending.get('tool') == 'lifecycle_wait': return 'Waiting for jobs'
    if pending or session._awaiting_plan_approval or session._pending_param_question: return '等待用户确认'
    runtime = session._runtime_snapshot() if session._runtime_snapshot else {}
    if runtime.get('status') == 'needs_user': return '等待用户协商编排'
    if runtime.get('status') == 'active':
        return '编排已暂停（主chat在岗）' if runtime.get('user_paused') else '编排分支运行中（主chat在岗）'
    return 'Completed' if session.task_complete else '任务未完成（主chat在岗）'


def _legacy_graph_projection(lines: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Recover a displayable DAG from persisted pre-DAG TaskLine records.

    This is deliberately a read model, not an execution approval.  It exposes
    work that was already persisted before executable DAG contracts became
    mandatory, while leaving ``approved_nodes`` and authorization untouched.
    Dependencies are taken from recorded edges first, then from shared job IDs
    and producer/consumer paths.  Diagnostic readers may depend on the latest
    preceding compute node, but unrelated compute nodes remain parallel roots.
    """
    raw = []
    for line in lines:
        for index, step in enumerate(line.get("steps", [])):
            if not step.get("step_id") or not step.get("tool") or step.get("status") == "superseded":
                continue
            node = copy.deepcopy(step)
            node["_legacy_order"] = len(raw)
            node["line_id"] = line.get("line_id", "")
            raw.append(node)
    if not raw:
        return {"source": "none", "nodes": []}

    ids = {node["step_id"] for node in raw}
    prior_job_owner: Dict[str, str] = {}
    latest_compute = ""
    diagnostic_tools = {
        "inspect_path", "inspect_run", "grep_search", "read_file",
        "task_line_query", "lifecycle_state",
    }

    def path_values(node):
        values = []
        for key in ("path", "input_dir", "output_dir", "work_dir", "job_work_dir", "cif", "output"):
            value = node.get("arguments", {}).get(key) or node.get(key)
            if isinstance(value, str) and value.strip():
                values.append(value.rstrip("/"))
        values.extend(str(value).rstrip("/") for value in node.get("output_files", []) if value)
        return values

    prior_nodes = []
    for node in raw:
        deps = [dep for dep in node.get("depends_on", []) if dep in ids and dep != node["step_id"]]
        jobs = [str(job) for job in node.get("job_ids", []) if str(job)]
        for job in jobs:
            owner = prior_job_owner.get(job)
            if owner and owner not in deps:
                deps.append(owner)

        # A persisted path inspection/read following a producer is a real
        # observation edge. Match exact paths and parent/child paths first.
        if not deps and node.get("tool") in diagnostic_tools:
            current_paths = path_values(node)
            for previous in reversed(prior_nodes):
                previous_paths = path_values(previous)
                if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
                       or a.endswith("/" + b) or b.endswith("/" + a)
                       for a in current_paths for b in previous_paths):
                    deps.append(previous["step_id"])
                    break
            if not deps and node.get("tool") != "inspect_path" and latest_compute:
                deps.append(latest_compute)

        if not deps and node.get("tool") not in diagnostic_tools:
            current_paths = path_values(node)
            for previous in reversed(prior_nodes):
                if previous.get("tool") not in diagnostic_tools:
                    continue
                previous_paths = path_values(previous)
                if any(a == b or a.startswith(b + "/") or b.startswith(a + "/")
                       or a.endswith("/" + b) or b.endswith("/" + a)
                       for a in current_paths for b in previous_paths):
                    deps.append(previous["step_id"])
                    break
        node["depends_on"] = deps
        node["legacy_recovered"] = True
        node.pop("_legacy_order", None)
        for job in jobs:
            prior_job_owner.setdefault(job, node["step_id"])
        if node.get("tool") not in diagnostic_tools:
            latest_compute = node["step_id"]
        prior_nodes.append(node)
    return {"source": "persisted_legacy_chain", "nodes": raw}


_api_durable_writes = set()


async def _finish_pending_io(pending):
    # Repeated cancellation must not cancel the to_thread Task itself: its
    # underlying writer cannot be interrupted and still owns the file lease.
    while not pending.done():
        try:
            await asyncio.shield(pending)
        except asyncio.CancelledError:
            if pending.cancelled():
                raise
    return pending.result()


async def _durable_io(function, *args, **kwargs):
    pending = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    _api_durable_writes.add(pending)
    pending.add_done_callback(_api_durable_writes.discard)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError as cancellation:
        try:
            await _finish_pending_io(pending)
        except BaseException as write_error:
            raise cancellation from write_error
        raise


def _owned_conversation_io(conv, function, *args, **kwargs):
    """Keep scope explicit when a default executor thread is reused."""
    from agents.watch_context import get_context, set_context, clear_context
    previous = get_context()
    try:
        set_context(conv.username, conv.conv_id, 'api_state',
                    getattr(conv.session, '_current_line_id', '') if conv.session else '')
        return function(*args, **kwargs)
    finally:
        if previous:
            set_context(**previous)
        else:
            clear_context()


from agents.workflow_projection import (ProjectionCache, projection_signature,
    thin_workflow, thin_runtime, thin_line)
_workflow_projection_cache = ProjectionCache(
    on_ready=lambda scope: _workflow_events.publish(scope[0], scope[1]))


_api_session_hydrations = {}


def _bound_api_session(conv):
    """An existing native binding is sufficient for an advisory GET, not control."""
    session=conv.session
    binding=getattr(conv,'_api_session_binding',None)
    if (session is not None and isinstance(binding,tuple) and len(binding)==5
            and binding[0] is session and binding[1:] == (conv.username,conv.conv_id,
                str(getattr(getattr(session,'config',None),'project_root','')),
                getattr(session,'_current_line_id',''))):
        return session
    return None


def _session_admission_source(conv):
    return (conv.session,getattr(conv,'current_uid',None),conv.username,conv.conv_id,
        bool(getattr(conv,'interrupt_requested',False)),str(getattr(conv,'interrupt_message','')),
        getattr(conv,'_workflow_user_control_epoch',0))


async def _prepare_api_session(conv, source):
    # Only a genuinely cold actor is prepared off-loop. Existing unbound
    # actors retain the complete legacy binding checks on their native path.
    actor=await _durable_io(_owned_conversation_io,conv,_ensure_session,conv,publish=False,_expected_source=source)
    if _session_admission_source(conv)!=source:
        raise ValueError('cold session preparation was superseded; no actor published')
    if actor is None:return None
    ready=getattr(actor,'_api_ready_binding',None)
    if (not isinstance(ready,tuple)or ready!=(actor,source[2],source[3],
            str(actor.config.project_root),actor._current_line_id)):
        raise ValueError('prepared session lacks its native scoped binding')
    # The event loop alone publishes the prepared actor after an identity CAS.
    conv.session=actor
    conv._api_session_binding=ready
    published_source=(actor,*source[1:])
    def after_bind():
        if _session_admission_source(conv)!=published_source:
            raise ValueError('prepared session control changed before after_bind')
        actor._api_after_bind()
        if _session_admission_source(conv)!=published_source:
            raise ValueError('prepared session control changed during after_bind')
    try:
        await _durable_io(_owned_conversation_io,conv,after_bind)
        if conv.session is not actor or _bound_api_session(conv)is not actor:
            raise ValueError('prepared session binding was replaced during after_bind')
    except BaseException:
        # A partially completed after_bind must never qualify for a warm GET.
        if getattr(conv,'_api_session_binding',None) is ready:
            conv._api_session_binding=None
        raise
    return actor


async def _ensure_api_session(conv, *, read_only=False):
    # Joining precedes the bound fast path: publication happens before native
    # after_bind I/O finishes, so that actor is not ready for another consumer.
    scope=(conv.username,conv.conv_id)
    key=(asyncio.get_running_loop(),id(conv))
    pending=_api_session_hydrations.get(key)
    if pending is None:
        bound=_bound_api_session(conv)
        if read_only and bound is not None:
            return bound
        if conv.session is not None:
            # Controls retain complete native recovery and approval checks.
            return _ensure_session(conv)
    if pending is None:
        pending=asyncio.create_task(_prepare_api_session(conv,_session_admission_source(conv)))
        _api_session_hydrations[key]=pending
        def retired(task):
            if _api_session_hydrations.get(key)is task:_api_session_hydrations.pop(key,None)
        pending.add_done_callback(retired)
    try:
        actor=await asyncio.shield(pending)
    except asyncio.CancelledError as cancellation:
        # Hydration may still own native file operations. A cancelled waiter
        # cannot release a caller's lease before the shared producer finishes.
        try:await _finish_pending_io(pending)
        except BaseException as error:raise cancellation from error
        raise
    if (conv.username,conv.conv_id)!=scope or actor is not conv.session:
        raise ValueError('session owner or actor changed after shared hydration')
    return actor


async def _set_owned_user_pause(conv, actor, paused, control_valid):
    """Fresh native pause writes use their revision CAS, never a UI snapshot."""
    if not control_valid():raise ValueError('user control was superseded before native pause')
    if actor is None or not actor._runtime_snapshot:return
    snapshot=await _durable_io(_owned_conversation_io,conv,actor._runtime_snapshot)
    if not control_valid():raise ValueError('user control was superseded during native pause inspection')
    if not snapshot:return
    if (snapshot.get('username'),snapshot.get('conv_id'))!=(conv.username,conv.conv_id):
        raise ValueError('native pause snapshot belongs to another owner')
    from agents.parallel_workflow import WorkflowStore
    store=WorkflowStore(actor.config.project_root/'data/state/parallel_workflows.json')
    identity=store.identity(conv.username,conv.conv_id)
    def commit():
        if not control_valid():raise ValueError('user control was superseded before pause commit')
        if not store.user_pause(identity,paused,expected_revision=snapshot.get('user_pause_revision',0)):
            raise ValueError('native pause revision changed before commit')
    await _durable_io(_owned_conversation_io,conv,commit)


async def _workflow_state_async(conv):
    # Warm GETs read an existing scoped actor without replaying control
    # recovery. Cold preparation is coalesced off-loop and CAS-published here.
    try:
        await _ensure_api_session(conv,read_only=True)
    except ValueError as error:
        raise HTTPException(status_code=409,detail=str(error)) from error
    def signature():
        return projection_signature(conv, _workflow_events.revision(conv.username, conv.conv_id))
    def build():
        return thin_workflow(_owned_conversation_io(conv, _workflow_state, conv, ensure_session=False))
    result = await _workflow_projection_cache.get((conv.username, conv.conv_id), signature, build)
    # Overlay current stream headers without rebuilding historical state. This
    # is presentation only; all control endpoints retain their native CAS.
    pending = getattr(conv.session, '_pending_user_interaction', None) or {}
    result['current_activity'] = {'agent': getattr(conv, 'current_agent', '') or 'lead-orchestrator',
        'status': getattr(conv, 'current_status', '') or 'Agents ready', 'is_processing': bool(getattr(conv, 'is_processing', False)),
        'waiting_for': pending.get('tool', '') if isinstance(pending, dict) else ''}
    return result


def _projection_lifecycle(session):
    # _lifecycle_state is a model/tool contract and rehydrates runtime+chain.
    # The UI needs role metadata and eight event headers, never event payloads.
    store = getattr(session, '_lifecycle_store', None)
    if store and callable(getattr(store, 'view_summary', None)):
        return store.view_summary()
    state = store.snapshot() if store else {}
    events = state.get('events', {})
    from collections import deque
    recent = deque(events.values(), maxlen=8)
    return {'agents': state.get('agents', {}), 'recent_events': [
        {key: event.get(key) for key in ('event_id','kind','created_at','main_chat','supervisor')}
        for event in recent]}


def _workflow_state(conv, *, ensure_session=True) -> Dict[str, Any]:
    """Structured, inspectable orchestration state for API/frontend."""
    if ensure_session:
        _ensure_session(conv)
    goal = {}
    errors = []
    awaiting = False
    line_id = ""
    if conv.session:
        try:
            goal = conv.session.goal_contract.to_dict()
            errors = list(conv.session._error_branches.values())
            awaiting = bool(conv.session._awaiting_plan_approval)
            line_id = str(conv.session._current_line_id or conv.conv_id)
        except Exception:
            pass
    lines = []
    try:
        from agents.task_line import get_store
        store = get_store()
        if line_id:
            line = store.get_line(line_id, username=conv.username, conv_id=conv.conv_id)
            if line:
                lines = [line]
        if not lines:
            lines = store.get_by_conv(conv.conv_id, username=conv.username)
    except FileNotFoundError:
        pass
    from agents.workflow_view import authoritative_line
    runtime = conv.session._runtime_snapshot() if conv.session and conv.session._runtime_snapshot else {}
    if runtime and any(runtime.get(key) != value for key, value in
                       (('username', conv.username), ('conv_id', conv.conv_id))):
        raise PermissionError('runtime projection belongs to a different owner/session scope')
    # Strip native result/attempt payloads before authoritative_line's deepcopy.
    proof_runtime = runtime
    runtime = thin_runtime(runtime)
    lines = [authoritative_line(thin_line(line), runtime, goal, proof_runtime,advisory_cache=True) for line in lines]
    from agents.orchestration_chain import read_last_graph
    from agents.workspace import session_root
    graph_projection = read_last_graph(session_root(conv.conv_id, conv.username),
                                      {'username': conv.username, 'conv_id': conv.conv_id}, strict=True) or {"source": "none", "nodes": []}
    if not runtime.get("nodes") and not goal.get("approved_nodes"):
        if not graph_projection.get('nodes'):
            draft = conv.session.context.get('workflow_draft', {}) if conv.session else {}
            if draft.get('nodes'):
                graph_projection = {**draft, 'source': 'workflow_draft'}
            else:
                # Flat diagnostic calls are activity history, not research DAGs.
                structured = [{**line, 'steps': [n for n in line.get('steps', [])
                    if 'expected_outputs' in n]} for line in lines]
                graph_projection = _legacy_graph_projection(structured)
        # Planning progress is metadata, never a synthetic executable node.
        if (not graph_projection.get("nodes") and goal.get("original_goal")
                and goal.get("execution_mode") == "workflow"
                and goal.get("requires_plan_approval")):
            planning_status = ("planning" if conv.is_processing else
                               "paused" if getattr(conv, "interrupt_requested", False) else
                               "planning_queued")
            graph_projection = {
                "source": "workflow_planning_state",
                "executable": False,
                "status": planning_status,
                "nodes": [],
            }
    recent_operations = []
    if conv.session:
        for item in conv.session.memory.tool_call_log[-16:]:
            recent_operations.append({
                "call_id": item.get("call_id", ""),
                "time": item.get("time"),
                "agent": item.get("agent", ""),
                "tool": item.get("tool", ""),
                "failed": bool(item.get("failed", False)),
                "params": item.get("params", {}),
                "evidence_path": item.get("evidence_path", ""),
            })
    pending = conv.session._pending_user_interaction if conv.session else None
    return {
        # Frontend responses carry an explicit ownership envelope.  The API
        # already resolves ``conv`` inside the authenticated user's scope; the
        # envelope additionally lets the browser reject a late response from a
        # conversation that was switched away while its request was in flight.
        "scope": {"username": conv.username, "conv_id": conv.conv_id},
        "snapshot_at": time.time(),
        "goal_contract": goal,
        "awaiting_plan_approval": awaiting,
        "error_branches": errors,
        "lines": lines,
        "recovery": conv.session.recovery_gate.snapshot() if conv.session else {},
        "lifecycle": _projection_lifecycle(conv.session),
        "pending_workflow_patch": conv.session._pending_workflow_patch if conv.session else None,
        "parallel_runtime": runtime,
        "graph_projection": graph_projection,
        "current_activity": {
            "agent": conv.current_agent or "lead-orchestrator",
            "status": conv.current_status or "Agents ready",
            "is_processing": bool(conv.is_processing),
            "waiting_for": (pending or {}).get("tool", "") if isinstance(pending, dict) else "",
        },
        "recent_operations": recent_operations,
        "memory_files": {
            name: str(conv.session._evidence_root.parent / filename)
            for name, filename in {'task': 'task_manifest.json', 'session': 'session_checkpoint.json',
                                   'supervisor': 'supervisor_checkpoint.json', 'recovery': 'recovery_state.json',
                                   'lifecycle': 'lifecycle.json', 'chain': 'orchestration_chain.json'}.items()
        } if conv.session and conv.session._evidence_root else {},
    }


def _recover_bound_session_owner_scope(conv, session):
    """Recheck missed hydration recovery without mutating an unverified actor."""
    if (getattr(conv,'is_processing',False)
            or not (session._pending_workflow_patch or session._pending_user_interaction)):
        return False
    from copy import copy, deepcopy
    def source():
        return (conv.session,getattr(conv,'current_uid',None),conv.username,conv.conv_id,
            getattr(conv,'interrupt_requested',False),getattr(conv,'interrupt_message',''),
            getattr(conv,'is_processing',False),str(session.config.project_root),session._current_line_id,
            deepcopy(session.goal_contract.to_dict()),deepcopy(session.context),
            deepcopy(session._pending_workflow_patch),deepcopy(session._pending_user_interaction),
            session._waiting_for_user_input,session._awaiting_plan_approval,
            deepcopy(session.memory.tool_call_log[-12:]))
    before=source()
    shadow=copy(session)
    shadow.goal_contract=deepcopy(session.goal_contract)
    shadow.context=deepcopy(session.context)
    shadow._pending_workflow_patch=deepcopy(session._pending_workflow_patch)
    shadow._pending_user_interaction=deepcopy(session._pending_user_interaction)
    shadow.memory=copy(session.memory)
    shadow.memory.tool_call_log=deepcopy(session.memory.tool_call_log[-12:])
    receipt=shadow._recover_committed_initial_pending(wake_after_restore=True)
    if not receipt:
        from agents.preparation_defaults import pending_default_review
        receipt=pending_default_review(shadow)
        if receipt:shadow.context['preflight_recovery_ready']=receipt
    if not receipt:
        return False
    if source()!=before or conv.session is not session:
        raise ValueError('bound recovery source was superseded; no actor metadata applied')
    session.context=shadow.context
    session._pending_workflow_patch=shadow._pending_workflow_patch
    session._pending_user_interaction=shadow._pending_user_interaction
    session._waiting_for_user_input=shadow._waiting_for_user_input
    return True


def _ensure_session(conv, *, publish=True, _expected_source=None):
    """Lazily initialize and bind one scoped Session exactly once."""
    # Startup/job-notification paths are not necessarily inside an API actor
    # thread. Hydration invokes owned native recovery before tool execution;
    # establish its authenticated conversation scope here, then restore the
    # caller's previous thread-local context in the common native wrapper.
    if _expected_source is not None and _session_admission_source(conv)!=_expected_source:
        raise ValueError('session preparation source changed before hydration')
    from agents.watch_context import get_context
    current_owner=get_context()
    if (current_owner.get('username'),current_owner.get('conv_id')) != (conv.username,conv.conv_id):
        return _owned_conversation_io(conv,_ensure_session,conv,publish=publish,_expected_source=_expected_source)
    session = conv.session
    binding = getattr(conv, '_api_session_binding', None)
    if (session is not None and binding and binding[0] is session
            and binding[1:] == (conv.username, conv.conv_id,
                str(getattr(getattr(session, 'config', None), 'project_root', '')),
                getattr(session, '_current_line_id', ''))):
        if not _recover_bound_session_owner_scope(conv,session):
            return session
        # A proved missed recovery needs the ordinary binding/after_bind path
        # once; all normal cached reads still return above.
    from copy import deepcopy
    initial_session = session
    initial_uid = getattr(conv, 'current_uid', None)
    initial_scope = (conv.username, conv.conv_id)
    initial_interrupt = (getattr(conv, 'interrupt_requested', False),
                         getattr(conv, 'interrupt_message', ''))
    def check_hydration_control():
        if _expected_source is not None and _session_admission_source(conv)!=_expected_source:
            raise ValueError('session preparation source changed during hydration')
        if (conv.session is not initial_session
                or getattr(conv, 'current_uid', None) != initial_uid
                or (conv.username, conv.conv_id) != initial_scope
                or (getattr(conv, 'interrupt_requested', False),
                    getattr(conv, 'interrupt_message', '')) != initial_interrupt):
            raise ValueError('session hydration was superseded; no actor was published')
    check_hydration_control()
    from agents.workspace import session_root
    from agents.config import get_config
    root = session_root(conv.conv_id, getattr(conv, 'username', ''))
    if not root.resolve().is_relative_to(get_config().project_root.resolve()):
        raise ValueError('conversation checkpoint must stay in the project workspace')
    checkpoint = root / 'session_checkpoint.json'
    if session is None:
        try:
            from agents.defns import ORCHESTRATOR
            from agents.session import Session
            session = Session(config=config_for_user(conv.username))
            restored = False
            saved_state = deepcopy(getattr(conv, 'session_state', None) or {})
            history = deepcopy(conv.messages_history)
            if checkpoint.exists():
                latest = json.loads(checkpoint.read_text())
                if latest.get('checkpoint_at', 0) > saved_state.get('checkpoint_at', 0):
                    saved_state = latest
            check_hydration_control()
            if saved_state:
                if saved_state.get('owner') and saved_state['owner'] != {'username': conv.username, 'conv_id': conv.conv_id}:
                    raise PermissionError('checkpoint belongs to a different user/session')
                restored = session.import_state(saved_state)
            if not restored and history:
                restored = session.hydrate_from_transcript(history)
            elif restored and history:
                # Session checkpoint and durable display transcript are written
                # on different boundaries. Merge any user turns that arrived
                # after the checkpoint (interrupts/redirects/job notices).
                session.reconcile_transcript(history, after_count=saved_state.get('transcript_message_count'))
            check_hydration_control()
            if not session.current_agent:
                session.current_agent = ORCHESTRATOR
            session._on_progress = conv._on_progress
        except PermissionError:
            raise
        except Exception as e:
            print(f"Warning: Could not init session: {e}")
            check_hydration_control()
            return None
    if session:
        check_hydration_control()
        from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
        from agents.lifecycle import LifecycleStore
        session.recovery_gate.path = root / 'recovery_state.json'
        if not session._current_line_id:
            session._current_line_id = conv.conv_id
        from agents.task_line import get_store
        line = get_store().begin_line(session._current_line_id, username=conv.username, conv_id=conv.conv_id, title=conv.title)
        check_hydration_control()
        session._current_line_id = line['line_id']
        if session._lifecycle_store is None:
            session._lifecycle_store = LifecycleStore(root / 'lifecycle.json')
            if (root / 'lifecycle.json').exists():
                session._lifecycle_store.recover_claims()
        check_hydration_control()
        session._evidence_root = root / 'evidence'
        session._on_lifecycle_event = lambda kind, payload: session._lifecycle_store.emit(kind, payload)
        from agents.parallel_workflow import WorkflowStore, resources_for, conflicts, READ_ONLY
        runtime_store = WorkflowStore(session.config.project_root / 'data/state/parallel_workflows.json')
        runtime_id = runtime_store.identity(conv.username, conv.conv_id)
        native_snapshot = lambda: runtime_store.snapshot(runtime_id)
        session._runtime_snapshot = native_snapshot
        from agents.workflow_reply_guard import bind_workflow_reply_guard
        session._workflow_reply_check = bind_workflow_reply_guard(runtime_store,
            username=conv.username,conv_id=conv.conv_id,
            is_current=lambda session=session,snapshot=native_snapshot:conv.session is session and session._runtime_snapshot is snapshot)
        session._restore_workflow_approval(runtime_store.snapshot(runtime_id),
            {'username': conv.username, 'conv_id': conv.conv_id})
        check_hydration_control()
        session._chain_state_is_primary = True
        session._chain_jobs = lambda: [j for j in _watch.list()
            if (j.get('username'), j.get('conv_id')) == (conv.username, conv.conv_id)]
        session._on_workflow_start = lambda plan_version, **kw: _start_workflow_runtime(conv, plan_version, **kw)
        session._on_resource_review = lambda request: _review_resources(conv, request)
        session._on_workflow_message = lambda step_id, text, kind='comment', message_id=None: _ensure_workflow_runtime(conv).send(step_id, text, kind, message_id)
        session._on_workflow_resolve = lambda step_id, decision_id, verification_call_id: _ensure_workflow_runtime(conv).resolve_local_write(step_id, decision_id, verification_call_id)
        session._on_local_preparation_requeue = lambda step_id, expected_token, failed_call_id, verification_call_id, **authority: _ensure_workflow_runtime(conv).requeue_local_preparation(step_id, expected_token, failed_call_id, verification_call_id, **authority)
        session._on_cdft_partial_prepare = lambda step_id, timeout_seconds, walltime, **authority: _ensure_workflow_runtime(conv).prepare_cdft_partial_recovery(step_id, timeout_seconds, walltime, **authority)
        from agents.workflow_view import bind_result_handoff
        bind_result_handoff(session, lambda: _ensure_workflow_runtime(conv))
        def tool_guard(tool, arguments):
            if conv.session is not session:
                return 'conversation actor was replaced; no tool control is granted'
            from agents.control_policy import role_denial, bypasses_data_lease
            role = getattr(session.current_agent, 'name', '')
            denied = role_denial(role, tool)
            if denied: return denied
            if bypasses_data_lease(role, tool): return ''
            # Handoff changes the reasoning/inspection owner, not the data.
            # Mutating scientific tools remain guarded at their real dispatch.
            if tool.startswith('handoff_to_'): return ''
            if tool in READ_ONLY or tool in {'execute_workflow', 'message_workflow_node', 'propose_workflow_patch', 'apply_workflow_patch', 'discard_workflow_patch', 'revalidate_workflow_node_outputs','finish_workflow_node',
                    'request_user_decision', 'resolve_preparation_defaults', 'reconcile_watched_job', 'prepare_retry', 'accept_recovered_result',
                    'recovery_state', 'lifecycle_state', 'task_line_query', 'resolve_local_workflow_write', 'retarget_queued_job', 'requeue_local_preparation', 'prepare_cdft_partial_recovery'}:
                return ''
            state = runtime_store.snapshot(runtime_id)
            if state and state['status'] in {'active', 'needs_user'} and (tool.startswith('handoff_to_') or
                    any(n['contract']['tool'] == tool and n['contract']['arguments'] == arguments for n in state['nodes'].values())):
                if tool != 'handoff_to_patcher':
                    return 'runtime owns this compiled node; do not re-execute it through the main Session'
            resources = resources_for({'tool': tool, 'arguments': arguments}, session.config.project_root, root)
            if any(conflicts(a, b) for lease in runtime_store.snapshot().get('leases', {}).values()
                   for a in resources for b in lease['resources']):
                return 'tool mutation conflicts with a live/unknown workflow resource lease'
            return ''
        session._on_tool_guard = tool_guard
        def request_primary_projection(reason):
            if conv.session is not session or (conv.username,conv.conv_id)!=initial_scope:
                raise ValueError('display request came from a replaced primary Session')
            runtime=_workflow_runtimes.get((conv.username,conv.conv_id))
            if runtime is None:runtime=_ensure_workflow_runtime(conv)
            return runtime.sync_chain(reason)
        session._on_display_request=request_primary_projection
        def _save_checkpoint(state, reason):
            defer_projection=callable(getattr(session,'_on_display_request',None))
            if conv.session is not session or (conv.username,conv.conv_id)!=initial_scope:
                raise ValueError('primary Session changed before its checkpoint')
            state['display_projection']={'schema_version':1,'pending':defer_projection,
                'checkpoint_at':state['checkpoint_at'],'reason':str(reason)[:256]}
            state['owner'] = {'username': conv.username, 'conv_id': conv.conv_id}
            state['transcript_message_count'] = len(conv.messages_history)
            references = compact_checkpoint_evidence(state, root / 'evidence')
            write_checkpoint(checkpoint, state)
            if not defer_projection:
                workflow = {}
                if session._current_line_id:
                    from agents.task_line import get_store
                    workflow = session._workflow_view()
                write_checkpoint(root / 'task_manifest.json', {
                    'updated_at': state['checkpoint_at'], 'goal_contract': state['goal_contract'],
                    'workflow': workflow, 'current_agent': state['current_agent'],
                    'chain_file': str(root / 'orchestration_chain.json'),
                    'pending_workflow_patch': state.get('pending_workflow_patch'),
                    'waiting_for_user_input': state.get('waiting_for_user_input'),
                    'pending_user_interaction': state.get('pending_user_interaction'),
                    'recovery_file': str(root / 'recovery_state.json'),
                    'lifecycle_file': str(root / 'lifecycle.json'),
                    'session_checkpoint_file': str(checkpoint),
                    'supervisor_checkpoint_file': str(root / 'supervisor_checkpoint.json'),
                    'tool_evidence_refs': references,
                    'delivery_owner': 'lead-orchestrator', 'negotiation_owner': 'lead-orchestrator',
                })
            apply_evidence_references(session.memory, references)
            if conv.session is session:
                conv.session_state = state
        session._on_checkpoint = _save_checkpoint
        session._on_chain_committed = lambda: _workflow_events.publish(conv.username, conv.conv_id)
        def _progress(*args, **kwargs):
            if getattr(session, '_defer_scientific_delivery', False) and len(args) >= 3 and args[2] == '正在回复':
                args = (args[0], '', '正在生成答复（待核验）', *args[3:])
            conv._on_progress(*args, **kwargs)
            _workflow_events.publish(conv.username, conv.conv_id)
        def _partial(*args, **kwargs):
            conv._on_partial_result(*args, **kwargs)
            _workflow_events.publish(conv.username, conv.conv_id)
        session._on_progress = _progress
        session._on_partial_result = _partial
        # Wire user-interrupt (Claude-Code-style Esc): session asks the
        # conversation whether the user clicked "中断" while the agent was
        # thinking. Returns True → the loop FREEZES (no further LLM/tool calls).
        from agents.workflow_user_control import bind_api_workflow_control
        bind_api_workflow_control(conv,session)
        session._on_interrupt_requested = (
            lambda c=conv, actor=session: bool(getattr(c, "interrupt_requested", False)
                or c.session is not actor
                or getattr(actor, '_api_turn_id', None)
                   and actor._api_turn_id != c.current_uid)
        )
        # Optional redirect message attached to the interrupt ("中断加话和需求").
        session._on_interrupt_message = (
            lambda c=conv: str(getattr(c, "interrupt_message", "") or "")
        )
        # After the redirect is consumed, clear it so it applies only once.
        def _clear_interrupt(c=conv, actor=session):
            if (c.session is not actor or getattr(actor, '_api_turn_id', None)
                    and actor._api_turn_id != c.current_uid):
                return
            c.interrupt_message = ""
            c.interrupt_requested = False
            if c.session and c.session._runtime_snapshot and c.session._runtime_snapshot():
                from agents.parallel_workflow import WorkflowStore
                store = WorkflowStore(c.session.config.project_root / 'data/state/parallel_workflows.json')
                store.user_pause(store.identity(c.username, c.conv_id), False)
        session._on_interrupt_clear = _clear_interrupt
        check_hydration_control()
        ready_binding = (session, conv.username, conv.conv_id,
                         str(session.config.project_root), session._current_line_id)
        session._api_ready_binding = ready_binding
        def after_bind():
            if (conv.session is not session
                    or getattr(conv, 'current_uid', None) != initial_uid
                    or (getattr(conv, 'interrupt_requested', False),
                        getattr(conv, 'interrupt_message', '')) != initial_interrupt):
                raise ValueError('prepared session was not bound to its unchanged owner')
            recovered = session.context.pop('preflight_recovery_ready', None)
            if recovered:
                session._emit_lifecycle_event('preflight_recovered', {
                    **recovered,
                    'requires_user': False,
                    'main_already_handling': False,
                    'instruction': (
                        recovered['instruction'] if isinstance(recovered.get('instruction'), str) and recovered['instruction'] else
                        'The current approved graph was already committed. Continue with execute_workflow for that approved version; do not propose the same graph or ask for the same permission again.'
                        if recovered.get('next_action') == 'execute_workflow' else
                        'The validated complete workflow draft was restored automatically. Continue with propose_workflow_patch and execution; do not ask the user about the repaired internal guard.'
                    ),
                })
                session._checkpoint('preflight_recovery_scheduled')
        session._api_after_bind = after_bind
        check_hydration_control()
        if publish:
            # Legacy synchronous callers remain supported. Async controls that
            # prepare in a worker must use publish=False and CAS on their loop.
            conv.session = session
            conv._api_session_binding = ready_binding
            after_bind()
    return session


# ── Models ──
class RegisterRequest(BaseModel):
    username: str
    password: str
    display_name: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class ModelConnectionRequest(BaseModel):
    mode: str = "default"
    base_url: str = ""
    model: str = ""
    api_key: str = ""

class QueryRequest(BaseModel):
    query: str
    conv_id: str = ""
    client_turn_id: Optional[str] = Field(default=None, min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_-]+$")

class QueryResponse(BaseModel):
    # Correlation only: never a permission token or a retry-idempotency key.
    client_turn_id: Optional[str] = None
    answer: str
    reasoning: Optional[str] = None
    agent_name: Optional[str] = None
    status: Optional[str] = None
    uid: Optional[str] = None
    done: bool = True
    conv_id: Optional[str] = None


# ── Auth endpoints ──
@app.post("/api/register")
async def api_register(req: RegisterRequest):
    result = register_user(req.username, req.password, req.display_name)
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["error"])
    session_manager.register_token(result["token"], result["username"])
    return result

@app.post("/api/login")
async def api_login(req: LoginRequest):
    result = login_user(req.username, req.password)
    if not result["ok"]:
        raise HTTPException(status_code=401, detail=result["error"])
    session_manager.register_token(result["token"], result["username"])
    return result

@app.post("/api/logout")
async def api_logout(authorization: Optional[str] = Header(None)):
    if authorization:
        parts = authorization.strip().split()
        if len(parts) == 2 and parts[0].lower() == "bearer":
            session_manager.remove_token(parts[1])
    return {"status": "logged out"}

@app.get("/api/me")
async def api_me(username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    from auth import _avatar_url
    entry = load_users().get(username, {})
    return {"username": username, "display_name": user.display_name or username,
            "avatar_url": _avatar_url(entry if isinstance(entry, dict) else {})}


@app.get("/api/model-connection")
async def get_model_connection(username: str = Depends(_get_user)):
    return public_model_connection(username)


@app.put("/api/model-connection")
async def put_model_connection(req: ModelConnectionRequest, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    if any(conv.is_processing for conv in user.conversations.values()):
        raise HTTPException(409, "该用户仍有模型轮次运行；请中断或等待结束后再切换连接")
    try:
        result = save_model_connection(username, req.mode, req.base_url, req.model, req.api_key)
    except ValueError as error:
        raise HTTPException(400, str(error))
    # Idle live objects hold an Anthropic client created from the old secret.
    # Drop only those objects; their durable checkpoints remain authoritative
    # and are restored on the next turn with the new per-user connection.
    for conv in user.conversations.values():
        conv.session = None
        conv.supervisor_session = None
    return result


# ── Conversation endpoints ──
@app.get("/api/conversations")
async def list_conversations(username: str = Depends(_get_user)):
    """List all conversations for current user."""
    user = session_manager.get_user(username)
    convs = []
    for cid, conv in user.conversations.items():
        # Count user messages (ignore legacy context placeholder blocks)
        user_msgs = sum(1 for m in conv.messages_history if m.get("role") == "user")
        convs.append({
            "conv_id": cid,
            "title": conv.title,
            "created_at": conv.created_at,
            "message_count": user_msgs,
            "is_current": cid == user.current_conv_id,
            "is_processing": conv.is_processing,
            "current_status": conv.current_status or "",
        })
    # Sort by created_at desc
    convs.sort(key=lambda c: c["created_at"], reverse=True)
    return {"conversations": convs, "current_conv_id": user.current_conv_id}


@app.post("/api/conversations")
async def create_conversation(username: str = Depends(_get_user)):
    """Create a new conversation."""
    user = session_manager.get_user(username)
    conv = user.new_conversation()
    (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
    return {"conv_id": conv.conv_id, "title": conv.title, "created_at": conv.created_at}


@app.get("/api/conversations/{conv_id}")
async def get_conversation(conv_id: str, username: str = Depends(_get_user)):
    """Get full message history of one conversation PLUS live state.

    Live state (is_processing / current_status / current_answer / reasoning /
    partial_results / execution_logs) is returned so the frontend can rebuild
    the full picture after a session switch — otherwise a conversation whose
    agent is still running in the background appears "dead" (task + responses
    seem to have disappeared) because only completed turns are in history.
    """
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    # Read live controls/history only after asynchronous hydration finishes.
    workflow = await _workflow_state_async(conv)
    # 返回完整对话历史（本地记录不设条数上限，压缩只针对 LLM 上下文）
    _messages = [m for m in conv.messages_history if m.get("role") != "context"]
    return {
        "conv_id": conv.conv_id,
        "title": conv.title,
        "messages": _messages,
        "is_processing": conv.is_processing,
        "current_status": conv.current_status,
        "current_answer": conv.current_answer or "",
        "current_reasoning": conv.current_reasoning or "",
        "current_agent": conv.current_agent or "",
        "current_uid": conv.current_uid or "",
        "partial_results": conv.partial_results[-10:],
        "execution_logs": conv.execution_logs[-15:],
        "workflow": workflow,
    }


@app.get("/api/conversations/{conv_id}/workflow")
async def get_workflow(conv_id: str, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return await _workflow_state_async(conv)


# Immutable native deliverables: authentication selects the conversation;
# requests never provide a filesystem path or a publication hash.
def _deliverable_context(username, conv_id):
    from agents.config import get_config
    from agents.parallel_workflow import WorkflowStore
    from agents.workspace import validate_scope_component
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, 'conversation not found')
    if (conv.username, conv.conv_id) != (username, conv_id):
        raise HTTPException(403, 'conversation owner mismatch')
    root = get_config().project_root
    session_dir = root / 'runs' / validate_scope_component(username) / validate_scope_component(conv_id)
    store = WorkflowStore(root / 'data/state/parallel_workflows.json')
    workflow_id = store.identity(username, conv_id)
    def snapshot():
        if user.get_conversation(conv_id) is not conv or (conv.username, conv.conv_id) != (username, conv_id):
            raise PermissionError('deliverable conversation ownership changed')
        return store.snapshot(workflow_id)
    return session_dir, {'username': username, 'conv_id': conv_id}, snapshot


@app.get('/api/conversations/{conv_id}/artifacts')
def conversation_deliverables(conv_id: str, username: str = Depends(_get_user)):
    from agents.result_delivery import list_deliverables
    root, owner, snapshot = _deliverable_context(username, conv_id)
    try:
        state = snapshot()
        if not state:
            return {'artifacts': []}
        return {'artifacts': list_deliverables(root, state, owner)}
    except PermissionError:
        raise HTTPException(403, 'deliverable owner scope is unavailable')
    except (OSError, ValueError, KeyError, TypeError):
        raise HTTPException(409, 'accepted deliverable inventory requires verification')


@app.get('/api/conversations/{conv_id}/artifacts/{artifact_id}')
def download_conversation_deliverable(conv_id: str, artifact_id: str,
                                     username: str = Depends(_get_user)):
    from fastapi.responses import Response
    from urllib.parse import quote
    from agents.result_delivery import read_deliverable
    root, owner, snapshot = _deliverable_context(username, conv_id)
    try:
        raw, media_type, name = read_deliverable(root, snapshot(), owner, artifact_id,
                                                current_state=snapshot)
        return Response(raw, media_type=media_type, headers={
            'Content-Disposition': "attachment; filename*=UTF-8''" + quote(name, safe=''),
            'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff'})
    except KeyError:
        raise HTTPException(404, 'deliverable not found')
    except PermissionError:
        raise HTTPException(403, 'deliverable owner scope is unavailable')
    except (OSError, ValueError, TypeError):
        raise HTTPException(409, 'deliverable differs from its accepted native publication')


@app.delete("/api/conversations/{conv_id}")
async def delete_conversation(conv_id: str, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    if conv_id in user.conversations:
        existing = user.conversations[conv_id]
        _ensure_session(existing)
        runtime = existing.session._runtime_snapshot() if existing.session and existing.session._runtime_snapshot else {}
        if runtime:
            from agents.parallel_workflow import WorkflowStore
            store = WorkflowStore(existing.session.config.project_root / 'data/state/parallel_workflows.json')
            try: store.retire(store.identity(username, conv_id))
            except ValueError as error: raise HTTPException(409, str(error))
        executor = _workflow_runtimes.pop((username, conv_id), None)
        if executor: executor.shutdown()
        del user.conversations[conv_id]
        with _conv_locks_lock:
            _conv_locks.pop((username, conv_id), None)
        if user.current_conv_id == conv_id:
            user.current_conv_id = ""
            user.new_conversation()
        (await _durable_io(session_manager.persist_conversations,
            username,
            conv_id=user.current_conv_id,
            delete_conv_id=conv_id,
        ))
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.post("/api/conversations/{conv_id}/rename")
async def rename_conversation(conv_id: str, req: dict, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    conv.title = req.get("title", conv.title)[:50]
    (await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id))
    return {"status": "renamed", "title": conv.title}


@app.post("/api/conversations/{conv_id}/switch")
async def switch_conversation(conv_id: str, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    if conv_id not in user.conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    user.current_conv_id = conv_id
    (await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id))
    return {"status": "switched", "conv_id": conv_id}


# ── Query endpoints ──
@app.get("/api/latest_answer")
async def get_latest_answer(username: str = Depends(_get_user), conv_id: str = ""):
    conv = _get_conv(username, conv_id)
    workflow = await _workflow_state_async(conv)
    return {
        "answer": conv.current_answer,
        "answer_phase": getattr(conv.session, "context", {}).get("answer_delivery", {}).get("phase", "published"),
        "review_status": getattr(conv.session, "context", {}).get("answer_delivery", {}).get("review_status", "not_required"),
        "reasoning": conv.current_reasoning,
        "agent_name": conv.current_agent,
        "status": conv.current_status,
        "uid": conv.current_uid or "",
        "done": not conv.is_processing,
        "logs": conv.execution_logs[-15:] if conv.execution_logs else [],
        "partial_results": conv.partial_results,  # streaming sub-agent results
        "conv_id": conv.conv_id,
        "interrupt_requested": bool(getattr(conv, "interrupt_requested", False)),
        "interrupt_message": str(getattr(conv, "interrupt_message", "")),
        "queued_redirect": bool(getattr(conv, "queued_redirect", "")),
        "workflow": workflow,
    }


@app.get('/api/conversations/{conv_id}/events')
async def conversation_events(request: Request, conv_id: str, username: str = Depends(_get_user)):
    """Push only this authenticated user's conversation revisions."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None:
        raise HTTPException(404, 'conversation not found')

    async def stream():
        revision = -1
        while not await request.is_disconnected():
            current = _workflow_events.revision(username, conv_id)
            if revision != current:
                workflow = await _workflow_state_async(conv)
                revision = _workflow_events.revision(username, conv_id)
                payload = {
                    'revision': revision,
                    'conv_id': conv_id,
                    'workflow': workflow,
                    'done': not conv.is_processing,
                    'agent_name': conv.current_agent,
                    'status': conv.current_status,
                    'answer_phase': getattr(conv.session, 'context', {}).get('answer_delivery', {}).get('phase', 'published'),
                    'review_status': getattr(conv.session, 'context', {}).get('answer_delivery', {}).get('review_status', 'not_required'),
                    'reply_preview': conv.current_reasoning if conv.current_status == '正在回复' else '',
                    'uid': conv.current_uid or '',
                    'interrupt_requested': bool(conv.interrupt_requested),
                    'queued_redirect': bool(getattr(conv, 'queued_redirect', '')),
                }
                yield 'event: workflow\ndata: ' + json.dumps(payload, ensure_ascii=False, default=str) + '\n\n'
                continue
            next_revision = await asyncio.to_thread(
                _workflow_events.wait, username, conv_id, revision, 20.0)
            if next_revision == revision:
                yield ': keepalive\n\n'

    return StreamingResponse(stream(), media_type='text/event-stream', headers={
        'Cache-Control': 'no-cache, no-store, no-transform',
        'X-Accel-Buffering': 'no',
    })


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, username: str = Depends(_get_user)):
    conv = _get_conv(username, request.conv_id)
    new_uid = str(datetime.now().timestamp())
    from agents.workflow_user_control import advance_user_control

    if conv.is_processing:
        # 用户中断（freeze）后发的新消息 = Claude-Code 的 Esc+typing 重定向。
        # 关键：绝不能 429 拒绝——否则用户"中断了发消息又说正在处理中"。
        # 这里清除冻结标志（这条新消息就是续写指令），排队等当前轮真正停止
        # 后自动执行；若当前轮只是卡在长工具调用（run_bash 等），排队会在
        # 其超时返回后立即跑，而不是把用户锁死。
        if getattr(conv, "interrupt_requested", False):
            conv.interrupt_requested = False
            conv.interrupt_message = ""
            advance_user_control(conv)
            conv.current_uid = new_uid  # 新消息接管本轮所有权
            conv.queued_redirect = request.query  # 前端可显示"已排队"
            conv.current_status = "⛔ 已中断；收到新消息，当前轮停止后立即继续"
            asyncio.create_task(_run_after_current_turn(username, conv.conv_id, request.query, new_uid,
                **({"client_turn_id": request.client_turn_id} if request.client_turn_id is not None else {})))
            return QueryResponse(
                answer="⛔ 已中断当前轮，收到你的新消息；当前工具调用结束后立即继续执行（无需再发消息）。",
                reasoning=f"已收到新需求（中断重定向）: {request.query}\n\n当前轮停止后立即执行...",
                agent_name="orchestrator",
                status="⛔ 已中断（新消息排队中）",
                uid=new_uid,
                client_turn_id=request.client_turn_id,
                done=False,
                conv_id=conv.conv_id,
            )
        raise HTTPException(status_code=429, detail="这个对话正在处理中，请稍等")

    # 非处理中：清掉任何残留的冻结标志——用户发的新消息就是续写/新委派，
    # 绝不能让上一轮的 interrupt_requested 残留把新 turn 第一轮就冻住
    #（曾导致"发消息又返回⛔用户中断"的死循环）。
    if getattr(conv, "interrupt_requested", False):
        conv.interrupt_requested = False
        conv.interrupt_message = ""
    advance_user_control(conv)
    conv.current_uid = new_uid
    conv.is_processing = True  # reserve before create_task yields to another request
    _workflow_events.publish(username, conv.conv_id)
    asyncio.create_task(run_agent_background(username, request.query, conv.conv_id,
        **({"client_turn_id": request.client_turn_id} if request.client_turn_id is not None else {})))

    return QueryResponse(
        answer="正在处理您的请求...",
        reasoning=f"收到请求: {request.query}\n\n开始分析...",
        agent_name="orchestrator",
        status="Processing",
        uid=new_uid,
        client_turn_id=request.client_turn_id,
        done=False,
        conv_id=conv.conv_id,
    )


async def _run_after_current_turn(username: str, conv_id: str, query: str, uid: str, *, client_turn_id=None):
    """Wait for the currently-frozen/stuck turn's thread to actually EXIT, then
    run the user's redirect message.

    Key design: the queue path holds `is_processing=True` continuously (the old
    turn's finally is uid-guarded and won't clear it), so the frontend keeps
    polling with no "gap". The only reliable signal that the old thread exited
    is that it released the per-conv lock — so we probe the lock. Bounded wait
    (~120s): a stuck tool call is capped at 180s per call and eventually times
    out, so the queued message runs shortly after; if it truly never releases,
    give the user an honest status instead of silently dropping it.
    """
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id) if user else None
    lock = _get_conv_lock(username, conv_id)
    for _ in range(240):  # up to 120s
        if conv is None or conv.current_uid != uid:
            return  # superseded / conversation gone
        if lock.acquire(blocking=False):  # old thread released → it exited
            lock.release()  # don't hold it; the redirect turn will acquire
            break
        await asyncio.sleep(0.5)
    if conv is None or conv.current_uid != uid:
        return
    if lock.acquire(blocking=False):
        lock.release()
    else:
        # Still busy after the cap — the tool call has not returned. Report
        # honestly so the user isn't left guessing (their message is preserved
        # in history; the agent WILL still see it when the current turn ends).
        conv.current_status = "⚠️ 当前工具调用仍未结束，你的消息已排队，稍后自动继续。"
        conv.current_reasoning = f"（排队中）: {query}"
        conv.execution_logs.append({
            "time": datetime.now().isoformat(),
            "message": "⚠️ 中断重定向排队超时（120s），当前工具调用仍未结束；消息已保留，轮次结束后自动继续。",
        })
        try:
            (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
        except Exception:
            pass
        return
    # Old thread exited — run the redirect as a normal turn (is_processing is
    # already held True by the queue path; run_agent_background keeps it True).
    conv.queued_redirect = ""
    await run_agent_background(username, query, conv_id,
        **({"client_turn_id": client_turn_id} if client_turn_id is not None else {}))


@app.post("/api/conversations/{conv_id}/interrupt")
async def interrupt_conversation(conv_id: str, request: Request, username: str = Depends(_get_user)):
    """User interrupt — Claude-Code-style (Esc): FREEZE the agent's thinking.

    Critical semantics (per user):
    - Does NOT stop previously-submitted tasks/jobs. SLURM jobs keep running.
    - Only makes the agent STOP its next-step thinking & further tool calls.
    - Freezes the current delegation, waiting for a subsequent delegation
      (the user's next message / redirect).
    - Optionally carries a redirect `message` (like Claude Code's Esc + typing
      a new requirement): the agent should switch to that new instruction.

    After the turn returns, the user can either (a) send a new message (which
    becomes the redirect and un-freezes the session) or (b) explicitly clear
    the freeze.
    """
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    from agents.workflow_user_control import advance_user_control
    advance_user_control(conv)
    conv.interrupt_requested = True
    conv.interrupt_message = ""  # default: plain freeze
    _redirect = ""
    # Accept optional redirect requirement: {"message": "...新需求..."}
    try:
        _body = await request.json()
        if isinstance(_body, dict) and _body.get("message"):
            conv.interrupt_message = str(_body["message"])[:2000]
            _redirect = conv.interrupt_message
    except Exception:
        pass
    # If the user typed a redirect into the interrupt prompt, do NOT leave the
    # conversation frozen waiting for a second message (anti-human). Freeze the
    # current turn, then automatically resume with the redirect once the frozen
    # turn has stopped — typing once feels like "continue thinking".
    control_source=_session_admission_source(conv)
    expected_epoch=control_source[-1]
    actor=None
    def current_control():
        return ((conv.username,conv.conv_id)==(username,conv_id)
            and conv.current_uid==control_source[1]
            and getattr(conv,'_workflow_user_control_epoch',0)==expected_epoch
            and conv.interrupt_requested is True and conv.interrupt_message==_redirect
            and (actor is None or conv.session is actor))
    try:
        actor=await _ensure_api_session(conv)
        await _set_owned_user_pause(conv,actor,True,current_control)
        if not current_control():raise ValueError('pause control was superseded during session preparation')
        if actor is not None:
            await _durable_io(_owned_conversation_io,conv,actor._emit_lifecycle_event,
                'user_paused',{'redirect':_redirect,'user_requested_freeze':True})
    except ValueError as error:
        raise HTTPException(status_code=409,detail=str(error)) from error
    if _redirect and conv.is_processing:
        conv.current_status = "⛔ 用户中断：冻结本轮，稍后自动用你的需求继续"
        try:
            (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
        except Exception:
            pass
        asyncio.create_task(_resume_after_freeze(username, conv_id, _redirect))
        return {
            "ok": True,
            "interrupt_requested": True,
            "redirect": _redirect,
            "conv_id": conv_id,
            "note": "已请求中断并附带重定向需求：agent 会先停止当前思考，"
                    "然后自动用你的需求继续执行（无需再发第二条消息）。"
                    "已提交的 SLURM 作业不受影响。",
        }
    conv.current_status = "⛔ 用户中断（冻结本轮，等待后续委派）"
    try:
        (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
    except Exception:
        pass
    return {
        "ok": True,
        "interrupt_requested": True,
        "conv_id": conv_id,
        "note": "已请求中断：agent 会停止本轮思考与进一步动作，已提交的 SLURM 作业不受影响。"
                "请发送新消息作为重定向（新需求）继续，agent 会从上次停下的地方接着思考。",
    }


@app.post("/api/conversations/{conv_id}/interrupt_clear")
async def clear_interrupt(conv_id: str, username: str = Depends(_get_user)):
    """Clear a pending interrupt/freeze flag (after the user resolves it)."""
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    from agents.workflow_user_control import advance_user_control
    advance_user_control(conv)
    conv.interrupt_requested = False
    conv.interrupt_message = ""
    control_source=_session_admission_source(conv)
    expected_epoch=control_source[-1]
    actor=None
    def current_control():
        return ((conv.username,conv.conv_id)==(username,conv_id)
            and conv.current_uid==control_source[1]
            and getattr(conv,'_workflow_user_control_epoch',0)==expected_epoch
            and conv.interrupt_requested is False and not conv.interrupt_message
            and (actor is None or conv.session is actor))
    try:
        actor=await _ensure_api_session(conv)
        await _set_owned_user_pause(conv,actor,False,current_control)
        if not current_control():raise ValueError('clear control was superseded during session preparation')
    except ValueError as error:
        raise HTTPException(status_code=409,detail=str(error)) from error
    try:
        (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
    except Exception:
        pass
    return {"ok": True, "interrupt_requested": bool(conv.interrupt_requested)}


# ── Background agent execution ──
def _api_turn_is_current(conv, turn_uid):
    return (conv.current_uid == turn_uid and not conv.interrupt_requested
            and not conv.interrupt_message)


async def _acquire_conversation_lease(lock, timeout=60):
    pending = asyncio.create_task(asyncio.to_thread(lock.acquire, timeout=timeout))
    _api_durable_writes.add(pending)
    pending.add_done_callback(_api_durable_writes.discard)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        acquired = await _finish_pending_io(pending)
        if acquired:
            lock.release()
        raise


async def _review_answer_delivery(conv, event):
    """Bounded independent read-only review, completed before final publication."""
    actor, uid = conv.session, conv.current_uid
    def observe():
        from agents.watch_context import set_context, clear_context
        from agents.session import Session
        from agents.goal_contract import GoalContract
        from agents.state_io import compact_checkpoint_evidence, write_checkpoint, apply_evidence_references
        from agents.answer_delivery import digest
        set_context(conv.username, conv.conv_id, 'answer_reviewer', actor._current_line_id)
        try:
            # Never share the lifecycle observer's mutable messages/checkpoint.
            reviewer = Session(config=actor.config, registry=actor.registry)
            reviewer.client = reviewer.client.with_options(timeout=60, max_retries=0)
            reviewer.goal_contract = GoalContract.from_dict(actor.goal_contract.to_dict())
            reviewer._current_line_id = actor._current_line_id
            reviewer._evidence_root = actor._evidence_root
            reviewer._chain_state_is_primary = False
            reviewer._on_interrupt_requested = lambda: (conv.session is not actor
                or conv.current_uid != uid or conv.interrupt_requested)
            identity = digest([uid, event.get('event_id'), event.get('payload',{}).get('answer')])
            path = Path(actor._evidence_root) / ('answer-review-' + identity + '.checkpoint.json')
            def checkpoint(state, reason):
                state['owner'] = {'username':conv.username,'conv_id':conv.conv_id,'role':'answer_reviewer'}
                refs = compact_checkpoint_evidence(state,actor._evidence_root)
                write_checkpoint(path,state)
                apply_evidence_references(reviewer.memory,refs)
            reviewer._on_checkpoint = checkpoint
            return reviewer.observe_lifecycle_event(event)
        finally:
            clear_context()
    pending = asyncio.create_task(asyncio.to_thread(observe))
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        await _finish_pending_io(pending)
        raise


async def _gate_answer_delivery(conv, turn_uid, query, result):
    from agents.answer_delivery import draft_identity, digest, sources_unchanged, persist_draft, limited_answer
    actor = conv.session
    make_event = getattr(actor, 'scientific_review_event', None)
    if not make_event or not isinstance(result, str):
        return result
    goal = actor.goal_contract.to_dict()
    goal_sha = digest(goal)
    def control_current():
        return (conv.session is actor and _api_turn_is_current(conv, turn_uid)
                and digest(actor.goal_contract.to_dict()) == goal_sha)
    if not control_current(): return None
    collecting = asyncio.create_task(asyncio.to_thread(make_event, query, result))
    try:
        event = await asyncio.shield(collecting)
    except asyncio.CancelledError:
        await _finish_pending_io(collecting)
        raise
    if not control_current(): return None
    if not event:
        if getattr(actor, '_answer_completion_candidate', False): actor.task_complete = True
        if hasattr(actor,'context'):
            actor.context['answer_delivery'] = {'turn_uid':turn_uid,'phase':'published','review_status':'not_required'}
        return result
    draft = draft_identity(conv.username, conv.conv_id, turn_uid, goal, result, event)
    original_complete = bool(getattr(actor, '_answer_completion_candidate', False) or actor.task_complete)
    context = actor.context
    def current():
        return (conv.session is actor and _api_turn_is_current(conv, turn_uid)
                and digest(actor.goal_contract.to_dict()) == draft['goal_sha256'])
    if not current(): return None
    previous = context.get('answer_delivery') or {}
    if previous.get('draft_id') == draft['draft_id']:
        if previous.get('phase') == 'verifying' and getattr(actor,'_active_answer_delivery_id',None) == draft['draft_id']:
            return None
        if previous.get('phase') in {'published','limited'} and await asyncio.to_thread(sources_unchanged, draft['sources']):
            if not current(): return None
            actor.task_complete = original_complete
            return result if previous['phase'] == 'published' else limited_answer(event, previous.get('limited_reason','证据核验不完整。'))
    actor.task_complete = False
    conv.current_status = '正在核验证据'
    conv.current_reasoning = '计算结果已保留，正在核对答复中的事实与限制。'
    record = {**draft, 'phase':'verifying', 'review_status':'pending', 'owner_pid':os.getpid()}
    actor._active_answer_delivery_id = draft['draft_id']
    context['answer_delivery'] = record
    _workflow_events.publish(conv.username, conv.conv_id)
    checkpoint = getattr(actor, '_checkpoint', None)
    async def save():
        if checkpoint and current():
            await _durable_io(_owned_conversation_io, conv, checkpoint, 'answer_delivery')
    try:
        record['evidence_path'] = await _durable_io(persist_draft, getattr(actor,'_evidence_root',None), draft, result, event)
        await save()
        if not current(): return None
        before = await asyncio.to_thread(sources_unchanged, draft['sources'])
        if event.get('payload',{}).get('source_gaps'):
            receipt = {'review_status':'insufficient_evidence','reason':'答复引用了已有结果，但当前原生证据无法验证；未发布未经核验的分析。'}
        elif not before:
            receipt = {'review_status':'insufficient_evidence', 'reason':'引用的原生证据已变化或超过核验边界。'}
        else:
            try: receipt = await _review_answer_delivery(conv, event)
            except asyncio.CancelledError: raise
            except Exception as error:
                receipt = {'review_status':'insufficient_evidence','reason':'独立核验暂不可用：'+type(error).__name__}
        if not current(): return None
        record['review_receipt_sha256'] = digest(receipt)
        record['review_evidence_call_ids'] = [c['call_id'] for c in receipt.get('evidence_calls',[])[:144] if c.get('call_id')]
        unchanged = await asyncio.to_thread(sources_unchanged, draft['sources'])
        if not current(): return None
        verified = (unchanged and receipt.get('review_status') == 'verified'
                    and receipt.get('scientific_review',{}).get('passed') is True)
        text = result if verified else limited_answer(event, receipt.get('reason') or '; '.join(
            receipt.get('scientific_review',{}).get('issues',[])) or '证据不完整或引用来源已变化。')
        # No await between final ownership/source recheck and selecting publication.
        record.update(phase='published' if verified else 'limited',
            review_status='verified' if verified else 'insufficient_evidence',
            publication_id=draft['draft_id'], published_text_sha256=hashlib.sha256(text.encode()).hexdigest())
        if not verified: record['limited_reason'] = receipt.get('reason') or '; '.join(receipt.get('scientific_review',{}).get('issues',[])) or '证据不完整或引用来源已变化。'
        actor.task_complete = original_complete
        await save()
        # Final source check follows all checkpoint I/O; publication runs on the
        # same loop continuation after this check and its ownership guard.
        final_sources = await asyncio.to_thread(sources_unchanged, draft['sources'])
        if not current(): return None
        if not final_sources:
            text = limited_answer(event, '引用来源在答复发布前发生变化，旧分析不能发布。')
            record.update(phase='limited',review_status='insufficient_evidence',limited_reason='引用来源已变化。',
                          published_text_sha256=hashlib.sha256(text.encode()).hexdigest())
        return text
    except asyncio.CancelledError:
        if conv.session is actor and context.get('answer_delivery') is record:
            record.update(phase='obsolete', review_status='cancelled')
        raise
    except Exception as error:
        if not current(): return None
        record.update(phase='limited',review_status='insufficient_evidence',limited_reason='证据交付暂不可用：'+type(error).__name__)
        actor.task_complete = original_complete
        text = limited_answer(event,record['limited_reason'])
        record.update(publication_id=draft['draft_id'],published_text_sha256=hashlib.sha256(text.encode()).hexdigest())
        return text
    finally:
        if getattr(actor,'_active_answer_delivery_id',None) == draft['draft_id']:
            actor._active_answer_delivery_id = None
        if not current() and context.get('answer_delivery') is record:
            record.update(phase='obsolete', review_status='superseded')


async def _attach_workflow_reply_status(conv, turn_uid, text):
    """Publish current failure/incompleteness even when the scientific draft is limited."""
    from agents.answer_delivery import digest, sources_unchanged
    from agents.workflow_reply_facts import attach_task_status
    actor=conv.session
    getter=getattr(actor,'workflow_reply_facts',None)
    if not callable(getter) or not isinstance(text,str):return text
    goal=actor.goal_contract.to_dict();goal_sha=digest(goal)
    def current():
        return (conv.session is actor and _api_turn_is_current(conv,turn_uid)
                and digest(actor.goal_contract.to_dict())==goal_sha)
    if not current():return None
    delivery=getattr(actor,'context',{}).get('answer_delivery')or {}
    scientific=(delivery.get('turn_uid')==turn_uid and delivery.get('review_status')not in {None,'not_required'})
    context_probe=getattr(actor,'_workflow_reply_context_required',None)
    if not scientific and callable(context_probe) and not context_probe():return text
    def gather():
        try:
            facts=getter(username=conv.username,conv_id=conv.conv_id,task_related_only=not scientific)
            error=False
        except Exception:
            facts,error=None,True
        # Status collection is additional I/O after the scientific gate. Do
        # not publish its old prose if a native source changed in that window.
        status_changed=False;guard_available=True;guard_receipt=None
        try:
            references=delivery.get('sources',[])
            source_valid=True
            has_task=bool(facts or goal.get('execution_authorized')is True and goal.get('approved_nodes'))
            if scientific and has_task and not error:
                checker=getattr(actor,'_workflow_reply_check',None)
                receipt=checker(goal=goal,evidence_root=getattr(actor,'_evidence_root',None),
                    references=references) if callable(checker) else {}
                guard_available=receipt.get('guarded')is True
                if guard_available:
                    latest=receipt['facts']
                    status_changed=digest(latest)!=digest(facts)
                    facts=latest;source_valid=receipt.get('sources_valid')is True
                    guard_receipt={k:receipt.get(k)for k in ('workflow_id','linearized_at')}
                else:
                    # Unknown bindings may report a fresh observation only;
                    # no scientific prose is released on a stat-only check.
                    source_valid=sources_unchanged(references)
                    facts=getter(username=conv.username,conv_id=conv.conv_id,task_related_only=False)
            elif scientific:
                # No approved workflow: retain the existing ordinary science
                # path, with the actual source SHA as its last file I/O.
                source_valid=sources_unchanged(references)
        except Exception:
            source_valid=False;error=True;facts=None
        return facts,error,source_valid,status_changed,guard_available,guard_receipt
    pending=asyncio.create_task(asyncio.to_thread(gather))
    try:
        facts,error,source_valid,status_changed,guard_available,guard_receipt=await asyncio.shield(pending)
    except asyncio.CancelledError:
        await _finish_pending_io(pending)
        raise
    if not current():return None
    if not guard_available:
        text='当前任务状态与原生来源尚未完成同源核验，此前科学草稿暂不发布。已有结果及回执已保留。'
        actor.task_complete=False
        delivery.update(phase='limited',review_status='insufficient_evidence',limited_reason='同源执行状态读取保护不可用。')
    elif not source_valid:
        text='原生来源未通过发布前的内容核验，暂不能发布此前的科学分析。已有计算及回执已保留。'
        actor.task_complete=False
        delivery.update(phase='limited',review_status='insufficient_evidence',limited_reason='原生来源在状态核对期间发生变化。')
    elif status_changed:
        text='任务执行状态在答复核验期间发生变化，此前草稿不能作为当前尝试的结论。已有结果与回执已保留。'
        actor.task_complete=False
        delivery.update(phase='limited',review_status='insufficient_evidence',limited_reason='当前尝试或任务状态在来源核对期间发生变化。')
    if error:
        actor.task_complete=False
        text='当前任务执行状态暂无法核对，不能确认整体完成。\n\n'+text
    elif facts and (facts.get('scope')!={'username':conv.username,'conv_id':conv.conv_id}
            or facts.get('goal_version')!=goal.get('version')
            or facts.get('plan_version')!=goal.get('approved_plan_version')):
        actor.task_complete=False
        facts=None
        text='当前任务状态的归属或版本无法核对，不能确认整体完成。\n\n'+text
    elif facts and facts.get('available')is False:
        actor.task_complete=False
        facts=None
        text='当前任务执行状态暂无法核对，不能确认整体完成。\n\n'+text
    if facts:
        if facts.get('complete')is not True:actor.task_complete=False
        text=attach_task_status(text,facts)
    if delivery.get('turn_uid')==turn_uid:
        delivery.update(workflow_status=facts,workflow_status_sha256=digest(facts),
                        published_text_sha256=hashlib.sha256(text.encode()).hexdigest())
        if guard_receipt:delivery['workflow_status_read_guard']=guard_receipt
    return text


async def run_agent_background(username: str, query: str, conv_id: str, lifecycle_event=None, *, client_turn_id=None):
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id) or user.get_current_conversation()
    _ensure_session(conv)

    conv.is_processing = True
    conv.last_progress_at = time.time()
    # 本轮所有权令牌：若用户在这轮还没结束时已用新消息接管（uid 更新），
    # 本轮结束时的输出必须丢弃，不能覆盖新 turn 的结果（写回前比对）。
    turn_uid = conv.current_uid
    conv.current_status = "Processing..."
    conv.current_reasoning = f"收到请求: {query}\n\n开始分析..."
    conv.current_answer = ""
    conv.execution_logs = []
    conv.partial_results = []  # Reset partial results for this query

    # ── 用户提问立即落盘（防"吞对话"）──────────────────────────────
    # 旧逻辑只在整轮 agent 跑完后才把 user 消息写入 messages_history；
    # 一旦后端中途重启/崩溃/超时，用户刚发的问题就从未进过历史，
    # 前端 reload 后问题和回复一起消失。现在 turn 一开始就把 user 消息
    # 写入并持久化，后续任何 exit path 都会补上 assistant 回复。
    archived_message = {"role": "system" if lifecycle_event else "user", "content": query}
    if not lifecycle_event:
        archived_message.update(turn_uid=turn_uid, timestamp=datetime.now().isoformat())
        if client_turn_id is not None:
            archived_message["client_turn_id"] = client_turn_id
    conv.messages_history.append(archived_message)
    # 仅清理遗留的旧 "已执行的计算" context 占位块（一次性脏数据）
    conv.messages_history = [m for m in conv.messages_history if m.get("role") != "context"]
    try:
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id)
    except asyncio.CancelledError:
        # The initial writer has stopped, but no conversation/model lease was
        # reached yet. Release only this reservation and persist that release.
        if conv.current_uid == turn_uid:
            conv.is_processing = False
            await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id)
        raise
    except Exception:
        pass

    if not _api_turn_is_current(conv, turn_uid):
        if conv.current_uid == turn_uid:
            conv.is_processing = False
            await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id)
        return

    # Build context from conversation history (already in session.messages, but keep for display)
    def _run_in_thread():
        # A redirect can also arrive after submit-to-pool but before this
        # worker starts. Do not let stale control enter the model or tools.
        if not _api_turn_is_current(conv, turn_uid):
            return None
        # Never replace process-global sys.stdout here. Multiple conversations
        # run in parallel, and swapping stdout in one worker caused logs from one
        # user/session to leak into another. Structured progress callbacks are
        # the supported logging path.
        # Publish thread-local context so tool executors can auto-register
        # SLURM jobs with JobWatch (with the right conv_id/user).
        from agents.watch_context import set_context, clear_context
        set_context(username, conv.conv_id, conv.current_agent or "orchestrator", conv.session._current_line_id)
        conv.session._api_turn_id = turn_uid
        conv.session._defer_scientific_delivery = True
        conv.session._answer_completion_candidate = False
        try:
            if lifecycle_event:
                result = conv.session.resume_lifecycle_event(lifecycle_event)
                conv.conversation_active = True
            elif conv.conversation_active:
                # Follow-up: session.reply() preserves full context
                from agents.defns import ORCHESTRATOR
                conv.session.current_agent = ORCHESTRATOR
                result = conv.session.reply(query, verbose=False)
            else:
                result = conv.session.start(query, verbose=False)
                conv.conversation_active = True
            return result
        finally:
            if getattr(conv.session,'_api_turn_id',None) == turn_uid:
                conv.session._defer_scientific_delivery = False
            clear_context()

    conv_lock = _get_conv_lock(username, conv.conv_id)
    # Per-conversation lock. The /api/query layer already rejects a second
    # query while this conversation is processing (429 via conv.is_processing),
    # so this lock is a belt-and-suspenders for the same-conversation race
    # only. It must NOT serialize across conversations: the harness and a
    # smoke-test conversation of the same user legitimately run together.
    # A bounded wait keeps the old "silent empty answer" failure visible:
    # if the lock can't be acquired, surface the reason in the answer.
    try:
        acquired = await _acquire_conversation_lease(conv_lock, timeout=60)
    except BaseException:
        if conv.current_uid == turn_uid:
            conv.is_processing = False
            await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id)
        raise
    if not acquired:
        if conv.current_uid != turn_uid:
            return
        conv.is_processing = False
        _busy = "⚠️ 系统繁忙：这个对话正在处理上一条消息，请稍后重试本条消息。"
        conv.current_answer = _busy
        conv.current_status = "系统繁忙"
        # 用户消息已在 turn 开始时落盘，这里补上回复，确保不吞对话
        conv.messages_history.append({"role": "assistant", "content": _busy})
        conv.execution_logs.append({
            "time": datetime.now().isoformat(),
            "message": "系统繁忙：同会话并发处理中，等待60s后放弃"
        })
        try:
            (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
        except Exception:
            pass
        return

    try:
        if not _api_turn_is_current(conv, turn_uid):
            return
        # Another API process may have finished this conversation while we waited
        # for its kernel lease. Restore its newer execution state before replying.
        from agents.workspace import session_root
        checkpoint = session_root(conv.conv_id, username) / 'session_checkpoint.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get('checkpoint_at', 0) > getattr(conv.session, '_last_checkpoint_at', 0):
                conv._api_session_binding = None
                conv.session.import_state(saved)
                _ensure_session(conv)
        conv.current_agent = "orchestrator"
        conv.current_status = "正在分析您的请求..."
        conv.execution_logs.append({"time": datetime.now().isoformat(), "message": "开始分析请求"})

        loop = asyncio.get_event_loop()
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREAD_WORKERS) as pool:
            result = await loop.run_in_executor(pool, _run_in_thread)

        # ── 所有权守卫：若用户已用新消息接管本轮（uid 更新），丢弃旧输出 ──
        # 中断后新消息排队→旧轮结束后才开新轮，正常情况下不会重叠；这里
        # 是兜底：万一旧轮晚归（长工具调用超时返回），不覆盖新轮的结果。
        if conv.current_uid != turn_uid:
            print(f"  [turn] conv {conv_id} superseded (uid {turn_uid} != {conv.current_uid}) — discarding old turn output", flush=True)
            return

        if not _api_turn_is_current(conv, turn_uid):
            return
        result = await _gate_answer_delivery(conv, turn_uid, query, result)
        if result is None or not _api_turn_is_current(conv, turn_uid):
            return
        result = await _attach_workflow_reply_status(conv, turn_uid, result)
        if result is None or not _api_turn_is_current(conv, turn_uid):
            return
        delivery = getattr(conv.session,'context',{}).get('answer_delivery',{})
        publication_id = delivery.get('publication_id') if delivery.get('turn_uid') == turn_uid else None
        if publication_id and any(m.get('publication_id') == publication_id for m in conv.messages_history):
            return
        conv.current_answer = result
        conv.session.last_text = result
        # 用户消息已在 turn 开始时落盘（防中途崩溃/重启吞掉提问），
        # 这里只补上 assistant 回复。本地对话历史完整落盘，不截断不设限；
        # 压缩只对"发给 LLM 的上下文"做（session._trim_messages）。
        conv.current_agent = conv.session.current_agent.name if conv.session.current_agent else "orchestrator"
        conv.current_status = _execution_status(conv.session)
        conv.messages_history.append({'role': 'assistant', 'content': result if result else '',
                                      'agent': 'lead-orchestrator', 'timestamp': time.time(),
                                      'status': conv.current_status, 'uid': turn_uid, 'publication_id':publication_id,
                                      'answer_phase':delivery.get('phase','published'),
                                      'delivery_owner': 'lead-orchestrator', 'negotiation_owner': 'lead-orchestrator'})
        conv.execution_logs.append({'time': datetime.now().isoformat(), 'message': conv.current_status})

        # Auto-title: first user message becomes the title
        if conv.title == "新对话" and conv.messages_history:
            first_user = next((m["content"] for m in conv.messages_history if m.get("role") == "user"), "")
            if first_user:
                conv.title = first_user[:30]

        # Persist conversation to log
        try:
            (await _durable_io(_append_conversation_log, [
                {"time": datetime.now().isoformat(), "username": username,
                 "conv_id": conv_id, "role": role, "content": content}
                for role, content in [("user", query), ("assistant", result or "")]
            ]))
        except Exception:
            pass

        # Persist conversations to users.json so a server restart does not
        # orphan JobWatch notifications (conv "gone" → dropped). Without this,
        # a job that fails hours after a restart never wakes its agent.
        try:
            (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
        except Exception:
            pass

    except Exception as e:
        if conv.current_uid == turn_uid:
            conv.current_status = f"Error: {str(e)}"
            _err = f"处理请求时出错: {str(e)}"
            conv.current_answer = _err
            # 用户消息已在 turn 开始时落盘，这里补上出错回复，保证问题+回复都不消失
            conv.messages_history.append({"role": "assistant", "content": _err})
            conv.execution_logs.append({"time": datetime.now().isoformat(), "message": f"❌ 错误: {str(e)}"})
            try:
                (await _durable_io(session_manager.persist_conversations, username, conv_id=conv.conv_id))
            except Exception:
                pass
    finally:
        try:
            await _finish_api_turn(conv, turn_uid, username)
            # Restart requests run only after the owned reply/chain commit.
            try:
                from agents._restart import consume
                if consume():
                    asyncio.create_task(_perform_backend_restart())
            except Exception as error:
                print(f"  [backend] restart consume error: {error}", flush=True)
        finally:
            try:
                conv_lock.release()
            except RuntimeError:
                pass


async def _finish_api_turn(conv, turn_uid, username):
    if conv.current_uid != turn_uid:
        return
    session = conv.session
    try:
        if session:
            session._api_turn_id = turn_uid
            await _durable_io(_owned_conversation_io, conv,
                              session._persist_chain, 'api_turn_end')
    finally:
        # A redirect can change ownership while disk I/O is pending. Never
        # clear its busy flag, pause or current turn identity from this turn.
        if conv.current_uid == turn_uid:
            if session and session._api_turn_id == turn_uid:
                session._api_turn_id = None
            conv.is_processing = False
            try:
                await _durable_io(session_manager.persist_conversations,
                                        username, conv_id=conv.conv_id)
            except Exception:
                pass
            _workflow_events.publish(username, conv.conv_id)


async def _resume_after_freeze(username: str, conv_id: str, redirect: str):
    """Interrupt + redirect (Claude-Code Esc+typing): after the currently-frozen
    turn has actually stopped, automatically resume the conversation with the
    redirect as the new delegation.

    The user typed ONE message into the interrupt prompt expecting the agent to
    "continue thinking" — the old design stored the redirect but then required a
    SECOND user message to un-freeze, which felt anti-human. Here we wait for the
    frozen turn to release (is_processing → False), then run the redirect
    immediately; context is preserved so the agent resumes where it stopped.
    """
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id) if user else None
    if conv is None:
        return
    # Wait for the frozen turn to stop (it holds the per-conv lock and sets
    # is_processing=False when it returns). Poll up to ~30s.
    for _ in range(60):
        if not conv.is_processing:
            break
        await asyncio.sleep(0.5)
    if conv.is_processing:
        print(f"  [interrupt] resume aborted: conv {conv_id} still busy after 30s", flush=True)
        return
    # Deliver the redirect now — clear the stored flag so reply() doesn't
    # double-merge it, and clear any leftover freeze intent.
    conv.interrupt_message = ""
    conv.interrupt_requested = False
    print(f"  [interrupt] auto-resuming conv {conv_id} with redirect: {redirect[:60]}", flush=True)
    await run_agent_background(username, redirect, conv_id)


async def _perform_backend_restart():
    """Restart uvicorn api:app detached, strictly AFTER the patcher's turn has
    fully written + persisted its reply (never mid-turn). Called from the
    turn's finally via asyncio.create_task."""
    await asyncio.sleep(2)  # let the HTTP response flush to the frontend
    busy = any(not task.done() for task in _api_durable_writes)
    busy = busy or any(c.is_processing for u in session_manager.list_users() for c in session_manager.get_user(u).conversations.values())
    busy = busy or any(agent.get('busy_event_id') for u in session_manager.list_users()
        for c in session_manager.get_user(u).conversations.values() if c.session and c.session._lifecycle_store
        for agent in c.session._lifecycle_store.snapshot().get('agents', {}).values())
    busy = busy or any(r.futures and any(not f.done() for f in r.futures.values()) for r in _workflow_runtimes.values())
    if busy:
        from agents._restart import request
        request()
        print('[backend] administrator restart deferred: other user/session turns still active', flush=True)
        return
    managed_restart = os.environ.get('BIMEM_MANAGED_RESTART_FILE')
    if managed_restart:
        from agents.state_io import write_checkpoint
        write_checkpoint(Path(managed_restart), {'requested_at': time.time(), 'pid': os.getpid()})
        print('[backend] restart requested from the existing compute allocation supervisor', flush=True)
        return
    # Never pgrep/head an arbitrary uvicorn belonging to another workspace.
    pid, root = os.getpid(), str(Path(__file__).resolve().parent)
    helper = '''import os,sys,time,signal,subprocess
from pathlib import Path
pid=int(sys.argv[1]);root=Path(sys.argv[2]);proc=Path('/proc')/str(pid)
assert (proc/'cwd').resolve()==root
assert b'uvicorn' in (proc/'cmdline').read_bytes()
os.kill(pid,signal.SIGTERM)
for _ in range(150):
    if not proc.exists():break
    time.sleep(.1)
else:raise RuntimeError('old backend did not exit; refusing force kill/duplicate backend')
with (root/'logs/api_backend.log').open('ab') as log:
    child=subprocess.Popen([sys.executable,'-m','uvicorn','api:app','--host','0.0.0.0','--port','8000'],cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
from agents.state_io import write_checkpoint
write_checkpoint(root/'logs/api_backend_state.json',{'pid':child.pid,'root':str(root),'updated_at':time.time()})
'''
    try:
        subprocess.Popen([sys.executable, '-c', helper, str(pid), root], cwd=root,
            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        print("[backend] patcher 已请求重启，正在重启 uvicorn api:app ...", flush=True)
    except Exception as e:
        print(f"[backend] restart failed to spawn: {e}", flush=True)


# ── System endpoints ──
@app.get("/api/system/status")
async def system_status():
    users = session_manager.list_users()
    active = sum(1 for u in users for c in session_manager.get_user(u).conversations.values() if c.is_processing)
    return {
        "status": "healthy",
        "version": app.version,
        "total_users_registered": await _durable_io(registered_user_count),
        "active_sessions": len(users),
        "active_queries": active,
        "max_concurrent_llm": MODEL_CALL_CONCURRENCY,
        "model_provider": provider_state(),
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/admin/conversations")
async def admin_conversations(username: str = Depends(_get_admin), limit: int = 100):
    """View all conversations across all users (from persistent log)."""
    entries = []
    if CONVERSATION_LOG.exists():
        lines = CONVERSATION_LOG.read_text().strip().split("\n")
        for line in lines[-limit:]:
            try:
                entries.append(json.loads(line))
            except:
                pass
    active = []
    for u in session_manager.list_users():
        user = session_manager.get_user(u)
        for cid, conv in user.conversations.items():
            active.append({
                "username": u, "conv_id": cid, "title": conv.title,
                "is_processing": conv.is_processing,
                "messages_count": len(conv.messages_history),
            })
    return {"log_entries": entries, "active_conversations": active}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat(), "version": app.version}


# ── JobWatch: async job monitor + fault-notification drain ──────────
from agents.job_watch import get_watch as _get_watch

_watch = _get_watch()
if os.environ.get('BIMEM_BACKGROUND_MONITOR', '1') == '1':
    _watch.start()


_notification_backlog: List[Dict[str, Any]] = []
_lifecycle_tasks = set()
_workflow_runtimes = {}


def _ensure_workflow_runtime(conv):
    from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
    from agents.workspace import session_root
    from agents.task_line import get_store
    guard=conv.__dict__.setdefault('_api_runtime_binding_lock',threading.RLock())
    with guard:
        scope=(conv.username,conv.conv_id)
        main=_ensure_session(conv)
        if main is None:raise ValueError('session could not be restored')
        if conv.session is not main or (conv.username,conv.conv_id)!=scope:
            raise ValueError('runtime binding owner changed during hydration')
        runtime=_workflow_runtimes.get(scope)
        if runtime is None:
            prepared=ParallelWorkflow(main,session_root(conv.conv_id,conv.username),
                WorkflowStore(main.config.project_root/'data/state/parallel_workflows.json'),
                main._lifecycle_store,get_store(),conv.username,conv.conv_id,job_watch=_watch,
                on_change=lambda reason:_workflow_events.publish(conv.username,conv.conv_id))
            if conv.session is not main or (conv.username,conv.conv_id)!=scope:
                prepared.shutdown()
                raise ValueError('runtime binding Session changed during preparation')
            runtime=prepared
            _workflow_runtimes[scope]=runtime
        if conv.session is not main or (conv.username,conv.conv_id)!=scope:
            raise ValueError('runtime binding Session was replaced')
        runtime.main=main
        return runtime


class WorkflowExecutionRequest(BaseModel):
    plan_version: int


class LocalPreparationRequeueRequest(BaseModel):
    expected_token: str = Field(min_length=1)
    failed_call_id: str = Field(min_length=1)
    verification_call_id: Optional[str] = Field(default=None, min_length=1)


class UserReportRetryRequest(BaseModel):
    plan_version: int = Field(gt=0, strict=True)
    expected_failure_call_id: str = Field(min_length=1, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    expected_failure_token: str = Field(min_length=32, max_length=32, pattern=r'^[a-f0-9]{32}$')
    request_id: str = Field(min_length=16, max_length=128, pattern=r'^[A-Za-z0-9_-]+$')
    confirm_same_report_only: StrictBool
    model_config = {'extra':'forbid'}


class WorkflowProposalDiscardRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class NativeResultRevalidationRequest(BaseModel):
    expected_token: str = Field(min_length=1)


class NativeResultRejectionRequest(BaseModel):
    expected_token: str = Field(min_length=1)
    goal_version: int = Field(gt=0)
    plan_version: int = Field(gt=0)


class CDFTPartialPrepareRequest(BaseModel):
    timeout_seconds: int = Field(gt=0)
    walltime: str = Field(min_length=1, max_length=32)


class PausedWorkflowStageRequest(BaseModel):
    base_version: int = Field(gt=0)
    goal_version: int = Field(gt=0)
    reason: str = Field(min_length=5, max_length=2000)
    changes: List[Dict[str, Any]] = Field(min_length=1, max_length=128)

    model_config = {'extra': 'forbid'}


def _start_workflow_runtime(conv, plan_version, *, preserve_user_pause=False, control_valid=None):
    if control_valid is not None and not control_valid():
        raise ValueError('workflow control changed before runtime binding')
    runtime = _ensure_workflow_runtime(conv)
    if control_valid is not None and not control_valid():
        raise ValueError('workflow control changed during runtime binding')
    jobs = [job for job in _watch.list()
            if (job.get('username'), job.get('conv_id')) == (conv.username, conv.conv_id)]
    options = {'preserve_user_pause': True} if preserve_user_pause else {}
    if control_valid is not None: options['control_valid'] = control_valid
    return runtime.start_and_tick(plan_version, jobs=jobs,
                                  paused=bool(conv.interrupt_requested), **options)


class WorkerMessageRequest(BaseModel):
    text: str
    kind: str = 'comment'
    message_id: Optional[str] = None


@app.post('/api/conversations/{conv_id}/workflow/execute')
async def execute_parallel_workflow(conv_id: str, request: WorkflowExecutionRequest, username: str = Depends(_get_user)):
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing or conv.interrupt_requested: raise HTTPException(409, 'main turn busy or user paused; approve through main chat')
    if (conv.username, conv.conv_id) != (username, conv_id):
        raise HTTPException(403, 'conversation owner does not match execution request')
    # Reserve before the first await, including while waiting for the lease.
    conv.is_processing = True
    uid = conv.current_uid
    initial_actor = conv.session
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    cancelled = threading.Event()
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise ValueError('conversation control lease is busy')
        def initial_control():
            return (not cancelled.is_set() and conv.current_uid == uid and conv.is_processing
                    and not conv.interrupt_requested and not conv.interrupt_message
                    and conv.session is initial_actor)
        if not initial_control(): raise ValueError('execution control was superseded or paused')
        # Lazy session hydration also reads durable evidence. Keep it off-loop;
        # the reservation and lease remain owned until its writer has stopped.
        actor = await _durable_io(_owned_conversation_io, conv, _ensure_session, conv)
        if (cancelled.is_set() or conv.current_uid != uid or not conv.is_processing
                or conv.interrupt_requested or conv.interrupt_message
                or actor is None or conv.session is not actor
                or initial_actor is not None and actor is not initial_actor):
            raise ValueError('session or execution control changed during hydration')
        goal = copy.deepcopy(actor.goal_contract.to_dict())
        if goal.get('execution_authorized') is not True:
            raise HTTPException(403, 'current goal is not authorized for execution')
        if goal.get('approved_plan_version') != request.plan_version:
            raise ValueError('execute requires the exact approved structured plan version')
        def control_valid():
            return (not cancelled.is_set() and conv.session is actor and conv.current_uid == uid
                    and conv.is_processing and not conv.interrupt_requested and not conv.interrupt_message
                    and actor.goal_contract.to_dict() == goal)
        def start():
            if not control_valid(): raise ValueError('execution control changed before startup')
            return _owned_conversation_io(conv, _start_workflow_runtime, conv,
                request.plan_version, control_valid=control_valid)
        pending = asyncio.create_task(asyncio.to_thread(start))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try: result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled.set()
            try: await _finish_pending_io(pending)
            except BaseException: pass
            raise
        if not control_valid(): raise ValueError('execution control changed during startup')
        actor.task_complete = False
        conv.conversation_active = True
        def checkpoint():
            if not control_valid(): raise ValueError('execution control changed before checkpoint')
            actor._checkpoint('workflow_api_start')
        await _durable_io(_owned_conversation_io, conv, checkpoint)
        if not control_valid(): raise ValueError('execution control changed before receipt persistence')
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id,
                          completed_turn_uid=uid)
        if not control_valid(): raise ValueError('execution control changed before receipt delivery')
        return result
    except ValueError as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/requeue-local-preparation')
async def requeue_workflow_local_preparation(conv_id: str, step_id: str, request: LocalPreparationRequeueRequest,
                                            username: str = Depends(_get_user)):
    """An authenticated owner can request an evidence-backed, unchanged retry."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing or conv.interrupt_requested:
        raise HTTPException(409, 'main turn busy or user paused')
    # Reserve synchronously before yielding: another query cannot claim this
    # conversation while its control transaction is in progress.
    conv.is_processing = True
    saved_uid = conv.current_uid
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise HTTPException(409, 'conversation control lease is busy')
        if conv.current_uid != saved_uid or conv.interrupt_requested or conv.interrupt_message:
            raise HTTPException(409, 'conversation control was superseded or paused')
        runtime = _ensure_workflow_runtime(conv)
        session = conv.session
        goal = session.goal_contract
        if not goal.execution_authorized:
            raise HTTPException(403, 'current goal is not authorized for execution')
        version = goal.version
        source = {'origin': 'user', 'turn_id': 'api-' + uuid.uuid4().hex,
                  'owner': {'username': username, 'conv_id': conv_id}, 'goal_version': version}
        def control_valid():
            return (conv.session is session and conv.current_uid == saved_uid and conv.is_processing
                    and not conv.interrupt_requested and not conv.interrupt_message
                    and session.goal_contract.version == version and session.goal_contract.execution_authorized)
        def inspect_and_requeue():
            verification_call_id = request.verification_call_id
            if not verification_call_id:
                from agents.local_preparation_recovery import preparation_capability
                from agents.recovery import failure_reason
                state = runtime.snapshot()
                node = state.get('nodes', {}).get(step_id, {})
                contract = node.get('contract', {})
                if (not control_valid() or node.get('token') != request.expected_token
                        or node.get('status') != 'failed' or not preparation_capability(contract)):
                    raise ValueError('local preparation changed before native inspection')
                arguments = copy.deepcopy(node.get('effective_arguments') or contract['arguments'])
                parameters = {'tool_name': contract['tool'], 'arguments': arguments}
                observed = session.registry.execute_dict('inspect_runtime', parameters)
                if not control_valid():
                    raise ValueError('control was superseded during native inspection')
                session.memory.record_tool_call('operations', 'inspect_runtime', parameters,
                    json.dumps(observed, ensure_ascii=False, default=str), failed=bool(failure_reason(observed)))
                verification_call_id = session.memory.tool_call_log[-1]['call_id']
                session._checkpoint('external_local_preparation_inspected')
            return runtime.requeue_local_preparation(
                step_id, request.expected_token, request.failed_call_id, verification_call_id,
                control_source=source, authority_role='lead-orchestrator',
                operations_authority=None, control_valid=control_valid)
        pending = asyncio.create_task(asyncio.to_thread(
            inspect_and_requeue))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try:
            receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await _finish_pending_io(pending)
            raise
        conv.session._checkpoint('local_preparation_requeued_by_api')
        if conv.current_uid == saved_uid:
            conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return receipt
    except PermissionError as error: raise HTTPException(403, str(error))
    except ValueError as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == saved_uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/retry-report-once')
async def retry_owned_report_once(conv_id: str, step_id: str, request: UserReportRetryRequest,
                                  username: str = Depends(_get_user)):
    """An authenticated, explicit owner decision; never a model-granted retry."""
    if request.confirm_same_report_only is not True:raise HTTPException(400,'explicit report retry confirmation required')
    conv=session_manager.get_user(username).get_conversation(conv_id)
    if conv is None:raise HTTPException(404,'conversation not found')
    if (conv.username,conv.conv_id)!=(username,conv_id):raise HTTPException(403,'conversation owner mismatch')
    if conv.is_processing or conv.interrupt_requested or conv.interrupt_message:
        raise HTTPException(409,'main turn busy or user paused')
    from agents.workflow_user_control import advance_user_control,workflow_control_guard
    advance_user_control(conv)
    conv.is_processing=True;saved_uid=conv.current_uid;lock=_get_conv_lock(username,conv_id)
    acquired=False;cancelled=threading.Event()
    try:
        acquired=await _acquire_conversation_lease(lock,timeout=60)
        if not acquired:raise ValueError('conversation control lease is busy')
        runtime=await _durable_io(_owned_conversation_io,conv,_ensure_workflow_runtime,conv)
        actor=conv.session
        if actor is None or runtime.main is not actor:raise ValueError('owned actor changed')
        goal=copy.deepcopy(actor.goal_contract.to_dict())
        native_control=workflow_control_guard(actor,username,conv_id)
        def control():
            return (not cancelled.is_set() and conv.is_processing and conv.current_uid==saved_uid
                and conv.session is actor and not conv.interrupt_requested and not conv.interrupt_message
                and native_control() and actor.goal_contract.to_dict()==goal)
        if not control() or goal.get('execution_authorized') is not True or goal.get('approved_plan_version')!=request.plan_version:
            raise ValueError('user control or approved plan changed')
        lifecycle=actor._lifecycle_store.snapshot() if actor._lifecycle_store else {}
        if any(v.get('busy_event_id') for v in (lifecycle.get('agents')or{}).values()):
            raise ValueError('native receiver is busy; retry after it finishes')
        owner={'username':username,'conv_id':conv_id}
        authorization={'kind':'authenticated_user_report_retry','owner':owner,'step_id':step_id,
            'plan_version':request.plan_version,'failure_call':request.expected_failure_call_id,
            'failure_token':request.expected_failure_token,'request_id':request.request_id,
            'confirmed':request.confirm_same_report_only}
        def execute():
            from agents.report_bounded_retry import capture_report_retry
            state=runtime.snapshot()
            proof,packet=capture_report_retry(runtime,state,step_id,control_valid=control,
                return_packet=True,user_authorization=authorization)
            def revalidate(current):
                if not control():raise ValueError('user control changed before report retry commit')
                return capture_report_retry(runtime,current,step_id,control_valid=control,
                    prepared_packet=packet,user_authorization=authorization)
            receipt=runtime.store.requeue_bounded_report(runtime.workflow_id,step_id,
                request.expected_failure_token,goal,owner,proof,revalidate,control_valid=control)
            # Only the explicitly approved report node may be claimed by this
            # wake-up. Ordinary background scheduling remains independent.
            wake='claimed_requires_reconciliation'
            try:
                # Even if user control was revoked after commit, the targeted
                # dispatcher must settle a provably unstarted claim immediately.
                wake=runtime.dispatch_user_report_only(step_id,control_valid=control,proof=proof,packet=packet,ticket=receipt['user_claim_ticket'])
            except Exception:pass  # The durable receipt still resolves an interrupted wake.
            return receipt,wake
        pending=asyncio.create_task(asyncio.to_thread(_owned_conversation_io,conv,execute))
        _api_durable_writes.add(pending);pending.add_done_callback(_api_durable_writes.discard)
        try:receipt,wake=await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled.set()
            try:await _finish_pending_io(pending)
            except BaseException:pass
            raise
        _workflow_events.publish(username,conv_id)
        return {'ok':True,'status':wake,'step_id':step_id,
            'request_id':request.request_id,'proof_sha256':receipt['proof_sha256'],'wake_status':wake,
            'no_scientific_resubmission':True,'scientific_acceptance':False}
    except PermissionError as error:raise HTTPException(403,str(error)) from None
    except ValueError as error:raise HTTPException(409,str(error)) from None
    finally:
        if acquired:lock.release()
        if conv.current_uid==saved_uid:conv.is_processing=False


@app.get('/api/conversations/{conv_id}/workflow/report-retry-options')
async def get_user_report_retry_options(conv_id: str,username: str=Depends(_get_user)):
    """Advisory owner-scoped report failures; never grants or triggers retry."""
    conv=session_manager.get_user(username).get_conversation(conv_id)
    if conv is None or (conv.username,conv.conv_id)!=(username,conv_id):raise HTTPException(404,'conversation not found')
    from agents.parallel_workflow import WorkflowStore
    from agents.report_bounded_retry import _digest
    from agents.workflow_patch import execution_contract
    root=Path(__file__).resolve().parent
    owner={'username':username,'conv_id':conv_id}
    def inspect():
        data=WorkflowStore(root/'data/state/parallel_workflows.json').snapshot()
        matches=[w for w in (data.get('workflows')or{}).values()
            if isinstance(w,dict) and (w.get('username'),w.get('conv_id'))==(username,conv_id)]
        if len(matches)>1:raise ValueError('workflow owner is ambiguous')
        if not matches:return {'plan_version':None,'options':[],'advisory':True}
        state=matches[0];ledgers=data.get('user_report_retry_ledger') or {}
        if not isinstance(ledgers,dict):raise ValueError('user report retry ledger malformed')
        ledger=ledgers.get(_digest(owner),[])
        if not isinstance(ledger,list) or any(not isinstance(v,dict)for v in ledger):raise ValueError('user report retry ledger malformed')
        options=[]
        for step,node in (state.get('nodes')or{}).items():
            contract=node.get('contract')or{};args=contract.get('arguments')or{};ref=node.get('result_ref')or{}
            token=node.get('token');call=ref.get('call_id');output=args.get('output_path')
            if (node.get('status')!='failed' or contract.get('tool')!='generate_scientific_report'
                    or not isinstance(step,str) or not isinstance(token,str) or not isinstance(call,str)
                    or not isinstance(output,str)):continue
            path=Path(os.path.abspath(output));owned=root/'runs'/username/conv_id
            if not path.is_relative_to(owned) or path==owned:continue
            contract_key=_digest(execution_contract(contract));output_key=_digest({'owner':owner,'output':str(path)})
            used=any(v.get('contract_key')==contract_key or v.get('output_identity_sha256')==output_key for v in ledger)
            options.append({'step_id':step,'label':step,'plan_version':state.get('plan_version'),
                'expected_failure_call_id':call,'expected_failure_token':token,
                'retry_state':'used' if used else 'check_required'})
        return {'plan_version':state.get('plan_version'),'options':options,'advisory':True,
            'note':'资格、来源、用户控制和输出将在确认时重新验证；此列表不授权重试。'}
    try:return await _durable_io(inspect)
    except ValueError as error:raise HTTPException(409,str(error)) from None


@app.get('/api/conversations/{conv_id}/workflow/report-retry-receipts/{request_id}')
async def get_user_report_retry_receipt(conv_id: str,request_id: str,username: str=Depends(_get_user)):
    """Resolve an interrupted HTTP response without reissuing authorization."""
    import re
    if not re.fullmatch(r'[A-Za-z0-9_-]{16,128}',request_id):raise HTTPException(400,'invalid request id')
    conv=session_manager.get_user(username).get_conversation(conv_id)
    if conv is None or (conv.username,conv.conv_id)!=(username,conv_id):raise HTTPException(404,'conversation not found')
    from agents.parallel_workflow import WorkflowStore
    from agents.report_bounded_retry import _digest
    root=Path(__file__).resolve().parent
    def read():
        from agents.state_io import json_transaction
        with json_transaction(WorkflowStore(root/'data/state/parallel_workflows.json').path) as data:
            ledgers=data.get('user_report_retry_ledger') or {};owner={'username':username,'conv_id':conv_id}
            if not isinstance(ledgers,dict):raise ValueError('user retry ledger malformed')
            records=ledgers.get(_digest(owner),[]);nonce=_digest({'owner':owner,'request_id':request_id})
            if not isinstance(records,list) or any(not isinstance(v,dict)for v in records):raise ValueError('user retry ledger malformed')
            matches=[v for v in records if v.get('nonce_key')==nonce]
            if len(matches)>1:raise ValueError('user retry receipt ambiguous')
            return copy.deepcopy(matches[0]) if matches else None
    try:record=await _durable_io(read)
    except ValueError as error:raise HTTPException(409,str(error)) from None
    if record is None:raise HTTPException(404,'report retry receipt not found')
    return {'ok':True,'request_id':request_id,'status':record.get('dispatch_status','claimed'),
        'proof_sha256':record['proof_sha256'],'scientific_acceptance':False}


@app.post('/api/conversations/{conv_id}/workflow/proposals/{plan_version}/discard')
async def discard_workflow_proposal(conv_id: str, plan_version: int, request: WorkflowProposalDiscardRequest,
                                    username: str = Depends(_get_user)):
    """Withdraw one exact pending proposal without changing the approved DAG."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing or conv.interrupt_requested: raise HTTPException(409, 'main turn busy or user paused')
    conv.is_processing = True
    saved_uid = conv.current_uid
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise HTTPException(409, 'conversation control lease is busy')
        if conv.current_uid != saved_uid or conv.interrupt_requested or conv.interrupt_message:
            raise HTTPException(409, 'conversation control was superseded or paused')
        session = _ensure_session(conv)
        source = {'origin': 'user', 'turn_id': 'api-' + uuid.uuid4().hex,
                  'owner': {'username': username, 'conv_id': conv_id}}
        def withdraw():
            from agents.watch_context import set_context, clear_context
            if conv.session is not session or conv.current_uid != saved_uid or conv.interrupt_requested or conv.interrupt_message:
                raise ValueError('conversation control was superseded or paused')
            set_context(username, conv_id)
            try: return session._discard_workflow_patch_as_user(plan_version, request.reason,
                          control_source=source, owner={'username': username, 'conv_id': conv_id})
            finally: clear_context()
        pending = asyncio.create_task(asyncio.to_thread(withdraw))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try: receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await _finish_pending_io(pending)
            raise
        if conv.current_uid == saved_uid: conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return receipt
    except PermissionError as error: raise HTTPException(403, str(error))
    except ValueError as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == saved_uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/reject-result')
async def reject_native_workflow_result(conv_id: str, step_id: str, request: NativeResultRejectionRequest,
                                        username: str = Depends(_get_user)):
    """Reject only a freshly proven invalid owned terminal result; never resubmit."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if (conv.username, conv.conv_id) != (username, conv_id): raise HTTPException(403, 'conversation owner mismatch')
    if conv.is_processing: raise HTTPException(409, 'conversation is busy')
    conv.is_processing = True
    uid, initial_actor = conv.current_uid, conv.session
    interrupted = (conv.interrupt_requested, conv.interrupt_message)
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    cancelled = threading.Event()
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise ValueError('conversation control lease is busy')
        if (conv.current_uid != uid or not conv.is_processing
                or (conv.interrupt_requested, conv.interrupt_message) != interrupted):
            raise ValueError('rejection control changed before session hydration')
        if initial_actor is None:
            actor = await _durable_io(_owned_conversation_io, conv, _ensure_session, conv, publish=False)
            if (actor is None or conv.session is not None or conv.current_uid != uid or not conv.is_processing
                    or (conv.username, conv.conv_id) != (username, conv_id)
                    or (conv.interrupt_requested, conv.interrupt_message) != interrupted):
                raise ValueError('conversation control changed during session preparation')
            # The loop owns this no-await identity check and publication.
            conv.session = actor
            conv._api_session_binding = actor._api_ready_binding
            await _durable_io(_owned_conversation_io, conv, actor._api_after_bind)
        else:
            actor = initial_actor
        if (actor is None or conv.session is not actor or conv.current_uid != uid or not conv.is_processing
                or (conv.interrupt_requested, conv.interrupt_message) != interrupted):
            raise ValueError('conversation session or control changed during hydration')
        goal = copy.deepcopy(actor.goal_contract.to_dict())
        if (goal.get('execution_authorized') is not True or goal.get('version') != request.goal_version
                or goal.get('approved_plan_version') != request.plan_version):
            raise ValueError('approved goal or plan changed')
        runtime = _ensure_workflow_runtime(conv)
        initial_runtime = await asyncio.to_thread(runtime.snapshot)
        pause_revision = initial_runtime.get('user_pause_revision', 0)
        def control_valid():
            return (not cancelled.is_set() and conv.session is actor and conv.current_uid == uid and conv.is_processing
                    and (conv.interrupt_requested, conv.interrupt_message) == interrupted
                    and actor.goal_contract.to_dict() == goal
                    and runtime.snapshot().get('user_pause_revision', 0) == pause_revision)
        def reject():
            from agents.watch_context import set_context, clear_context
            set_context(username, conv_id, 'operations', actor._current_line_id)
            try:
                if not control_valid(): raise ValueError('rejection control changed')
                return runtime.reject_workflow_result(step_id, expected_token=request.expected_token,
                                                      control_valid=control_valid)
            finally: clear_context()
        pending = asyncio.create_task(asyncio.to_thread(reject))
        try:
            receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled.set()
            try: await _finish_pending_io(pending)
            except BaseException: pass
            raise
        if not control_valid(): raise ValueError('rejection control changed after commit')
        actor.memory.record_tool_call('operations', 'reject_workflow_result',
            {'step_id':step_id, 'expected_token':request.expected_token,
             'goal_version':request.goal_version, 'plan_version':request.plan_version},
            json.dumps(receipt, ensure_ascii=False, default=str), failed=False)
        await _durable_io(actor._checkpoint, 'native_result_rejected_by_api')
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id,
                          completed_turn_uid=uid)
        return receipt
    except PermissionError as error: raise HTTPException(403, str(error))
    except (KeyError, ValueError) as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/revalidate-gcmc')
async def revalidate_gcmc_native_outputs(conv_id: str, step_id: str, request: NativeResultRevalidationRequest,
                                         username: str = Depends(_get_user)):
    """Review current owned GCMC artifacts; the server chooses native paths."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing or conv.interrupt_requested: raise HTTPException(409, 'main turn busy or user paused')
    conv.is_processing = True
    saved_uid = conv.current_uid
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise HTTPException(409, 'conversation control lease is busy')
        if conv.current_uid != saved_uid or conv.interrupt_requested or conv.interrupt_message:
            raise HTTPException(409, 'conversation control was superseded or paused')
        runtime = _ensure_workflow_runtime(conv)
        session = conv.session
        version = session.goal_contract.version
        if not session.goal_contract.execution_authorized:
            raise HTTPException(403, 'current goal is not authorized for execution')
        def control_valid():
            return (conv.session is session and conv.current_uid == saved_uid and conv.is_processing
                    and not conv.interrupt_requested and not conv.interrupt_message
                    and session.goal_contract.version == version and session.goal_contract.execution_authorized)
        def review():
            from agents.watch_context import set_context, clear_context
            set_context(username, conv_id)
            try:
                node = runtime.snapshot().get('nodes', {}).get(step_id, {})
                if (node.get('contract', {}).get('tool') not in {'run_gcmc_isotherm', 'run_gcmc_batch'}
                        or node.get('token') != request.expected_token or not control_valid()):
                    raise ValueError('current owned GCMC attempt does not match the request')
                return runtime.revalidate_outputs(step_id, None, expected_token=request.expected_token,
                                                  control_valid=control_valid)
            finally: clear_context()
        pending = asyncio.create_task(asyncio.to_thread(review))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try: receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await _finish_pending_io(pending)
            raise
        session._checkpoint('gcmc_native_outputs_revalidated_by_api')
        if conv.current_uid == saved_uid: conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return receipt
    except PermissionError as error: raise HTTPException(403, str(error))
    except (KeyError, ValueError) as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == saved_uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/prepare-cdft-partial')
async def prepare_cdft_partial_retry(conv_id: str, step_id: str, request: CDFTPartialPrepareRequest,
                                    username: str = Depends(_get_user)):
    """Prepare an owned evidenced failed-only subset; no scientific job starts."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing: raise HTTPException(409, 'main turn busy')
    conv.is_processing = True
    saved_uid = conv.current_uid
    saved_interrupt = (conv.interrupt_requested, conv.interrupt_message)
    lock = _get_conv_lock(username, conv_id)
    acquired = False
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise HTTPException(409, 'conversation control lease is busy')
        if conv.current_uid != saved_uid or (conv.interrupt_requested, conv.interrupt_message) != saved_interrupt:
            raise HTTPException(409, 'conversation control was superseded')
        runtime = _ensure_workflow_runtime(conv)
        session = conv.session
        version = session.goal_contract.version
        if not session.goal_contract.execution_authorized:
            raise HTTPException(403, 'current goal is not authorized for execution')
        source = {'origin': 'user', 'turn_id': 'api-' + uuid.uuid4().hex,
                  'owner': {'username': username, 'conv_id': conv_id}, 'goal_version': version}
        def control_valid():
            return (conv.session is session and conv.current_uid == saved_uid and conv.is_processing
                    and (conv.interrupt_requested, conv.interrupt_message) == saved_interrupt
                    and session.goal_contract.version == version and session.goal_contract.execution_authorized)
        pending = asyncio.create_task(asyncio.to_thread(runtime.prepare_cdft_partial_recovery,
            step_id, request.timeout_seconds, request.walltime, control_source=source,
            authority_role='lead-orchestrator', operations_authority=None, control_valid=control_valid,
            allow_paused=True))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try: receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            await _finish_pending_io(pending)
            raise
        if not control_valid(): raise ValueError('control changed before partial recovery receipt was recorded')
        from agents.recovery import failure_reason
        session.memory.record_tool_call('operations', 'prepare_cdft_partial_recovery',
            {'step_id': step_id, 'timeout_seconds': request.timeout_seconds, 'walltime': request.walltime},
            json.dumps(receipt, ensure_ascii=False, default=str), failed=bool(failure_reason(receipt)))
        session._checkpoint('cdft_partial_recovery_prepared_by_api')
        if conv.current_uid == saved_uid: conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return receipt
    except PermissionError as error: raise HTTPException(403, str(error))
    except (KeyError, ValueError) as error: raise HTTPException(409, str(error))
    finally:
        if acquired: lock.release()
        if conv.current_uid == saved_uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/stage-paused-plan')
async def stage_paused_workflow_plan(conv_id: str, request: PausedWorkflowStageRequest,
                                     username: str = Depends(_get_user)):
    """Compile and stage an explicitly supplied owned plan; keep all execution paused.

    TaskLine and runtime are separate durable files. A partial commit is a
    recoverable inconsistency, never reported as an atomic success or dispatched.
    """
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing: raise HTTPException(409, 'main turn is busy')
    if len(json.dumps(request.changes, ensure_ascii=False).encode()) > 2*1024*1024:
        raise HTTPException(422, 'plan changes exceed bounded control request size')
    actor = conv.session
    if actor is None or not actor.goal_contract.execution_authorized:
        raise HTTPException(403, 'an existing authorized execution goal is required')
    conv.is_processing = True
    uid = conv.current_uid
    interrupt = (conv.interrupt_requested, conv.interrupt_message)
    lock = _get_conv_lock(username, conv_id)
    acquired = False; cancelled = threading.Event()
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise HTTPException(409, 'conversation control lease is busy')
        runtime = _ensure_workflow_runtime(conv)
        initial = await asyncio.to_thread(runtime.snapshot)
        pause = (bool(initial.get('user_paused')), initial.get('user_pause_revision', 0))
        goal = actor.goal_contract.to_dict()
        allowed_plans = {request.base_version}
        if (not pause[0] or initial.get('plan_version') != request.base_version
                or goal.get('version') != request.goal_version
                or goal.get('approved_plan_version') != request.base_version
                or (initial.get('username'), initial.get('conv_id')) != (username, conv_id)):
            raise HTTPException(409, 'paused native runtime and requested approved goal/plan do not match')
        if actor._pending_workflow_patch:
            raise HTTPException(409, 'resolve the exact existing pending proposal before staging another plan')

        def control_valid():
            if (cancelled.is_set() or conv.session is not actor or conv.current_uid != uid or not conv.is_processing
                    or (conv.interrupt_requested, conv.interrupt_message) != interrupt
                    or not actor.goal_contract.execution_authorized
                    or actor.goal_contract.version != request.goal_version): return False
            current = runtime.snapshot()
            return ((current.get('username'), current.get('conv_id')) == (username, conv_id)
                and (bool(current.get('user_paused')), current.get('user_pause_revision', 0)) == pause
                and current.get('plan_version') in allowed_plans)

        def stage():
            from agents.watch_context import set_context, clear_context
            from agents.task_line import get_store
            from agents.control_policy import issue_operations_authority
            from agents.defns import OPERATIONS
            from agents.workflow_patch import execution_contract
            from agents.recovery import is_submission
            from agents.answer_delivery import digest
            from agents.workflow_audit import verified_job_observation
            set_context(username, conv_id, 'operations', actor._current_line_id)
            saved_agent = actor.current_agent
            saved_source = copy.deepcopy(getattr(actor, '_control_source', None))
            saved_authority = copy.deepcopy(getattr(actor, '_operations_authority', None))
            source = {'origin': 'user', 'turn_id': 'api-' + uuid.uuid4().hex,
                      'owner': {'username': username, 'conv_id': conv_id}, 'goal_version': request.goal_version,
                      'api_operation': 'stage_paused_plan', 'requested_changes_sha256': digest(request.changes)}
            try:
                if not control_valid(): raise ValueError('paused plan control changed before compilation')
                line = get_store().get_line(actor._current_line_id, username=username, conv_id=conv_id) or {}
                if (line.get('plan_version') != request.base_version
                        or any(item.get('job_ids') and item.get('step_id') not in initial.get('nodes', {})
                               and not verified_job_observation(actor,runtime,initial,item)
                               for item in line.get('steps', []))):
                    raise ValueError('only the complete current native plan may use paused staging; legacy jobs require reconciliation')
                approved = {item.get('step_id'): item for item in goal.get('approved_nodes', [])}
                if not runtime.approved_contracts_match(initial, goal.get('approved_nodes', [])):
                    raise ValueError('approved goal and current native executable contracts differ')
                if not control_valid() or actor.goal_contract.to_dict() != goal:
                    raise ValueError('paused plan control changed during native contract proof')
                actor.current_agent = OPERATIONS
                actor._control_source = source
                actor._operations_authority = issue_operations_authority(source, source['owner'], request.goal_version)
                previous_review = actor.context.get('resource_review')
                proposal = actor._propose_workflow_patch(request.base_version, request.reason,
                    copy.deepcopy(request.changes), control_valid=control_valid)
                if not control_valid(): raise ValueError('paused plan control changed after compilation')
                patch = proposal.get('patch') if isinstance(proposal, dict) else None
                if (not isinstance(proposal, dict) or proposal.get('ok') is not True or proposal.get('applied')
                        or not isinstance(patch, dict) or patch != actor._pending_workflow_patch
                        or patch.get('base_version') != request.base_version or patch.get('goal_version') != request.goal_version
                        or type(patch.get('new_version')) is not int or patch['new_version'] <= request.base_version):
                    raise ValueError('compiler did not return the exact current unapplied proposal')
                submissions = [item['node'] for item in patch.get('changes', [])
                    if item.get('operation') == 'upsert' and item.get('node')
                    and is_submission(item['node'].get('tool'), item['node'].get('arguments', {}))]
                if submissions:
                    review = actor.context.get('resource_review')
                    if not isinstance(review, dict) or review is previous_review:
                        callback = getattr(actor, '_on_resource_review', None)
                        if not callback: raise ValueError('fresh resource review is required before staging submission changes')
                        review = callback({'kind': 'before_submission', 'review_phase': 'planning',
                                           'nodes': submissions, 'goal_version': request.goal_version})
                    if (not isinstance(review, dict) or review.get('status') != 'ready'
                            or any(review.get('resource_allocations', {}).get(item['step_id'], {}).get('budget_final') is not True
                                   for item in submissions)):
                        raise ValueError('resource review is not ready with final workload-backed budgets')
                if not control_valid(): raise ValueError('paused plan control changed after resource review')
                allowed_plans.add(patch['new_version'])
                actor.memory.record_tool_call('operations', 'propose_workflow_patch',
                    {'base_version': request.base_version, 'reason': request.reason, 'changes': request.changes},
                    json.dumps(proposal, ensure_ascii=False, default=str), failed=False)
                applied = actor._apply_workflow_patch(patch['new_version'], preserve_user_pause=True, control_valid=control_valid)
                if not control_valid(): raise ValueError('paused plan control changed while staging durable files')
                current = runtime.snapshot()
                latest = get_store().get_line(actor._current_line_id, username=username, conv_id=conv_id) or {}
                final_goal = actor.goal_contract.to_dict()
                approved_final = {item['step_id']: item for item in final_goal.get('approved_nodes', [])}
                if (not isinstance(applied, dict) or applied.get('ok') is not True or applied.get('applied') is not True
                        or not applied.get('execution', {}).get('plan_staged')
                        or applied.get('execution', {}).get('scheduled') is not False
                        or {current.get('plan_version'), latest.get('plan_version'), final_goal.get('approved_plan_version')} != {patch['new_version']}
                        or any(item.get('job_ids')and item.get('step_id')not in current.get('nodes',{})
                               and not verified_job_observation(actor,runtime,current,item)
                               for item in latest.get('steps',[]))
                        or not runtime.approved_contracts_match(current, final_goal.get('approved_nodes', []))):
                    raise ValueError('plan files are not consistently staged; retain pause and reconcile approved runtime')
                if not control_valid() or actor.goal_contract.to_dict() != final_goal:
                    raise ValueError('paused plan control changed during final native contract proof')
                actor.memory.record_tool_call('operations', 'apply_workflow_patch', {'plan_version': patch['new_version']},
                    json.dumps(applied, ensure_ascii=False, default=str), failed=False)
                # Persist the conversation's prior role/source, not this API's
                # temporary controller identity. Approval proof and tool audit
                # retain the actual authenticated request independently.
                actor.current_agent = saved_agent
                actor._control_source = saved_source
                actor._operations_authority = saved_authority
                actor._checkpoint('owner_api_paused_plan_staged')
                if not control_valid(): raise ValueError('paused plan control changed after final checkpoint')
                return {'ok': True, 'status': 'staged_paused', 'plan_version': patch['new_version'],
                        'dispatch_started': False, 'user_paused': True, 'proposal': patch,
                        'cross_file_atomic_transaction': False}
            finally:
                actor.current_agent = saved_agent
                actor._control_source = saved_source
                actor._operations_authority = saved_authority
                clear_context()

        pending = asyncio.create_task(asyncio.to_thread(stage))
        _api_durable_writes.add(pending)
        pending.add_done_callback(_api_durable_writes.discard)
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled.set()
            try: await _finish_pending_io(pending)
            except Exception: pass
            raise
        if conv.current_uid == uid:
            conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return result
    except PermissionError as error:
        raise HTTPException(403, str(error))
    except (ValueError, KeyError) as error:
        raise HTTPException(409, {'status': 'not_staged_or_recovery_required', 'recovery_required': True,
                                  'dispatch_started': False, 'reason': str(error)})
    finally:
        if acquired: lock.release()
        if conv.current_uid == uid: conv.is_processing = False


class RestoreApprovedPausedPlanRequest(BaseModel):
    plan_version: int = Field(gt=0)
    goal_version: int = Field(gt=0)
    model_config = {'extra': 'forbid'}


@app.post('/api/conversations/{conv_id}/workflow/restore-approved-paused-plan')
async def restore_approved_paused_plan(conv_id: str, request: RestoreApprovedPausedPlanRequest,
                                       username: str = Depends(_get_user)):
    """Restore an already durable approval, keeping execution paused throughout."""
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing: raise HTTPException(409, 'main turn is busy')
    actor = conv.session
    if actor is None: raise HTTPException(409, 'owned session is not loaded')
    conv.is_processing = True
    uid = conv.current_uid; interrupt = (conv.interrupt_requested, conv.interrupt_message)
    lock = _get_conv_lock(username, conv_id); acquired = False; cancelled = threading.Event()
    try:
        acquired = await _acquire_conversation_lease(lock, timeout=60)
        if not acquired: raise ValueError('conversation control lease is busy')
        runtime = _ensure_workflow_runtime(conv)
        goal = copy.deepcopy(actor.goal_contract.to_dict())
        proof = copy.deepcopy(actor.context.get('approved_execution_proof'))
        state = await asyncio.to_thread(runtime.snapshot)
        owner = {'username': username, 'conv_id': conv_id}
        pause = (bool(state.get('user_paused')), state.get('user_pause_revision', 0))
        plans = {state.get('plan_version'), request.plan_version}

        def control_valid():
            if (cancelled.is_set() or conv.session is not actor or conv.current_uid != uid or not conv.is_processing
                    or (conv.interrupt_requested, conv.interrupt_message) != interrupt
                    or actor.goal_contract.to_dict() != goal or actor.context.get('approved_execution_proof') != proof
                    or actor._pending_workflow_patch or actor._awaiting_plan_approval): return False
            visible = runtime.snapshot()
            return ((visible.get('username'), visible.get('conv_id')) == (username, conv_id)
                and (bool(visible.get('user_paused')), visible.get('user_pause_revision', 0)) == pause
                and visible.get('plan_version') in plans)

        def restore():
            from agents.watch_context import set_context, get_context
            from agents.task_line import get_store
            from agents.parallel_workflow import _approval_digest
            from agents.workflow_patch import execution_contract
            previous = get_context(); set_context(username, conv_id, 'operations', actor._current_line_id)
            try:
                if not control_valid(): raise ValueError('recovery control was superseded')
                if (goal.get('execution_authorized') is not True or goal.get('version') != request.goal_version
                        or goal.get('approved_plan_version') != request.plan_version or not pause[0]
                        or type(state.get('plan_version')) is not int or state['plan_version'] > request.plan_version):
                    raise ValueError('recovery requires the exact saved authorized plan and a paused owned runtime')
                source = proof.get('control_source', {}) if isinstance(proof, dict) else {}
                if (not isinstance(proof, dict) or proof.get('schema_version') != 1
                        or proof.get('kind') != 'main_model_applied_plan' or proof.get('owner') != owner
                        or proof.get('plan_version') != request.plan_version or proof.get('goal_version') != request.goal_version
                        or proof.get('approval_sha256') != _approval_digest(goal)
                        or type(proof.get('base_plan_version')) is not int
                        or not 0 <= proof['base_plan_version'] < request.plan_version
                        or source.get('origin') != 'user' or source.get('owner') != owner
                        or source.get('goal_version') != request.goal_version or not source.get('turn_id')):
                    raise PermissionError('saved approval has no matching authenticated native user provenance')
                protected = ('method', 'gases', 'parameters', 'execution_mode', 'execution_authorized')
                saved = state.get('goal_contract', {})
                if (any(saved.get(key) != goal.get(key) for key in protected)
                        or type(saved.get('version')) is not int or saved['version'] > goal['version']):
                    raise ValueError('protected scientific goal changed; use a new formal plan')
                approved_list = goal.get('approved_nodes') or []
                approved = {item['step_id']: execution_contract(item) for item in approved_list}
                line = get_store().get_line(actor._current_line_id, username=username, conv_id=conv_id) or {}
                executable = [item for item in line.get('steps', []) if item.get('arguments') is not None and 'expected_outputs' in item]
                if (not approved or len(approved) != len(approved_list) or line.get('plan_version') != request.plan_version
                        or len(executable) != len(approved)
                        or {item['step_id']: execution_contract(item) for item in executable} != approved):
                    raise ValueError('saved TaskLine executable contracts do not match the approved graph')
                if not control_valid(): raise ValueError('recovery control changed before native activation')
                result = runtime.start_and_tick(request.plan_version, paused=True,
                    preserve_user_pause=True, control_valid=control_valid)
                if not control_valid(): raise ValueError('recovery control changed during native activation')
                current = runtime.snapshot()
                latest = get_store().get_line(actor._current_line_id, username=username, conv_id=conv_id) or {}
                if (not result.get('plan_staged') or result.get('scheduled') is not False
                        or result.get('dispatch_started') is not False or current.get('plan_version') != request.plan_version
                        or latest.get('plan_version') != request.plan_version
                        or not runtime.approved_contracts_match(current, approved_list)
                        or {item['step_id']: execution_contract(item) for item in latest.get('steps', [])
                            if item.get('arguments') is not None and 'expected_outputs' in item} != approved):
                    raise ValueError('approved runtime graph was not consistently restored; keep pause and reconcile')
                receipt = {'ok': True, 'status': 'staged_paused', 'plan_version': request.plan_version,
                    'user_paused': True, 'dispatch_started': False, 'cross_file_atomic_transaction': False,
                    'needs_recovery': bool(result.get('needs_recovery')), 'blocked_nodes': result.get('blocked_nodes', [])}
                actor.memory.record_tool_call('operations', 'restore_approved_paused_plan',
                    request.model_dump(), json.dumps(receipt), failed=False)
                actor._checkpoint('owner_restored_saved_approved_plan_paused')
                if not control_valid(): raise ValueError('recovery control changed before receipt delivery')
                return receipt
            finally: set_context(**previous)

        pending = asyncio.create_task(asyncio.to_thread(restore))
        _api_durable_writes.add(pending); pending.add_done_callback(_api_durable_writes.discard)
        try: result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled.set()
            try: await _finish_pending_io(pending)
            except Exception: pass
            raise
        if conv.current_uid == uid: conv.is_processing = False
        await _durable_io(session_manager.persist_conversations, username, conv_id=conv_id)
        return result
    except PermissionError as error: raise HTTPException(403, str(error))
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(409, {'status': 'recovery_required', 'reason': str(error),
            'dispatch_started': False, 'cross_file_atomic_transaction': False})
    finally:
        if acquired: lock.release()
        if conv.current_uid == uid: conv.is_processing = False


@app.post('/api/conversations/{conv_id}/workflow/nodes/{step_id}/messages')
async def message_parallel_worker(conv_id: str, step_id: str, request: WorkerMessageRequest, username: str = Depends(_get_user)):
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    try:
        return _ensure_workflow_runtime(conv).send(step_id, request.text, request.kind, request.message_id)
    except KeyError: raise HTTPException(404, 'runtime node not found')
    except ValueError as error: raise HTTPException(409, str(error))


def _ensure_supervisor(conv):
    from agents.session import Session
    from agents.goal_contract import GoalContract
    from agents.workspace import session_root
    from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
    from agents.defns import SUPERVISOR
    main = _ensure_session(conv)
    root = session_root(conv.conv_id, conv.username)
    path = root / 'supervisor_checkpoint.json'
    if conv.supervisor_session is None:
        observer = Session(config=main.config, registry=main.registry)
        if path.exists():
            saved = json.loads(path.read_text())
            if saved.get('owner') and saved['owner'] != {'username': conv.username, 'conv_id': conv.conv_id, 'role': 'supervisor'}:
                raise PermissionError('supervisor checkpoint belongs to a different user/session')
            observer.import_state(saved)
        conv.supervisor_session = observer
    observer = conv.supervisor_session
    observer.current_agent = SUPERVISOR
    observer.goal_contract = GoalContract.from_dict(main.goal_contract.to_dict())
    observer._current_line_id = main._current_line_id or conv.conv_id
    observer.recovery_gate.path = root / 'recovery_state.json'
    observer._evidence_root = root / 'evidence'
    observer._lifecycle_store = main._lifecycle_store
    observer._runtime_snapshot = main._runtime_snapshot
    observer._on_display_request=getattr(main,'_on_display_request',None)
    observer._chain_state_is_primary = False
    observer._chain_jobs = main._chain_jobs
    # A user freeze covers the full-lifecycle pair, not only main chat.  The
    # observer stops at the next model/tool boundary and its durable event is
    # retried after resume; it must not keep draining a backlog during deploy.
    observer._on_interrupt_requested = lambda c=conv: bool(getattr(c, 'interrupt_requested', False))
    observer._on_interrupt_message = lambda c=conv: str(getattr(c, 'interrupt_message', '') or '')
    observer.context['main_evidence_calls'] = main._recovery_evidence_calls()
    def save(state, reason):
        state['owner'] = {'username': conv.username, 'conv_id': conv.conv_id, 'role': 'supervisor'}
        references = compact_checkpoint_evidence(state, root / 'evidence')
        write_checkpoint(path, state)
        apply_evidence_references(observer.memory, references)
    observer._on_checkpoint = save
    return observer


def _review_resources(conv, request):
    """Delegate a bounded read-only resource Session, persist and share facts."""
    from agents.session import Session
    from agents.resource_review import run_resource_review
    from agents.workspace import session_root
    from agents.state_io import ConversationLock, write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
    from agents.watch_context import get_context, set_context
    import copy
    root = session_root(conv.conv_id,conv.username)
    lock=ConversationLock(root/'resource_review.lock')
    if not lock.acquire(blocking=False):
        return {'status': 'review_busy', 'review_in_progress': True, 'retryable': True,
                'retry_after_seconds': 5,
                'reason': 'resource agent is already reviewing this session; wait for that review without consuming a failed-review attempt'}
    previous=get_context()
    try:
        return _review_resources_locked(conv, request, root, copy.deepcopy(conv.session.goal_contract))
    finally:
        set_context(**previous)
        lock.release()


def _review_resources_locked(conv, request, root, reviewed_goal):
    from agents.session import Session
    from agents.resource_review import run_resource_review, allocate_reviewed_resources
    from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
    from agents.watch_context import set_context
    from agents.workload_budget import _digest
    from agents.resource_intent import resource_intent
    from agents.node_inventory import load_policy
    import copy
    policy = load_policy(conv.session.config.project_root)
    if request.get('kind') == 'before_submission':
        from agents.workload_context import enrich_resource_request
        from agents.installations import cdft_installation_files
        executable = None
        if any(n.get('tool') == 'run_cdft' and n.get('arguments', {}).get('action') == 'submit'
               for n in request.get('nodes', [])):
            try:
                executable = cdft_installation_files(conv.session.config.project_root,
                    {'action': 'submit'})['cdft_executable'][0]
            except (KeyError, ValueError, OSError):
                pass  # Missing installation remains an explicit unverified budget.
        request = enrich_resource_request(request, owned_root=root, cdft_executable=executable)
        # Request labels do not establish user/operator authority. The server
        # supplies authenticated constraints separately from model arguments.
        request.pop('resource_constraints', None)
        request.pop('resource_pins', None)
        request.update(resource_intent(reviewed_goal, request.get('nodes', [])))
    set_context(conv.username,conv.conv_id,'monitor',conv.session._current_line_id)
    observer=Session(config=conv.session.config,registry=conv.session.registry)
    observer.client=observer.client.with_options(timeout=60,max_retries=0)
    observer.goal_contract=copy.deepcopy(reviewed_goal)
    observer._evidence_root=root/'evidence'
    def checkpoint(state, reason):
        state['owner']={'username':conv.username,'conv_id':conv.conv_id,'role':'resource_monitor'}
        refs=compact_checkpoint_evidence(state,root/'evidence')
        write_checkpoint(root/'resource_monitor_checkpoint.json',state)
        apply_evidence_references(observer.memory,refs)
    observer._on_checkpoint=checkpoint
    try: receipt=run_resource_review(observer,request)
    except Exception as error: receipt={'status':'review_retry','retryable':True,'reason':f'resource agent unavailable: {error}','not_resubmitted':True}
    if request.get('kind')=='before_submission':
        receipt = allocate_reviewed_resources(observer, request, receipt)
        for node in request.get('nodes', []):
            allocation = receipt.get('resource_allocations', {}).get(node.get('step_id'), {})
            if allocation.get('budget_final') is True:
                allocation.update(goal_version=reviewed_goal.version,
                    operator_policy_sha256=_digest(policy),
                    resource_intent_sha256=_digest(resource_intent(reviewed_goal, [node])))
    if (conv.session.goal_contract.to_dict() != reviewed_goal.to_dict()
            or load_policy(conv.session.config.project_root) != policy):
        receipt = {**receipt, 'status':'review_retry', 'retryable':True,
            'reason':'Goal or operator resource policy changed during review; reassess the current request.',
            'resource_allocations':{}}
    write_checkpoint(root/'resource_review.json',receipt)
    known={c.get('call_id') for c in conv.session.memory.tool_call_log}
    for call in receipt.get('evidence_calls',[]):
        if call.get('call_id') not in known:conv.session.memory.tool_call_log.append(copy.deepcopy(call))
    conv.session._checkpoint('resource_monitor_receipt')
    return receipt


async def _supervise_lifecycle_event(conv, event):
    store = conv.session._lifecycle_store
    recovery = event.get('payload', {}).get('failure_recovery') or {}
    if event.get('kind') == 'worker_failure_recovery' and recovery.get('stage') in {'diagnosis', 'repair'}:
        # A pending scientific question must not suppress read-only diagnosis.
        # Do not issue a provisional main-chat receipt: the diagnostic stage
        # owns its event until native evidence has been committed.
        from agents.failure_recovery import FailureRecovery
        actor = conv.session
        turn_uid = conv.current_uid
        goal = copy.deepcopy(actor.goal_contract.to_dict())
        controller = FailureRecovery(_ensure_workflow_runtime(conv))
        def valid():
            return (conv.session is actor and conv.current_uid == turn_uid
                    and not conv.interrupt_requested and not conv.interrupt_message
                    and actor.goal_contract.to_dict() == goal)
        cancelled = False
        try:
            if not valid() or not await asyncio.to_thread(controller.current_event, event):
                await _durable_io(store.receipt, event['event_id'], 'supervisor', 'obsolete',
                                  {'reason': 'failure diagnosis superseded', 'not_resubmitted': True})
                await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete',
                                  {'reason': 'failure diagnosis superseded', 'not_resubmitted': True})
                return
            observer = _ensure_supervisor(conv)
            observer._on_interrupt_requested = lambda: not valid()
            run_stage = controller.run_diagnosis if recovery['stage'] == 'diagnosis' else controller.run_repair
            pending = asyncio.create_task(asyncio.to_thread(run_stage, observer,
                recovery.get('incident_id'), recovery.get('revision'),
                checkpoint_path=observer._evidence_root.parent / 'supervisor_checkpoint.json',
                control_valid=valid))
            try:
                result = await asyncio.shield(pending)
            except asyncio.CancelledError:
                cancelled = True
                result = await _finish_pending_io(pending)
            await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', result)
            await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete',
                              {'reason': 'recovery stage completed; next stage owns continuation',
                               'not_resubmitted': True})
            if cancelled:
                raise asyncio.CancelledError
        except Exception as error:
            still_current = valid() and await asyncio.to_thread(controller.current_event, event)
            status = 'retry' if still_current else 'obsolete'
            await _durable_io(store.receipt, event['event_id'], 'supervisor', status,
                              {'reason': 'native diagnostic stage did not complete',
                               'error_type': type(error).__name__, 'not_resubmitted': True})
            if not still_current:
                await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete',
                                  {'reason': 'diagnostic stage superseded', 'not_resubmitted': True})
            if cancelled:
                raise asyncio.CancelledError
        return
    action = ('ask_user' if event.get('payload', {}).get('requires_user') else
              'advance_dependencies' if event['kind'] in {'job_completed', 'result_verified'} else
              'wait_existing' if event['kind'] in {'tool_uncertain', 'stalled', 'user_paused', 'waiting_jobs'} else 'diagnose_and_fix')
    provisional = {'next_action': action, 'reason': '控制层只允许诊断/等终态/推进依赖；详细只读审核异步进行，不阻塞主chat推进',
                   'evidence_refs': [event['event_id']]}
    (await _durable_io(store.provisional, event['event_id'], provisional))
    try:
        def observe():
            from agents.watch_context import set_context, clear_context
            set_context(conv.username, conv.conv_id, 'supervisor', conv.session._current_line_id)
            try:
                if event['kind'] in {'job_pending_warning','job_completed'}:
                    resource_receipt=_review_resources(conv,{'kind':event['kind'],'job_id':event['payload']['job']['job_id'],
                        'pending_reason':event['payload']['notification'].get('pending_reason'),
                        'instruction':'Review actual exit/return state for a completed job, or constraints for a pending job; only report facts, never cancel/update/resubmit. Result acceptance belongs to the scientific owner.'})
                    event['payload']['resource_review']=resource_receipt
                    event['payload']['requires_user']=resource_receipt.get('status') == 'needs_user'
                observer = _ensure_supervisor(conv)
                if event['kind'] == 'scientific_review':
                    return conv.session.audit_scientific_answer(observer, event)
                return observer.observe_lifecycle_event(event)
            finally:
                clear_context()
        receipt = await asyncio.to_thread(observe)
        if event['kind']=='job_pending_warning':
            resource_receipt=event['payload'].get('resource_review',{})
            receipt['resource_review']=resource_receipt
            receipt['not_resubmitted']=True
            if resource_receipt.get('status') == 'propose_resource_patch':
                receipt.update(next_action='diagnose_and_fix',
                    reason=resource_receipt.get('reason','apply the evidence-backed resource patch to the same queued job'))
                conv.session.context['resource_review']={**resource_receipt,'job_id':event['payload']['job']['job_id']}
            elif resource_receipt.get('status') == 'needs_user':
                receipt.update(next_action='ask_user',reason=resource_receipt.get('reason','resource review requires user decision'))
        (await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', receipt))
        if event['kind'] == 'scientific_review':
            action = receipt.get('review_action')
            if action in {'verified', 'obsolete'}:
                (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'delivered' if action == 'verified' else 'obsolete',
                              {'status': action, 'delivery_owner': 'lead-orchestrator', 'not_resubmitted': True}))
            elif action in {'correct_main', 'ask_user'}:
                payload = {**event['payload'], 'main_already_handling': False, 'supervisor_decision': receipt,
                           'requires_user': action == 'ask_user'}
                review_id = (await _durable_io(store.emit, 'scientific_review_failed', payload, event_id=event['event_id'] + ':feedback'))
                (await _durable_io(store.receipt, review_id, 'supervisor', 'delivered', receipt))
                (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'delivered', {'status':'forwarded_to_main', 'delivery_owner':'lead-orchestrator'}))
            (await _durable_io(session_manager.persist_conversations, conv.username, conv_id=conv.conv_id))
            return
        latest = (await asyncio.to_thread(store.snapshot))['events'][event['event_id']]
        if (event['kind'] not in {'user_paused', 'stalled', 'waiting_jobs', 'tool_uncertain', 'supervisor_review'}
                and latest.get('main_chat') == 'delivered'
                and (receipt.get('next_action') != provisional['next_action'] or receipt.get('suggested_changes')
                     or receipt.get('scientific_review', {}).get('passed') is False)):
            review_id = (await _durable_io(store.emit, 'supervisor_review', {
                **event.get('payload', {}), 'parent_event_id': event['event_id'], 'supervisor_decision': receipt,
            }, event_id=event['event_id'] + ':review'))
            (await _durable_io(store.receipt, review_id, 'supervisor', 'delivered', receipt))
        conv.execution_logs.append({'time': datetime.now().isoformat(), 'tool': 'supervisor',
                                    'message': f"监督交回主chat：{receipt['next_action']} — {receipt['reason']}"})
    except Exception as error:
        # A read-only supervisor outage gets an explicit fallback receipt so the
        # main agent can diagnose/advance instead of waiting forever for an LLM.
        fallback = {
            'next_action': 'diagnose_and_fix', 'reason': '监督服务暂不可用，主chat按结构化事件继续',
            'observer_error': str(error), 'evidence_refs': [event['event_id']],
        }
        if event['kind'] == 'scientific_review':
            fallback.update(review_action='ask_user', review_status='insufficient_evidence',
                            scientific_review={'passed':False,'issues':['Supervisor infrastructure failure; main computations were not invalidated.']})
            review_id = (await _durable_io(store.emit, 'scientific_review_failed', {**event['payload'], 'main_already_handling':False,
                                   'requires_user':True}, event_id=event['event_id'] + ':infrastructure'))
            (await _durable_io(store.receipt, review_id, 'supervisor', 'delivered', fallback))
        (await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', fallback))


async def _continue_lifecycle_event(conv, event, reservation=None):
    store = conv.session._lifecycle_store
    turn_uid = reservation['uid'] if reservation else conv.current_uid
    def rollback_unscheduled():
        if conv.current_uid == turn_uid:
            conv.is_processing = False
            if reservation:
                conv.current_uid = reservation['previous_uid']
    if reservation:
        try:
            changed = await _lifecycle_controls_changed(conv, event, reservation['goal_version'])
            if changed or conv.current_uid != turn_uid or not conv.is_processing:
                await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                                  {'reason': 'reservation superseded before lifecycle turn started'})
                rollback_unscheduled()
                return
        except BaseException:
            try:
                await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                                  {'reason': 'dispatcher stopped before lifecycle turn started'})
            finally:
                rollback_unscheduled()
            raise
    try:
        await run_agent_background(conv.username, f"[生命周期事件] {event['kind']}：监督已交回主chat处理",
                                   conv.conv_id, lifecycle_event=event)
        if not _api_turn_is_current(conv, turn_uid):
            await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                              {'reason': 'lifecycle turn superseded or paused'})
            return
        await _durable_io(store.receipt, event['event_id'], 'main_chat', 'delivered', {
            'turn_uid': turn_uid, 'status': conv.current_status,
            'negotiation_owner': 'lead-orchestrator', 'delivery_owner': 'lead-orchestrator',
        })
    except BaseException as error:
        try:
            await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                              {'reason': 'lifecycle turn did not complete', 'error': str(error)})
        finally:
            if conv.current_uid == turn_uid:
                conv.is_processing = False
        if isinstance(error, asyncio.CancelledError) or not isinstance(error, Exception):
            raise


def _schedule_lifecycle(coro):
    task = asyncio.create_task(coro)
    _lifecycle_tasks.add(task)
    task.add_done_callback(_lifecycle_tasks.discard)


_lifecycle_dispatch_inflight = set()


async def _dispatch_lifecycle_once():
    # A large or slow mailbox must not delay another user's ready work.
    # Keep one receiver sweep per conversation and bound total I/O. Claims
    # and receipts remain serialized by the original durable mailbox locks.
    semaphore = asyncio.Semaphore(8)
    pending = []

    async def dispatch(username, conv, store):
        key = (username, conv.conv_id)
        if key in _lifecycle_dispatch_inflight:
            return
        _lifecycle_dispatch_inflight.add(key)
        try:
            async with semaphore:
                try:
                    await _dispatch_conversation_lifecycle(username, conv, store)
                except Exception as error:
                    # Keep the affected conversation's state for recovery.
                    print(f'[Lifecycle] conversation {username}/{conv.conv_id} dispatch error: {error}', flush=True)
        finally:
            _lifecycle_dispatch_inflight.discard(key)

    for username in session_manager.list_users():
        user = session_manager.get_user(username)
        for conv in list(user.conversations.values()):
            if not conv.session:
                continue
            store = conv.session._lifecycle_store
            if not store:
                continue
            pending.append(dispatch(username, conv, store))
    if pending:
        tasks = [asyncio.create_task(item) for item in pending]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            # Even a child-raised CancelledError must not leave other sweeps
            # running in the background. Their durable I/O uses shield and
            # settles or returns its claim before cancellation completes.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


async def _claim_lifecycle_receiver(store, event_id, receiver):
    """Cancellation never abandons a claim still committing in a worker."""
    pending = asyncio.create_task(asyncio.to_thread(store.claim, event_id, receiver))
    _api_durable_writes.add(pending)
    pending.add_done_callback(_api_durable_writes.discard)
    try:
        return await asyncio.shield(pending)
    except asyncio.CancelledError:
        claimed = await _finish_pending_io(pending)
        if claimed:
            await _durable_io(store.receipt, event_id, receiver, 'retry',
                                    {'reason': 'dispatcher cancelled before scheduling'})
        raise


async def _lifecycle_controls_changed(conv, event, goal_version):
    from agents.lifecycle import superseded_runtime_event, current_goal_runtime_event
    snapshot = conv.session._runtime_snapshot
    runtime = (await asyncio.to_thread(snapshot) if snapshot else {}) or {}
    metadata_current=True
    if ((event.get('payload')or{}).get('recovery')or{}).get('action')=='inspect_forward_metadata_alignment':
        from agents.metadata_recovery import metadata_event_current
        native_runtime=_workflow_runtimes.get((conv.username,conv.conv_id))
        metadata_current=bool(native_runtime)and await _durable_io(
            _owned_conversation_io,conv,metadata_event_current,native_runtime,event)
    origin=(event.get('payload') or {}).get('origin_goal_version',conv.session.goal_contract.version)
    scope_current=(origin is None or origin == conv.session.goal_contract.version or await _durable_io(
        _owned_conversation_io,conv,current_goal_runtime_event,conv.session,event,runtime))
    return bool(conv.interrupt_requested or conv.interrupt_message or not scope_current
        or runtime.get('user_paused') or conv.session.goal_contract.version != goal_version
        or superseded_runtime_event(event, runtime) or not metadata_current)



async def _claim_current_lifecycle_receiver(conv, event, store, receiver, goal_version):
    claimed = False
    try:
        # Keep unchanged controls from repeatedly claiming and rewriting the
        # durable mailbox. Still recheck after claiming to cover races.
        if await _lifecycle_controls_changed(conv, event, goal_version):
            return False
        claimed = await _claim_lifecycle_receiver(store, event['event_id'], receiver)
        if claimed and await _lifecycle_controls_changed(conv, event, goal_version):
            await _durable_io(store.receipt, event['event_id'], receiver, 'retry',
                                    {'reason': 'controls changed while claiming lifecycle event'})
            return False
        return claimed
    except BaseException:
        if claimed:
            await _durable_io(store.receipt, event['event_id'], receiver, 'retry',
                                    {'reason': 'dispatcher stopped before scheduling'})
        raise


async def _reserve_and_schedule_lifecycle_main(conv, event, store):
    # Control decisions and the reservation stay on the API loop. No second
    # dispatcher or query may start this conversation while disk claim waits.
    if conv.is_processing or conv.interrupt_requested or conv.interrupt_message:
        return False
    previous_uid = conv.current_uid
    reserved_uid = 'lifecycle:' + uuid.uuid4().hex
    goal_version = conv.session.goal_contract.version
    conv.is_processing = True
    conv.current_uid = reserved_uid
    def rollback_reservation():
        if conv.current_uid == reserved_uid:
            conv.is_processing = False
            conv.current_uid = previous_uid
    claimed = False
    try:
        claimed = await _claim_current_lifecycle_receiver(conv, event, store, 'main_chat', goal_version)
        if not claimed:
            rollback_reservation()
            return False
        if conv.current_uid != reserved_uid or not conv.is_processing:
            await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                                    {'reason': 'controls changed before main scheduling'})
            rollback_reservation()
            return False
        _schedule_lifecycle(_continue_lifecycle_event(conv, event, reservation={
            'uid': reserved_uid, 'previous_uid': previous_uid, 'goal_version': goal_version}))
        return True
    except BaseException:
        try:
            if claimed:
                await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry',
                                        {'reason': 'dispatcher stopped before main scheduling'})
        finally:
            rollback_reservation()
        raise


async def _dispatch_conversation_lifecycle(username, conv, store):
    (await _durable_io(store.recover_claims))
    (await _durable_io(store.heartbeat))
    runtime_state = (await asyncio.to_thread(conv.session._runtime_snapshot)) if conv.session._runtime_snapshot else {}
    if runtime_state:
        runtime = _ensure_workflow_runtime(conv)
        jobs = [j for j in (await asyncio.to_thread(_watch.list)) if j.get('username') == username and j.get('conv_id') == conv.conv_id]
        await asyncio.to_thread(_owned_conversation_io, conv, runtime.tick, jobs, bool(conv.interrupt_requested), (),
                                runtime_state.get("user_pause_revision", 0))
        # tick() can complete or pause the graph, so use its current
        # durable state for all wake-up decisions below.
        runtime_state = (await asyncio.to_thread(conv.session._runtime_snapshot)) or {}
    formal_runtime_active = bool(
        runtime_state.get('nodes')
        and runtime_state.get('status') in {'active', 'needs_user'}
        and not runtime_state.get('user_paused')
    )
    # Freeze is session-wide: keep durable mailbox events pending, but
    # do not claim fresh main/supervisor work until the user resumes.
    if conv.interrupt_requested or conv.interrupt_message or runtime_state.get('user_paused'):
        return
    if conv.is_processing and conv.last_progress_at and time.time() - conv.last_progress_at > 360:
        (await _durable_io(store.emit, 'stalled', {'workflow_id': conv.session._current_line_id,
                              'current_agent': conv.current_agent, 'last_progress_at': conv.last_progress_at},
                   event_id='stalled:' + str(conv.current_uid)))
    from agents.workflow_priority import order_events
    from agents.workflow_review_stage import WorkflowReviewStage
    review_controller = WorkflowReviewStage(runtime) if runtime_state and hasattr(runtime, 'root') else None
    if review_controller is not None:
        review_changed = bool(await asyncio.to_thread(review_controller.recover, observed_state=runtime_state))
        for step, node in runtime_state.get('nodes', {}).items():
            if node.get('status') == 'prefinish':
                created_review = await asyncio.to_thread(review_controller.observe, step, observed_state=runtime_state)
                review_changed = bool(created_review) or review_changed
        if review_changed:
            await asyncio.to_thread(runtime.flush_events)
    events = await asyncio.to_thread(store.pending)
    review_reserved = any(record.get('phase') == 'reviewing'
        for record in (runtime_state.get('scientific_owner_reviews') or {}).values())
    for event in order_events(events, runtime_state):
        from agents.retained_prerequisite_scope import defer_pending_prerequisite_event
        if defer_pending_prerequisite_event(conv.session,event,runtime_state):
            # The currently owned prerequisite may still be valid under a
            # pending condition edit. Keep its exact durable notice untouched;
            # never promote it or grant execution until the new graph is proved.
            continue
        if ((event.get('payload')or{}).get('recovery')or{}).get('action')=='inspect_forward_metadata_alignment':
            if await _lifecycle_controls_changed(conv,event,conv.session.goal_contract.version):
                for receiver in ('supervisor','main_chat'):
                    await _durable_io(store.receipt,event['event_id'],receiver,'obsolete',
                        {'reason':'metadata capability source, owner, approval, controls or native proof superseded','not_resubmitted':True})
                continue
        if event.get('kind') == 'worker_scientific_owner_review' and review_controller is not None:
            if not await asyncio.to_thread(review_controller.current_event, event):
                for role in ('scientific_owner', 'supervisor', 'main_chat'):
                    await _durable_io(store.receipt, event['event_id'], role, 'obsolete',
                        {'reason':'scientific review attempt or controls superseded', 'not_resubmitted':True})
                continue
            if event.get('scientific_owner','pending') in {'pending','retry'} and not review_reserved:
                if await _claim_current_lifecycle_receiver(conv, event, store, 'scientific_owner', conv.session.goal_contract.version):
                    _schedule_lifecycle(_run_scientific_owner_review(conv, event))
                    review_reserved = True
            continue
        if event.get('kind') == 'worker_prefinish' and review_controller is not None:
            # A native durable acceptance ticket owns this attempt. Do not send
            # the same review through main where unrelated patch work can win.
            current_review = await asyncio.to_thread(runtime.snapshot)
            step = (event.get('payload') or {}).get('step_id')
            owned_review = any(record.get('binding',{}).get('step_id') == step
                and record.get('binding',{}).get('token') == current_review.get('nodes',{}).get(step,{}).get('token')
                and record.get('phase') in {'required','retryable','reviewing'}
                for record in (current_review.get('scientific_owner_reviews') or {}).values())
            if owned_review:
                for role in ('supervisor','main_chat'):
                    if event.get(role) not in {'delivered','obsolete'}:
                        await _durable_io(store.receipt, event['event_id'], role, 'obsolete',
                            {'reason':'exact native scientific owner review lane owns acceptance', 'not_resubmitted':True})
                continue
        from agents.answer_delivery import completed_notification_binding, sources_unchanged, digest as delivery_digest
        delivered = getattr(conv.session,'context',{}).get('answer_delivery') or {}
        delivery_actor, delivery_uid = conv.session, conv.current_uid
        delivered_goal = conv.session.goal_contract.to_dict() if delivered.get('native_bindings') else {}
        notice = completed_notification_binding(event,runtime_state,delivered,goal_sha256=delivery_digest(delivered_goal))
        if notice:
            unchanged = await asyncio.to_thread(sources_unchanged,delivered.get('sources',[]))
            fresh_runtime = (await asyncio.to_thread(conv.session._runtime_snapshot)) or {}
            if (unchanged and conv.session is delivery_actor and conv.current_uid == delivery_uid
                    and conv.session.goal_contract.to_dict() == delivered_goal
                    and getattr(conv.session,'context',{}).get('answer_delivery') is delivered
                    and completed_notification_binding(event,fresh_runtime,delivered,goal_sha256=delivery_digest(delivered_goal))):
                proof={'status':'notification_only','reason':'exact completed native attempt already delivered',
                       'publication_id':delivered.get('publication_id'),'native_binding':notice,'not_resubmitted':True}
                for receiver in ('supervisor','main_chat'):
                    await _durable_io(store.receipt,event['event_id'],receiver,'delivered',proof)
                continue
        from agents.lifecycle import superseded_runtime_event
        if superseded_runtime_event(event, runtime_state):
            for receiver in ('main_chat', 'supervisor'):
                if event.get(receiver) not in {'delivered', 'obsolete'}:
                    (await _durable_io(store.receipt, event['event_id'], receiver, 'obsolete', {
                        'reason': 'newer durable node attempt or phase supersedes this event',
                        'not_resubmitted': True}))
            continue
        # A default-review event is allowed to reach its current native main
        # even while the original question remains pending. Revalidate source
        # ownership/science before both restart replay and negotiation gates.
        default_review_binding=None
        if (event.get('kind')=='preflight_recovered'
                and (event.get('payload')or{}).get('status')=='preparation_default_review_required'):
            from agents.preparation_defaults import preparation_review_binding
            review_actor,review_uid=conv.session,conv.current_uid
            review_goal=copy.deepcopy(review_actor.goal_contract.to_dict())
            review_question=copy.deepcopy(review_actor._pending_user_interaction)
            review_patch=copy.deepcopy(review_actor._pending_workflow_patch)
            proof=await _durable_io(_owned_conversation_io,conv,preparation_review_binding,review_actor,event)
            if (proof and conv.session is review_actor and conv.current_uid==review_uid
                    and review_actor.goal_contract.to_dict()==review_goal
                    and review_actor._pending_user_interaction==review_question
                    and review_actor._pending_workflow_patch==review_patch):
                default_review_binding=proof
        # A dead process may have left a receiver in `processing`; its
        # claim is recovered as `retry`.  Replaying that event for an
        # ordinary/non-DAG chat caused the observed "old session spoke
        # by itself" loop.  Archive it for audit instead.  An active
        # formal DAG is deliberately exempt, because its persisted
        # node/job state proves there is real work to continue.
        if (float(event.get('time', 0) or 0) < _PROCESS_STARTED_AT
                and not formal_runtime_active and not default_review_binding):
            details = {
                'reason': 'pre-restart event has no active formal workflow',
                'process_started_at': _PROCESS_STARTED_AT,
                'not_resubmitted': True,
            }
            if event.get('supervisor') not in {'delivered', 'obsolete'}:
                (await _durable_io(store.receipt, event['event_id'], 'supervisor', 'obsolete', details))
            if event.get('main_chat') not in {'delivered', 'obsolete'}:
                (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete', details))
            continue
        if (runtime_state and getattr(conv.session, '_evidence_root', None)
                and event['kind'] == 'worker_prefinish' and event.get('supervisor') in {'pending', 'retry'}):
            from agents.dossier_event import verify_executor_dossier_event
            observed_session, observed_uid = conv.session, conv.current_uid
            observed_goal = conv.session.goal_contract.version
            observed_approval = copy.deepcopy(observed_session.goal_contract.to_dict())
            proof = await asyncio.to_thread(verify_executor_dossier_event, event, runtime_state,
                owned_root=observed_session._evidence_root.parent,
                goal_contract=observed_approval, jobs=jobs, session=observed_session)
            if proof:
                try:
                    changed = await _lifecycle_controls_changed(conv, event, observed_goal)
                    if (changed or conv.session is not observed_session or conv.current_uid != observed_uid
                            or conv.session.goal_contract.to_dict() != observed_approval):
                        continue
                except Exception:
                    continue  # No stale identity receipt or scheduling on uncertain controls.
            if proof:
                await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', {
                    'next_action': 'advance_dependencies',
                    'reason': 'native executor dossier identity verified; scientific owner must still review and finish',
                    'identity_verified': True, 'needs_scientific_owner_review': True,
                    'evidence_refs': [proof['dossier_call_id'], proof['producer_call_id']],
                    'executor_identity': proof, 'not_resubmitted': True,
                })
                event = (await asyncio.to_thread(store.snapshot)).get('events', {}).get(event['event_id'], event)
        if (event['kind'] == 'worker_validation_dossier_ready'
                and event.get('supervisor') in {'pending', 'retry'}):
            dossier = event.get('payload', {}).get('validation_dossier') or {}
            (await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', {
                'next_action': 'advance_dependencies' if dossier.get('verdict') == 'pass' else 'diagnose_and_fix',
                'reason': 'executor-bound validation dossier is already kernel verified',
                'evidence_refs': [dossier.get('evidence_call_id')],
                'scientific_review': {'passed': dossier.get('verdict') == 'pass', 'issues': dossier.get('failed_items', [])},
            }))
            event = (await asyncio.to_thread(store.snapshot)).get('events', {}).get(event['event_id'], event)
        if (event['kind'] == 'worker_recovery_ready'
                and event.get('supervisor') in {'pending', 'retry'}):
            recovery = event.get('payload', {}).get('recovery') or {}
            (await _durable_io(store.receipt, event['event_id'], 'supervisor', 'delivered', {
                'next_action': 'diagnose_and_fix',
                'reason': 'durable workflow state exposes a bounded recovery capability',
                'evidence_refs': [event['event_id']],
                'suggested_changes': [{
                    'action': 'call_tool', 'tool': recovery.get('tool'),
                    'arguments': copy.deepcopy(recovery.get('arguments') or {}),
                    'step_id': event.get('payload', {}).get('step_id'),
                }],
                'diagnostic': True, 'recovery_capability_verified': True,
            }))
            event = (await asyncio.to_thread(store.snapshot)).get('events', {}).get(event['event_id'], event)
        if event.get('supervisor') in {'pending', 'retry'}:
            supervisor_goal = conv.session.goal_contract.version
            if await _claim_current_lifecycle_receiver(conv, event, store, 'supervisor', supervisor_goal):
                _schedule_lifecycle(_supervise_lifecycle_event(conv, event))
                continue
        if (event.get('supervisor') not in ({'delivered'} if event['kind']=='job_pending_warning' else {'delivered','processing'}) or not event.get('supervisor_receipt')
                or event.get('main_chat') not in {'pending', 'retry', 'waiting_user'}):
            continue
        payload = event.get('payload', {})
        if payload.get('runtime_id') and event['kind'] == 'workflow_started':
            (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'delivered', {'status': 'runtime scheduled; main remains owner'}))
            continue
        if payload.get('main_already_handling') or event['kind'] in {'user_paused', 'tool_uncertain', 'stalled', 'waiting_jobs', 'plan_patch_applied'}:
            (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'delivered', {'status': 'notification_only', 'not_resubmitted': True}))
            continue
        old_goal = payload.get('origin_goal_version', payload.get('goal_contract', {}).get('version'))
        node = runtime_state.get('nodes', {}).get(payload.get('step_id'), {})
        reconciled_event = bool(node.get('token')
            and payload.get('plan_version') == runtime_state.get('plan_version')
            and runtime_state.get('goal_contract', {}).get('version') == conv.session.goal_contract.version
            and event['event_id'] == f"runtime:{payload.get('runtime_id')}:{node['token']}:{node.get('status')}"
            and any(h.get('from_version') == old_goal for h in runtime_state.get('goal_reconciliation_history', [])))
        retained_event=False
        if old_goal is not None and old_goal != conv.session.goal_contract.version and not reconciled_event:
            from agents.lifecycle import current_goal_runtime_event
            retained_event=await _durable_io(_owned_conversation_io,conv,current_goal_runtime_event,
                conv.session,event,runtime_state,jobs=jobs)
        if old_goal is not None and old_goal != conv.session.goal_contract.version and not reconciled_event and not retained_event:
            (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete', {'reason': 'user superseded the goal'}))
            continue
        from agents.lifecycle import obsolete_internal_question_for_recovery
        if obsolete_internal_question_for_recovery(conv.session, event, runtime=runtime_state):
            conv.current_status = '恢复工作流'
        pending = conv.session._pending_user_interaction or {}
        if (conv.session._awaiting_plan_approval or pending.get('tool') in {'failure_decision', 'workflow_patch_decision', 'user_decision'}
                and not default_review_binding):
            (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'waiting_user', {'negotiation_owner': 'lead-orchestrator'}))
            continue
        if conv.is_processing or conv.interrupt_requested or conv.interrupt_message:
            continue
        if event.get('main_chat') == 'waiting_user':
            (await _durable_io(store.receipt, event['event_id'], 'main_chat', 'retry'))
        await _reserve_and_schedule_lifecycle_main(conv, event, store)


async def _drain_lifecycle_events():
    while True:
        from agents.lifecycle import wait_for_lifecycle_change
        # emit() wakes this immediately.  The bounded timeout is only a safety
        # sweep for externally-written state; a two-second full scan rewrote
        # every historical mailbox and consumed a CPU core while idle.
        await asyncio.to_thread(wait_for_lifecycle_change,10)
        try:
            await _dispatch_lifecycle_once()
            from agents._restart import consume
            if consume(): _schedule_lifecycle(_perform_backend_restart())
        except Exception as error:
            print(f'[Lifecycle] dispatcher error: {error}', flush=True)


async def _drain_job_notifications():
    """Continuously drain JobWatch notifications and wake the responsible
    conversation so the agent runs its fault-handling rules even after its
    original turn already ended. This is the '及时提醒智能体' mechanism.
    If the conversation is busy (user mid-turn), the notification is kept in
    a backlog and retried on the next cycle."""
    global _notification_backlog
    while True:
        await asyncio.sleep(5)
        try:
            notifs = (await asyncio.to_thread(_watch.drain_notifications))
        except Exception as e:
            notifs = []
            print(f"  [JobWatch] drain error: {e}", flush=True)
        # Merge new + retry backlog
        notifs = _notification_backlog + notifs
        _notification_backlog = []
        for n in notifs:
            conv_id = n.get("conv_id", "")
            username = n.get("username", "")
            msg = n.get("message", "")
            if not conv_id or not username:
                print(f"  [JobWatch] notification {n.get('job_id')} has no conv/user — dropped", flush=True)
                (await asyncio.to_thread(_watch.ack_notification, str(n.get("job_id", "")),n.get('notification_id','')))
                continue
            try:
                # Wake the conversation (session.reply → agent fault handling).
                user = session_manager.get_user(username)
                conv = user.get_conversation(conv_id) if user else None
                if conv is None:
                    # conversation deleted while job was running
                    print(f"  [JobWatch] conv {conv_id} gone — notification dropped", flush=True)
                    (await asyncio.to_thread(_watch.ack_notification, str(n.get("job_id", "")),n.get('notification_id','')))
                    continue
                _ensure_session(conv)
                rec = (await asyncio.to_thread(_watch.get, str(n.get('job_id')))) or {}
                kind=n.get('kind') or ('job_failed' if rec.get('failed') else 'job_completed')
                conv.session._lifecycle_store.emit(kind, {
                    'job': rec, 'notification': n, 'workflow_id': conv.session._current_line_id,
                    'goal_contract': conv.session.goal_contract.to_dict(),
                    'instruction':'PENDING is nonterminal: delegate resource review, preserve leases and original job, never label completion or auto-resubmit.' if kind=='job_pending_warning' else '',
                }, event_id=f"job:{n.get('job_id')}:{n.get('notification_id') or n.get('state')}")
                # Ack only after the durable two-receiver mailbox has accepted it.
                (await asyncio.to_thread(_watch.ack_notification, str(n.get("job_id", "")),n.get('notification_id','')))
                print(f"  [JobWatch] notified conv {conv_id} about job {n.get('job_id')}", flush=True)
            except Exception as e:
                print(f"  [JobWatch] notify conv {conv_id} failed: {e}", flush=True)
                _notification_backlog.append(n)


@app.on_event("startup")
async def _startup_watch():
    await asyncio.to_thread(migrate_auth_storage)
    session_manager._tokens = load_tokens()
    if os.environ.get('BIMEM_BACKGROUND_MONITOR', '1') != '1':
        return
    # Restore lifecycle mailboxes even when the browser has not revisited a
    # conversation. A job/tool completion must not wait for a manual page load.
    for username in load_users():
        user = session_manager.get_user(username)
        for conv in user.conversations.values():
            if conv.conversation_active:
                _ensure_session(conv)
                if conv.session:
                    conv.session._lifecycle_store.initialize()
                    with conv.session.recovery_gate.transaction():
                        pass
                    conv.session._checkpoint('backend_restore')
                    _ensure_supervisor(conv)._checkpoint('supervisor_restore')
                    for entry in conv.session.recovery_gate.snapshot().values():
                        if entry.get('status') not in {'reserved', 'uncertain'}:
                            continue
                        alive = False
                        try:
                            if entry.get('owner_pid'):
                                os.kill(int(entry['owner_pid']), 0)
                                alive = True
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            alive = True
                        if not alive:
                            conv.session._lifecycle_store.emit('dispatch_unknown', {
                                'recovery_key': entry['key'], 'attempt': entry,
                                'workflow_id': conv.session._current_line_id,
                                'goal_contract': conv.session.goal_contract.to_dict(),
                                'next_action': 'inspect original receipt/logs; main chat requests user identity decision if necessary',
                            }, event_id='dispatch:' + entry['attempt_id'] + ':unknown')
    asyncio.create_task(_drain_job_notifications())
    asyncio.create_task(_drain_lifecycle_events())


@app.get("/api/jobs")
async def list_watched_jobs(username: str = Depends(_get_user), conv_id: str = ''):
    """List jobs currently tracked by JobWatch (any state)."""
    jobs = []
    for j in (await asyncio.to_thread(_watch.list)):
        if j.get('username') != username or (conv_id and j.get('conv_id') != conv_id): continue
        jobs.append({
            "job_id": j.get("job_id"),
            "state": j.get("state"),
            "last_state": j.get("last_state"),
            "terminal": j.get("terminal"),
            "failed": j.get("failed"),
            "gas": j.get("gas"),
            "cif": j.get("cif"),
            "work_dir": j.get("work_dir"),
            "submitted_at": j.get("submitted_at"),
            "updated_at": j.get("updated_at"),
            "conv_id": j.get("conv_id"),
            "diagnosis": j.get("diagnosis"),
            "pending_reason": j.get('pending_reason',''),
            "pending_age_seconds": j.get('pending_age_seconds',0),
            "resource_warning": j.get('resource_warning'),
            "estimated_start": j.get('estimated_start',''),
        })
    return {"count": len(jobs), "jobs": jobs}


@app.get("/api/jobs/{job_id}/explain")
async def explain_job(job_id: str, username: str = Depends(_get_user)):
    """运行只读 monitor 小智能体，检查作业进度并给出简洁中文结论。

    - 只读：monitor 只有 check_job / list_my_jobs / diagnose_job / read_file / grep_search，
      无 submit/scancel/write/run_bash；本端点还会剥离其 handoff 工具，杜绝委派连锁。
    - 前端任务栏"AI 看进度"按钮调用；其他 Agent 也可在会话内委派 monitor。
    """
    rec = (await asyncio.to_thread(_watch.get, str(job_id).strip()))
    if not rec:
        return {"job_id": job_id, "error": "该作业不在当前监控列表中（可能已完成并被清理）", "done": True}
    if rec.get('username') != username:
        raise HTTPException(status_code=403, detail='只能查看本用户作业')

    def _run() -> str:
        from agents.session import Session
        from agents.agent import Agent
        from agents.defns import MONITOR
        read_only = Agent(
            name="monitor",
            instructions=MONITOR.instructions,
            functions=[f for f in MONITOR.functions
                       if not getattr(f, "__name__", "").startswith("handoff_to")],
            model=MONITOR.model,
            max_turns=6,
        )
        s = Session(config=config_for_user(username))
        prompt = (
            f"请检查 SLURM 作业 {job_id} 的当前进度与健康状况，用简洁中文回答。\n"
            f"已知作业信息：状态={rec.get('state')}，gas={rec.get('gas')}，cif={rec.get('cif')}，"
            f"工作目录={rec.get('work_dir')}，提交时间={rec.get('submitted_at')}，最后更新={rec.get('updated_at')}。\n"
            f"请用只读工具查证后回答：①现在在不在正常跑 ②进度如何 ③有没有异常苗头 ④预计还要多久。"
            f"不要委派、不要修复、不要提交或取消作业。总回答 ≤ 150 字。"
        )
        try:
            from agents.watch_context import set_context, clear_context
            set_context(username, rec.get('conv_id', ''), 'monitor')
            try:
                return s.run_readonly_observer(prompt, read_only, max_rounds=6)
            finally:
                clear_context()
        except Exception as e:
            return f"monitor 分析失败：{e}"

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        verdict = await loop.run_in_executor(pool, _run)
    return {
        "job_id": job_id,
        "state": rec.get("state"),
        "gas": rec.get("gas"),
        "cif": rec.get("cif"),
        "work_dir": rec.get("work_dir"),
        "verdict": verdict,
        "done": True,
    }


@app.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, username: str = Depends(_get_user)):
    """Interrupt a running computation job (scancel). Frontend '中断' button."""
    record = (await asyncio.to_thread(_watch.get, job_id))
    if not record or record.get('username') != username:
        raise HTTPException(status_code=403, detail='只能取消本用户作业')
    def control():
        from agents.job_control import pause_owned_workflow
        pause_owned_workflow(Path(__file__).parent, username, record['conv_id'])
        return _watch.cancel(job_id, control_root=Path(__file__).parent / 'runs' / username / record['conv_id'])
    # Scheduler commands must not block API heartbeats or independent queries.
    return await asyncio.to_thread(control)


# ── Static files ──
STATIC_DIR = Path(__file__).parent / "frontend" / "build"
FRONTEND_READY = (STATIC_DIR / "index.html").is_file() and (STATIC_DIR / "static").is_dir()

@app.get("/")
async def serve_index():
    if FRONTEND_READY:
        return FileResponse(STATIC_DIR / "index.html")
    return {"status": "healthy", "frontend": "not_built", "version": app.version}

if FRONTEND_READY:
    app.mount("/static", StaticFiles(directory=STATIC_DIR / "static"), name="static")

@app.get("/{full_path:path}")
async def serve_static(full_path: str):
    file_path = STATIC_DIR / full_path
    if FRONTEND_READY and file_path.is_file():
        return FileResponse(file_path)
    if FRONTEND_READY:
        return FileResponse(STATIC_DIR / "index.html")
    raise HTTPException(status_code=404, detail="Frontend has not been built")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)


def _ensure_scientific_owner_reviewer(conv, runtime, event):
    """Fresh isolated role state; no reuse of a busy main/supervisor actor."""
    from agents.session import Session
    from agents.goal_contract import GoalContract
    from agents.state_io import compact_checkpoint_evidence, apply_evidence_references, write_checkpoint
    binding = event['payload']['review_binding']
    actor = Session(config=conv.session.config, registry=conv.session.registry)
    actor.goal_contract = GoalContract.from_dict(conv.session.goal_contract.to_dict())
    actor._current_line_id = conv.session._current_line_id
    actor._evidence_root = runtime.root / 'evidence'
    actor._runtime_snapshot = runtime.snapshot
    actor._persist_chain = lambda *args, **kwargs: None
    # Attempt identity comes from the controller, never a model-supplied path.
    path = runtime.root / ('scientific_review_' + uuid.uuid4().hex + '.json')
    def save(state, reason):
        state['owner'] = {'username':conv.username, 'conv_id':conv.conv_id, 'step_id':binding['step_id']}
        references = compact_checkpoint_evidence(state, actor._evidence_root)
        write_checkpoint(path, state)
        apply_evidence_references(actor.memory, references)
    actor._on_checkpoint = save
    return actor, path


async def _run_scientific_owner_review(conv, event):
    """Independent acceptance lane; never reserve or clear a live main turn."""
    from agents.workflow_review_stage import WorkflowReviewStage
    main = conv.session
    from agents.workflow_user_control import workflow_control_guard
    user_control=workflow_control_guard(main,conv.username,conv.conv_id)
    goal = copy.deepcopy(main.goal_contract.to_dict())
    runtime = _ensure_workflow_runtime(conv)
    controller = WorkflowReviewStage(runtime)
    store = main._lifecycle_store
    stopped = threading.Event()
    def authority_current():
        return (conv.session is main and user_control()
                and main.goal_contract.to_dict() == goal)
    def valid():
        return not stopped.is_set() and authority_current()
    def run():
        from agents.watch_context import get_context, set_context
        previous = get_context()
        set_context(conv.username, conv.conv_id, 'scientific-owner', main._current_line_id)
        try:
            if not valid(): raise ValueError('owner review controls changed before actor creation')
            actor, path = _ensure_scientific_owner_reviewer(conv, runtime, event)
            return controller.run(actor, event, checkpoint_path=path, control_valid=valid)
        finally:
            set_context(**previous)
    cancelled = False
    pending = asyncio.create_task(asyncio.to_thread(run))
    _api_durable_writes.add(pending)
    pending.add_done_callback(_api_durable_writes.discard)
    try:
        try:
            receipt = await asyncio.shield(pending)
        except asyncio.CancelledError:
            cancelled = True; stopped.set()
            receipt = await _finish_pending_io(pending)
        await _durable_io(store.receipt, event['event_id'], 'scientific_owner', 'delivered', receipt)
        await _durable_io(store.receipt, event['event_id'], 'supervisor', 'obsolete',
            {'reason':'independent scientific owner lane completed', 'not_resubmitted':True})
        await _durable_io(store.receipt, event['event_id'], 'main_chat', 'obsolete',
            {'reason':'scientific owner lane completed; authoritative DAG state owns continuation', 'not_resubmitted':True})
    except Exception as error:
        from agents.owner_review_diagnostics import failure_record
        current = authority_current() and await asyncio.to_thread(controller.current_event, event)
        await _durable_io(store.receipt, event['event_id'], 'scientific_owner', 'retry' if current else 'obsolete',
            {**failure_record(error),'not_resubmitted':True})
    finally:
        # This transports durable events only; display rendering has its own lane.
        await asyncio.to_thread(runtime.flush_events)
    if cancelled: raise asyncio.CancelledError
