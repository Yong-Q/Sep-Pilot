"""
BiMemAgent API - Multi-user, Multi-conversation Backend
"""
from fastapi import FastAPI, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
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
from datetime import datetime
from pathlib import Path

from auth import (
    session_manager, register_user, login_user,
    load_users, USER_DB, CONVERSATION_LOG, public_model_connection,
    save_model_connection, config_for_user, migrate_auth_storage, load_tokens,
)

app = FastAPI(title="BiMemAgent API", description="多用户会话隔离、持久化混合串并行编排与全生命周期协调", version="3.4.76")

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
        "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'self'"
    )
    if request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store"
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


def _workflow_state(conv) -> Dict[str, Any]:
    """Structured, inspectable orchestration state for API/frontend."""
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
    except Exception:
        pass
    from agents.workflow_view import authoritative_line
    runtime = conv.session._runtime_snapshot() if conv.session and conv.session._runtime_snapshot else {}
    if runtime:
        from agents.parallel_workflow import runtime_summary
        runtime = {**runtime, 'live_reports': runtime_summary(runtime).get('live_reports', [])}
    lines = [authoritative_line(line, runtime) for line in lines]
    from agents.orchestration_chain import read_last_graph
    from agents.workspace import session_root
    graph_projection = read_last_graph(session_root(conv.conv_id, conv.username),
                                      {'username': conv.username, 'conv_id': conv.conv_id}) or {"source": "none", "nodes": []}
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
        "lifecycle": conv.session._lifecycle_state() if conv.session else {},
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


def _ensure_session(conv):
    """Lazily init session on a conversation."""
    from agents.workspace import session_root
    from agents.config import get_config
    root = session_root(conv.conv_id, getattr(conv, 'username', ''))
    if not root.resolve().is_relative_to(get_config().project_root.resolve()):
        raise ValueError('conversation checkpoint must stay in the project workspace')
    checkpoint = root / 'session_checkpoint.json'
    if conv.session is None:
        try:
            from agents.defns import ORCHESTRATOR
            from agents.session import Session
            conv.session = Session(config=config_for_user(conv.username))
            restored = False
            saved_state = getattr(conv, 'session_state', None) or {}
            if checkpoint.exists():
                latest = json.loads(checkpoint.read_text())
                if latest.get('checkpoint_at', 0) > saved_state.get('checkpoint_at', 0):
                    saved_state = latest
            if saved_state:
                if saved_state.get('owner') and saved_state['owner'] != {'username': conv.username, 'conv_id': conv.conv_id}:
                    raise PermissionError('checkpoint belongs to a different user/session')
                restored = conv.session.import_state(saved_state)
            if not restored and conv.messages_history:
                restored = conv.session.hydrate_from_transcript(conv.messages_history)
            elif restored and conv.messages_history:
                # Session checkpoint and durable display transcript are written
                # on different boundaries. Merge any user turns that arrived
                # after the checkpoint (interrupts/redirects/job notices).
                conv.session.reconcile_transcript(conv.messages_history, after_count=saved_state.get('transcript_message_count'))
            if not conv.session.current_agent:
                conv.session.current_agent = ORCHESTRATOR
            conv.session._on_progress = conv._on_progress
        except PermissionError:
            conv.session = None
            raise
        except Exception as e:
            print(f"Warning: Could not init session: {e}")
            conv.session = None
    if conv.session:
        from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
        from agents.lifecycle import LifecycleStore
        conv.session.recovery_gate.path = root / 'recovery_state.json'
        if not conv.session._current_line_id:
            conv.session._current_line_id = conv.conv_id
        from agents.task_line import get_store
        line = get_store().begin_line(conv.session._current_line_id, username=conv.username, conv_id=conv.conv_id, title=conv.title)
        conv.session._current_line_id = line['line_id']
        if conv.session._lifecycle_store is None:
            conv.session._lifecycle_store = LifecycleStore(root / 'lifecycle.json')
            if (root / 'lifecycle.json').exists():
                conv.session._lifecycle_store.recover_claims()
        conv.session._evidence_root = root / 'evidence'
        conv.session._on_lifecycle_event = lambda kind, payload: conv.session._lifecycle_store.emit(kind, payload)
        from agents.parallel_workflow import WorkflowStore, resources_for, conflicts, READ_ONLY
        runtime_store = WorkflowStore(conv.session.config.project_root / 'data/state/parallel_workflows.json')
        runtime_id = runtime_store.identity(conv.username, conv.conv_id)
        conv.session._runtime_snapshot = lambda: runtime_store.snapshot(runtime_id)
        conv.session._restore_workflow_approval(runtime_store.snapshot(runtime_id),
            {'username': conv.username, 'conv_id': conv.conv_id})
        conv.session._chain_state_is_primary = True
        conv.session._chain_jobs = lambda: [j for j in _watch.list()
            if (j.get('username'), j.get('conv_id')) == (conv.username, conv.conv_id)]
        conv.session._on_workflow_start = lambda plan_version: _start_workflow_runtime(conv, plan_version)
        conv.session._on_resource_review = lambda request: _review_resources(conv, request)
        conv.session._on_workflow_message = lambda step_id, text, kind='comment', message_id=None: _ensure_workflow_runtime(conv).send(step_id, text, kind, message_id)
        conv.session._on_workflow_resolve = lambda step_id, decision_id, verification_call_id: _ensure_workflow_runtime(conv).resolve_local_write(step_id, decision_id, verification_call_id)
        from agents.workflow_view import bind_result_handoff
        bind_result_handoff(conv.session, lambda: _ensure_workflow_runtime(conv))
        def tool_guard(tool, arguments):
            from agents.control_policy import role_denial, bypasses_data_lease
            role = getattr(conv.session.current_agent, 'name', '')
            denied = role_denial(role, tool)
            if denied: return denied
            if bypasses_data_lease(role, tool): return ''
            # Handoff changes the reasoning/inspection owner, not the data.
            # Mutating scientific tools remain guarded at their real dispatch.
            if tool.startswith('handoff_to_'): return ''
            if tool in READ_ONLY or tool in {'execute_workflow', 'message_workflow_node', 'propose_workflow_patch', 'apply_workflow_patch', 'discard_workflow_patch', 'revalidate_workflow_node_outputs','finish_workflow_node',
                    'request_user_decision', 'reconcile_watched_job', 'prepare_retry', 'accept_recovered_result',
                    'recovery_state', 'lifecycle_state', 'task_line_query', 'resolve_local_workflow_write', 'retarget_queued_job'}:
                return ''
            state = runtime_store.snapshot(runtime_id)
            if state and state['status'] in {'active', 'needs_user'} and (tool.startswith('handoff_to_') or
                    any(n['contract']['tool'] == tool and n['contract']['arguments'] == arguments for n in state['nodes'].values())):
                if tool != 'handoff_to_patcher':
                    return 'runtime owns this compiled node; do not re-execute it through the main Session'
            resources = resources_for({'tool': tool, 'arguments': arguments}, conv.session.config.project_root, root)
            if any(conflicts(a, b) for lease in runtime_store.snapshot().get('leases', {}).values()
                   for a in resources for b in lease['resources']):
                return 'tool mutation conflicts with a live/unknown workflow resource lease'
            return ''
        conv.session._on_tool_guard = tool_guard
        def _save_checkpoint(state, reason):
            state['owner'] = {'username': conv.username, 'conv_id': conv.conv_id}
            state['transcript_message_count'] = len(conv.messages_history)
            references = compact_checkpoint_evidence(state, root / 'evidence')
            write_checkpoint(checkpoint, state)
            workflow = {}
            if conv.session._current_line_id:
                from agents.task_line import get_store
                workflow = conv.session._workflow_view()
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
            apply_evidence_references(conv.session.memory, references)
            conv.session_state = state
        conv.session._on_checkpoint = _save_checkpoint
        conv.session._on_chain_committed = lambda: _workflow_events.publish(conv.username, conv.conv_id)
        def _progress(*args, **kwargs):
            conv._on_progress(*args, **kwargs)
            _workflow_events.publish(conv.username, conv.conv_id)
        def _partial(*args, **kwargs):
            conv._on_partial_result(*args, **kwargs)
            _workflow_events.publish(conv.username, conv.conv_id)
        conv.session._on_progress = _progress
        conv.session._on_partial_result = _partial
        # Wire user-interrupt (Claude-Code-style Esc): session asks the
        # conversation whether the user clicked "中断" while the agent was
        # thinking. Returns True → the loop FREEZES (no further LLM/tool calls).
        conv.session._on_interrupt_requested = (
            lambda c=conv: bool(getattr(c, "interrupt_requested", False))
        )
        # Optional redirect message attached to the interrupt ("中断加话和需求").
        conv.session._on_interrupt_message = (
            lambda c=conv: str(getattr(c, "interrupt_message", "") or "")
        )
        # After the redirect is consumed, clear it so it applies only once.
        def _clear_interrupt(c=conv):
            c.interrupt_message = ""
            c.interrupt_requested = False
            if c.session and c.session._runtime_snapshot and c.session._runtime_snapshot():
                from agents.parallel_workflow import WorkflowStore
                store = WorkflowStore(c.session.config.project_root / 'data/state/parallel_workflows.json')
                store.user_pause(store.identity(c.username, c.conv_id), False)
        conv.session._on_interrupt_clear = _clear_interrupt
        recovered = conv.session.context.pop('preflight_recovery_ready', None)
        if recovered:
            conv.session._emit_lifecycle_event('preflight_recovered', {
                **recovered,
                'requires_user': False,
                'main_already_handling': False,
                'instruction': (
                    'The validated complete workflow draft was restored automatically. '
                    'Continue with propose_workflow_patch and execution; do not ask the user '
                    'about the repaired internal guard.'
                ),
            })
            conv.session._checkpoint('preflight_recovery_scheduled')
    return conv.session


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

class QueryResponse(BaseModel):
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
    session_manager.persist_conversations(username, conv_id=conv.conv_id)
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
        "workflow": _workflow_state(conv),
    }


@app.get("/api/conversations/{conv_id}/workflow")
async def get_workflow(conv_id: str, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    return _workflow_state(conv)


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
        session_manager.persist_conversations(
            username,
            conv_id=user.current_conv_id,
            delete_conv_id=conv_id,
        )
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="会话不存在")


@app.post("/api/conversations/{conv_id}/rename")
async def rename_conversation(conv_id: str, req: dict, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    conv = user.get_conversation(conv_id)
    if not conv:
        raise HTTPException(status_code=404, detail="会话不存在")
    conv.title = req.get("title", conv.title)[:50]
    session_manager.persist_conversations(username, conv_id=conv_id)
    return {"status": "renamed", "title": conv.title}


@app.post("/api/conversations/{conv_id}/switch")
async def switch_conversation(conv_id: str, username: str = Depends(_get_user)):
    user = session_manager.get_user(username)
    if conv_id not in user.conversations:
        raise HTTPException(status_code=404, detail="会话不存在")
    user.current_conv_id = conv_id
    session_manager.persist_conversations(username, conv_id=conv_id)
    return {"status": "switched", "conv_id": conv_id}


# ── Query endpoints ──
@app.get("/api/latest_answer")
async def get_latest_answer(username: str = Depends(_get_user), conv_id: str = ""):
    conv = _get_conv(username, conv_id)
    return {
        "answer": conv.current_answer,
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
        "workflow": _workflow_state(conv),
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
                revision = current
                payload = {
                    'revision': revision,
                    'conv_id': conv_id,
                    'workflow': _workflow_state(conv),
                    'done': not conv.is_processing,
                    'agent_name': conv.current_agent,
                    'status': conv.current_status,
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
        'Cache-Control': 'no-cache, no-store',
        'X-Accel-Buffering': 'no',
    })


@app.post("/api/query", response_model=QueryResponse)
async def process_query(request: QueryRequest, username: str = Depends(_get_user)):
    conv = _get_conv(username, request.conv_id)
    new_uid = str(datetime.now().timestamp())

    if conv.is_processing:
        # 用户中断（freeze）后发的新消息 = Claude-Code 的 Esc+typing 重定向。
        # 关键：绝不能 429 拒绝——否则用户"中断了发消息又说正在处理中"。
        # 这里清除冻结标志（这条新消息就是续写指令），排队等当前轮真正停止
        # 后自动执行；若当前轮只是卡在长工具调用（run_bash 等），排队会在
        # 其超时返回后立即跑，而不是把用户锁死。
        if getattr(conv, "interrupt_requested", False):
            conv.interrupt_requested = False
            conv.interrupt_message = ""
            conv.current_uid = new_uid  # 新消息接管本轮所有权
            conv.queued_redirect = request.query  # 前端可显示"已排队"
            conv.current_status = "⛔ 已中断；收到新消息，当前轮停止后立即继续"
            asyncio.create_task(_run_after_current_turn(username, conv.conv_id, request.query, new_uid))
            return QueryResponse(
                answer="⛔ 已中断当前轮，收到你的新消息；当前工具调用结束后立即继续执行（无需再发消息）。",
                reasoning=f"已收到新需求（中断重定向）: {request.query}\n\n当前轮停止后立即执行...",
                agent_name="orchestrator",
                status="⛔ 已中断（新消息排队中）",
                uid=new_uid,
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
    conv.current_uid = new_uid
    conv.is_processing = True  # reserve before create_task yields to another request
    _workflow_events.publish(username, conv.conv_id)
    asyncio.create_task(run_agent_background(username, request.query, conv.conv_id))

    return QueryResponse(
        answer="正在处理您的请求...",
        reasoning=f"收到请求: {request.query}\n\n开始分析...",
        agent_name="orchestrator",
        status="Processing",
        uid=new_uid,
        done=False,
        conv_id=conv.conv_id,
    )


async def _run_after_current_turn(username: str, conv_id: str, query: str, uid: str):
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
            session_manager.persist_conversations(username, conv_id=conv.conv_id)
        except Exception:
            pass
        return
    # Old thread exited — run the redirect as a normal turn (is_processing is
    # already held True by the queue path; run_agent_background keeps it True).
    conv.queued_redirect = ""
    await run_agent_background(username, query, conv_id)


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
    conv.interrupt_requested = True
    if conv.session and conv.session._runtime_snapshot and conv.session._runtime_snapshot():
        from agents.parallel_workflow import WorkflowStore
        store = WorkflowStore(conv.session.config.project_root / 'data/state/parallel_workflows.json')
        store.user_pause(store.identity(username, conv_id))
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
    _ensure_session(conv)
    conv.session._emit_lifecycle_event('user_paused', {'redirect': _redirect, 'user_requested_freeze': True})
    if _redirect and conv.is_processing:
        conv.current_status = "⛔ 用户中断：冻结本轮，稍后自动用你的需求继续"
        try:
            session_manager.persist_conversations(username, conv_id=conv.conv_id)
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
        session_manager.persist_conversations(username, conv_id=conv.conv_id)
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
    conv.interrupt_requested = False
    conv.interrupt_message = ""
    _ensure_session(conv)
    if conv.session and conv.session._runtime_snapshot and conv.session._runtime_snapshot():
        from agents.parallel_workflow import WorkflowStore
        store = WorkflowStore(conv.session.config.project_root / 'data/state/parallel_workflows.json')
        store.user_pause(store.identity(username, conv_id), False)
    try:
        session_manager.persist_conversations(username, conv_id=conv.conv_id)
    except Exception:
        pass
    return {"ok": True, "interrupt_requested": False}


# ── Background agent execution ──
async def run_agent_background(username: str, query: str, conv_id: str, lifecycle_event=None):
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
    conv.messages_history.append({"role": "system" if lifecycle_event else "user", "content": query})
    # 仅清理遗留的旧 "已执行的计算" context 占位块（一次性脏数据）
    conv.messages_history = [m for m in conv.messages_history if m.get("role") != "context"]
    try:
        session_manager.persist_conversations(username, conv_id=conv.conv_id)
    except Exception:
        pass

    # Build context from conversation history (already in session.messages, but keep for display)
    def _run_in_thread():
        # Never replace process-global sys.stdout here. Multiple conversations
        # run in parallel, and swapping stdout in one worker caused logs from one
        # user/session to leak into another. Structured progress callbacks are
        # the supported logging path.
        # Publish thread-local context so tool executors can auto-register
        # SLURM jobs with JobWatch (with the right conv_id/user).
        from agents.watch_context import set_context, clear_context
        set_context(username, conv.conv_id, conv.current_agent or "orchestrator", conv.session._current_line_id)
        conv.session._api_turn_id = turn_uid
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
            clear_context()

    conv_lock = _get_conv_lock(username, conv.conv_id)
    # Per-conversation lock. The /api/query layer already rejects a second
    # query while this conversation is processing (429 via conv.is_processing),
    # so this lock is a belt-and-suspenders for the same-conversation race
    # only. It must NOT serialize across conversations: the harness and a
    # smoke-test conversation of the same user legitimately run together.
    # A bounded wait keeps the old "silent empty answer" failure visible:
    # if the lock can't be acquired, surface the reason in the answer.
    if not await asyncio.to_thread(conv_lock.acquire, timeout=60):
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
            session_manager.persist_conversations(username, conv_id=conv.conv_id)
        except Exception:
            pass
        return

    try:
        # Another API process may have finished this conversation while we waited
        # for its kernel lease. Restore its newer execution state before replying.
        from agents.workspace import session_root
        checkpoint = session_root(conv.conv_id, username) / 'session_checkpoint.json'
        if checkpoint.exists():
            saved = json.loads(checkpoint.read_text())
            if saved.get('checkpoint_at', 0) > getattr(conv.session, '_last_checkpoint_at', 0):
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

        conv.current_answer = result
        # 用户消息已在 turn 开始时落盘（防中途崩溃/重启吞掉提问），
        # 这里只补上 assistant 回复。本地对话历史完整落盘，不截断不设限；
        # 压缩只对"发给 LLM 的上下文"做（session._trim_messages）。
        conv.current_agent = conv.session.current_agent.name if conv.session.current_agent else "orchestrator"
        conv.current_status = _execution_status(conv.session)
        conv.messages_history.append({'role': 'assistant', 'content': result if result else '',
                                      'agent': 'lead-orchestrator', 'timestamp': time.time(),
                                      'status': conv.current_status, 'uid': turn_uid,
                                      'delivery_owner': 'lead-orchestrator', 'negotiation_owner': 'lead-orchestrator'})
        conv.execution_logs.append({'time': datetime.now().isoformat(), 'message': conv.current_status})

        # Auto-title: first user message becomes the title
        if conv.title == "新对话" and conv.messages_history:
            first_user = next((m["content"] for m in conv.messages_history if m.get("role") == "user"), "")
            if first_user:
                conv.title = first_user[:30]

        # Persist conversation to log
        try:
            _append_conversation_log([
                {"time": datetime.now().isoformat(), "username": username,
                 "conv_id": conv_id, "role": role, "content": content}
                for role, content in [("user", query), ("assistant", result or "")]
            ])
        except Exception:
            pass

        # Persist conversations to users.json so a server restart does not
        # orphan JobWatch notifications (conv "gone" → dropped). Without this,
        # a job that fails hours after a restart never wakes its agent.
        try:
            session_manager.persist_conversations(username, conv_id=conv.conv_id)
        except Exception:
            pass

    except Exception as e:
        conv.current_status = f"Error: {str(e)}"
        _err = f"处理请求时出错: {str(e)}"
        conv.current_answer = _err
        # 用户消息已在 turn 开始时落盘，这里补上出错回复，保证问题+回复都不消失
        conv.messages_history.append({"role": "assistant", "content": _err})
        conv.execution_logs.append({"time": datetime.now().isoformat(), "message": f"❌ 错误: {str(e)}"})
        try:
            session_manager.persist_conversations(username, conv_id=conv.conv_id)
        except Exception:
            pass
    finally:
        # 只有仍拥有本轮所有权时才清 is_processing/interrupt；若已被新消息
        # 接管（uid 更新），绝不动新轮的状态（否则会把新 turn 标成空闲/清掉
        # 它的中断，造成"发了又处理中/又冻结"）。
        if conv.current_uid == turn_uid:
            conv.is_processing = False
            if conv.session:
                conv.session._api_turn_id = turn_uid
                conv.session._persist_chain('api_turn_end')
                conv.session._api_turn_id = None
            # A turn always ends — clear any pending user-interrupt flag so a later
            # "继续" message doesn't abort immediately.
            # Preserve an explicit user freeze until their next directive;
            # scheduler/supervisor events may not silently unfreeze it.
            # ⚠️ 关键：上面的 is_processing=False 必须在**清标志后**再持久化一次。
            # 否则 users.json 会一直存着 is_processing=True（第574行 persist 跑在
            # finally 之前），服务重启时 auth 自愈会把**已正常完成**的对话误标成
            # "已中断（服务重启）"——用户切 session / 刷新后"响应消失"。
            try:
                session_manager.persist_conversations(username, conv_id=conv.conv_id)
            except Exception:
                pass
            _workflow_events.publish(username, conv.conv_id)
        # A framework patcher may have requested a backend restart so its code
        # edits take effect. Do it ONLY after the reply is written + persisted
        # (never mid-turn), and detached so the new process survives us.
        try:
            from agents._restart import consume
            if consume():
                asyncio.create_task(_perform_backend_restart())
        except Exception as _e:
            print(f"  [backend] restart consume error: {_e}", flush=True)
        try:
            conv_lock.release()
        except RuntimeError:
            pass  # already released by an overlapping force-release


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
    busy = any(c.is_processing for u in session_manager.list_users() for c in session_manager.get_user(u).conversations.values())
    busy = busy or any(agent.get('busy_event_id') for u in session_manager.list_users()
        for c in session_manager.get_user(u).conversations.values() if c.session and c.session._lifecycle_store
        for agent in c.session._lifecycle_store.snapshot().get('agents', {}).values())
    busy = busy or any(r.futures and any(not f.done() for f in r.futures.values()) for r in _workflow_runtimes.values())
    if busy:
        from agents._restart import request
        request()
        print('[backend] administrator restart deferred: other user/session turns still active', flush=True)
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
        "total_users_registered": len(load_users()) if USER_DB.exists() else 0,
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
    main = _ensure_session(conv)
    if main is None:
        raise ValueError('session could not be restored')
    key = (conv.username, conv.conv_id)
    if key not in _workflow_runtimes:
        _workflow_runtimes[key] = ParallelWorkflow(main, session_root(conv.conv_id, conv.username),
            WorkflowStore(main.config.project_root / 'data/state/parallel_workflows.json'),
            main._lifecycle_store, get_store(), conv.username, conv.conv_id, job_watch=_watch,
            on_change=lambda reason: _workflow_events.publish(conv.username, conv.conv_id))
    _workflow_runtimes[key].main = main
    return _workflow_runtimes[key]


class WorkflowExecutionRequest(BaseModel):
    plan_version: int


def _start_workflow_runtime(conv, plan_version):
    runtime = _ensure_workflow_runtime(conv)
    jobs = [job for job in _watch.list()
            if (job.get('username'), job.get('conv_id')) == (conv.username, conv.conv_id)]
    return runtime.start_and_tick(plan_version, jobs=jobs,
                                  paused=bool(conv.interrupt_requested))


class WorkerMessageRequest(BaseModel):
    text: str
    kind: str = 'comment'
    message_id: Optional[str] = None


@app.post('/api/conversations/{conv_id}/workflow/execute')
async def execute_parallel_workflow(conv_id: str, request: WorkflowExecutionRequest, username: str = Depends(_get_user)):
    conv = session_manager.get_user(username).get_conversation(conv_id)
    if conv is None: raise HTTPException(404, 'conversation not found')
    if conv.is_processing or conv.interrupt_requested: raise HTTPException(409, 'main turn busy or user paused; approve through main chat')
    try:
        result = _start_workflow_runtime(conv, request.plan_version)
        conv.session.task_complete = False
        conv.conversation_active = True
        conv.session._checkpoint('workflow_api_start')
        session_manager.persist_conversations(username, conv_id=conv.conv_id)
        return result
    except ValueError as error: raise HTTPException(409, str(error))


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
    observer._chain_state_is_primary = False
    observer._chain_jobs = main._chain_jobs
    # A user freeze covers the full-lifecycle pair, not only main chat.  The
    # observer stops at the next model/tool boundary and its durable event is
    # retried after resume; it must not keep draining a backlog during deploy.
    observer._on_interrupt_requested = lambda c=conv: bool(getattr(c, 'interrupt_requested', False))
    observer._on_interrupt_message = lambda c=conv: str(getattr(c, 'interrupt_message', '') or '')
    observer.context['main_evidence_calls'] = main._recovery_state()['evidence_calls']
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
    if not lock.acquire(blocking=False): return {'status':'review_retry','retryable':True,'reason':'resource agent is reviewing this session; do not dispatch duplicate work'}
    previous=get_context()
    try:
        set_context(conv.username,conv.conv_id,'monitor',conv.session._current_line_id)
        observer=Session(config=conv.session.config,registry=conv.session.registry)
        observer.client=observer.client.with_options(timeout=60,max_retries=0)
        observer.goal_contract=copy.deepcopy(conv.session.goal_contract)
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
            from agents.resource_review import allocate_reviewed_resources
            receipt = allocate_reviewed_resources(observer, request, receipt)
        write_checkpoint(root/'resource_review.json',receipt)
        known={c.get('call_id') for c in conv.session.memory.tool_call_log}
        for call in receipt.get('evidence_calls',[]):
            if call.get('call_id') not in known:conv.session.memory.tool_call_log.append(copy.deepcopy(call))
        conv.session._checkpoint('resource_monitor_receipt')
        return receipt
    finally:
        set_context(**previous)
        lock.release()


async def _supervise_lifecycle_event(conv, event):
    store = conv.session._lifecycle_store
    action = ('ask_user' if event.get('payload', {}).get('requires_user') else
              'advance_dependencies' if event['kind'] in {'job_completed', 'result_verified'} else
              'wait_existing' if event['kind'] in {'tool_uncertain', 'stalled', 'user_paused', 'waiting_jobs'} else 'diagnose_and_fix')
    provisional = {'next_action': action, 'reason': '控制层只允许诊断/等终态/推进依赖；详细只读审核异步进行，不阻塞主chat推进',
                   'evidence_refs': [event['event_id']]}
    store.provisional(event['event_id'], provisional)
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
                    event['payload']['requires_user']=resource_receipt.get('status') in {'needs_user','propose_resource_patch'}
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
        store.receipt(event['event_id'], 'supervisor', 'delivered', receipt)
        if event['kind'] == 'scientific_review':
            action = receipt.get('review_action')
            if action in {'verified', 'obsolete'}:
                store.receipt(event['event_id'], 'main_chat', 'delivered' if action == 'verified' else 'obsolete',
                              {'status': action, 'delivery_owner': 'lead-orchestrator', 'not_resubmitted': True})
            elif action in {'correct_main', 'ask_user'}:
                payload = {**event['payload'], 'main_already_handling': False, 'supervisor_decision': receipt,
                           'requires_user': action == 'ask_user'}
                review_id = store.emit('scientific_review_failed', payload, event_id=event['event_id'] + ':feedback')
                store.receipt(review_id, 'supervisor', 'delivered', receipt)
                store.receipt(event['event_id'], 'main_chat', 'delivered', {'status':'forwarded_to_main', 'delivery_owner':'lead-orchestrator'})
            session_manager.persist_conversations(conv.username, conv_id=conv.conv_id)
            return
        latest = store.snapshot()['events'][event['event_id']]
        if (event['kind'] not in {'user_paused', 'stalled', 'waiting_jobs', 'tool_uncertain', 'supervisor_review'}
                and latest.get('main_chat') == 'delivered'
                and (receipt.get('next_action') != provisional['next_action'] or receipt.get('suggested_changes')
                     or receipt.get('scientific_review', {}).get('passed') is False)):
            review_id = store.emit('supervisor_review', {
                **event.get('payload', {}), 'parent_event_id': event['event_id'], 'supervisor_decision': receipt,
            }, event_id=event['event_id'] + ':review')
            store.receipt(review_id, 'supervisor', 'delivered', receipt)
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
            review_id = store.emit('scientific_review_failed', {**event['payload'], 'main_already_handling':False,
                                   'requires_user':True}, event_id=event['event_id'] + ':infrastructure')
            store.receipt(review_id, 'supervisor', 'delivered', fallback)
        store.receipt(event['event_id'], 'supervisor', 'delivered', fallback)


async def _continue_lifecycle_event(conv, event):
    store = conv.session._lifecycle_store
    try:
        await run_agent_background(conv.username, f"[生命周期事件] {event['kind']}：监督已交回主chat处理",
                                   conv.conv_id, lifecycle_event=event)
        store.receipt(event['event_id'], 'main_chat', 'delivered', {
            'turn_uid': conv.current_uid, 'status': conv.current_status,
            'negotiation_owner': 'lead-orchestrator', 'delivery_owner': 'lead-orchestrator',
        })
    except Exception as error:
        store.receipt(event['event_id'], 'main_chat', 'retry', {'error': str(error)})
        conv.is_processing = False


def _schedule_lifecycle(coro):
    task = asyncio.create_task(coro)
    _lifecycle_tasks.add(task)
    task.add_done_callback(_lifecycle_tasks.discard)


async def _dispatch_lifecycle_once():
    for username in session_manager.list_users():
        user = session_manager.get_user(username)
        for conv in list(user.conversations.values()):
            if not conv.session:
                continue
            store = conv.session._lifecycle_store
            if not store:
                continue
            store.recover_claims()
            store.heartbeat()
            runtime_state = conv.session._runtime_snapshot() if conv.session._runtime_snapshot else {}
            if runtime_state:
                runtime = _ensure_workflow_runtime(conv)
                jobs = [j for j in _watch.list() if j.get('username') == username and j.get('conv_id') == conv.conv_id]
                await asyncio.to_thread(runtime.tick, jobs, bool(conv.interrupt_requested))
                # tick() can complete or pause the graph, so use its current
                # durable state for all wake-up decisions below.
                runtime_state = conv.session._runtime_snapshot() or {}
            formal_runtime_active = bool(
                runtime_state.get('nodes')
                and runtime_state.get('status') in {'active', 'needs_user'}
                and not runtime_state.get('user_paused')
            )
            # Freeze is session-wide: keep durable mailbox events pending, but
            # do not claim fresh main/supervisor work until the user resumes.
            if conv.interrupt_requested or conv.interrupt_message or runtime_state.get('user_paused'):
                continue
            if conv.is_processing and conv.last_progress_at and time.time() - conv.last_progress_at > 360:
                store.emit('stalled', {'workflow_id': conv.session._current_line_id,
                                      'current_agent': conv.current_agent, 'last_progress_at': conv.last_progress_at},
                           event_id='stalled:' + str(conv.current_uid))
            for event in store.pending():
                # A dead process may have left a receiver in `processing`; its
                # claim is recovered as `retry`.  Replaying that event for an
                # ordinary/non-DAG chat caused the observed "old session spoke
                # by itself" loop.  Archive it for audit instead.  An active
                # formal DAG is deliberately exempt, because its persisted
                # node/job state proves there is real work to continue.
                if float(event.get('time', 0) or 0) < _PROCESS_STARTED_AT and not formal_runtime_active:
                    details = {
                        'reason': 'pre-restart event has no active formal workflow',
                        'process_started_at': _PROCESS_STARTED_AT,
                        'not_resubmitted': True,
                    }
                    if event.get('supervisor') not in {'delivered', 'obsolete'}:
                        store.receipt(event['event_id'], 'supervisor', 'obsolete', details)
                    if event.get('main_chat') not in {'delivered', 'obsolete'}:
                        store.receipt(event['event_id'], 'main_chat', 'obsolete', details)
                    continue
                if (event['kind'] == 'worker_validation_dossier_ready'
                        and event.get('supervisor') in {'pending', 'retry'}):
                    dossier = event.get('payload', {}).get('validation_dossier') or {}
                    store.receipt(event['event_id'], 'supervisor', 'delivered', {
                        'next_action': 'advance_dependencies' if dossier.get('verdict') == 'pass' else 'diagnose_and_fix',
                        'reason': 'executor-bound validation dossier is already kernel verified',
                        'evidence_refs': [dossier.get('evidence_call_id')],
                        'scientific_review': {'passed': dossier.get('verdict') == 'pass', 'issues': dossier.get('failed_items', [])},
                    })
                    event = store.snapshot().get('events', {}).get(event['event_id'], event)
                if (event['kind'] == 'worker_recovery_ready'
                        and event.get('supervisor') in {'pending', 'retry'}):
                    recovery = event.get('payload', {}).get('recovery') or {}
                    store.receipt(event['event_id'], 'supervisor', 'delivered', {
                        'next_action': 'diagnose_and_fix',
                        'reason': 'durable workflow state exposes a bounded recovery capability',
                        'evidence_refs': [event['event_id']],
                        'suggested_changes': [{
                            'action': 'call_tool', 'tool': recovery.get('tool'),
                            'step_id': event.get('payload', {}).get('step_id'),
                        }],
                        'scientific_review': {'passed': True, 'issues': []},
                    })
                    event = store.snapshot().get('events', {}).get(event['event_id'], event)
                if event.get('supervisor') in {'pending', 'retry'} and store.claim(event['event_id'], 'supervisor'):
                    _schedule_lifecycle(_supervise_lifecycle_event(conv, event))
                    continue
                if (event.get('supervisor') not in ({'delivered'} if event['kind']=='job_pending_warning' else {'delivered','processing'}) or not event.get('supervisor_receipt')
                        or event.get('main_chat') not in {'pending', 'retry', 'waiting_user'}):
                    continue
                payload = event.get('payload', {})
                if payload.get('runtime_id') and event['kind'] == 'workflow_started':
                    store.receipt(event['event_id'], 'main_chat', 'delivered', {'status': 'runtime scheduled; main remains owner'})
                    continue
                if payload.get('main_already_handling') or event['kind'] in {'user_paused', 'tool_uncertain', 'stalled', 'waiting_jobs', 'plan_patch_applied'}:
                    store.receipt(event['event_id'], 'main_chat', 'delivered', {'status': 'notification_only', 'not_resubmitted': True})
                    continue
                old_goal = payload.get('origin_goal_version', payload.get('goal_contract', {}).get('version'))
                node = runtime_state.get('nodes', {}).get(payload.get('step_id'), {})
                reconciled_event = bool(node.get('token')
                    and payload.get('plan_version') == runtime_state.get('plan_version')
                    and runtime_state.get('goal_contract', {}).get('version') == conv.session.goal_contract.version
                    and event['event_id'] == f"runtime:{payload.get('runtime_id')}:{node['token']}:{node.get('status')}"
                    and any(h.get('from_version') == old_goal for h in runtime_state.get('goal_reconciliation_history', [])))
                if old_goal is not None and old_goal != conv.session.goal_contract.version and not reconciled_event:
                    store.receipt(event['event_id'], 'main_chat', 'obsolete', {'reason': 'user superseded the goal'})
                    continue
                from agents.lifecycle import obsolete_internal_question_for_recovery
                if obsolete_internal_question_for_recovery(conv.session, event):
                    conv.current_status = '恢复工作流'
                pending = conv.session._pending_user_interaction or {}
                if conv.session._awaiting_plan_approval or pending.get('tool') in {'failure_decision', 'workflow_patch_decision', 'user_decision'}:
                    store.receipt(event['event_id'], 'main_chat', 'waiting_user', {'negotiation_owner': 'lead-orchestrator'})
                    continue
                if conv.is_processing or conv.interrupt_requested or conv.interrupt_message:
                    continue
                if event.get('main_chat') == 'waiting_user':
                    store.receipt(event['event_id'], 'main_chat', 'retry')
                if store.claim(event['event_id'], 'main_chat'):
                    # Reserve synchronously, before any await/task scheduling.
                    # Otherwise several events could start overlapping main turns.
                    conv.is_processing = True
                    conv.current_uid = str(time.time())
                    _schedule_lifecycle(_continue_lifecycle_event(conv, event))


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
            notifs = _watch.drain_notifications()
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
                _watch.ack_notification(str(n.get("job_id", "")),n.get('notification_id',''))
                continue
            try:
                # Wake the conversation (session.reply → agent fault handling).
                user = session_manager.get_user(username)
                conv = user.get_conversation(conv_id) if user else None
                if conv is None:
                    # conversation deleted while job was running
                    print(f"  [JobWatch] conv {conv_id} gone — notification dropped", flush=True)
                    _watch.ack_notification(str(n.get("job_id", "")),n.get('notification_id',''))
                    continue
                _ensure_session(conv)
                rec = _watch.get(str(n.get('job_id'))) or {}
                kind=n.get('kind') or ('job_failed' if rec.get('failed') else 'job_completed')
                conv.session._lifecycle_store.emit(kind, {
                    'job': rec, 'notification': n, 'workflow_id': conv.session._current_line_id,
                    'goal_contract': conv.session.goal_contract.to_dict(),
                    'instruction':'PENDING is nonterminal: delegate resource review, preserve leases and original job, never label completion or auto-resubmit.' if kind=='job_pending_warning' else '',
                }, event_id=f"job:{n.get('job_id')}:{n.get('notification_id') or n.get('state')}")
                # Ack only after the durable two-receiver mailbox has accepted it.
                _watch.ack_notification(str(n.get("job_id", "")),n.get('notification_id',''))
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
    for j in _watch.list():
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
    rec = _watch.get(str(job_id).strip())
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
    record = _watch.get(job_id)
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
