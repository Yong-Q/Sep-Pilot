# -*- coding: utf-8 -*-
"""
TaskLine — 流水线任务线记录器。

解决的核心问题：agent 之间只通过 handoff 文本传上下文，工具返回的产物路径
没有落盘，导致下游 agent 不知道上一步产物在哪，只能全局 grep 撞历史文件
（job 3669 就 grep 到了 5-29 的 GCMC 输出，误诊成力场缺失）。

TaskLine 提供一条**持久化、可查询的任务线**：
  - 一条线 = 一个流水线（line_id 即任务号）
  - 每步记录：tool / job_ids / 输入目录 / 输出目录 / 产物文件 / done(BOOL)
  - done 由 JobWatch 定格驱动（terminal+COMPLETED→True, FAILED→False）
  - 整条线的 agent/tool 都可按 line_id 或 conv_id 查询，拿到路径直接继续

状态持久化到 JSON 文件，重启不丢。
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from .slurm import normalize_job_id
from typing import Any, Dict, List, Optional

LINE_FILE = str(Path(__file__).resolve().parents[1] / 'data' / 'state' / 'task_lines.json')
LEGACY_LINE_FILE = os.path.join(tempfile.gettempdir(), 'bimem_task_line.json')


def _scope(username=None, conv_id=None):
    if username is None and conv_id is None:
        from .watch_context import get_context
        context = get_context()
        return context.get('username', ''), context.get('conv_id', '')
    return username or '', conv_id or ''


class TaskLineStore:
    """Persistent store of pipeline task lines."""

    def __init__(self, file_path: str = ""):
        self._file = file_path or LINE_FILE
        self._lines: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load()

    # ── persistence ────────────────────────────────────────────────
    def _load(self) -> None:
        try:
            source = self._file
            if self._file == LINE_FILE and not os.path.isfile(source):
                source = LEGACY_LINE_FILE
            if os.path.isfile(source):
                d = json.loads(Path(source).read_text())
                if isinstance(d, dict):
                    self._lines = d
                    if (self._file == LINE_FILE and source == self._file and os.path.isfile(LEGACY_LINE_FILE)
                            and os.path.getmtime(LEGACY_LINE_FILE) > os.path.getmtime(self._file)):
                        legacy = json.loads(Path(LEGACY_LINE_FILE).read_text())
                        for key, value in legacy.items():
                            if key not in d or str(value.get('updated_at', '')) >= str(d[key].get('updated_at', '')):
                                self._lines[key] = value
                else:
                    raise ValueError('task line file must contain a JSON object')
        except Exception as error:
            raise RuntimeError(f'task line load failed; preserving source file {source}: {error}') from error

    def _save(self) -> None:
        try:
            target = Path(self._file)
            target.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(prefix=target.name + ".", dir=str(target.parent))
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._lines, f, ensure_ascii=False, indent=1)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, target)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        except Exception as error:
            raise RuntimeError(f'task line commit failed: {self._file}: {error}') from error

    @contextmanager
    def _process_guard(self):
        """Cross-process guard for the JSON store (Linux deployment)."""
        lock_path = self._file + ".lock"
        Path(lock_path).parent.mkdir(parents=True, exist_ok=True)
        fh = open(lock_path, "a+")
        try:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
            # Other workers may have committed since this process last loaded.
            self._load()
            yield
        finally:
            try:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            fh.close()

    # ── lifecycle ──────────────────────────────────────────────────
    def _scoped_key(self, line_id, username, conv_id):
        if not username or not conv_id: return line_id
        import hashlib
        line = self._lines.get(line_id)
        if line is None or (line.get('username') == username and line.get('conv_id') == conv_id): return line_id
        return 'scope:' + hashlib.sha256(json.dumps([username, conv_id, line_id]).encode()).hexdigest()[:32]

    def begin_line(self, line_id: str, username: str = "", conv_id: str = "",
                   title: str = "") -> Dict[str, Any]:
        """Open a new pipeline line (idempotent)."""
        line_id = str(line_id).strip()
        if not line_id:
            line_id = conv_id or f"line_{int(time.time())}"
        with self._lock, self._process_guard():
            line_id = self._scoped_key(line_id, username, conv_id)
            if line_id not in self._lines:
                self._lines[line_id] = {
                    "line_id": line_id,
                    "username": username,
                    "conv_id": conv_id,
                    "title": title or line_id,
                    "steps": [],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._save()
            else:
                # Fill ownership fields if this line was created earlier
                # without them (e.g. a bare upsert_step before context existed).
                cur = self._lines[line_id]
                changed = False
                if username and not cur.get("username"):
                    cur["username"] = username; changed = True
                if conv_id and not cur.get("conv_id"):
                    cur["conv_id"] = conv_id; changed = True
                if changed:
                    self._save()
            return self._lines[line_id]

    def upsert_step(self, line_id: str, step_id: str, *, tool: str = "",
                    job_ids: Optional[list] = None, input_dir: str = "",
                    output_dir: str = "", output_files: Optional[list] = None,
                    done: bool = False, note: str = "", username: str = "",
                    conv_id: str = "", status: str = "", agent: str = "",
                    depends_on: Optional[list] = None,
                    arguments: Optional[Dict[str, Any]] = None,
                    validation: Optional[Dict[str, Any]] = None,
                    path_manifest: Optional[Dict[str, Any]] = None,
                    resource_allocation: Optional[Dict[str, Any]] = None,
                    effective_arguments: Optional[Dict[str, Any]] = None,
                    plan_version: int = 0, branch_parent: str = "") -> None:
        """Create or update one step on a line. Paths + BOOL are the payload.

        username/conv_id are captured from the thread-local execution context
        (api.py → watch_context) so a line never loses who owns it — the
        TaskLine query relies on conv_id to find the line for this conversation.
        """
        line_id = str(line_id).strip() or conv_id or f"line_{int(time.time())}"
        with self._lock, self._process_guard():
            ctx_user, ctx_conv = _scope()
            username, conv_id = username or ctx_user, conv_id or ctx_conv
            line_id = self._scoped_key(line_id, username, conv_id)
            line = self._lines.get(line_id)
            if line is None:
                line = {
                    "line_id": line_id, "username": username, "conv_id": conv_id,
                    "title": line_id, "steps": [],
                    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                self._lines[line_id] = line
            elif username and not line.get("username"):
                line["username"] = username
            if conv_id and not line.get("conv_id"):
                line["conv_id"] = conv_id
            for st in line["steps"]:
                # Explicit step IDs are authoritative. Two serial nodes can use
                # the very same tool/arguments before and after a file mutation.
                # Tool-name fallback used to overwrite the first node here.
                if st.get("step_id") == step_id:
                    if tool:
                        st["tool"] = tool
                    if job_ids:
                        st["job_ids"] = sorted(set(st.get("job_ids", []) + job_ids))
                    if input_dir:
                        st["input_dir"] = input_dir
                    if output_dir:
                        st["output_dir"] = output_dir
                    if output_files:
                        st["output_files"] = sorted(set(st.get("output_files", []) + output_files))
                    st["done"] = bool(done)
                    if status:
                        st["status"] = status
                    elif done:
                        st["status"] = "completed"
                    elif job_ids:
                        st["status"] = "submitted"
                    if agent:
                        st["agent"] = agent
                    if depends_on is not None:
                        st["depends_on"] = list(depends_on)
                    if arguments is not None:
                        st["arguments"] = arguments
                    if path_manifest is not None:
                        st['path_manifest'] = dict(path_manifest)
                    if validation is not None:
                        st["validation"] = validation
                    if resource_allocation is not None:
                        st["resource_allocation"] = dict(resource_allocation)
                    if effective_arguments is not None:
                        st["effective_arguments"] = dict(effective_arguments)
                    if plan_version:
                        st["plan_version"] = int(plan_version)
                    if branch_parent:
                        st["branch_parent"] = branch_parent
                    if note:
                        st["note"] = note
                    line["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save()
                    return
            line["steps"].append({
                "step_id": str(step_id),
                "tool": tool,
                "job_ids": list(job_ids or []),
                "input_dir": input_dir,
                "output_dir": output_dir,
                "output_files": list(output_files or []),
                "path_manifest": dict(path_manifest or {}),
                "done": bool(done),
                "status": status or ("completed" if done else ("submitted" if job_ids else "pending")),
                "agent": agent,
                "depends_on": list(depends_on or []),
                "arguments": dict(arguments or {}),
                "validation": dict(validation or {}),
                "resource_allocation": dict(resource_allocation or {}),
                "effective_arguments": dict(effective_arguments or {}),
                "plan_version": int(plan_version or 0),
                "branch_parent": branch_parent,
                "note": note or "",
            })
            line["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()

    def mark_job_done(self, job_id: str, done: bool, state: str = "",
                      failed: bool = False, output_files: Optional[list] = None) -> None:
        """Update every step that references this job_id. Called by JobWatch
        when a job reaches its terminal state (COMPLETED→done=True)."""
        job_id = str(job_id)
        _norm = normalize_job_id(job_id)
        with self._lock, self._process_guard():
            touched = False
            for line in self._lines.values():
                for st in line.get("steps", []):
                    _ids = st.get("job_ids", []) or []
                    if job_id in _ids or _norm in _ids or \
                       any(normalize_job_id(x) == _norm for x in _ids):
                        states = st.setdefault('job_states', {})
                        states[_norm] = {'done': bool(done), 'failed': bool(failed), 'state': state}
                        all_done = bool(_ids) and all(states.get(normalize_job_id(j), {}).get('done') for j in _ids)
                        any_failed = any(x.get('failed') for x in states.values())
                        prerequisite = st.get('validation', {}).get('chain_status') == 'waiting'
                        st['done'] = all_done and not prerequisite
                        st['status'] = ('failed' if any_failed else 'ready' if all_done and prerequisite else
                                        'completed' if all_done else 'unconfirmed' if state.upper() in {'UNKNOWN', 'UNCONFIRMED'} else 'running')
                        if state:
                            st["note"] = f"job {job_id} → {state}"
                        if output_files:
                            st["output_files"] = sorted(set(st.get("output_files", []) + output_files))
                        touched = True
                        line['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            if touched:
                self._save()

    def supersede_plan_versions(self, line_id: str, current_version: int) -> None:
        """Mark unfinished nodes from older plan versions as superseded."""
        with self._lock, self._process_guard():
            line = self._lines.get(str(line_id))
            if not line:
                return
            changed = False
            for st in line.get("steps", []):
                version = int(st.get("plan_version", 0) or 0)
                if version and version < int(current_version) and st.get("status") not in {
                    "completed", "failed", "superseded"
                }:
                    st["status"] = "superseded"
                    st["note"] = (st.get("note", "") + " [superseded by plan v%d]" % current_version).strip()
                    changed = True
            if changed:
                line["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._save()

    def apply_workflow_patch(self, line_id, changes, base_version, new_version, reason, project_root=None, *, runtime_nodes=None):
        from .workflow_patch import patched_graph
        with self._lock, self._process_guard():
            username, conv_id = _scope()
            line_id = self._scoped_key(str(line_id), username, conv_id)
            line = self._lines.get(str(line_id))
            if not line:
                raise ValueError('workflow line does not exist')
            version = int(line.get('plan_version') or max((s.get('plan_version', 0) for s in line.get('steps', [])), default=0))
            if version != int(base_version):
                raise ValueError('workflow changed since patch proposal; propose again')
            nodes, affected = patched_graph(line.get('steps', []), changes, project_root)
            for old in line.get('steps', []):
                if old.get('step_id') in affected and old.get('status') in {'running', 'submitted'}:
                    actual = (runtime_nodes or {}).get(old['step_id'], {})
                    if not (actual.get('status') == 'pending' and not actual.get('token')
                            and not actual.get('job_ids') and not actual.get('dispatch_phase') and not old.get('job_ids')):
                        raise ValueError('cannot replace a live node; first reconcile its jobs or ask user to cancel')
                    old.setdefault('status_reconciliation_history', []).append({'reported': old['status'],
                        'actual': 'pending', 'reason': 'executor has never claimed/dispatched this node', 'time': time.time()})
                    old.update(status='pending', done=False)
            line.setdefault('plan_history', []).append({
                'version': version, 'steps': line.get('steps', []), 'reason': reason,
            })
            for node in nodes:
                if node['step_id'] in affected:
                    node.update({'status': 'pending', 'done': False, 'job_ids': [], 'output_files': []})
                node['plan_version'] = int(new_version)
            line['steps'] = nodes
            line['plan_version'] = int(new_version)
            line['updated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self._save()
            return json.loads(json.dumps(line))

    def repair_output_contract(self,line_id,step_id,outputs,verification,plan_version, *, restoration=None):
        """Record an evidenced output-contract revision without re-dispatch."""
        with self._lock,self._process_guard():
            username,conv_id=_scope()
            key=self._scoped_key(str(line_id),username,conv_id)
            line=self._lines.get(key)
            if not line or int(line.get('plan_version',0))!=plan_version:
                raise ValueError('output repair requires the current scoped plan')
            step=next(n for n in line['steps'] if n['step_id']==step_id)
            step.setdefault('output_contract_history',[]).append({'before':step.get('expected_outputs'),
                'after':outputs,'verification':verification,'time':time.time()})
            step['expected_outputs']=outputs
            if restoration:
                step.setdefault('attempt_history', []).append({k: step.get(k) for k in ('status', 'done', 'job_ids', 'validation')})
                step.update(status='prefinish', done=False, job_ids=[restoration['completed_job_id']],
                            output_files=list(outputs), restored_result=restoration)
            self._save()

    # ── query (any agent/tool can call) ────────────────────────────
    def get_line(self, line_id: str, username=None, conv_id=None) -> Optional[Dict[str, Any]]:
        with self._lock, self._process_guard():
            username, conv_id = _scope(username, conv_id)
            key = self._scoped_key(str(line_id), username, conv_id)
            line = self._lines.get(key)
            if username and (line or {}).get('username') != username: return None
            if conv_id and (line or {}).get('conv_id') != conv_id: return None
            return json.loads(json.dumps(line)) if line else None

    def get_by_conv(self, conv_id: str, username=None) -> List[Dict[str, Any]]:
        with self._lock, self._process_guard():
            if username is None: username = _scope()[0]
            return [json.loads(json.dumps(l))
                    for l in self._lines.values() if l.get("conv_id") == conv_id and (not username or l.get('username') == username)]

    def get_by_job(self, job_id: str) -> List[Dict[str, Any]]:
        job_id = str(job_id)
        _norm = normalize_job_id(job_id)
        with self._lock, self._process_guard():
            return [json.loads(json.dumps(l))
                    for l in self._lines.values()
                    if any(job_id in (st.get("job_ids") or []) or
                           _norm in (st.get("job_ids") or []) or
                           any(normalize_job_id(x) == _norm
                               for x in (st.get("job_ids") or []))
                           for st in l.get("steps", []))]

    def summarize(self, line_id: str) -> str:
        """Human-readable digest an agent can paste into its context."""
        line = self.get_line(line_id)
        if not line:
            return f"[TaskLine] 线 {line_id} 不存在"
        rows = []
        for st in line.get("steps", []):
            flag = "✅完成" if st.get("done") else "⏳未完成"
            rows.append(
                f"  [{st.get('step_id')}] agent={st.get('agent') or '-'} "
                f"tool={st.get('tool','')} status={st.get('status') or ('completed' if st.get('done') else 'pending')} "
                f"depends_on={st.get('depends_on', [])} job={st.get('job_ids')} {flag}\n"
                f"      in:  {st.get('input_dir') or '-'}\n"
                f"      out: {st.get('output_dir') or '-'}"
                f"{'  ('+str(len(st.get('output_files',[])))+' files)' if st.get('output_files') else ''}"
            )
        return (f"[TaskLine] {line_id}（{line.get('title','')}）\n"
                + "\n".join(rows) if rows else f"[TaskLine] {line_id} 空")

    def all_lines(self) -> List[Dict[str, Any]]:
        with self._lock, self._process_guard():
            return [json.loads(json.dumps(l)) for l in self._lines.values()]


# Global singleton
_store = TaskLineStore()


def get_store() -> TaskLineStore:
    return _store
