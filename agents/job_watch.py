"""JobWatch — background SLURM job monitor.

The problem: a computation agent submits a job, ends its turn, and nothing
watches the job afterwards. If the job fails 10 minutes later, nobody tells
the agent → the failure goes unhandled (exactly the bug the user reported:
"提交的都直接C了 但是又没有调用故障处理机制").

JobWatch solves it:
  1. Every submitted SLURM job is auto-registered (via registry tool executors).
  2. A daemon thread polls check_job_status every 30s.
  3. When a job hits a terminal FAILURE state, JobWatch emits a *notification*
     into `drain_notifications()`. api.py drains it and wakes the responsible
     conversation (session.reply with the failure text) so the agent runs its
     fault-handling rules (diagnose → fix → retry, max retries enforced by
     session._failure_retries).
  4. `cancel()` gives the user/frontend an interrupt mechanism (scancel).

State is persisted to a JSON file so a server restart doesn't forget jobs.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
import threading
import time
from typing import Any, Dict, List, Optional
from pathlib import Path

from . import slurm

POLL_INTERVAL = 5           # seconds between polls (batch squeue is TTL-cached & local, cheap) — task bar sync within seconds
WATCH_FILE = str(Path(__file__).resolve().parents[1] / 'data' / 'state' / 'job_watch.json')
LEGACY_WATCH_FILE = os.path.join(tempfile.gettempdir(), 'bimem_job_watch.json')


class JobWatch:
    def __init__(self, watch_file: str = WATCH_FILE):
        self._watch_file = watch_file
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._pending_notifications: List[Dict[str, Any]] = []
        self._inflight_notifications: set[str] = set()
        self._deleted: set[str] = set()
        self._leader_fh = None
        self._load()

    # ── public API ──────────────────────────────────────────────────
    def register(self, job_id: str, work_dir: str = "", gas: str = "",
                 cif: str = "", temperature: float = 0.0, username: str = "",
                 conv_id: str = "", agent_name: str = "",
                 tool: str = "", n_retries: int = 0,
                 output_dir: str = "", charged_cifs: Optional[list] = None) -> None:
        """Track a submitted job. Auto-created on first register.

        output_dir / charged_cifs are the REAL product locations produced by
        the submitting tool (e.g. run_pacman_charge writes *_pacmof.cif into
        output_dir). They let the completion reminder tell the agent exactly
        where to find the previous step's artifacts — otherwise the agent
        blind-greps the tree and hits stale results from OTHER runs.
        """
        job_id = slurm.normalize_job_id(job_id)
        if not job_id:
            return
        from .watch_context import get_context
        context = get_context()
        if context.get('recovery_key') and context.get('recovery_path'):
            from .recovery import RecoveryGate
            RecoveryGate(path=context['recovery_path']).register_job(
                context['recovery_key'], context['attempt_id'], job_id, str(work_dir))
        with self._lock:
            self._load()
            if job_id in self._jobs:
                # Update metadata but keep first-seen time
                self._jobs[job_id].update({
                    "work_dir": work_dir or self._jobs[job_id].get("work_dir", ""),
                    "output_dir": output_dir or self._jobs[job_id].get("output_dir", ""),
                    "charged_cifs": charged_cifs or self._jobs[job_id].get("charged_cifs", []),
                    "gas": gas or self._jobs[job_id].get("gas", ""),
                    "cif": cif or self._jobs[job_id].get("cif", ""),
                    "username": username or self._jobs[job_id].get("username", ""),
                    "conv_id": conv_id or self._jobs[job_id].get("conv_id", ""),
                    "agent_name": agent_name or self._jobs[job_id].get("agent_name", ""),
                    'tool': tool or self._jobs[job_id].get('tool', ''),
                    'recovery_key': context.get('recovery_key') or self._jobs[job_id].get('recovery_key', ''),
                    'attempt_id': context.get('attempt_id') or self._jobs[job_id].get('attempt_id', ''),
                })
                self._save()
                return
            self._jobs[job_id] = {
                "job_id": job_id,
                'recovery_key': context.get('recovery_key', ''), 'attempt_id': context.get('attempt_id', ''),
                "work_dir": work_dir,
                "output_dir": output_dir,
                "charged_cifs": list(charged_cifs or []),
                "gas": gas,
                "cif": cif,
                "temperature": temperature,
                "username": username,
                "conv_id": conv_id,
                "agent_name": agent_name,
                "tool": tool,
                "n_retries": n_retries,
                "state": "PENDING",
                "last_state": "PENDING",
                "submitted_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "terminal": False,
                "failed": False,
                "notified": False,
                "diagnosis": None,
                "diagnosed": False,
                "diagnosed_at": "",
            }
            self._save()

    def unregister(self, job_id: str) -> None:
        with self._lock:
            key = str(job_id)
            self._jobs.pop(key, None)
            self._deleted.add(key)
            self._save()

    def list(self) -> List[Dict[str, Any]]:
        with self._lock:
            self._load()
            return [dict(j) for j in self._jobs.values()]

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            self._load()
            j = self._jobs.get(str(job_id))
            return dict(j) if j else None

    def cancel(self, job_id: str, host: str = "", control_root=None) -> Dict[str, Any]:
        """Request cancellation; only scheduler evidence proves termination.

        Never releases a workflow data lease or infers CANCELLED from exit 0.
        Local jobs stay local; a configured remote default is not job identity.
        """
        job_id = slurm.normalize_job_id(job_id)
        if not job_id or not job_id.isdigit():
            return {'cancelled': False, 'request_sent': False, 'error': 'numeric job ID required'}
        rec = self.get(job_id)
        if not rec:
            return {'job_id': job_id, 'cancelled': False, 'request_sent': False, 'error': 'untracked job'}
        from .state_io import ConversationLock
        root = Path(control_root) if control_root else Path(self._watch_file).parent
        lock = ConversationLock(root / 'job_control' / f'{job_id}.lock')
        if not lock.acquire(blocking=False):
            return {'job_id': job_id, 'cancelled': False, 'request_sent': False, 'blocked': True,
                    'message': '同一作业正在执行调度控制；保留运行租约，稍后核对。'}
        try:
            host = host or rec.get('scheduler_host', '')
            if rec.get('cancel_request',{}).get('status')=='confirmed' and rec.get('terminal') and rec.get('state')=='CANCELLED':
                return {'job_id':job_id,'cancelled':True,'request_sent':True,'scheduler_status':'CANCELLED',
                        'notified':False,'message':'原取消已由调度器确认；不重复发送。'}
            if rec.get('cancel_request', {}).get('status') in {'intent', 'sent', 'uncertain'}:
                evidence = slurm.check_job_status(job_id, host=host, work_dir=rec.get('work_dir', ''))
                confirmed = bool(evidence.get('terminal')) and str(evidence.get('status', '')).upper().startswith('CANCELLED')
                return {'job_id': job_id, 'cancelled': confirmed, 'request_sent': rec['cancel_request']['status'] == 'sent',
                        'scheduler_status': evidence.get('status', 'UNKNOWN'), 'notified': self.mark_cancelled(job_id) if confirmed else False,
                        'message': '调度器已确认取消。' if confirmed else '已有取消请求尚待核验；不重复发送、不释放运行租约。'}
            argv = ['ssh', host, f'scancel {job_id}'] if host else ['scancel', job_id]
            # Durable intent survives timeout/process restart; state is not terminal.
            with self._lock:
                self._jobs[job_id]['cancel_request'] = {'status': 'intent', 'requested_at': time.time(), 'host': host}
                self._save()
            proc = subprocess.run(argv,
                                  capture_output=True, text=True, timeout=15)
            ok = proc.returncode == 0
            with self._lock:
                self._jobs[job_id]['cancel_request'].update(status='sent' if ok else 'rejected',
                    exit_code=proc.returncode, error=proc.stderr[:2000])
                if ok: self._jobs[job_id]['cancelled_by_user'] = True
                self._save()
            if not ok:
                return {'job_id': job_id, 'cancelled': False, 'request_sent': False,
                        'message': proc.stderr.strip() or 'scancel failed', 'notified': False}
            evidence = slurm.check_job_status(job_id, host=host, work_dir=rec.get('work_dir', ''))
            confirmed = bool(evidence.get('terminal')) and str(evidence.get('status', '')).upper().startswith('CANCELLED')
            return {'job_id': job_id, 'cancelled': confirmed, 'request_sent': True,
                    'scheduler_status': evidence.get('status', 'UNKNOWN'),
                    'message': '调度器已确认取消。' if confirmed else '取消请求已发送，尚未确认终止；保留运行租约并继续监控。',
                    'notified': self.mark_cancelled(job_id) if confirmed else False}
        except Exception as e:
            return {"job_id": job_id, "cancelled": False, "outcome_unknown": True, "error": str(e),
                    'message': '取消结果未确认；核对原作业，不释放租约、不重提。'}
        finally:
            lock.release()

    def mark_diagnosed(self, job_id: str) -> bool:
        """Persist that a failed job was formally diagnosed (diagnose_job tool).

        A terminal+failed job that has been diagnosed is no longer an
        *unhandled* failure: the report gate lets it pass (the agent has
        analysed the root cause and will report the conclusion to the user).
        Stored on disk so the resolution survives a server restart — this is
        what unblocks conversations whose only "failure" is an already-diagnosed
        job (e.g. a defect-repro sample like zeo++ 3790).
        """
        job_id = slurm.normalize_job_id(job_id)
        if not job_id:
            return False
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return False
            rec["diagnosed"] = True
            rec["diagnosed_at"] = rec.get("diagnosed_at") or time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()
            return True

    def mark_cancelled(self, job_id: str) -> bool:
        """Mark a job as cancelled and queue a notification for the agent."""
        with self._lock:
            rec = self._jobs.get(str(job_id))
            if not rec:
                return False
            rec["state"] = "CANCELLED"
            if rec.get('cancel_request'):
                rec['cancel_request']['status'] = 'confirmed'
            rec["terminal"] = True
            rec["failed"] = True
            rec["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save()
        self._notify(job_id, "CANCELLED", "作业已被取消（scancel）。", {})
        return True

    def drain_notifications(self) -> List[Dict[str, Any]]:
        """Pop all pending notifications (called by api.py)."""
        with self._lock:
            out, self._pending_notifications = self._pending_notifications, []
            self._inflight_notifications.update(str(n.get("job_id")) for n in out if n.get("job_id"))
            return out

    def ack_notification(self, job_id: str, notification_id: str = '') -> None:
        """Acknowledge only after the responsible conversation accepted it."""
        job_id = str(job_id)
        with self._lock:
            rec = self._jobs.get(job_id)
            if rec and (not notification_id or rec.get('pending_notification', {}).get('notification_id') == notification_id):
                rec.pop("pending_notification", None)
                rec["notification_delivered_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._inflight_notifications.discard(job_id)
            if rec and rec.get('pending_notification') and notification_id:
                pending = rec['pending_notification']
                if not any(n.get('notification_id')==pending.get('notification_id') for n in self._pending_notifications):
                    self._pending_notifications.append(pending)
            self._save()

    def _queue_notification(self, rec: Dict[str, Any], payload: Dict[str, Any]) -> None:
        import uuid
        payload.setdefault('notification_id', uuid.uuid4().hex)
        job_id = str(payload.get("job_id", ""))
        rec["pending_notification"] = payload
        already_queued = any(str(n.get("job_id")) == job_id for n in self._pending_notifications)
        if not already_queued and job_id not in self._inflight_notifications:
            self._pending_notifications.append(payload)
        self._save()

    # ── internals ───────────────────────────────────────────────────
    def _notify(self, job_id: str, state: str, summary: str, diagnosis: Dict[str, Any]) -> None:
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            # Only notify once per failure unless the agent explicitly re-arms
            if rec.get("notified"):
                return
            rec["notified"] = True
            _gas = rec.get("gas") or ""
            _cif = rec.get("cif") or ""
            _agent = rec.get("agent_name") or ""
            _who = f"该作业由 **{_agent}** 提交" if _agent else "该作业由系统监控发现"
            _gasdesc = f"{_gas}@{_cif}" if (_gas and _cif) else (_gas or _cif or "通用作业")
            _work = rec.get("work_dir") or ""
            _prod = []
            if rec.get("output_dir"):
                _prod.append(f"产物目录: {rec['output_dir']}")
            _cc = rec.get("charged_cifs") or []
            if _cc:
                _prod.append(f"产物文件({len(_cc)}个): {_cc[0]}{' …' if len(_cc) > 1 else ''}")
            _loc = "、".join(x for x in ([f"工作目录: {_work}"] if _work else []) + _prod if x)
            msg = (
                f"⚠️ [作业故障提醒] SLURM 作业 {job_id} ({_gasdesc}) 状态为 **{state}**。"
                f"{_who}。\n"
                f"{summary}\n"
                f"{_loc}\n"
                "请立即按故障处理规则处理：委派提交该作业的专业Agent（如 adsorption/analyst/harness），"
                "用 diagnose_job(job_id=..., work_dir=...) 获取完整 stderr/run.log 和软件级原因，"
                "修复后重新提交；若无法修复，如实告知用户失败原因并给出替代方案，不要编造数据。"
            )
            if rec.get('cancelled_by_user'):
                msg = (f'⚠️ [作业故障提醒] 用户主动取消的作业{job_id}，调度器状态为{state}。{summary}\n{_loc}\n'
                       '这是用户停止请求，不是需要自动修复重提的计算故障。保留证据，向主chat交付取消状态；不得自动重提或解除用户暂停。')
            payload = {
                "job_id": job_id,
                "conv_id": rec.get("conv_id", ""),
                "username": rec.get("username", ""),
                "state": state,
                "gas": rec.get("gas", ""),
                "cif": rec.get("cif", ""),
                "message": msg,
                "diagnosis": diagnosis,
                "cancelled_by_user": bool(rec.get('cancelled_by_user')),
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._queue_notification(rec, payload)

    def _notify_completion(self, job_id: str, state: str) -> None:
        """Notify that a job finished SUCCESSFULLY (COMPLETED) so the responsible
        agent wakes up and reviews the results.

        Mirrors _notify (one-shot, notified flag) but for the success path —
        previously a completed job just printed "(completed)" and nobody ever
        read the results unless the submitting agent happened to still be in a
        tracking loop. With this, the drain loop wakes the conversation and the
        orchestrator routes result extraction back to the SUBMITTING specialist
        (analyst only when the task itself needs deep scientific analysis).
        """
        with self._lock:
            rec = self._jobs.get(job_id)
            if not rec:
                return
            if rec.get("notified"):
                return
            rec["notified"] = True
            _gas = rec.get("gas") or ""
            _cif = rec.get("cif") or ""
            _agent = rec.get("agent_name") or ""
            _who = f"该作业由 **{_agent}** 提交" if _agent else "该作业由系统监控发现"
            _gasdesc = f"{_gas}@{_cif}" if (_gas and _cif) else (_gas or _cif or "通用作业")
            _work = rec.get("work_dir") or ""
            # Product locations: output_dir + charged_cifs are the REAL artifacts
            # of this job (e.g. *_pacmof.cif written into output_dir). Without
            # them the agent blind-greps the whole tree and finds STALE results
            # from other runs — the exact bug seen on job 3669 (grep hit a
            # May-29 GCMC output instead of this job's charged CIFs).
            _prod = []
            if rec.get("output_dir"):
                _prod.append(f"产物目录: {rec['output_dir']}")
            _cc = rec.get("charged_cifs") or []
            if _cc:
                _prod.append(f"产物文件({len(_cc)}个): {_cc[0]}{' …' if len(_cc) > 1 else ''}")
            _work_line = f"工作目录: {_work}" if _work else ""
            _prod_line = "\n".join(_prod)
            _loc = "\n".join(x for x in (_work_line, _prod_line) if x)
            _loc = ("\n" + _loc) if _loc else ""
            msg = (
                f"✅ [作业完成提醒] SLURM 作业 {job_id} ({_gasdesc}) 已完成（state=**{state}**）。"
                f"{_who}。\n{_loc}\n"
                f"1) **结果提取（默认，必做）**：委派提交该作业的专业Agent（{_agent or 'adsorption/analyst/harness'}）"
                "读取工作目录/产物目录的结果文件，"
                "验证数值合理性（吸附量/能量是否非零、是否收敛、有无异常或静默失败），提取关键结果并如实汇报。\n"
                "2) **是否需要 analyst（按需，绝不强制）**：只有当任务本身要求深度科学分析"
                "（吸附趋势对比、选择性/工作容量分析、与文献数据对齐、多结构筛选排名等）时，才委派 analyst；"
                "普通结果提取（报数值 + 验证收敛）由提交该作业的专业Agent自己完成即可，不必每次都叫 analyst。\n"
                "3) **无需分析的任务**：如果该作业只是中间步骤/结构准备/纯几何量/无需进一步解读，"
                "专业Agent简要确认结果即可，不要强行分析。\n"
                "4) 如果你已经在本轮汇报过该作业的结果，只需简要确认，不要重复冗长报告。"
            )
            payload = {
                "job_id": job_id,
                "conv_id": rec.get("conv_id", ""),
                "username": rec.get("username", ""),
                "state": state,
                "gas": rec.get("gas", ""),
                "cif": rec.get("cif", ""),
                "message": msg,
                "diagnosis": None,
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            self._queue_notification(rec, payload)

    def _poll_once(self) -> None:
        with self._lock:
            # Poll non-terminal jobs as usual, BUT also re-examine jobs that
            # were loaded as already-terminal from the persisted watch file:
            # if the server restarted while they were failing, nobody ever
            # notified the agent (their transition happened in a previous
            # process). Sweep them once so a restart never orphans a failure.
            now = time.time()
            jobs = [dict(j) for j in self._jobs.values() if not j.get("terminal")
                    or (j.get("terminal") and not j.get("notified")
                        and now >= float(j.get("unconfirmed_next_poll", 0) or 0))]
        if not jobs:
            return
        # ai2-kit absorb: ONE batch squeue query (TTL-cached) for all watched
        # jobs instead of spawning a subprocess per job. Jobs still visible in
        # the scheduler in a non-failed state (PENDING/RUNNING/COMPLETED) get
        # their cheap state straight from the batch. Failed terminal jobs and
        # jobs that vanished from the queue get the full software-level
        # check_job_status (job.done indicator, .err/run.log diagnosis,
        # zero-loading detection) — that's the minority, so the cost stays low.
        try:
            batch = slurm.batch_fetch_states([r["job_id"] for r in jobs])
        except Exception as e:
            print(f"  [JobWatch] batch state query failed: {e}", flush=True)
            batch = {}
        for rec in jobs:
            job_id = rec["job_id"]
            hit = batch.get(job_id)
            if hit is not None and not hit.get("failed"):
                # Cheap path: PENDING/RUNNING/COMPLETED from the batch query.
                st = hit
            else:
                # Full path: job vanished from the queue (finished & gone —
                # gpu2 has no accounting) OR the batch saw a terminal failure
                # that needs software-level diagnosis before notifying.
                try:
                    st = slurm.check_job_status(job_id, work_dir=rec.get("work_dir", ""))
                except Exception as e:
                    print(f"  [JobWatch] poll error {job_id}: {e}", flush=True)
                    continue
            state = st.get("status", "UNKNOWN")
            terminal = st.get("terminal", False)
            failed = st.get("failed", False)
            with self._lock:
                cur = self._jobs.get(job_id)
                if not cur:
                    continue
                cur["last_state"] = cur.get("state", "PENDING")
                cur["state"] = state
                cur["terminal"] = terminal
                cur["failed"] = failed
                cur["updated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                cur['pending_reason'] = st.get('pending_reason', '')
                cur['estimated_start'] = st.get('estimated_start', '')
                if state == 'PENDING':
                    if cur.get('last_state') != 'PENDING' or not cur.get('pending_since_epoch'):
                        from datetime import datetime
                        try: submitted = datetime.strptime(cur.get('submitted_at',''), '%Y-%m-%d %H:%M:%S').timestamp()
                        except ValueError: submitted = time.time()
                        cur['pending_since_epoch'] = submitted if cur.get('last_state')=='PENDING' else time.time()
                    cur['pending_age_seconds'] = max(0,time.time()-cur['pending_since_epoch'])
                    reason=cur.get('pending_reason') or 'Unknown'
                    changed=cur.get('pending_warning_reason') != reason or cur.get('pending_warning_policy_version')!=1
                    if cur['pending_age_seconds']>=300 and (changed or time.time()-cur.get('pending_warning_at',0)>=900) and not cur.get('pending_notification'):
                        cur.update(pending_warning_at=time.time(),pending_warning_reason=reason,pending_warning_policy_version=1,
                            resource_warning={'kind':'pending','reason':reason,'age_seconds':cur['pending_age_seconds'],'not_failed':True})
                        self._queue_notification(cur,{'job_id':job_id,'state':'PENDING','kind':'job_pending_warning',
                            'username':cur.get('username'),'conv_id':cur.get('conv_id'),
                            'pending_reason':reason,'pending_age_seconds':cur['pending_age_seconds'],
                            'message':f'作业{job_id}仍排队，原因{reason}；委派资源monitor复核，不当成完成/失败，不自动重提。'})
                else:
                    cur.pop('pending_since_epoch',None)
                    cur.pop('pending_age_seconds',None)
                    cur.pop('resource_warning',None)
                if not terminal: self._save()
                if terminal and failed and not cur.get("notified"):
                    cur["diagnosis"] = st.get("diagnosis") or {}
                    self._save()
            if not terminal:
                continue
            # Sync terminal state into TaskLine: done=True ONLY for a confirmed
            # success state (COMPLETED / COMPLETE / DONE). ⚠️ UNKNOWN / missing
            # (job vanished without accounting) must NOT be marked done=True —
            # otherwise the task line shows "all steps ✅" and a taking-over
            # agent thinks there's nothing left to do (the CO2/N2 misdirection).
            # UNKNOWN stays ⏳未完成 with a note so the next agent re-checks it.
            try:
                from .task_line import get_store as _get_taskline
                if failed:
                    _get_taskline().mark_job_done(job_id, done=False, state=state, failed=True)
                elif state.upper() in ("COMPLETED", "COMPLETE", "DONE", "SUCCESS"):
                    _cc = rec.get("charged_cifs") or []
                    _get_taskline().mark_job_done(
                        job_id, done=True, state=state,
                        output_files=_cc if _cc else None)
                else:
                    # UNKNOWN / UNCONFIRMED terminal — keep the step open, flag it.
                    with self._lock:
                        cur = self._jobs.get(job_id)
                        if cur is not None:
                            # Scheduler accounting may be unavailable. Avoid a
                            # permanent 5-second hot loop while retaining periodic
                            # rechecks in case a job.done/result artifact appears.
                            cur["unconfirmed_next_poll"] = time.time() + 300
                    _get_taskline().mark_job_done(
                        job_id, done=False, state=state, failed=False,
                        output_files=rec.get("charged_cifs") or None)
            except Exception as _te:
                print(f"  [TaskLine] sync error {job_id}: {_te}", flush=True)
            if failed:
                diag = st.get("diagnosis") or {}
                summary = "stderr 摘要: " + (st.get("error") or "")[:400]
                if diag.get("cause"):
                    summary += f"\n推测原因: {diag['cause']}\n修复建议: {', '.join((diag.get('fixes') or [])[:3])}"
                self._notify(job_id, state, summary, diag)
                print(f"  [JobWatch] job {job_id} → {state} (failure notification queued)", flush=True)
            else:
                # Successful-completion wake-up — but ONLY for CONFIRMED
                # terminal states. UNKNOWN / UNCONFIRMED means the job could
                # not be found (malformed id, vanished from accounting):
                # notifying "✅ completed" misleads the agent into believing
                # results exist. Keep the step open instead.
                if state.upper() in ("UNKNOWN", "UNCONFIRMED", ""):
                    print(f"  [JobWatch] job {job_id} → {state} (terminal but unconfirmed; no completion notice, step kept open)", flush=True)
                else:
                    self._notify_completion(job_id, state)
                    print(f"  [JobWatch] job {job_id} → {state} (completion notification queued)", flush=True)
            with self._lock:
                self._save()

    def start(self) -> "JobWatch":
        if self._thread and self._thread.is_alive():
            return self
        # Exactly one API worker owns the polling loop. Other workers may still
        # register/list jobs through the shared atomic state file, but duplicate
        # pollers must not emit duplicate completion/failure notifications.
        leader = None
        try:
            import fcntl
            Path(self._watch_file).parent.mkdir(parents=True, exist_ok=True)
            leader = open(self._watch_file + ".leader.lock", "a+")
            fcntl.flock(leader.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            self._leader_fh = leader
        except Exception:
            try:
                if leader is not None:
                    leader.close()
            except Exception:
                pass
            print("  [JobWatch] another process owns the poller; registration-only mode", flush=True)
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="job-watch", daemon=True)
        self._thread.start()
        print(f"  [JobWatch] started (poll every {POLL_INTERVAL}s, {len(self._jobs)} jobs)", flush=True)
        return self

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._poll_once()
            except Exception as e:
                print(f"  [JobWatch] poll loop error: {e}", flush=True)
            self._stop.wait(POLL_INTERVAL)

    def stop(self) -> None:
        self._stop.set()
        if self._leader_fh is not None:
            try:
                import fcntl
                fcntl.flock(self._leader_fh.fileno(), fcntl.LOCK_UN)
                self._leader_fh.close()
            except Exception:
                pass
            self._leader_fh = None

    # ── persistence ─────────────────────────────────────────────────
    def _save(self) -> None:
        lock_fh = None
        try:
            import fcntl
            from pathlib import Path
            Path(self._watch_file).parent.mkdir(parents=True, exist_ok=True)
            lock_fh = open(self._watch_file + ".lock", "a+")
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            disk = {}
            if os.path.exists(self._watch_file):
                try:
                    with open(self._watch_file, encoding="utf-8") as f:
                        loaded = json.load(f)
                        if isinstance(loaded, dict):
                            disk = loaded
                except Exception:
                    disk = {}
            baseline = getattr(self, '_baseline', {})
            for key, record in self._jobs.items():
                previous = baseline.get(key)
                if previous is None:
                    disk[key] = record
                elif previous != record:
                    current = disk.setdefault(key, {})
                    # Apply only locally changed fields, not stale snapshots of
                    # other workers' notification acknowledgements/job states.
                    for field, value in record.items():
                        if previous.get(field) != value or field not in previous:
                            current[field] = value
                    for field in set(previous) - set(record):
                        current.pop(field, None)
            for key in self._deleted:
                disk.pop(key, None)
            fd, tmp = tempfile.mkstemp(
                prefix=os.path.basename(self._watch_file) + ".",
                dir=os.path.dirname(self._watch_file) or ".",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(disk, f, ensure_ascii=False, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self._watch_file)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
            self._jobs = disk
            self._baseline = json.loads(json.dumps(disk))
            self._deleted.clear()
        except Exception as error:
            raise RuntimeError(f'job watch commit failed; scheduler jobs must not be resubmitted: {error}') from error
        finally:
            if lock_fh is not None:
                try:
                    import fcntl
                    fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    pass
                lock_fh.close()

    def _load(self) -> None:
        try:
            source = self._watch_file
            if self._watch_file == WATCH_FILE and not os.path.exists(source):
                source = LEGACY_WATCH_FILE
            if os.path.exists(source):
                with open(source) as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    self._jobs = {k: v for k, v in data.items() if isinstance(v, dict)}
                    if (self._watch_file == WATCH_FILE and source == self._watch_file and os.path.exists(LEGACY_WATCH_FILE)
                            and os.path.getmtime(LEGACY_WATCH_FILE) > os.path.getmtime(self._watch_file)):
                        with open(LEGACY_WATCH_FILE) as stream:
                            legacy = json.load(stream)
                        for key, value in legacy.items():
                            if key not in self._jobs or str(value.get('updated_at', '')) >= str(self._jobs[key].get('updated_at', '')):
                                self._jobs[key] = value
                    queued = {str(n.get("job_id")) for n in self._pending_notifications}
                    for job_id, rec in self._jobs.items():
                        payload = rec.get("pending_notification")
                        if (isinstance(payload, dict) and job_id not in queued
                                and job_id not in self._inflight_notifications):
                            self._pending_notifications.append(payload)
                            queued.add(job_id)
                else:
                    raise ValueError('job watch state must contain a JSON object')
            self._baseline = json.loads(json.dumps(self._jobs))
        except Exception as error:
            raise RuntimeError(f'job watch load failed; preserving source file {source}: {error}') from error
        # Self-heal contradictory records persisted by older code:
        #   1. job_id keys carrying an sbatch prefix ("Submitted batch job
        #      3758" instead of "3758") — normalize the key so squeue lookups
        #      and TaskLine matching work again (legacy zombie records).
        #   2. a NON-terminal state name (PENDING/RUNNING) with a mis-set
        #      terminal flag would be skipped by _poll_once forever — reset it.
        # UNKNOWN is not success, but a log-confirmed failure must remain failed.
        # Unconfirmed records are rechecked with backoff; never re-arm their
        # notifications on every list()/load() call.
        with self._lock:
            _healed = False
            for _key, rec in list(self._jobs.items()):
                _norm = slurm.normalize_job_id(_key)
                if _norm != _key:
                    self._jobs.pop(_key, None)
                    if _norm not in self._jobs:
                        self._jobs[_norm] = rec
                        rec["job_id"] = _norm
                    _healed = True
                _rec = self._jobs.get(_norm) or rec
                # 4. legacy records (pre-diagnosed) lack the field → default False
                if 'diagnosed' not in _rec:
                    _rec["diagnosed"] = False
                    _rec.setdefault("diagnosed_at", "")
                    _healed = True
                _st = str(_rec.get("state", "")).upper()
                if _st in ('PENDING', 'RUNNING') and _rec.get('terminal') and not _rec.get('failed'):
                    _rec["terminal"] = False
                    _rec["failed"] = False
                    _healed = True
            if _healed:
                self._save()


# ── global singleton ────────────────────────────────────────────────
_watch: Optional[JobWatch] = None
_watch_lock = threading.Lock()


def get_watch() -> JobWatch:
    global _watch
    with _watch_lock:
        if _watch is None:
            _watch = JobWatch(watch_file=os.environ.get('BIMEM_JOB_WATCH_FILE',WATCH_FILE))
        return _watch
