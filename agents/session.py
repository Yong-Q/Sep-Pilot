"""Interactive session — human-in-the-loop multi-agent orchestration."""
from __future__ import annotations

import json
import re
import time
import uuid
import os
import copy
import hashlib
from typing import Any, Dict, List, Optional

import anthropic

from .agent import Agent, AgentResult
from .goal_contract import GoalContract
from .registry import ToolRegistry, get_registry
from .config import AgentConfig, get_config
from .recovery import RecoveryGate, failure_reason, input_fingerprint, is_submission, result_object


class LLMUnavailableError(Exception):
    """Raised when the LLM endpoint fails (timeout / connection error) so the
    run can retry or degrade gracefully instead of dying silently."""


class LLMRateLimitError(LLMUnavailableError):
    """Provider-wide transient limit; retry without polluting model context."""


class ExecutionBudgetExceeded(Exception):
    """Controlled local budget stop, never a remote LLM endpoint failure."""


def is_job_status_polling(tool_name, params):
    """File/input discovery is not scheduler-status polling."""
    if tool_name in {'check_job', 'check_job_status', 'list_my_jobs'}: return True
    if tool_name != 'run_bash': return False
    import shlex
    try:
        lexer = shlex.shlex(str(params.get('command') or params.get('cmd') or ''), posix=True, punctuation_chars=';&|')
        lexer.whitespace_split = True
        words = list(lexer)
    except ValueError:
        return False
    command_start = True
    for index, word in enumerate(words):
        if word in {';', '&&', '||', '|'}:
            command_start = True
        elif command_start:
            if word in {'squeue', 'sacct', 'qstat'} or word == 'scontrol' and words[index+1:index+2] == ['show']:
                return True
            command_start = False
    return False


class SessionMemory:
    """Tracks completed steps, tool calls, and agent handoffs for context preservation.

    Improved compact module:
    - Smart compression that preserves important information
    - Handles interruptions and recovery
    - Maintains critical context across long sessions
    """

    def __init__(self):
        self.completed_steps: List[Dict[str, Any]] = []
        self.tool_call_log: List[Dict[str, Any]] = []
        self.agent_history: List[str] = []
        self.errors: List[str] = []
        self.context_summary: str = ""
        self.critical_results: List[Dict[str, Any]] = []  # Store important results
        self.user_preferences: Dict[str, Any] = {}  # Store user choices
        self.interruption_point: Optional[str] = None  # Track where interruption occurred
        # ── Per-agent memory partition ──────────────────────────────────
        # The user requires: "记忆放每个session中按不同智能体再细分" — memories
        # must be partitioned per session AND per agent, never mixed. Each agent
        # keeps its own tool log, steps, errors, critical results and a handoff
        # "exit report" it writes before leaving so the next entry (same-session
        # re-entry or re-invocation by another agent) can pick up where it left.
        self.agent_memories: Dict[str, Dict[str, Any]] = {}

    def _agent_mem(self, agent_name: str) -> Dict[str, Any]:
        """Get (create on first use) an agent's own memory partition."""
        am = self.agent_memories.get(agent_name)
        if am is None:
            am = {
                "tool_call_log": [],
                "completed_steps": [],
                "errors": [],
                "critical_results": [],
                "exit_reports": [],   # 退场报告 list (append-only)
                "agent_history": [],
            }
            self.agent_memories[agent_name] = am
        return am

    def record_tool_call(self, agent_name: str, tool_name: str, params: Dict, result: str,
                         failed: bool = False):
        entry = {
            "call_id": uuid.uuid4().hex,
            "time": time.time(),
            "agent": agent_name, "tool": tool_name,
            "params": copy.deepcopy(params or {}), "result_preview": result[:500],
            "result": result[:50000],
            "failed": bool(failed),
        }
        self.tool_call_log.append(entry)
        # Per-agent partition — never mixed across agents
        self._agent_mem(agent_name)["tool_call_log"].append(entry)
        # Auto-detect and store critical results
        self._extract_critical_results(tool_name, result, agent_name)

    def _extract_critical_results(self, tool_name: str, result: str, agent_name: str = ""):
        """Extract and store critical results for compact summary."""
        critical_keywords = ["选择性", "吸附量", "扩散系数", "结合能", "电荷", "预测", "R2", "MAE"]
        if any(kw in result for kw in critical_keywords):
            cr = {
                "tool": tool_name,
                "result": result[:300],
                "timestamp": len(self.tool_call_log),
                "agent": agent_name,
            }
            self.critical_results.append(cr)
            if agent_name:
                self._agent_mem(agent_name)["critical_results"].append(cr)

    def record_step(self, step_name: str, status: str, details: str = "", agent_name: str = ""):
        entry = {"step": step_name, "status": status, "details": details[:300], "agent": agent_name}
        self.completed_steps.append(entry)
        if agent_name:
            self._agent_mem(agent_name)["completed_steps"].append(entry)

    def record_error(self, error: str, agent_name: str = ""):
        self.errors.append(error)
        if agent_name:
            self._agent_mem(agent_name)["errors"].append(error)

    def record_handoff(self, from_agent: str, to_agent: str):
        hop = f"{from_agent} → {to_agent}"
        self.agent_history.append(hop)
        # Record the hop in BOTH agents' partitions (who left / who received)
        if from_agent:
            self._agent_mem(from_agent)["agent_history"].append(hop)
        if to_agent:
            self._agent_mem(to_agent)["agent_history"].append(hop)

    def record_user_preference(self, key: str, value: Any):
        """Record user choices for future reference."""
        self.user_preferences[key] = value

    def record_interruption(self, point: str):
        """Record where interruption occurred for recovery."""
        self.interruption_point = point

    # ── Agent exit protocol ────────────────────────────────────────────
    # 计算智能体只有在委派的任务都正常结束并取得结果后才退场；退场前写一份
    # 简要报告，方便下次进场读取（同 session 多次进场、或被其他智能体再次调用）。
    def record_exit_report(self, agent_name: str, report_text: str, meta: Optional[Dict[str, Any]] = None):
        """Write the agent's handoff/exit report into its own partition.

        The report is the ONLY durable artifact the next entry reads to resume:
        it must state what was delegated, what was done, which jobs were
        submitted (id + state), what results were obtained, and what remains.
        """
        if not agent_name:
            return
        import time as _t
        am = self._agent_mem(agent_name)
        am["exit_reports"].append({
            "time": _t.strftime("%Y-%m-%d %H:%M:%S"),
            "report": str(report_text)[:3000],
            "meta": meta or {},
        })
        # Keep only the last 3 exit reports per agent (avoid unbounded growth)
        if len(am["exit_reports"]) > 3:
            am["exit_reports"] = am["exit_reports"][-3:]

    def get_exit_report(self, agent_name: str) -> str:
        """Return the agent's most recent exit report (for re-entry injection)."""
        am = self.agent_memories.get(agent_name)
        if not am or not am.get("exit_reports"):
            return ""
        return am["exit_reports"][-1]["report"]

    def compact_summary(self, agent_name: str = "") -> str:
        """Generate intelligent compact summary that preserves important information.

        Per-agent partition: when agent_name is given, ONLY that agent's own
        partition is summarized (never mixed with other agents' memories).
        When agent_name is "" → global summary across the whole session.
        """
        def _fmt(tool_log, critical, errors, steps):
            out = []
            if critical:
                out.append("📊 Critical Results:")
                for r in critical[-5:]:
                    out.append(f"  • {r['tool']}: {r['result'][:100]}")
            recent_tools = [t['tool'] for t in tool_log[-5:]]
            if recent_tools:
                out.append(f"🔧 Recent Tools: {', '.join(dict.fromkeys(recent_tools))}")
            if errors:
                out.append(f"⚠️ Errors ({len(errors)}): {'; '.join(errors[-3:])}")
            completed = sum(1 for s in steps if s['status'] == 'done')
            out.append(f"✅ Progress: {completed}/{len(steps)} steps completed")
            return out

        if agent_name:
            am = self.agent_memories.get(agent_name, {})
            lines = [f"[记忆分区 · {agent_name}]"]
            lines += _fmt(
                am.get("tool_call_log", []),
                am.get("critical_results", []),
                am.get("errors", []),
                am.get("completed_steps", []),
            )
            # Exit report of THIS agent (if any) — the key artifact for re-entry
            _exit = self.get_exit_report(agent_name)
            if _exit:
                lines.append("📝 上次退场报告（进场必读）:")
                lines.append(f"  {_exit[:600]}")
            _hops = am.get("agent_history", [])[-5:]
            if _hops:
                lines.append("🔄 参与链路: " + " ; ".join(_hops))
        else:
            lines = ["[SESSION MEMORY - Intelligent Compact Summary]"]
            lines += _fmt(
                self.tool_call_log,
                self.critical_results,
                self.errors,
                self.completed_steps,
            )
            if self.user_preferences:
                lines.append(f"👤 User Preferences: {self.user_preferences}")
            if self.agent_history:
                unique_agents = set()
                for step in self.agent_history:
                    parts = step.split(' → ')
                    unique_agents.update(parts)
                lines.append(f"🤖 Agents Used: {', '.join(unique_agents)}")

        self.context_summary = "\n".join(lines)
        return self.context_summary

    def get_stats(self) -> Dict[str, Any]:
        return {
            "steps_completed": len(self.completed_steps),
            "tool_calls": len(self.tool_call_log),
            "errors": len(self.errors),
            "agent_handoffs": len(self.agent_history),
            "critical_results": len(self.critical_results),
            "user_preferences": len(self.user_preferences),
            "agents": {a: len(v.get("tool_call_log", [])) for a, v in self.agent_memories.items()},
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe durable representation used by ConversationState.

        Memory is intentionally persisted separately from the chat transcript:
        the transcript is a UI/audit artifact, while this object is execution
        state that agents require after a backend restart.
        """
        return {
            "completed_steps": self.completed_steps,
            "tool_call_log": self.tool_call_log,
            "agent_history": self.agent_history,
            "errors": self.errors,
            "context_summary": self.context_summary,
            "critical_results": self.critical_results,
            "user_preferences": self.user_preferences,
            "interruption_point": self.interruption_point,
            "agent_memories": self.agent_memories,
        }

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "SessionMemory":
        obj = cls()
        if not isinstance(data, dict):
            return obj
        for name in (
            "completed_steps", "tool_call_log", "agent_history", "errors",
            "critical_results",
        ):
            value = data.get(name)
            if isinstance(value, list):
                setattr(obj, name, value)
        if isinstance(data.get("user_preferences"), dict):
            obj.user_preferences = data["user_preferences"]
        if isinstance(data.get("agent_memories"), dict):
            obj.agent_memories = data["agent_memories"]
        obj.context_summary = str(data.get("context_summary", "") or "")
        obj.interruption_point = data.get("interruption_point")
        return obj


class Session:
    """Interactive session with human-in-the-loop.

    Flow:
        session.start("task description") → agent runs tools, returns text
        session.reply("user answer")       → agent continues, returns text
        session.task_complete == True      → done

    Features:
        - Interactive parameter asking (TST/MD, cdft/gcmc selection)
        - Smart compact module
        - Interrupt handling and recovery
        - Active tool creation capability
    """

    def __init__(
        self,
        config: AgentConfig | None = None,
        registry: ToolRegistry | None = None,
    ):
        self.config = config or get_config()
        self.registry = registry or get_registry()
        # Bound each LLM call so a hung endpoint (e.g. a degraded network path
        # with TCP retransmits) can never block the whole multi-round run forever.
        # 300s per call + the SDK's built-in retries is a sane ceiling.
        self.client = anthropic.Anthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url if self.config.base_url else None,
            timeout=300,
        )
        self.messages: List[Dict[str, Any]] = []
        self.current_agent: Optional[Agent] = None
        self.context: Dict[str, Any] = {}
        self.task_complete: bool = False
        self.last_text: str = ""
        self.memory = SessionMemory()
        # Durable objective contract.  Unlike chat history, this survives
        # context trimming and is injected into every agent invocation.
        self.goal_contract = GoalContract()
        self._prev_agent_name: str = ""
        self._pending_handoff = None
        self._waiting_for_user_input: bool = False  # Track if waiting for user
        self._user_input_queue: List[str] = []  # Queue for user responses
        self._pending_user_interaction = None
        # Progress callback: fn(agent_name, reasoning, status, log_message)
        self._on_progress = None
        self._on_checkpoint = None
        self.recovery_gate = RecoveryGate()
        self._active_tool_futures = []
        self._on_lifecycle_event = None
        self._lifecycle_store = None
        self._pending_workflow_patch = None
        self._evidence_root = None
        self._on_workflow_start = None
        self._on_resource_review = None
        self._on_workflow_message = None
        self._on_workflow_resolve = None
        self._on_workflow_revalidate = None
        self._on_workflow_finish = None
        self._runtime_snapshot = None
        self._on_tool_guard = None
        self._control_source = None
        # Partial result callback: fn(agent_name, result_text) — called when a sub-agent completes
        self._on_partial_result = None
        # User-interrupt callback: fn() → bool. When True, the running loop
        # FREEZES cleanly (Claude-Code-style Esc): it stops making further
        # LLM/tool calls, returns an honest partial report, and leaves any
        # previously-submitted SLURM jobs running. Wired from the API layer to
        # the ConversationState.interrupt_requested flag.
        self._on_interrupt_requested = None
        # Optional redirect callback: fn() → str — the Claude-Code-style message
        # the user attached to the interrupt ("中断加话和需求"). Consumed by
        # reply() and injected as the new delegation's redirect context.
        self._on_interrupt_message = None
        # Optional clear callback: fn() — clears the pending redirect after it
        # has been consumed (so it only applies once).
        self._on_interrupt_clear = None
        # Fault-handling: track retries per failed tool signature
        self._failure_retries: Dict[str, int] = {}
        self.failure_history: List[Dict[str, Any]] = []  # structured record of all detected failures

        # ReAct discipline: consecutive rounds where the model acted with no
        # reasoning text (pure tool_use, no [思考]/[观察]) before a nudge fires.
        self._silent_act_rounds = 0

        # Method-choice interaction: set True once the user has answered the
        # GCMC-vs-cDFT question so we never ask it twice in one session.
        self._method_choice_made = False

        # Hard-gate param question (start()/reply()): set by _build_param_question
        # when thermodynamic params / method / gas / material are missing; reply()
        # re-verifies the user's next message actually supplies them (or the user
        # authorizes self-decision) before the agent is allowed to run.
        self._pending_param_question = ""

        # TaskLine integration: auto-parse plans into pipeline steps
        self._current_line_id: str = ""
        self._plan_steps_created: bool = False

        # Error branch tracking: when a step fails, create a dedicated error branch
        self._error_branches: Dict[str, Dict[str, Any]] = {}
        self._error_branch_counter: int = 0
        # A multi-step plan pauses here until the user explicitly approves it,
        # unless the user's goal already says "直接执行".
        self._awaiting_plan_approval: bool = False
        self._pending_plan_text: str = ""

    def _interrupt_requested(self) -> bool:
        """True if the user asked to interrupt (freeze) the current thinking turn."""
        cb = getattr(self, "_on_interrupt_requested", None)
        if cb is None:
            return False
        try:
            return bool(cb())
        except Exception:
            return False

    def _interrupt_message(self) -> str:
        """Optional Claude-Code-style redirect attached to the interrupt.

        The user pressed ⛔ and typed a new requirement ("加话和需求"): the next
        delegation should carry this message as the redirect context.
        """
        cb = getattr(self, "_on_interrupt_message", None)
        if cb is None:
            return ""
        try:
            return str(cb() or "")
        except Exception:
            return ""

    def _build_interrupt_report(self, agent_name: str) -> str:
        """Honest partial report at interrupt — NO new LLM call, NO tool calls.

        Claude-Code-style Esc: the agent's turn stops RIGHT HERE. Previously
        submitted SLURM jobs keep running (interrupt never cancels them). The
        report is assembled from real state: pending jobs + their status +
        completed steps, so nothing is fabricated.
        """
        lines = [
            "⛔ [系统·用户中断] 本轮思考已冻结。",
            f"  • 当前 Agent: {agent_name}",
            "  • 已停止进一步的思考与工具调用。",
            "  • **之前已提交的 SLURM 作业不受影响，继续在计算节点运行。**",
        ]
        jobs = self._pending_jobs_for_conv()
        if jobs:
            lines.append(f"\n📋 本会话已提交的作业 ({len(jobs)}):")
            from . import slurm as _slurm
            for j in jobs:
                jid = j.get("job_id")
                tool = j.get("tool", "?")
                st = _slurm.check_job_status(jid, work_dir=j.get("work_dir", ""))
                state = st.get("status", "UNKNOWN")
                lines.append(f"  • {jid} [{tool}] → {state}")
        else:
            lines.append("\n📋 尚未提交任何 SLURM 作业。")
        if self.memory.completed_steps:
            lines.append("\n✅ 已完成步骤:")
            for s in self.memory.completed_steps[-6:]:
                lines.append(f"  • {s.get('step')} ({s.get('status')})")
        if self.memory.tool_call_log:
            lines.append("\n🔧 最近工具调用:")
            for t in self.memory.tool_call_log[-4:]:
                lines.append(f"  • {t['agent']}.{t['tool']}: {t['result_preview'][:80]}")
        lines.append("\n你可以在 agent 停止后发送新消息作为重定向/继续委派。")
        return "\n".join(lines)

    def _build_agent_exit_report(self, agent_name: str, final_text: str) -> str:
        """Brief handoff/exit report a compute agent writes before leaving.

        Readable on the agent's next entry — either a same-session re-entry, or
        a re-invocation by another agent (the report is injected into that
        agent's system prompt via compact_summary(agent_name=...)). It captures
        what was delegated, what was done, which jobs are running/pending, what
        was obtained, and what remains — so the next entry resumes cleanly
        instead of restarting from scratch.
        """
        jobs = self._pending_jobs_for_conv()
        job_lines = []
        if jobs:
            from . import slurm as _slurm
            for j in jobs:
                jid = j.get("job_id")
                tool = j.get("tool", "?")
                try:
                    st = _slurm.check_job_status(jid, work_dir=j.get("work_dir", ""))
                    state = st.get("status", "UNKNOWN")
                except Exception:
                    state = "UNKNOWN"
                job_lines.append(f"{jid}[{tool}]→{state}")
        _tool_n = len(self.memory.agent_memories.get(agent_name, {}).get("tool_call_log", []))
        lines = [
            f"[{agent_name} 退场报告]",
            f"· 委派任务完成。本轮工具调用: {_tool_n} 次。",
            f"· 已提交/跟踪作业: {'; '.join(job_lines) if job_lines else '无'}",
            f"· 交付内容摘要: {final_text[:800] if final_text else '(无文本交付)'}",
        ]
        return "\n".join(lines)

    def _report_partial(self, agent_name: str, result_text: str):
        """Emit a sub-agent's completed result in real-time (before final answer)."""
        if self._on_partial_result and result_text:
            try:
                self._on_partial_result(agent_name, result_text[:3000])
            except Exception:
                pass

    def _progress(self, agent_name: str = "", reasoning: str = "", status: str = "", log: str = ""):
        """Report progress to the API layer via callback."""
        if self._on_progress:
            try:
                self._on_progress(agent_name, reasoning, status, log)
            except Exception:
                pass  # Don't let callback errors break execution

    def _checkpoint(self, reason: str, partial_results=None):
        """Persist after each tool and before submission; never silently ignore I/O failure."""
        if not self._on_checkpoint:
            return
        state = self.export_state()
        if partial_results is not None:
            messages = self._json_safe(self.messages)
            if messages and messages[-1].get('role') == 'assistant':
                results = self._json_safe(partial_results)
                have = {r.get('tool_use_id') for r in results}
                for block in messages[-1].get('content', []):
                    if block.get('type') == 'tool_use' and block.get('id') not in have:
                        results.append({'type': 'tool_result', 'tool_use_id': block['id'],
                                        'content': '[CHECKPOINT] call not executed or outcome unknown; inspect recovery ledger before resuming'})
                messages.append({'role': 'user', 'content': results})
            state['messages'] = self._pair_trimmed_blocks(self._trim_messages(messages, max_tokens=100000))
        state['checkpoint_reason'] = reason
        self._on_checkpoint(state, reason)
        self._persist_chain(reason, state)
        if getattr(self, '_on_chain_committed', None):
            self._on_chain_committed()
        self._last_checkpoint_at = state['checkpoint_at']

    def _persist_chain(self, reason, state=None):
        if not (self._evidence_root and self._current_line_id and self._runtime_snapshot):
            return
        from .orchestration_chain import sync_chain
        from .task_line import get_store
        # Only the main conversation owns goals, branches and pending decisions.
        # Supervisor checkpoints must not replace these with observer-local state.
        primary = getattr(self, '_chain_state_is_primary', True)
        job_supplier = getattr(self, '_chain_jobs', None)
        jobs = job_supplier() if job_supplier else None
        return sync_chain(self._evidence_root.parent, lambda: get_store().get_line(self._current_line_id) or {},
            lambda: self._runtime_snapshot() or {}, reason,
            state=(state or self.export_state()) if primary else None, jobs=jobs,
            turn_id=getattr(self, '_api_turn_id', None) or self.context.get('active_turn_id'))

    def _emit_lifecycle_event(self, kind, details):
        if not self._on_lifecycle_event:
            return
        workflow = {}
        if self._current_line_id:
            from .task_line import get_store
            workflow = get_store().get_line(self._current_line_id) or {}
        payload = {
            'workflow_id': self._current_line_id,
            'goal_contract': self.goal_contract.to_dict(),
            'current_agent': self.current_agent.name if self.current_agent else 'lead-orchestrator',
            'workflow': workflow, 'recovery': self.recovery_gate.snapshot(), **details,
        }
        return self._on_lifecycle_event(kind, payload)

    def _lifecycle_state(self):
        # Event payloads are historical snapshots, not the current DAG. Returning
        # the entire mailbox made old, unapproved proposals look executable.
        state = self._lifecycle_store.snapshot() if self._lifecycle_store else {'events': {}, 'agents': {}}
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        from .parallel_workflow import runtime_summary
        from .task_line import get_store
        line = self._workflow_view()
        return {'agents': state.get('agents', {}), 'workflow_runtime': runtime_summary(runtime),
                'current_workflow': {'plan_version': (line or {}).get('plan_version'),
                    'nodes': [{k: n.get(k) for k in ('step_id', 'agent', 'tool', 'arguments', 'depends_on',
                        'expected_outputs', 'status', 'done', 'job_ids')} for n in (line or {}).get('steps', [])
                        if 'expected_outputs' in n]},
                'pending_patch': ({k: self._pending_workflow_patch.get(k) for k in
                    ('base_version', 'new_version', 'changes', 'reason', 'affected_nodes')} if self._pending_workflow_patch else None),
                'recent_events': [{k: e.get(k) for k in ('event_id', 'kind', 'created_at', 'main_chat', 'supervisor')}
                    for e in list(state.get('events', {}).values())[-8:]],
                'note': 'Current workflow/runtime are authoritative. Recent events are history, not applied plans.'}

    def _workflow_view(self):
        from .task_line import get_store
        from .workflow_view import authoritative_line
        line = get_store().get_line(self._current_line_id) if self._current_line_id else {}
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        return authoritative_line(line, runtime)

    def _restore_workflow_approval(self, runtime, owner):
        from .workflow_view import restore_approved_nodes
        recovered = restore_approved_nodes(self.goal_contract, runtime, owner)
        if recovered:
            self.goal_contract.approved_nodes = recovered
            self.context['workflow_approval_restored'] = {
                'plan_version': runtime['plan_version'], 'node_ids': [n['step_id'] for n in recovered],
                'basis': 'same scoped/versioned scientific contract; no execution authority granted'}
        return bool(recovered)

    def _workflow_completion_state(self):
        from .task_line import get_store
        from .workflow_view import workflow_completion
        line = get_store().get_line(self._current_line_id) or {}
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        self._restore_workflow_approval(runtime, {k: line.get(k) for k in ('username', 'conv_id')})
        if self.goal_contract.approved_plan_version and not self.goal_contract.approved_nodes:
            return {'ok': False, 'reason': 'approved_contract_missing', 'blockers': []}
        return workflow_completion(line, runtime, self.goal_contract.approved_nodes,
                                   self.goal_contract.approved_plan_version)

    def _task_line_query(self, **params):
        from .registry import _exec_task_line_query
        result = _exec_task_line_query(params)
        if result.get('ok') and 'steps' in result:
            from .workflow_view import authoritative_line
            line = {**result, **{k: self._workflow_view().get(k) for k in ('username', 'conv_id')}}
            view = authoritative_line(line, self._runtime_snapshot() if self._runtime_snapshot else {})
            result.update({k: v for k, v in view.items() if k not in {'username', 'conv_id'}})
            runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
            if runtime:
                actual_ids = set(runtime.get('nodes', {}))
                result['diagnostic_steps'] = [n for n in result['steps'] if n.get('branch_parent')]
                result['historical_steps'] = [n for n in result['steps']
                    if n['step_id'] not in actual_ids and not n.get('branch_parent')]
                result['steps'] = [n for n in result['steps'] if n['step_id'] in actual_ids]
                result['execution_completion'] = self._workflow_completion_state()
                result['note'] = 'steps are the current executable DAG; diagnostic/historical records are not delegations and cannot be removed with a DAG patch.'
        return result

    def _task_line_update(self, **params):
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        if params.get('step_id') in runtime.get('nodes', {}):
            return {'ok': False, 'blocked': True, 'executed': False,
                    'reason': 'Executor-owned node status cannot be changed by a conversational annotation.',
                    'actual_status': runtime['nodes'][params['step_id']]['status'],
                    'next_action': 'Use message_workflow_node for comments, finish_workflow_node for verified results, or propose_workflow_patch for repairs.'}
        from .registry import _exec_task_line_update
        return _exec_task_line_update(params)

    def scientific_review_event(self, question, answer):
        from .scientific_review import SCIENCE_TOOLS, digest, knowledge_fingerprint, is_source_result, canonical_source_inputs
        start = int(self.context.get('scientific_tool_start', 0))
        if self.context.get('scientific_correction_scope'):
            start = 0  # The correction can rely on its initial, unchanged evidence.
        calls = [c for c in self.memory.tool_call_log[start:] if c.get('tool') in SCIENCE_TOOLS
                 and is_source_result(c['tool'], result_object(self._load_evidence_call(c).get('result')))]
        if not calls:
            return None
        active = self.context.get('active_scientific_review_scope')
        states = self.context.setdefault('scientific_reviews', {})
        state = states.get(active, {})
        if state.get('goal_version') != self.goal_contract.version or state.get('status') != 'correcting_main':
            active = uuid.uuid4().hex
            state = {'scope_id': active, 'goal_version': self.goal_contract.version, 'question': question,
                     'initial_answer': answer, 'corrections': 0, 'verification_retries': 0, 'status': 'pending'}
            states[active] = state
        self.context['active_scientific_review_scope'] = active
        inputs = []
        for call in calls:
            loaded = self._load_evidence_call(call)
            obj = result_object(loaded.get('result'))
            path = obj.get('cif_path') or obj.get('asset_path')
            if path and obj.get('sha256'):
                inputs.append({'path': path, 'sha256': obj['sha256']})
        tools = (['validate_framework_charges'] if any(c['tool'] == 'validate_framework_charges' for c in calls)
                 else ['inspect_forcefield'] if any(c['tool'] == 'inspect_forcefield' for c in calls)
                 else [calls[-1]['tool']])
        return {'event_id': f'science:{active}:{digest(answer)}', 'kind': 'scientific_review', 'payload': {
            'scope_id': active, 'origin_goal_version': state['goal_version'], 'question': state['question'], 'answer': answer,
            'verification_tools': tools, 'source_inputs': canonical_source_inputs(inputs),
            'knowledge_fingerprint': knowledge_fingerprint(self.config.project_root),
            'main_already_handling': True,
            'instruction': '审核当前调查回答而不是完整可运行模型。自主调用正式科学工具核实实际主张。有具体类型/数值就必须inspect，不能用discover摘要替代。metadata_status=indexed_not_scientifically_assigned是未完成化学分配，不是未解析；实际atom_types和numeric/raw系数就是已查证数值，读取parsed_sections判断解析范围。错误必须引用原回答和相反证据；监督未查证不是主任务失败。公式/单位/原始数值也要核对，不输出私有思维链。'}}

    def scientific_input_receipt(self):
        """Machine-verified immutable input receipt, not a model promise."""
        from pathlib import Path
        from .scientific_review import is_source_result
        start = int(self.context.get('scientific_tool_start', 0))
        rows = []
        seen = set()
        for call in self.memory.tool_call_log[start:]:
            if call.get('tool') != 'validate_framework_charges': continue
            obj = result_object(self._load_evidence_call(call).get('result'))
            if not is_source_result(call['tool'], obj) or obj['cif_path'] in seen: continue
            seen.add(obj['cif_path'])
            path = Path(obj['cif_path'])
            unchanged = path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == obj['sha256']
            rows.append(f"{obj['cif_path']}：" + ('保持原样，输入未修改' if unchanged else '输入已变化，旧校验证据不得沿用')
                        + f"；evidence_ref.call_id={call['call_id']}")
        return ('\n\n[SDK输入完整性回执]\n' + '\n'.join(rows)) if rows else ''

    def audit_scientific_answer(self, observer, event):
        """Same bounded policy for live API and standalone SDK acceptance."""
        from .scientific_review import review_action
        payload = event['payload']
        scope = payload['scope_id']
        states = self.context.setdefault('scientific_reviews', {})
        state = states.setdefault(scope, {'scope_id': scope, 'goal_version': payload['origin_goal_version'],
            'question': payload['question'], 'initial_answer': payload['answer'], 'corrections': 0, 'verification_retries': 0})
        for attempt in range(2):
            receipt = observer.observe_lifecycle_event(event)
            if self.goal_contract.version != state['goal_version']:
                return {**receipt, 'review_action': 'obsolete', 'scope_id': scope}
            action = review_action(state, receipt)
            receipt = {**receipt, 'review_action': action, 'scope_id': scope}
            self._checkpoint('scientific_audit_receipt')
            if action != 'retry_supervisor':
                return receipt
        return receipt

    def _recover_validated_preflight_draft(self, wake_after_restore=False):
        # A validated preflight draft is enough to resume planning. It is an
        # internal checkpoint, not a scientific choice or external identity
        # decision, so recover it rather than handing a framework deadlock to
        # the user. This also repairs sessions created before workflow-draft
        # controls were exempted from the legacy force-handoff guard.
        if (self.goal_contract.execution_authorized and not self.goal_contract.approved_nodes
                and self._current_line_id):
            from .task_line import get_store
            line = get_store().get_line(self._current_line_id) or {}
            persisted = next((step for step in reversed(line.get('steps', []))
                if step.get('tool') == 'record_workflow_draft'
                and step.get('validation', {}).get('goal_contract') == 'passed'
                and (step.get('validation', {}).get('schema') == 'passed'
                     or step.get('validation', {}).get('execution') == 'passed')
                and (step.get('arguments') or {}).get('nodes')
                and (step.get('arguments') or {}).get('completion_criteria')
                and not any(node.get('missing_parameters') for node in
                            (step.get('arguments') or {}).get('nodes', []))), None)
            if persisted:
                arguments = persisted['arguments']
                draft = self._record_workflow_draft(
                    arguments['nodes'], arguments['completion_criteria'])
                get_store().upsert_step(
                    self._current_line_id, persisted['step_id'], status='completed', done=True,
                    validation={**persisted.get('validation', {}),
                                'autonomous_recovery': 'restored validated preflight draft'},
                )
                self._waiting_for_user_input = False
                self._pending_user_interaction = None
                receipt = {'ok': True, 'status': 'autonomous_repair',
                        'recovered_step': persisted['step_id'],
                        'draft_version': draft['version'],
                        'next_action': 'propose_workflow_patch'}
                if wake_after_restore:
                    self.context['preflight_recovery_ready'] = copy.deepcopy(receipt)
                return receipt
        return None

    def _request_user_decision(self, question, reason, related_nodes, recovery_key='', candidate_job_id=''):
        if self.current_agent and self.current_agent.name != 'lead-orchestrator':
            raise ValueError('main chat owns user negotiation')
        if not recovery_key and not candidate_job_id:
            recovered = self._recover_validated_preflight_draft()
            if recovered:
                return recovered
        decision_id = uuid.uuid4().hex
        params = {'decision_id': decision_id, 'question': question, 'reason': reason,
                  'related_nodes': related_nodes, 'recovery_key': recovery_key, 'candidate_job_id': candidate_job_id}
        self._waiting_for_user_input = True
        self._pending_user_interaction = {'tool': 'user_decision', 'params': params, 'prompt': question}
        self._emit_lifecycle_event('user_decision_required', {**params, 'requires_user': True})
        return {'ok': True, 'status': 'needs_user', 'decision_id': decision_id, 'negotiation_owner': 'lead-orchestrator'}

    def _reconcile_watched_job(self, recovery_key, job_id, decision_id, diagnosis_call_id):
        from .job_watch import get_watch
        from .watch_context import get_context
        from datetime import datetime
        if self.current_agent and self.current_agent.name != 'lead-orchestrator':
            raise ValueError('main chat owns dispatch reconciliation')
        decision = self.memory.user_preferences.get('decision:' + decision_id, {})
        answer = str(decision.get('answer', ''))
        if (decision.get('source') != 'authenticated_user' or decision.get('recovery_key') != recovery_key
                or decision.get('candidate_job_id') and str(decision['candidate_job_id']) != str(job_id)
                or any(k in answer for k in ('不是', '不确定', '不知道', '不要', '拒绝'))
                or not (re.search(rf'(?<!\d){re.escape(str(job_id))}(?!\d)', answer)
                        or decision.get('candidate_job_id') == job_id and any(k in answer for k in ('确认', '同意', '是的')))):
            raise ValueError('a genuine user identity decision is required, not an Agent claim of approval')
        entry = self.recovery_gate.snapshot().get(recovery_key)
        job = get_watch().get(job_id)
        context = get_context()
        if not entry or not job or job.get('username') != context.get('username') or job.get('conv_id') != context.get('conv_id'):
            raise ValueError('can only reconcile an existing watched job owned by this user/conversation')
        stamped = job.get('attempt_id') == entry['attempt_id'] and job.get('recovery_key') == recovery_key
        legacy = str(entry.get('legacy_job_id')) == str(job_id)
        if not stamped and not legacy:
            try:
                submitted = datetime.fromisoformat(job['submitted_at']).timestamp()
            except (KeyError, ValueError):
                submitted = 0
            if job.get('tool') != entry['tool'] or submitted < entry['started_at'] - 5:
                raise ValueError('job identity does not match the unresolved dispatch')
        call = self._load_evidence_call(next((c for c in self.memory.tool_call_log if c.get('call_id') == diagnosis_call_id), None))
        obj = result_object(call.get('result', '')) if call else {}
        if (not call or call.get('tool') != 'diagnose_job' or str(call.get('params', {}).get('job_id')) != str(job_id)
                or call.get('time', 0) < entry['started_at']
                or str(obj.get('status', '')).upper() in {'', 'UNKNOWN', 'UNCONFIRMED'} and not obj.get('failed')):
            raise ValueError('fresh matching job diagnosis is required; UNKNOWN without error evidence cannot establish identity/outcome')
        confirmed = {**job, 'state': str(obj.get('status', job.get('state'))).upper(),
                     'terminal': bool(obj.get('terminal')), 'failed': bool(obj.get('failed'))}
        self.recovery_gate.reconcile_watched_job(recovery_key, confirmed, {
            'decision_id': decision_id, 'diagnosis_call_id': diagnosis_call_id, 'diagnostic_time': call['time'],
        })
        self._emit_lifecycle_event('dispatch_reconciled', {'recovery_key': recovery_key, 'job_id': job_id, 'main_already_handling': True})
        return {'ok': True, 'job_id': job_id, 'new_submission': False}

    def _set_control_source(self, user_message):
        from .watch_context import get_context
        from .goal_contract import _is_automated_message
        owner = get_context()
        self._control_source = {'origin': 'system' if _is_automated_message(user_message) else 'user',
            'turn_id': uuid.uuid4().hex, 'owner': {'username': owner.get('username'), 'conv_id': owner.get('conv_id')},
            'goal_version': self.goal_contract.version}

    def _cancel_watched_job(self, job_id):
        from .watch_context import get_context
        from .job_watch import get_watch
        owner = get_context()
        watch = get_watch()
        job = watch.get(str(job_id)) or {}
        if not owner.get('username') or not owner.get('conv_id') or (job.get('username'), job.get('conv_id')) != (owner['username'], owner['conv_id']):
            raise PermissionError('cancellation is restricted to the executing user/session')
        if getattr(self.current_agent, 'name', '') != 'lead-orchestrator':
            return {'blocked': True, 'executed': False, 'reason': 'only main chat may cancel an owned job'}
        from .control_policy import require_user_control_source
        require_user_control_source(self._control_source, owner)
        from .job_control import pause_owned_workflow
        pause_owned_workflow(self.config.project_root, owner['username'], owner['conv_id'])
        result = watch.cancel(str(job_id), control_root=self.config.project_root / 'runs' / owner['username'] / owner['conv_id'])
        self.context['job_cancel'] = result
        self.context['pending_control_delivery'] = f"作业 {job_id}：{result.get('message') or result.get('error') or '取消结果尚待核验。'}\n科研任务未宣称完成；后继派发已暂停。"
        self._emit_lifecycle_event('job_cancel_control', {**result, 'main_already_handling': True})
        self._checkpoint('job_cancel_control')
        return result

    def _retarget_queued_job(self, job_id, nodelist, partition, memory_mb=None):
        from .defns import ORCHESTRATOR
        if self.current_agent and self.current_agent.name!='lead-orchestrator':
            return AgentResult(value='调度变更交回主chat核对真实用户授权，不执行shell取消/重提。',agent=ORCHESTRATOR,
                context_variables={'job_retarget_proposal':{'job_id':job_id,'nodelist':nodelist,'partition':partition}})
        if self.goal_contract.execution_mode in {'read_only','prepare_only'}:
            return {'blocked':True,'executed':False,'reason':'current user scope does not authorize scheduler mutation'}
        from .watch_context import get_context
        from .job_control import retarget_queued_job
        owner=get_context()
        from .control_policy import require_user_control_source
        if not self._control_source or self._control_source.get('origin') != 'user':
            # A background lifecycle turn may only apply the exact resource
            # patch produced from fresh monitor evidence and an operator-owned
            # tool profile. It cannot invent a node, RAM value, or science edit.
            receipt=self.context.get('resource_review') or {}
            suggested=receipt.get('suggested_resources') or {}
            exact=(receipt.get('status')=='propose_resource_patch'
                   and str(receipt.get('job_id') or job_id)==str(job_id)
                   and suggested.get('nodelist')==nodelist
                   and suggested.get('partition')==partition
                   and suggested.get('memory_mb')==memory_mb
                   and self.goal_contract.execution_mode=='workflow'
                   and self.goal_contract.execution_authorized)
            if not exact:
                require_user_control_source(self._control_source, owner)
        else:
            require_user_control_source(self._control_source, owner)
        result=retarget_queued_job(job_id,nodelist,partition,owner.get('username',''),owner.get('conv_id',''),self.config.project_root,memory_mb=memory_mb)
        self.context['job_retarget']=result
        if result.get('ok'):
            memory_note = f'，内存申请按工具资源画像调整为 {memory_mb} MiB' if memory_mb is not None else '，内存申请保持不变'
            self.context['pending_control_delivery'] = f'作业 {job_id} 已核验调往 {partition}/{nodelist}{memory_note}；未取消、未重提，科学参数不变。仍需等待调度器分配，主chat/监督继续监控，科研任务未完成。'
        self._emit_lifecycle_event('job_resources_updated',{**result,'main_already_handling':True})
        self._checkpoint('job_resource_update')
        return result

    def _record_workflow_draft(self, nodes, completion_criteria=""):
        if getattr(self.current_agent, 'name', '') != 'lead-orchestrator':
            raise PermissionError('main chat owns research planning')
        completion_criteria = str(completion_criteria or '').strip()
        if not completion_criteria:
            raise ValueError('draft requires a concrete final delivery/completion criterion')
        ids = [n['step_id'] for n in nodes]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError('draft requires unique node IDs')
        remaining = {n['step_id']: set(n.get('depends_on', [])) for n in nodes}
        if any(deps - set(ids) for deps in remaining.values()):
            raise ValueError('draft dependency references missing node')
        while remaining:
            roots = {k for k, deps in remaining.items() if not deps}
            if not roots: raise ValueError('draft dependency cycle')
            remaining = {k: deps - roots for k, deps in remaining.items() if k not in roots}
        previous = self.context.get('workflow_draft', {})
        draft = {'version': previous.get('version', 0) + 1, 'goal_version': self.goal_contract.version,
                 'executable': False, 'completion_criteria': completion_criteria,
                 'nodes': [{**copy.deepcopy(n), 'status': 'draft', 'done': False} for n in nodes]}
        self.context['workflow_draft'] = draft
        self._checkpoint('workflow_draft_saved')
        return {'ok': True, **draft}

    def _propose_workflow_patch(self, base_version: int, reason: str, changes: List[Dict]):
        if self.current_agent and self.current_agent.name != 'lead-orchestrator':
            from .defns import ORCHESTRATOR
            return AgentResult(value='专业Agent编排补丁建议交回主chat审核协商，未应用。', agent=ORCHESTRATOR,
                               context_variables={'workflow_patch_proposal': {'base_version': base_version, 'reason': reason, 'changes': changes}})
        from .task_line import get_store
        from .workflow_patch import patched_graph
        from .defns import resolve_agent
        line = get_store().get_line(self._current_line_id) if self._current_line_id else None
        if not line:
            raise ValueError('query/create the current structured workflow before proposing a patch')
        # Upgrade-era sessions may have submitted work before executable DAGs
        # became mandatory. Reconcile their persisted TaskLine labels from the
        # authoritative JobWatch receipts before recompiling; a terminal job is
        # history, not a live-node lock. This never changes or resubmits a job.
        watched = {str(job.get('job_id')): job for job in self._pending_jobs_for_conv() if job.get('job_id')}
        reconciled = False
        for history in line.get('steps', []):
            job_ids = [str(job_id) for job_id in history.get('job_ids', []) if str(job_id)]
            records = [watched.get(job_id) for job_id in job_ids]
            if not job_ids or not all(records) or not all(record.get('terminal') is True for record in records):
                continue
            failed = any(record.get('failed') is True or str(record.get('state', '')).upper().startswith(('FAIL', 'CANCEL', 'TIMEOUT'))
                         for record in records)
            status = 'failed' if failed else 'completed'
            if history.get('status') != status or history.get('done') == failed:
                get_store().upsert_step(
                    self._current_line_id, history['step_id'], status=status, done=not failed,
                    validation={**history.get('validation', {}), 'legacy_job_reconciled': True,
                                'job_states': {job_id: watched[job_id].get('state') for job_id in job_ids}},
                    note='terminal legacy job reconciled before DAG compilation',
                )
                reconciled = True
        if reconciled:
            line = get_store().get_line(self._current_line_id)
        ledger = self.recovery_gate.snapshot()
        runtime_history = self._runtime_snapshot() if self._runtime_snapshot else {}
        for history in line.get('steps', []):
            tool = history.get('tool')
            validation = history.get('validation', {})
            if ('expected_outputs' not in history and not history.get('job_ids') and not history.get('output_files')
                    and not history.get('recovery_key') and history.get('status') == 'running'
                    and validation.get('schema') == 'passed' and validation.get('execution') != 'passed'
                    and is_submission(tool, history.get('arguments', {}))
                    and history['step_id'] not in runtime_history.get('nodes', {})
                    and not any(e.get('tool') == tool for e in ledger.values())
                    and not any(c.get('tool') == tool for c in self.memory.tool_call_log)
                    and not any(j.get('tool') == tool for j in self._pending_jobs_for_conv())
                    and not any(not f.done() for f in self._active_tool_futures)):
                history.update(status='blocked', done=False)
                validation = {**validation, 'dispatch_not_entered': True,
                    'reason': 'pre-dispatch label has no tool invocation, durable intent or runtime claim'}
                history['validation'] = validation
                get_store().upsert_step(self._current_line_id, history['step_id'], status='blocked', done=False, validation=validation)
        current = int(line.get('plan_version') or max((s.get('plan_version', 0) for s in line.get('steps', [])), default=0))
        if current != int(base_version):
            raise ValueError('patch base version is stale')
        changes = copy.deepcopy(changes)
        from .workspace import validate_scope_component
        from .watch_context import get_context
        owner_context = get_context()
        from pathlib import Path
        conversation_root = Path(self.config.project_root) / 'runs' / validate_scope_component(owner_context.get('username') or 'default') / validate_scope_component(owner_context.get('conv_id') or self._current_line_id or 'default')
        # TaskLine also contains control/audit records such as
        # record_workflow_draft.  Only typed workflow nodes belong in the
        # executable graph; feeding control records to the compiler creates a
        # phantom DAG node and makes the initial graph differ from its draft.
        executable_steps = [
            step for step in line.get('steps', [])
            if 'expected_outputs' in step and not step.get('branch_parent')
        ]
        from .workflow_compiler import compile_workflow_changes
        compilation = compile_workflow_changes(
            changes, self.registry, self.config.project_root, conversation_root,
            existing_steps=executable_steps,
        )
        changes = compilation.changes
        compiler_receipt = {'defaults_applied': compilation.defaults_applied}
        for change in changes:
            if change['operation'] != 'upsert':
                continue
            node = change.get('node') or {}
            owner = resolve_agent(node.get('agent', ''))
            if node.get('tool') not in {f.__name__ for f in owner.functions}:
                raise ValueError('workflow node tool is not exposed to its assigned agent')
            if node.get('agent') in {'monitor', 'supervisor'} and is_submission(node.get('tool'), node.get('arguments', {})):
                raise ValueError('monitor/supervisor cannot own compute submission nodes')
            # Missing scientific choices are checked against the typed node
            # that would actually submit work, not guessed from words in the
            # user's message. Read-only investigations therefore never ask for
            # an irrelevant GCMC/cDFT choice, while real compute still cannot
            # invent protected conditions.
            if is_submission(node.get('tool'), node.get('arguments', {})):
                from .goal_contract import _TOOL_METHOD, extract_gases
                method = _TOOL_METHOD.get(node.get('tool'), '')
                node_gases = extract_gases(' '.join(str(node.get('arguments', {}).get(key, ''))
                                                     for key in ('gas', 'gases')))
                missing_choices = []
                if method and not self.goal_contract.method:
                    missing_choices.append('scientific method')
                if node_gases and not self.goal_contract.gases:
                    missing_choices.append('gas scope')
                if node.get('arguments', {}).get('temperature') is not None and 'temperature_K' not in self.goal_contract.parameters:
                    missing_choices.append('temperature')
                if missing_choices:
                    from .workflow_patch import WorkflowContractError
                    raise WorkflowContractError('compute node uses scientific choices not fixed by the user goal', {
                        'step_id': change['step_id'], 'tool': node.get('tool'),
                        'missing_choices': missing_choices,
                        'next_action': 'Main chat asks one concise scientific question for these missing choices. Do not dispatch or invent defaults; read-only work may continue independently.',
                    })
            ambiguous_locks = set(node.get('resource_locks', [])) & set(self.registry.all_names())
            if ambiguous_locks:
                from .workflow_patch import WorkflowContractError
                raise WorkflowContractError('resource lock must identify a shared resource, not a bare tool name', {
                    'step_id': change['step_id'], 'ambiguous_locks': sorted(ambiguous_locks),
                    'next_action': 'use real shared paths or an explicitly scoped resource name; remove accidental tool-wide locks when independent workspaces share no mutable resource. Dispatch concurrency is already limited by the executor.',
                    'calculation_dir': node.get('arguments', {}).get('work_dir') or node.get('arguments', {}).get('job_work_dir')})
        nodes, affected = patched_graph(executable_steps, changes, self.config.project_root)
        if current == 0 or self.context.get('workflow_recompile_required'):
            # Initial execution is compiled from one complete, persisted DAG.
            # Runtime failures may later create versioned local patches, but a
            # model may not start with one convenient step and promise to append
            # the rest after it runs.
            draft = self.context.get('workflow_draft') or {}
            draft_nodes = draft.get('nodes') or []
            from .workflow_patch import WorkflowContractError
            if (draft.get('goal_version') != self.goal_contract.version
                    or not str(draft.get('completion_criteria') or '').strip()):
                raise WorkflowContractError('initial executable DAG requires a current complete workflow draft', {
                    'next_action': (
                        'Call record_workflow_draft first with the entire path from inputs/preparation through '
                        'calculation, validation, analysis and the requested final deliverable. Do not compile '
                        'only the first runnable step or promise to append later.'),
                })
            missing_parameters = {item.get('step_id'): item.get('missing_parameters', [])
                                  for item in draft_nodes if item.get('missing_parameters')}
            compiled = {item.get('step_id'): item for item in nodes}
            outlined = {item.get('step_id'): item for item in draft_nodes}
            structural_mismatch = []
            if set(compiled) != set(outlined):
                structural_mismatch.append({
                    'draft_only': sorted(set(outlined) - set(compiled)),
                    'compiled_only': sorted(set(compiled) - set(outlined)),
                })
            for step_id in sorted(set(compiled) & set(outlined)):
                draft_node, compiled_node = outlined[step_id], compiled[step_id]
                if list(draft_node.get('depends_on', [])) != list(compiled_node.get('depends_on', [])):
                    structural_mismatch.append({'step_id': step_id, 'field': 'depends_on'})
                if draft_node.get('agent') and draft_node.get('agent') != compiled_node.get('agent'):
                    structural_mismatch.append({'step_id': step_id, 'field': 'agent'})
                if draft_node.get('tool') and draft_node.get('tool') != compiled_node.get('tool'):
                    structural_mismatch.append({'step_id': step_id, 'field': 'tool'})
            if missing_parameters or structural_mismatch:
                raise WorkflowContractError('initial executable DAG does not match the complete persisted draft', {
                    'missing_parameters': missing_parameters,
                    'structural_mismatch': structural_mismatch,
                    'completion_criteria': draft.get('completion_criteria'),
                    'next_action': (
                        'Resolve only genuinely missing scientific choices, then persist one revised complete draft '
                        'and compile every draft node in the same proposal. Runtime-derived file names belong in '
                        'artifact contracts; they are not a reason to omit downstream nodes.'),
                })
        from .workflow_patch import workflow_efficiency_issues
        efficiency_issues = workflow_efficiency_issues(nodes, self.config.project_root)
        if efficiency_issues:
            from .workflow_patch import WorkflowContractError
            raise WorkflowContractError(
                'DAG contains redundant dispatches or incorrect data-flow edges',
                {
                    'issues': efficiency_issues,
                    'next_action': (
                        'Recompile the structured DAG: merge every objectively batchable group into one node/tool call; '
                        'keep independent nodes dependency-free so the executor can run them in parallel; add depends_on '
                        'only for actual producer→consumer data flow. Do not ask the user to repeat unchanged science.'
                    ),
                },
            )
        # A charge generator and an electrostatic simulation are distinct
        # scientific operations.  Prompt guidance alone is not sufficient:
        # require a typed, read-only charge validation node on every dependency
        # path from PACMOF/PACMAN into GCMC.  This prevents a model from treating
        # successful charge-file creation as proof of the declared cell charge.
        node_by_id = {n.get('step_id'): n for n in nodes if n.get('step_id')}

        def _ancestors(step_id):
            seen, pending = set(), list(node_by_id.get(step_id, {}).get('depends_on', []))
            while pending:
                dependency = pending.pop()
                if dependency in seen:
                    continue
                seen.add(dependency)
                pending.extend(node_by_id.get(dependency, {}).get('depends_on', []))
            return seen

        for candidate in nodes:
            if candidate.get('tool') not in {'run_gcmc_isotherm', 'run_gcmc_batch'}:
                continue
            ancestry = _ancestors(candidate.get('step_id'))
            generated = [step for step in ancestry
                         if node_by_id.get(step, {}).get('tool') == 'run_pacman_charge']
            if not generated:
                continue
            validators = [step for step in ancestry
                          if node_by_id.get(step, {}).get('tool') == 'validate_framework_charges'
                          and set(generated) & _ancestors(step)]
            if not validators:
                from .workflow_patch import WorkflowContractError
                raise WorkflowContractError(
                    'PACMOF/PACMAN output must be validated before GCMC; charge generation is not charge-model verification',
                    {
                        'step_id': candidate.get('step_id'),
                        'charge_steps': generated,
                        'required_tool': 'validate_framework_charges',
                        'required_dependency_order': 'run_pacman_charge -> validate_framework_charges -> GCMC',
                        'expected_net_charge': self.goal_contract.parameters.get('expected_cell_net_charge'),
                        'next_action': (
                            'Query validate_framework_charges schema, add a read-only validation node with the declared '
                            'expected cell charge and PACMOF/PACMAN provenance, then make this GCMC node depend on it. '
                            'If expected cell charge is unknown, ask the user; never force residual charge onto an atom.'
                        ),
                    },
                )
        resource_nodes=[{**c['node'], 'step_id': c['step_id']} for c in changes if c.get('operation')=='upsert' and c.get('node',{}).get('tool')=='run_cdft' and c['node'].get('arguments',{}).get('action','pipeline') in {'pipeline','submit'}]
        if self._on_resource_review and resource_nodes:
            resource_receipt=self._on_resource_review({'kind':'before_submission','nodes':resource_nodes,'goal_version':self.goal_contract.version})
            self.context['resource_review']=resource_receipt
            if resource_receipt.get('status')!='ready':
                if resource_receipt.get('status')=='review_retry':
                    return {'ok':False,'status':'resource_review_retry','retryable':True,'executed':False,'resource_review':resource_receipt,
                            'next_action':'Resource tool/schema exchange failed; repair reviewer evidence delivery without asking user to repeat an existing budget.'}
                self.context['pending_resource_plan'] = {'base_version': base_version, 'reason': reason, 'changes': changes}
                self._checkpoint('resource_plan_waiting')
                return {'ok':False,'status':'waiting_resources','retryable':True,'executed':False,'resource_review':resource_receipt,
                        'next_action':'Resource agent owns allocation. Preserve the candidate and obtain fresh resource evidence; do not ask the user for RAM or alter science.'}
            for change in changes:
                if change.get('operation')=='upsert' and change.get('node',{}).get('tool')=='run_cdft':
                    change['node']['arguments']['resource_review_id']=resource_receipt['resource_review_id']
                    allocation = resource_receipt.get('resource_allocations', {}).get(change['step_id'], {})
                    for key in ('memory_mb', 'nodelist', 'partition'):
                        if allocation.get(key) is not None:
                            change['node']['arguments'].setdefault(key, allocation[key])
            nodes, affected=patched_graph(executable_steps,changes,self.config.project_root)
        if self._runtime_snapshot:
            from .parallel_workflow import ACTIVE
            runtime = self._runtime_snapshot()
            if any(key in affected and value['status'] in ACTIVE for key, value in runtime.get('nodes', {}).items()):
                raise ValueError('affected runtime node still owns running/unknown resources; reconcile before patching')
        new_version = max(current + 1, self.goal_contract.pending_plan_version + 1)
        patch = {'base_version': current, 'new_version': new_version, 'changes': changes,
                 'reason': reason, 'affected_nodes': sorted(affected), 'proposed_nodes': nodes,
                 'compiler_receipt': compiler_receipt}
        self.context.pop('workflow_patch_proposal', None)
        # Initial compilation may use the user's explicit "直接执行" authority;
        # later repairs always go back through main-chat negotiation.
        existing_ids = {n['step_id'] for n in executable_steps}
        no_live_legacy_job = not any(
            n.get('job_ids') and any(
                not watched.get(str(job_id), {}).get('terminal', False)
                for job_id in n.get('job_ids', [])
            )
            for n in line.get('steps', [])
        )
        fresh_initial = all(c['operation'] == 'upsert' and c['step_id'] not in existing_ids for c in changes)
        legacy_recompile = bool(line.get('steps')) and not runtime_history.get('nodes') and no_live_legacy_job
        initial = (current == 0 and all(c['operation'] == 'upsert' for c in changes)
                   and (fresh_initial or legacy_recompile)
                   and not any(n.get('status') in {'running', 'submitted', 'unconfirmed'} and n.get('job_ids') for n in line.get('steps', [])))
        # A repair proposal necessarily differs from the currently approved
        # node.  Check protected science here; exact identity is enforced again
        # after the new graph version is committed.
        within_goal = all(self.goal_contract.guard_tool_call(
                              n.get('tool', ''), n.get('arguments', {}),
                              enforce_approved_node=False)[0]
                          for n in nodes if n.get('arguments'))
        covered_gases = set()
        from .goal_contract import extract_gases
        for n in nodes:
            covered_gases |= extract_gases(' '.join(str(n.get('arguments', {}).get(k, '')) for k in ('gas', 'gases')))
        complete_gas_scope = not self.goal_contract.gases or covered_gases == set(self.goal_contract.gases)
        non_submission_plan = all(not is_submission(node.get('tool'), node.get('arguments', {})) for node in nodes)
        plan_authorized = self.goal_contract.execution_authorized or (
            self.goal_contract.execution_mode in {'read_only', 'prepare_only'} and non_submission_plan
        )
        if initial and plan_authorized and within_goal and complete_gas_scope:
            applied = get_store().apply_workflow_patch(self._current_line_id, changes, current, new_version, reason, self.config.project_root)
            self.goal_contract.approved_nodes = [
                n for n in applied['steps']
                if n.get('arguments') is not None
                and 'expected_outputs' in n
                and not n.get('branch_parent')
            ]
            self.goal_contract.approved_plan_version = new_version
            self.goal_contract.pending_plan_version = new_version
            self._plan_steps_created = True
            self.context.pop('workflow_recompile_required', None)
            self._emit_lifecycle_event('plan_patch_applied', {'patch': patch, 'main_already_handling': True, 'initial_user_authorized': True})
            activation = self._activate_approved_workflow()
            return {'ok': True, 'status': 'approved', 'patch': patch, 'applied': True,
                    'execution': activation, 'proposed_nodes': nodes,
                    'compiler_receipt': compiler_receipt}
        self._pending_workflow_patch = patch
        self.goal_contract.pending_plan_version = new_version
        # A repair inside an already approved execution scope is staged for the
        # main model to apply, not automatically turned into another user veto.
        from .workflow_patch import operational_patch
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        if (self.goal_contract.execution_authorized and within_goal and not runtime.get('user_paused')
                and operational_patch(line.get('steps', []), changes, self.config.project_root)):
            self._pending_user_interaction = None
            self._waiting_for_user_input = False
            self._emit_lifecycle_event('plan_patch_proposed', {'patch': patch, 'requires_user': False,
                'main_already_handling': True, 'basis': 'unchanged scientific contract, existing execution scope'})
            return {'ok': True, 'status': 'ready_to_apply', 'patch': patch, 'applied': False,
                    'proposed_nodes': nodes, 'compiler_receipt': compiler_receipt,
                    'next_action': 'Main chat applies this pending version under the existing user instruction; do not repeat the same confirmation.'}
        diff_lines=[]
        for change in changes:
            if change['operation']=='remove':diff_lines.append('移除节点：'+change['step_id'])
            else:
                node=change['node'];args=node.get('arguments',{})
                summary='；'.join(f'{key}={value}' for key,value in args.items() if key not in {'resource_review_id'})
                diff_lines.append(f"更新节点 {change['step_id']}：{node.get('agent')} → {node.get('tool')}；依赖：{', '.join(node.get('depends_on',[])) or '无'}\n参数：{summary}")
        prompt = (f'建议修补编排 v{current} → v{new_version}，尚未执行。\n原因：{reason}\n'
                  +'\n'.join(diff_lines)+'\n是否按这个安排继续？也可以直接说明希望改变的地方；无需使用特定回复词。')
        self._waiting_for_user_input = True
        self._pending_user_interaction = {'tool': 'workflow_patch_decision', 'params': {}, 'prompt': prompt}
        self._emit_lifecycle_event('plan_patch_proposed', {'patch': patch, 'requires_user': True})
        return {'ok': True, 'status': 'needs_user', 'patch': patch, 'applied': False,
                'proposed_nodes': nodes, 'compiler_receipt': compiler_receipt}

    def _finish_workflow_node(self,step_id,evidence_call_ids,conclusion):
        if not self._on_workflow_finish:raise ValueError('no owned workflow runtime')
        receipt=self._on_workflow_finish(step_id,evidence_call_ids,conclusion)
        if receipt.get('already_finished'):
            return receipt
        if receipt.get('phase')=='finish':
            runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
            if runtime.get('status') == 'completed' and self._workflow_completion_state()['ok']:
                self.context.pop('pending_control_delivery', None)
                return {**receipt, 'workflow_completed': True, 'next_step_state': 'completed',
                        'next_action': 'Deliver the verified existing results; no scientific nodes remain.'}
            next_state = receipt.get('next_step_state', 'triggered')
            if next_state == 'user_paused':
                # No keyword-based resume and no forced end: the main model
                # reads the latest real request and chooses execute_workflow
                # if the user asked to continue, otherwise reports the pause.
                return receipt
            self.context['pending_control_delivery'] = ('这一阶段的结果已检查通过，正在推进下一步。整体科研任务尚未完成。' if next_state == 'triggered' else
                '这一阶段的结果已检查通过；按要求保留暂停，不会启动后续计算。' if next_state == 'user_paused' else
                '这一阶段的结果已检查通过，但下一步还未启动。原因：' + receipt.get('resume_error', '状态待核对'))
        return receipt

    def _apply_workflow_patch(self, plan_version):
        """Main model chooses the operation; kernel checks provenance/version."""
        from .control_policy import require_user_control_source
        from .watch_context import get_context
        from .task_line import get_store
        if getattr(self.current_agent,'name','')!='lead-orchestrator':
            raise PermissionError('main chat owns workflow negotiation')
        require_user_control_source(self._control_source,get_context())
        patch=copy.deepcopy(self._pending_workflow_patch)
        if not patch or patch['new_version']!=plan_version:
            raise ValueError('a matching pending structured patch is required; no text plan or inferred approval')
        # A patch may have been proposed before a hot upgrade normalized
        # project-anchored output contracts. Re-normalize at the actual apply
        # boundary so persisted ``runs/...`` paths cannot be joined twice.
        from .workflow_patch import canonical_expected_outputs
        for change in patch.get('changes', []):
            if change.get('operation') == 'upsert' and change.get('node'):
                change['node']['expected_outputs'] = canonical_expected_outputs(
                    change['node'].get('expected_outputs', []), self.config.project_root)
        self._pending_workflow_patch = patch
        if self._runtime_snapshot:
            from .parallel_workflow import ACTIVE
            if any(node.get('status') in ACTIVE for key,node in self._runtime_snapshot().get('nodes',{}).items() if key in patch['affected_nodes']):
                raise ValueError('affected node still owns live/unknown resources; reconcile before applying')
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        line=get_store().apply_workflow_patch(self._current_line_id,patch['changes'],patch['base_version'],plan_version,patch['reason'],self.config.project_root,
                                            runtime_nodes=runtime.get('nodes', {}))
        # Old diagnostic tool receipts are audit history, not delegated nodes.
        self.goal_contract.approved_nodes=[n for n in line['steps'] if n.get('arguments') is not None and 'expected_outputs' in n]
        self.goal_contract.approved_plan_version=plan_version
        self.goal_contract.pending_plan_version=plan_version
        self.goal_contract.execution_authorized=True
        self.context.pop('workflow_recompile_required', None)
        self._pending_workflow_patch=None
        self._pending_user_interaction=None
        self._waiting_for_user_input=False
        self._emit_lifecycle_event('plan_patch_applied',{'patch':patch,'main_already_handling':True})
        activation=self._activate_approved_workflow()
        if activation.get('scheduled'):
            self.context['pending_control_delivery'] = '执行安排已保存并交给执行器，尚不能确认计算已经启动；实际提交以作业回执为准。'
        self._checkpoint('model_applied_workflow_patch')
        return {'ok':bool(activation.get('scheduled')),'applied':True,'plan_version':plan_version,
                'status': 'execution_staged' if activation.get('scheduled') else 'activation_blocked', 'execution':activation,
                'next_action': 'Await actual dispatch receipt.' if activation.get('scheduled') else
                    'Plan was saved, execution was NOT started. Diagnose the returned blocked nodes and restore/repair them; do not ask for the same approval again.'}

    def _discard_workflow_patch(self, plan_version, reason):
        if getattr(self.current_agent, 'name', '') != 'lead-orchestrator':
            raise PermissionError('main chat owns candidate plan decisions')
        patch = self._pending_workflow_patch
        if not patch or patch['new_version'] != plan_version:
            raise ValueError('only the exact current pending proposal can be withdrawn')
        self.context.setdefault('withdrawn_workflow_patches', []).append({'patch': patch, 'reason': reason})
        self._pending_workflow_patch = None
        if (self._pending_user_interaction or {}).get('tool') == 'workflow_patch_decision':
            self._pending_user_interaction = None
            self._waiting_for_user_input = False
        self._emit_lifecycle_event('plan_patch_discarded', {'plan_version': plan_version, 'reason': reason,
            'main_already_handling': True, 'execution_unchanged': True})
        self._checkpoint('model_withdrew_workflow_proposal')
        return {'ok': True, 'discarded': True, 'plan_version': plan_version, 'execution_unchanged': True}

    def _activate_approved_workflow(self):
        """Approval commits before scheduling; never fall back to legacy dispatch."""
        if not self._on_workflow_start:
            return {'status': 'not_attached', 'scheduled': False}
        self._checkpoint('approved_workflow_before_activation')
        try:
            runtime = self._on_workflow_start(self.goal_contract.approved_plan_version)
            if runtime.get('scheduled') is False or runtime.get('status') in {'needs_recovery', 'activation_blocked'}:
                return {'status': 'activation_blocked', 'scheduled': False, 'reason': runtime.get('reason', 'Execution requires recovery'), 'runtime': runtime}
            return {'status': 'scheduled', 'scheduled': True, 'runtime': runtime}
        except Exception as error:
            receipt = {'status': 'activation_blocked', 'scheduled': False, 'reason': str(error),
                       'next_action': 'reconcile_legacy_dispatch_or_propose_workflow_patch',
                       'legacy_fallback_allowed': False}
            self._emit_lifecycle_event('workflow_activation_blocked', {**receipt, 'main_already_handling': True})
            return receipt

    def resume_lifecycle_event(self, event):
        from .defns import ORCHESTRATOR
        resource_receipt=event.get('supervisor_receipt',{}).get('resource_review',{})
        if event.get('kind')=='job_pending_warning' and resource_receipt:
            self.current_agent=ORCHESTRATOR
            self.context['resource_review']=resource_receipt
            self.context['resource_queue_instruction'] = 'Normal scheduler PENDING is not a user decision. Preserve the original job; assess resources and use authorized operational controls when evidence supports them. Do not ask the user for RAM or resubmit queued work.'
        review = event.get('supervisor_receipt', {}).get('scientific_review')
        receipt = event.get('supervisor_receipt', {})
        if receipt.get('review_action') == 'ask_user':
            self.current_agent = ORCHESTRATOR
            self.task_complete = False
            issues = '; '.join(review.get('issues', [])) if review else receipt.get('reason', '')
            result = self._request_user_decision('独立科学审查仍未完成或修正后仍未通过。请确认补充资料、调整方案，或继续查证预算。',
                issues or '科学审查预算已达到上限；禁止自动继续提交。', [])
            self.last_text = self._pending_user_interaction['prompt'] + '\n具体原因：' + issues
            self._checkpoint('scientific_review_needs_user')
            return self.last_text
        if receipt.get('review_action') == 'correct_main':
            self.context['active_scientific_review_scope'] = receipt['scope_id']
            self.context['scientific_correction_scope'] = receipt['scope_id']
        if review and review.get('passed') is False:
            event = copy.deepcopy(event)
            audit_incomplete = event.get('supervisor_receipt', {}).get('review_status') == 'insufficient_evidence'
            event.setdefault('payload', {})['correction_contract'] = {
                'action': 'supervisor_source_verification_incomplete' if audit_incomplete else 'correct_the_actual_claims_against_authoritative_evidence',
                'issues': review.get('issues', []), 'new_submission_allowed': False,
                'goal_scope_change_allowed': False,
                'note': ('The supervisor did not verify its sources; this is NOT proof the main answer or completed work failed. Preserve prior facts and results; handle the audit gap, do not repeat main computations or erase the prior answer.' if audit_incomplete else
                         'Revise only unsupported professional claims or report uncertainty; do not regenerate scientific results or silently replace a forcefield.')}
        # Supervisor evidence is shared structured tool output, not its private reasoning.
        known = {c.get('call_id') for c in self.memory.tool_call_log}
        incoming = list(event.get('supervisor_receipt', {}).get('evidence_calls', []))
        if event.get('payload', {}).get('evidence_call'):
            incoming.append(event['payload']['evidence_call'])
        for call in incoming:
            if call.get('call_id') not in known:
                shared = copy.deepcopy(call)
                self.memory.tool_call_log.append(shared)
                self.memory._agent_mem(shared.get('agent') or 'supervisor')['tool_call_log'].append(shared)
                known.add(call.get('call_id'))
        if self._pending_user_interaction and self._pending_user_interaction.get('tool') == 'lifecycle_wait':
            self._pending_user_interaction = None
            self._waiting_for_user_input = False
        self.current_agent = ORCHESTRATOR
        self.messages.append({'role': 'user', 'content': '[生命周期事件]\n' + json.dumps(event, ensure_ascii=False, default=str)})
        self.task_complete = False
        try:
            return self._execute_loop(ORCHESTRATOR)
        finally:
            self.context.pop('scientific_correction_scope', None)

    def observe_lifecycle_event(self, event):
        """Independent, durable, bounded read-only supervisor loop."""
        from .defns import SUPERVISOR
        self.current_agent = SUPERVISOR
        self.messages.append({'role': 'user', 'content': '[生命周期事件]\n' + json.dumps(event, ensure_ascii=False, default=str)})
        start = len(self.memory.tool_call_log)
        verification_tools = set(event.get('payload', {}).get('verification_tools', []))
        self._scientific_verification_tools = verification_tools if event.get('kind') == 'scientific_review' else set()
        from .scientific_review import knowledge_fingerprint, is_source_result, canonical_source_inputs
        from .watch_context import get_context
        owner = get_context()
        source_scope = hashlib.sha256(json.dumps({'question': event.get('payload', {}).get('question'),
            'owner': [owner.get('username'), owner.get('conv_id')],
            'knowledge': knowledge_fingerprint(self.config.project_root), 'inputs': canonical_source_inputs(event.get('payload', {}).get('source_inputs', [])),
            'goal_version': self.goal_contract.version, 'verification_tools': sorted(verification_tools)},
            sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        prior_source = self.context.get('scientific_review_source', {})
        existing_calls = {c['call_id'] for c in self.memory.tool_call_log if c.get('tool') in verification_tools
            and is_source_result(c['tool'], result_object(self._load_evidence_call(c).get('result')))}
        verified_sources = [cid for cid in prior_source.get('call_ids', []) if cid in existing_calls] if prior_source.get('scope') == source_scope else []
        from pathlib import Path
        for item in event.get('payload', {}).get('source_inputs', []):
            path = Path(item['path'])
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                verified_sources = []
                break
        delivered_verdict = False
        self._scientific_source_required = bool(verification_tools and not verified_sources)
        default_actions = {'job_completed': 'advance_dependencies', 'tool_failed': 'diagnose_and_fix',
                           'tool_outcome': 'diagnose_and_fix', 'plan_patch_proposed': 'diagnose_and_fix',
                           'tool_uncertain': 'wait_existing', 'user_paused': 'wait_existing', 'stalled': 'wait_existing', 'waiting_jobs': 'wait_existing', 'result_verified': 'advance_dependencies'}
        decision = {'next_action': default_actions.get(event['kind'], 'diagnose_and_fix'),
                    'reason': 'Resume from the structured workflow; never duplicate an unresolved submission',
                    'evidence_refs': [event['event_id']]}
        allowed = {'task_line_query', 'recovery_state', 'lifecycle_state', 'check_job', 'diagnose_job', 'read_file',
                   'discover_forcefield', 'inspect_forcefield', 'validate_framework_charges', 'convert_physical_units',
                   'query_literature', 'supervisor_decision'}
        try:
            for observation_round in range(3):
                self._supervisor_finalizing = observation_round == 2
                if self._supervisor_finalizing:
                    self.messages.append({'role': 'user', 'content':
                        '只读观察预算的最后一轮：现在仅能调用supervisor_decision交付，不再查询生命周期或其他工具。'
                        'scientific_review事件必须填写scientific_review.passed/issues；证据不足就passed=false并说明实际缺项，不得默认通过。'})
                response = self._call_api(SUPERVISOR)
                calls = [b for b in response.content if getattr(b, 'type', '') == 'tool_use']
                if not calls:
                    self.messages.append({'role': 'assistant', 'content': response.content})
                    self.messages.append({'role': 'user', 'content':
                        '观察尚未交付：请调用supervisor_decision提交结构化结论；scientific_review事件必须包含passed和issues。不要重复查询已获得的证据。'})
                    continue
                self.messages.append({'role': 'assistant', 'content': response.content})
                results = []
                completed_decision = False
                for call in calls:
                    if call.name not in allowed:
                        raw = {'blocked': True, 'reason': 'supervisor is read-only; no writes/submissions/handoffs'}
                    else:
                        issues = self.registry.validate_params(call.name, call.input)
                        if issues:
                            raw = {'error': 'supervisor parameters invalid', 'issues': issues,
                                   'input_schema': self.registry.get(call.name).input_schema}
                        elif call.name == 'supervisor_decision':
                            missing_verdict = event.get('kind') == 'scientific_review' and 'scientific_review' not in call.input
                            missing_source = (event.get('kind') == 'scientific_review' and verification_tools
                                and not verified_sources and (call.input.get('scientific_review', {}).get('passed') is True
                                    or not self._supervisor_finalizing))
                            if missing_verdict or missing_source:
                                raw = {'error': 'scientific_review requires a structured verdict backed by successful independent source evidence' if missing_source else 'scientific_review event requires a structured passed/issues verdict',
                                       'required_verification_tools': sorted(verification_tools),
                                       'input_schema': self.registry.get('supervisor_decision').input_schema}
                                content = json.dumps(raw, ensure_ascii=False)
                                self.memory.record_tool_call('supervisor', call.name, call.input, content, failed=True)
                                results.append({'type':'tool_result','tool_use_id':call.id,'content':content})
                                continue
                            decision = dict(call.input)
                            raw = {'ok': True, 'delivered_to': 'main_chat'}
                            completed_decision = True
                            delivered_verdict = True
                        elif call.name == 'recovery_state':
                            raw = self._recovery_state()
                        elif call.name == 'lifecycle_state':
                            raw = self._lifecycle_state()
                        elif call.name == 'task_line_query':
                            raw = self._task_line_query(**call.input)
                        else:
                            raw = self.registry.execute_dict(call.name, call.input)
                    content = json.dumps(raw, ensure_ascii=False, default=str)
                    self.memory.record_tool_call('supervisor', call.name, call.input, content,
                                                 failed=bool(failure_reason(raw)))
                    if call.name in verification_tools and is_source_result(call.name, raw):
                        verified_sources.append(self.memory.tool_call_log[-1]['call_id'])
                        self._scientific_source_required = False
                    results.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': content})
                self.messages.append({'role': 'user', 'content': results})
                self._checkpoint('supervisor_observation')
                if completed_decision:
                    break
        except Exception as error:
            decision['observer_error'] = str(error)
            # Supervisor/API outage must not permanently stall the main workflow.
        finally:
            self._supervisor_finalizing = False
            self._scientific_verification_tools = set()
            self._scientific_source_required = False
        decision = self._audit_supervisor_decision(decision, event)
        if event.get('kind') == 'scientific_review' and 'scientific_review' not in decision:
            decision['scientific_review'] = {'passed': False, 'issues': ['scientific review incomplete; no structured verdict obtained']}
        decision['evidence_calls'] = self.memory.tool_call_log[start:]
        if verification_tools:
            self.context['scientific_review_source'] = {'scope': source_scope, 'call_ids': verified_sources}
            decision['independent_source_call_ids'] = verified_sources
            decision['review_status'] = ('insufficient_evidence' if not verified_sources or not delivered_verdict else
                'verified' if decision.get('scientific_review', {}).get('passed') is True else 'rejected_content')
        self.memory.record_step('supervisor_decision', 'done', json.dumps(decision, default=str), 'supervisor')
        self.memory.record_exit_report('supervisor', json.dumps(decision, ensure_ascii=False, default=str))
        self._checkpoint('supervisor_receipt')
        return decision

    def _audit_supervisor_decision(self, decision, event):
        """A TaskLine label is not a scheduler receipt. LLM advice is not fact."""
        decision = copy.deepcopy(decision)
        jobs = self._pending_jobs_for_conv()
        self.recovery_gate.sync_jobs(jobs)
        entries = self.recovery_gate.snapshot()
        runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
        pending = [j for j in jobs if j.get('terminal') is not True and j.get('job_id')]
        failed = [key for key, entry in entries.items() if entry.get('status') == 'failed']
        uncertain = [key for key, entry in entries.items() if entry.get('status') in {'reserved', 'uncertain', 'submitted'}]
        active_nodes = [key for key, node in runtime.get('nodes', {}).items()
                        if node.get('status') in {'running', 'waiting_jobs', 'waiting_prerequisite', 'uncertain'}]
        facts = {'jobs': [{key: j.get(key) for key in ('job_id', 'gas', 'state', 'terminal', 'failed')} for j in jobs],
                 'failed_recovery_keys': failed, 'unresolved_recovery_keys': uncertain,
                 'active_runtime_nodes': active_nodes, 'runtime_status': runtime.get('status', 'not_started')}
        decision['authoritative_facts'] = facts
        requires_user = bool(event.get('payload', {}).get('requires_user'))
        action = decision.get('next_action')
        if requires_user and action != 'ask_user':
            decision.update(next_action='ask_user', reason='主chat需要处理已持久化的用户确认，监督不得用等待作业替代协商。', audited=True)
        elif action == 'ask_user' and not requires_user and event.get('kind') in {
                'tool_failed', 'tool_outcome', 'reflection_stalled',
                'workflow_activation_blocked', 'plan_patch_proposed'}:
            decision.update(
                next_action='diagnose_and_fix', audited=True,
                reason=(
                    '该事件没有新的科研方法/条件缺项。schema、路径、依赖、环境或框架错误必须由主chat自行修复；'
                    '连续无法修复时委派patcher，不得把内部错误升级给用户。'
                ),
            )
        elif action == 'wait_existing' and not (pending or uncertain or active_nodes):
            decision.update(next_action='diagnose_and_fix' if failed else 'advance_dependencies',
                reason='实际回执没有未完成作业或活动节点；不能按TaskLine的running文本继续等待。' +
                       ('已有明确失败，应诊断修复并由主chat协商编排。' if failed else '应检查产物并推进剩余依赖。'), audited=True)
        elif action == 'verified_complete' and (runtime.get('status') != 'completed' or failed or uncertain or pending):
            decision.update(next_action='diagnose_and_fix', reason='尚无完整运行时产物验收，或存在失败/未知派发；不能确认任务已交付。', audited=True)
        return decision

    def run_readonly_observer(self, prompt, agent, max_rounds=6):
        """Resource monitor never hands control to the main/compute agents."""
        self.current_agent = agent
        self.messages = [{'role': 'user', 'content': prompt}]
        allowed = {'check_job', 'list_my_jobs', 'diagnose_job', 'read_file', 'grep_search', 'resource_health','assess_job_resources'}
        if agent.name == 'harness-maintainer':
            allowed.add('run_project_regressions')
            allowed.add('build_project_frontend')
        for _ in range(max_rounds):
            response = self._call_api(agent)
            calls = [b for b in response.content if getattr(b, 'type', '') == 'tool_use']
            if not calls:
                return '\n'.join(b.text for b in response.content if getattr(b, 'type', '') == 'text') or '暂无足够的资源/作业证据，不能确认正常。'
            self.messages.append({'role': 'assistant', 'content': response.content})
            results = []
            for call in calls:
                if call.name not in allowed or call.name not in {f.__name__ for f in agent.functions}:
                    raw = {'blocked': True, 'reason': 'monitor only returns resource/job health; no handoff, submit, writes or plan patches'}
                else:
                    raw = self.registry.execute_dict(call.name, call.input)
                result = json.dumps(raw, ensure_ascii=False, default=str)
                self.memory.record_tool_call(agent.name, call.name, call.input, result, failed=bool(failure_reason(raw)))
                results.append({'type': 'tool_result', 'tool_use_id': call.id, 'content': result})
            self.messages.append({'role': 'user', 'content': results})
        return '监控只读检查达到上限；请查看实际工具状态，未知状态不能视为正常。'

    _CHAIN_CONTROL_TOOLS = frozenset({
        'record_workflow_draft',
        'get_tool_schema', 'task_line_query', 'task_line_update', 'check_job',
        'list_my_jobs', 'diagnose_job', 'recovery_state', 'prepare_retry',
        'accept_recovered_result', 'lifecycle_state', 'propose_workflow_patch',
        'apply_workflow_patch', 'discard_workflow_patch', 'request_user_decision',
        'reconcile_watched_job', 'execute_workflow', 'message_workflow_node',
        'resolve_local_workflow_write', 'retarget_queued_job', 'cancel_watched_job',
        'revalidate_workflow_node_outputs', 'finish_workflow_node',
    })

    def _chain_required_block(self, tool: str) -> str:
        """Require a session-owned DAG before production tools or handoffs."""
        if (not self._current_line_id or not self._on_workflow_start
                or tool in self._CHAIN_CONTROL_TOOLS):
            return ''
        if not self.goal_contract.approved_nodes:
            return (
                '[CHAIN_REQUIRED] No real tool or agent handoff may execute before '
                'the current session has an approved structured DAG. Use '
                'get_tool_schema as needed, then task_line_query -> '
                'propose_workflow_patch with concrete agent/tool/arguments/'
                'depends_on/expected_outputs. A one-tool task still needs one node; '
                'plain conversation needs no DAG.'
            )
        if tool.startswith('handoff_to_'):
            return (
                '[WORKFLOW_BLOCK] The approved DAG owns delegation. Start or resume '
                'it with execute_workflow; do not create a parallel legacy handoff.'
            )
        return ''

    def _claim_submission(self, tool, params):
        if not is_submission(tool, params):
            return None, None, ''
        chain_block = self._chain_required_block(tool)
        if chain_block:
            return None, None, chain_block
        if self._runtime_snapshot and self._runtime_snapshot().get('status') in {'active', 'needs_user'}:
            return None, None, '[WORKFLOW_BLOCK] compiled runtime owns dispatch; main chat cannot duplicate its worker submissions'
        if self._on_workflow_start and self.goal_contract.approved_nodes:
            return None, None, '[WORKFLOW_BLOCK] approved DAG must execute through execute_workflow; activation failure is not permission for legacy serial submissions'
        if self._pending_workflow_patch:
            return None, None, '[RETRY_BLOCK] workflow patch is still under user negotiation; do not dispatch its new nodes'
        if self.goal_contract.requires_plan_approval and not self.goal_contract.approved_nodes:
            return None, None, '[WORKFLOW_BLOCK] compile concrete agent/tool/arguments/depends_on nodes with propose_workflow_patch before dispatching a multi-step goal'
        self.recovery_gate.sync_jobs(self._pending_jobs_for_conv())
        entries = self.recovery_gate.snapshot()
        if any(e.get('status') in {'reserved', 'uncertain'} for e in entries.values()):
            return None, None, '[RETRY_BLOCK] an earlier dispatch still has an unknown outcome; reconcile it before any new submission'
        if tool == 'run_cdft' and params.get('action') == 'submit' and any(
            e.get('goal_version') == self.goal_contract.version and e.get('tool') == 'run_cdft'
            and e.get('params', {}).get('action') == 'pipeline'
            and e.get('status') in {'failed', 'submitted'} for e in entries.values()
        ):
            return None, None, '[RETRY_BLOCK] do not bypass an unresolved cDFT pipeline by switching to action=submit; review the original pipeline'
        # Raw submission is not an escape hatch around a failing scientific tool.
        if tool == 'submit_job' and any(
            e.get('goal_version') == self.goal_contract.version
            and e.get('tool') != 'submit_job'
            and e.get('status') in {'failed', 'uncertain', 'reserved', 'submitted'}
            for e in entries.values()
        ):
            return None, None, '[RETRY_BLOCK] use the original scientific tool; raw submit_job cannot bypass its recovery gate'
        key = self.recovery_gate.key(tool, params, self.goal_contract.version,
                                     node=self._workflow_node_for_call(tool, params).get('step_id', ''), project_root=self.config.project_root)
        fingerprint = input_fingerprint(params, self.config.project_root)
        if key not in entries:
            # Adopt known legacy jobs instead of duplicating them on the first
            # call after upgrading. Match actual target metadata, not job name.
            def _same_path(a, b):
                if not a or not b:
                    return False
                root = self.config.project_root
                pa, pb = Path(a), Path(b)
                return (pa if pa.is_absolute() else root / pa).resolve() == (pb if pb.is_absolute() else root / pb).resolve()
            from pathlib import Path
            legacy = [j for j in self._pending_jobs_for_conv()
                      if j.get('tool') == tool and (j.get('failed') or not j.get('terminal'))
                      and not (j.get('failed') and any(str(j.get('job_id')) in e.get('job_ids', []) for e in entries.values()))
                      and (_same_path(j.get('work_dir'), params.get('work_dir'))
                           or j.get('gas') == params.get('gas') and _same_path(j.get('cif'), params.get('cif') or params.get('cif_path')))]
            if legacy:
                job = max(legacy, key=lambda j: str(j.get('submitted_at', '')))
                adopted, _ = self.recovery_gate.claim(key, tool, params, fingerprint, self.goal_contract.version, self.config.project_root, legacy_job_id=job['job_id'])
                if adopted:
                    self.recovery_gate.outcome(key, adopted, {'submitted': True, 'job_id': job['job_id']})
                    self.recovery_gate.sync_jobs(legacy)
        attempt, reason = self.recovery_gate.claim(key, tool, params, fingerprint,
                                                  goal_version=self.goal_contract.version, project_root=self.config.project_root)
        return key, attempt, reason

    def _recovery_state(self):
        self.recovery_gate.sync_jobs(self._pending_jobs_for_conv())
        evidence = [
            {'call_id': c.get('call_id'), 'time': c.get('time'), 'tool': c.get('tool'),
             'params': c.get('params'), 'result_preview': c.get('result_preview'), 'evidence_path': c.get('evidence_path')}
            for c in self.memory.tool_call_log[-50:]
            if c.get('tool') in {'diagnose_job', 'read_file', 'inspect_path', 'run_bash', 'validate_method', 'validate_gcmc_results'}
        ]
        evidence.extend(self.context.get('main_evidence_calls', []))
        attempts = self.recovery_gate.snapshot()
        candidates = {}
        for key, entry in attempts.items():
            cutoff = entry.get('failed_at', entry.get('started_at', 0))
            fresh = [c for c in evidence if c.get('call_id') and c.get('time', 0) >= cutoff]
            candidates[key] = {'diagnostic_call_ids': [c['call_id'] for c in fresh],
                'required_order': 'failed_at <= diagnosis.time < verification.time; two DIFFERENT actual call_ids',
                'actual_job_ids': entry.get('job_ids', []), 'failed_arguments': entry.get('params', {}),
                'actual_result': entry.get('last_result', {}),
                'next_action': 'diagnose_and_verify_repair' if entry.get('status') == 'failed' else
                               'collect_and_validate_actual_job_outputs' if entry.get('status') == 'completed_unverified' else 'reconcile_existing_dispatch'}
        return {'attempts': attempts, 'evidence_calls': evidence, 'recovery_guidance': candidates,
                'note': 'Use evidence_ref.call_id from real tool replies, NEVER recovery_key/tool_use_id/invented strings. A successful cDFT collect of the SAME job is output verification. Do not use the same call as diagnosis and verification. submitted/UNKNOWN are not success.'}

    def _prepare_retry(self, recovery_key: str, tool_name: str, arguments: Dict,
                       diagnosis: str, diagnosis_call_id: str, fix_summary: str,
                       verification_call_id: str):
        """Grant a single-use permit only from real, fresh tool evidence."""
        self.recovery_gate.sync_jobs(self._pending_jobs_for_conv())
        entry = self.recovery_gate.snapshot().get(recovery_key)
        if not entry or entry.get('tool') != tool_name:
            raise ValueError('unknown recovery key/tool; read recovery_state first')
        if len(diagnosis.strip()) < 10 or len(fix_summary.strip()) < 10:
            raise ValueError('root cause and concrete fix summary are required')
        calls = {c.get('call_id'): c for c in self.memory.tool_call_log if c.get('call_id')}
        diag = self._load_evidence_call(calls.get(diagnosis_call_id))
        verify = self._load_evidence_call(calls.get(verification_call_id))
        evidence_tools = {'read_file', 'inspect_path', 'run_bash', 'validate_method', 'validate_gcmc_results'}
        if not diag or diag.get('tool') not in evidence_tools | {'diagnose_job'}:
            raise ValueError('diagnosis must reference an actual diagnostic/inspection tool call')
        failed_at = entry.get('failed_at', entry['started_at'])
        if diag.get('time', 0) < failed_at:
            raise ValueError('diagnosis predates the failure; acquire fresh logs')
        dobj = result_object(diag.get('result', ''))
        if entry.get('job_ids'):
            if (diag.get('tool') != 'diagnose_job'
                    or str(diag.get('params', {}).get('job_id')) not in entry['job_ids']
                    or dobj.get('terminal') is not True or dobj.get('failed') is not True
                    or not any(dobj.get(k) for k in ('error', 'stderr_tail', 'run_log_tail', 'diagnosis'))):
                raise ValueError('diagnose the failed job ID with terminal failure and real error evidence; UNKNOWN is insufficient')
        elif failure_reason(diag.get('result', '')) or not dobj:
            raise ValueError('diagnostic tool failed or returned no structured evidence')
        if (not verify or verify.get('tool') not in evidence_tools
                or verify.get('time', 0) <= diag.get('time', 0)
                or failure_reason(verify.get('result', ''))):
            raise ValueError('a successful verification call after diagnosis is required')
        vobj = result_object(verify.get('result', ''))
        if not vobj or any(vobj.get(k) is False for k in ('valid', 'passed', 'validation_passed', 'ok')):
            raise ValueError('verification did not pass')
        if verify.get('tool') == 'run_bash':
            command = str(verify.get('params', {}).get('command', '')).strip()
            if vobj.get('exit_code') != 0 or re.match(r'^(echo|printf|true)\b', command):
                raise ValueError('verification must run a real check, not echo a claim')
        # An unrelated successful command is not proof that these inputs were fixed.
        if verify.get('tool') in {'read_file', 'inspect_path', 'run_bash'}:
            verification_text = ' '.join(str(v) for v in verify.get('params', {}).values())
            targets = [str(arguments[k]) for k in ('cif', 'cif_path', 'cif_dir', 'input_path', 'input_dir', 'data_csv', 'work_dir')
                       if arguments.get(k)]
            if targets and not any(t in verification_text for t in targets):
                raise ValueError('verification must reference the corrected input or work directory')
        issues = self.registry.validate_params(tool_name, arguments)
        allowed, reason = self.goal_contract.guard_tool_call(tool_name, arguments)
        expected_key = self.recovery_gate.key(tool_name, arguments, self.goal_contract.version,
                                              node=self._workflow_node_for_call(tool_name, arguments).get('step_id', ''), project_root=self.config.project_root)
        if issues or not allowed or expected_key != recovery_key:
            raise ValueError(f'retry must keep the approved task identity and valid scientific parameters: {issues or reason}')
        fingerprint = input_fingerprint(arguments, self.config.project_root)
        review = {'diagnosis': diagnosis, 'fix_summary': fix_summary,
                  'diagnosis_call_id': diagnosis_call_id, 'verification_call_id': verification_call_id}
        permit = self.recovery_gate.approve(recovery_key, fingerprint, review)
        self.memory.record_step('recovery_review', 'done', json.dumps(review, ensure_ascii=False),
                                agent_name=self.current_agent.name if self.current_agent else '')
        return permit

    def _accept_recovered_result(self, recovery_key: str, verification_call_id: str, conclusion: str):
        self.recovery_gate.sync_jobs(self._pending_jobs_for_conv())
        entry = self.recovery_gate.snapshot().get(recovery_key)
        call = self._load_evidence_call(next((c for c in self.memory.tool_call_log if c.get('call_id') == verification_call_id), None))
        if (not entry or not call or call.get('time', 0) <= entry['started_at']
                or call.get('tool') not in {'run_bash', 'validate_gcmc_results', 'run_cdft', 'read_file'}
                or failure_reason(call.get('result', '')) or len(conclusion.strip()) < 10):
            raise ValueError('fresh, successful output validation and a factual conclusion are required')
        obj = result_object(call.get('result', ''))
        if call['tool'] == 'validate_gcmc_results' and obj.get('validation_passed') is not True:
            raise ValueError('GCMC result validation did not pass')
        if call['tool'] in {'run_cdft', 'read_file'}:
            self._validate_cdft_output_evidence(entry, call, obj)
        if call['tool'] == 'run_bash':
            command = str(call.get('params', {}).get('command', '')).strip()
            targets = [str(entry.get('last_result', {}).get(k) or '') for k in ('work_dir', 'output_dir', 'output_csv')]
            if (obj.get('exit_code') != 0 or re.match(r'^(echo|printf|true)\b', command)
                    or not any(t and t in command for t in targets)):
                raise ValueError('output validation must check the actual result directory, not echo a claim')
        receipt = self.recovery_gate.accept_result(recovery_key, {
            'verification_call_id': verification_call_id, 'conclusion': conclusion,
        })
        self._emit_lifecycle_event('result_verified', {'recovery_key': recovery_key, 'receipt': receipt, 'main_already_handling': True})
        for bid, branch in self._error_branches.items():
            if branch.get('recovery_key') == recovery_key and branch.get('status') != 'resolved':
                self._update_error_branch(bid, resolution=f'Confirmed outputs: {conclusion}; evidence={verification_call_id}')
        return receipt

    def _validate_cdft_output_evidence(self, entry, call, obj):
        """Link collection/read evidence to the actual job, not just any CSV."""
        import csv
        import io
        import math
        from pathlib import Path
        if entry.get('tool') != 'run_cdft':
            raise ValueError('read_file only verifies linked cDFT CSV results; use the scientific validation tool for other calculations')
        def path(value):
            p = Path(value)
            return (p if p.is_absolute() else self.config.project_root / p).resolve()
        job_dir = entry.get('last_result', {}).get('work_dir')
        if not job_dir:
            raise ValueError('cDFT validation requires the actual scheduler work_dir receipt')
        if call['tool'] == 'run_cdft':
            if call.get('params', {}).get('action') != 'collect':
                raise ValueError('cDFT submission cannot serve as output verification; use collect')
            if path(call.get('params', {}).get('job_work_dir') or '') != path(job_dir):
                raise ValueError('cDFT collect must reference this exact job work_dir')
            rows = obj.get('rows')
        else:
            if obj.get('truncated') or obj.get('offset', 0) != 0:
                raise ValueError('output verification needs a complete CSV read, not a partial snippet')
            target = path(call.get('params', {}).get('path') or '')
            linked = target == path(job_dir) / 'results.csv'
            for previous in self.memory.tool_call_log:
                if previous.get('tool') != 'run_cdft' or previous.get('params', {}).get('action') != 'collect':
                    continue
                saved = self._load_evidence_call(previous)
                result = result_object(saved.get('result', ''))
                if (previous.get('time', 0) > entry['started_at'] and previous.get('time', 0) < call['time']
                        and path(previous['params'].get('job_work_dir') or '') == path(job_dir)
                        and result.get('output_csv') and path(result['output_csv']) == target
                        and not failure_reason(saved.get('result', ''))):
                    linked = True
            if not linked:
                raise ValueError('CSV read is not linked to the actual job or its fresh collection receipt')
            rows = list(csv.DictReader(io.StringIO(obj.get('content', ''))))
            if obj.get('total_lines') and len(rows) + 1 < obj['total_lines']:
                raise ValueError('output verification needs all CSV rows')
        args = entry.get('params', {})
        gases = args.get('gases') or ([args['gas']] if args.get('gas') else [])
        if not rows or not gases:
            raise ValueError('cDFT verification requires nonempty material rows and the submitted gas scope')
        names = []
        for row in rows:
            names.append(row.get('MOF'))
            for gas in gases:
                value = row.get(gas + '_henry_mol_L_atm')
                try:
                    valid = not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0
                except (TypeError, ValueError):
                    valid = False
                if not valid:
                    raise ValueError('cDFT CSV has missing/nonfinite/negative Henry values for the submitted gas scope')
        if not all(names) or len(set(names)) != len(names):
            raise ValueError('cDFT CSV contains missing/duplicate material identities')
        expected_count = entry.get('last_result', {}).get('n_mofs')
        if expected_count and len(rows) != expected_count:
            raise ValueError('cDFT CSV does not cover every submitted material')

    def _load_evidence_call(self, call):
        if not call or call.get('result') is not None:
            return call
        if not call.get('evidence_path') or not self._evidence_root:
            return call
        from pathlib import Path
        path = Path(call['evidence_path']).resolve()
        if not path.is_relative_to(Path(self._evidence_root).resolve()):
            raise ValueError('evidence reference escaped the conversation workspace')
        saved = json.loads(path.read_text())
        if saved.get('call_id') != call.get('call_id'):
            raise ValueError('evidence call ID mismatch')
        return {**call, **saved}

    # ── Plan → TaskLine Auto-Conversion ─────────────────────────────────
    # 当orchestrator输出方案时，自动解析步骤并创建TaskLine流水线

    def _parse_plan_to_taskline(self, plan_text: str, line_id: str = "") -> List[Dict[str, Any]]:
        """解析orchestrator的方案文本，提取步骤并创建TaskLine。

        方案格式通常为：
        1. 步骤1描述（如：查找CIF文件）
        2. 步骤2描述（如：运行PACMAN赋电荷）
        3. 步骤3描述（如：运行GCMC等温线计算）

        返回解析出的步骤列表。
        """
        steps = []
        plan_version = self.goal_contract.pending_plan_version or 1
        # 匹配数字编号的步骤
        step_patterns = [
            r'(?:^|\n)\s*(\d+)[\.、)）]\s*(.+?)(?=\n\s*\d+[\.、)）]|$)',
            r'(?:^|\n)\s*[-•]\s*(.+?)(?=\n\s*[-•]|$)',
            r'(?:^|\n)\s*Step\s*(\d+)[：:]\s*(.+?)(?=\n\s*Step\s*\d+|$)',
        ]

        for pattern in step_patterns:
            matches = re.findall(pattern, plan_text, re.MULTILINE | re.DOTALL)
            if matches:
                for i, match in enumerate(matches):
                    if isinstance(match, tuple):
                        step_id = match[0].strip()
                        step_desc = match[1].strip()
                    else:
                        step_id = str(i + 1)
                        step_desc = match.strip()

                    # 识别步骤类型
                    step_type = self._classify_step_type(step_desc)
                    steps.append({
                        "step_id": f"v{plan_version}_step_{step_id}",
                        "description": step_desc[:200],
                        "type": step_type,
                        "tool_hint": self._suggest_tool_for_step(step_desc),
                        "status": "pending",
                    })
                break

        # 创建TaskLine
        if steps and line_id:
            try:
                from .task_line import get_store
                from .watch_context import get_context
                store = get_store()
                ctx = get_context()
                store.begin_line(
                    line_id,
                    username=ctx.get("username", ""),
                    conv_id=ctx.get("conv_id", ""),
                    title=f"Workflow plan v{self.goal_contract.pending_plan_version or 1} ({len(steps)} steps)",
                )
                store.supersede_plan_versions(line_id, plan_version)
                previous_id = ""
                for s in steps:
                    store.upsert_step(
                        line_id,
                        s["step_id"],
                        tool=s["tool_hint"],
                        status="pending",
                        agent=self._suggest_agent_for_step(s["type"]),
                        depends_on=[previous_id] if previous_id else [],
                        validation={"goal_contract": "pending", "arguments": "pending"},
                        plan_version=plan_version,
                        note=s["description"],
                        username=ctx.get("username", ""),
                        conv_id=ctx.get("conv_id", ""),
                    )
                    previous_id = s["step_id"]
                self._plan_steps_created = True
                print(f"  📋 TaskLine创建: {len(steps)} 步骤已解析 (line_id={line_id})", flush=True)
            except Exception as e:
                print(f"  ⚠️ TaskLine创建失败: {e}", flush=True)

        return steps

    @staticmethod
    def _suggest_agent_for_step(step_type: str) -> str:
        if step_type in {"gcmc", "henry", "pore_analysis"}:
            return "adsorption"
        if step_type in {"charge", "analysis", "validation"}:
            return "analyst"
        if step_type == "literature":
            return "communicator"
        if step_type == "find_cif":
            return "harness-maintainer"
        return "lead-orchestrator"

    def _classify_step_type(self, description: str) -> str:
        """根据描述文本分类步骤类型。"""
        desc_lower = description.lower()
        if any(k in desc_lower for k in ["查找cif", "find_cif", "搜索材料", "查找材料"]):
            return "find_cif"
        elif any(k in desc_lower for k in ["赋电荷", "pacman", "charge", "电荷计算"]):
            return "charge"
        elif any(k in desc_lower for k in ["gcmc", "等温线", "isotherm", "吸附计算"]):
            return "gcmc"
        elif any(k in desc_lower for k in ["henry", "亨利"]):
            return "henry"
        elif any(k in desc_lower for k in ["孔隙分析", "pore", "zeo++"]):
            return "pore_analysis"
        elif any(k in desc_lower for k in ["文献", "literature", "查询"]):
            return "literature"
        elif any(k in desc_lower for k in ["分析", "analysis", "结果", "报告"]):
            return "analysis"
        elif any(k in desc_lower for k in ["验证", "validate", "检查"]):
            return "validation"
        else:
            return "other"

    def _suggest_tool_for_step(self, description: str) -> str:
        """根据步骤描述建议使用的工具。"""
        step_type = self._classify_step_type(description)
        tool_map = {
            "find_cif": "find_cif",
            "charge": "run_pacman_charge",
            "gcmc": "run_gcmc_isotherm",
            "henry": "run_henry",
            "pore_analysis": "run_pore_analysis",
            "literature": "query_literature",
            "analysis": "generate_scientific_report",
            "validation": "validate_gcmc_results",
        }
        return tool_map.get(step_type, "run_bash")

    def _delegation_fingerprint(self, name, arguments):
        from .defns import resolve_agent
        target = resolve_agent(name.removeprefix('handoff_to_')).name
        from .reflection_progress import digest
        evidence = next((digest([c.get('tool'), c.get('params'), self._load_evidence_call(c).get('result') or c.get('result_preview')])
                         for c in reversed(self.memory.tool_call_log)
                         if not c.get('failed') and not c.get('tool', '').startswith('handoff_to_')), None)
        payload = [target, arguments, self.goal_contract.version,
                   self.goal_contract.approved_plan_version, evidence]
        return 'delegation:v2:' + hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    def _workflow_node_for_call(self, tool_name, arguments):
        if not self._current_line_id:
            return {}
        from .task_line import get_store
        line = self._workflow_view()
        matches = [s for s in line.get('steps', []) if s.get('tool') == tool_name
                   and s.get('status') != 'superseded' and not s.get('branch_parent')]
        exact = [s for s in matches if s.get('arguments') == arguments]
        candidates = exact or [s for s in matches if not s.get('arguments')]
        active = [s for s in candidates if not s.get('done') and s.get('status') != 'completed']
        by_id = {s.get('step_id'): s for s in line.get('steps', [])}
        ready = [s for s in active if all(by_id.get(dep, {}).get('contract_matches_execution') is not False and
                 (by_id.get(dep, {}).get('done') or by_id.get(dep, {}).get('status') == 'completed') for dep in s.get('depends_on', []))]
        eligible = ready or active or candidates
        if len(eligible) > 1:
            raise ValueError(f'[AMBIGUOUS_NODE] {tool_name} arguments match multiple workflow nodes: '
                             + ', '.join(s['step_id'] for s in eligible))
        return eligible[0] if eligible else {}

    def _check_taskline_dependencies(self, tool_name: str, arguments=None) -> tuple[bool, str]:
        """Enforce declared serial/DAG dependencies before tool execution."""
        if not self._current_line_id or tool_name.startswith("handoff_to_"):
            return True, ""
        try:
            from .task_line import get_store
            line = self._workflow_view()
            if not line:
                return True, ""
            steps = line.get("steps", [])
            target = self._workflow_node_for_call(tool_name, arguments or {})
            if not target:
                return True, ""
            by_id = {s.get("step_id"): s for s in steps}
            blocked = []
            for dep in target.get("depends_on", []) or []:
                state = by_id.get(dep, {})
                if state.get('contract_matches_execution') is False or not state.get("done") and state.get("status") != "completed":
                    blocked.append(f"{dep}:{state.get('status', 'missing')}")
            if blocked:
                return False, (
                    f"[DEPENDENCY_BLOCK] {target.get('step_id')} cannot run {tool_name}; "
                    f"unfinished dependencies: {', '.join(blocked)}"
                )
            return True, ""
        except Exception as error:
            return False, f'[DEPENDENCY_BLOCK] workflow state could not be verified: {error}'

    # ── Error Branch Management ─────────────────────────────────────────
    # 当步骤失败时，创建独立的错误处理分支

    def _create_error_branch(self, failed_step: str, error: str, diagnosis: str = "",
                             category: str = "unknown") -> str:
        """为失败的步骤创建错误处理分支线。

        返回分支ID。
        """
        self._error_branch_counter += 1
        branch_id = f"error_branch_{self._error_branch_counter}"

        branch = {
            "branch_id": branch_id,
            "created_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
            "failed_step": failed_step,
            "error": error[:500],
            "diagnosis": diagnosis[:500],
            "category": category,
            "status": "open",
            "fix_attempts": [],
            "resolution": None,
        }

        self._error_branches[branch_id] = branch

        # 同时记录到TaskLine
        if self._current_line_id:
            try:
                from .task_line import get_store
                store = get_store()
                store.upsert_step(
                    self._current_line_id,
                    f"error_{branch_id}",
                    status="failed",
                    branch_parent=failed_step,
                    validation={"error": error[:500], "diagnosis": diagnosis[:500]},
                    note=f"[错误分支] {failed_step}: {error[:100]}",
                )
            except Exception:
                pass

        print(f"  🔀 错误分支创建: {branch_id} (失败步骤: {failed_step})", flush=True)
        return branch_id

    @staticmethod
    def _classify_failure_scope(error: str) -> str:
        """Separate framework defects from task/input/external failures."""
        low = str(error or "").lower()
        if any(k in low for k in (
            "schema validation", "unexpected keyword", "attributeerror", "traceback",
            "tool not available", "input generation bug", "goal_contract",
        )):
            return "framework"
        if any(k in low for k in (
            "permission", "auth", "license", "connection", "network", "dns",
            "command not found", "no such file", "glibc", "module not found",
        )):
            return "external_environment"
        if any(k in low for k in (
            "out of memory", "oom", "node_fail", "timeout", "cancelled", "resource",
        )):
            return "infrastructure"
        if any(k in low for k in (
            "invalid", "missing", "cif", "force field", "forcefield", "parameter",
            "convergence", "nan", "zero loading",
        )):
            return "task_or_scientific_input"
        return "unknown"

    def _update_error_branch(self, branch_id: str, fix_attempt: str = "",
                              resolution: str = "") -> None:
        """更新错误分支的状态。"""
        if branch_id not in self._error_branches:
            return

        branch = self._error_branches[branch_id]
        if fix_attempt:
            branch["fix_attempts"].append({
                "time": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
                "attempt": fix_attempt[:300],
            })
        if resolution:
            branch["resolution"] = resolution[:300]
            branch["status"] = "resolved"
            print(f"  ✅ 错误分支已解决: {branch_id}", flush=True)
            if self._current_line_id:
                try:
                    from .task_line import get_store
                    get_store().upsert_step(
                        self._current_line_id,
                        f"error_{branch_id}",
                        status="completed",
                        done=True,
                        note=f"[错误分支已解决] {resolution[:180]}",
                    )
                except Exception:
                    pass

    def _resolve_error_branches_for_tool(self, tool_name: str, result: str, arguments=None) -> None:
        """Close prior branches when the same tool later succeeds."""
        obj = result_object(result)
        # sbatch accepting a retry is NOT proof that its calculation succeeded.
        if obj.get('submitted') or obj.get('job_id') or obj.get('job_ids') or obj.get('chain_status') == 'submitted':
            return
        if is_submission(tool_name, obj):
            return
        failed = bool(failure_reason(result))
        if failed:
            return
        for branch_id, branch in list(self._error_branches.items()):
            if branch.get("status") == "open" and branch.get("failed_step") == tool_name:
                if (branch.get('category') != 'contract_validation'
                        and 'failed_arguments' in branch and branch['failed_arguments'] != arguments):
                    continue
                self._update_error_branch(
                    branch_id,
                    fix_attempt=f"{tool_name} retry returned without a failure marker",
                    resolution=f"{tool_name} succeeded after recovery",
                )

    def _get_error_branch_summary(self) -> str:
        """获取所有错误分支的摘要。"""
        if not self._error_branches:
            return ""

        lines = ["=== 错误分支摘要 ==="]
        for bid, branch in self._error_branches.items():
            status_icon = "✅" if branch["status"] == "resolved" else "❌"
            lines.append(f"{status_icon} {bid}: {branch['failed_step']}")
            lines.append(f"   错误: {branch['error'][:100]}")
            if branch["diagnosis"]:
                lines.append(f"   诊断: {branch['diagnosis'][:100]}")
            if branch["resolution"]:
                lines.append(f"   解决: {branch['resolution'][:100]}")
            lines.append(f"   修复尝试: {len(branch['fix_attempts'])} 次")

        return "\n".join(lines)

    # ── Fault Handling ──────────────────────────────────────────────────
    # The computation tools submit SLURM jobs. Jobs routinely FAIL (cancelled,
    # timeout, RASPA crash, node down, bad params). Without explicit handling,
    # the agent would report success on a job that produced nothing. This
    # mechanism detects failures, tells the model to diagnose + recover, and
    # caps retries so the agent never loops forever.

    _FAILURE_MARKERS = (
        "CANCELLED", "FAILED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
        "REJECTED", "FATAL", "simulation failed", "no valid gcmc loading data",
        '"failed": true', '"submitted": false',
        "returned non-zero", "segmentation", "core dumped", "job not found",
        "does not exist", "command not found",
    )

    # Diagnostic/query tools whose normal output CONTAINS '"failed": true'
    # (the *job* failed, but the tool call itself succeeded). Marking those
    # results as "tool failures" would make the agent loop: diagnose_job says
    # failed → we inject a recovery message → agent diagnoses again → failed
    # again … ad infinitum. For these tools the agent must interpret the
    # failure itself (e.g. via diagnose_job) instead of us flagging the call.
    _DIAGNOSTIC_TOOLS = {
        "check_job", "check_job_status", "diagnose_job", "inspect_run",
        "inspect", "get_job_status", "list_jobs",
    }

    def _detect_tool_failures(self, tool_results, assistant_content) -> list:
        """Scan a batch of tool_results for job failures.

        Returns a list of (tool_name, error_snippet) for every failed tool call.
        Uses both SLURM state markers and our explicit failure keywords.
        Diagnostic tools (check_job/diagnose_job/…) are excluded: their
        '"failed": true' output describes the JOB, not the tool call.
        Handoff/delegation tools (handoff_to_*) are excluded too: their brief
        and receipt are prose that quotes failure words as context — not logs.
        """
        # Map tool_use_id → tool name from the assistant message
        id_to_name: Dict[str, str] = {}
        for b in (assistant_content or []):
            btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
            if btype == "tool_use":
                bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", None)
                bname = b.get("name") if isinstance(b, dict) else getattr(b, "name", "tool")
                if bid:
                    id_to_name[bid] = bname

        failures = []
        for tr in tool_results:
            name = id_to_name.get(tr.get("tool_use_id"))
            # If the call cannot be attributed, do not invent a tool failure.
            # Query results may describe failed JOBS or contain arbitrary files.
            if not name or name in self._DIAGNOSTIC_TOOLS or name.startswith('handoff_to_'):
                continue
            reason = failure_reason(tr.get('content', ''))
            if reason:
                failures.append((name, reason))
        return failures

    def _pending_jobs_for_conv(self) -> List[Dict[str, Any]]:
        """Jobs auto-registered by THIS conversation (JobWatch records conv_id).

        Every compute tool executor registers its job_id with JobWatch under the
        current thread-local conv_id, so before producing a final report we can
        enumerate exactly which SLURM jobs this task spawned.
        """
        try:
            from .watch_context import get_context
            from .job_watch import get_watch
            ctx = get_context()
            conv = ctx.get("conv_id", "")
            if not conv:
                return []
            return [j for j in get_watch().list() if j.get("conv_id") == conv
                    and (not ctx.get('username') or j.get('username') == ctx['username'])]
        except Exception:
            return []

    def _has_verified_runtime_computation(self):
        state = self._runtime_snapshot() if self._runtime_snapshot else {}
        if state.get('status') != 'completed' or not state.get('nodes'): return False
        ledger = self.recovery_gate.snapshot()
        jobs = {str(job['job_id']): job for job in self._pending_jobs_for_conv() if job.get('job_id')}
        computed = False
        for node in state['nodes'].values():
            if node.get('status') != 'succeeded' or not node.get('result_ref', {}).get('call_id'): return False
            contract = node.get('contract', {})
            if not is_submission(contract.get('tool'), contract.get('arguments', {})): continue
            entry = ledger.get(node.get('recovery_key'), {})
            ids = entry.get('job_ids', [])
            if (entry.get('status') != 'completed' or not entry.get('result_verification') or not node.get('artifacts')
                    or not ids or any(jobs.get(str(jid), {}).get('terminal') is not True or jobs[str(jid)].get('failed') for jid in ids)):
                return False
            computed = True
        return computed

    @staticmethod
    def _claims_execution_result(text):
        """Only positive execution assertions, not domain nouns or safe refusals.

        This is a conservative fabrication signal, not scientific validation;
        scheduler/output/structured-plan gates remain authoritative.
        """
        pattern = r'提交成功|成功提交|(?:计算|运行|仿真)(?:已)?完成|已完成(?:计算|仿真|运行)|\bsubmitted\b|\bjob_id\s*[:=：]\s*[`\"\']?\d+'
        for match in re.finditer(pattern, text, re.IGNORECASE):
            prefix = re.split(r'[。；;\n]', text[:match.start()])[-1][-32:]
            if re.search(r'不能|不应|不可|不要|未能|尚未|没有|并未|未|如果|假如|例如|示例|not\s|never\s|without\s|if\s', prefix, re.IGNORECASE):
                continue
            return True
        return False

    def _gate_jobs_before_report(self, jobs: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Report gate: poll every SLURM job spawned by this conversation.

        Returns:
            {"ok": True}                              → all jobs COMPLETED normally.
            {"ok": False, "reason": "running", ...}   → some jobs still executing.
            {"ok": False, "reason": "failed", ...}    → some jobs entered a failure state.
        The caller must NOT let the model output a final report while ok=False —
        instead inject the returned message and continue the loop.
        """
        from . import slurm as _slurm
        jobs = jobs if jobs is not None else self._pending_jobs_for_conv()
        if not jobs:
            return {"ok": True, "jobs": []}
        running: List[str] = []
        failed: List[str] = []
        completed: List[str] = []
        resolved: List[str] = []
        for j in jobs:
            jid = j.get("job_id")
            if not jid:
                continue
            tool = j.get("tool", "?")
            # ① 优先用 JobWatch 已定格的终态（terminal=True）。终态在作业结束那
            #    一刻就持久化落盘，时间再久也不会失效——时间久的作业 SLURM 实时
            #    查询（squeue/sacct/scontrol）必然查不到（本机 sacct 还 disabled），
            #    所以绝不能靠实时命令去"确认"老作业，而应信任这份定格记录。
            if j.get("terminal") is True:
                if j.get("failed"):
                    # ①a 已正式诊断（diagnose_job）的定格失败不算"未处理失败"：
                    #    agent 已分析根因并会向用户报告结论，放行终报。否则该
                    #    conv 被永久打回——正是 3790 反复输出失败信号的根因。
                    if j.get("diagnosed"):
                        resolved.append(f"{jid} [{tool}] → {j.get('state', 'FAILED')}（已诊断·不阻塞终报）")
                    else:
                        failed.append(f"{jid} [{tool}] → {j.get('state', 'FAILED')}（定格·未诊断）")
                else:
                    if str(j.get('state', '')).upper() in {'COMPLETED', 'COMPLETE', 'DONE', 'SUCCESS'}:
                        completed.append(f"{jid} [{tool}] → {j.get('state')}（定格）")
                    else:
                        running.append(f"{jid} [{tool}] → {j.get('state', 'UNKNOWN')}（终态未确认，需核对产物）")
                continue
            # ② 只有尚未定格（可能还在跑）的作业才实时查询——这些是近期作业，
            #    squeue 还能查到，实时查询才有效。
            try:
                st = _slurm.check_job_status(jid, work_dir=j.get("work_dir", ""))
            except Exception as e:
                running.append(f"{jid} (poll error: {e})")
                continue
            state = st.get("status", "UNKNOWN")
            if st.get("failed") or state in _slurm.FAILURE_STATES:
                failed.append(f"{jid} [{tool}] → {state}")
            elif st.get("terminal") and state in ("COMPLETED", ""):
                completed.append(f"{jid} [{tool}] → {state}")
            elif st.get("terminal") and state in ("", "UNKNOWN"):
                # ③ 实时查不到但 terminal=True，且 JobWatch 记录没有定格标记
                #    （j.get("terminal") 不是 True 也不是 False，即无该字段/异常）。
                #    此时它不可能还在跑，绝不能当 running 卡死终报；标记为已终态
                #    但需产物验证，交给 agent 用文件系统证据确认。
                #    注意：若 JobWatch 明确 terminal=False（上次还看到在跑），实时却
                #    查不到 UNKNOWN，这是矛盾信号——保守起见仍算 running，等 JobWatch
                #    下次 poll 定格，避免把可能失败的作业误放行。
                if j.get("terminal") is False:
                    running.append(f"{jid} [{tool}] → {state}（JobWatch 未定格，保守等待）")
                else:
                    completed.append(f"{jid} [{tool}] → {state}（已终态·请用工作目录产物验证）")
            else:
                running.append(f"{jid} [{tool}] → {state}")
        if running:
            return {
                "ok": False, "reason": "running",
                "running": running, "failed": failed, "completed": completed,
                "message": (
                    "[系统·作业门禁] 以下计算作业仍在运行或尚无可验证终态，不能输出成功终报：\n"
                    + "\n".join(f"  ⏳ {r}" for r in running[:10])
                    + (f"\n  （还有 {len(running)-10} 个…）" if len(running) > 10 else "")
                    + "\n请调用 check_job(job_id=..., work_dir=...) 逐一确认结果；"
                      "等所有作业 COMPLETED 并读取到真实结果后再输出报告。"
                      "禁止编造或提前总结未完成的计算。"
                ),
            }
        if failed:
            return {
                "ok": False, "reason": "failed",
                "running": running, "failed": failed, "completed": completed,
                "message": (
                    "[系统·作业门禁] 以下计算作业**执行失败**，最终报告不能声称成功：\n"
                    + "\n".join(f"  ❌ {f}" for f in failed[:10])
                    + "\n请调用 diagnose_job(job_id=..., work_dir=...) 分析原因并修复重提，"
                      "或如实向用户说明失败项；不要输出包含编造结果的成功报告。"
                ),
            }
        if resolved:
            # 所有定格失败项均已正式诊断、无 running、无未诊断失败 → 放行终报，
            # 已诊断失败项随返回带回，agent 如实向用户汇报诊断结论即可。
            return {"ok": True, "jobs": completed, "resolved_failed": resolved}
        return {"ok": True, "jobs": completed}

    def _handle_failures(self, tool_results, assistant_content, agent_name: str) -> int:
        """Post-batch fault handling.

        - Records failures in memory (tool_call_log + failure_history)
        - Emits frontend progress + partial results so the user SEES recovery
        - Appends a recovery instruction user-message for the model
        - Caps retries per failure signature (default 3)
        Returns the number of failure instructions injected.
        """
        failures = self._detect_tool_failures(tool_results, assistant_content)
        if not failures:
            return 0

        # Parallel calls of one tool may concern different files/nodes. Match
        # their persisted evidence IDs instead of assigning every error to the
        # last call and exhausting an unrelated retry budget.
        evidence = {c.get('call_id'): c for c in self.memory.tool_call_log}
        correlated = {}
        for reply in tool_results:
            obj = result_object(reply.get('content'))
            call = evidence.get(obj.get('evidence_ref', {}).get('call_id'))
            if call and (failure_reason(reply.get('content')) or obj.get('error')):
                correlated.setdefault(call['tool'], []).append(call)

        for fname, ferr in failures:
            # Error text and agent segment changes must never replenish retries.
            last_call = (correlated[fname].pop(0) if correlated.get(fname) else
                         next((c for c in reversed(self.memory.tool_call_log)
                               if c.get('tool') == fname and isinstance(c.get('params'), dict)), {}))
            contract_failure = any(marker in ferr.lower() for marker in (
                'parameter validation failed', 'schema validation',
                'not valid under any of the given schemas',
                'workflow contains unresolved path placeholders',
                'patch contains incomplete/unscientific tool arguments',
                'workflow contract', 'contract must include',
            ))
            if contract_failure:
                # Count one broken node contract across successive field edits;
                # hashing all arguments reset the counter every time the model
                # corrected one field and exposed the next schema error.
                contract_step = last_call.get('workflow_step_id') or fname
                key = f'contract:{self.goal_contract.version}:{fname}:{contract_step}'
            else:
                key = last_call.get('recovery_key') or self.recovery_gate.key(
                    fname, last_call.get('params', {}), self.goal_contract.version,
                    project_root=self.config.project_root,
                )
            failed_attempt = self.recovery_gate.snapshot().get(key)
            n = failed_attempt['attempts'] if failed_attempt and failed_attempt.get('status') == 'failed' else self._failure_retries.get(key, 0) + 1
            self._failure_retries[key] = n
            self.failure_history.append({
                "tool": fname, "error": ferr[:300],
                "retries": n, "round": len(self.messages),
            })
            # Record so the per-agent compact memory summarizes the failure
            self.memory.record_error(f"{fname} FAILED (attempt {n}): {ferr[:120]}", agent_name=agent_name)

            # RASPA-aware diagnosis: map the error text to a concrete cause + fixes
            raspa_block = ""
            try:
                from .raspa_errors import summarize_run_log
                ra = summarize_run_log(ferr)
                if ra["raspa_error"]:
                    fixes_txt = "\n".join(f"    - {f}" for f in ra["fixes"][:4])
                    raspa_block = (
                        f"⚠️ 根据 RASPA 错误签名判断的失败原因：\n"
                        f"    {ra['cause']}\n"
                        f"    可尝试的修复方案：\n{fixes_txt}\n"
                    )
            except Exception:
                pass

            # 递归 sbatch 调用检测
            recursive_block = ""
            if "sbatch submit.sh" in ferr or "sbatch ./submit.sh" in ferr:
                recursive_block = (
                    f"🚨 递归调用错误: 检测到 'sbatch submit.sh' 命令，这会导致无限递归提交。\n"
                    f"    原因: submit.sh 脚本中包含了调用自身的命令。\n"
                    f"    修复方案:\n"
                    f"    - 检查 submit.sh 文件，确保其中的命令是实际的模拟命令（如 run_sim simulation.input）\n"
                    f"    - 不要在 submit.sh 中使用 'sbatch submit.sh' 或类似递归调用\n"
                    f"    - 使用 run_henry 工具而不是手动创建 submit.sh\n"
                )

            # ── 创建错误分支线 ──
            _failure_category = 'contract_validation' if contract_failure else 'actual_tool_failure'
            _error_branch_id = self._create_error_branch(
                failed_step=fname,
                error=ferr,
                diagnosis=raspa_block + recursive_block,
                category=_failure_category,
            )
            self._error_branches[_error_branch_id].update(failed_arguments=copy.deepcopy(last_call.get('params', {})),
                failed_call_id=last_call.get('call_id'))
            if failed_attempt and failed_attempt.get('status') == 'failed':
                self._error_branches[_error_branch_id]['recovery_key'] = key
            self._emit_lifecycle_event('tool_failed', {
                'tool': fname, 'error': ferr, 'category': _failure_category,
                'branch_id': _error_branch_id, 'requires_user': False, 'main_already_handling': True,
            })

            # Live feedback for the frontend
            self._progress(
                agent_name=agent_name,
                reasoning="",
                status=f"⚠️ 任务失败: {fname}",
                log=f"实际工具操作失败: {fname} — {ferr[:120]}",
            )
            self._report_partial(
                agent_name,
                f"⚠️ [故障处理] {fname} 失败 (第{n}次尝试)。正在诊断原因并恢复…\n"
                f"错误分支: {_error_branch_id}",
            )

            # 错误分支摘要
            _branch_summary = self._get_error_branch_summary()

            if contract_failure:
                recovery = (
                    f"[系统·参数合同修复] {fname} 在执行前被拒绝；没有提交作业，也没有产生科研失败。\n"
                    f"校验信息: {ferr[:1200]}\n\n"
                    "直接按 issues、missing_parameters 和 input_schema 修正节点。"
                    "可由 expected_outputs、depends_on 或会话目录唯一推导的路径字段必须自行补全。"
                    "只有缺少气体、温度、压力、方法等新的科研选择时才询问用户。"
                    "本错误不占用计算重试额度，不得因格式错误删掉交付物、改科学方案或宣称无法继续。"
                    + ("同一参数合同已多次修复未果：立即 handoff_to_patcher，附上完整 issues/input_schema 和失败参数，修复框架后继续原节点。"
                       if n >= 3 else "")
                )
                if n >= 3 and agent_name != 'patcher' and not self._pending_handoff:
                    from .defns import PATCHER
                    self._pending_handoff = PATCHER
                    self.context['automatic_handoff_task'] = (
                        '修复阻塞当前DAG的参数/schema/路径合同缺陷；不要运行用户科研任务。'
                        f'失败工具={fname}；节点={last_call.get("workflow_step_id") or "unknown"}；'
                        f'失败参数={json.dumps(last_call.get("params", {}), ensure_ascii=False, default=str)}；'
                        f'错误={ferr[:4000]}。修复框架并运行定向测试后交回主chat，从原DAG继续。'
                    )
            elif n >= 3:
                recovery = (
                    f"[系统·故障处理] 工具 {fname} 已连续失败 {n} 次，禁止再重试。\n"
                    f"最后一次失败信息: {ferr[:400]}\n\n"
                    f"错误分支: {_error_branch_id}\n"
                    f"{_branch_summary}\n\n"
                    "必须继续自主恢复：先基于真实诊断选择同一科研方法/条件下的修复或备选执行路径；"
                    "若属于工具/API/框架缺陷则立即委派patcher。只有缺少新的科研方法或条件时才能询问用户。"
                    "不要编造计算结果，也不要声称任务成功。"
                )
                self.context['reflection_required'] = {'tool': fname, 'actual_error': ferr,
                    'branch_id': _error_branch_id, 'attempts': n,
                    'instruction': 'Do not repeat an unchanged failed operation. Diagnose actual evidence and choose result restoration or a verified repair. A retry budget is not evidence that the user must change the scientific goal.'}
                if (self._classify_failure_scope(ferr) == 'framework'
                        and agent_name != 'patcher' and not self._pending_handoff):
                    from .defns import PATCHER
                    self._pending_handoff = PATCHER
                    self.context['automatic_handoff_task'] = (
                        '修复当前工具/API框架缺陷；不要执行用户科研计算。'
                        f'工具={fname}；错误={ferr[:4000]}；'
                        '完成代码修复和定向测试后交回主chat，从原DAG继续。'
                    )
            else:
                # Trust the model to reason from the raw error. The RASPA catalog
                # hint is optional; the error text itself is the real signal.
                # Tool errors are not necessarily failed scientific jobs. Share
                # facts, not an exhaustive reason classifier or private chain.
                recovery = (
                    f"[系统·故障处理] 实际工具操作失败：\n"
                    f"工具: {fname}\n"
                    f"失败信息: {ferr[:500]}\n\n"
                    f"错误分支: {_error_branch_id}\n"
                    f"{raspa_block}"
                    f"{recursive_block}"
                    "依据实际错误和当前任务判断原因，调用相关检查工具验证后选择恢复、修正操作或修补节点合同。\n"
                    "文件读取/接口错误不要默认当调度器或科学方法失败；没有关联作业就不要查无关job。\n"
                    "已有成功结果优先查证复用，不盲目重新提交，不擅自换力场、方法或关键条件。\n"
                    "真正计算重提仍须既有诊断、具体修正、验证和单次重提许可；新科学选择或外部权限才交用户协商。\n"
                    "给用户简短事实、已采取的修正和实际结果，不输出私有思维链或限定原因分类。"
                )
            self.messages.append({"role": "user", "content": recovery})
        return len(failures)

    # ── ReAct Discipline ───────────────────────────────────────────────
    # ReAct (Reasoning + Acting) is the base working loop of every agent:
    # think → act → observe → think again. The prompt-level rules in
    # _COMMON_RULES handle the "how". This runtime guard catches the failure
    # mode where the model calls tools back-to-back with ZERO reasoning text
    # (纯工具连点, no [思考]/[观察]) and injects a nudge to force it to stop
    # and reason before the next action.

    _REACT_SILENT_LIMIT = 3  # consecutive textless tool rounds before nudging

    # Cheap general-purpose tools that are EXEMPT from the aggressive
    # "same tool ≥2 calls → skip" anti-loop rule. Multi-step scientific
    # workflows (explore dirs → read inputs → write scripts → monitor jobs)
    # legitimately call these many times; the hard caps in `limits` and the
    # exact-duplicate-params skip still bound them.
    _GENERAL_LIMIT_EXEMPT = frozenset({
        "read_file", "write_file", "run_bash", "grep_search", "inspect_path",
    })

    def _count_text_blocks(self, assistant_content) -> int:
        """Number of reasoning-carrying blocks in an assistant message.

        Both visible 'text' (the [思考]/[观察] the ReAct prompt asks for) and the
        model's internal 'thinking' block count as reasoning — a round with a
        thinking block is NOT a silent tool chain and must not be nudged.
        """
        n = 0
        for b in (assistant_content or []):
            btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", "")
            if btype == "text":
                txt = (b.get("text") if isinstance(b, dict) else getattr(b, "text", "") or "") or ""
            elif btype == "thinking":
                # ThinkingBlock stores its content in .thinking, not .text
                txt = (b.get("thinking") if isinstance(b, dict) else getattr(b, "thinking", "") or "") or ""
            else:
                continue
            if txt.strip():
                n += 1
        return n

    def _react_nudge(self, assistant_content, agent_name: str) -> int:
        """ReAct loop guard: after tool_results are injected, if the model has
        been acting WITHOUT any reasoning text for N consecutive rounds, append
        a light [系统·ReAct] message demanding think-before-act (or a final answer).

        Returns the number of nudges injected (0 or 1).
        """
        self._silent_act_rounds = getattr(self, '_silent_act_rounds', 0)
        if self._count_text_blocks(assistant_content) > 0:
            # This round had visible reasoning → healthy ReAct, reset counter
            self._silent_act_rounds = 0
            return 0

        self._silent_act_rounds += 1
        if self._silent_act_rounds < self._REACT_SILENT_LIMIT:
            return 0

        n = self._silent_act_rounds
        self._silent_act_rounds = 0  # reset after nudging

        # Sanity: don't nudge when the batch contained handoffs — a sub-agent
        # chain legitimately calls tools across agents without new text.
        _had_handoff = False
        for b in (assistant_content or []):
            if (b.get("type") if isinstance(b, dict) else getattr(b, "type", "")) == "tool_use":
                bname = b.get("name") if isinstance(b, dict) else getattr(b, "name", "")
                if bname and bname.startswith("handoff_to_"):
                    _had_handoff = True
                    break
        if _had_handoff:
            self._silent_act_rounds = 0
            return 0

        self.messages.append({"role": "user", "content":
            f"[系统·ReAct] 你已经连续 {n} 轮只调用工具、没有输出任何思考文本，"
            "这是典型的'蒙头连点'，违反 ReAct 工作循环。\n"
            "在继续之前，必须先输出一段思考文本，格式如下（必须写）：\n"
            "  [思考] 根据已观察到的工具结果，现在离目标还缺什么？下一步为什么这样做？\n"
            "  [观察] 上一个工具结果说明了什么？\n"
            "如果已有信息已经足够回答用户，请直接输出最终报告，不要再调用任何工具。"
        })
        self._progress(
            agent_name=agent_name,
            reasoning="",
            status=f"🧠 ReAct 打断：连续 {n} 轮无思考工具调用，要求先思考",
            log=f"ReAct guard: {n} consecutive textless tool rounds → injected think-before-act nudge"
        )
        return 1

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token, conservative)."""
        return len(text) // 3

    def _detect_missing_params(self, msg: str) -> str:
        """Detect missing critical parameters in user message.
        Returns description of missing params, or empty string if sufficient.
        """
        msg_lower = msg.lower()
        # Strip filesystem paths before any substring matching: a CIF path like
        # "/tmp/cdft_10mof/MOF_0123_bex_pacman.cif" contains "10", "mc", "K"…
        # which would be misread as a pressure value / method / temperature and
        # make the gate skip asking — exactly the bug that let agents self-supply
        # 298K and submit. Match only the natural-language portion of the message.
        # A *.cif/*.xsf/*.json path itself IS strong material evidence (a concrete
        # structure file), so remember it and let has_material see it.
        _had_structure_file = bool(re.search(
            r"\S+\.(cif|xsf|json|pdb|xyz)\b", msg, re.IGNORECASE))
        _pathless = msg
        for _tok in msg.split():
            # 只剥真正的文件路径（以 / 开头、或以文件扩展名结尾）。
            # 注意 "CO2/N2"、"N2/CH4" 这类**混合气写法**不是路径——若仅因含 "/"
            # 就剥掉，has_gas 会误判气体缺失，硬门禁就会错误地要求补气体。
            if (_tok.startswith(("/", "./", "../"))
                    or _tok.lower().endswith((".cif", ".xsf", ".json", ".txt", ".out", ".err", ".pdb", ".xyz"))):
                _pathless = _pathless.replace(_tok, " ")
        msg = _pathless
        msg_lower = msg.lower()

        # Check if this is a calculation/analysis request
        action_keywords = ["计算", "筛选", "分析", "模拟", "优化", "训练", "预测",
                          "评估", "对比", "设计", "生成", "调研", "帮我", "做",
                          "设置", "查询", "多少", "怎么样", "如何", "怎样",
                          "是什么", "有哪些", "哪些", "求", "找", "研究", "跑",
                          "算", "算一下",
                          "calculate", "screen", "analyze", "investigate"]
        is_calculation = any(k in msg for k in action_keywords)
        if not is_calculation:
            return ""  # Not a calculation request, skip check

        # Detect what's present - use specific material names (not generic "MOF")
        # Completeness is evaluated against the durable structured contract,
        # not only the latest chat fragment.  A user may provide material,
        # gases and thermodynamic conditions over several turns.
        _contract = getattr(self, "goal_contract", None)
        _contract_params = getattr(_contract, "parameters", {}) or {}
        has_gas = bool(getattr(_contract, "gases", [])) or any(g in msg for g in
            ["CO2", "CH4", "N2", "H2", "H2O", "O2", "CO", "SO2", "NH3",
             "Xe", "Kr", "C2H4", "C2H6", "二氧化碳", "甲烷", "氮气", "氢气",
             "水蒸气", "乙烯", "乙烷"])
        # Specific materials (not just "MOF" which is too generic)
        specific_materials = ["MOF-5", "Ni-MOF-74", "Mg-MOF-74", "Co-MOF-74",
                              "CuBTC", "Cu-BTC", "HKUST", "MFI", "ZSM-5", "UiO",
                              "ZIF-8", "ZIF-90", "MIL-120", "MIL-101", "COF",
                              "Zeolite", "CALF-20", "NiMOF", "CuMOF", "FeMOF",
                              "HOF", "u-HOF", "COF", "u-COF"]
        has_specific_material = bool(_contract_params.get("material")) or any(
            m.lower() in msg_lower for m in specific_materials)
        # Generic material references (less reliable)
        has_generic_material = any(m in msg for m in ["数据库", "cif", "CIF",
                                                        "结构文件", "框架结构"])
        # A user-supplied structure file path (e.g. "xxx.cif") is the strongest
        # possible material evidence — treat it as material present.
        has_material = has_specific_material or has_generic_material or _had_structure_file

        has_temp = "temperature_K" in _contract_params or any(
            t in msg for t in ["298", "300", "273", "350", "400", "K"])
        # Pressure: match a pressure VALUE (digits before bar/Pa/atm) — a bare
        # "10" or "0.1" in "10个MOF"/"cif_0.1" is NOT pressure evidence.
        has_pressure = "pressure_range" in _contract_params or "pressure" in _contract_params or bool(re.search(
            r"\d+(\.\d+)?\s*(bar|Pa|atm|MPa|kPa)\b", msg, re.IGNORECASE))
        has_target = any(t in msg for t in ["选择性", "吸附量", "扩散系数", "结合能",
                                             "电荷", "Henry", "等温线", "能垒",
                                             "吸附热", "渗透性", "性能"])

        missing_parts = []

        # Method (GCMC vs cDFT) is a decision for the USER when not specified —
        # the agent should ASK instead of silently picking one.
        # "mc" is a common method abbreviation, but bare-substring matching makes
        # it false-positive on words like "pacman" — use a word-boundary regex.
        has_method = bool(getattr(_contract, "method", "")) or any(m in msg for m in
                         ["GCMC", "cDFT", "蒙特卡洛", "密度泛函", "经典密度泛函",
                          "monte carlo", "用gcmc", "用cdft"]) or bool(
            re.search(r"(^|[\s,，。])mc([\s,，。]|$)", msg, re.IGNORECASE)
        )
        # The GCMC-vs-cDFT question only applies to adsorption/screening-style
        # tasks. Diffusion (MD/TST), VASP/DFT, ML etc. have their own method
        # implied by the task itself — forcing a GCMC/cDFT choice there would be
        # a false positive that blocks valid tasks.
        _method_relevant_task = any(
            k in msg for k in ["吸附", "等温线", "筛选", "吸附量", "吸附热",
                               "loading", "adsorption", "选择性", "性能", "评估"]
        ) or (any(k in msg for k in ["计算", "帮我", "跑"]) and not any(
            k in msg for k in ["扩散", "MD", "TST", "分子动力学", "过渡态",
                               "VASP", "DFT", "电荷", "结合能", "能垒", "势场",
                               "训练", "预测", "特征"])
        )
        if is_calculation and not has_method and _method_relevant_task:
            missing_parts.append("计算方法(GCMC 或 cDFT)")

        # For adsorption/GCMC tasks: need gas + material + temperature
        adsorption_kw = ["吸附", "GCMC", "等温线", "Henry", "选择性", "adsorption",
                         "吸附量", "吸附性能", "吸附热", "loading"]
        is_adsorption = any(k in msg for k in adsorption_kw)
        if is_adsorption:
            if not has_gas:
                missing_parts.append("气体种类(CO2/CH4/N2等)")
            if not has_material:
                missing_parts.append("具体MOF材料名称")
            if not has_temp:
                missing_parts.append("温度(如298K)")
            if not has_pressure and "等温线" in msg:
                missing_parts.append("压力范围(如0.1-10 bar)")

        # Classical cDFT, framework-charge assignment, and binding-energy
        # calculations are distinct methods, although all require a material.
        cdft_kw = ["cDFT", "经典密度泛函", "classical density functional"]
        charge_kw = ["电荷", "charge", "DDEC6", "CM5", "PACMAN", "PACMOF"]
        binding_kw = ["结合能", "binding", "adsorption energy"]
        if any(k in msg for k in cdft_kw + charge_kw + binding_kw):
            if not has_material:
                missing_parts.append("具体MOF材料名称")
            if not has_gas and any(k in msg for k in cdft_kw + binding_kw + ["吸附"]):
                missing_parts.append("气体种类")

        # For diffusion tasks: need material + gas
        diffusion_kw = ["扩散", "diffusion", "能垒", "势场"]
        is_diffusion = any(k in msg for k in diffusion_kw)
        if is_diffusion:
            if not has_material:
                missing_parts.append("具体MOF材料名称")
            if not has_gas:
                missing_parts.append("气体种类")

        # For MD/TST: need material + gas
        md_tst_kw = ["MD", "TST", "分子动力学", "过渡态"]
        is_md_tst = any(k in msg for k in md_tst_kw)
        if is_md_tst:
            if not has_material:
                missing_parts.append("具体MOF材料名称")
            if not has_gas:
                missing_parts.append("气体种类")

        # For VASP/DFT: need material
        vasp_kw = ["VASP", "DFT", "结构优化", "单点能", "能带"]
        is_vasp = any(k in msg for k in vasp_kw)
        if is_vasp:
            if not has_material:
                missing_parts.append("具体MOF材料名称")

        # For ML tasks: need target or material
        ml_kw = ["ML", "机器学习", "训练", "预测", "特征", "GBR", "RF", "XGBoost"]
        is_ml = any(k in msg for k in ml_kw)
        if is_ml:
            if not has_material and not has_target:
                missing_parts.append("目标材料或预测属性")

        # For general screening: need gas + material
        if any(k in msg for k in ["筛选", "screen"]):
            if not has_gas:
                missing_parts.append("目标气体种类")
            if not has_material:
                missing_parts.append("材料范围(特定MOF或数据库)")

        # If no specific operation detected but message is very vague
        if not missing_parts and not has_gas and not has_material and is_calculation:
            if len(msg) < 30:  # Very short vague message
                missing_parts.append("气体种类")
                missing_parts.append("具体MOF材料")
                missing_parts.append("计算目标")

        # Edge case: has generic "MOF" but no specific material name
        if not missing_parts and not has_specific_material:
            generic_mof = "MOF" in msg and not has_specific_material
            if generic_mof and (is_adsorption or is_diffusion or is_md_tst or is_vasp):
                if not has_material:
                    missing_parts.append("具体MOF材料名称(如MOF-5, Ni-MOF-74等)")

        # Several scientific categories can identify the same missing field
        # (for example adsorption + cDFT + diffusion).  Ask once per field.
        return "、".join(dict.fromkeys(missing_parts))

    def _contract_has_parameter(self, part: str) -> bool:
        """Whether the durable goal contract already satisfies a pending field."""
        goal = getattr(self, "goal_contract", None)
        params = getattr(goal, "parameters", {}) or {}
        if "计算方法" in part:
            return bool(getattr(goal, "method", ""))
        if "温度" in part:
            return "temperature_K" in params
        if "压力" in part:
            return "pressure_range" in params or "pressure" in params
        if "气体" in part:
            return bool(getattr(goal, "gases", []))
        if "材料" in part:
            return bool(params.get("material"))
        return False

    def _build_param_question(self, missing: str, user_message: str) -> str:
        """Build the user-facing clarifying question for missing params.

        Used by the start() HARD GATE: thermodynamic params / method / gas /
        material must be user-confirmed before any compute job. This question is
        returned directly (never goes through the LLM loop), so an agent CANNOT
        ignore it and delegate straight to a specialist that self-supplies 298K.
        """
        parts = [p for p in missing.split("、") if p]
        lines = []
        for p in parts:
            if "计算方法" in p:
                lines.append(
                    "  - 计算方法：你想用 **GCMC**（构型采样）还是 **经典 cDFT**"
                    "（求孔内平衡流体密度/吸附热力学）？cDFT 在这里不是电子结构 DFT，"
                    "也不直接计算框架电荷或单构型结合能。（必须由你指定）"
                )
            elif "温度" in p:
                lines.append("  - 温度（如 298K）")
            elif "压力" in p:
                lines.append("  - 压力范围（如 0.1–10 bar）")
            elif "气体" in p:
                lines.append("  - 气体种类（CO₂ / CH₄ / N₂ / H₂ 等）")
            elif "材料" in p:
                lines.append("  - 具体材料（MOF 名称或结构文件）")
            else:
                lines.append(f"  - {p}")
        question = (
            "为了准确执行你的计算任务，我需要先和你确认以下参数（在你确认前不会提交任何计算作业）：\n"
            + "\n".join(lines)
            + "\n\n你可以这样回复：『用 GCMC，298K，压力 0.1–10 bar』；"
              "如果你希望我来推荐，可以回复『你推荐』，我会查文献给出建议值再请你确认。"
        )
        # Record the pending question so a reply() that still lacks params can
        # re-ask instead of silently proceeding to submit a job.
        self._pending_param_question = missing
        return question

    def _detect_user_interaction_needed(self, tool_name: str, params: Dict) -> Optional[str]:
        """Detect if user interaction is needed for certain tools.

        Returns:
            Interaction prompt if needed, None otherwise
        """
        # TST vs MD selection - only when user needs to choose between methods
        if tool_name in ["run_string_tst"]:
            if "method" not in params and "tst" not in str(params).lower():
                return (
                    "为了选择合适的扩散计算方法，请告诉我：\n"
                    "1. 你更关心什么？\n"
                    "   a) 扩散系数和机制 (TST)\n"
                    "   b) 动态轨迹和时间相关性 (MD)\n"
                    "2. 计算资源限制？\n"
                    "   a) 有限资源 (TST)\n"
                    "   b) 充足资源 (MD)\n"
                    "请回复选项（如 '1a, 2a'）或直接指定方法（如 'TST' 或 'MD'）"
                )

        # cDFT vs GCMC method selection — auto-select based on task context
        # instead of asking the user. The agent should plan first (in its
        # reasoning), then execute with the selected method. This prevents the
        # "agent asks user for method" anti-pattern that blocks autonomous execution.
        if tool_name in ("run_cdft", "run_gcmc_isotherm", "run_gcmc_batch", "run_henry"):
            low = str(params).lower()
            has_method = any(k in low for k in
                             ("method", "cdft", "gcmc", "functional", "mode", "theory"))
            if not has_method:
                # Auto-select: GCMC for adsorption tools, cDFT for charge tools
                if tool_name in ("run_gcmc_isotherm", "run_gcmc_batch", "run_henry"):
                    # These are GCMC tools — method is implicit
                    pass  # No interaction needed
                elif tool_name == "run_cdft":
                    # This is a cDFT tool — method is implicit
                    pass  # No interaction needed
                # If method is truly ambiguous, let the agent decide in its planning
                # (the agent's system prompt already guides method selection)

        return None

    def _parse_user_choice(self, user_input: str, tool_type: str) -> Dict[str, Any]:
        """Parse user choice and return appropriate parameters."""
        result = {}
        user_input_lower = user_input.lower()

        if tool_type == "diffusion":
            if "tst" in user_input_lower or "1a" in user_input_lower:
                result["method"] = "tst"
                result["explanation"] = "用户选择TST方法（关心扩散机制和能垒）"
            elif "md" in user_input_lower or "1b" in user_input_lower:
                result["method"] = "md"
                result["explanation"] = "用户选择MD方法（关心动态轨迹和时间相关性）"
            else:
                result["method"] = "tst"  # Default
                result["explanation"] = "默认选择TST方法"

        elif tool_type == "adsorption":
            if "cdft" in user_input_lower or "1a" in user_input_lower:
                result["method"] = "cdft"
                result["explanation"] = "用户选择经典cDFT方法（求孔内平衡流体密度与吸附热力学）"
            elif "gcmc" in user_input_lower or "1b" in user_input_lower:
                result["method"] = "gcmc"
                result["explanation"] = "用户选择GCMC方法（用构型采样计算吸附容量和等温线）"
            else:
                result["method"] = "gcmc"  # Default
                result["explanation"] = "默认选择GCMC方法"

        elif tool_type == "dft":
            if "vasp" in user_input_lower or "dft" in user_input_lower or "1a" in user_input_lower:
                result["method"] = "vasp"
                result["explanation"] = "用户选择VASP量子DFT方法（电子结构）"
            else:
                result["method"] = "vasp"
                result["explanation"] = "电子结构任务默认路由到VASP量子DFT；经典cDFT不属于电子结构方法"

        return result

    def _handle_interrupt(self, interruption_point: str):
        """Handle interruption and save state for recovery."""
        self.memory.record_interruption(interruption_point)
        # Save critical state
        self.context["interruption_point"] = interruption_point
        self.context["messages_count"] = len(self.messages)
        self.context["tool_calls_count"] = len(self.memory.tool_call_log)

    def _create_custom_tool(self, tool_name: str, description: str, parameters: Dict) -> bool:
        """Create a custom tool dynamically.

        This allows agents to create new tools when needed.
        """
        try:
            # Create a simple function that returns a placeholder
            def custom_tool(**kwargs):
                return f"Custom tool '{tool_name}' executed with parameters: {kwargs}"

            custom_tool.__name__ = tool_name
            custom_tool.__doc__ = description

            # Add to registry (simplified - in real implementation, would add to ToolRegistry)
            print(f"  🔧 Created custom tool: {tool_name}", flush=True)
            return True
        except Exception as e:
            print(f"  ⚠️ Failed to create custom tool: {e}", flush=True)
            return False

    def _trim_messages(self, messages: List[Dict], max_tokens: int = 80000) -> List[Dict]:
        """Trim messages to fit within token budget, preserving system/user first messages
        and most recent messages. Based on AutoGen's middle-removal pattern.

        IMPROVED (fixes "吞对话"): the old version kept only first-2 + last-6 and
        summarized ONLY tool dumps — so the agent LOST the actual user requests and
        assistant answers from the middle of a long conversation and replied as if
        it had forgotten earlier turns. Now we keep first-2 + last-12, and the
        removed middle is compressed into a dialogue summary that preserves the
        user's actual questions + assistant conclusions (not just tool results)."""
        total = sum(self._estimate_tokens(json.dumps(m, default=str)) for m in messages)
        if total <= max_tokens:
            return messages

        # Keep first 2 messages (user question + system reminder) and recent messages
        if len(messages) <= 4:
            return messages

        # Strategy: keep first 2, remove from middle until under budget
        kept = messages[:2]  # First user message + possible system reminder
        recent = messages[-12:] if len(messages) > 12 else messages[2:]
        middle = messages[2:-12] if len(messages) > 14 else []

        # Compact summary of the REMOVED middle: preserve the real dialogue
        # (user queries + assistant text), not just tool dumps.
        if middle:
            removed_tools = 0
            dialogue_lines: List[str] = []
            for m in middle:
                role = m.get("role")
                content = m.get("content")
                if isinstance(content, list):
                    # Tool message → count tool_result blocks only
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "tool_result":
                            removed_tools += 1
                    continue
                text = str(content or "").strip()
                if not text:
                    continue
                if role == "user":
                    # Skip system-injected prompts / JobWatch notifications /
                    # freeze reports (not real dialogue)
                    if (text.startswith("[系统") or text.startswith("[上下文压缩]")
                            or text.startswith("<reflection") or text.startswith("<plan")
                            or text.startswith("⛔") or text.startswith("[JobWatch]")):
                        continue
                    dialogue_lines.append(f"用户: {text[:300]}")
                elif role == "assistant":
                    # Skip the freeze/"⛔" reports which are transient UI text
                    if text.startswith("⛔"):
                        continue
                    dialogue_lines.append(f"助手: {text[:300]}")
            summary_msg = {
                "role": "user",
                "content": (
                    "[历史对话摘要(上下文压缩)] " +
                    ("；".join(dialogue_lines[-16:]) if dialogue_lines else "") +
                    (f"。期间共执行 {removed_tools} 个工具调用（结果已压缩，细节见各作业输出）。"
                     if removed_tools else "")
                )[:3000]
            }
            kept.append(summary_msg)

        kept.extend(recent)
        return kept

    def _pair_trimmed_blocks(self, msgs: List[Dict]) -> List[Dict]:
        """Repair tool_use↔tool_result pairing after middle-removal trimming.

        The API rejects a history where an assistant message contains a tool_use
        without a matching tool_result in the IMMEDIATELY following message
        (400: "tool_use ids were found without tool_result blocks immediately
        after"), and also rejects a tool_result-only user message whose tool_use
        was removed ("Each tool_result block must have a corresponding tool_use
        block in the previous message").

        Two passes:
          Pass 1 (assistant side): strip any tool_use block whose id does NOT
              appear in the tool_result blocks of the immediately-following
              message. Reasoning text is preserved.
          Pass 2 (user side): drop any tool_result-only user message that is not
              immediately preceded by an assistant message that still contains a
              tool_use.
        """
        def _btype(b):
            if isinstance(b, dict):
                return b.get("type")
            return getattr(b, "type", None)

        def _bid(b):
            if isinstance(b, dict):
                return b.get("id")
            return getattr(b, "id", None)

        def _btrid(b):
            if isinstance(b, dict):
                return b.get("tool_use_id")
            return getattr(b, "tool_use_id", None)

        # ── Pass 1: fix orphan tool_use on the assistant side ──
        out: List[Dict] = []
        n = len(msgs)
        for i, msg in enumerate(msgs):
            content = msg.get("content")
            role = msg.get("role")
            if role == "assistant" and isinstance(content, list):
                tool_use_blocks = [b for b in content if _btype(b) == "tool_use"]
                if tool_use_blocks:
                    tu_ids = {_bid(b) for b in tool_use_blocks}
                    nxt = msgs[i + 1] if i + 1 < n else None
                    nxt_tr_ids = set()
                    if nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list):
                        for b in nxt["content"]:
                            if _btype(b) == "tool_result":
                                nxt_tr_ids.add(_btrid(b))
                    missing = tu_ids - nxt_tr_ids
                    if missing:
                        print(
                            f"  🩹 post-trim: stripped orphan tool_use ids {sorted(missing)} "
                            f"(their tool_result was trimmed away)", flush=True
                        )
                        new_content = [
                            b for b in content
                            if not (_btype(b) == "tool_use" and _bid(b) in missing)
                        ]
                        if not new_content:
                            # Message would be empty → drop entirely
                            continue
                        msg = {"role": role, "content": new_content}
            out.append(msg)
        msgs = out

        # ── Pass 2: fix orphan tool_result on the user side ──
        out = []
        prev_has_tool_use = False
        for msg in msgs:
            content = msg.get("content")
            role = msg.get("role")
            _is_tool_result_msg = (
                role == "user" and isinstance(content, list) and len(content) > 0
                and all(_btype(b) == "tool_result" for b in content)
            )
            if _is_tool_result_msg and not prev_has_tool_use:
                print(f"  🩹 post-trim: dropped orphan tool_result user msg", flush=True)
                continue
            out.append(msg)
            prev_has_tool_use = (
                role == "assistant" and isinstance(content, list)
                and any(_btype(b) == "tool_use" for b in content)
            )
        return out

    def _call_api(self, agent: Agent) -> anthropic.types.Message:
        model = self.config.model if self.config.force_model_override else (agent.model or self.config.model)
        system = agent.get_instructions(self.context)
        if self._runtime_snapshot:
            runtime = self._runtime_snapshot()
            if runtime:
                from .parallel_workflow import runtime_summary
                system += '\n[PARALLEL RUNTIME: main chat retains lifecycle ownership]\n' + json.dumps(runtime_summary(runtime), ensure_ascii=False, default=str)

        # The goal contract is the authoritative source of truth.  Always inject
        # it, even when chat history is trimmed or the active specialist has no
        # private memory yet.  This is the runtime protection against the
        # evaluator4 failure (Kr/Xe+cDFT silently drifting to CO2/SO2+GCMC).
        if self.goal_contract.original_goal:
            system = system + "\n\n" + self.goal_contract.prompt_block()
        if self.context.get('execution_budget'):
            system += '\n[调用预算信息，不是执行授权]\n' + json.dumps(self.context['execution_budget'], ensure_ascii=False)
            system += '\n保留一次答复机会交付已查事实、具体缺项与下一步。预算即将耗尽时不要继续无关查询或重新猜参数；未完成不得宣称完成，也不得为了赶进度编造数据。'
        if self.context.get('reflection_required'):
            system += '\n[ACTUAL FAILURE / NO-PROGRESS REFLECTION]\n' + json.dumps(self.context['reflection_required'], ensure_ascii=False, default=str)
        if agent.name == 'lead-orchestrator' and self.context.get('workflow_patch_proposal'):
            system += '\n[待主chat审核的专业Agent结构化补丁建议；不是批准命令]\n' + json.dumps(self.context['workflow_patch_proposal'], ensure_ascii=False, default=str)[:12000]
        if agent.name == 'lead-orchestrator' and self._pending_workflow_patch:
            system += '\n[PENDING PATCH, NOT YET APPLIED]\n' + json.dumps({k: self._pending_workflow_patch.get(k)
                for k in ('base_version', 'new_version', 'changes', 'reason')}, ensure_ascii=False, default=str)
            system += '\n这只是候选方案，不是必须应用的命令。理解当前真实用户要求后选择应用、修改、撤回或恢复既有结果。拒绝/暂缓不能被当同意；恢复已完成节点应查证历史结果，不能靠upsert把状态变为完成或重新生成。不要从历史提案猜当前版本。'
        if agent.name == 'lead-orchestrator' and self.context.get('user_decision_proposal'):
            system += '\n[专业Agent交回的协商建议；尚未向用户发问/未批准]\n' + json.dumps(self.context['user_decision_proposal'], ensure_ascii=False, default=str)[:8000]
        if (agent.name == 'lead-orchestrator' and self._current_line_id and self._on_workflow_start
                and not self.goal_contract.approved_nodes):
            system += '\n[EXECUTABLE WORKFLOW REQUIRED] 除纯对话外，任何真实工具或委派之前都必须先用record_workflow_draft持久化从输入/准备一直到验证、分析和用户最终交付物的完整DAG及completion_criteria，再读所需get_tool_schema，并用一次propose_workflow_patch编译草案中的全部节点：每个节点包含agent/tool/完整arguments/depends_on/expected_outputs。禁止只编译第一步并称后续再追加。未知的运行期文件用目录/模式产物合同表达；不得用编号文本、旧式handoff或直接工具绕过。执行后只有真实故障或用户改变科学合同才产生版本化局部补丁。'
        if self._lifecycle_store and agent.name != 'supervisor':
            notices = [
                {'event_id': e['event_id'], 'kind': e['kind'],
                 'supervisor': {k:v for k,v in e.get('supervisor_receipt', {}).items() if k != 'evidence_calls'}}
                for e in list(self._lifecycle_store.snapshot().get('events', {}).values())[-5:] if e.get('supervisor_receipt')
                and e.get('main_chat') not in {'obsolete'}
                and e.get('payload', {}).get('origin_goal_version', self.goal_contract.version) == self.goal_contract.version
            ]
            system += '\n[SUPERVISOR MAILBOX]\n' + json.dumps(notices, ensure_ascii=False, default=str)[:8000]
        system += '\n\n[RECOVERY GATE] Failed submissions require recovery_state → fresh diagnosis → concrete fix → successful verification → prepare_retry → one resubmission. Use evidence_ref.call_id from actual replies, NEVER recovery keys/provider tool_use IDs. Diagnosis and verification need DIFFERENT calls, verification later. A successful run_cdft(action=collect, job_work_dir=ACTUAL receipt work_dir) is output evidence. A prose reflection is not permission. Never launch scheduler commands through run_bash or nest sbatch in submit_job. Approved DAGs activate automatically; activation failure requires reconciliation, not serial fallback. expected_outputs must be actual files or {kind:directory,path:...,pattern:*.dat,min_count:N}, never prose. Independent cDFT gases require disjoint explicit job_work_dir/input_dir and exact inputs→submit directory agreement.'

        from .watch_context import get_context
        owner_context = get_context()
        if owner_context.get('username') and owner_context.get('conv_id'):
            root = self.config.project_root / 'runs' / owner_context['username'] / owner_context['conv_id']
            system += '\n[SESSION WORKSPACE]\n' + json.dumps({'username': owner_context['username'],
                'conv_id': owner_context['conv_id'], 'absolute_root': str(root),
                'directory_tool': 'inspect_path', 'shell_cwd': 'own session root by default',
                'cdft_contract': 'pipeline/submit collects results.csv; empty expected_outputs and absent job_work_dir are completed before approval with distinct per-node directories. Never clear an approved DAG to bypass execution.'}, ensure_ascii=False)
        from datetime import datetime
        system += '\n[SERVER CLOCK] 当前服务端时间：'+datetime.now().astimezone().isoformat()+'。日期差和预计等待时长引用工具计算结果，不用模型默认日期；预计启动时间不是保证。'
        system += ('\n[SCHEDULING CONTROL] 主chat使用retarget_queued_job调整同一个自有PENDING作业；不必让用户手动scancel，也不要经run_bash改产物目录或取消重提。'
                   '通常保留CPU/内存申请；若resource_review给出带实际证据且与operator tool profile完全一致的suggested_resources，可在自动科研模式就地修正隐式整节点内存并迁移。'
                   '不得自行猜RAM或节点，RUNNING/UNKNOWN不得直接迁移。科学方法/输入/力场/产物合同不变时，这是调度操作，不是方法切换或DAG科学补丁。')
        system += '\n[ROLE CONTROL] 计算目录租约不阻塞只读诊断或主chat的正式调度控制。用户明确取消某个自有作业时使用cancel_watched_job，停止后继派发，核验调度器终态；request_sent不等于cancelled。主chat不是OS root，不能跨会话写入或绕锁改结果。monitor/supervisor只读查证和提建议，不直接取消、重提或改科学合同。'
        system += '\n[RESULT HANDOFF] worker_prefinish表示任务运行结束但尚未验收，立即让负责该计算的专家查证实际结果、单位和既定条件。生成器分组CIF与输出匹配不符时用revalidate_workflow_node_outputs修合同，不能重跑完成的作业，不能用cDFT专属CSV恢复入口验证生成日志。若已停止的重复尝试覆盖了节点状态，可传completed_job_id恢复原自有已完成作业的历史验收合同；程序冻结原始文件集合，不纳入重复尝试的额外文件。查证通过后finish_workflow_node引用实际evidence_call_ids结束节点，依赖队列自动继续；task_line_update显示完成不能替代真实节点验收。已有科学合同完全不变时，主模型可选择execute_workflow恢复原计划，普通对话版本变化无需重新询问科学条件。向用户只说结构生成/正在检查/准备下一步，不要求用户提供工具或版本术语。'
        system += '\n[CONTROL RESPONSIBILITY] 主chat依据真实用户回合、历史问题和决策原文理解意图，选择具体操作；不输出授权分类bool，不依赖限定词。询问能力、描述错误、引用他人命令不等于要求执行；意图不明时request_user_decision，明确后直接控制，不重复协商。待定编排补丁由你理解用户回答后调用apply_workflow_patch(plan_version)，应用成功会激活执行器；不要先execute尚未应用的版本。程序仅查证来源、归属、权限、状态和回执。reconcile_watched_job仅绑定未决派发身份，不能取消/解锁；取消请求call_id不是诊断证据。编排/工具能力以当前get_tool_schema为准。'
        system += '\n[USER INTERACTION] 向用户默认提供简明节点/执行者/依赖/状态及具体变更摘要，不粘贴整个编排JSON要求用户读代码。完整JSON仅在用户明确请求或高级详情中呈现。换节点说明是否更新原job、是否需要确认、失败实际原因；不把内部守卫失败转成用户手动shell责任。'
        # Shared blackboard: every agent sees the same structured workflow
        # state, while private exploratory memory remains partitioned per agent.
        if self._current_line_id:
            try:
                from .task_line import get_store
                from .workflow_view import line_digest
                graph = self._workflow_view()
                digest = line_digest(graph)
                system = system + "\n\n[SHARED WORKFLOW MEMORY]\n" + digest[:8000]
                if self._runtime_snapshot and self._runtime_snapshot():
                    system += '\n[CURRENT EXECUTION COMPLETION]\n' + json.dumps(
                        self._workflow_completion_state(), ensure_ascii=False)
                    system += '\nThis is executable-DAG completion, not a claim of scientific answer quality. Diagnostic history is not a delegation. Use recorded actual output/evidence paths; do not rediscover the entire workspace or clear historical records to deliver existing results.'
                from .defns import resolve_agent
                assigned = []
                for n in graph.get('steps', []):
                    if 'expected_outputs' in n and n.get('agent') and resolve_agent(n['agent']).name == agent.name and n.get('status') != 'superseded':
                        assigned.append(n)
                envelope = {'workflow_id': self._current_line_id, 'version': self.goal_contract.approved_plan_version,
                            'agent': agent.name, 'nodes': assigned}
                encoded = json.dumps(envelope, ensure_ascii=False, default=str)
                system += '\n[EXECUTABLE NODE ENVELOPE]\n' + (encoded if len(encoded) < 12000 else
                          json.dumps({'workflow_id': self._current_line_id, 'read_required': 'task_line_query contains full arguments; do not guess truncated params'}))
            except Exception:
                pass

        # Inject memory summary into system prompt — PER-AGENT partition, never
        # mixed. Each agent only sees ITS OWN tool log / results / errors, plus
        # its own most-recent exit report (the "进场必读" for re-entry). Inject
        # whenever this agent has ANY partition data (tool calls OR an exit
        # report), so a re-invoked agent always reads its last exit report even
        # if it has made no new tool calls this session.
        _agent_data = self.memory.agent_memories.get(agent.name, {})
        _has_agent_mem = bool(_agent_data.get("tool_call_log")) or bool(_agent_data.get("exit_reports"))
        compact = self.memory.compact_summary(agent_name=agent.name)
        if compact and _has_agent_mem:
            system = system + "\n\n" + compact

        tools = self.registry.claude_tools(
            [f.__name__ for f in agent.functions]
        ) if agent.functions else []

        # Hard guard: strip any assistant tool_use blocks not immediately followed by tool_result
        self._sanitize_messages()
        _sanitized = []
        for i, msg in enumerate(self.messages):
            content = msg.get("content")
            role = msg.get("role")
            if role == "assistant" and isinstance(content, list):
                has_tool_use = any(
                    (isinstance(b, dict) and b.get("type") == "tool_use") or getattr(b, "type", "") == "tool_use"
                    for b in content
                )
                if has_tool_use:
                    nxt = self.messages[i + 1] if i + 1 < len(self.messages) else None
                    # A user message containing ANY tool_result counts as a valid
                    # pairing, even if it also carries text (e.g. the interaction
                    # reply merged into the tool_result message). Using all() here
                    # would drop the tool_use and orphan the tool_result → API 400.
                    nxt_ok = nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list) and any(
                        (isinstance(b, dict) and b.get("type") == "tool_result") or getattr(b, "type", "") == "tool_result"
                        for b in nxt["content"]
                    )
                    if not nxt_ok:
                        # Keep only text blocks
                        text_blocks = [b for b in content if not (
                            (isinstance(b, dict) and b.get("type") == "tool_use") or getattr(b, "type", "") == "tool_use"
                        )]
                        if text_blocks:
                            _sanitized.append({"role": role, "content": text_blocks})
                        continue
            _sanitized.append(msg)
        self.messages = _sanitized

        # Trim messages to fit context window
        trimmed = self._trim_messages(self.messages)

        # Post-trim cleanup: middle-removal can break tool_use↔tool_result pairing
        # in BOTH directions:
        #   1. A user tool_result whose tool_use was trimmed away → "Each
        #      tool_result block must have a corresponding tool_use block in the
        #      previous message".
        #   2. An assistant tool_use whose tool_result was trimmed away → "tool_use
        #      ids were found without tool_result blocks immediately after".
        # _pair_trimmed_blocks repairs both: it strips tool_use blocks with no
        # matching tool_result in the immediately-following message (keeping text
        # reasoning), and drops tool_result-only user messages with no preceding
        # tool_use.
        trimmed = self._pair_trimmed_blocks(trimmed)

        kwargs: Dict[str, Any] = {
            "model": model,
            "max_tokens": self.config.max_tokens,
            "system": system,
            "messages": trimmed,
        }
        if tools:
            kwargs["tools"] = tools
            if getattr(self,'_resource_review_tools',None):
                kwargs['tools']=[tool for tool in tools if tool['name'] in self._resource_review_tools]
            if getattr(self,'_resource_review_final',False):
                kwargs['tools']=[tool for tool in tools if tool['name']=='resource_review_decision']
                kwargs['tool_choice']={'type':'tool','name':'resource_review_decision'}
            if agent.name == 'supervisor' and getattr(self, '_scientific_verification_tools', None):
                verification_capabilities = self._scientific_verification_tools | {'supervisor_decision'}
                if 'inspect_forcefield' in verification_capabilities:
                    verification_capabilities.add('discover_forcefield')  # Resolve canonical names; does not replace required inspection evidence.
                kwargs['tools'] = [tool for tool in tools if tool['name'] in verification_capabilities]
                if getattr(self, '_scientific_source_required', False) and not getattr(self, '_supervisor_finalizing', False):
                    # First verify independently; neither acceptance nor rejection
                    # may substitute the main agent's prose for source evidence.
                    kwargs['tools'] = [tool for tool in kwargs['tools'] if tool['name'] != 'supervisor_decision']
            if agent.name == 'supervisor' and getattr(self, '_supervisor_finalizing', False):
                # Some compatible providers ignore tool_choice. Restrict the
                # actual exposed capability as well as requesting the tool.
                kwargs['tools'] = [tool for tool in tools if tool['name'] == 'supervisor_decision']
                kwargs['tool_choice'] = {'type': 'tool', 'name': 'supervisor_decision'}

        # DEBUG: dump message structure before sending
        try:
            for i, m in enumerate(self.messages if os.environ.get('BIMEM_DEBUG_MESSAGES') == '1' else []):
                c = m.get("content")
                if isinstance(c, list):
                    ts = [getattr(b, "type", "dict") if not isinstance(b, dict) else b.get("type") for b in c]
                    print(f"    [DBG] messages[{i}] role={m['role']} blocks={ts}", flush=True)
                    # Dump tool_use / tool_result IDs to catch pairing mismatches
                    for b in c:
                        btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                        if btype == "tool_use":
                            bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", None)
                            bname = b.get("name") if isinstance(b, dict) else getattr(b, "name", None)
                            print(f"        [DBG]   tool_use id={bid} name={bname}", flush=True)
                        elif btype == "tool_result":
                            bid = b.get("tool_use_id") if isinstance(b, dict) else getattr(b, "tool_use_id", None)
                            print(f"        [DBG]   tool_result tool_use_id={bid}", flush=True)
                        elif btype == "thinking":
                            if isinstance(b, dict):
                                inner = [x.get("type") for x in b.get("content", [])] if isinstance(b.get("content"), list) else None
                                has_nested = any(x.get("type") == "tool_use" for x in b.get("content", []) if isinstance(x, dict))
                                print(f"        [DBG]   thinking inner_types={inner} nested_tool_use={has_nested}", flush=True)
                            else:
                                inner = [getattr(x, "type", None) for x in getattr(b, "content", [])]
                                has_nested = any(getattr(x, "type", None) == "tool_use" for x in getattr(b, "content", []))
                                print(f"        [DBG]   thinking inner_types={inner} nested_tool_use={has_nested}", flush=True)
                else:
                    print(f"    [DBG] messages[{i}] role={m['role']} text={str(c)[:40]}", flush=True)
        except Exception:
            pass

        try:
            from .model_gate import call_model, ModelGateInterrupted, ProviderRateLimited

            def waiting(seconds: int, attempt: int):
                self._progress(
                    agent_name=agent.name,
                    reasoning="供应端 429；共享熔断防止多 session 同时重试。",
                    status=f"模型服务限流；{seconds}秒后自动继续（共享冷却 {attempt}）",
                    log="",
                )

            return call_model(
                lambda: self._stream_api(**kwargs),
                on_wait=waiting,
                on_queue=lambda position: self._progress(agent_name=agent.name, reasoning='',
                    status=f'等待模型调用名额（排队位置 {position}）', log=''),
                interrupted=self._interrupt_requested,
            )
        except ModelGateInterrupted as interrupted:
            raise LLMRateLimitError("共享模型冷却等待已被用户中断") from interrupted
        except anthropic.APITimeoutError as _te:
            raise LLMUnavailableError(f"LLM API 超时（{_te}）") from _te
        except anthropic.APIConnectionError as _ce:
            raise LLMUnavailableError(f"LLM API 连接失败（{_ce}）") from _ce
        except ProviderRateLimited as limited:
            raise LLMRateLimitError(str(limited)) from limited

    def _stream_api(self, **kwargs) -> anthropic.types.Message:
        """Stream the LLM response, honoring user-interrupt (freeze) mid-stream.

        WHY: the old `messages.create()` is a single BLOCKING call — it cannot
        be stopped until the FULL generation (thinking + all tool_use blocks)
        has finished. So after clicking ⛔中断 the agent would keep generating
        (even submitting more jobs) and only freeze afterwards — "中断了他还
        继续思考/提交，最后才把思考和话砍掉", which felt anti-human.

        Streaming lets us break between RAW stream events (fine granularity,
        including mid-thinking), so the freeze cuts the generation almost
        immediately. The caller's post-`_call_api` check then builds the
        freeze report right away. The partial Message returned is still valid
        for history (text-only, no dangling tool_use).
        """
        interrupted = False
        streamed_text = ''
        last_stream_update = 0.0
        with self.client.messages.stream(**kwargs) as stream:
            for _ev in stream:  # raw events → check interrupt on every chunk
                if self._interrupt_requested():
                    interrupted = True
                    break
                delta = getattr(_ev, 'delta', None)
                if getattr(delta, 'type', '') == 'text_delta':
                    streamed_text += delta.text
                    if time.monotonic() - last_stream_update >= .25:
                        self._progress(agent_name=getattr(self.current_agent, 'name', ''),
                            reasoning=streamed_text, status='正在回复', log='')
                        last_stream_update = time.monotonic()
            # get_final_message() drains/waits for the remote stream. Calling it
            # after an interrupt defeats the whole freeze protocol and can hold
            # the sole provider slot until its five-minute HTTP timeout.
            if not interrupted:
                response = stream.get_final_message()
        if interrupted:
            class InterruptedResponse:
                content = []
                stop_reason = 'interrupted'
            return InterruptedResponse()
        else:
            if not self._interrupt_requested() and not any(
                    getattr(block, 'type', '') == 'tool_use' or getattr(block, 'type', '') == 'text' and block.text.strip()
                    for block in response.content):
                raise LLMUnavailableError(f'模型没有返回可执行调用或有效答复（stop_reason={response.stop_reason}）；未完成任务，不改变原目标。')
            return response

    def _sanitize_messages(self):
        """Remove dangling tool_use blocks / orphan tool_results so the API doesn't reject the history.

        Called at the start of each execution loop. Handles both dict blocks and
        Pydantic block objects (TextBlock / ToolUseBlock / ToolResultBlock).
        """
        if not self.messages:
            return

        def _btype(b):
            """Get block type whether it's a dict or an object."""
            if isinstance(b, dict):
                return b.get("type")
            return getattr(b, "type", None)

        def _is_tool_use(b):
            return _btype(b) == "tool_use"

        def _is_tool_result(b):
            return _btype(b) == "tool_result"

        cleaned = []
        for i, msg in enumerate(self.messages):
            content = msg.get("content")
            role = msg.get("role")

            # Assistant message with tool_use blocks that were never answered → drop tool_use, keep text
            if role == "assistant" and isinstance(content, list):
                if any(_is_tool_use(b) for b in content):
                    # Check if the NEXT message is a tool_result user message pairing these
                    # (a mixed tool_result + text user message, e.g. the user's reply
                    # to an interaction prompt, also counts as a valid pairing)
                    nxt = self.messages[i + 1] if i + 1 < len(self.messages) else None
                    nxt_is_tool_result = (
                        nxt and nxt.get("role") == "user" and isinstance(nxt.get("content"), list)
                        and any(_is_tool_result(b) for b in nxt["content"])
                    )
                    if not nxt_is_tool_result:
                        # Drop tool_use blocks, keep text-only blocks
                        text_blocks = [b for b in content if not _is_tool_use(b)]
                        if text_blocks:
                            cleaned.append({"role": role, "content": text_blocks})
                        # If only tool_use with no text, drop entirely
                        continue
            cleaned.append(msg)

        # Second pass: drop orphan tool_result user messages (no preceding tool_use)
        final = []
        prev_was_tool_use_assistant = False
        for msg in cleaned:
            content = msg.get("content")
            role = msg.get("role")
            if role == "user" and isinstance(content, list) and all(_is_tool_result(b) for b in content):
                if not prev_was_tool_use_assistant:
                    continue  # orphan tool_result
            final.append(msg)
            prev_was_tool_use_assistant = (
                role == "assistant" and isinstance(content, list) and any(_is_tool_use(b) for b in content)
            )
        self.messages = final

    def _execute_loop(self, agent: Agent, max_rounds: int = 45) -> str:
        self.context['active_turn_id'] = getattr(self, '_api_turn_id', None) or uuid.uuid4().hex
        self._persist_chain('turn_begin')
        try:
            return self._execute_loop_body(agent, max_rounds)
        finally:
            self._persist_chain('turn_end')

    def _execute_loop_body(self, agent: Agent, max_rounds: int = 45) -> str:
        """核心循环：调 API → 执行工具 → 再调 API，直到 agent 产出纯文本。

        Features:
        - Interactive parameter asking
        - Smart compact module
        - Interrupt handling
        - Active tool creation

        Returns: agent 的最终文本回复
        """
        self._active_tool_futures = [f for f in self._active_tool_futures if not f.done()]
        if self._active_tool_futures:
            self.task_complete = True
            self.last_text = '⛔ 上一次超时工具仍在后台运行。本轮不调用模型、不执行工具、不重新提交；等待该调用结束后再继续。'
            return self.last_text
        self.current_agent = agent
        # Clean dangling tool_use/tool_result so the API accepts the message history
        self._sanitize_messages()
        func_map = agent.function_map()

        # Track consecutive skips across ALL rounds (not just within one round)
        total_consecutive_skips = 0
        max_consecutive_skips = 5
        # Track user interaction requests
        _pending_user_interaction = None

        # ── 监管清单（委派清单）────────────────────────────────────────
        # 每次委派记一步，子Agent回执时标记 done，并把"已完成/未完成"进度
        # 注入给调度器。这样多步研究方案能在同一轮内链式推进，而不是
        # 一轮只做一件事、下一轮就忘了整体方案。
        _turn_delegations = []  # [{agent, task, status: pending/done}]
        # 本轮开始时的历史长度快照 —— 让 all_agents_called 只统计"本轮"的
        # 委派，而不是把整个会话历史算进来（否则 4 个专业 Agent 在任意以往
        # 轮次被调用过后，之后每个子任务回执都会触发 _all_agents_done →
        # 强制提前出终报 = "一轮只做一件事"）。
        _turn_tool_log_start = len(self.memory.tool_call_log)
        _turn_agent_history_start = len(self.memory.agent_history)
        # ── 委派段内工具配额快照 ─────────────────────────────────────
        # 用户红线："每次委派刷新配额"——run_bash/grep_search 等工具的上限计数
        # 必须以**当前委派段**为窗口，而不是整个 session 历史（否则历史轮次积累
        # 到 30/30 后，新委派的 agent 一次 run_bash 都调不了 = "配额有问题"）。
        # 在每次 handoff（切 agent）处重新打快照，配额从 0 重新起算。
        _segment_tool_log_start = len(self.memory.tool_call_log)

        # ── Polling Loop Detection ──────────────────────────────────────
        # The agent often gets stuck calling check_job / run_bash (squeue/tail)
        # alternately to monitor SLURM jobs, burning 50+ tool calls without
        # making progress. Detect this pattern and inject a "wait then check
        # once" instruction to break the loop.
        _polling_streak = 0  # consecutive polling-style tool calls
        _MAX_POLLING_STREAK = 2  # after this many, force a wait instruction
        _total_polling_calls = 0  # total polling calls in this delegation segment
        _MAX_TOTAL_POLLING = 5  # reduced from 8 to 5

        # ── Total Tool Call Budget per Delegation Segment ─────────────────
        # Even general-purpose tools (read_file, run_bash, etc.) must have a
        # hard cap per delegation segment to prevent runaway loops. S01 test
        # showed99 tool calls for a single Henry calculation — most were
        # redundant file reads. This cap applies to ALL tools combined.
        _MAX_TOTAL_TOOLS_PER_SEGMENT = 30  # reduced from 40 to 30
        _segment_total_tools = 0  # counter for all tools in current segment

        # ── 基础工具软上限 ─────────────────────────────────────────────
        # 基础通用工具虽然豁免配额，但设置软上限提醒
        _GENERAL_TOOL_SOFT_LIMIT = 12  # 软上限：超过后提醒优化（从15降到12）
        _general_tool_counts = {}  # 记录各基础工具调用次数

        # Task interpretation belongs to the model; the current typed goal,
        # schema and actual DAG are supplied in _call_api, not keyword hints.

        # ── 快速委派提醒 ─────────────────────────────────────────────
        # 调度器如果连续多轮只读文件不委派，提醒它尽快委派
        _orchestrator_no_handoff_rounds = 0
        _MAX_NO_HANDOFF_ROUNDS = 3  # 最多3轮没有委派就提醒

        def _is_polling_call(tool_name: str, params: dict) -> bool:
            """Detect if a tool call is polling-style (monitoring a job)."""
            return is_job_status_polling(tool_name, params)

        # ── 轮次独立计数（用户红线）────────────────────────────────────
        # max_rounds 只约束"当前委派段"：每次调用(start/reply)与每次委派
        # (handoff 切换 Agent) 都从 0 起算，二次委派绝不把之前 Agent 消耗的
        # 轮次叠加记数。_loop_round 是整轮总迭代（单调，仅供内部比较与安全
        # 上限），_segment_rounds 是当前段内轮次（切换 Agent 时归零）。
        _loop_round = 0
        _segment_rounds = 0
        from .reflection_progress import ReflectionProgress
        reflection_progress = ReflectionProgress()
        _TOTAL_ROUND_SAFETY = 500  # 整轮安全上限，防死循环
        for _loop_round in range(_TOTAL_ROUND_SAFETY):
            # 段内轮次耗尽处理：子Agent 交回调度器继续总方案；调度器段耗尽才
            # break 走强制报告。
            if _segment_rounds >= max_rounds:
                if agent.name != "lead-orchestrator":
                    print(f"  🕒 {agent.name} 本委派段轮次已尽({max_rounds}) → 交回调度器", flush=True)
                    self.messages.append({"role": "user", "content":
                        f"[系统] {agent.name} 已达到本委派段的轮次上限({max_rounds}轮)。"
                        "立即停止调用工具，把当前已有的真实结果整理成简要交付交回调度器；"
                        "未完成项如实说明，不要编造。"
                    })
                    try:
                        _seg_resp = self._call_api(agent)
                        _seg_text = " ".join(
                            b.text for b in _seg_resp.content if hasattr(b, "text")
                        ).strip()
                        if _seg_text:
                            self._report_partial(agent.name, _seg_text)
                            self.memory.record_exit_report(
                                agent.name,
                                f"[{agent.name} 退场报告·轮次耗尽]\n{_seg_text[:800]}",
                            )
                    except Exception as _seg_e:
                        print(f"  ⚠️ 子Agent轮次耗尽交回失败: {_seg_e}", flush=True)
                    agent = ORCHESTRATOR
                    self.current_agent = agent
                    func_map = agent.function_map()
                    _segment_rounds = 0
                    _segment_tool_log_start = len(self.memory.tool_call_log)  # 新委派段配额从0起算
                    _polling_streak = 0; _total_polling_calls = 0  # Reset polling counters
                    continue
                break
            _segment_rounds += 1
            # No forced handoff — orchestrator decides when to delegate

            # User interrupt (Claude-Code-style Esc): if the user requested a
            # freeze, STOP the loop immediately. Crucially:
            #   - NO further LLM call and NO further tool calls (stall thinking)
            #   - Previously submitted SLURM jobs KEEP RUNNING (never cancelled)
            #   - The current delegation is FROZEN, awaiting a subsequent
            #     delegation (the user's next message / redirect).
            _interrupt = self._interrupt_requested()
            if _interrupt:
                print(f"  ⛔ 用户中断（round {_segment_rounds}）—— 冻结本轮，停止进一步思考/动作", flush=True)
                self.memory.record_interruption(f"round {_segment_rounds} by {agent.name}")
                self._progress(
                    agent_name=agent.name,
                    reasoning="用户中断：本轮冻结，已提交作业不受影响。",
                    status=f"⛔ 用户中断 (round {_segment_rounds}) — 等待后续委派",
                    log="用户中断：冻结本轮，输出真实进展（job_id/状态），不编造。",
                )
                final = self._build_interrupt_report(agent.name)
                self.task_complete = True
                self.last_text = final
                return final

            self._progress(
                agent_name=agent.name,
                reasoning="",
                status=f"🔄 {agent.name} 思考中... (委派段轮次 {_segment_rounds}/{max_rounds})",
                log=f"委派段轮次 {_segment_rounds}: {agent.name} 正在分析..."
            )

            # Resilient LLM call: on endpoint timeout/connection failure, inject
            # a retry notice (with a cap) instead of letting the exception kill
            # the whole multi-round run.
            self._api_failures = getattr(self, "_api_failures", 0)
            try:
                response = self._call_api(agent)
                self._api_failures = 0
                # User interrupt may have fired MID-stream (streaming cut the
                # generation short). Freeze NOW — don't process the partial
                # response or execute any tools from it.
                if self._interrupt_requested():
                    print(f"  ⛔ 用户中断（LLM 生成中被截断）—— 立即冻结本轮", flush=True)
                    self.memory.record_interruption(f"stream-cut by {agent.name}")
                    self._progress(
                        agent_name=agent.name,
                        reasoning="用户中断：生成被截断，本轮冻结。",
                        status="⛔ 用户中断 — 已停止思考",
                        log="LLM 生成中收到中断，立即停止。",
                    )
                    final = self._build_interrupt_report(agent.name)
                    self.task_complete = True
                    self.last_text = final
                    return final
            except ExecutionBudgetExceeded as error:
                self.task_complete = False
                self._waiting_for_user_input = True
                self.last_text = f'执行预算已达到上限：{error}。原目标与已查证据已保存；尚未完成，不是LLM API故障。请确认继续查证或调整方案。'
                self._pending_user_interaction = {'tool': 'budget_decision', 'params': {}, 'prompt': self.last_text}
                self._emit_lifecycle_event('budget_exhausted', {'requires_user': True, 'main_already_handling': True})
                self._checkpoint('execution_budget_exhausted')
                return self.last_text
            except LLMRateLimitError as _rate_e:
                self._api_failures += 1
                if self._interrupt_requested():
                    self.memory.record_interruption(f"provider cooldown by {agent.name}")
                    self.task_complete = False
                    self.last_text = self._build_interrupt_report(agent.name)
                    return self.last_text
                try:
                    max_rate_retries = max(0, int(os.environ.get("BIMEM_LLM_RATE_LIMIT_RETRIES", "0")))
                except (TypeError, ValueError):
                    max_rate_retries = 0
                if max_rate_retries and self._api_failures >= max_rate_retries:
                    self.task_complete = False
                    self.last_text = (f"⚠️ 模型服务持续限流，已自动探测 {self._api_failures} 次。"
                                      "原目标、DAG 和证据已保存，本轮未完成。")
                    self._checkpoint('provider_rate_limit_exhausted')
                    return self.last_text
                retry_label = str(max_rate_retries) if max_rate_retries else "∞"
                status = (f"模型服务限流；已进入共享冷却，将自动继续 "
                          f"({self._api_failures}/{retry_label})")
                print(f"  ⏳ {status}", flush=True)
                self._progress(agent_name=agent.name, reasoning=str(_rate_e), status=status,
                               log="provider_rate_limit_cooldown")
                self._checkpoint('provider_rate_limit_cooldown')
                continue
            except LLMUnavailableError as _llm_e:
                self._api_failures += 1
                if self._api_failures >= 3:
                    self.task_complete = True
                    self.last_text = (f"⚠️ LLM API 连续失败 {self._api_failures} 次，无法继续执行。"
                                      f"最后错误：{_llm_e}。请稍后重试。")
                    print(f"  ⛔ {self.last_text}", flush=True)
                    return self.last_text
                notice = (f"[系统] ⚠️ LLM API 调用失败（第 {self._api_failures} 次）：{_llm_e}。"
                          f"请稍候，自动重试中…")
                print(f"  🔁 {notice}", flush=True)
                self.messages.append({"role": "user", "content": notice})
                continue
            # 提取文本和工具调用
            # 注意：ThinkingBlock 有 .thinking 属性而非 .text
            text_parts = []
            thinking_parts = []
            tool_calls = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif hasattr(block, "thinking"):
                    # Keep private model reasoning out of final/user-visible text.
                    # The frontend receives the structured decision trace
                    # (GoalContract + TaskLine + tool/error events) instead.
                    thinking_parts.append(block.thinking)
                elif block.type == "tool_use":
                    tool_calls.append(block)

            # Report only model-authored visible text to the frontend. Internal
            # ThinkingBlock content remains internal and is never persisted as a
            # user-facing answer.
            if text_parts:
                thinking_text = "\n".join(text_parts)[:500]
                self._progress(
                    agent_name=agent.name,
                    reasoning=thinking_text,
                    status=f"💡 {agent.name} 正在分析...",
                    log=f"{agent.name} 思考: {thinking_text[:100]}..."
                )

            # Operational delivery is not completion of the scientific plan.
            # Handle it before knowledge-answer and runtime completion shortcuts.
            if agent.name == 'lead-orchestrator' and not tool_calls:
                control_delivery = self.context.pop('pending_control_delivery', None)
                if control_delivery:
                    self.task_complete = False
                    self.last_text = control_delivery
                    self._checkpoint('delivered_scheduler_control_receipt')
                    return self.last_text

            # If orchestrator generated substantial text (>300 chars), accept it as the answer
            # even if it also generated tool calls (LLM often generates both)
            # BUT: block if user asked for computation (needs workflow, not knowledge)
            if agent.name == "lead-orchestrator" and text_parts:
                combined_text = "\n".join(text_parts)
                # Check ALL user messages (the original task + any follow-up
                # replies after an interaction), not just messages[0]. Otherwise
                # a follow-up reply like "方法方面我想用cDFT来算" is ignored and
                # a computation task wrongly accepts a direct answer, skipping
                # the analyst delegation entirely.
                def _text_of(msg) -> str:
                    c = msg.get("content")
                    if isinstance(c, str):
                        return c
                    if isinstance(c, list):
                        # Mixed tool_result + text blocks (interaction reply) → keep text
                        return " ".join(
                            b.get("text", "") for b in c
                            if (isinstance(b, dict) and b.get("type") == "text")
                        )
                    return ""
                _user_msg = " ".join(_text_of(m) for m in self.messages
                                    if m.get("role") == "user").lower()
                _user_wants_computation = any(kw in _user_msg for kw in
                    ["计算", "等温线", "gcmc", "henry", "扩散", "吸附量", "电荷", "cdft", "vasp",
                     "模拟", "优化", "训练", "预测", "帮我做", "帮我算", "submit", "run_"])
                _user_wants_knowledge = any(kw in _user_msg for kw in
                    ["是什么", "原理", "特点", "介绍", "解释", "什么是", "有哪些"])
                # Accept direct answer ONLY for knowledge questions (not computation)
                if len(combined_text) > 300 and not _user_wants_computation:
                    # Check if tool calls are all handoffs (can be skipped)
                    all_handoffs = all(tc.name.startswith('handoff_to_') for tc in tool_calls) if tool_calls else True
                    if all_handoffs:
                        print(f"  ✅ Orchestrator direct answer accepted ({len(combined_text)} chars, skipping {len(tool_calls)} calls)", flush=True)
                        self._progress(
                            agent_name=agent.name,
                            reasoning=combined_text[:500],
                            status="✅ 直接回答",
                            log=f"调度器直接回答 ({len(combined_text)} chars)"
                        )
                        self.task_complete = True
                        self.last_text = combined_text
                        return combined_text

            # 没有工具调用 → agent想结束，但检查是否真的完成了
            if not tool_calls:
                if agent.name == 'lead-orchestrator' and self._runtime_snapshot:
                    runtime = self._runtime_snapshot()
                    if runtime.get('status') in {'active', 'needs_user'}:
                        self.task_complete = False
                        self.last_text = ('\n'.join(text_parts) + '\n编排执行器仍负责独立分支；主chat继续在岗，收到节点事件后处理，未宣称整体完成。').strip()
                        self._checkpoint('parallel_workflow_wait')
                        return self.last_text
                total_tool_calls = len(self.memory.tool_call_log)
                unique_tools = set(t['tool'] for t in self.memory.tool_call_log)

                # ── FABRICATION GUARD: agent claims to have submitted / computed
                # something but never actually called the corresponding tool.
                # e.g. harness writes "✅ 提交成功, job_id=123" with zero tool calls.
                # NOTE: only applies to agents that OWN computation tools AND no
                # real computation/submission tool was called by ANY agent recently
                # (handoffs count as evidence the work was delegated & executed).
                _agent_tools = {f.__name__ for f in agent.functions}
                _claimed_text = "\n".join(text_parts) if text_parts else ""
                _recent_tools = [t.get('tool', '') for t in self.memory.tool_call_log[-8:] if not t.get('failed')]
                _any_real_tool = any(
                    t.startswith(("run_", "submit", "check", "diagnose", "generate",
                                  "build", "ml_", "extract", "handoff_to_"))
                    for t in _recent_tools
                )
                _any_real_tool = _any_real_tool or self._has_verified_runtime_computation()
                _claim_risk = (
                    self._claims_execution_result(_claimed_text)
                    and not _any_real_tool
                    and _claimed_text.strip()
                )
                if _claim_risk and len(_claimed_text) > 120:
                    print(f"  🛡️ 防伪: {agent.name} 声称完成但未实际调用工具，强制继续 (round {_segment_rounds})", flush=True)
                    self._progress(
                        agent_name=agent.name,
                        reasoning="",
                        status=f"🛡️ 检测到疑似虚构结果: {agent.name}",
                        log=f"{agent.name} 声称{_claimed_text[:80]}... 但未调用任何工具"
                    )
                    self.messages.append({"role": "user", "content":
                        f"[系统·防伪] 你的文本声称已完成任务（'{_claimed_text[:180]}...'），"
                        "但当前没有支持这项执行主张的计算/提交证据；只读查询不等于计算完成。"
                        "绝不能只写文本编造成功结果。"
                        "请撤回没有证据的执行主张；仅当用户已授权计算时，才按结构化编排获取真实输出。"
                        "如果你无法调用工具，请如实告知失败原因，不要声称任务成功。"
                    })
                    continue

                # Sub-agent finished → hand back to orchestrator for further delegation
                if agent.name != "lead-orchestrator":
                    finished_agent = agent.name
                    from .defns import ORCHESTRATOR
                    print(f"  🔄 {finished_agent} finished → returning to orchestrator", flush=True)
                    self._progress(
                        agent_name=finished_agent,
                        reasoning="",
                        status=f"✅ {finished_agent} 完成任务",
                        log=f"{finished_agent} 已完成，返回调度器"
                    )
                    # Emit partial result: sub-agent's final text in real-time
                    _finished_text = "\n".join(text_parts) if text_parts else ""
                    if _finished_text:
                        self._report_partial(finished_agent, _finished_text)
                    # ── AGENT EXIT PROTOCOL ───────────────────────────────
                    # The compute agent is exiting (handing back to orchestrator).
                    # Per the user's directive, it must write a brief exit report
                    # BEFORE leaving so the next entry (same-session re-entry or
                    # re-invocation by another agent) can read where it left off.
                    _exit_report = self._build_agent_exit_report(
                        finished_agent, _finished_text
                    )
                    self.memory.record_exit_report(finished_agent, _exit_report)
                    self.memory.record_handoff(finished_agent, "lead-orchestrator")
                    agent = ORCHESTRATOR
                    self.current_agent = agent
                    func_map = agent.function_map()
                    _segment_rounds = 0  # 交回调度器 = 新委派段，轮次从 0 重新记
                    _segment_tool_log_start = len(self.memory.tool_call_log)  # 新委派段配额从0起算
                    _polling_streak = 0; _total_polling_calls = 0  # Reset polling counters

                    # Check if all4 agents have been called (from handoffs).
                    # 只统计"本轮"的委派（_turn_tool_log_start / _turn_agent_history_start
                    # 是本轮开始时的快照），避免历史轮次的委派污染本轮判断。
                    agents_called = set()
                    for t in self.memory.tool_call_log[_turn_tool_log_start:]:
                        agents_called.add(t['agent'])
                    # Also check handoff targets in THIS turn
                    for step in self.memory.agent_history[_turn_agent_history_start:]:
                        parts = step.split(' → ')
                        for p in parts:
                            if p != 'lead-orchestrator':
                                agents_called.add(p)
                    all_agents_called = {'adsorption', 'analyst', 'communicator', 'harness-maintainer'}.issubset(agents_called)

                    if all_agents_called:
                        # Coverage is advisory. Four agent receipts do not prove
                        # completion of the executable DAG or verified outputs.
                        self.messages.append({"role": "user", "content":
                            f"[系统] 所有被调用的专业Agent都已完成任务。"
                            "**但如果你的总体研究方案还有后续步骤（如：材料已构建→还差赋电荷→"
                            "还差cDFT/GCMC计算→还差选择性与物理特征分析），必须继续委派下一步，"
                            "不能就此停手。**只有整体方案全部步骤都完成、结果都已读取并分析，"
                            "才输出最终报告（报告应基于实际计算结果，不要编造数据）。"
                        })
                    else:
                        self.messages.append({"role": "user", "content":
                            f"[系统] {finished_agent} 已完成任务。"
                            "**不要就此停手汇报'已完成'。**"
                            "请对照你向用户给出的整体方案：后续步骤（构建材料→赋电荷→计算→"
                            "选择性/特征分析…）还有哪些没执行？必须继续委派下一步骤，"
                            "直到总体研究方案全部完成，最后再输出最终报告。"
                        })

                    # ── 监管清单：标记该子Agent本次委派为 done，并把整体进度注入给调度器 ──
                    for _d in reversed(_turn_delegations):
                        if _d["agent"] == finished_agent and _d["status"] == "pending":
                            _d["status"] = "done"
                            break
                    if _turn_delegations:
                        _cl_lines = ["[系统·进度清单] 本轮研究方案的委派进度："]
                        for _d in _turn_delegations:
                            _mark = "✅ 已完成" if _d["status"] == "done" else "⬜ 待执行"
                            _cl_lines.append(f"  {_d['agent']} {_mark}：{_d['task']}")
                        _pending_left = [d for d in _turn_delegations if d["status"] == "pending"]
                        if _pending_left:
                            _cl_lines.append(
                                f"还有 {len(_pending_left)} 个步骤未执行。"
                                "请对照整体方案继续委派下一未完成步骤，直到全部 ✅ 完成，"
                                "再输出最终报告。"
                            )
                        else:
                            _cl_lines.append(
                                "全部步骤已 ✅ 完成。若你给出的整体方案还有未列入清单的步骤，"
                                "仍应继续执行；否则可输出最终报告。"
                            )
                        self.messages.append({"role": "user", "content": "\n".join(_cl_lines)})

                    # Main model reviews the specialist delivery against shared
                    # evidence; prose keywords never trigger reruns or method changes.
                    continue

                # Dialogue intent and planning are model-owned. Plans are saved
                # with record_workflow_draft/propose_workflow_patch, not parsed
                # from numbered prose or confirmation keywords.

                # ── REPORT GATE: before letting the orchestrator finish, verify
                # every SLURM job this conversation spawned has COMPLETED
                # normally (polled via sacct/squeue + result-file validation).
                # No final report until all jobs are accounted for. ──
                _gate = self._gate_jobs_before_report()
                if not _gate["ok"]:
                    print(f"  🚦 报告门禁拦截: {_gate['reason']} "
                          f"(running={len(_gate.get('running', []))}, failed={len(_gate.get('failed', []))})", flush=True)
                    self._progress(
                        agent_name=agent.name,
                        reasoning="",
                        status=f"🚦 作业未全部完成 ({_gate['reason']})",
                        log=_gate["message"][:140],
                    )
                    if _gate.get('reason') == 'running':
                        self._waiting_for_user_input = True
                        self._pending_user_interaction = {
                            'tool': 'lifecycle_wait', 'params': {'jobs': _gate.get('running', [])},
                            'prompt': '⏳ 计算仍在运行，编排与节点细节已保存。主chat/监督Agent由终态事件自动唤醒并继续下一节点，无需重复提交。\n'
                                      + '\n'.join(_gate.get('running', [])[:10]),
                        }
                        self.task_complete = False
                        self.last_text = self._pending_user_interaction['prompt']
                        self._emit_lifecycle_event('waiting_jobs', {'jobs': self._pending_jobs_for_conv()})
                        self._checkpoint('waiting_for_scheduler_terminal')
                        return self.last_text
                    self.messages.append({"role": "user", "content": _gate["message"]})
                    continue

                # ── STRUCTURED PLAN COMPLETION GATE ─────────────────────
                # Text saying "done" is insufficient while approved TaskLine
                # nodes remain pending/running/submitted. This is the durable
                # replacement for relying on the model to remember its plan.
                if self._current_line_id and (self.goal_contract.approved_plan_version or self.goal_contract.execution_mode == 'workflow'):
                    try:
                        _completion = self._workflow_completion_state()
                        if not _completion['ok'] and not _completion['blockers']:
                            self.task_complete = False
                            self.context['workflow_resume_error'] = _completion
                            self.last_text = '保存的批准方案与执行记录不一致，暂不能确认任务完成。已有结果已保留，不会重新计算；需要恢复一致的编排记录。'
                            self._checkpoint('workflow_contract_inconsistent')
                            return self.last_text
                        _open_steps = _completion['blockers']
                        if _open_steps:
                            _summary = "; ".join(
                                f"{s.get('step_id')}={s.get('status', 'pending')}({s.get('tool') or 'unassigned'})"
                                for s in _open_steps[:10]
                            )
                            self.messages.append({"role": "user", "content":
                                "[系统·结构化完成门禁] 批准方案仍有未完成节点："
                                f"{_summary}。请按 depends_on 继续下一个可执行节点；"
                                "若节点处于 needs_user，必须停止并向用户提问。"
                            })
                            continue
                    except Exception as _wg_e:
                        self.task_complete = False
                        self.last_text = f'执行链核验失败，已有结果保留，不能宣称完成：{_wg_e}'
                        self._checkpoint('workflow_completion_check_failed')
                        return self.last_text

                # Scientific claims and answer quality are audited by the
                # independent model/evidence review below. Word presence,
                # character counts and numeric formatting cannot prove quality
                # and must not manufacture a new execution/rewriting loop.

                # Let the agent decide what to do next — no forced tool/agent requirements
                self.task_complete = True
                final_text = "\n".join(text_parts) if text_parts else "(no text)"
                final_text += self.scientific_input_receipt()
                self.last_text = final_text
                self._progress(
                    agent_name=agent.name,
                    reasoning=final_text[:500],
                    status="✅ 任务完成",
                    log=f"{agent.name} 输出最终结果"
                )
                # Agent exit protocol: write a brief report before leaving so the
                # next entry (same-session re-entry / re-invocation) reads it.
                self.memory.record_exit_report(
                    agent.name,
                    self._build_agent_exit_report(agent.name, final_text),
                )
                if self._on_lifecycle_event:
                    question = self.context.get('scientific_review_question') or self.goal_contract.active_goal
                    event = self.scientific_review_event(question, final_text)
                    if event:
                        self._emit_lifecycle_event(event['kind'], event['payload'])
                        self._checkpoint('scientific_review_queued')
                return final_text
            
            # 有工具调用 → 执行工具
            self.messages.append({"role": "assistant", "content": response.content})

            # A Session has one active agent, so handoffs are explicitly serial.
            # If the model emits several at once, preserve the FIRST declared
            # dependency rather than the old behavior of silently executing only
            # the last one (which inverted plans and skipped prerequisites).
            handoff_calls_in_batch = [tc for tc in tool_calls if tc.name.startswith('handoff_to_')]
            _batch_skipped = {}
            if len(handoff_calls_in_batch) > 1:
                first_id = handoff_calls_in_batch[0].id
                for tc in handoff_calls_in_batch[1:]:
                    _batch_skipped[tc.id] = {'skipped': True, 'status': 'deferred',
                        'reason': 'one active agent per Session; finish the first delegation before requesting this one',
                        'blocking_tool_use_id': first_id, 'tool': tc.name, 'arguments': tc.input,
                        'executed': False, 'requires_reissue': True}
                tool_calls = [
                    tc for tc in tool_calls
                    if not tc.name.startswith('handoff_to_') or tc.id == first_id
                ]

            # Batch-level dedup: remove duplicate tool calls within same batch.
            # NOTE: the assistant message already contains ALL tool_use blocks,
            # so deduped calls must STILL get a synthetic tool_result later,
            # otherwise the API rejects the history (tool_use without tool_result).
            consecutive_skips = 0  # init per-round counter BEFORE dedup can += 1
            seen_in_batch = {}
            _batch_readonly = {'read_file', 'grep_search', 'inspect_path', 'inspect_run',
                'task_line_query', 'recovery_state', 'lifecycle_state', 'check_job',
                'diagnose_job', 'list_my_jobs', 'resource_health'}
            deduped_tool_calls = []
            _batch_dup_ids = set()  # tool_use ids removed by batch dedup → need synthetic results
            for tc in tool_calls:
                if tc.name.startswith('handoff_to_'):
                    seen_in_batch.clear()
                    deduped_tool_calls.append(tc)
                    continue
                if tc.name not in _batch_readonly:
                    # Repeated mutations may be intentional. Only submissions
                    # have durable semantic idempotency via RecoveryGate.
                    seen_in_batch.clear()
                    deduped_tool_calls.append(tc)
                    continue
                batch_key = (tc.name, json.dumps(tc.input, sort_keys=True, separators=(',', ':'), default=str))
                if batch_key not in seen_in_batch:
                    seen_in_batch[batch_key] = tc.id
                    deduped_tool_calls.append(tc)
                else:
                    print(f"  ⏭️  {tc.name} (skipped: duplicate in batch)", flush=True)
                    _batch_dup_ids.add(tc.id)
                    _batch_skipped[tc.id] = {'deduplicated': True, 'executed': False,
                        'original_tool_use_id': seen_in_batch[batch_key],
                        'reason': 'same tool and complete arguments in this batch'}
                    consecutive_skips += 1
                    total_consecutive_skips += 1
            tool_calls = deduped_tool_calls

            # Consecutive skip limit: force break if model keeps retrying limited tools
            consecutive_skips = 0  # Reset per-round counter

            # Argument completeness is checked against the actual tool schema,
            # not a word list in the first greeting or a historical message.

            tool_results = []
            _effective_agent = agent  # Tracks agent through handoff chain within this batch
            _original_func_map = func_map  # Keep original func_map for handoff calls

            # Force handoff mode: after 4+ tool calls with no handoffs, only allow handoff tools.
            # General-purpose tools (read_file/write_file/run_bash/grep_search) are every
            # agent's base capability and do NOT count toward the delegation requirement.
            # BUT: to cut the orchestrator's ineffective exploration, general tools DO count
            # toward a higher cap (8) — otherwise the orchestrator can burn unlimited rounds
            # on "environment verification" without ever delegating a computation.
            _GENERAL_TOOL_NAMES = {"read_file", "write_file", "run_bash", "grep_search"}
            _ho_count = sum(1 for t in self.memory.tool_call_log if t['tool'].startswith('handoff_to_'))
            _tc_count = sum(1 for t in self.memory.tool_call_log if t['tool'] not in _GENERAL_TOOL_NAMES)
            _all_tc_count = sum(1 for t in self.memory.tool_call_log)
            _force_handoff_mode = (
                agent.name == "lead-orchestrator" and _ho_count == 0
                and (_tc_count >= 4 or _all_tc_count >= 8)
                and not (self._runtime_snapshot and self._runtime_snapshot())
            )

            # User interrupt arrived while this batch of tool calls was about to
            # run — stop executing ANY further tools. Synthesize "skipped"
            # tool_results so the assistant tool_use pairing stays valid, then
            # let the round-top check build the freeze report.
            if self._interrupt_requested():
                print(f"  ⛔ 用户中断（工具执行前）—— 跳过本批 {len(tool_calls)} 个工具", flush=True)
                for _sk in tool_calls:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": _sk.id,
                        "content": "⛔ [用户中断] 该工具未执行（本轮已冻结）。",
                    })
                self.messages.append({"role": "user", "content": tool_results})
                continue

            for tc in tool_calls:
                # ── Goal-contract hard gate ───────────────────────────────
                capability_map = _original_func_map if tc.name.startswith('handoff_to_') else func_map
                if tc.name not in capability_map:
                    if tc.name == 'request_user_decision' and agent.name != 'lead-orchestrator':
                        from .defns import ORCHESTRATOR
                        self.context['user_decision_proposal'] = {'agent': agent.name, 'proposal': copy.deepcopy(tc.input),
                            'evidence_refs': [c.get('call_id') for c in self.memory.tool_call_log[-5:]]}
                        self._pending_handoff = ORCHESTRATOR
                    tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                        'content': json.dumps({'blocked': True, 'executed': False,
                            'reason': f"tool {tc.name} is not exposed to {agent.name}",
                            'next_action': 'handoff_to_lead-orchestrator with facts and a structured decision proposal; main chat owns user negotiation'})})
                    continue
                chain_block = self._chain_required_block(tc.name)
                if chain_block:
                    tool_results.append({'type':'tool_result','tool_use_id':tc.id,
                        'content':json.dumps({'blocked':True,'executed':False,'reason':chain_block},ensure_ascii=False)})
                    self._progress(agent_name=agent.name,reasoning='',status='先建立结构化编排图',
                                   log=f'{tc.name} blocked until the current-session DAG exists')
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue
                # This is code-level enforcement, not a prompt suggestion.
                # A compute/delegation call that changes protected gases or the
                # selected method is rejected before any executor/job submission.
                try:
                    _node_for_call = self._workflow_node_for_call(tc.name, dict(tc.input or {}))
                except ValueError as ambiguity:
                    tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                        'content': json.dumps({'blocked': True, 'reason': str(ambiguity), 'executed': False})})
                    continue
                _node_step_id = _node_for_call.get('step_id') or tc.name
                if _node_for_call.get('agent') and is_submission(tc.name, tc.input or {}):
                    from .defns import resolve_agent
                    if resolve_agent(_node_for_call['agent']).name != agent.name:
                        tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                                             'content': json.dumps({'blocked': True, 'reason': 'node belongs to another agent; delegate its executable envelope instead'})})
                        continue
                _goal_ok, _goal_reason = self.goal_contract.guard_tool_call(
                    tc.name, dict(tc.input or {})
                )
                if self.context.get('scientific_correction_scope') and (is_submission(tc.name, tc.input or {}) or tc.name.startswith('handoff_to_') or tc.name in {'run_bash','execute_workflow','propose_workflow_patch'}):
                    _goal_ok, _goal_reason = False, '[SCIENTIFIC_CORRECTION_BLOCK] Answer correction cannot submit, delegate, execute shell or change the approved workflow.'
                if not _goal_ok:
                    print(f"  🧭 {tc.name} blocked by goal contract: {_goal_reason[:160]}", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": _goal_reason,
                    })
                    self.memory.record_error(
                        f"goal drift blocked: {tc.name}: {_goal_reason[:220]}",
                        agent_name=agent.name,
                    )
                    self._progress(
                        agent_name=agent.name,
                        reasoning="",
                        status="🧭 已阻止偏离原始目标的调用",
                        log=f"目标契约阻止 {tc.name}: {_goal_reason[:140]}",
                    )
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue

                _dep_ok, _dep_reason = self._check_taskline_dependencies(tc.name, dict(tc.input or {}))
                if not _dep_ok:
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": _dep_reason,
                    })
                    self._progress(
                        agent_name=agent.name,
                        reasoning="",
                        status="⛓️ 等待前置步骤完成",
                        log=_dep_reason[:160],
                    )
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue

                _schema_issues: List[str] = []
                if tc.name.startswith('handoff_to_'):
                    _schema_issues = self.registry.validate_handoff_params(tc.input)
                elif self.registry.get(tc.name):
                    _schema_issues = self.registry.validate_params(tc.name, dict(tc.input or {}))
                if _schema_issues:
                    _schema = self.registry.get(tc.name).input_schema if self.registry.get(tc.name) else {}
                    _validation_error = json.dumps(
                        self.registry._validation_receipt(tc.name, _schema_issues, _schema),
                        ensure_ascii=False, indent=2,
                    )
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": _validation_error,
                    })
                    if self._current_line_id:
                        try:
                            from .task_line import get_store as _get_tl
                            _get_tl().upsert_step(
                                self._current_line_id, _node_step_id, tool=tc.name,
                                status="failed", agent=agent.name,
                                arguments=dict(tc.input or {}),
                                validation={"goal_contract": "passed", "schema": "failed", "issues": _schema_issues},
                                plan_version=self.goal_contract.approved_plan_version,
                                note="参数校验失败，未执行工具",
                            )
                        except Exception:
                            pass
                    self.memory.record_error(
                        f"{tc.name} parameter validation: {'; '.join(_schema_issues)}",
                        agent_name=agent.name,
                    )
                    self.memory.record_tool_call(
                        agent.name, tc.name, dict(tc.input or {}),
                        _validation_error, failed=True,
                    )
                    self.memory.tool_call_log[-1]['workflow_step_id'] = _node_step_id
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue

                _is_coordination_tool = tc.name in {
                    "discover_forcefield", "inspect_forcefield",
                    "get_tool_schema",
                    "task_line_query", "task_line_update", "check_job",
                    "list_my_jobs", "diagnose_job", "recovery_state", "prepare_retry", "accept_recovered_result", "lifecycle_state", "propose_workflow_patch", "apply_workflow_patch", "discard_workflow_patch", "request_user_decision", "reconcile_watched_job", "execute_workflow", "message_workflow_node", "resolve_local_workflow_write", "retarget_queued_job", "cancel_watched_job", "revalidate_workflow_node_outputs","finish_workflow_node",
                }
                if self._current_line_id and not tc.name.startswith("handoff_to_") and not _is_coordination_tool:
                    try:
                        from .task_line import get_store as _get_tl
                        _get_tl().upsert_step(
                            self._current_line_id,
                            _node_step_id,
                            tool=tc.name,
                            status="running",
                            agent=agent.name,
                            arguments=dict(tc.input or {}),
                            validation={"goal_contract": "passed", "schema": "passed"},
                            plan_version=self.goal_contract.approved_plan_version,
                        )
                    except Exception:
                        pass

                # Block intermediate handoff calls in a batch (only last one is applied)
                if tc.name.startswith('handoff_to_') and self._pending_handoff:
                    print(f"  ⏭️  {tc.name} (skipped: handoff to {self._pending_handoff.name} already pending)", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": f"A handoff to {self._pending_handoff.name} is already pending. Complete this handoff first.",
                    })
                    continue

                # ── 委派去重机制 ──────────────────────────────────────────
                # 防止重复委派相同的任务给相同的Agent
                # 提取任务描述的关键信息进行去重
                if tc.name.startswith('handoff_to_'):
                    _task_desc = tc.input.get('task', '') if tc.input else ''
                    _target_agent = tc.name.replace('handoff_to_', '')

                    # Full envelope + canonical agent + graph/goal/evidence
                    # versions. Keywords alone conflate different temperatures,
                    # files, gases and repair stages; legacy keys are not reused.
                    _task_fingerprint = self._delegation_fingerprint(tc.name, tc.input or {})

                    # 检查是否已经委派过相同的任务
                    if not hasattr(self, '_delegated_tasks'):
                        self._delegated_tasks = set()

                    if _task_fingerprint in self._delegated_tasks:
                        print(f"  ⏭️  {tc.name} (skipped: duplicate delegation)", flush=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": (
                                f"[系统·委派去重] 相同任务已委派给 {_target_agent} Agent。\n"
                                "完整委派参数、目标/编排版本与工具证据均未变化。\n\n"
                                "**不要重复委派相同任务**。请选择：\n"
                                "1. 等待之前委派的结果返回\n"
                                "2. 委派一个**不同的**子任务\n"
                                "3. 直接输出最终报告\n\n"
                                "如果之前的委派失败了，请先分析失败原因，再决定是否重新委派。"
                            ),
                        })
                        consecutive_skips += 1
                        total_consecutive_skips += 1
                        continue
                    else:
                        # 记录本次委派
                        self._delegated_tasks.add(_task_fingerprint)

                # In force handoff mode, block all non-handoff tools
                # Planning/control calls must stay available while the legacy
                # delegation nudge is active. Derive this set from the same
                # source used by the DAG gate so newly added controls cannot be
                # omitted here and create a handoff↔planning deadlock.
                _control_tools = set(self._CHAIN_CONTROL_TOOLS) | {
                    'discover_forcefield', 'inspect_forcefield',
                }
                if _force_handoff_mode and not tc.name.startswith('handoff_to_') and tc.name not in _control_tools:
                    print(f"  ⏭️  {tc.name} (blocked: handoff required)", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": (
                            "⛔ You must delegate to a specialist agent first. "
                            "Call handoff_to_adsorption(task='...', context='...') now. "
                            "Example: handoff_to_adsorption(task='计算CuBTC CO2吸附', context='cif=CuBTC, gas=CO2')"
                        ),
                    })
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue
                # Interactive parameter asking for specific tools
                interaction_prompt = self._detect_user_interaction_needed(tc.name, tc.input)
                if interaction_prompt:
                    print(f"  ❓ Asking user for parameters: {tc.name}", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": interaction_prompt,
                    })
                    self._waiting_for_user_input = True
                    self._pending_user_interaction = {
                        "tool": tc.name,
                        "params": tc.input,
                        "prompt": interaction_prompt
                    }
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue
                # Anti-loop: count how many times this tool has been called
                # 配额窗口 = 当前委派段（_segment_tool_log_start 在每次 handoff 处
                # 重新打点）。只统计本段内的调用，历史轮次/历史委派的调用不叠加。
                # 用户红线："每次委派刷新配额"——run_bash/grep_search 等工具
                # 不能让 session 级别的历史调用把新委派段锁死。
                _segment_log = self.memory.tool_call_log[_segment_tool_log_start:]
                tool_call_count = sum(1 for t in _segment_log if t['tool'] == tc.name)

                # ── Total Tool Call Budget per Delegation Segment ─────────
                # Even general-purpose tools must have a hard cap to prevent
                # runaway loops. S01 test showed99 tool calls for a single
                # Henry calculation — most were redundant file reads.
                _segment_total_tools = len(_segment_log)
                # Early warning at 2/3 capacity
                if _segment_total_tools >= int(_MAX_TOTAL_TOOLS_PER_SEGMENT * 0.67):
                    print(f"  ⚠️ Tool budget approaching: {_segment_total_tools}/{_MAX_TOTAL_TOOLS_PER_SEGMENT}", flush=True)
                if _segment_total_tools >= _MAX_TOTAL_TOOLS_PER_SEGMENT:
                    print(f"  🛑 Total tool budget exhausted: {_segment_total_tools}/{_MAX_TOTAL_TOOLS_PER_SEGMENT} tools in this segment", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": (
                            f"[系统·工具预算耗尽] 本委派段已执行 {_segment_total_tools} 次工具调用，达到上限({_MAX_TOTAL_TOOLS_PER_SEGMENT})。\n"
                            "**立即停止调用所有工具**。基于已有结果输出报告：\n"
                            "1. 已完成的计算和获取的结果\n"
                            "2. 已提交的作业 job_id 和当前状态\n"
                            "3. 未完成项的原因说明\n"
                            "不要编造数据，如实报告。"
                        ),
                    })
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue

                # Hard limits per tool (专业工具统一 10 次/委派段)
                # 专业工具（计算/建库/ML 等）每个委派段最多 10 次，保留失败重试
                # 的可能（提交作业→查状态→失败→修正参数→重新提交…都在 10 次内），
                # 又不至于无限刷。防无限循环由轮次安全上限兜底
                # （max_rounds=45 / _TOTAL_ROUND_SAFETY=500 / ReAct nudge / 重复参数跳过）。
                limits = {
                    "check_job": 10, "inspect_path": 10, "inspect_run": 10,
                    "run_henry": 10, "run_gcmc_isotherm": 10,
                    "find_cif": 10, "extract_features": 10,
                    "run_pore_analysis": 10, "query_literature": 10,
                    "run_cdft": 10, "run_pacman_charge": 10,
                    "run_md_optimize": 10, "run_xtb_optimize": 10,
                    "build_guest_forcefield": 10, "run_vasp": 10,
                    "run_string_tst": 10, "run_external_potential": 10,
                    "calc_binding_energy": 10, "generate_structure": 10,
                    "ml_train": 10, "ml_predict": 10,
                    "ml_feature_importance": 10, "ml_active_learning": 10,
                    "run_gcmc_batch": 10, "submit_job": 10,
                }
                # 基础通用工具（read_file/write_file/run_bash/grep_search/inspect_path）
                # **不设配额上限**：这些是 agent 的"手"，多步科学流程（探索目录→读输入
                # →写脚本→监控作业）会合法调用很多次。防无限循环由轮次安全上限兜底
                # （max_rounds=45 / _TOTAL_ROUND_SAFETY=500 / ReAct nudge / 重复参数跳过），
                # 不需要靠调用次数配额去卡它。专业昂贵工具用上面统一的 10 次/段上限。
                if tc.name in self._GENERAL_LIMIT_EXEMPT:
                    max_calls = 10**9  # 基础工具不限配额
                    # 基础工具软上限：超过后提醒优化
                    if tc.name not in _general_tool_counts:
                        _general_tool_counts[tc.name] = 0
                    _general_tool_counts[tc.name] += 1
                    if _general_tool_counts[tc.name] > _GENERAL_TOOL_SOFT_LIMIT:
                        print(f"  ⚠️ {tc.name} 调用次数过多 ({_general_tool_counts[tc.name]}次)，提醒优化", flush=True)
                        # A warning is not a second tool_result for the same ID.
                        self._progress(agent_name=agent.name, reasoning='',
                            status='工具调用次数提醒', log=f'{tc.name}: {_general_tool_counts[tc.name]} calls; reduce redundant queries')
                else:
                    max_calls = limits.get(tc.name, 10)

                if tool_call_count >= max_calls:
                    # Build list of available (non-limited) tools for guidance
                    # (同样只统计当前委派段，保证提示与实际配额一致)
                    available = []
                    for fname, flimit in limits.items():
                        fcount = sum(1 for t in _segment_log if t['tool'] == fname)
                        if fcount < flimit and fname in func_map:
                            available.append(fname)
                    avail_str = ", ".join(available[:10]) if available else "none remaining"
                    msg = (
                        f"Tool '{tc.name}' has reached its call limit this delegation ({tool_call_count}/{max_calls}). "
                        f"DO NOT call '{tc.name}' again. "
                        f"Available tools with remaining calls: [{avail_str}]. "
                        f"Choose a different tool to continue the task."
                    )
                    print(f"  ⏭️  {tc.name} (skipped: limit {max_calls})", flush=True)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tc.id,
                        "content": msg,
                    })
                    consecutive_skips += 1
                    total_consecutive_skips += 1
                    continue

                # Skip exact duplicate params (same tool + same params = waste).
                # 只比较**当前委派段**内的调用：跨委派重复参数是合法的（新委派的
                # agent 可能要对同一文件/同一作业重新读取确认），不能拿整个 session
                # 的历史去卡它。基础通用工具（read/write/run_bash 等）完全豁免——
                # 同一路径前后两次读取（提交前/提交后）是合法流程，不因参数相同被拦。
                if tc.name in self._GENERAL_LIMIT_EXEMPT or is_submission(tc.name, tc.input or {}) or tc.name == 'diagnose_job':
                    pass  # 基础工具不查重复
                else:
                    call_key = f"{tc.name}:{json.dumps(tc.input, sort_keys=True, default=str)}"
                    all_keys = [
                        f"{t['tool']}:{json.dumps(t.get('params', {}), sort_keys=True, default=str)}"
                        for t in _segment_log
                    ]
                    if call_key in all_keys:
                        print(f"  ⏭️  {tc.name} (skipped: exact duplicate)", flush=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": f"Tool '{tc.name}' with these exact parameters was already called. Use different parameters or choose another tool.",
                        })
                        consecutive_skips += 1
                        total_consecutive_skips += 1
                        continue

                    # ── 重试参数变化验证 ──────────────────────────────────
                    # 如果同一工具已失败过，重试时必须改变关键参数
                    # 防止智能体用相同错误参数反复重试
                    _failed_calls = [t for t in _segment_log
                                    if t['tool'] == tc.name and t.get('failed', False)]
                    if _failed_calls:
                        _last_failed = _failed_calls[-1]
                        _last_params = _last_failed.get('params', {})
                        if not isinstance(_last_params, dict):
                            _last_params = {}
                        _current_params = tc.input or {}

                        # 提取关键参数进行比较
                        _key_params = ['cif', 'gas', 'temperature', 'pressure', 'forcefield',
                                      'unit_cells', 'n_cycles', 'molecule_definition']
                        _last_key = {k: _last_params.get(k) for k in _key_params if k in _last_params}
                        _current_key = {k: _current_params.get(k) for k in _key_params if k in _current_params}

                        # 如果关键参数完全相同，强制要求改变
                        if _last_key == _current_key and _last_key:
                            print(f"  ⏭️  {tc.name} (blocked: same params as failed call)", flush=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": (
                                    f"[系统·参数未变] 工具 '{tc.name}' 使用相同参数已失败过。\n"
                                    f"上次失败参数: {json.dumps(_last_key, default=str)}\n\n"
                                    "**重试前必须改变关键参数**。请选择以下之一：\n"
                                    "1. 改变力场（如 GenericMOFs → UFF）\n"
                                    "2. 改变气体定义（如 TraPPE → ExampleDefinitions）\n"
                                    "3. 改变单元格大小（如 2 2 2 → 3 3 3）\n"
                                    "4. 改变温度/压力\n"
                                    "5. 换一个完全不同的工具或方法\n\n"
                                    "如果无法改变参数，请如实报告失败原因，不要重复尝试。"
                                ),
                            })
                            consecutive_skips += 1
                            total_consecutive_skips += 1
                            continue

                # Skip if same tool called too many times with similar params.
                # NOTE: cheap general-purpose tools (file read/write, shell, grep,
                # inspect) are EXEMPT — multi-step scientific workflows (explore
                # dirs → read inputs → write scripts → monitor jobs) legitimately
                # need many calls. 专业工具的"相似参数"拦截上限对齐 limits 的
                # 10 次/段预算：第一次失败 → 修正参数重试 → 失败再修，都在 10 次
                # 内进行，不会因为第 2 次就"2 calls then stop"而失去重试机会。
                if tc.name not in self._GENERAL_LIMIT_EXEMPT:
                    same_tool_count = sum(1 for t in _segment_log if t['tool'] == tc.name)
                    if same_tool_count >= 10:
                        print(f"  ⏭️  {tc.name} (skipped: already called {same_tool_count} times)", flush=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": f"Tool '{tc.name}' has already been called {same_tool_count} times with different parameters this delegation. Retry budget exhausted. Choose a different tool or report the failure.",
                        })
                        consecutive_skips += 1
                        total_consecutive_skips += 1
                        continue

                # ── Polling loop detection ──────────────────────────────
                # If the agent keeps calling check_job / run_bash(squeue) in a
                # loop, inject a "wait then check once" instruction after
                # _MAX_POLLING_STREAK consecutive polling calls. Also enforce a
                # hard cap on total polling calls per delegation segment.
                if _is_polling_call(tc.name, tc.input):
                    _polling_streak += 1
                    _total_polling_calls += 1
                    if _total_polling_calls >= _MAX_TOTAL_POLLING:
                        print(f"  🔄 Polling hard cap: {_total_polling_calls} total polling calls → blocking", flush=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": (
                                f"[系统·轮询硬上限] 本委派段已执行 {_total_polling_calls} 次轮询调用，达到上限({_MAX_TOTAL_POLLING})。\n"
                                "**禁止再调用任何检查/监控工具**（check_job / run_bash squeue 等）。\n"
                                "请基于已有结果输出报告：\n"
                                "1. 已提交的作业 job_id 和当前状态\n"
                                "2. 已获取的计算结果\n"
                                "3. 未完成项的原因说明\n"
                                "不要编造数据，如实报告。"
                            ),
                        })
                        _polling_streak = 0
                        consecutive_skips += 1
                        total_consecutive_skips += 1
                        continue
                    if _polling_streak >= _MAX_POLLING_STREAK:
                        print(f"  🔄 Polling loop detected: {_polling_streak} consecutive polling calls → injecting wait instruction", flush=True)
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tc.id,
                            "content": (
                                f"[系统·轮询打断] 你已经连续 {_polling_streak} 次调用检查/监控工具"
                                "（check_job / run_bash squeue 等），这是典型的轮询循环。\n"
                                "**立即停止轮询**。不要再次调用 check_job 或 run_bash 来监控作业状态。\n"
                                "你有两个选择：\n"
                                "1. 如果你已知道 job_id，直接用 `check_job(job_id=..., work_dir=...)` 做**最后一次**确认，"
                                "然后基于已有结果输出报告；\n"
                                "2. 如果作业确实还在运行，先输出当前已有结果（提交了哪些作业、job_id、预期完成时间），"
                                "让用户决定是等待还是调整方案。**禁止继续轮询。**"
                            ),
                        })
                        _polling_streak = 0
                        consecutive_skips += 1
                        total_consecutive_skips += 1
                        continue
                else:
                    _polling_streak = 0  # Reset on non-polling tool call

                # Reset consecutive skip counter on successful execution
                consecutive_skips = 0
                total_consecutive_skips = 0

                print(f"  🔧 {tc.name}", flush=True)
                self._progress(
                    agent_name=agent.name,
                    reasoning="",
                    status=f"🔧 执行工具: {tc.name}",
                    log=f"执行工具: {tc.name}({json.dumps(tc.input, default=str)[:100]})"
                )

                # Use original func_map for handoff calls (in case agent changes mid-batch)
                _use_map = _original_func_map if tc.name.startswith('handoff_to_') else func_map
                func = _use_map.get(tc.name)
                session_bindings = {
                    'record_workflow_draft': self._record_workflow_draft,
                    'task_line_query': self._task_line_query,
                    'task_line_update': self._task_line_update,
                    'prepare_retry': self._prepare_retry, 'recovery_state': self._recovery_state,
                    'accept_recovered_result': self._accept_recovered_result,
                    'lifecycle_state': self._lifecycle_state,
                    'propose_workflow_patch': self._propose_workflow_patch,
                    'apply_workflow_patch': self._apply_workflow_patch,
                    'discard_workflow_patch': self._discard_workflow_patch,
                    'request_user_decision': self._request_user_decision,
                    'reconcile_watched_job': self._reconcile_watched_job,
                    'execute_workflow': self._on_workflow_start,
                    'retarget_queued_job': self._retarget_queued_job,
                    'cancel_watched_job': self._cancel_watched_job,
                    'message_workflow_node': self._on_workflow_message,
                    'resolve_local_workflow_write': self._on_workflow_resolve,
                    'revalidate_workflow_node_outputs': self._on_workflow_revalidate,
                    'finish_workflow_node': self._finish_workflow_node,
                }
                # Binding is not authorization. Only exposed capabilities can
                # be replaced with their session-aware implementation.
                if func is not None and tc.name in session_bindings:
                    func = session_bindings[tc.name]
                recovery_key = attempt_id = None
                if func is not None:
                    try:
                        from .control_policy import role_denial
                        denied = role_denial(agent.name, tc.name)
                        if denied:
                            tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                                'content': json.dumps({'blocked': True, 'executed': False, 'reason': denied})})
                            continue
                        if self._on_tool_guard:
                            guard = self._on_tool_guard(tc.name, dict(tc.input or {}))
                            if guard:
                                tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                                    'content': json.dumps({'blocked': True, 'reason': guard, 'executed': False})})
                                continue
                        recovery_key, attempt_id, recovery_block = self._claim_submission(tc.name, dict(tc.input or {}))
                        if recovery_block:
                            from .task_line import get_store
                            history_line = get_store().get_line(self._current_line_id) if self._current_line_id else {}
                            history_step = next((n for n in (history_line or {}).get('steps', []) if n['step_id'] == _node_step_id), {})
                            if self._current_line_id and 'expected_outputs' not in history_step:
                                get_store().upsert_step(self._current_line_id, _node_step_id, status='blocked', done=False,
                                    validation={'dispatch_not_entered': True, 'reason': recovery_block})
                            tool_results.append({'type': 'tool_result', 'tool_use_id': tc.id,
                                                 'content': json.dumps({'blocked': True, 'reason': recovery_block,
                                                                        'recovery_key': recovery_key}, ensure_ascii=False)})
                            self._progress(agent.name, '', '⛔ 重提门禁：先诊断与验证修复', recovery_block)
                            continue
                        self._checkpoint('before_tool_dispatch', tool_results)
                    except Exception as checkpoint_error:
                        # No tool is dispatched if its intent/checkpoint cannot be
                        # persisted. A reserved claim remains blocked on restart.
                        prompt = f'已停止执行：提交意图/检查点无法安全持久化：{checkpoint_error}。请修复存储后继续。'
                        self._waiting_for_user_input = True
                        self._pending_user_interaction = {'tool': 'failure_decision', 'params': {}, 'prompt': prompt}
                        break
                if func is None:
                    result_str = f"Error: tool '{tc.name}' not available for '{agent.name}'"
                else:
                    try:
                        # Timeout wrapper: prevent tools from blocking forever
                        import concurrent.futures
                        _tool_timeout = 180  # 3 minutes max per tool call
                        # Propagate thread-local watch context (username/conv_id)
                        # into the tool worker thread so SLURM submissions get
                        # auto-registered with JobWatch under the right conv.
                        try:
                            from .watch_context import get_context, set_context as _set_ctx
                            _outer_ctx = get_context()
                            _set_ctx(
                                username=_outer_ctx.get("username", ""),
                                conv_id=_outer_ctx.get("conv_id", ""),
                                agent_name=agent.name,
                                line_id=self._current_line_id or _outer_ctx.get("line_id", ""),
                                tool_name=tc.name, recovery_key=recovery_key or '', attempt_id=attempt_id or '',
                                recovery_path=str(self.recovery_gate.path or ''),
                                step_id=_node_step_id,
                            )
                            _watch_ctx = get_context()
                        except Exception:
                            _watch_ctx = None

                        def _run_tool_with_ctx(*args, **kwargs):
                            if _watch_ctx:
                                _set_ctx(**_watch_ctx)
                            return func(*args, **kwargs)

                        _pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
                        _future = _pool.submit(_run_tool_with_ctx, **tc.input)
                        try:
                            raw = _future.result(timeout=_tool_timeout)
                        except concurrent.futures.TimeoutError:
                            result_str = (
                                f"⚠️ Tool '{tc.name}' timed out after {_tool_timeout}s. "
                                "The orchestration turn has been released; any already-submitted "
                                "scheduler job continues independently and remains in JobWatch."
                            )
                            self.memory.record_error(f"{tc.name}: timeout after {_tool_timeout}s", agent_name=agent.name)
                            _future.cancel()
                            if recovery_key and attempt_id:
                                self.recovery_gate.outcome(recovery_key, attempt_id, {'error': result_str}, uncertain=True)
                                # Cancellation does not stop a running Python thread.
                                # Preserve the uncertain claim until that exact call
                                # returns; never create another concurrent submission.
                                def _late_outcome(f, key=recovery_key, aid=attempt_id, origin=self.goal_contract.version):
                                    try:
                                        self.recovery_gate.outcome(key, aid, f.result())
                                    except Exception as e:
                                        self.recovery_gate.outcome(key, aid, {'error': str(e)})
                                    self._emit_lifecycle_event('tool_outcome', {'recovery_key': key, 'attempt_id': aid, 'origin_goal_version': origin})
                                _future.add_done_callback(_late_outcome)
                            else:
                                def _late_tool(f, tool=tc.name, params=dict(tc.input or {}), origin=self.goal_contract.version):
                                    try:
                                        outcome = self._json_safe(f.result())
                                    except Exception as error:
                                        outcome = {'error': str(error)}
                                    self._emit_lifecycle_event('tool_outcome', {'tool': tool, 'arguments': params,
                                                                            'result': outcome, 'origin_goal_version': origin})
                                _future.add_done_callback(_late_tool)
                            self._active_tool_futures.append(_future)
                            # Context-manager shutdown(wait=True) used to negate the
                            # timeout and block until the hung call actually ended.
                            _pool.shutdown(wait=False, cancel_futures=True)
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tc.id,
                                "content": result_str,
                            })
                            self._waiting_for_user_input = True
                            self._pending_user_interaction = {'tool': 'lifecycle_wait', 'params': {},
                                                              'prompt': result_str + ' 当前调用未决，禁止重提；主chat和监督Agent将收到终态事件并自动继续，无需重复发送任务。'}
                            self._emit_lifecycle_event('tool_uncertain', {'tool': tc.name, 'arguments': dict(tc.input or {}), 'recovery_key': recovery_key})
                            self.memory.record_tool_call(agent.name, tc.name, dict(tc.input or {}), result_str, failed=True)
                            self._checkpoint('tool_outcome_uncertain', tool_results)
                            break
                        else:
                            _pool.shutdown(wait=False)
                        if isinstance(raw, AgentResult):
                            # Track handoff chain within batch
                            if raw.agent:
                                # Record handoff from current effective agent to target
                                _prev_agent = _effective_agent.name
                                _effective_agent = raw.agent  # Update effective agent FIRST
                                self.memory.record_handoff(_prev_agent, raw.agent.name)
                                self._pending_handoff = raw.agent
                                # ── 监管清单：记录本次委派 ──
                                _turn_delegations.append({
                                    "agent": raw.agent.name,
                                    "task": str(tc.input.get("task", ""))[:150],
                                    "status": "pending",
                                })
                                # Emit partial result: the sub-agent produced output BEFORE delegating
                                if raw.value and len(str(raw.value)) > 20:
                                    self._report_partial(_effective_agent.name, str(raw.value))
                                print(f"  ✈️  → {raw.agent.name} (pending)", flush=True)
                            self._progress(
                                agent_name=agent.name,
                                reasoning="",
                                status=f"✈️ 委派给: {raw.agent.name}",
                                log=f"任务委派: {agent.name} → {raw.agent.name}"
                            )
                            self.context.update(raw.context_variables)
                            result_str = raw.value
                        elif isinstance(raw, Agent):
                            _prev_agent = _effective_agent.name
                            _effective_agent = raw
                            self.memory.record_handoff(_prev_agent, raw.name)
                            self._pending_handoff = raw
                            # ── 监管清单：记录本次委派 ──
                            _turn_delegations.append({
                                "agent": raw.name,
                                "task": str(tc.input.get("task", ""))[:150],
                                "status": "pending",
                            })
                            print(f"  ✈️  → {raw.name} (pending)", flush=True)
                            result_str = json.dumps({"handed_off_to": raw.name})
                        else:
                            result_str = (
                                raw if isinstance(raw, str)
                                else json.dumps(raw, ensure_ascii=False, default=str)
                            )
                    except Exception as e:
                        result_str = json.dumps(e.details if hasattr(e, 'details') else {
                            'error': str(e), 'exception_type': type(e).__name__}, ensure_ascii=False, default=str)
                        self.memory.record_error(f"{tc.name}: {e}", agent_name=agent.name)
                
                if len(result_str) > 50000:
                    result_str = result_str[:50000] + "\n... (truncated)"

                # Record structured parameters + failure bit in memory.  Older
                # code stored a truncated string, making parameter-aware retry
                # checks impossible and causing identical bad calls to repeat.
                _call_failed = bool(failure_reason(result_str)) and tc.name not in self._DIAGNOSTIC_TOOLS
                result_facts = result_object(result_str)
                not_executed = bool(result_facts.get('blocked') or result_facts.get('skipped') or result_facts.get('executed') is False)
                if recovery_key and attempt_id:
                    self.recovery_gate.outcome(recovery_key, attempt_id, result_str)
                self.memory.record_tool_call(
                    agent.name, tc.name, dict(tc.input or {}), result_str,
                    failed=_call_failed,
                )
                # The model needs the *persisted* evidence identity, not the
                # provider's tool_use ID or a guessed recovery key. Keep the
                # original result in evidence; metadata is only the wire reply.
                evidence_call = self.memory.tool_call_log[-1]
                try:
                    wire_result = json.loads(result_str)
                except (ValueError, TypeError):
                    wire_result = {'result': result_str}
                if not isinstance(wire_result, dict):
                    wire_result = {'result': wire_result}
                wire_result['evidence_ref'] = {'call_id': evidence_call['call_id'],
                                             'time': evidence_call['time'], 'tool': tc.name}
                if recovery_key:
                    self.memory.tool_call_log[-1]['recovery_key'] = recovery_key
                    self.memory.tool_call_log[-1]['workflow_step_id'] = _node_step_id
                if not _call_failed and not not_executed:
                    self._resolve_error_branches_for_tool(tc.name, result_str, dict(tc.input or {}))
                if self._current_line_id and not tc.name.startswith("handoff_to_") and not _is_coordination_tool and not (not_executed and not _node_for_call):
                    try:
                        from .task_line import get_store as _get_tl
                        try:
                            _result_obj = json.loads(result_str) if isinstance(result_str, str) else result_str
                        except Exception:
                            _result_obj = {}
                        _result_obj = _result_obj if isinstance(_result_obj, dict) else {}
                        _result_obj = result_object(_result_obj)
                        _jid = _result_obj.get("job_id")
                        _jids = list(_result_obj.get("job_ids") or ([] if not _jid else [_jid]))
                        _ctx_jobs = _result_obj.get('context') or {}
                        _jids = _jids or _ctx_jobs.get('gcmc_job_ids') or ([str(_ctx_jobs['henry_job_id'])] if _ctx_jobs.get('henry_job_id') else []) or ([str(_ctx_jobs['charge_job_id'])] if _ctx_jobs.get('charge_job_id') else [])
                        _submitted = bool(_result_obj.get("submitted") or _jids or _result_obj.get('chain_status') in {'waiting', 'submitted'})
                        _get_tl().upsert_step(
                            self._current_line_id,
                            _node_step_id,
                            tool=tc.name,
                            status="blocked" if not_executed else "failed" if _call_failed else ("submitted" if _submitted else "completed"),
                            done=not _call_failed and not _submitted and not not_executed,
                            job_ids=[str(x) for x in _jids if x],
                            output_dir=str(_result_obj.get("output_dir") or _result_obj.get("work_dir") or ""),
                            output_files=list(_result_obj.get("output_files") or _result_obj.get("charged_cifs") or []),
                            arguments=dict(tc.input or {}),
                            validation={
                                "goal_contract": "passed",
                                "execution": "failed" if _call_failed else "passed",
                                'chain_status': _result_obj.get('chain_status'),
                            },
                            plan_version=self.goal_contract.approved_plan_version,
                        )
                    except Exception:
                        pass
                # 用户红线：每次工具调用都重置段内轮次计数 —— 只要 Agent 持续有
                # 实质进展（工具执行/委派），轮次预算就从当前调用重新起算，长任务
                # 不会被历史调用数叠加打爆；只有"连续 max_rounds 轮无工具进展"
                # （卡死/空转）才会触发段内上限。

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": tc.id,
                    "content": json.dumps(wire_result, ensure_ascii=False, default=str),
                })
                self._checkpoint('after_tool_result', tool_results)

                # 用户中断（每执行完一个工具立即检查）：长工具调用无法抢占，
                # 但一旦返回就立刻停止本批剩余工具并交给轮顶冻结——比整批跑完
                # 再停更快，中断体验更跟手。reconciliation 会给剩余 tool_use 补
                # 合成结果保持配对有效，随后轮顶检查构建冻结报告。
                if self._interrupt_requested():
                    print(f"  ⛔ 用户中断（工具 {tc.name} 完成后）—— 跳过本批剩余工具，冻结", flush=True)
                    break

            # ── Reconciliation: EVERY tool_use in the last assistant message MUST
            # have a matching tool_result, or the API rejects the history (400).
            # Any tool_use that slipped through (batch dedup, partial failures,
            # skipped paths) gets a synthetic tool_result appended here.
            try:
                _last_msg = self.messages[-1]
                _last_content = _last_msg.get("content") if isinstance(_last_msg, dict) else None
                if _last_msg.get("role") == "assistant" and isinstance(_last_content, list):
                    _have_ids = set()
                    for tr in tool_results:
                        _tid = tr.get("tool_use_id")
                        if _tid:
                            _have_ids.add(_tid)
                    for b in _last_content:
                        btype = b.get("type") if isinstance(b, dict) else getattr(b, "type", None)
                        if btype == "tool_use":
                            bid = b.get("id") if isinstance(b, dict) else getattr(b, "id", None)
                            if bid and bid not in _have_ids:
                                # Guarantee a synthetic result for this orphan tool_use
                                bname = b.get("name") if isinstance(b, dict) else getattr(b, "name", "tool")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": bid,
                                    "content": json.dumps(_batch_skipped[bid], ensure_ascii=False) if bid in _batch_skipped else (
                                        f"⏭️ 工具 '{bname}' 在本轮中已被跳过（重复调用或超出限制）。"
                                        "请使用之前调用的结果，或选择其他工具。"
                                    ),
                                })
                                _have_ids.add(bid)
                                print(f"  🩹 reconciliation: added synthetic tool_result for {bid} ({bname})", flush=True)
            except Exception as _recon_e:
                print(f"  ⚠️ reconciliation pass failed: {_recon_e}", flush=True)

            self.messages.append({"role": "user", "content": tool_results})

            # Count new evidence/real execution changes, not fresh call IDs.
            # Guard failures and schema rejections participate too, so their
            # actual reasons reach the next model's reflection context.
            runtime = self._runtime_snapshot() if self._runtime_snapshot else {}
            execution_state = {'plan_version': runtime.get('plan_version'), 'status': runtime.get('status'),
                'user_paused': runtime.get('user_paused'), 'nodes': {k: {field: n.get(field) for field in
                    ('status', 'phase', 'job_ids', 'contract', 'error')} for k, n in runtime.get('nodes', {}).items()}}
            replies = {r.get('tool_use_id'): r.get('content') for r in tool_results if r.get('type') == 'tool_result'}
            for call in tool_calls:
                if call.id not in replies: continue
                receipt = reflection_progress.observe(call.name, call.input, replies[call.id], execution_state)
                if receipt['progress']:
                    _segment_rounds = 0
                    self.context.pop('reflection_required', None)
                elif receipt['stalled_calls'] >= 3:
                    self.context['reflection_required'] = receipt
            if reflection_progress.stalled >= 8:
                receipt = self.context.get('reflection_required', {})
                from .reflection_progress import digest
                key = digest([execution_state, receipt.get('last_problem')])
                escalations = self.context.setdefault('reflection_escalations', [])
                if key not in escalations:
                    escalations.append(key)
                    self._emit_lifecycle_event('reflection_stalled', {'actual_failure': receipt, 'task_complete': False,
                        'instruction': 'Supervisor reviews actual state/error; main chooses result restoration or a verified contract repair. No blind resubmission, no automatic repeat approval.'})
                # This used to return to the user despite saying "handoff".
                # Perform a real internal handoff and keep the original DAG.
                from .defns import ORCHESTRATOR, PATCHER
                target = PATCHER if agent.name != 'patcher' else ORCHESTRATOR
                if not self._pending_handoff:
                    self._pending_handoff = target
                    self.context['automatic_handoff_task'] = (
                        ('修复反复阻塞当前DAG的工具/API/框架问题；不要运行用户科研任务。' if target is PATCHER else
                         'patcher未能在预算内修复；依据已有证据选择同一科研条件下的其他恢复路径，不要询问用户处理内部错误。')
                        + '实际停滞证据=' + json.dumps(receipt, ensure_ascii=False, default=str)[:6000]
                    )
                reflection_progress.stalled = 0
                self.context.pop('reflection_required', None)
                self._checkpoint('reflection_stalled_auto_handoff')

            # Completed control/acceptance operation hands continuation to the
            # durable executor. No extra model round just to watch it move.
            control_delivery=self.context.pop('pending_control_delivery',None)
            if control_delivery and agent.name=='lead-orchestrator':
                self.task_complete=False
                self.last_text=control_delivery
                self._checkpoint('delivered_operation_receipt')
                return self.last_text

            # ── 快速委派提醒 ─────────────────────────────────────────────
            # 调度器如果连续多轮只调用基础工具不委派，提醒它尽快委派
            if (agent.name == "lead-orchestrator"
                    and not (self._runtime_snapshot and self._runtime_snapshot().get('status') in {'active','needs_user'})
                    and (self.goal_contract.approved_nodes or any(
                        is_submission(c.get('tool'), c.get('params', {})) for c in self.memory.tool_call_log))
                    and any(f.__name__.startswith('handoff_to_') for f in agent.functions)):
                _had_handoff_in_batch = any(
                    tc.name.startswith('handoff_to_') for tc in tool_calls
                )
                if _had_handoff_in_batch:
                    _orchestrator_no_handoff_rounds = 0
                else:
                    _orchestrator_no_handoff_rounds += 1
                    if _orchestrator_no_handoff_rounds >= _MAX_NO_HANDOFF_ROUNDS:
                        print(f"  ⚡ 调度器已 {_orchestrator_no_handoff_rounds} 轮未委派，提醒尽快委派", flush=True)
                        self.messages.append({"role": "user", "content":
                            "[系统·快速委派] 你已连续多轮未委派专业Agent。"
                            "请立即调用 handoff_to_* 委派计算/分析任务。\n"
                            "例如：handoff_to_adsorption(task='计算Henry系数')"
                        })
                        _orchestrator_no_handoff_rounds = 0

            # ── FAULT HANDLING: detect failed/cancelled computation jobs and
            # force the model to diagnose + recover instead of reporting success ──
            try:
                _assistant_content = response.content
                self._handle_failures(tool_results, _assistant_content, agent.name)
            except Exception as _fh_e:
                print(f"  ⚠️ fault-handling pass failed: {_fh_e}", flush=True)

            # ── ReAct DISCIPLINE: if the model acted for N rounds with no
            # reasoning text, inject a think-before-act nudge. (Failure recovery
            # messages above already demand <reflection>+<plan>; a nudge on top
            # is harmless and only triggers on the silent-toolchain pattern.)
            try:
                self._react_nudge(_assistant_content, agent.name)
            except Exception as _rn_e:
                print(f"  ⚠️ react-guard pass failed: {_rn_e}", flush=True)

            self._checkpoint('after_batch_recovery')

            # If waiting for user input, break out of loop to wait for reply()
            if self._waiting_for_user_input:
                break

            # Apply pending handoff chain (deferred from tool execution)
            if hasattr(self, '_pending_handoff') and self._pending_handoff:
                # Record the FULL handoff chain from original agent to final target
                original_agent_name = agent.name if agent else "unknown"
                self.memory.record_handoff(original_agent_name, self._pending_handoff.name)
                agent = self._pending_handoff
                self.current_agent = agent
                func_map = agent.function_map()
                self._pending_handoff = None
                _segment_rounds = 0  # 新委派段从 0 起算，不叠加调度器的轮次
                _segment_tool_log_start = len(self.memory.tool_call_log)  # 新委派段配额从0起算
                _polling_streak = 0; _total_polling_calls = 0  # Reset polling counters

                # Inject task context for the new agent
                handoff_task = self.context.pop('automatic_handoff_task', '')
                for tc in tool_calls:
                    if tc.name.startswith('handoff_to_') and hasattr(tc, 'input'):
                        if 'task' in tc.input:
                            handoff_task = tc.input['task']
                            break
                # ⚠️ 任务线注入：接管 agent 必须看到当前会话 TaskLine 的真实状态
                # （每步 tool/job_ids/产物路径/done），否则它只拿到调度器手写的
                # handoff_task 文本，不知道上游产物在哪、哪步已完成哪步未完成，
                # 只能全局 grep 撞历史文件（CO2/N2 误判就是这种丢失任务的体现）。
                # 把 TaskLine 摘要直接附进委派上下文，让接管 agent 拿着路径继续。
                _tl_digest = ""
                _line_id = ""
                try:
                    from .watch_context import get_context as _gctx2
                    from .task_line import get_store as _tlstore
                    _line_id = _gctx2().get("line_id") or _gctx2().get("conv_id", "")
                    if _line_id:
                        _tl_digest = _tlstore().summarize(_line_id)
                except Exception:
                    _tl_digest = ""
                _taskline_note = (
                    f"\n\n[任务线 {_line_id or '当前会话'}]\n{_tl_digest}\n"
                    "（以上是当前会话流水线各步骤的真实产物路径与完成状态。"
                    "上游产物请直接用任务线给出的路径，禁止全局 grep 撞历史文件。"
                    "任务线显示 ⏳未完成 的步骤正是你本次委派需要推进的部分。）"
                ) if _tl_digest else ""
                self.messages.append({"role": "user", "content":
                    f"[系统] 任务已委派给 {agent.name}。委派任务：{handoff_task}"
                    f"{_taskline_note}\n"
                    f"你现在的角色是 {agent.name}，请使用你的专业工具完成上述任务。"
                    "完成后不要调用任何工具，直接输出文本结果返回给调度器。"
                })
                if self._current_line_id:
                    from .task_line import get_store
                    from .defns import resolve_agent
                    line = get_store().get_line(self._current_line_id) or {}
                    self.context['delegation_envelope'] = {
                        'workflow_id': self._current_line_id, 'plan_version': self.goal_contract.approved_plan_version,
                        'goal_version': self.goal_contract.version, 'agent': agent.name,
                        'nodes': [n for n in line.get('steps', []) if n.get('agent') and resolve_agent(n['agent']).name == agent.name],
                    }

            # No forced handoff reminders — let the agent decide

            # Force break if too many consecutive skips (model stuck in loop)
            if total_consecutive_skips >= max_consecutive_skips:
                self.messages.append({"role": "user", "content":
                    "[系统强制要求] 你连续调用了多次已达到限制的工具，全部被跳过。"
                    f"已经跳过了 {total_consecutive_skips} 次。"
                    "这是浪费资源的行为。请立即停止重试这些工具，"
                    "改用其他可用工具完成任务，或者输出最终报告。"
                })
                total_consecutive_skips = 0  # Reset after force break

            # 委派段软提醒：当前段轮次接近上限时，提醒尽快收尾（硬上限由循环顶部的
            # 段内耗尽处理负责 —— 子Agent 交回调度器、调度器走强制报告）。
            _agent_budget = 24
            if _segment_rounds > _agent_budget:
                self.messages.append({"role": "user", "content":
                    f"[系统] 当前委派段({agent.name})已运行 {_segment_rounds} 轮，接近上限({max_rounds})。"
                    "尽快输出当前结果；不要再调用新的工具。"
                })

        # ── USER INTERACTION: agent asked the user a question (missing params /
        # method ambiguity). Return the question to the frontend now and WAIT for
        # the user's reply — do NOT force a final report. reply() resumes.
        if self._waiting_for_user_input and self._pending_user_interaction:
            prompt = self._pending_user_interaction.get("prompt", "")
            self.task_complete = True
            self.last_text = prompt
            print(f"  ❓ Waiting for user input: {prompt[:80]}...", flush=True)
            return prompt

        # ── 完成不了判定（用户红线）─────────────────────────────────────
        # "完成不了"只有在 **超过轮次 + patcher 修补后仍不行** 时才成立。
        # 若轮次已尽但从未委派过 patcher 修补，则不能宣判"完成不了"——
        # 给一次追加轮次去委派 patcher（重置本轮预算，让调度器有空间执行委派）。
        # 若已给过追加仍没委派 patcher，或已委派过 patcher 仍失败，才允许
        # 按"完成不了 / 待修补"输出。
        _patcher_tried = any(
            'handoff_to_patcher' in t.get('tool', '')
            for t in self.memory.tool_call_log
        )
        _patcher_prompted = getattr(self, "_patcher_prompted", False)
        if not _patcher_tried and not _patcher_prompted:
            self._patcher_prompted = True
            self.messages.append({"role": "user", "content":
                "[系统·完成不了判定] 轮次已到上限，但你**从未委派过 patcher 修补**。"
                "按规则：'完成不了'只有在**超过轮次 + patcher 修补后仍不行**时才成立。"
                "因此现在不许宣告完成不了，系统已为你追加一轮，请：\n"
                "1. 判断当前阻塞问题是否属于工具/框架缺陷（工具行为不符、输入文件错误、"
                "静默失败、结果文件缺失、cDFT/GCMC 工具产出错误等）→ "
                "**立即 handoff_to_patcher 委派修补**；\n"
                "2. 若属于任务参数/数据问题 → 提出备选方案继续执行，或询问用户调整；\n"
                "3. patcher 修补后问题解决 → 继续推进整体方案剩余步骤；"
                "若修补后仍无法解决，才允许如实标注'该步骤完成不了'并说明原因与修补尝试。"
            })
            print(f"  🛠️ 完成不了判定：轮次尽但未委派 patcher → 追加一轮委派 patcher", flush=True)
            return self._execute_loop(self.current_agent, max_rounds=max_rounds)

        # Max rounds reached — force final report generation. But first gate on
        # unfinished jobs: if computations are still running/failed, tell the
        # model to report honestly (not fabricate) rather than claiming success.
        # 到这里 patcher 已被委派过（或已给过追加仍未委派），此时才能宣判。
        if _patcher_tried:
            _complete_note = (
                "已委派过 patcher 修补仍未能解决的部分，可如实标注'完成不了'，"
                "并说明失败原因与已尝试的修补路径。"
            )
        else:
            _complete_note = (
                "注意：你仍未委派 patcher 修补，未完成项只能标注为'待修补'，"
                "**不得宣判'完成不了'**；请如实说明剩余问题属于框架缺陷、需委派 patcher 处理。"
            )
        _gate = self._gate_jobs_before_report()
        if not _gate["ok"]:
            self.messages.append({"role": "user", "content":
                _gate["message"]
                + "\n[系统] 已达到最大轮次限制，但仍有关键计算未完成。"
                  "请如实输出：哪些作业完成、哪些未完成，附已有真实数据；"
                  "明确标注未完成项，不要编造缺失的结果。\n"
                  + _complete_note
            })
        else:
            self.messages.append({"role": "user", "content":
                "[系统] 已达到最大轮次限制。你必须立即停止所有工具调用，"
                "基于已有的所有计算结果和分析，输出一份完整的研究报告。\n\n"
                "**报告质量要求（必须满足）：**\n"
                "1. **数据表格**：必须包含结构化数据表（用Markdown表格格式）\n"
                "2. **定量结论**：必须有明确的数值评判（如：'选择性为X，优于/劣于Y'）\n"
                "3. **综合评判**：必须基于计算结果给出明确的推荐/不推荐结论\n"
                "4. **与文献对比**：如有文献数据，必须与计算值对比并分析偏差\n"
                "5. **应用场景**：必须说明该材料适合/不适合什么应用场景\n\n"
                "不要省略任何已有数据。\n"
                + _complete_note
            })
        try:
            response = self._call_api(agent)
            final_text = ""
            for block in response.content:
                if hasattr(block, "text"):
                    final_text += block.text
            self.task_complete = True
            self.last_text = final_text
            print(f"  📝 Forced report at max rounds ({len(final_text)} chars)", flush=True)
            return final_text
        except Exception as e:
            return f"(max rounds ({max_rounds}) reached, report generation failed: {e})"
    
    def start(self, user_message: str, agent: Agent | None = None, max_rounds: int = 45, verbose: bool = True) -> str:
        """启动会话，agent 自主执行直到需要用户输入或完成。"""
        self.current_agent = agent or self.current_agent
        self.messages = [{"role": "user", "content": user_message}]
        self.task_complete = False
        self.context = {}
        self.context['scientific_review_question'] = user_message
        self.context['scientific_tool_start'] = len(self.memory.tool_call_log)
        self.goal_contract = GoalContract.from_user_message(user_message)
        self._set_control_source(user_message)
        self._awaiting_plan_approval = False
        self._pending_plan_text = ""
        self._plan_steps_created = False
        from .watch_context import get_context
        self._current_line_id = get_context().get('line_id', '')
        # 轮次独立计数在 _execute_loop 内实现：每次 start()/reply() 以及每次委派
        # (handoff) 都从 0 起算，max_rounds 只约束当前委派段，二次委派不叠加。

        # Smart parameter completeness check
        # NOTE: check the EFFECTIVE agent. api.py calls start(query) WITHOUT
        # passing `agent` (agent=None) — a bare `if agent and ...` makes the
        # hard gate dead code in production and the agent silently self-supplies
        # missing thermodynamic params instead of asking the user ("没有AI问询").
        # self.current_agent was set above (agent or self.current_agent), so this
        # fires for API turns AND explicit agent=... calls, while an explicit
        # non-orchestrator (e.g. monitor in /explain) correctly skips the gate.
        _gate_agent = agent or self.current_agent
        if _gate_agent and _gate_agent.name == "lead-orchestrator":
            missing = self._detect_missing_params(user_message) if self.goal_contract.execution_mode == 'workflow' else ''
            if missing:
                # This detector is advisory only.  It must never decide task
                # intent before the model has understood the request: terms
                # such as GCMC/cDFT may occur in a force-field audit or schema
                # report that submits no compute. Typed DAG-node validation is
                # the authoritative gate immediately before any submission.
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[结构化参数提示] 词法扫描发现这些条件可能未固定：{missing}。"
                        "先根据用户的真实目标判断是否会创建计算提交节点。若只是只读核验、schema审计或报告，"
                        "不要询问无关计算条件，直接编译只读DAG；若确实要提交计算，缺失科学条件只能向用户一次性询问，"
                        "不得自行补默认值。最终以具体DAG节点和工具schema校验为准。"
                    )
                })
        
        if verbose:
            print(f"\n🤖 [{self.current_agent.name}]", flush=True)

        return self._execute_loop(self.current_agent, max_rounds=max_rounds)

    def reply(self, user_message: str, max_rounds: int = 45, verbose: bool = True) -> str:
        """用户回复，继续执行。

        Features:
        - Handles responses to interaction requests
        - Records user preferences
        - Resumes from interruption point
        - Consumes a pending interrupt redirect (Claude-Code-style Esc + 需求):
          if the user interrupted the last turn with an attached message, that
          message is injected as the redirect context for this delegation.
        """
        # 轮次独立计数：本次 reply() 是一个新的调用段，_execute_loop 内从 0 起算。
        # ── Consume pending interrupt/freeze ───────────────────────────
        # A freeze is "awaiting a subsequent delegation": the user's new message
        # IS that delegation, so the freeze is released here. If the interrupt
        # carried an attached redirect (Claude-Code Esc + 需求), it is stored and
        # merged into this delegation's user message below (so we never create
        # two consecutive user turns).
        _redirect = self._interrupt_message()
        if _redirect:
            print(f"  🔀 用户中断附带重定向需求: {_redirect[:80]}...", flush=True)
        # Whether or not there was a redirect, the freeze is now consumed —
        # clear it so the new delegation runs normally (not instantly re-freeze).
        _clear = getattr(self, "_on_interrupt_clear", None)
        if _clear:
            try:
                _clear()
            except Exception:
                pass

        # Update the durable goal contract BEFORE any LLM/tool call. Automated
        # JobWatch messages are ignored by GoalContract; genuine user changes
        # create a new objective version and immediately constrain all agents.
        _goal_text = user_message + ("\n" + _redirect if _redirect else "")
        self.context['scientific_review_question'] = _goal_text
        self.context['scientific_tool_start'] = len(self.memory.tool_call_log)
        self.context.pop('active_scientific_review_scope', None)
        _before_constraints = (
            tuple(self.goal_contract.gases), self.goal_contract.method,
            json.dumps(self.goal_contract.parameters, sort_keys=True, default=str),
            self.goal_contract.active_goal,
        )
        self.goal_contract.apply_user_message(_goal_text)
        self._set_control_source(_goal_text)
        # A frontend user addresses the lifetime main chat, even if the last
        # interrupted exploration left a specialist as current_agent. Worker
        # Sessions do not own this orchestration callback.
        if self._on_workflow_start:
            from .defns import ORCHESTRATOR
            self.current_agent = ORCHESTRATOR
        from .goal_contract import _is_automated_message
        if not _is_automated_message(_goal_text):
            self.context['last_authenticated_user_input'] = {'text': _goal_text, 'goal_version': self.goal_contract.version, 'time': time.time()}
        _after_constraints = (
            tuple(self.goal_contract.gases), self.goal_contract.method,
            json.dumps(self.goal_contract.parameters, sort_keys=True, default=str),
            self.goal_contract.active_goal,
        )
        _scope_changed = _before_constraints != _after_constraints

        # A pending structured patch is not approved/refused by parsing words.
        # The normal main-agent turn reads the real answer and, if appropriate,
        # chooses apply_workflow_patch or proposes a revision. Keep the draft.

        # Explicit plan approval gate. A multi-step plan presented in the prior
        # turn cannot execute until the user approves it. A non-approval reply is
        # treated as a revision request and causes a fresh plan version.
        if self._awaiting_plan_approval:
            _approval_words = (
                "确认", "同意", "批准", "执行", "开始", "可以", "approve", "approved", "go ahead",
            )
            _approved = any(w in _goal_text.lower() for w in _approval_words) and not _scope_changed
            if _approved:
                self.goal_contract.approve_pending_plan()
                self._awaiting_plan_approval = False
                approved_plan = self._pending_plan_text
                self._pending_plan_text = ""
                self.messages.append({
                    "role": "user",
                    "content": (
                        f"[系统·用户已批准方案 v{self.goal_contract.approved_plan_version}]\n"
                        f"批准的方案：\n{approved_plan[:6000]}\n\n"
                        "严格按该版本执行；任何方法、气体或范围变更必须重新征得用户确认。"
                    ),
                })
            else:
                self._awaiting_plan_approval = False
                self._pending_plan_text = ""
                self.goal_contract.execution_authorized = False
                self._plan_steps_created = False
                self._current_line_id = ""

        # ── Hard gate for a pending param question (reply side) ────────
        # If start() returned a param question (_pending_param_question set),
        # verify the user's reply actually supplies the missing thermodynamic
        # params / method. "确认，跑吧" or "好的" is NOT a valid answer — the
        # agent would self-supply 298K and submit. Re-ask until the params are
        # really provided (or the user explicitly authorizes "你推荐/你自己定").
        _pq = getattr(self, "_pending_param_question", "")
        if _pq and self.goal_contract.execution_mode in {'read_only', 'prepare_only'}:
            self._pending_param_question = ""
            _pq = ""
        if _pq:
            _auth_words = ["你自己定", "你定", "你来定", "你推荐", "推荐一下",
                           "自己决定", "你决定", "自行决定"]
            # The user's reply may BE the interrupt redirect message (e.g.
            # "温度用300K，压力0.1-2bar，确认用GCMC直接跑") — or the redirect may
            # carry the params while the follow-up message is just "确认，跑吧".
            # Merge both so a redirect that fully specifies params is honored.
            _check_text = user_message + (" " + _redirect if _redirect else "")
            _authorized = any(w in _check_text for w in _auth_words)
            # Does this reply carry any concrete value for each missing part?
            _still_missing = []
            for part in _pq.split("、"):
                if not part:
                    continue
                if self._contract_has_parameter(part):
                    continue
                if "计算方法" in part and not any(k in _check_text.upper()
                                                  for k in ["GCMC", "CDFT", "蒙特卡洛", "密度泛函"]):
                    _still_missing.append(part)
                elif "温度" in part and not any(t in _check_text for t in ["298", "300", "273", "350", "400", "K", "k"]):
                    _still_missing.append(part)
                elif "压力" in part and not any(p in _check_text for p in ["bar", "Pa", "atm", "0.1", "1.0", "5.0", "10", "MPa"]):
                    _still_missing.append(part)
                elif "气体" in part and not any(g in _check_text.upper()
                                                for g in ["CO2", "CH4", "N2", "H2", "CO", "SO2", "NH3",
                                                          "C2H4", "C2H6", "乙烯", "乙烷", "氩", "氦"]):
                    _still_missing.append(part)
                elif "材料" in part and not any(m in _check_text for m in
                                                ["MOF", "COF", "CIF", "cif", "Ni-MOF", "Mg-MOF", "Co-MOF",
                                                 "ZIF", "UiO", "MIL", "HKUST", "CuBTC", "Cu-BTC", "MFI", "ZSM-5"]):
                    _still_missing.append(part)
            if _still_missing and not _authorized:
                self.task_complete = True
                self.last_text = self._build_param_question("、".join(_still_missing), user_message)
                return self.last_text
            # Either fully provided, or user authorized self-decision → proceed.
            self._pending_param_question = ""

        # Check if this is a response to an interaction request
        if self._waiting_for_user_input and self._pending_user_interaction:
            tool_info = self._pending_user_interaction
            tool_name = tool_info["tool"]
            if tool_name == 'user_decision':
                decision = {**tool_info['params'], 'answer': user_message, 'source': 'authenticated_user', 'time': time.time(),
                            'goal_version': self.goal_contract.version}
                self.memory.record_user_preference('decision:' + decision['decision_id'], decision)

            if tool_name == "failure_decision":
                branch_id = tool_info.get("params", {}).get("branch_id", "")
                if branch_id in self._error_branches:
                    self._error_branches[branch_id]["status"] = "open"
                    self._update_error_branch(
                        branch_id,
                        fix_attempt=f"用户确认的恢复方向: {user_message}",
                    )

            # A reply to a budget/failure/general question is not a scientific
            # method. Preserve the answer for the model to interpret and choose
            # actual tool parameters; never promote arbitrary prose into method.
            parsed = {'answer': user_message}

            # Record user preference
            self.memory.record_user_preference(tool_name, parsed)
            print(f"  📝 User choice recorded: {parsed}", flush=True)

            # If the user chose an adsorption/electronic method, remember it so we
            # never re-ask the GCMC-vs-cDFT question in this session.
            self._method_choice_made = bool(self.goal_contract.method)

            # Update the pending tool with user's choice
            self._pending_user_interaction["params"].update(parsed)

            # Clear waiting state
            self._waiting_for_user_input = False
            self._pending_user_interaction = None

            # Add user response to messages — merge into the existing tool_result
            # user message (content is a list) so the API doesn't see two
            # consecutive user turns, which would be rejected.
            if _redirect:
                user_message = f"[用户中断后重定向需求] {_redirect}\n\n[用户回复] {user_message}"
            if self.messages and self.messages[-1].get("role") == "user" and isinstance(self.messages[-1].get("content"), list):
                self.messages[-1]["content"] = self.messages[-1]["content"] + [
                    {"type": "text", "text": user_message}
                ]
            else:
                self.messages.append({"role": "user", "content": user_message})
            self.task_complete = False

            if verbose:
                print(f"\n🤖 [{self.current_agent.name}]", flush=True)

            return self._execute_loop(self.current_agent, max_rounds=max_rounds)

        # Regular user response
        if _redirect:
            user_message = f"[用户中断后重定向需求] {_redirect}\n\n[用户后续委派] {user_message}"
        # Keep Anthropic role alternation valid when the plan-approval branch
        # already injected a system user message above.
        if (self.messages and self.messages[-1].get("role") == "user"
                and isinstance(self.messages[-1].get("content"), str)):
            self.messages[-1]["content"] += f"\n\n[用户回复] {user_message}"
        else:
            self.messages.append({"role": "user", "content": user_message})
        self.task_complete = False

        # If the user's reply resolves the method ambiguity, mark it done.
        if any(m in user_message.lower() for m in ("cdft", "gcmc", "用 cDFT", "用 GCMC")):
            self._method_choice_made = True

        if verbose:
            print(f"\n🤖 [{self.current_agent.name}]", flush=True)

        return self._execute_loop(self.current_agent, max_rounds=max_rounds)
    
    def run_until_complete(self, user_message: str, agent: Agent | None = None, max_rounds: int = 45, verbose: bool = True) -> str:
        """一次性跑完（自动模式）。"""
        self.current_agent = agent
        self.messages = [{"role": "user", "content": user_message}]
        self.task_complete = False
        self.context = {}
        self.context['scientific_review_question'] = user_message
        self.context['scientific_tool_start'] = len(self.memory.tool_call_log)
        self.goal_contract = GoalContract.from_user_message(user_message)
        self._set_control_source(user_message)
        
        if verbose:
            print(f"\n🤖 [{self.current_agent.name}]", flush=True)
        
        return self._execute_loop(self.current_agent, max_rounds=max_rounds)

    @staticmethod
    def _json_safe(value: Any) -> Any:
        """Convert SDK content blocks and arbitrary state to JSON-safe values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): Session._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [Session._json_safe(v) for v in value]
        if hasattr(value, "model_dump"):
            try:
                return Session._json_safe(value.model_dump())
            except Exception:
                pass
        if hasattr(value, "dict"):
            try:
                return Session._json_safe(value.dict())
            except Exception:
                pass
        return str(value)

    def export_state(self) -> Dict[str, Any]:
        """Durable execution checkpoint for backend restart recovery."""
        # Persist the already-trimmed LLM context, not an unbounded duplicate of
        # the UI transcript. The goal contract separately preserves objectives.
        safe_messages = self._json_safe(self.messages)
        try:
            safe_messages = self._trim_messages(safe_messages, max_tokens=100000)
        except Exception:
            pass
        return {
            "schema_version": 2,
            "checkpoint_at": time.time(),
            "messages": safe_messages,
            "current_agent": self.current_agent.name if self.current_agent else "lead-orchestrator",
            "context": self._json_safe(self.context),
            "control_source": self._json_safe(self._control_source),
            "task_complete": bool(self.task_complete),
            "last_text": self.last_text,
            "memory": self._json_safe(self.memory.to_dict()),
            "goal_contract": self.goal_contract.to_dict(),
            "failure_retries": self._failure_retries,
            "failure_history": self.failure_history,
            "recovery_gate": self.recovery_gate.snapshot(),
            "current_line_id": self._current_line_id,
            "plan_steps_created": self._plan_steps_created,
            "error_branches": self._error_branches,
            "error_branch_counter": self._error_branch_counter,
            "awaiting_plan_approval": self._awaiting_plan_approval,
            "pending_plan_text": self._pending_plan_text,
            "pending_param_question": self._pending_param_question,
            "waiting_for_user_input": self._waiting_for_user_input,
            "pending_user_interaction": self._json_safe(self._pending_user_interaction),
            "pending_workflow_patch": self._pending_workflow_patch,
            "delegated_tasks": sorted(getattr(self, "_delegated_tasks", set())),
        }

    def import_state(self, data: Optional[Dict[str, Any]]) -> bool:
        """Restore a checkpoint. Returns False for absent/invalid state."""
        if not isinstance(data, dict) or not data:
            return False
        self._last_checkpoint_at = data.get('checkpoint_at', 0)
        self._control_source = data.get('control_source')
        messages = data.get("messages")
        if isinstance(messages, list):
            # Runtime routing/auto-confirm prompts are ephemeral decisions from
            # the old process. Recompute them from the restored GoalContract;
            # carrying stale prompts across restart can resurrect a superseded
            # method (e.g. old Henry/GCMC routing after user restored cDFT).
            stale_prefixes = (
                "[系统·快速路由]", "[系统·自动确认]", "[系统·强制执行]",
                "[系统·目标契约路由]", "[系统·方案确认门禁]",
                "[系统·结构化完成门禁]",
                "[系统·完成验证]",
            )
            self.messages = [
                m for m in messages
                if not (
                    m.get("role") == "user"
                    and isinstance(m.get("content"), str)
                    and m["content"].lstrip().startswith(stale_prefixes)
                )
            ]
        self.context = data.get("context") if isinstance(data.get("context"), dict) else {}
        self.task_complete = bool(data.get("task_complete", False))
        self.last_text = str(data.get("last_text", "") or "")
        self.memory = SessionMemory.from_dict(data.get("memory"))
        self.goal_contract = GoalContract.from_dict(data.get("goal_contract"))
        self._failure_retries = dict(data.get("failure_retries", {}) or {})
        self.failure_history = list(data.get("failure_history", []) or [])
        self.recovery_gate = RecoveryGate(data.get('recovery_gate'))
        self._current_line_id = str(data.get("current_line_id", "") or "")
        self._plan_steps_created = bool(data.get("plan_steps_created", False))
        self._error_branches = dict(data.get("error_branches", {}) or {})
        self._error_branch_counter = int(data.get("error_branch_counter", 0) or 0)
        self._awaiting_plan_approval = bool(data.get("awaiting_plan_approval", False))
        self._pending_plan_text = str(data.get("pending_plan_text", "") or "")
        self._pending_param_question = str(data.get("pending_param_question", "") or "")
        self._waiting_for_user_input = bool(data.get("waiting_for_user_input", False))
        pending_interaction = data.get("pending_user_interaction")
        self._pending_user_interaction = pending_interaction if isinstance(pending_interaction, dict) else None
        self._pending_workflow_patch = data.get('pending_workflow_patch')
        # Quoted historical job failure inside ok:true TaskLine responses used
        # to generate spurious recovery branches. Preserve the record but retire
        # only these provably misclassified query failures on upgrade.
        for branch_id, branch in self._error_branches.items():
            if (branch.get('failed_step') == 'task_line_query'
                    and branch.get('status') != 'resolved'
                    and re.match(r'^\s*\{\s*"ok"\s*:\s*true\b', str(branch.get('error', '')))):
                self._update_error_branch(branch_id, resolution='Legacy false alarm: successful TaskLine query quoted a historical failure; no compute failure occurred')
        if (self._pending_user_interaction and self._pending_user_interaction.get('params', {}).get('failed_tool') == 'task_line_query'):
            bid = self._pending_user_interaction['params'].get('branch_id')
            if self._error_branches.get(bid, {}).get('status') == 'resolved':
                self._pending_user_interaction = None
                self._waiting_for_user_input = False
        self._delegated_tasks = set(data.get("delegated_tasks", []) or [])
        try:
            from .defns import resolve_agent
            self.current_agent = resolve_agent(str(data.get("current_agent") or "lead-orchestrator"))
        except Exception:
            from .defns import ORCHESTRATOR
            self.current_agent = ORCHESTRATOR
        pending = self._pending_user_interaction or {}
        params = pending.get('params', {}) if isinstance(pending, dict) else {}
        if (pending.get('tool') == 'user_decision'
                and not params.get('recovery_key') and not params.get('candidate_job_id')):
            self._recover_validated_preflight_draft(wake_after_restore=True)
        return True

    def hydrate_from_transcript(self, messages_history: List[Dict[str, Any]]) -> bool:
        """Legacy fallback when no session checkpoint exists after restart."""
        if not isinstance(messages_history, list) or not messages_history:
            return False
        self.messages = []
        for m in messages_history:
            role = m.get("role")
            content = m.get("content", "")
            if role not in {"user", "assistant"} or not str(content).strip():
                continue
            if (self.messages and self.messages[-1]["role"] == role
                    and isinstance(self.messages[-1].get("content"), str)):
                self.messages[-1]["content"] += "\n\n" + str(content)
            else:
                self.messages.append({"role": role, "content": content})
        first_user = next((m.get("content", "") for m in self.messages if m.get("role") == "user"), "")
        self.goal_contract = GoalContract.from_user_message(str(first_user))
        # Replay only genuine user directives into the compact contract. This is
        # deterministic and does not require an LLM during server startup.
        seen_first = False
        for m in self.messages:
            if m.get("role") != "user":
                continue
            if not seen_first:
                seen_first = True
                continue
            self.goal_contract.apply_user_message(str(m.get("content", "")))
        return True

    def reconcile_transcript(self, messages_history: List[Dict[str, Any]], after_count=None) -> int:
        """Merge display-history turns newer than the persisted Session checkpoint.

        A background turn can checkpoint before a user's interrupt/redirect is
        appended to ``messages_history``. On restart, the checkpoint remains
        useful for tools/memory, but the later user instruction must win.
        Returns the number of merged transcript messages.
        """
        if not isinstance(messages_history, list):
            return 0
        if after_count is None:
            # Legacy/trimmed checkpoint: derive the goal from the ENTIRE ordered
            # user transcript, not just messages missing from its trimmed tail.
            # Replaying only missing old turns can otherwise overwrite a newer
            # contract whose latest turn is already present in the checkpoint.
            old = self.goal_contract
            rebuilt = GoalContract()
            for message in messages_history:
                if message.get('role') != 'user':
                    continue
                text = str(message.get('content', ''))
                if not rebuilt.original_goal:
                    from .goal_contract import _is_automated_message
                    if _is_automated_message(text):
                        continue
                    rebuilt = GoalContract.from_user_message(text)
                else:
                    rebuilt.apply_user_message(text)
            if rebuilt.original_goal:
                rebuilt.approved_plan_version = old.approved_plan_version
                rebuilt.pending_plan_version = old.pending_plan_version
                # Transcript reconstruction is not compilation or revocation.
                # Preserve approvals only when protected science is unchanged.
                if all(getattr(rebuilt, k) == getattr(old, k) for k in ('gases', 'method', 'parameters', 'execution_mode')):
                    rebuilt.approved_nodes = copy.deepcopy(old.approved_nodes)
                    rebuilt.execution_authorized = old.execution_authorized
                    rebuilt.requires_plan_approval = old.requires_plan_approval
                self.goal_contract = rebuilt
            suffix = messages_history
        else:
            suffix = messages_history[max(0, int(after_count)):]
        existing = {
            (m.get("role"), str(m.get("content", "")))
            for m in self.messages
            if isinstance(m.get("content"), str)
        }
        merged = 0
        for m in suffix:
            role = m.get("role")
            content = m.get("content", "")
            if role not in {"user", "assistant"} or not isinstance(content, str) or not content.strip():
                continue
            key = (role, content)
            if key in existing:
                continue
            if role == "user" and after_count is not None:
                self.goal_contract.apply_user_message(content)
            if (self.messages and self.messages[-1].get("role") == role
                    and isinstance(self.messages[-1].get("content"), str)):
                self.messages[-1]["content"] += "\n\n" + content
            else:
                self.messages.append({"role": role, "content": content})
            existing.add(key)
            merged += 1
        return merged
    
    def _get_final_text(self) -> str:
        return self.last_text
    
    def reset(self):
        self.messages = []
        self.current_agent = None
        self.context = {}
        self.task_complete = False
        self.last_text = ""
        self.memory = SessionMemory()
        self.goal_contract = GoalContract()
        self._prev_agent_name = ""
        self._awaiting_plan_approval = False
        self._pending_plan_text = ""

    def get_memory_summary(self) -> Dict[str, Any]:
        return {
            "stats": self.memory.get_stats(),
            "compact": self.memory.compact_summary(),
            "last_text": self.last_text[:500],
            "goal_contract": self.goal_contract.to_dict(),
            "awaiting_plan_approval": self._awaiting_plan_approval,
            "error_branches": list(self._error_branches.values()),
        }
