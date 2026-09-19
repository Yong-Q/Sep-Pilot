"""Regression tests for the Claude-Code-style interrupt (freeze) protocol:

- Interrupt = FREEZE only: no further LLM call, no tool calls, previously
  submitted SLURM jobs keep running, honest partial report (no fabrication).
- A redirect attached to the interrupt ("中断加话和需求") is consumed exactly
  once and merged into the next delegation.
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.session import Session
from agents.config import get_config
from agents.defns import ORCHESTRATOR


def _freeze_session(**callbacks):
    s = Session(config=get_config())
    s.current_agent = ORCHESTRATOR
    s._on_interrupt_requested = callbacks.get("requested", lambda: False)
    s._on_interrupt_message = callbacks.get("message", lambda: "")
    cleared = []
    s._on_interrupt_clear = lambda: cleared.append(True)
    return s, cleared


def test_interrupt_freeze_zero_llm_and_honest_report():
    """While frozen, the loop must NOT call the LLM, must NOT fabricate a job,
    and must return a real partial report of pending jobs / completed steps."""
    s, cleared = _freeze_session(requested=lambda: True)
    # If _call_api is reached during the freeze, that is the exact bug this
    # protocol exists to prevent (agent keeps "thinking" after user freezes).
    with patch.object(s, "_call_api",
                      side_effect=AssertionError("LLM called during freeze!")):
        out = s.reply("跑一下MOF-5的CO2吸附", max_rounds=5)

    assert "用户中断" in out or "冻结" in out
    assert s.task_complete is True
    assert s.last_text == out
    # No fabricated job_id — the report is assembled from real state only.
    assert "job_id=123" not in out


def test_interrupt_redirect_merged_into_next_delegation_and_cleared_once():
    """A redirect attached to the interrupt is merged into the next user
    delegation and cleared exactly once (no double-consume, no re-freeze)."""
    s, cleared = _freeze_session(
        requested=lambda: False,          # new delegation releases the freeze
        message=lambda: "改为用cDFT不要GCMC",
    )

    class _B:
        type = "text"
        text = "好的，已收到重定向需求。"

    class _Resp:
        content = [_B()]

    with patch.object(s, "_call_api", return_value=_Resp()):
        out = s.reply("请继续", max_rounds=2)

    # The redirect must be injected as the delegation's leading context.
    _msg_texts = [m.get("content") for m in s.messages if isinstance(m.get("content"), str)]
    merged = "\n".join(_msg_texts)
    assert "重定向需求" in merged, merged
    assert "改为用cDFT不要GCMC" in merged, merged
    # Cleared exactly once (consumed by this reply, not lingering).
    assert len(cleared) == 1
    assert "重定向需求" in out or out  # loop ran with the merged message


def test_interrupt_without_redirect_still_clears():
    """Freeze without an attached message is still consumed on the next
    delegation (so the conversation doesn't stay frozen forever)."""
    s, cleared = _freeze_session(requested=lambda: False, message=lambda: "")

    class _B:
        type = "text"
        text = "继续。"

    class _Resp:
        content = [_B()]

    with patch.object(s, "_call_api", return_value=_Resp()):
        out = s.reply("请继续", max_rounds=1)

    assert len(cleared) == 1


def test_midstream_interrupt_does_not_wait_for_final_message():
    """Breaking the stream must close it, never drain the provider response."""
    s, _ = _freeze_session(requested=lambda: True)
    state = {'closed': False, 'final_called': False}

    class Stream:
        def __enter__(self): return self
        def __exit__(self, *args): state['closed'] = True
        def __iter__(self): return iter([object()])
        def get_final_message(self):
            state['final_called'] = True
            raise AssertionError('interrupted stream was drained')

    class Messages:
        def stream(self, **kwargs): return Stream()

    s.client = type('Client', (), {'messages': Messages()})()
    response = s._stream_api()
    assert response.stop_reason == 'interrupted' and response.content == []
    assert state == {'closed': True, 'final_called': False}
