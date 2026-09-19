from types import SimpleNamespace

import anthropic
import httpx
import pytest
from anthropic.types import TextBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.session import LLMRateLimitError, Session
from agents import model_gate
import threading
import time


def test_fifo_model_admission_and_cancelled_waiter_cleanup():
    slots = model_gate.FairSlots(1)
    slots.acquire()
    order, threads = [], []
    cancelled = threading.Event()
    def worker(index):
        try:
            slots.acquire(cancelled.is_set if index == 1 else None)
        except model_gate.ModelGateInterrupted:
            return
        try: order.append(index)
        finally: slots.release()
    for index in range(4):
        thread = threading.Thread(target=worker, args=(index,))
        thread.start()
        threads.append(thread)
        deadline = time.monotonic() + 2
        while len(slots.waiters) < index + 1 and time.monotonic() < deadline:
            time.sleep(.005)
        assert len(slots.waiters) == index + 1
    cancelled.set()
    slots.release()
    for thread in threads:
        thread.join(2)
        assert not thread.is_alive()
    assert order == [0, 2, 3]
    assert slots.active == 0 and not slots.waiters


def test_provider_rate_limit_cooldown_resumes_without_prompt_pollution(tmp_path, monkeypatch):
    session = Session(config=AgentConfig(api_key="test", project_root=tmp_path))
    agent = Agent(name="lead-orchestrator", functions=[])
    calls = 0
    progress = []

    def response(_agent):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise LLMRateLimitError("429 Too many requests")
        return SimpleNamespace(content=[TextBlock(type="text", text="done")])

    session._call_api = response
    session._on_progress = lambda *args, **kwargs: progress.append((args, kwargs))
    monkeypatch.setenv("BIMEM_LLM_RATE_LIMIT_RETRIES", "2")
    monkeypatch.setattr("agents.session.time.sleep", lambda _seconds: None)

    result = session.run_until_complete("read-only audit", agent=agent, max_rounds=2)

    assert result == "done"
    assert calls >= 2
    assert any("限流" in str(item) and "自动继续" in str(item) for item in progress)
    assert not any(
        "LLM API 调用失败" in str(message.get("content", ""))
        for message in session.messages
    )


def test_shared_provider_circuit_opens_and_is_interruptible(monkeypatch):
    monkeypatch.setattr(model_gate, "_cooldown_until", 0.0)
    monkeypatch.setattr(model_gate, "_consecutive_limits", 0)
    response = httpx.Response(429, request=httpx.Request("POST", "https://provider.invalid/messages"))

    def limited():
        raise anthropic.RateLimitError("limited", response=response, body={"error": {"type": "limitation"}})

    with pytest.raises(model_gate.ProviderRateLimited) as caught:
        model_gate.call_model(limited)
    assert caught.value.wait_seconds == 15
    assert model_gate.provider_state()["consecutive_limits"] == 1

    with pytest.raises(model_gate.ModelGateInterrupted):
        model_gate.call_model(lambda: "must not run", interrupted=lambda: True)

    monkeypatch.setattr(model_gate, "_cooldown_until", 0.0)
    assert model_gate.call_model(lambda: "ok") == "ok"
    assert model_gate.provider_state()["consecutive_limits"] == 0
