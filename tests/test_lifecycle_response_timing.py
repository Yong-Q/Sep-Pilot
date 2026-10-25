"""Deterministic dispatch timing; no external model, scheduler or live sessions."""
import asyncio
import threading
import time
from types import SimpleNamespace

import pytest

from agents.config import AgentConfig
from agents.lifecycle import LifecycleStore, wait_for_lifecycle_change
from agents.session import Session


@pytest.mark.parametrize("kind", [
    "tool_failed", "worker_validation_failed", "plan_patch_proposed", "worker_failed",
])
def test_provisional_repair_receipt_wakes_dispatcher_without_model_completion(tmp_path, kind):
    store = LifecycleStore(tmp_path / "lifecycle.json")
    event_id = store.emit(kind, {"step_id": "case"})
    assert wait_for_lifecycle_change(0)
    assert store.claim(event_id, "supervisor")
    store.provisional(event_id, {"next_action": "diagnose_and_fix"})
    started = time.monotonic()
    woke = wait_for_lifecycle_change(.2)
    elapsed = time.monotonic() - started
    assert woke, f"{kind} did not wake dispatcher; waited {elapsed:.3f}s"
    event = store.snapshot()["events"][event_id]
    assert event["supervisor"] == "processing"
    assert event["supervisor_receipt"]["audit_pending"]
    assert event["supervisor_provisional_at"] >= event["supervisor_claimed_at"]
    assert event["main_chat"] == "pending"


def test_missing_provisional_event_does_not_create_false_wakeup(tmp_path):
    store = LifecycleStore(tmp_path / "lifecycle.json")
    wait_for_lifecycle_change(0)
    store.provisional("missing", {"next_action": "diagnose_and_fix"})
    assert not wait_for_lifecycle_change(0)


def test_real_dispatcher_resumes_main_while_supervisor_model_is_still_waiting(tmp_path, monkeypatch, record_property):
    # The provider is a controlled slow stand-in. Store, claim, provisional
    # receipt, dispatcher and main delivery are the real production methods.
    import agents.job_watch
    monkeypatch.setattr(agents.job_watch.JobWatch, "start", lambda self: self)
    import api

    main = Session(config=AgentConfig(api_key="test", project_root=tmp_path))
    store = main._lifecycle_store = LifecycleStore(tmp_path / "lifecycle.json")
    event_id = store.emit("worker_failed", {"step_id": "runtime_case"})
    wait_for_lifecycle_change(0)
    conv = SimpleNamespace(session=main, username="timing_test", conv_id="isolated",
        is_processing=False, current_uid="initial", current_agent="lead-orchestrator",
        last_progress_at=0, interrupt_requested=False, interrupt_message="",
        current_status="pending", execution_logs=[])
    user = SimpleNamespace(conversations={"isolated": conv})
    monkeypatch.setattr(api.session_manager, "list_users", lambda: ["timing_test"])
    monkeypatch.setattr(api.session_manager, "get_user", lambda username: user)
    monkeypatch.setattr(api, "_lifecycle_tasks", set())
    provider_release = threading.Event()
    continued = []

    def observe(event):
        assert provider_release.wait(3), "test provider release timed out"
        return {"next_action": "diagnose_and_fix", "reason": "controlled audit complete"}

    monkeypatch.setattr(api, "_ensure_supervisor", lambda c: SimpleNamespace(observe_lifecycle_event=observe))

    async def run_main(username, query, conv_id, lifecycle_event=None):
        assert not provider_release.is_set()
        continued.append(lifecycle_event)
        conv.current_status = "repair_received"
        conv.is_processing = False

    monkeypatch.setattr(api, "run_agent_background", run_main)

    async def scenario():
        try:
            await api._dispatch_lifecycle_once()
            await asyncio.sleep(0)  # start the supervisor task, not a polling sweep
            assert await asyncio.to_thread(wait_for_lifecycle_change, .5), "main waited for the safety sweep"
            await api._dispatch_lifecycle_once()
            await asyncio.sleep(0)
            assert len(continued) == 1
            event = store.snapshot()["events"][event_id]
            assert event["main_chat"] == "delivered"
            assert event["supervisor"] == "processing"
            assert event["supervisor_claimed_at"] >= event["time"]
            assert event["main_chat_claimed_at"] >= event["supervisor_provisional_at"] >= event["supervisor_claimed_at"]
            assert event["main_chat_updated_at"] >= event["main_chat_claimed_at"]
            assert event["main_chat_updated_at"] - event["time"] < 1
            for stage in ("supervisor_claimed_at", "supervisor_provisional_at", "main_chat_claimed_at", "main_chat_updated_at"):
                record_property(stage + "_seconds_from_event", event[stage] - event["time"])
            assert continued[0]["supervisor_receipt"]["audit_pending"]
        finally:
            provider_release.set()
            if api._lifecycle_tasks:
                await asyncio.gather(*list(api._lifecycle_tasks))
    asyncio.run(scenario())
    assert store.snapshot()["events"][event_id]["supervisor"] == "delivered"
