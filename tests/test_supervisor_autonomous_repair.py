import pytest

from agents.config import AgentConfig
from agents.session import Session


@pytest.mark.parametrize("kind", ["worker_failed", "worker_validation_failed", "worker_recovery_ready", "plan_patch_proposed"])
def test_recoverable_internal_failure_stays_with_agents(tmp_path, monkeypatch, kind):
    session = Session(config=AgentConfig(api_key="unused", project_root=tmp_path))
    monkeypatch.setattr(session, "_pending_jobs_for_conv", lambda: [])
    result = session._audit_supervisor_decision(
        {"next_action": "ask_user", "reason": "tool needs a path"},
        {"kind": kind, "payload": {"requires_user": False, "step_id": "case"}})
    assert result["next_action"] == "diagnose_and_fix"


def test_genuine_decision_is_not_overridden_by_autonomous_repair(tmp_path, monkeypatch):
    session = Session(config=AgentConfig(api_key="unused", project_root=tmp_path))
    monkeypatch.setattr(session, "_pending_jobs_for_conv", lambda: [])
    result = session._audit_supervisor_decision(
        {"next_action": "diagnose_and_fix"},
        {"kind": "worker_uncertain", "payload": {"requires_user": True}})
    assert result["next_action"] == "ask_user"
