"""The primary provider must preserve the host's parameter-repair loop."""
import json
import subprocess
from pathlib import Path

from agents.agent import Agent
from agents.config import AgentConfig
from agents.registry import ToolDef, ToolRegistry
from agents.session import Session, LLMUnavailableError


def test_codex_invalid_arguments_receive_tool_feedback_then_recover(tmp_path, monkeypatch):
    executions, prompts = [], []
    registry = ToolRegistry()
    registry.register(ToolDef("measure_fixture", "Read a fixture measurement", {
        "type": "object", "properties": {"sample_id": {"type": "string"}},
        "required": ["sample_id"], "additionalProperties": False,
    }, lambda p: executions.append(p) or {"value": 2.0, "fixture_only": True}))

    def measure_fixture(**kwargs):
        return registry.execute("measure_fixture", kwargs)

    replies = iter([
        {"text": "", "tool_calls": [{"name": "measure_fixture", "arguments_json": '{"sample":"fixture-a"}'}]},
        {"text": "", "tool_calls": [{"name": "measure_fixture", "arguments_json": '{"sample_id":"fixture-a"}'}]},
        {"text": "Fixture read successfully; no physical calculation was performed.", "tool_calls": []},
    ])
    def invoke(command, **kwargs):
        prompts.append(kwargs["input"])
        (Path(command[1]) / "answer.json").write_text(json.dumps(next(replies)))
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agents.codex_provider._isolated_command", lambda root, model: ["fixture-cli", str(root)])
    monkeypatch.setattr("agents.codex_provider._invoke_cli", invoke)
    session = Session(config=AgentConfig(model_provider="codex_cli", api_key="unused", project_root=tmp_path), registry=registry)
    agent = Agent(name="lead-orchestrator", instructions="Read the requested fixture and return its actual evidence.", functions=[measure_fixture])
    answer = session.run_until_complete("Read fixture-a without any calculation.", agent=agent, max_rounds=4, verbose=False)

    assert executions == [{"sample_id": "fixture-a"}]
    assert "sample_id" in prompts[1] and "required" in prompts[1]
    assert "Codex proposed invalid arguments" not in prompts[1]
    transcript = json.loads(prompts[1].split("\n\n", 1)[1])["conversation"]
    feedback = [block for message in transcript if isinstance(message.get("content"), list)
                for block in message["content"]]
    assert any(block.get("type") == "tool_result" and "sample_id" in block["content"] for block in feedback)
    assert any(c.get("failed") for c in session.memory.tool_call_log)
    assert "Fixture read successfully" in answer


def test_provider_outage_does_not_mark_unfinished_task_complete(tmp_path, monkeypatch):
    session = Session(config=AgentConfig(api_key="unused", project_root=tmp_path))
    def offline(agent):
        raise LLMUnavailableError("controlled unavailable provider")
    monkeypatch.setattr(session, "_call_api", offline)
    checkpoints = []
    session._on_checkpoint = lambda state, reason: checkpoints.append((state, reason))
    session.run_until_complete("Inspect the unfinished workflow.", agent=Agent(name="lead-orchestrator"), max_rounds=4, verbose=False)
    assert session.task_complete is False
    assert any(reason == "provider_unavailable" for state, reason in checkpoints)
