"""Contract tests for using Codex as the Session's model, not its agent runtime."""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agents.codex_provider import CodexProviderError, _CodexInterrupted, _child_env, _invoke_cli, call_codex_model


TOOLS = [{"name": "read_file", "description": "Read a file", "input_schema": {
    "type": "object", "properties": {"path": {"type": "string"}},
    "required": ["path"], "additionalProperties": False,
}}]


@pytest.fixture(autouse=True)
def isolated_cli_runtime(tmp_path, monkeypatch):
    """Provide synthetic mount sources without requiring host CLI installations."""
    runtime = tmp_path / 'runtime'
    binaries = runtime / 'bin'
    binaries.mkdir(parents=True)
    for name in ('bwrap', 'node', 'codex'):
        executable = binaries / name
        executable.write_text('#!/bin/sh\nexit 97\n')
        executable.chmod(0o700)
    auth_home = tmp_path / 'fake-codex-home'
    auth_home.mkdir()
    (auth_home / 'auth.json').write_text('{"fixture": true}')
    monkeypatch.setenv('CODEX_HOME', str(auth_home))
    monkeypatch.setattr('agents.codex_provider.shutil.which',
                        lambda name: str(binaries / name) if name in {'bwrap', 'node', 'codex'} else None)
    return binaries, auth_home


def _host_answer(command):
    return Path(command[command.index("--bind") + 1]) / "answer.json"


def test_tool_proposal_returns_claude_compatible_block(monkeypatch):
    captured = {}

    def fake_run(command, **kwargs):
        captured.update(command=command, kwargs=kwargs)
        output = _host_answer(command)
        with open(output, "w", encoding="utf-8") as f:
            json.dump({"text": "", "tool_calls": [
                {"name": "read_file", "arguments_json": '{"path":"data/manifest.json"}'}]}, f)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("agents.codex_provider._invoke_cli", fake_run)
    reply = call_codex_model("You are a scientist", [{"role": "user", "content": "Check manifest"}], TOOLS)
    assert reply.stop_reason == "tool_use"
    assert len(reply.content) == 1
    assert reply.content[0].type == "tool_use"
    assert reply.content[0].name == "read_file"
    assert reply.content[0].input == {"path": "data/manifest.json"}
    cmd = captured["command"]
    assert cmd[0] == "bwrap"
    assert "--unshare-all" in cmd
    assert "--share-net" in cmd
    assert "--ro-bind" in cmd
    assert "/codexhome/auth.json" in cmd
    assert "/opt/node/bin/codex" in cmd
    assert "--ephemeral" in cmd
    assert "--ignore-user-config" in cmd
    cli = cmd[cmd.index('/opt/node/bin/codex') + 1:]
    disabled = {cli[index + 1] for index, value in enumerate(cli[:-1]) if value == '--disable'}
    assert disabled == {'shell_tool', 'shell_snapshot', 'apps', 'plugins', 'hooks',
                        'multi_agent', 'code_mode', 'view_image', 'image_generation',
                        'browser_use', 'computer_use', 'sleep_tool', 'tool_suggest'}
    overrides = {cli[index + 1] for index, value in enumerate(cli[:-1]) if value == '-c'}
    assert {'mcp_servers={}', 'web_search="disabled"', 'agents.enabled=false'} <= overrides
    assert 'unified_exec' not in disabled
    assert [cmd[cmd.index("--sandbox") + 1], cmd[cmd.index("--model") + 1]] == ["read-only", "gpt-6-sol"]
    assert "model_reasoning_effort=\"medium\"" in cmd
    assert "OPENAI_API_KEY" not in captured["kwargs"]["env"]
    assert "ANTHROPIC_API_KEY" not in captured["kwargs"]["env"]
    assert captured["kwargs"]["input"]
    assert "You are a scientist" in captured["kwargs"]["input"]
    assert "tmp/bimem-codex-" in str(_host_answer(cmd))
    assert not _host_answer(cmd).parent.exists()


def test_invalid_tool_is_rejected_without_execution(monkeypatch):
    def fake_run(command, **kwargs):
        with open(_host_answer(command), "w", encoding="utf-8") as f:
            json.dump({"text": "", "tool_calls": [{"name": "shell", "arguments_json": "{}"}]}, f)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("agents.codex_provider._invoke_cli", fake_run)
    with pytest.raises(CodexProviderError, match="unavailable tool"):
        call_codex_model("", [{"role": "user", "content": "hi"}], TOOLS)


def test_schema_invalid_arguments_reach_host_validation(monkeypatch):
    def fake_run(command, **kwargs):
        with open(_host_answer(command), "w", encoding="utf-8") as f:
            json.dump({"text": "", "tool_calls": [{"name": "read_file", "arguments_json": '{"bad":"x"}'}]}, f)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("agents.codex_provider._invoke_cli", fake_run)
    response = call_codex_model("", [{"role": "user", "content": "hi"}], TOOLS)
    assert response.content[0].name == "read_file"
    assert response.content[0].input == {"bad": "x"}


def test_cli_error_does_not_reveal_stderr_or_prompt(monkeypatch):
    def fake_run(command, **kwargs):
        return subprocess.CompletedProcess(command, 1, "private stdout", "private stderr token")
    monkeypatch.setattr("agents.codex_provider._invoke_cli", fake_run)
    with pytest.raises(CodexProviderError) as error:
        call_codex_model("secret prompt", [{"role": "user", "content": "hi"}], TOOLS)
    assert "private" not in str(error.value)
    assert "secret" not in str(error.value)


def test_plain_text_reply_and_tool_result_history(monkeypatch):
    captured = {}
    def fake_run(command, **kwargs):
        captured["prompt"] = kwargs["input"]
        with open(_host_answer(command), "w", encoding="utf-8") as f:
            json.dump({"text": "Evidence is insufficient.", "tool_calls": []}, f)
        return subprocess.CompletedProcess(command, 0, "", "")
    monkeypatch.setattr("agents.codex_provider._invoke_cli", fake_run)
    messages = [
        {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "x"}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_1", "content": "found"}]},
    ]
    reply = call_codex_model("system", messages, TOOLS)
    assert reply.stop_reason == "end_turn"
    assert reply.content[0].text == "Evidence is insufficient."
    assert "toolu_1" in captured["prompt"]


def test_interruption_prevents_cli_call(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("CLI must not run")
    monkeypatch.setattr("agents.codex_provider._invoke_cli", unexpected)
    reply = call_codex_model("", [], TOOLS, interrupted=lambda: True)
    assert reply.stop_reason == "interrupted"
    assert reply.content == []


def test_interruption_stops_running_child(tmp_path):
    started = time.monotonic()
    command = [sys.executable, "-c", "import time; time.sleep(20)"]
    with pytest.raises(_CodexInterrupted):
        _invoke_cli(command, input="", text=True, stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE, timeout=10, env={}, cwd=tmp_path,
                    interrupted=lambda: time.monotonic() - started > .2)
    assert time.monotonic() - started < 3


def test_proxy_userinfo_is_never_forwarded(monkeypatch):
    proxy = "http://" + "name" + ":" + "password" + "@proxy.example:8080"
    monkeypatch.setenv("HTTPS_PROXY", proxy)
    with pytest.raises(CodexProviderError, match="Proxy credentials") as error:
        _child_env()
    assert "name" not in str(error.value)
    assert "password" not in str(error.value)


def test_missing_runtime_is_rejected_before_cli(monkeypatch):
    monkeypatch.setattr('agents.codex_provider.shutil.which', lambda _: None)
    monkeypatch.setattr('agents.codex_provider._invoke_cli',
                        lambda *a, **kw: pytest.fail('unavailable runtime must not execute'))
    with pytest.raises(CodexProviderError, match='runtime is unavailable'):
        call_codex_model('', [], TOOLS)


def test_unsupported_runtime_layout_is_rejected(tmp_path, monkeypatch):
    outside = tmp_path / 'outside-codex'
    outside.write_text('synthetic executable')
    original = __import__('agents.codex_provider', fromlist=['shutil']).shutil.which
    monkeypatch.setattr('agents.codex_provider.shutil.which',
                        lambda name: str(outside) if name == 'codex' else original(name))
    with pytest.raises(CodexProviderError, match='layout is unsupported'):
        call_codex_model('', [], TOOLS)


def test_missing_synthetic_auth_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setenv('CODEX_HOME', str(tmp_path / 'missing-home'))
    with pytest.raises(CodexProviderError, match='account is unavailable'):
        call_codex_model('', [], TOOLS)
