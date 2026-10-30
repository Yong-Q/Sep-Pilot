"""Offline wire failures must never become executable ordinary tool input."""
import dataclasses
import json
import subprocess

import pytest

from agents import codex_provider as provider


TOOLS = [
    {"name": "read_file", "input_schema": {"type": "object"}},
    {"name": "list_jobs", "input_schema": {"type": "object", "properties": {}}},
]


@pytest.mark.parametrize("arguments", ['{"path":', '[]', 'null', '"text"',
                                       '{"amount":NaN}', '{"amount":Infinity}'])
def test_invalid_wire_arguments_return_explicit_unexecuted_call_error(arguments):
    payload = {"text": "", "tool_calls": [
        {"name": "list_jobs", "arguments_json": arguments}]}
    with pytest.raises(provider.CodexProviderError) as caught:
        provider._validate_reply(payload, TOOLS, None)
    error = caught.value
    assert type(error).__name__ == "CodexToolArgumentError"
    assert error.executed is False
    assert len(error.tool_errors) == 1
    failed = error.tool_errors[0]
    assert failed.name == "list_jobs"
    assert failed.call_id.startswith("toolu_")
    assert "JSON" in failed.error
    assert not hasattr(failed, "input")


def test_mixed_batch_is_rejected_as_a_whole_without_losing_any_proposal():
    payload = {"text": "", "tool_calls": [
        {"name": "read_file", "arguments_json": '{"path":"fixture.txt"}'},
        {"name": "list_jobs", "arguments_json": '{'},
        {"name": "list_jobs", "arguments_json": '{}'},
    ]}
    with pytest.raises(provider.CodexProviderError) as caught:
        provider._validate_reply(payload, TOOLS, None)
    error = caught.value
    assert type(error).__name__ == "CodexToolArgumentError"
    assert [item.name for item in error.tool_errors] == ["read_file", "list_jobs", "list_jobs"]
    assert len({item.call_id for item in error.tool_errors}) == 3
    assert all("not executed" in item.error for item in error.tool_errors)
    with pytest.raises(dataclasses.FrozenInstanceError):
        error.tool_errors[0].name = "different_tool"


def test_wire_error_does_not_retain_sensitive_arguments_or_parser_context():
    sensitive = "private-fixture-value-271828"
    payload = {"text": sensitive, "tool_calls": [
        {"name": "read_file", "arguments_json": '{"path":"' + sensitive},
        {"name": "list_jobs", "arguments_json": json.dumps({"unused": sensitive})},
    ]}
    with pytest.raises(provider.CodexProviderError) as caught:
        provider._validate_reply(payload, TOOLS, None)
    error = caught.value
    assert type(error).__name__ == "CodexToolArgumentError"
    assert sensitive not in str(error)
    assert sensitive not in repr(vars(error))
    assert sensitive not in repr(error.args)
    assert error.__context__ is None
    assert error.__cause__ is None


@pytest.mark.parametrize("choice, name, message", [
    (None, "unavailable", "unavailable tool"),
    ({"type": "tool", "name": "read_file"}, "list_jobs", "required choice"),
])
def test_tool_authority_is_validated_before_recoverable_argument_errors(choice, name, message):
    payload = {"text": "", "tool_calls": [
        {"name": "read_file", "arguments_json": '{'},
        {"name": name, "arguments_json": '{}'},
    ]}
    with pytest.raises(provider.CodexProviderError, match=message) as caught:
        provider._validate_reply(payload, TOOLS, choice)
    assert type(caught.value) is provider.CodexProviderError


def test_provider_preserves_safe_wire_error_and_stable_contract_prefix(monkeypatch):
    captured = {}

    def command(root, model):
        captured["root"] = root
        return ["fixture-cli"]

    def invoke(cmd, **kwargs):
        prompt = kwargs["input"]
        captured["contract"] = json.loads(prompt.split("\n\n", 1)[1])
        (captured["root"] / "answer.json").write_text(json.dumps({
            "text": "", "tool_calls": [{"name": "list_jobs", "arguments_json": '{'}]}))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(provider, "_isolated_command", command)
    monkeypatch.setattr(provider, "_invoke_cli", invoke)
    with pytest.raises(provider.CodexProviderError) as caught:
        provider.call_codex_model("dynamic instructions", [{"role": "user", "content": "status"}], TOOLS)
    assert type(caught.value).__name__ == "CodexToolArgumentError"
    assert caught.value.executed is False
    assert list(captured["contract"]) == ["available_tools", "system", "conversation", "tool_choice"]
    assert captured["contract"]["available_tools"] == TOOLS
    assert captured["contract"]["system"] == "dynamic instructions"
    assert not captured["root"].exists()
