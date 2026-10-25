"""Use the signed-in Codex CLI as a bounded model backend for the existing Session.

This adapter only asks the model for text and proposed tool calls. The host
Session remains responsible for tool execution, permissions, history, and DAGs.
No account credential is read, copied, or written by this module.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from urllib.parse import urlsplit

import jsonschema
from anthropic.types import TextBlock, ToolUseBlock


class CodexProviderError(RuntimeError):
    """A safe-to-display provider failure without prompt or credential content."""


class _CodexInterrupted(Exception):
    pass


def _invoke_cli(command, *, input, text, stdout, stderr, timeout, env, cwd,
                interrupted=None):
    """Run CLI with short interrupt checks; never expose captured streams."""
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=stdout,
                               stderr=stderr, text=text, env=env, cwd=cwd)
    deadline = time.monotonic() + timeout
    pending_input = input
    try:
        while True:
            if interrupted and interrupted():
                raise _CodexInterrupted()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(command, timeout)
            try:
                out, err = process.communicate(input=pending_input, timeout=min(.25, remaining))
                return subprocess.CompletedProcess(command, process.returncode, out, err)
            except subprocess.TimeoutExpired:
                pending_input = None
    except BaseException:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
        raise


_REPLY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "tool_calls": {"type": "array", "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}},
            "required": ["name", "arguments_json"],
            "additionalProperties": False,
        }},
    },
    "required": ["text", "tool_calls"],
    "additionalProperties": False,
}


def _as_json(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _as_json(value.model_dump(exclude_none=True))
    if isinstance(value, dict):
        return {str(k): _as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise CodexProviderError("Conversation contains an unsupported block type")


def _child_env() -> dict[str, str]:
    # Explicit allowlist: arbitrary caller environment must never enter the
    # model process (or its own tool environment).
    safe = {"LANG", "LC_ALL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
            "NO_PROXY", "http_proxy", "https_proxy", "all_proxy", "no_proxy"}
    env = {name: value for name, value in os.environ.items() if name in safe}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                 "http_proxy", "https_proxy", "all_proxy"):
        value = env.get(name)
        if value:
            parsed = urlsplit(value if "://" in value else "//" + value)
            if parsed.username is not None or parsed.password is not None:
                raise CodexProviderError("Proxy credentials cannot be forwarded to Codex")
    env.update(PATH="/usr/bin:/bin", HOME="/work", NO_COLOR="1")
    return env


def _isolated_command(root: Path, model: str) -> list[str]:
    """Expose the CLI runtime and signed-in auth file, never the project tree.

    Codex needs to update installation_id in CODEX_HOME, so its home is a
    private tmpfs. The existing auth file is bind-mounted read-only. No auth
    bytes are copied into the project or the scratch directory.
    """
    bwrap = shutil.which("bwrap")
    node = shutil.which("node")
    codex = shutil.which("codex")
    if not (bwrap and node and codex):
        raise CodexProviderError("Isolated Codex CLI runtime is unavailable")
    node_root = Path(node).resolve().parent.parent
    if not Path(codex).resolve().is_relative_to(node_root):
        raise CodexProviderError("Codex CLI runtime layout is unsupported")
    auth_home = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))).expanduser()
    auth_file = auth_home / "auth.json"
    if not auth_file.is_file():
        raise CodexProviderError("Signed-in Codex account is unavailable")
    command = [
        "bwrap", "--unshare-all", "--share-net", "--die-with-parent",
        "--ro-bind", "/usr", "/usr",
        "--ro-bind", "/lib", "/lib",
        "--ro-bind", "/lib64", "/lib64",
        "--ro-bind", "/bin", "/bin",
        "--ro-bind", "/etc", "/etc",
        "--ro-bind", str(node_root), "/opt/node",
        "--proc", "/proc", "--dev", "/dev",
        "--tmpfs", "/tmp", "--tmpfs", "/codexhome",
        "--ro-bind", str(auth_file), "/codexhome/auth.json",
        "--bind", str(root), "/work",
        "--setenv", "HOME", "/work",
        "--setenv", "CODEX_HOME", "/codexhome",
        "--setenv", "TMPDIR", "/tmp",
        "--setenv", "PATH", "/opt/node/bin:/usr/bin:/bin",
        "--setenv", "NO_COLOR", "1",
        "--chdir", "/work",
        "/opt/node/bin/codex", "exec", "--ephemeral", "--ignore-user-config",
        "--ignore-rules", "--sandbox", "read-only", "--skip-git-repo-check",
        "--cd", "/work", "--model", model,
        "-c", 'model_reasoning_effort="medium"',
        "--output-schema", "/work/reply.schema.json",
        "-o", "/work/answer.json", "-",
    ]
    return command


def _validate_reply(payload: Any, tools: list[dict], tool_choice: dict | None) -> list[Any]:
    try:
        jsonschema.validate(payload, _REPLY_SCHEMA)
    except jsonschema.ValidationError as exc:
        raise CodexProviderError("Codex returned an invalid response shape") from None
    allowed = {tool["name"]: tool for tool in tools}
    if payload["tool_calls"] and not tools:
        raise CodexProviderError("Codex proposed an unavailable tool")
    blocks: list[Any] = []
    if payload["text"].strip():
        blocks.append(TextBlock(type="text", text=payload["text"]))
    for call in payload["tool_calls"]:
        name = call["name"]
        if name not in allowed:
            raise CodexProviderError("Codex proposed an unavailable tool")
        if tool_choice and tool_choice.get("type") == "tool" and name != tool_choice.get("name"):
            raise CodexProviderError("Codex selected a tool outside the required choice")
        try:
            arguments = json.loads(call["arguments_json"])
            if not isinstance(arguments, dict):
                raise ValueError("arguments must be object")
            jsonschema.validate(arguments, allowed[name].get("input_schema", {"type": "object"}))
        except (ValueError, jsonschema.ValidationError, jsonschema.SchemaError):
            raise CodexProviderError("Codex proposed invalid arguments") from None
        blocks.append(ToolUseBlock(type="tool_use", id="toolu_" + uuid.uuid4().hex,
                                   name=name, input=arguments))
    if tool_choice and tool_choice.get("type") == "tool" and not payload["tool_calls"]:
        raise CodexProviderError("Codex omitted the required tool call")
    if not blocks:
        raise CodexProviderError("Codex returned an empty response")
    return blocks


def call_codex_model(
    system: str,
    messages: list[dict],
    tools: list[dict] | None = None,
    *,
    model: str = "gpt-6-sol",
    timeout: int = 180,
    interrupted: Callable[[], bool] | None = None,
    tool_choice: dict | None = None,
) -> SimpleNamespace:
    """Return Claude-compatible content blocks from one isolated CLI model turn.

    The CLI is ephemeral and read-only. Its own stdout/stderr is captured and
    discarded, never persisted to the scientific session. Only the validated
    JSON response enters the host agent's history.
    """
    if interrupted and interrupted():
        return SimpleNamespace(content=[], stop_reason="interrupted")
    tools = list(tools or [])
    if tool_choice and tool_choice.get("type") == "tool":
        required_name = tool_choice.get("name")
        if required_name not in {tool.get("name") for tool in tools}:
            raise CodexProviderError("Required tool is unavailable")
    try:
        transcript = _as_json(messages)
        json.dumps(transcript, ensure_ascii=False)
    except (TypeError, ValueError) as exc:
        raise CodexProviderError("Conversation could not be serialized") from None

    contract = {
        "system": system,
        "conversation": transcript,
        "available_tools": tools,
        "tool_choice": tool_choice or {"type": "auto"},
    }
    prompt = (
        "You are the language model inside a scientific multi-agent application. "
        "Follow the system instructions and conversation exactly. The listed available_tools "
        "are the only application tools you may propose. You do not execute them yourself. "
        "Return one JSON object matching the required output schema: text is visible assistant "
        "text, and tool_calls is an array of tool proposals. Each tool proposal has a name "
        "and arguments_json, a JSON-encoded object string matching that tool's input_schema. "
        "If a tool is required, propose it. "
        "Treat prior tool_result blocks as observations. Do not access local files or invoke "
        "Codex tools; application tools are executed by the host after your response.\n\n"
        + json.dumps(contract, ensure_ascii=False, separators=(",", ":"))
    )
    scratch_parent = Path(__file__).resolve().parent.parent / "tmp"
    scratch_parent.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bimem-codex-", dir=scratch_parent) as temp:
        root = Path(temp)
        schema_file = root / "reply.schema.json"
        answer_file = root / "answer.json"
        schema_file.write_text(json.dumps(_REPLY_SCHEMA), encoding="utf-8")
        command = _isolated_command(root, model)
        try:
            completed = _invoke_cli(command, input=prompt, text=True,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=timeout, env=_child_env(), cwd=root,
                                       interrupted=interrupted)
        except _CodexInterrupted:
            return SimpleNamespace(content=[], stop_reason="interrupted")
        except subprocess.TimeoutExpired:
            raise CodexProviderError("Codex model call timed out") from None
        except OSError:
            raise CodexProviderError("Codex CLI is unavailable") from None
        if interrupted and interrupted():
            return SimpleNamespace(content=[], stop_reason="interrupted")
        if completed.returncode != 0:
            raise CodexProviderError("Codex model call failed")
        try:
            payload = json.loads(answer_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise CodexProviderError("Codex returned no valid JSON response") from None
        blocks = _validate_reply(payload, tools, tool_choice)
        return SimpleNamespace(content=blocks,
                               stop_reason="tool_use" if payload["tool_calls"] else "end_turn")
