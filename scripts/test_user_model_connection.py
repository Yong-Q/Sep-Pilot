"""Real SDK acceptance for platform-default and encrypted per-user model connections."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import time
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import auth
from agents.agent import Agent
from agents.config import AgentConfig
import agents.config as config_module
from agents.session import Session
from agents.state_io import write_checkpoint


def safe_endpoint(url: str) -> str:
    parsed = urlsplit(url)
    return f"{parsed.scheme}://{parsed.hostname or 'default'}{parsed.path or ''}"


def probe(config: AgentConfig, label: str) -> dict:
    agent = Agent(name="connection-probe", instructions=(
        "You are a connection health probe. Reply with exactly CONNECTION_OK and do not add anything."
    ), functions=[], model=None, max_turns=1)
    session = Session(config=config)
    session.messages = [{"role": "user", "content": "Reply exactly CONNECTION_OK."}]
    response = session._call_api(agent)
    answer = "\n".join(block.text for block in response.content if getattr(block, "type", "") == "text")
    return {"mode": label, "ok": "CONNECTION_OK" in answer, "model": config.model,
            "endpoint": safe_endpoint(config.base_url), "key_present": bool(config.api_key),
            "force_model_override": bool(config.force_model_override)}


def main() -> None:
    default = AgentConfig.from_env_dir()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "runs" / "sdk_connection" / stamp
    run_dir.mkdir(parents=True, exist_ok=False)
    results = [probe(default, "platform_default")]
    original = (auth.USER_DB, auth.TOKEN_DB, auth.CONVERSATION_LOG, auth.CREDENTIAL_KEY_FILE, config_module._config)
    try:
        with tempfile.TemporaryDirectory(prefix="credential-test-", dir=run_dir) as scratch:
            scratch = Path(scratch)
            auth.USER_DB = scratch / "users.json"
            auth.TOKEN_DB = scratch / "tokens.json"
            auth.CONVERSATION_LOG = scratch / "conversation_logs.jsonl"
            auth.CREDENTIAL_KEY_FILE = scratch / "credential_master.key"
            config_module._config = default
            auth.USER_DB.write_text(json.dumps({"connection-test": {
                "password_hash": auth._hash_password("temporary-password")
            }}))
            auth.save_model_connection("connection-test", "custom", default.base_url,
                                       default.model, default.api_key)
            raw = auth.USER_DB.read_text()
            if default.api_key and default.api_key in raw:
                raise RuntimeError("API key leaked into user database")
            custom = auth.config_for_user("connection-test")
            results.append(probe(custom, "encrypted_user_custom"))
    finally:
        auth.USER_DB, auth.TOKEN_DB, auth.CONVERSATION_LOG, auth.CREDENTIAL_KEY_FILE, config_module._config = original
    report = {"kind": "REAL_SDK_USER_MODEL_CONNECTION", "passed": all(item["ok"] for item in results),
              "results": results, "note": "No API key is included in this report."}
    write_checkpoint(run_dir / "test_result.json", report)
    print(json.dumps({"passed": report["passed"], "report": str(run_dir / "test_result.json")}, ensure_ascii=False))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
