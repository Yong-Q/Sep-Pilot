import hashlib
import json
import stat

import auth
from agents.config import AgentConfig


def isolate(monkeypatch, tmp_path):
    monkeypatch.setattr(auth, "USER_DB", tmp_path / "users.json")
    monkeypatch.setattr(auth, "TOKEN_DB", tmp_path / "tokens.json")
    monkeypatch.setattr(auth, "CONVERSATION_LOG", tmp_path / "conversation_logs.jsonl")
    monkeypatch.setattr(auth, "CREDENTIAL_KEY_FILE", tmp_path / "credential_master.key")


def test_passwords_are_bcrypt_and_login_upgrades_metadata(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    result = auth.register_user("alice", "correct-horse", "Alice")
    assert result["ok"]
    saved = json.loads(auth.USER_DB.read_text())["alice"]
    assert saved["password_hash"].startswith("bcrypt$")
    assert saved["avatar_id"] in auth.AVATAR_IDS
    assert result["avatar_url"] == "/avatars/" + saved["avatar_id"]
    assert "correct-horse" not in auth.USER_DB.read_text() and "password" not in saved
    assert auth.login_user("alice", "wrong")["ok"] is False
    assert auth.login_user("alice", "correct-horse")["ok"] is True
    assert stat.S_IMODE(auth.USER_DB.stat().st_mode) == 0o600


def test_legacy_password_and_plain_token_are_hardened_without_plaintext(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    salt, password = "abc123", "legacy-pass"
    digest = hashlib.sha256((salt + password).encode()).hexdigest()
    auth.USER_DB.write_text(json.dumps({"legacy": {"password": f"{salt}:{digest}", "display_name": "Legacy"}}))
    auth.TOKEN_DB.write_text(json.dumps({"raw-session-token": "legacy"}))
    report = auth.migrate_auth_storage()
    users = json.loads(auth.USER_DB.read_text())
    tokens = json.loads(auth.TOKEN_DB.read_text())
    assert report == {"passwords": 1, "tokens": 1}
    assert users["legacy"]["avatar_id"] in auth.AVATAR_IDS
    assert users["legacy"]["password_hash"].startswith("bcrypt-sha256$")
    assert "password" not in users["legacy"]
    assert "raw-session-token" not in tokens
    assert auth.login_user("legacy", password)["ok"]


def test_tokens_are_digest_only_and_expire(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    auth.TOKEN_DB.write_text("{}")
    manager = auth.UserSessionManager()
    manager.register_token("browser-secret", "alice")
    persisted = json.loads(auth.TOKEN_DB.read_text())
    assert "browser-secret" not in auth.TOKEN_DB.read_text()
    assert next(iter(persisted)).startswith("sha256:")
    assert manager.get_username("browser-secret") == "alice"
    entry = next(iter(persisted.values())); entry["expires_at"] = 1
    auth.TOKEN_DB.write_text(json.dumps(persisted)); manager._tokens = persisted
    assert manager.get_username("browser-secret") is None


def test_custom_model_key_is_encrypted_and_user_scoped(monkeypatch, tmp_path):
    isolate(monkeypatch, tmp_path)
    import agents.config as config_module
    monkeypatch.setattr(config_module, "_config", AgentConfig(api_key="platform-key", base_url="https://platform.invalid", model="platform-model"))
    users = {
        "alice": {"password_hash": auth._hash_password("password-a")},
        "bob": {"password_hash": auth._hash_password("password-b")},
    }
    auth.USER_DB.write_text(json.dumps(users))
    visible = auth.save_model_connection("alice", "custom", "https://models.example/v1", "research-model", "alice-secret-key")
    raw = auth.USER_DB.read_text()
    assert "alice-secret-key" not in raw
    assert visible["api_key_hint"] == "••••-key" and visible["key_configured"]
    alice = auth.config_for_user("alice")
    bob = auth.config_for_user("bob")
    assert (alice.api_key, alice.base_url, alice.model, alice.force_model_override) == (
        "alice-secret-key", "https://models.example/v1", "research-model", True)
    assert (bob.api_key, bob.model, bob.force_model_override) == ("platform-key", "platform-model", False)
    assert stat.S_IMODE(auth.CREDENTIAL_KEY_FILE.stat().st_mode) == 0o600
