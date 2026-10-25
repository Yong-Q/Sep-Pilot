import json
from types import SimpleNamespace

from agents.agent import Agent
from agents.config import AgentConfig
from agents.session import Session


def test_private_settings_can_select_codex_as_primary_model(tmp_path):
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps({"model_provider": "codex_cli", "env": {"ANTHROPIC_AUTH_TOKEN": "expired"}}))
    config = AgentConfig.from_env_dir(tmp_path)
    assert config.model_provider == "codex_cli"
    assert config.model == "gpt-6-sol"
    assert config.force_model_override


def test_provider_override_is_normalized_before_role_models_are_selected(tmp_path):
    (tmp_path / "settings.json").write_text(json.dumps({"env": {"ANTHROPIC_MODEL": "old-api-model"}}))
    config = AgentConfig.from_env_dir(tmp_path, model_provider="codex_cli")
    assert config.model == "gpt-6-sol"
    assert config.orchestrator_model == config.specialist_model == "gpt-6-sol"
    assert config.force_model_override


def test_environment_only_provider_override_is_normalized(tmp_path, monkeypatch):
    import agents.config
    monkeypatch.setattr(agents.config, "ENV_DIR", tmp_path)
    monkeypatch.setenv("ANTHROPIC_MODEL", "old-api-model")
    config = AgentConfig.from_env(model_provider="codex_cli")
    assert config.model == config.orchestrator_model == config.specialist_model == "gpt-6-sol"


def test_session_routes_primary_turn_through_codex_provider(tmp_path, monkeypatch):
    from agents import codex_provider
    seen = {}
    def fake_provider(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Science answer")], stop_reason="end_turn")
    monkeypatch.setattr(codex_provider, "call_codex_model", fake_provider)
    session = Session(config=AgentConfig(api_key="expired", project_root=tmp_path,
                                         model="gpt-6-sol", model_provider="codex_cli"))
    monkeypatch.setattr(session, "_stream_api", lambda **kwargs: (_ for _ in ()).throw(AssertionError("old proxy used")))
    session.messages = [{"role": "user", "content": "What can uptake show about a membrane?"}]
    result = session._call_api(Agent(name="analyst", instructions="Answer from evidence.", functions=[]))
    assert result.content[0].text == "Science answer"
    assert seen["model"] == "gpt-6-sol"
    assert seen["messages"]


def test_user_supplied_api_stays_on_its_own_provider(monkeypatch):
    import auth
    import agents.config
    monkeypatch.setattr(agents.config, "get_config", lambda: AgentConfig(model_provider="codex_cli"))
    monkeypatch.setattr(auth, "load_users", lambda: {"scientist": {"model_connection": {
        "mode": "custom", "base_url": "https://example.invalid/api", "model": "user-model",
        "api_key_ciphertext": "encrypted-placeholder"}}})
    monkeypatch.setattr(auth, "_decrypt_secret", lambda value: "user-provided-key")
    config = auth.config_for_user("scientist")
    assert config.model_provider == "anthropic"
    assert config.model == "user-model"


def test_default_connection_displays_codex_without_old_proxy(monkeypatch):
    import auth
    import agents.config
    monkeypatch.setattr(agents.config, "get_config", lambda: AgentConfig(
        api_key="expired", base_url="https://old-proxy.invalid",
        model_provider="codex_cli"))
    monkeypatch.setattr(auth, "load_users", lambda: {"scientist": {}})
    public = auth.public_model_connection("scientist")
    assert public["model"] == "gpt-6-sol"
    assert public["base_url"] == ""
    assert "expired" not in json.dumps(public)
