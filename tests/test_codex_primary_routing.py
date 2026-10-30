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


def test_planner_receives_current_draft_schemas_without_schema_tool_roundtrip(tmp_path, monkeypatch):
    from agents.defns import ORCHESTRATOR
    from agents.registry import get_registry
    seen = {}
    def provider(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='Plan ready')], stop_reason='end_turn')
    monkeypatch.setattr('agents.codex_provider.call_codex_model', provider)
    session = Session(config=AgentConfig(model_provider='codex_cli', project_root=tmp_path))
    session.context['workflow_draft'] = {'nodes': [
        {'tool': 'generate_structure'}, {'tool': 'run_cdft'}, {'tool': 'run_cdft'}]}
    session.messages = [{'role': 'user', 'content': 'Compile the saved draft.'}]
    session._call_api(ORCHESTRATOR)
    provided = json.loads(seen['system'].split('[CURRENT DRAFT TOOL SCHEMAS]\n')[1].split('\n', 1)[0])
    assert [item['name'] for item in provided] == ['generate_structure', 'run_cdft']
    assert provided[1]['input_schema'] == get_registry().get('run_cdft').input_schema
    assert not session.memory.tool_call_log


def test_model_timing_records_sizes_without_prompt_or_credentials(tmp_path, monkeypatch):
    monkeypatch.setattr('agents.codex_provider.call_codex_model', lambda **kwargs: SimpleNamespace(
        content=[SimpleNamespace(type='text', text='Ready')], stop_reason='end_turn'))
    session = Session(config=AgentConfig(model_provider='codex_cli', project_root=tmp_path))
    session.messages = [{'role': 'user', 'content': 'private scientific request'}]
    session._call_api(Agent(name='analyst', instructions='private instructions', functions=[]))
    metric = session.context['model_call_metrics'][-1]
    assert metric['elapsed_seconds'] >= 0
    assert metric['input_characters'] > 0
    assert metric['tool_count'] == 0
    assert 'private' not in json.dumps(metric)


def test_read_only_request_does_not_require_research_dag(tmp_path, monkeypatch):
    from agents.defns import ORCHESTRATOR
    from agents.goal_contract import GoalContract
    seen = {}
    def provider(**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type='text', text='No jobs')], stop_reason='end_turn')
    monkeypatch.setattr('agents.codex_provider.call_codex_model', provider)
    session = Session(config=AgentConfig(model_provider='codex_cli', project_root=tmp_path))
    session._current_line_id = 'read-only'
    session._on_workflow_start = lambda *_: None
    session.goal_contract = GoalContract.from_user_message('只查看当前会话作业状态，不提交计算')
    session.messages = [{'role': 'user', 'content': '只查看当前会话作业状态，不提交计算'}]
    assert session.goal_contract.execution_mode == 'read_only'
    assert not session._chain_required_block('inspect_path')
    assert session._chain_required_block('generate_structure')
    session._call_api(ORCHESTRATOR)
    assert '[EXECUTABLE WORKFLOW REQUIRED]' not in seen['system']


def test_user_supplied_api_stays_on_its_own_provider(monkeypatch):
    import auth
    import agents.config
    monkeypatch.setattr(agents.config, "get_config", lambda: AgentConfig(model_provider="codex_cli"))
    metadata_calls=[]
    def metadata_snapshot(username):
        metadata_calls.append(username)
        assert username == "scientist"
        return {"model_connection": {"mode": "custom", "base_url": "https://example.invalid/api",
            "model": "user-model", "api_key_ciphertext": "encrypted-placeholder"}}
    monkeypatch.setattr(auth, "user_metadata_snapshot", metadata_snapshot)
    monkeypatch.setattr(auth, "_decrypt_secret", lambda value: "user-provided-key")
    config = auth.config_for_user("scientist")
    assert config.model_provider == "anthropic"
    assert config.model == "user-model"
    assert metadata_calls == ["scientist"]
    assert agents.config.get_config().model_provider == "codex_cli"


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
