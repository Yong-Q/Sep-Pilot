import json

from agents.registry import get_registry


def _manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({
        "split": {"train": ["a", "b"], "valid": ["v"], "test": ["t"]},
        "normalization": {"source": "train_labels_only"},
        "source_csv_sha256": "a" * 64,
        "label_source": "measured uptake",
    }))
    return path


def test_model_template_tool_reads_manifest_and_returns_reviewable_stages(tmp_path):
    path = _manifest(tmp_path)
    result = get_registry().execute_dict("ml_template_plan", {
        "manifest_path": str(path), "scientific_target": "adsorption_uptake",
        "architecture": "joint_physics",
    })
    assert result["execution"] == "template_only"
    assert result["model_blocks"][-1] == "physics_residual_adapter"
    assert result["test_used_for_selection"] is False
    assert str(path) not in json.dumps(result)


def test_model_template_can_record_missing_label_source_as_unverified_declaration(tmp_path):
    path = _manifest(tmp_path)
    manifest = json.loads(path.read_text())
    manifest.pop("label_source")
    path.write_text(json.dumps(manifest))
    without = get_registry().execute_dict("ml_template_plan", {
        "manifest_path": str(path), "scientific_target": "adsorption_uptake"})
    assert "error" in without
    with_declaration = get_registry().execute_dict("ml_template_plan", {
        "manifest_path": str(path), "scientific_target": "adsorption_uptake",
        "label_source": "simulated adsorption uptake from documented source dataset"})
    assert with_declaration["execution"] == "template_only"
    assert with_declaration["label_source_recorded_from"] == "tool_argument_unverified"


def test_active_learning_template_tool_rejects_test_candidate(tmp_path):
    path = _manifest(tmp_path)
    result = get_registry().execute_dict("ml_active_learning_plan", {
        "manifest_path": str(path), "revealed_labels": {"a": [1.0]},
        "candidate_ids": ["t"],
    })
    assert "error" in result
    assert "TRAIN" in result["error"]


def test_template_tools_are_available_to_scientific_agents():
    from agents.defns import ANALYST, ORCHESTRATOR
    assert {"ml_template_plan", "ml_active_learning_plan"} <= {
        fn.__name__ for fn in ANALYST.functions}
    assert "ml_template_plan" in {fn.__name__ for fn in ORCHESTRATOR.functions}


def test_manifest_path_obeys_private_session_scope(tmp_path):
    from agents.workspace import tool_scope_issues
    other = tmp_path / "runs" / "other" / "conversation" / "manifest.json"
    issues = tool_scope_issues({"manifest_path": str(other)}, "current", "conversation", tmp_path)
    assert issues


def test_direct_template_execution_cannot_read_other_session(tmp_path, monkeypatch):
    from agents.watch_context import set_context, clear_context
    from agents import registry
    other = tmp_path / "runs" / "other" / "conversation" / "manifest.json"
    other.parent.mkdir(parents=True)
    other.write_text("{}")
    monkeypatch.setattr(registry, "get_config", lambda: type("Config", (), {"project_root": tmp_path})())
    try:
        set_context("current", "conversation", "analyst")
        result = get_registry().execute_dict("ml_template_plan", {
            "manifest_path": str(other), "scientific_target": "uptake"})
    finally:
        clear_context()
    assert "error" in result
    assert "other" not in json.dumps(result)


def test_planning_tools_do_not_claim_mutating_runtime_resources():
    from agents.parallel_workflow import READ_ONLY
    assert {"ml_template_plan", "ml_active_learning_plan"} <= READ_ONLY
