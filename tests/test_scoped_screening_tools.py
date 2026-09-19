import csv
from pathlib import Path

from agents import registry
from agents.defns import ADSORPTION, ANALYST, COMMUNICATOR, HARNESS
from agents.workflow_patch import workflow_efficiency_issues


def _scope(tmp_path, monkeypatch):
    root = tmp_path / "runs" / "u" / "c"
    monkeypatch.setattr("agents.workspace.session_root", lambda *a, **k: root)
    monkeypatch.setattr(registry, "get_config", lambda: type("C", (), {"project_root": tmp_path})())
    return root


def test_stage_cif_subset_is_sorted_hashed_scoped_and_idempotent(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch)
    source = tmp_path / "source"; source.mkdir()
    for name in ("c.cif", "a.cif", "b.cif"):
        (source / name).write_text("data_" + name)
    output = root / "inputs" / "first_two"
    args = {"cif_dir": str(source), "output_dir": str(output), "limit": 2}
    result = registry._exec_stage_cif_subset(args)
    assert result["status"] == "created"
    assert [row["name"] for row in result["files"]] == ["a.cif", "b.cif"]
    assert all(len(row["sha256"]) == 64 for row in result["files"])
    assert registry._exec_stage_cif_subset(args)["status"] == "reused"
    (output / "a.cif").write_text("tampered")
    assert registry._exec_stage_cif_subset(args)["error_code"] == "OUTPUT_NOT_EMPTY"
    outside = registry._exec_stage_cif_subset({**args, "output_dir": str(tmp_path / "outside")})
    assert outside["error_code"] == "OUTPUT_OUTSIDE_SESSION"


def _write_result(root: Path, gas: str, material: str, loading: float, sd: float):
    path = root / gas / "298K" / material / "Output" / "System_0" / "result.data"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"Average loading absolute [mol/kg framework] {loading} +/- {sd} [-]\n")


def test_gcmc_screening_analysis_uses_real_molkg_evidence_and_declares_ratio(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch); work = root / "gcmc" / "batch"
    _write_result(work, "CO2", "m1", 4.0, .2); _write_result(work, "N2", "m1", .5, .05)
    _write_result(work, "CO2", "m2", 3.0, .1); _write_result(work, "N2", "m2", 1.0, .1)
    output_csv, output_md = root / "analysis" / "ranking.csv", root / "analysis" / "ranking.md"
    result = registry._exec_analyze_gcmc_screening({
        "work_dir": str(work), "gases": ["CO2", "N2"],
        "output_csv": str(output_csv), "output_markdown": str(output_md)})
    assert result["ok"] and result["synthetic_data_used"] is False
    assert result["rows"][0]["material"] == "m1"
    assert result["rows"][0]["CO2_N2_uptake_ratio"] == 8.0
    assert "not IAST selectivity" in output_md.read_text()
    assert list(csv.DictReader(output_csv.open()))[0]["loading_unit"] == "mol/kg_framework"


def test_incomplete_gcmc_evidence_fails_without_partial_report(tmp_path, monkeypatch):
    root = _scope(tmp_path, monkeypatch); work = root / "gcmc" / "batch"
    _write_result(work, "CO2", "m1", 4.0, .2)
    result = registry._exec_analyze_gcmc_screening({
        "work_dir": str(work), "gases": ["CO2", "N2"],
        "output_csv": str(root / "a.csv"), "output_markdown": str(root / "a.md")})
    assert result["error_code"] == "INCOMPLETE_GCMC_EVIDENCE"
    assert not (root / "a.csv").exists() and not (root / "a.md").exists()


def test_scientific_shell_after_batch_is_rejected_in_favor_of_typed_parser(tmp_path):
    nodes = [
        {"step_id": "batch", "agent": "adsorption", "tool": "run_gcmc_batch",
         "arguments": {"cif_dir": "cifs", "gases": ["CO2", "N2"], "temperature": 298, "pressure": 1},
         "depends_on": [], "expected_outputs": [], "resource_locks": []},
        {"step_id": "fake", "agent": "harness", "tool": "run_bash",
         "arguments": {"command": "echo placeholder"}, "depends_on": ["batch"],
         "expected_outputs": [], "resource_locks": []},
    ]
    assert any(i["kind"] == "untyped_gcmc_analysis" for i in workflow_efficiency_issues(nodes, tmp_path))


def test_new_tools_are_exposed_only_to_appropriate_agents():
    assert "stage_cif_subset" in {f.__name__ for f in HARNESS.functions}
    assert "analyze_gcmc_screening" in {f.__name__ for f in ADSORPTION.functions}
    assert "analyze_gcmc_screening" in {f.__name__ for f in ANALYST.functions}
    assert "generate_scientific_report" in {f.__name__ for f in COMMUNICATOR.functions}
