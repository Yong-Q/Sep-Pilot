from agents.workflow_patch import workflow_efficiency_issues, canonical_expected_outputs, operational_patch
from agents.parallel_workflow import resources_for, conflicts
from agents.registry import get_registry


def node(step_id, tool, arguments, depends_on=None, outputs=None):
    return {
        "step_id": step_id,
        "agent": "adsorption",
        "tool": tool,
        "arguments": arguments,
        "depends_on": depends_on or [],
        "expected_outputs": outputs or [],
        "resource_locks": [],
    }


def test_identical_dispatch_contracts_are_rejected_as_redundant():
    nodes = [
        node("a", "run_henry", {"cif": "a.cif", "gas": "CO2", "temperature": 298}),
        node("b", "run_henry", {"cif": "a.cif", "gas": "CO2", "temperature": 298}),
    ]
    issues = workflow_efficiency_issues(nodes, "/project")
    assert any(issue["kind"] == "duplicate_dispatch" for issue in issues)


def test_gcmc_batch_gases_must_be_one_node_and_one_submission():
    common = {"cif_dir": "/project/cifs", "temperature": 298, "pressure": 1.0}
    nodes = [
        node("co2", "run_gcmc_batch", {**common, "gases": ["CO2"]}),
        node("n2", "run_gcmc_batch", {**common, "gases": ["N2"]}),
    ]
    issues = workflow_efficiency_issues(nodes, "/project")
    fragmented = next(issue for issue in issues if issue["kind"] == "fragmented_batch")
    assert fragmented["gases"] == ["CO2", "N2"]


def test_output_directory_alone_cannot_disguise_a_duplicate_batch(tmp_path):
    common = {"cif_dir": "/project/cifs", "gases": ["CO2"],
              "temperature": 298, "pressure": 1.0, "force_field": "UFF"}
    nodes = [
        node("a", "run_gcmc_batch", {**common, "output_dir": str(tmp_path / "a")}),
        node("b", "run_gcmc_batch", {**common, "output_dir": str(tmp_path / "b")}),
    ]
    issues = workflow_efficiency_issues(nodes, "/project")
    assert any(issue["kind"] == "fragmented_batch" for issue in issues)


def test_complete_isotherm_pressure_scan_is_one_node():
    common = {"cif": "/project/a.cif", "gas": "CO2", "temperature": 298}
    nodes = [
        node("low", "run_gcmc_isotherm", {**common, "pressure_start": .1, "pressure_end": 1, "n_pressure_points": 5}),
        node("high", "run_gcmc_isotherm", {**common, "pressure_start": 1, "pressure_end": 10, "n_pressure_points": 5}),
    ]
    issues = workflow_efficiency_issues(nodes, "/project")
    assert any(issue["kind"] == "fragmented_scan" for issue in issues)


def test_explicit_artifact_consumer_requires_dependency(tmp_path):
    structures = tmp_path / "structures"
    nodes = [
        node("generate", "generate_structure", {"output_dir": str(structures)},
             outputs=[{"kind": "directory", "path": str(structures), "pattern": "*.cif", "min_count": 1}]),
        node("charge", "run_pacman_charge", {"cif_dir": str(structures), "method": "pacmof"}),
    ]
    issues = workflow_efficiency_issues(nodes, tmp_path)
    missing = next(issue for issue in issues if issue["kind"] == "missing_data_dependency")
    assert missing["producer"] == "generate"
    assert missing["consumer"] == "charge"

    nodes[1]["depends_on"] = ["generate"]
    assert not [issue for issue in workflow_efficiency_issues(nodes, tmp_path)
                if issue["kind"] == "missing_data_dependency"]


def test_independent_nodes_are_not_forced_into_serial_dependencies(tmp_path):
    nodes = [
        node("co2", "run_henry", {"cif": str(tmp_path / "a.cif"), "gas": "CO2", "temperature": 298}),
        node("n2", "run_henry", {"cif": str(tmp_path / "b.cif"), "gas": "N2", "temperature": 298}),
    ]
    assert workflow_efficiency_issues(nodes, tmp_path) == []


def test_dynamic_report_contract_uses_upstream_step_evidence(tmp_path):
    report = tmp_path / "session" / "acceptance_report.md"
    arguments = {
        "source_steps": ["inspect", "convert"],
        "output_path": str(report),
        "title": "可核查验收报告",
    }
    assert get_registry().validate_params("generate_scientific_report", arguments) == []
    contract = node("report", "generate_scientific_report", arguments,
                    depends_on=["inspect", "convert"], outputs=[str(report)])
    resources = resources_for(contract, tmp_path, tmp_path / "session")
    assert {item["key"]: item["mode"] for item in resources}["path:" + str(report.resolve())] == "write"


def test_report_schema_names_missing_output_path_instead_of_vague_anyof_error():
    issues = get_registry().validate_params('generate_scientific_report', {
        'source_steps': ['collect'], 'title': 'Results',
    })
    assert any('missing required parameter(s)' in issue and 'output_path' in issue for issue in issues)
    assert not any('not valid under any of the given schemas' in issue for issue in issues)
    receipt = get_registry().execute_dict('generate_scientific_report', {
        'source_steps': ['collect'], 'title': 'Results',
    })
    assert receipt['error_kind'] == 'schema_validation'
    assert receipt['missing_parameters'] == ['output_path']


def test_workflow_draft_schema_accepts_output_contracts_used_by_compiler():
    issues = get_registry().validate_params('record_workflow_draft', {
        'completion_criteria': 'A verified report',
        'nodes': [{
            'step_id': 'report', 'description': 'write report', 'agent': 'communicator',
            'tool': 'generate_scientific_report', 'depends_on': [],
            'arguments': {'title': 'Report'},
            'input_bindings': {
                'output_path': {'step_id': 'source', 'output_index': 0},
            },
            'expected_outputs': [{'kind': 'file', 'path': 'report.md'}],
            'resource_locks': ['report-path'],
        }],
    })
    assert issues == []


def test_workflow_patch_schema_accepts_structured_input_bindings():
    issues = get_registry().validate_params('propose_workflow_patch', {
        'base_version': 0,
        'reason': 'Bind one approved producer artifact into its consumer',
        'changes': [{'operation': 'upsert', 'step_id': 'consume', 'node': {
            'agent': 'analyst', 'tool': 'build_mof_database', 'arguments': {},
            'input_bindings': {
                'cif_dir': {'step_id': 'generate', 'output_index': 0},
            },
            'depends_on': ['generate'], 'expected_outputs': [],
        }}],
    })
    assert issues == []


def test_gcmc_batch_branches_with_distinct_outputs_can_run_in_parallel(tmp_path):
    session = tmp_path / "runs" / "u" / "c"
    shared = tmp_path / "charged"
    generic = node("generic", "run_gcmc_batch", {
        "cif_dir": str(shared), "gases": ["CO2"], "temperature": 298,
        "pressure": 1.0, "force_field": "GenericMOFs",
        "output_dir": str(session / "gcmc" / "generic"),
    })
    uff = node("uff", "run_gcmc_batch", {
        "cif_dir": str(shared), "gases": ["CO2"], "temperature": 298,
        "pressure": 1.0, "force_field": "UFF",
        "output_dir": str(session / "gcmc" / "uff"),
    })
    a = resources_for(generic, tmp_path, session)
    b = resources_for(uff, tmp_path, session)
    assert not any(conflicts(x, y) for x in a for y in b)


def test_gcmc_batch_schema_exposes_branch_directory_not_ignored_csv():
    schema = get_registry().get("run_gcmc_batch").input_schema
    assert "output_dir" in schema["properties"]
    assert "output_csv" not in schema["properties"]


def test_project_anchored_expected_outputs_are_not_rejoined_to_artifact_base(tmp_path):
    values = canonical_expected_outputs([
        "runs/u/c/result.csv",
        {"kind": "directory", "path": "runs/u/c/subset", "pattern": "*.cif", "min_count": 2},
        "bare.csv",
    ], tmp_path)
    assert values[0] == str((tmp_path / "runs/u/c/result.csv").resolve())
    assert values[1]["path"] == str((tmp_path / "runs/u/c/subset").resolve())
    assert values[2] == "bare.csv"


def test_path_normalization_only_is_an_operational_patch(tmp_path):
    before = node("stage", "stage_cif_subset", {
        "cif_dir": str(tmp_path / "source"),
        "output_dir": str(tmp_path / "runs/u/c/subset"), "limit": 6})
    before["agent"] = "harness"
    before["expected_outputs"] = [{"kind": "directory", "path": "runs/u/c/subset", "pattern": "*.cif", "min_count": 6}]
    after = {**before, "expected_outputs": [{"kind": "directory", "path": str(tmp_path / "runs/u/c/subset"), "pattern": "*.cif", "min_count": 6}]}
    assert operational_patch([before], [{"operation": "upsert", "step_id": "stage", "node": after}], tmp_path)
    weakened = {**after, "expected_outputs": [{"kind": "directory", "path": str(tmp_path / "runs/u/c/subset"), "pattern": "*.cif", "min_count": 1}]}
    assert not operational_patch([before], [{"operation": "upsert", "step_id": "stage", "node": weakened}], tmp_path)
