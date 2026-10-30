from agents.workflow_patch import workflow_efficiency_issues, canonical_expected_outputs, operational_patch
from agents.parallel_workflow import resources_for, conflicts
from agents.registry import get_registry
from agents.workflow_compiler import compile_dataflow_bindings, expand_parallel_report_nodes


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


def test_batch_charge_validation_accepts_directory_artifact_binding(tmp_path):
    root = tmp_path / "runs" / "u" / "c"
    structures = root / "charged"
    report = root / "charge_validation.json"
    changes = [
        {"operation": "upsert", "step_id": "charge", "node": node(
            "charge", "run_pacman_charge", {"cif_dir": str(root / "input")},
            outputs=[{"kind": "directory", "path": str(structures),
                      "pattern": "*_pacmof.cif", "min_count": 1}])},
        {"operation": "upsert", "step_id": "validate", "node": node(
            "validate", "validate_framework_charges", {
                "expected_net_charge": 0,
                "charge_source": "pacmof",
                "output_path": str(report),
            }, depends_on=["charge"], outputs=[str(report)])},
    ]
    changes[1]["node"]["input_bindings"] = {
        "cif_path": {"step_id": "charge", "output_index": 0},
    }

    compiled = compile_dataflow_bindings(changes, get_registry(), tmp_path, root)

    assert compiled[1]["node"]["arguments"]["cif_path"] == str(structures)


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


def test_parallel_report_expands_independent_source_fragments(tmp_path):
    report = tmp_path / 'session' / 'report.md'
    changes = [
        {'operation': 'upsert', 'step_id': 'a', 'node': node(
            'a', 'read_file', {'path': str(tmp_path / 'a.txt')}, outputs=[str(tmp_path / 'a.out')])},
        {'operation': 'upsert', 'step_id': 'b', 'node': node(
            'b', 'read_file', {'path': str(tmp_path / 'b.txt')}, outputs=[str(tmp_path / 'b.out')])},
        {'operation': 'upsert', 'step_id': 'final', 'node': node(
            'final', 'generate_scientific_report', {
                'source_steps': ['a', 'b'], 'output_path': str(report), 'title': 'Results',
            }, depends_on=['a', 'b'], outputs=[str(report)])},
    ]

    expanded = expand_parallel_report_nodes(changes, tmp_path / 'session')
    nodes = {change['step_id']: change['node'] for change in expanded}

    assert nodes['final__fragment__a']['depends_on'] == ['a']
    assert nodes['final__fragment__b']['depends_on'] == ['b']
    assert nodes['final']['depends_on'] == [
        'final__fragment__a', 'final__fragment__b']
    assert nodes['final']['arguments']['output_path'] == str(report)
    assert nodes['final']['report_role'] == 'assembly'
    assert nodes['final__fragment__a']['report_role'] == 'fragment'
    assert nodes['final__fragment__a']['report_parent'] == 'final'
    assert nodes['final']['arguments']['report_mode'] == 'assembly'
    assert nodes['final__fragment__a']['arguments']['report_mode'] == 'fragment'


def test_parallel_report_keeps_single_source_contract_compatible(tmp_path):
    report = tmp_path / 'session' / 'report.md'
    changes = [{'operation': 'upsert', 'step_id': 'final', 'node': node(
        'final', 'generate_scientific_report', {
            'source_steps': ['a'], 'output_path': str(report), 'title': 'Results',
        }, depends_on=['a'], outputs=[str(report)])}]

    assert expand_parallel_report_nodes(changes, tmp_path / 'session') == changes


def test_report_compilation_is_idempotent_when_replaying_saved_patch(tmp_path):
    root = tmp_path / 'session'
    changes = [{'operation': 'upsert', 'step_id': 'final', 'node': node(
        'final', 'generate_scientific_report', {
            'source_steps': ['simulation', 'analysis'],
            'output_path': str(root / 'report.md'), 'title': 'Results',
        }, depends_on=['simulation', 'analysis'], outputs=[str(root / 'report.md')])}]
    compiled = expand_parallel_report_nodes(changes, root)

    assert expand_parallel_report_nodes(compiled, root) == compiled


def test_parallel_report_patch_adds_ml_branch_without_replacing_existing_fragments(tmp_path):
    root = tmp_path / 'session'
    report = root / 'report.md'
    initial = expand_parallel_report_nodes([{
        'operation': 'upsert', 'step_id': 'final', 'node': node(
            'final', 'generate_scientific_report', {
                'source_steps': ['simulation', 'analysis'],
                'output_path': str(report), 'title': 'Results',
            }, depends_on=['simulation', 'analysis'], outputs=[str(report)])
    }], root)
    existing = [change['node'] for change in initial]
    ml = node('train_ml', 'ml_train', {
        'data_csv': str(root / 'training.csv'), 'target': 'uptake',
        'model_type': 'RF', 'output_dir': str(root / 'ml'),
    }, depends_on=['analysis'], outputs=[str(root / 'ml')])
    patch = expand_parallel_report_nodes([
        {'operation': 'upsert', 'step_id': 'train_ml', 'node': ml},
        {'operation': 'upsert', 'step_id': 'final', 'node': node(
            'final', 'generate_scientific_report', {
                'source_steps': ['simulation', 'analysis', 'train_ml'],
                'output_path': str(report), 'title': 'Results',
            }, depends_on=['simulation', 'analysis', 'train_ml'], outputs=[str(report)])},
    ], root, existing_steps=existing)
    changed = {change['step_id']: change['node'] for change in patch}

    assert changed['final__fragment__simulation'] == existing[0]
    assert changed['final__fragment__analysis'] == existing[1]
    assert changed['final__fragment__train_ml']['depends_on'] == ['train_ml']
    assert changed['final']['depends_on'][-1] == 'final__fragment__train_ml'
    assert changed['final']['arguments']['report_mode'] == 'assembly'


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


def _revision_report(root):
    return [{'operation': 'upsert', 'step_id': 'final', 'node': node(
        'final', 'generate_scientific_report', {
            'source_steps': ['simulation', 'analysis'],
            'output_path': str(root / 'report.md'), 'title': 'Results',
        }, depends_on=['simulation', 'analysis'], outputs=[str(root / 'report.md')])}]


def test_fragment_revision_new_plan_and_source_get_distinct_immutable_paths(tmp_path):
    root = tmp_path / 'session'
    original = expand_parallel_report_nodes(_revision_report(root), root, target_plan_version=1,
        source_evidence={'simulation': {'result_ref': {'call_id': 'old'}}})
    existing = [change['node'] for change in original]
    revised = expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=2, source_evidence={'simulation': {'result_ref': {'call_id': 'new'}}})
    assert original[0]['node']['arguments']['output_path'] != revised[0]['node']['arguments']['output_path']
    changed_same_version = expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=1, source_evidence={'simulation': {'result_ref': {'call_id': 'new'}}})
    assert original[0]['node']['arguments']['output_path'] != changed_same_version[0]['node']['arguments']['output_path']
    assert expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=2, source_evidence={'simulation': {'result_ref': {'call_id': 'new'}}}) == revised


def test_fragment_revision_reuses_verified_unchanged_source_only(tmp_path):
    root = tmp_path / 'session'
    existing = [c['node'] for c in expand_parallel_report_nodes(_revision_report(root), root)]
    runtime = {
        'simulation': {'status': 'succeeded', 'execution_fingerprint': 'attempt',
                       'result_ref': {'call_id': 'source-call'}},
        'final__fragment__simulation': {'status': 'succeeded', 'contract': existing[0],
            'phase': 'finish', 'artifacts': {existing[0]['arguments']['output_path']: {'size': 123}},
            'result': {'source_step': 'simulation', 'source_attempt_fingerprint': 'attempt',
                       'source_evidence_call_id': 'source-call', 'fragment_fingerprint': 'a' * 64,
                       'report_file': existing[0]['arguments']['output_path']}},
    }
    updated = expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=2, source_evidence=runtime)
    assert updated[0]['node'] == existing[0]
    runtime['simulation']['result_ref']['call_id'] = 'different-call'
    changed = expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=2, source_evidence=runtime)
    assert changed[0]['node']['arguments']['output_path'] != existing[0]['arguments']['output_path']


def test_fragment_revision_keeps_formally_proposed_recovery_path(tmp_path):
    root = tmp_path / 'session'
    initial = expand_parallel_report_nodes(_revision_report(root), root)
    existing = [c['node'] for c in initial]
    import copy
    repaired = copy.deepcopy(initial[0])
    replacement = str(root / 'reports' / 'new-approved-fragment.md')
    repaired['node']['arguments']['output_path'] = replacement
    repaired['node']['expected_outputs'] = [replacement]
    patch = expand_parallel_report_nodes([repaired] + _revision_report(root), root, existing,
        target_plan_version=2)
    matches = [c for c in patch if c.get('step_id') == repaired['step_id']]
    assert len(matches) == 1
    assert matches[0] == repaired


def test_fragment_revision_ignores_unrelated_runtime_text_and_preserves_old_file(tmp_path):
    root = tmp_path / 'session'
    first = expand_parallel_report_nodes(_revision_report(root), root, target_plan_version=1,
        source_evidence={'simulation': {'result_ref': {'call_id': 'same'}, 'result': 'old text'}})
    from pathlib import Path
    old = Path(first[0]['node']['arguments']['output_path'])
    old.parent.mkdir(parents=True)
    old.write_text('immutable earlier report')
    second = expand_parallel_report_nodes(_revision_report(root), root, target_plan_version=1,
        source_evidence={'simulation': {'result_ref': {'call_id': 'same'}, 'result': 'new unrelated text'}})
    assert second == first
    newer = expand_parallel_report_nodes(_revision_report(root), root, target_plan_version=2)
    assert newer[0]['node']['arguments']['output_path'] != str(old)
    assert old.read_text() == 'immutable earlier report'


def test_fragment_revision_does_not_reuse_legacy_receipt_without_exact_source_call(tmp_path):
    root = tmp_path / 'session'
    initial = [c['node'] for c in expand_parallel_report_nodes(_revision_report(root), root)]
    runtime = {'simulation': {'status': 'succeeded', 'execution_fingerprint': 'a',
                              'result_ref': {'call_id': 'source'}},
        'final__fragment__simulation': {'status': 'succeeded', 'node_verification': {'verified': True},
            'result': {'source_step': 'simulation', 'source_attempt_fingerprint': 'a',
                       'report_file': initial[0]['arguments']['output_path']}}}
    changed = expand_parallel_report_nodes(_revision_report(root), root, initial,
        target_plan_version=2, source_evidence=runtime)
    assert changed[0]['node']['arguments']['output_path'] != initial[0]['arguments']['output_path']


def test_fragment_revision_rejects_receipt_from_different_runtime_contract(tmp_path):
    import copy
    root = tmp_path / 'session'
    existing = [c['node'] for c in expand_parallel_report_nodes(_revision_report(root), root)]
    old_contract = copy.deepcopy(existing[0])
    old_contract['arguments']['title'] = 'Different prior title'
    runtime = {'simulation': {'status': 'succeeded', 'execution_fingerprint': 'attempt',
                              'result_ref': {'call_id': 'source-call'}},
        'final__fragment__simulation': {'status': 'succeeded', 'contract': old_contract,
            'node_verification': {'verified': True},
            'result': {'source_step': 'simulation', 'source_attempt_fingerprint': 'attempt',
                       'source_evidence_call_id': 'source-call',
                       'report_file': existing[0]['arguments']['output_path']}}}
    changed = expand_parallel_report_nodes(_revision_report(root), root, existing,
        target_plan_version=2, source_evidence=runtime)
    assert changed[0]['node']['arguments']['output_path'] != existing[0]['arguments']['output_path']
