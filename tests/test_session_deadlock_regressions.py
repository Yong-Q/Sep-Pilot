import copy
from pathlib import Path

import pytest

from agents.registry import get_registry
from agents.workspace import canonical_session_shell, scoped_shell_contract, tool_scope_issues
from agents.workflow_compiler import (
    compile_dataflow_bindings, compile_workflow_changes, materialize_safe_defaults,
)
from agents.workflow_patch import (
    WorkflowContractError, patched_graph, complete_compute_contract,
    resolve_workflow_placeholders,
)


def test_main_can_inspect_directories_without_contradictory_routing_instruction():
    from agents.defns import ORCHESTRATOR
    assert {'inspect_path', 'inspect_run'} <= set(ORCHESTRATOR.function_map())
    assert '你没有 submit_job/check_job/diagnose_job/read_file' not in ORCHESTRATOR.instructions


def test_input_discovery_does_not_consume_job_polling_budget():
    from agents.session import is_job_status_polling
    assert not is_job_status_polling('inspect_path', {'path':'charged'})
    for command in ('ls charged', 'cat input.dat', 'head results.csv', 'tail run.log', 'grep -i energy input.dat', 'ls "squeue results"'):
        assert not is_job_status_polling('run_bash', {'command': command})
    for command in ('qstat 123', 'squeue', 'sacct -j 123', 'scontrol show job 123'):
        assert is_job_status_polling('run_bash', {'command': command})
    assert is_job_status_polling('check_job', {'job_id':'123'})


def test_absolute_own_session_is_allowed_but_sibling_and_prefix_alias_are_not(tmp_path):
    own = tmp_path / 'runs/alice/c1'
    assert not tool_scope_issues({'path': str(own / 'charged')}, 'alice', 'c1', tmp_path)
    for path in ('runs/alice/c2/charged', 'runs/alice/c10/charged', 'runs/bob/c1/charged'):
        assert tool_scope_issues({'path': str(tmp_path / path)}, 'alice', 'c1', tmp_path)


def test_shell_project_operand_has_same_anchor_as_native_tool(tmp_path):
    command = 'ls runs/alice/c1/charged/ | head -20'
    actual = canonical_session_shell(command, tmp_path)
    assert str(tmp_path / 'runs/alice/c1/charged') in actual
    cwd, issues = scoped_shell_contract(command, '', 'alice', 'c1', tmp_path)
    assert not issues and cwd == str(tmp_path / 'runs/alice/c1')
    assert scoped_shell_contract('ls runs/bob/c2/charged', '', 'alice', 'c1', tmp_path)[1]


def test_call_history_is_not_a_worker_contract_even_when_failed(tmp_path):
    history = [
        {'step_id': 'shell', 'tool': 'run_bash', 'arguments': {'command': 'ls charged'}, 'status': 'failed'},
        {'step_id': 'read', 'tool': 'read_file', 'arguments': {'path': 'charged'}, 'status': 'failed'},
        {'step_id': 'run_cdft', 'tool': 'run_cdft', 'status': 'blocked', 'validation': {'dispatch_not_entered': True}},
        {'step_id': 'execute_workflow', 'tool': 'execute_workflow', 'status': 'failed'},
    ]
    node = {'agent': 'analyst', 'tool': 'run_cdft', 'arguments': {'action': 'pipeline', 'gas': 'Kr'},
            'depends_on': [], 'expected_outputs': ['results.csv']}
    graph, _ = patched_graph(history, [{'operation': 'upsert', 'step_id': 'kr', 'node': node}], tmp_path)
    assert [n['step_id'] for n in graph] == ['kr']
    history[2]['job_ids'] = ['known-job']
    graph, _ = patched_graph(history, [{'operation': 'upsert', 'step_id': 'kr', 'node': node}], tmp_path)
    assert 'run_cdft' in [n['step_id'] for n in graph]


def test_cdft_defaults_are_disjoint_and_verified_before_approval(tmp_path):
    node = {'tool': 'run_cdft', 'arguments': {'action': 'pipeline', 'gas': 'Kr'}, 'expected_outputs': []}
    root = tmp_path / 'runs/alice/c1'
    a = complete_compute_contract(node, 'kr', tmp_path, root)
    b = complete_compute_contract(node, 'xe', tmp_path, root)
    assert a['arguments']['job_work_dir'] != b['arguments']['job_work_dir']
    assert a['expected_outputs'][0]['path'] == str(Path(a['arguments']['job_work_dir']) / 'results.csv')
    node['expected_outputs'] = [{'kind': 'directory', 'path': 'wrong', 'pattern': 'output*.dat', 'min_count': 10}]
    repaired = complete_compute_contract(node, 'kr', tmp_path, root)
    assert repaired['expected_outputs'][-1] == {
        'kind': 'file',
        'path': str(Path(repaired['arguments']['job_work_dir']) / 'results.csv'),
    }


def test_workflow_compiler_resolves_session_and_upstream_path_placeholders(tmp_path):
    root = tmp_path / 'runs' / 'alice' / 'c1'
    changes = [
        {'operation': 'upsert', 'step_id': 'generate', 'node': {
            'arguments': {'output_dir': '{session}/structures'},
            'expected_outputs': [{'kind': 'directory', 'path': '{session}/structures',
                                  'pattern': '*.cif', 'min_count': 1}]},
        },
        {'operation': 'upsert', 'step_id': 'charge', 'node': {
            'arguments': {'cif_dir': '{generate.output_dir}', 'output_dir': '{session}/charged'},
            'expected_outputs': ['{session}/charged/result.csv']},
        },
    ]
    repaired = resolve_workflow_placeholders(changes, root)
    assert repaired[0]['node']['arguments']['output_dir'] == str(root / 'structures')
    assert repaired[1]['node']['arguments']['cif_dir'] == str(root / 'structures')
    assert repaired[1]['node']['expected_outputs'] == [str(root / 'charged/result.csv')]


def test_report_contract_derives_output_and_direct_evidence_dependencies(tmp_path):
    root = tmp_path / 'runs' / 'alice' / 'c1'
    report = root / 'report' / 'final.md'
    node = {
        'tool': 'generate_scientific_report',
        'arguments': {'title': 'Final', 'source_steps': ['old', 'analysis']},
        'depends_on': ['analysis'],
        'expected_outputs': [str(report)],
    }
    repaired = complete_compute_contract(node, 'report', tmp_path, root)
    assert repaired['arguments']['output_path'] == str(report)
    assert repaired['arguments']['source_steps'] == ['analysis']


def test_compiler_materializes_only_allowlisted_schema_defaults(tmp_path):
    registry = get_registry()
    node = {'tool': 'build_mof_database', 'arguments': {'cif_dir': str(tmp_path)}}
    compiled, receipt = materialize_safe_defaults(node, registry, 'inventory')
    assert compiled['arguments']['recursive'] is False
    assert compiled['arguments']['max_files'] == 1000
    assert {item['parameter'] for item in receipt} == {'recursive', 'max_files'}


def test_compiler_never_materializes_scientific_schema_defaults():
    registry = get_registry()
    node = {'tool': 'run_henry_chain', 'arguments': {'material': 'MOF-5'}}
    compiled, receipt = materialize_safe_defaults(node, registry, 'henry')
    assert 'gas' not in compiled['arguments']
    assert 'temperature' not in compiled['arguments']
    assert receipt == []


def test_compiler_does_not_default_scientific_analysis_window_or_dimensions():
    node = {'tool': 'analyze_diffusion_msd', 'arguments': {
        'trajectory_path': 'trajectory.csv', 'timestep_ps': 0.001,
    }}
    compiled, receipt = materialize_safe_defaults(
        node, get_registry(), 'diffusion')
    assert compiled['arguments']['format'] == 'auto'
    assert 'dimensions' not in compiled['arguments']
    assert 'fit_start_fraction' not in compiled['arguments']
    assert 'fit_end_fraction' not in compiled['arguments']
    assert [item['parameter'] for item in receipt] == ['format']


def _serial_directory_workflow(root):
    return [
        {'operation': 'upsert', 'step_id': 'generate', 'node': {
            'agent': 'harness', 'tool': 'generate_structure',
            'arguments': {'material_type': 'MOF', 'n_structures': 1,
                          'output_dir': str(root / 'structures')},
            'depends_on': [],
            'expected_outputs': [{'kind': 'directory', 'path': str(root / 'structures'),
                                  'pattern': '*.cif', 'min_count': 1}],
        }},
        {'operation': 'upsert', 'step_id': 'consume', 'node': {
            'agent': 'analyst', 'tool': 'build_mof_database',
            'arguments': {}, 'depends_on': ['generate'], 'expected_outputs': [],
        }},
    ]


def _ambiguous_directory_workflow(root):
    changes = _serial_directory_workflow(root)
    second = copy.deepcopy(changes[0])
    second['step_id'] = 'generate_b'
    second['node']['arguments']['output_dir'] = str(root / 'structures_b')
    second['node']['expected_outputs'][0]['path'] = str(root / 'structures_b')
    changes.insert(1, second)
    changes[-1]['node']['depends_on'] = ['generate', 'generate_b']
    return changes


def test_compiler_binds_serial_output_to_tool_argument(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes[1]['node']['input_bindings'] = {
        'cif_dir': {'step_id': 'generate', 'output_index': 0},
    }
    compiled = compile_dataflow_bindings(changes, get_registry(), tmp_path, root)
    assert compiled[1]['node']['arguments']['cif_dir'] == str(root / 'structures')


def test_compiler_auto_binds_one_compatible_upstream_directory(tmp_path):
    root = tmp_path / 'runs/u/c'
    compiled = compile_dataflow_bindings(
        _serial_directory_workflow(root), get_registry(), tmp_path, root)
    assert compiled[1]['node']['arguments']['cif_dir'] == str(root / 'structures')


@pytest.mark.parametrize('binding', [
    {'step_id': 'missing', 'output_index': 0},
    {'step_id': 'generate', 'output_index': -1},
    {'step_id': 'generate', 'output_index': True},
    {'step_id': 'generate', 'output_index': 99},
])
def test_compiler_rejects_invalid_explicit_binding(tmp_path, binding):
    root = tmp_path / 'runs/u/c'
    changes = _ambiguous_directory_workflow(root)
    changes[-1]['node']['input_bindings'] = {'cif_dir': binding}
    with pytest.raises(WorkflowContractError) as error:
        compile_dataflow_bindings(changes, get_registry(), tmp_path, root)
    assert error.value.details['error_kind'] == 'workflow_contract'
    assert error.value.details['step_id'] == 'consume'


def test_compiler_rejects_ambiguous_automatic_binding(tmp_path):
    root = tmp_path / 'runs/u/c'
    with pytest.raises(WorkflowContractError) as error:
        compile_dataflow_bindings(
            _ambiguous_directory_workflow(root), get_registry(), tmp_path, root)
    assert error.value.details['missing_parameters'] == ['cif_dir']
    assert len(error.value.details['binding_candidates']) == 2


def test_compiler_accepts_indirect_ancestor_and_ignores_unrelated_output(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes.insert(1, {'operation': 'upsert', 'step_id': 'middle', 'node': {
        'agent': 'analyst', 'tool': 'inspect_path',
        'arguments': {'path': str(root)}, 'depends_on': ['generate'],
        'expected_outputs': [],
    }})
    changes[-1]['node']['depends_on'] = ['middle']
    changes.insert(0, {'operation': 'upsert', 'step_id': 'unrelated', 'node': {
        'agent': 'harness', 'tool': 'generate_structure',
        'arguments': {'output_dir': str(root / 'unrelated')}, 'depends_on': [],
        'expected_outputs': [{'kind': 'directory', 'path': str(root / 'unrelated'),
                              'pattern': '*.cif', 'min_count': 1}],
    }})
    compiled = compile_dataflow_bindings(changes, get_registry(), tmp_path, root)
    assert compiled[-1]['node']['arguments']['cif_dir'] == str(root / 'structures')


def test_compiler_binds_file_glob_parent_to_directory_argument(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes[0]['node']['expected_outputs'] = [str(root / 'structures' / '*.cif')]
    changes[1]['node']['input_bindings'] = {
        'cif_dir': {'step_id': 'generate', 'output_index': 0},
    }
    compiled = compile_dataflow_bindings(changes, get_registry(), tmp_path, root)
    assert compiled[1]['node']['arguments']['cif_dir'] == str(root / 'structures')


def test_compiler_auto_binds_validation_work_dir(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes[1]['node'].update(
        tool='validate_gcmc_results', arguments={}, input_bindings={})
    compiled = compile_dataflow_bindings(
        changes, get_registry(), tmp_path, root)
    assert compiled[1]['node']['arguments']['work_dir'] == str(root / 'structures')


def test_compiler_can_bind_from_unchanged_ancestor_during_local_patch(tmp_path):
    root = tmp_path / 'runs/u/c'
    full = _serial_directory_workflow(root)
    producer = {'step_id': 'generate', **copy.deepcopy(full[0]['node'])}
    compiled = compile_dataflow_bindings(
        [full[1]], get_registry(), tmp_path, root, existing_steps=[producer])
    assert compiled[0]['node']['arguments']['cif_dir'] == str(root / 'structures')


def test_compiler_rejects_directory_output_for_file_only_argument(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes[1]['node'].update(tool='read_file', arguments={}, input_bindings={
        'path': {'step_id': 'generate', 'output_index': 0},
    })
    with pytest.raises(WorkflowContractError, match='incompatible or missing output'):
        compile_dataflow_bindings(changes, get_registry(), tmp_path, root)


def test_compiler_never_wires_upstream_artifact_into_output_destination(tmp_path):
    root = tmp_path / 'runs/u/c'
    changes = _serial_directory_workflow(root)
    changes[0]['node']['expected_outputs'] = [str(root / 'structures' / 'source.txt')]
    changes[1]['node'].update(tool='write_file', arguments={'content': 'new'},
                              input_bindings={})
    with pytest.raises(WorkflowContractError) as error:
        compile_dataflow_bindings(changes, get_registry(), tmp_path, root)
    assert error.value.details['missing_parameters'] == ['path']
    assert error.value.details['binding_candidates'] == []


def test_workflow_compiler_returns_concrete_arguments_and_default_receipt(tmp_path):
    root = tmp_path / 'runs/u/c'
    compilation = compile_workflow_changes(
        _serial_directory_workflow(root), get_registry(), tmp_path, root)
    consumer = compilation.changes[1]['node']
    assert consumer['arguments'] == {
        'cif_dir': str(root / 'structures'), 'recursive': False, 'max_files': 1000,
    }
    assert {item['parameter'] for item in compilation.defaults_applied} == {
        'recursive', 'max_files',
    }


def test_workflow_compiler_reports_unknown_tool_as_contract_error(tmp_path):
    changes = [{'operation': 'upsert', 'step_id': 'unknown', 'node': {
        'agent': 'analyst', 'tool': 'missing_tool', 'arguments': {},
        'depends_on': [], 'expected_outputs': [],
    }}]
    with pytest.raises(WorkflowContractError) as error:
        compile_workflow_changes(changes, get_registry(), tmp_path, tmp_path / 'runs/u/c')
    assert error.value.details['tool'] == 'missing_tool'
    assert error.value.details['step_id'] == 'unknown'


def test_local_patch_placeholder_can_reference_unchanged_ancestor(tmp_path):
    root = tmp_path / 'runs/u/c'
    producer = {'step_id': 'generate', 'agent': 'harness',
        'tool': 'generate_structure',
        'arguments': {'material_type': 'MOF', 'n_structures': 1,
                      'output_dir': str(root / 'structures')},
        'depends_on': [],
        'expected_outputs': [{'kind': 'directory', 'path': str(root / 'structures'),
                              'pattern': '*.cif', 'min_count': 1}]}
    changes = [{'operation': 'upsert', 'step_id': 'consume', 'node': {
        'agent': 'analyst', 'tool': 'build_mof_database',
        'arguments': {'cif_dir': '{generate.output_dir}'},
        'depends_on': ['generate'], 'expected_outputs': [],
    }}]
    compilation = compile_workflow_changes(
        changes, get_registry(), tmp_path, root, existing_steps=[producer])
    assert compilation.changes[0]['node']['arguments']['cif_dir'] == str(root / 'structures')


def test_backend_release_hash_covers_workflow_compiler(tmp_path):
    import importlib.util
    module_path = Path(__file__).resolve().parents[1] / 'scripts/restart_idle_backend.py'
    spec = importlib.util.spec_from_file_location('restart_idle_backend_under_test', module_path)
    restart_idle_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restart_idle_backend)
    for relative in restart_idle_backend.RELEASE_SOURCE_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding='utf-8')
    hashes = restart_idle_backend.release_source_hashes(tmp_path)
    assert 'agents/workflow_compiler.py' in hashes
    assert len(hashes['agents/workflow_compiler.py']) == 64


def test_backend_restart_reuses_actual_running_python_not_stale_conda_prefix(tmp_path):
    import importlib.util
    import sys
    module_path = Path(__file__).resolve().parents[1] / 'scripts/restart_idle_backend.py'
    spec = importlib.util.spec_from_file_location('restart_idle_backend_python_test', module_path)
    restart_idle_backend = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(restart_idle_backend)
    proc = tmp_path / 'proc'
    proc.mkdir()
    (proc / 'exe').symlink_to(Path(sys.executable))

    selected = restart_idle_backend.running_python(
        proc, {'CONDA_PREFIX': str(tmp_path / 'wrong-env')}, tmp_path / 'fallback-python')

    assert selected == Path(sys.executable).resolve()
