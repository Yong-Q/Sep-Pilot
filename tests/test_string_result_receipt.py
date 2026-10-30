"""Native protocol fixtures are synthetic processes, not GPU/physics validation."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from agents.workflow_patch import complete_compute_contract


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'tools/run_string_tst_batch.py'
    spec = importlib.util.spec_from_file_location('string_receipt_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def manifest_at(tmp_path, count=1, input_text='synthetic input\n'):
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['structure', 'direction', 'input_dat'])
        writer.writeheader()
        for index in range(count):
            source = tmp_path / f'frame_{index}/dir1/input.dat'
            source.parent.mkdir(parents=True)
            source.write_text(input_text)
            writer.writerow({'structure': f'frame_{index}', 'direction': 1, 'input_dat': str(source)})
    return manifest


def solver_at(tmp_path, body=None):
    solver = tmp_path / 'synthetic_solver'
    solver.write_text('#!' + sys.executable + '\nimport pathlib,sys\n' + (body or
        'pathlib.Path(sys.argv[2]).write_text("0 0 0 0 0 0 1\\n1 0 0 0 0 0 2\\n")\n'
        'print("info:\\t1\\t0\\t424\\t0.1")\nprint("info:\\t0\\t20\\t424\\t0.2")\n'))
    solver.chmod(0o700)
    return solver


def test_complete_receipt_binds_real_paths_hashes_points_and_last_native_flag(tmp_path, runner):
    manifest = manifest_at(tmp_path, count=2)
    receipt = runner.run_manifest(manifest, solver_at(tmp_path))
    saved = json.loads((tmp_path / 'string_results.json').read_text())
    assert saved == receipt and saved['success'] is True
    assert len(saved['results']) == 2
    assert saved['manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    for row in saved['results']:
        assert row['point_count'] == 2
        assert row['input_sha256'] == hashlib.sha256(Path(row['input_path']).read_bytes()).hexdigest()
        assert row['output_sha256'] == hashlib.sha256(Path(row['output_path']).read_bytes()).hexdigest()
        assert Path(row['run_log_path']).is_file()
        assert row['native_info'] == {'converged': False, 'iterations': 20, 'expanded_atoms': 424}
    assert 'scientific convergence' in saved['validation_scope']
    assert not list(tmp_path.glob('.string-results-*'))


def test_new_receipt_declares_expected_protocol_without_claiming_binary_source_binding(tmp_path, runner):
    manifest=manifest_at(tmp_path)
    solver=solver_at(tmp_path)
    result=runner.run_manifest(manifest,solver)
    protocol=result['output_protocol']
    assert protocol['column_count']==7 and len(protocol['columns'])==7
    assert protocol['columns']==['fractional_a','fractional_b','fractional_c','orientation_alpha',
        'orientation_beta','orientation_gamma','potential']
    assert 's' not in protocol['columns'] and protocol['energy_unit'] is None
    assert protocol['provenance']=={'declared_by_adapter':True,'source_verified':False,'binary_source_binding':False}
    assert result['executable_sha256']==hashlib.sha256(solver.read_bytes()).hexdigest()
    assert result['results'][0]['adapter_native_wall_elapsed_seconds']>=0


def test_missing_native_convergence_marker_is_unknown_not_success(tmp_path, runner):
    manifest = manifest_at(tmp_path)
    solver = solver_at(tmp_path, 'pathlib.Path(sys.argv[2]).write_text("0 0 0 0 0 0 1\\n1 0 0 0 0 0 2\\n")\n')
    receipt = runner.run_manifest(manifest, solver)
    assert receipt['results'][0]['native_info']['converged'] is None


@pytest.mark.parametrize('contents', ['0 0 0 0 0 0 nan\n1 0 0 0 0 0 2\n',
                                     '0 0 0 0 0 1\n1 0 0 0 0 2\n', '0 0 0 0 0 0 1\n'])
def test_invalid_native_output_never_publishes_success_receipt(tmp_path, runner, contents):
    manifest = manifest_at(tmp_path)
    solver = solver_at(tmp_path, f'pathlib.Path(sys.argv[2]).write_text({contents!r})\n')
    with pytest.raises(ValueError):
        runner.run_manifest(manifest, solver)
    assert not (tmp_path / 'string_results.json').exists()


def test_later_native_failure_preserves_evidence_without_partial_success_receipt(tmp_path, runner):
    manifest = manifest_at(tmp_path, count=2)
    solver = solver_at(tmp_path,
        'pathlib.Path(sys.argv[2]).write_text("0 0 0 0 0 0 1\\n1 0 0 0 0 0 2\\n")\n'
        'raise SystemExit(7 if "frame_1" in sys.argv[1] else 0)\n')
    with pytest.raises(Exception):
        runner.run_manifest(manifest, solver)
    assert not (tmp_path / 'string_results.json').exists()
    assert (tmp_path / 'frame_0/dir1/string_path.dat').is_file()
    assert (tmp_path / 'frame_1/dir1/run.log').is_file()


def test_existing_receipt_is_not_overwritten_or_native_solver_rerun(tmp_path, runner, monkeypatch):
    manifest = manifest_at(tmp_path)
    result = tmp_path / 'string_results.json'
    result.write_text('original results')
    monkeypatch.setattr(runner.subprocess, 'run', lambda *a, **kw: pytest.fail('existing results must reject before execution'))
    with pytest.raises(FileExistsError):
        runner.run_manifest(manifest, tmp_path / 'unused')
    assert result.read_text() == 'original results'


def test_receipt_publish_race_never_overwrites_and_cleans_own_temporary_file(tmp_path, runner):
    manifest = manifest_at(tmp_path)
    target = tmp_path / 'string_results.json'
    solver = solver_at(tmp_path,
        'pathlib.Path(sys.argv[2]).write_text("0 0 0 0 0 0 1\\n1 0 0 0 0 0 2\\n")\n'
        f'pathlib.Path({str(target)!r}).write_text("concurrent evidence")\n')
    with pytest.raises(FileExistsError):
        runner.run_manifest(manifest, solver)
    assert target.read_text() == 'concurrent evidence'
    assert not list(tmp_path.glob('.string-results-*'))


def test_manifest_input_point_count_must_match_native_path_when_declared(tmp_path, runner):
    manifest = manifest_at(tmp_path, input_text='#_of_points delta_frac delta_angle_degree\n3 0.001 1\n')
    with pytest.raises(ValueError, match='point'):
        runner.run_manifest(manifest, solver_at(tmp_path))
    assert not (tmp_path / 'string_results.json').exists()


def test_string_compiler_uses_relative_canonical_receipts_and_preserves_science(tmp_path):
    arguments = {'cif_path': str(tmp_path / 'input.cif'), 'output_dir': str(tmp_path / 'runs/u/c/string'),
                 'gas': 'CH4', 'temperature': 310, 'directions': [2], 'running_steps': 17, 'n_string': 9}
    node = {'tool': 'run_string_tst', 'arguments': arguments, 'expected_outputs': [
        {'kind': 'directory', 'path': arguments['output_dir'], 'pattern': 'string_path.dat'}]}
    compiled = complete_compute_contract(node, 'string', tmp_path, tmp_path / 'runs/u/c')
    assert compiled['arguments'] == arguments
    assert compiled['expected_outputs'] == [
        {'kind': 'file', 'path': 'string_input_CH4/manifest.csv'},
        {'kind': 'file', 'path': 'string_input_CH4/string_results.json'}]
    assert node['expected_outputs'][0]['kind'] == 'directory'


def test_string_compiler_supplies_only_missing_operational_directory(tmp_path):
    node = {'tool': 'run_string_tst', 'arguments': {'gas': 'CH4', 'cif_path': str(tmp_path / 'input.cif')}}
    result = complete_compute_contract(node, 'string', tmp_path, tmp_path / 'runs/u/c')
    assert Path(result['arguments']['output_dir']).is_relative_to(tmp_path / 'runs/u/c')
    assert 'directions' not in result['arguments'] and 'temperature' not in result['arguments']
    missing_gas = complete_compute_contract({'tool': 'run_string_tst', 'arguments': {}}, 'x', tmp_path, tmp_path / 'runs/u/c')
    assert 'gas' not in missing_gas['arguments']


def test_string_submission_predicts_canonical_result_without_claiming_it_exists(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agents import registry
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path,
        string_tst_config={'generator': '/installed/generate', 'executable': '/installed/solver', 'python': sys.executable}))
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: {'submitted': True, 'job_id': '42'})
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    receipt = registry._exec_string_tst({'cif_dir': str(tmp_path / 'inputs'), 'output_dir': str(tmp_path / 'output'), 'gas': 'CH4'})
    assert receipt['result_json_path'] == str(tmp_path / 'output/string_input_CH4/string_results.json')
    assert not Path(receipt['result_json_path']).exists()
