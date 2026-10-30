"""Captured String scheduler commands; no GPU or physical solver execution."""
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from agents import registry


@pytest.fixture
def submission(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(
        project_root=tmp_path, string_tst_config={'generator': '/installed/generate.py',
                                               'executable': '/installed/gpu-string', 'python': '/installed/python'}))
    monkeypatch.setattr(registry, '_ws', lambda _: str(tmp_path / 'tst'))
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    monkeypatch.setattr(registry.slurm, 'submit_and_return',
                        lambda **kw: captured.append(kw) or {'submitted': True, 'job_id': '42'})
    return captured


def generator_arguments(command):
    line = next(line for line in command.splitlines() if '--cif-dir' in line)
    return shlex.split(line)


def flag(arguments, name):
    return arguments[arguments.index(name) + 1]


def test_single_cif_and_explicit_budgets_reach_generator_without_changing_gpu_resources(tmp_path, submission):
    source = tmp_path / 'shared inputs/framework_clean.cif'
    source.parent.mkdir()
    source.write_text('synthetic protocol fixture')
    output = tmp_path / 'runs/alice/one/string'
    args = {'cif_path': str(source), 'output_dir': str(output), 'gas': 'CH4', 'temperature': 310,
            'directions': [2], 'running_steps': 17, 'n_string': 9, 'nmax': 8, 'walltime': '00:07:00'}
    assert registry.get_registry().validate_params('run_string_tst', args) == []
    receipt = registry._exec_string_tst(args)
    tokens = generator_arguments(submission[0]['command'])
    assert flag(tokens, '--cif') == str(source)
    assert flag(tokens, '--cif-dir') == str(source.parent)
    assert flag(tokens, '--running-steps') == '17' and flag(tokens, '--n-string') == '9'
    assert flag(tokens, '--nmax') == '8'
    assert tokens[tokens.index('--directions') + 1:tokens.index('--directions') + 2] == ['2']
    assert submission[0]['walltime'] == '00:07:00'
    assert submission[0]['partition'] == 'gpu' and submission[0]['gres'] == 'gpu:1'
    assert submission[0]['cpus_per_task'] == 8
    assert receipt['manifest_path'] == str(output / 'string_input_CH4/manifest.csv')
    assert 'converg' in registry.get_registry().get('run_string_tst').description.lower()


def test_original_string_defaults_remain_explicit_and_unchanged(tmp_path, submission):
    registry._exec_string_tst({'cif_dir': str(tmp_path / 'inputs'), 'output_dir': str(tmp_path / 'out'), 'gas': 'CH4'})
    tokens = generator_arguments(submission[0]['command'])
    assert flag(tokens, '--running-steps') == '10000'
    assert flag(tokens, '--n-string') == '401'
    assert flag(tokens, '--nmax') == '70'
    start = tokens.index('--directions') + 1
    assert tokens[start:start + 3] == ['1', '2', '3']
    assert '--cif' not in tokens
    assert submission[0]['walltime'] == '48:00:00'


def test_relative_single_cif_and_outputs_use_project_anchor_and_ignore_batch_directory(tmp_path, submission):
    source = tmp_path / 'inputs/framework.cif'
    source.parent.mkdir()
    source.write_text('synthetic fixture')
    receipt = registry._exec_string_tst({
        'cif_path': 'inputs/framework.cif', 'cif_dir': 'entire-library',
        'output_dir': 'runs/alice/one/string', 'gas': 'CH4',
        'directions': [3, 1], 'walltime': '1-00:10:00'})
    tokens = generator_arguments(submission[0]['command'])
    assert flag(tokens, '--cif') == str(source)
    assert flag(tokens, '--cif-dir') == str(source.parent)
    assert receipt['work_dir'] == str(tmp_path / 'runs/alice/one/string')
    start = tokens.index('--directions') + 1
    assert tokens[start:start + 2] == ['3', '1']
    assert submission[0]['walltime'] == '1-00:10:00'


@pytest.mark.parametrize('extra', [
    {'directions': []}, {'directions': [0]}, {'directions': [4]}, {'directions': [1, 1]},
    {'directions': [True]}, {'running_steps': 0}, {'running_steps': -1}, {'running_steps': 1.5},
    {'n_string': 1}, {'n_string': False}, {'nmax': 0}, {'walltime': 'five minutes'},
    {'walltime': '00:61:00'}, {'gas': 'CH4$(touch /unwanted)'},
])
def test_invalid_budgets_are_rejected_before_scheduler_entry(tmp_path, submission, extra):
    args = {'cif_dir': str(tmp_path / 'inputs'), 'output_dir': str(tmp_path / 'out'), 'gas': 'CH4', **extra}
    assert registry.get_registry().validate_params('run_string_tst', args)
    receipt = registry._exec_string_tst(args)
    assert receipt.get('blocked') and receipt.get('executed') is False
    assert submission == []


def test_existing_string_results_are_preserved_in_fresh_attempt(tmp_path, submission):
    output = tmp_path / 'out'
    old = output / 'string_input_CH4/manifest.csv'
    old.parent.mkdir(parents=True)
    old.write_text('original manifest evidence')
    receipt = registry._exec_string_tst({'cif_dir': str(tmp_path / 'inputs'), 'output_dir': str(output), 'gas': 'CH4'})
    actual = Path(receipt['work_dir'])
    assert actual.parent == output and actual.name.startswith('attempt_')
    assert old.read_text() == 'original manifest evidence'
    assert receipt['manifest_path'] == str(actual / 'string_input_CH4/manifest.csv')


def test_nonexistent_selected_cif_does_not_reserve_gpu(tmp_path, submission):
    receipt = registry._exec_string_tst({'cif_path': str(tmp_path / 'missing.cif'),
        'output_dir': str(tmp_path / 'out'), 'gas': 'CH4'})
    assert receipt['blocked'] and receipt['executed'] is False
    assert not submission


def test_single_cif_scope_guard_rejects_another_conversation(tmp_path, submission):
    from agents.watch_context import clear_context, set_context
    set_context('alice', 'one', 'adsorption', '')
    try:
        receipt = registry.get_registry().execute_dict('run_string_tst', {
            'cif_path': str(tmp_path / 'runs/bob/two/private.cif'),
            'output_dir': str(tmp_path / 'runs/alice/one/string'), 'gas': 'CH4'})
        assert receipt.get('blocked') or receipt.get('error')
        assert not submission
    finally:
        clear_context()
