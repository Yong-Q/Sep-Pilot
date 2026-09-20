import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / 'tools' / 'pacmof_batch.py'
SPEC = importlib.util.spec_from_file_location('pacmof_batch_under_test', MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_best_effort_batch_records_failures_and_accepts_valid_successes(tmp_path):
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'charged'
    inputs.mkdir()
    for name in ('ok.cif', 'ok2.cif', 'bad.cif'):
        (inputs / name).write_text(name)

    def charge(path, destination):
        if path.name == 'bad.cif':
            raise ValueError('feature contains NaN')
        (destination / f'{path.stem}_pacmof.cif').write_text('charged')

    result = MODULE.run_batch(inputs, output, charge, 'best_effort')

    assert result['accepted'] is True
    assert result['attempted_count'] == 3
    assert result['successful_count'] == 2
    assert result['failed_count'] == 1
    assert result['artifact_presence_floor'] == 1
    assert result['failures'][0]['file'] == 'bad.cif'
    assert json.loads((output / 'charge_manifest.json').read_text()) == result


def test_strict_batch_rejects_partial_results(tmp_path):
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'charged'
    inputs.mkdir()
    (inputs / 'bad.cif').write_text('bad')

    result = MODULE.run_batch(
        inputs, output, lambda *_: (_ for _ in ()).throw(ValueError('NaN')), 'strict')

    assert result['accepted'] is False
    assert result['successful_count'] == 0


def test_best_effort_records_sparse_success_for_agent_review(tmp_path):
    inputs = tmp_path / 'inputs'
    output = tmp_path / 'charged'
    inputs.mkdir()
    for index in range(5):
        (inputs / f'{index}.cif').write_text('input')

    def charge(path, destination):
        if int(path.stem) >= 2:
            raise ValueError('unsupported structure')
        (destination / f'{path.stem}_pacmof.cif').write_text('charged')

    result = MODULE.run_batch(inputs, output, charge, 'best_effort')

    assert result['artifact_presence_floor'] == 1
    assert result['successful_count'] == 2
    assert result['accepted'] is True
