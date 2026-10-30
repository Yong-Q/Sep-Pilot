"""MD adapter contracts with fake scheduler receipts, never simulated physics."""
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from agents import registry


@pytest.fixture
def capture_submission(tmp_path, monkeypatch):
    captured = []
    config = SimpleNamespace(project_root=tmp_path, lammps_config={
        'executable': '/configured/lammps', 'python': '/configured/python'})
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry, '_ws', lambda _: str(tmp_path / 'md'))
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    monkeypatch.setattr(registry.slurm, 'submit_and_return',
                        lambda **kw: captured.append(kw) or {'submitted': True, 'job_id': '42'})
    return captured


def parameters(submission):
    tokens = shlex.split(submission['command'])
    return json.loads(tokens[tokens.index('--parameters') + 1])


@pytest.mark.parametrize('mode,suffix', [('single', '_opt.cif'), ('md', '_md.cif')])
def test_explicit_md_directory_has_predictable_expected_paths(tmp_path, monkeypatch, capture_submission, mode, suffix):
    relative = Path('runs/alice/one/quick-relaxation')
    expected = tmp_path / relative
    monkeypatch.setattr(registry, '_ws', lambda _: pytest.fail('explicit output must not create a default workspace'))
    args = {'cif_path': '/shared/framework_clean.cif', 'mode': mode, 'output_dir': str(relative)}
    if mode == 'md':
        args['temperature'] = 298
    receipt = registry._exec_md_optimize(args)
    assert capture_submission[0]['work_dir'] == str(expected)
    assert parameters(capture_submission[0])['output_dir'] == str(expected)
    assert receipt['expected_structure_path'] == str(expected / ('framework_clean' + suffix))
    assert receipt['result_json_path'] == str(expected / 'md_result.json')
    assert not Path(receipt['expected_structure_path']).exists()
    assert 'output_cif' not in receipt and 'success' not in receipt and 'energy' not in receipt
    assert receipt['guest_diffusion_supported'] is False and receipt['trajectory_path'] is None


def test_nonempty_explicit_md_directory_keeps_original_outputs(tmp_path, capture_submission):
    output = tmp_path / 'runs/alice/one/relaxation'
    output.mkdir(parents=True)
    original = output / 'md_result.json'
    original.write_text('{"success":false,"error":"original failure"}')
    receipt = registry._exec_md_optimize({'cif_path': '/shared/framework.cif', 'output_dir': str(output)})
    actual = Path(receipt['work_dir'])
    assert actual.parent == output and actual.name.startswith('attempt_')
    assert receipt['expected_structure_path'] == str(actual / 'framework_opt.cif')
    assert receipt['result_json_path'] == str(actual / 'md_result.json')
    assert parameters(capture_submission[0])['output_dir'] == str(actual)
    assert original.read_text() == '{"success":false,"error":"original failure"}'


def test_default_md_directory_retains_same_output_contract(tmp_path, capture_submission):
    receipt = registry._exec_md_optimize({'cif_path': '/shared/framework.cif'})
    assert receipt['expected_structure_path'] == str(tmp_path / 'md/framework_opt.cif')
    assert receipt['result_json_path'] == str(tmp_path / 'md/md_result.json')


def test_md_schema_exposes_outputs_and_rejects_cross_session_directory(tmp_path, capture_submission):
    from agents.watch_context import set_context, clear_context
    tool = registry.get_registry().get('run_md_optimize')
    assert 'output_dir' in tool.input_schema['properties']
    assert all(value in tool.description for value in ('_opt.cif', '_md.cif', 'md_result.json', 'energy', 'output_cif'))
    set_context('alice', 'one', 'adsorption', '')
    try:
        result = registry.get_registry().execute_dict('run_md_optimize', {
            'cif_path': str(tmp_path / 'cifs/framework.cif'),
            'output_dir': str(tmp_path / 'runs/bob/two/md')})
        assert result.get('blocked') or result.get('error')
        assert not capture_submission
    finally:
        clear_context()
