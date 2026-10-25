"""Synthetic process fixtures verify String manifest dispatch, not GPU physics."""
import csv
from pathlib import Path
import subprocess
import sys
import pytest

RUNNER = Path(__file__).resolve().parents[1] / 'tools/run_string_tst_batch.py'


@pytest.mark.parametrize('solver_exit', [0, 7])
def test_manifest_uses_each_input_file_and_propagates_solver_failure(tmp_path, solver_exit):
    inputs = tmp_path / 'MOF with spaces/dir1'
    inputs.mkdir(parents=True)
    native = inputs / 'input.dat'
    native.write_text('synthetic protocol input\n')
    manifest = tmp_path / 'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['structure','direction','cif','input_dat','natoms'])
        writer.writeheader()
        writer.writerow({'structure':'MOF with spaces','direction':1,'cif':'source.cif','input_dat':str(native),'natoms':1})
    solver = tmp_path / 'fake_solver'
    solver.write_text('#!' + sys.executable + '\nimport pathlib,sys\n'
        'assert pathlib.Path(sys.argv[1]).is_file()\n'
        'assert pathlib.Path.cwd() == pathlib.Path(sys.argv[1]).parent\n'
        'print("synthetic solver invoked", flush=True)\n'
        'path = pathlib.Path(sys.argv[2])\n'
        'path.write_text("0 0 0 0 0 0 0\\n1 0 0 0 0 0 1\\n")\n'
        f'raise SystemExit({solver_exit})\n')
    solver.chmod(0o700)

    result = subprocess.run([sys.executable,str(RUNNER),'--manifest',str(manifest),
                             '--executable',str(solver)],capture_output=True,text=True)

    if solver_exit == 0:
        assert result.returncode == 0, result.stderr
    else:
        assert result.returncode != 0
    assert 'synthetic solver invoked' in (inputs/'run.log').read_text()


def test_string_submission_receipt_exposes_workdir_and_registers_watch(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from agents import registry
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path,
        string_tst_config={'generator': '/installed/generate.py', 'executable': '/installed/solver', 'python': sys.executable}))
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kwargs: {'submitted': True, 'job_id': '42'})
    watches = []
    monkeypatch.setattr(registry, '_register_jobwatch', lambda receipt, **kwargs: watches.append(dict(receipt)))
    output = tmp_path / 'string'

    receipt = registry._exec_string_tst({'cif_dir': str(tmp_path / 'inputs'), 'output_dir': str(output), 'gas': 'CH4'})

    assert receipt['work_dir'] == str(output)
    assert watches[0]['work_dir'] == str(output)


def test_string_uses_existing_canonical_software_paths(tmp_path, monkeypatch):
    import json
    from bimem_agent import paths
    from agents.config import AgentConfig
    from agents import registry
    selected = tmp_path / 'runtime.json'
    selected.write_text(json.dumps({'software': {'string_tst_executable': '/installed/string-gpu',
                                                'string_tst_generate_script': '/installed/generate.py'},
                                   'paths': {'agent_env_python': sys.executable}}))
    monkeypatch.setattr(paths, 'RUNTIME_CONFIG_PATH', selected)
    paths.load_runtime_config.cache_clear()
    config = AgentConfig(project_root=tmp_path)
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *args, **kwargs: None)
    commands = []
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: commands.append(kw['command']) or {'submitted': False})

    registry._exec_string_tst({'cif_dir': str(tmp_path/'inputs'), 'output_dir': str(tmp_path/'output'), 'gas': 'CH4'})

    assert '/installed/string-gpu' in commands[0]
    assert '/installed/generate.py' in commands[0]
