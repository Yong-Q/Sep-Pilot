"""Synthetic executable fixtures exercise argv/configuration, never MD physics."""
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import os

import pytest

from agents.config import AgentConfig


def configure_runtime(tmp_path, monkeypatch, values):
    from bimem_agent import paths
    mapping = {'executable': 'lammps_executable', 'mpi_executable': 'mpi_executable',
               'interface_executable': 'lammps_interface_executable', 'library_path': 'lammps_library_path'}
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps({'software': {mapping[k]: v for k, v in values.items() if k in mapping},
                                   'paths': {'agent_env_python': values.get('python', sys.executable)}}))
    monkeypatch.setattr(paths, 'RUNTIME_CONFIG_PATH', runtime)
    monkeypatch.setattr(paths, 'PROJECT_ROOT', tmp_path)
    paths.load_runtime_config.cache_clear()
    return runtime


def test_lammps_configuration_reads_canonical_runtime(tmp_path, monkeypatch):
    values = {'executable': '/opt/user/lmp_mpi', 'mpi_executable': '/opt/user/mpiexec'}
    configure_runtime(tmp_path, monkeypatch, values)
    config = AgentConfig(project_root=tmp_path)
    assert config.lammps_config['executable'] == values['executable']
    assert config.lammps_config['mpi_executable'] == values['mpi_executable']
    assert config.lammps_config['python'] == sys.executable


def test_canonical_runtime_environment_override_is_used(tmp_path):
    runtime = tmp_path / 'selected-runtime.json'
    runtime.write_text(json.dumps({'software': {'lammps_executable': '/chosen/lmp',
                                               'mpi_executable': '/chosen/mpiexec'},
                                   'paths': {'agent_env_python': '/chosen/python'}}))
    environment = dict(os.environ, BIMEM_RUNTIME_CONFIG=str(runtime))
    process = subprocess.run([sys.executable, '-c',
        'import json; from agents.config import AgentConfig; print(json.dumps(AgentConfig().lammps_config))'],
        capture_output=True, text=True, check=True, env=environment)
    assert json.loads(process.stdout) == {'executable': '/chosen/lmp',
                                          'mpi_executable': '/chosen/mpiexec', 'python': '/chosen/python'}


def test_unconfigured_md_does_not_submit_or_use_developer_installation(tmp_path, monkeypatch):
    from agents import registry
    configure_runtime(tmp_path, monkeypatch, {})
    config = AgentConfig(project_root=tmp_path)
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: pytest.fail('must reject before submission'))
    result = registry._exec_md_optimize({'cif_path': 'input.cif'})
    assert result['submitted'] is False
    assert 'software.lammps_executable' in result['error']


def test_md_submission_uses_configured_runtime_and_registers_watch(tmp_path, monkeypatch):
    from agents import registry
    values = {'executable': '/user software/lmp', 'mpi_executable': '/user software/mpirun',
              'python': sys.executable, 'interface_executable': '/user software/interface'}
    configure_runtime(tmp_path, monkeypatch, values)
    config = AgentConfig(project_root=tmp_path)
    captured, watches = {}, []
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry, '_ws', lambda kind: str(tmp_path / kind))
    def submit(**kwargs):
        captured.update(kwargs)
        return {'submitted': True, 'job_id': '42'}
    monkeypatch.setattr(registry.slurm, 'submit_and_return', submit)
    monkeypatch.setattr(registry, '_register_jobwatch', lambda result, **kw: watches.append(result.copy()))
    result = registry._exec_md_optimize({'cif_path': 'input.cif', 'mode': 'md', 'temperature': 298,
                                        'pressure': 1, 'npt_ps': .1, 'timestep': .5})
    command = captured['command']
    assert 'md_runtime.py' in command
    assert '/user software/lmp' in command and '/user software/mpirun' in command
    assert 'conda activate Agent' not in command and 'module load' not in command
    assert result['work_dir'] == str(tmp_path / 'md')
    assert watches[0]['job_id'] == '42'


def test_existing_session_reads_saved_runtime_before_next_submission(tmp_path, monkeypatch):
    from agents import registry
    runtime = configure_runtime(tmp_path, monkeypatch, {})
    config = AgentConfig(project_root=tmp_path)
    _ = config.lammps_config  # Prime the existing runtime cache before saving.
    runtime.write_text(json.dumps({'software': {'lammps_executable': '/new/lmp', 'mpi_executable': '/new/mpi'}}))
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry, '_ws', lambda kind: str(tmp_path / kind))
    commands = []
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: commands.append(kw['command']) or {'submitted': False})
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    registry._exec_md_optimize({'cif_path': 'input.cif'})
    assert '/new/lmp' in commands[0] and '/new/mpi' in commands[0]


def test_md_retry_uses_fresh_directory_and_retains_failed_logs(tmp_path, monkeypatch):
    from agents import registry
    configure_runtime(tmp_path, monkeypatch, {'executable': '/configured/lmp'})
    monkeypatch.setattr(registry, 'get_config', lambda: AgentConfig(project_root=tmp_path))
    monkeypatch.setattr(registry, '_ws', lambda _: str(tmp_path / 'md'))
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: {'submitted': True, 'job_id': '43'})
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **kw: None)
    work = tmp_path / 'md';work.mkdir()
    (work / 'interface.log').write_text('original failure')
    (work / 'md_result.json').write_text('{"success":false}')
    result = registry._exec_md_optimize({'cif_path': 'input.cif'})
    assert Path(result['work_dir']).is_relative_to(work)
    assert Path(result['work_dir']) != work
    assert (work / 'interface.log').read_text() == 'original failure'
    assert (work / 'md_result.json').read_text() == '{"success":false}'


@pytest.mark.parametrize('invocation', ['python', 'cli'])
@pytest.mark.parametrize('n_procs', [1, 2, 4])
def test_configured_executables_receive_argv_without_shell_or_path_override(tmp_path, invocation, n_procs, monkeypatch):
    # This fixture checks argv at explicit process counts, independent of the
    # allocation used to run pytest; allocation capping has separate tests.
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    module_path = Path(__file__).resolve().parents[1] / 'tools/md_runtime.py'
    spec = importlib.util.spec_from_file_location('configured_md_test', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def executable(name, source):
        path = tmp_path / name
        path.write_text('#!' + sys.executable + '\n' + source)
        path.chmod(0o700)
        return str(path)
    interface = executable('custom interface', 'import pathlib,sys\np=pathlib.Path(sys.argv[-1])\n'
                           'pathlib.Path("data."+p.stem).write_text("synthetic native data")\n'
                           'pathlib.Path("in."+p.stem).write_text("units real\\nread_data data."+p.stem+"\\nlog log."+p.stem+"\\n")\n')
    solver = executable('custom lammps', 'import pathlib,sys,re,shlex\n'
                        'assert sys.argv[1]=="-in"\ns=pathlib.Path(sys.argv[2]).read_text()\n'
                        'assert "run 200\\n" in s and "temp 298 298" in s\n'
                        'out=shlex.split(re.findall(r"^write_data (.+)$",s,re.M)[-1])[0]\n'
                        'pathlib.Path(out).write_text("synthetic final data")\n'
                        'print("Temp Press Volume PotEng KinEng\\n298 1 100 -2 3")\n')
    mpi = executable('custom mpi', 'import subprocess,sys,os\n'
                     f'assert sys.argv[1:3]==["-np","{n_procs}"]\n'
                     'assert all(os.environ[k]=="1" for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"))\n'
                     'raise SystemExit(subprocess.run(sys.argv[3:]).returncode)\n')
    cif = tmp_path / 'input.cif'; cif.write_text('synthetic fixture')
    parameters = dict(cif_path=str(cif), output_dir=str(tmp_path / 'work'), mode='md', temperature=298,
                      pressure=1, npt_ps=.1, timestep=.5, n_procs=n_procs,
                      executable=solver, mpi_executable=mpi,
                      interface_executable=interface, skip_convert=True)
    if invocation == 'python':
        result = module.run(**parameters)
    else:
        process = subprocess.run([sys.executable, str(module_path), '--parameters', json.dumps(parameters)],
                                 capture_output=True, text=True)
        assert process.returncode == 0, process.stdout + process.stderr
        result = json.loads(process.stdout)
    assert result['success'] is True
    assert Path(result['output_cif']).read_text() == 'synthetic final data'
    assert result['energy'] == -2
