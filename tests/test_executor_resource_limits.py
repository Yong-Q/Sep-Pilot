"""Resource envelopes with fake adapters/scheduler, never scientific jobs."""
import importlib.util
import json
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from agents import registry
from agents.watch_context import set_context


@pytest.mark.parametrize('tool,default', [('md', 4), ('xtb', 1), ('string', 8)])
@pytest.mark.parametrize('cpus', [None, 1, 2])
def test_executor_process_or_thread_budget_matches_submission(tmp_path, monkeypatch, tool, default, cpus):
    captured = []
    config = SimpleNamespace(project_root=tmp_path,
        lammps_config={'executable': '/fake/lmp', 'python': '/fake/python'},
        string_tst_config={'generator': '/fake/generate.py', 'executable': '/fake/string', 'python': '/fake/python'})
    monkeypatch.setattr(registry, 'get_config', lambda: config)
    monkeypatch.setattr(registry, '_installation_path', lambda *a: '/fake/xtb')
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *a, **k: None)
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: captured.append(kw) or {'submitted': True})
    allocation = {'cpus': cpus, 'resource_review_id': 'review'} if cpus is not None else None
    set_context(resource_allocation=allocation)
    args = {'output_dir': str(tmp_path / 'output')}
    if tool == 'md': registry._exec_md_optimize({**args, 'cif_path': '/fake/input.cif'})
    elif tool == 'xtb': registry._exec_xtb({**args, 'input_path': '/fake/input.xyz'})
    else: registry._exec_string_tst({**args, 'cif_dir': str(tmp_path / 'cifs'), 'gas': 'CO2'})
    request = captured[0]
    expected = default if cpus is None else cpus
    assert request['cpus_per_task'] == expected
    if tool == 'md':
        words = shlex.split(request['command'])
        assert json.loads(words[words.index('--parameters') + 1])['n_procs'] == expected
        assert 'OMP_NUM_THREADS=1' in request['command']
    else:
        assert f'OMP_NUM_THREADS={expected}' in request['command']
    assert 'OPENBLAS_NUM_THREADS=1' in request['command']
    assert 'MKL_NUM_THREADS=1' in request['command']
    script = registry.slurm.render_sbatch_script(request['job_name'], request['command'], request['work_dir'],
                                                cpus_per_task=request['cpus_per_task'])
    assert f'#SBATCH --cpus-per-task={expected}' in script


@pytest.mark.parametrize('cpus,expected', [('1', 1), ('2', 2), (None, 4), ('0', None), ('invalid', None)])
def test_md_runtime_caps_stale_script_processes_inside_slurm(tmp_path, monkeypatch, cpus, expected):
    path = Path(__file__).resolve().parents[1] / 'tools/md_runtime.py'
    spec = importlib.util.spec_from_file_location('resource_limited_md', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    if cpus is None: monkeypatch.delenv('SLURM_CPUS_PER_TASK', raising=False)
    else: monkeypatch.setenv('SLURM_CPUS_PER_TASK', cpus)
    monkeypatch.setenv('OMP_NUM_THREADS', '128')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '128')
    monkeypatch.setenv('MKL_NUM_THREADS', '128')
    monkeypatch.setattr(module, '_executable', str)
    cif = tmp_path / 'input.cif'; cif.write_text('synthetic CIF fixture')
    work = tmp_path / 'work'; calls = []
    def fake_process(argv, **kw):
        calls.append((argv, kw['env']))
        if argv[0] == '/fake/interface':
            stage = Path(kw['cwd'])
            (stage / 'in.input').write_text('units real\nread_data data.input\n')
            (stage / 'data.input').write_text('synthetic native input')
        else:
            (work / 'opt_data/input.lammps-data').write_text('synthetic final structure')
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', fake_process)
    result = module.run(str(cif), str(work), executable='/fake/lmp', mpi_executable='/fake/mpi',
                        interface_executable='/fake/interface', n_procs=4, skip_convert=True)
    if expected is None:
        assert not result['success']
        assert not calls
        return
    assert result['success'], result
    assert result['n_procs'] == expected
    command = calls[1][0]
    assert command[:3] == ['/fake/mpi', '-np', str(expected)]
    for _, environment in calls:
        assert all(environment[name] == '1' for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'))


@pytest.mark.parametrize('cpus', [True, 0, -1, '1'])
def test_invalid_allocation_never_enters_submission(tmp_path, monkeypatch, cpus):
    monkeypatch.setattr(registry, 'get_config', lambda: SimpleNamespace(project_root=tmp_path,
        lammps_config={'executable': '/fake/lmp'}))
    monkeypatch.setattr(registry.slurm, 'submit_and_return', lambda **kw: pytest.fail('invalid allocation must not submit'))
    set_context(resource_allocation={'cpus': cpus, 'resource_review_id': 'review'})
    with pytest.raises(ValueError, match='resource_allocation.cpus'):
        registry._exec_md_optimize({'cif_path': '/fake/input.cif', 'output_dir': str(tmp_path / 'work')})
