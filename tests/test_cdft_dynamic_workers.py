"""Batch parallelism follows the actual allocation, without running a solver."""
import importlib.util
import os
from pathlib import Path
import subprocess

import pytest


EXECUTION_SENTINEL = '# controlled execution boundary; native dispatcher must not run in this fixture'


@pytest.mark.parametrize('allocated,expected', [('2', 2), ('7', 7), (None, 3)])
def test_cdft_worker_count_follows_allocation(tmp_path, allocated, expected):
    path = Path(__file__).resolve().parents[1] / 'tools/cdft/cDFT_Initialization/cdft_submit.py'
    spec = importlib.util.spec_from_file_location('cdft_worker_template_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    script = module._PBS_TEMPLATE.format(job_name='fixture', partition='compute',
        num_processes=3, walltime='00:01:00', nodelist_line='', memory_line='', dependency_line='',
        work_dir=str(tmp_path), execution_command=EXECUTION_SENTINEL)
    # Execute only initialization from the real generated batch, before any
    # scientific function or input directory is touched.
    assert script.count(EXECUTION_SENTINEL) == 1
    initialization = script.split(EXECUTION_SENTINEL, 1)[0]
    env = dict(os.environ); env.pop('SLURM_CPUS_PER_TASK', None)
    if allocated is not None: env['SLURM_CPUS_PER_TASK'] = allocated
    result = subprocess.run(['bash', '-c', initialization + '\nprintf "%s" "$NUM_PROCESSES"'],
                            env=env, capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert int(result.stdout) == expected


@pytest.mark.parametrize('allocated', ['0', '-1', 'many', '2.5'])
def test_invalid_allocation_never_becomes_unbounded_xargs(tmp_path, allocated):
    path = Path(__file__).resolve().parents[1] / 'tools/cdft/cDFT_Initialization/cdft_submit.py'
    spec = importlib.util.spec_from_file_location('cdft_bad_allocation_test', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    script = module._PBS_TEMPLATE.format(job_name='fixture', partition='compute',
        num_processes=3, walltime='00:01:00', nodelist_line='', memory_line='', dependency_line='',
        work_dir=str(tmp_path), execution_command=EXECUTION_SENTINEL)
    result = subprocess.run(['bash', '-c', script.split(EXECUTION_SENTINEL, 1)[0]],
        env={**os.environ, 'SLURM_CPUS_PER_TASK': allocated}, capture_output=True,
        text=True, timeout=3)
    assert result.returncode != 0
    assert 'CPU allocation' in result.stderr
