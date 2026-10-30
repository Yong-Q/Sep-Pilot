"""Capture dispatch and native shell concurrency with an inert xargs fixture."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import pytest
from agents.watch_context import set_context, clear_context


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[1]/'tools/zeopp.py'
    spec = importlib.util.spec_from_file_location('pore_budget_under_test', path)
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'find_zeopp', lambda: '/inert/not-executed-zeo')
    inputs = tmp_path/'cifs';inputs.mkdir()
    for i in range(8): (inputs/f'{i}.cif').write_text('fixture only')
    requests = []
    monkeypatch.setattr('agents.slurm.submit_and_return', lambda **kw: requests.append(kw) or {'submitted': False})
    clear_context()
    yield module, inputs, requests
    clear_context()


def execute_inert_command(tmp_path, command, slurm_cpus):
    fixtures = tmp_path/'bin';fixtures.mkdir()
    executable = fixtures/'xargs'
    executable.write_text('#!/usr/bin/env python3\nimport os,json,sys\njson.dump({"args":sys.argv[1:],"threads":{k:os.environ.get(k) for k in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"]}},open(os.environ["CAPTURE_FILE"],"w"))\n')
    executable.chmod(0o755)
    record = tmp_path/'capture.json'
    env = {**os.environ, 'PATH': str(fixtures)+os.pathsep+os.environ['PATH'], 'CAPTURE_FILE': str(record),
           'SLURM_CPUS_PER_TASK': str(slurm_cpus), 'SLURM_JOB_ID': 'fixture-not-a-job'}
    result = subprocess.run(['bash', '-c', command], capture_output=True, text=True, env=env, timeout=10)
    assert result.returncode == 0, result.stderr
    return json.loads(record.read_text())


@pytest.mark.parametrize('reviewed,expected', [(None,4),(1,1),(2,2),(4,4)])
def test_batch_workers_and_requested_cpu_share_review_budget(prepared, tmp_path, reviewed, expected):
    module, inputs, requests = prepared
    if reviewed is not None: set_context(resource_allocation={'resource_review_id':'proof','cpus':reviewed})
    module.submit_pore_analysis(cif_dir=str(inputs), work_dir=str(tmp_path/'work'))
    request = requests[0]
    assert request['cpus_per_task'] == expected
    captured = execute_inert_command(tmp_path, request['command'], expected)
    args = captured['args']
    assert int(args[args.index('-P')+1]) == expected
    assert set(captured['threads'].values()) == {'1'}


def test_job_environment_caps_a_larger_prepared_worker_budget(prepared, tmp_path):
    module, inputs, requests = prepared
    set_context(resource_allocation={'resource_review_id':'proof','cpus':4})
    module.submit_pore_analysis(cif_dir=str(inputs), work_dir=str(tmp_path/'work'))
    captured = execute_inert_command(tmp_path, requests[0]['command'], 1)
    args = captured['args']
    assert int(args[args.index('-P')+1]) == 1


@pytest.mark.parametrize('invalid', [0,-1,True,'4'])
def test_invalid_parallelism_cannot_reach_submission(prepared, tmp_path, invalid):
    module, inputs, requests = prepared
    with pytest.raises(ValueError):
        module.submit_pore_analysis(cif_dir=str(inputs), work_dir=str(tmp_path/'work'), n_threads=invalid)
    assert not requests
