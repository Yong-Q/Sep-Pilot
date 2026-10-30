from tests.native_child_capture import native_source
"""Synthetic adapters test path plumbing, never scientific calculations."""
import contextlib
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents import registry
from agents.installations import executor_fingerprint, inspect_installation


@pytest.fixture
def installation(tmp_path, monkeypatch):
    from bimem_agent import paths
    repo, installed = tmp_path/'repo', tmp_path/'installed'
    adapter = repo/'tools/cdft/cDFT_Initialization'
    adapter.mkdir(parents=True)
    (installed/'Molecule_FF/UFF').mkdir(parents=True)
    for path in [installed/'Molecule_FF/data_ff_Coarsening_gas', installed/'data_ff_UFF',
                 installed/'Molecule_FF/UFF/CO2.def', installed/'Molecule_FF/UFF/N2.def']:
        path.write_text('test-only asset fixture\n')
    (installed/'data_input.py').write_text("raise AssertionError('external Python must not execute')\n")
    (adapter/'data_input.py').write_text('''
_CG_FF = _UFF_DIR = _FF_UFF = 'unconfigured'
def _load_cg_ff(path):
    assert path == _CG_FF
    return {}
def generate_cdft_inputs(**kwargs):
    from pathlib import Path
    assert Path(_CG_FF).is_file() and Path(_FF_UFF).is_file()
    assert Path(_UFF_DIR, 'CO2.def').is_file()
    print('ASSET_ROOT=' + str(Path(_CG_FF).parents[1]))
    return ['test-only-input.dat']
def expand_cif_dir(*args, **kwargs):
    return {'asset_root': str(__import__('pathlib').Path(_FF_UFF).parent)}
''')
    (adapter/'cdft_submit.py').write_text('def submit_cdft_batch(**kwargs): return {"submitted":False}\n')
    (adapter/'collect_cdft_results.py').write_text('def collect_results(*args, **kwargs): return []\n')
    # Inspection verifies the actual public implementation closure, not empty
    # placeholder modules. Synthetic adapters above still prevent any compute.
    from agents.installations import IMPLEMENTATIONS
    public_root=Path(__file__).resolve().parents[1]
    for relative in IMPLEMENTATIONS['run_cdft']:
        target=repo/relative
        if target.exists():continue
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((public_root/relative).read_bytes())
    executable = installed/'DM_cdft'
    executable.write_text('#!/bin/sh\nexit 0\n'); executable.chmod(0o700)
    values = {'cdft_init_dir':installed, 'cdft_python':Path(sys.executable), 'cdft_executable':executable}
    monkeypatch.setattr(paths, 'get_runtime_section', lambda section: values if section=='software' else {})
    monkeypatch.setattr(paths, 'get_config_path', lambda section,key: values[key] if section=='software' else Path(sys.executable))
    monkeypatch.setattr(registry, 'get_config', lambda:SimpleNamespace(project_root=repo))
    monkeypatch.setattr(registry, '_register_jobwatch', lambda *args,**kwargs:None)
    calls = []
    def child(command, **kwargs):
        calls.append(command)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exec(compile(native_source(command, kwargs), '<isolated-adapter-test>', 'exec'), {'__name__': '__main__'})
        return SimpleNamespace(returncode=0, stdout=stdout.getvalue(), stderr='')
    monkeypatch.setattr('subprocess.run', child)
    return repo,installed,calls


def test_cdft_input_uses_configured_assets_and_public_adapter_despite_cached_module(installation, monkeypatch):
    repo,installed,calls = installation
    monkeypatch.setitem(sys.modules, 'data_input', SimpleNamespace(
        _CG_FF='poisoned cache',generate_cdft_inputs=lambda **kwargs:pytest.fail('cached module executed')))
    result = registry._exec_cdft({'action':'inputs','cif_dir':str(repo/'inputs'), 'gas':'CO2',
        'bulk_densities':[1.0], 'input_dir':str(repo/'owned/input'), 'job_work_dir':str(repo/'owned/work')})
    assert result.get('n_inputs') == 1 and not result.get('failed')
    assert 'ASSET_ROOT=' + str(installed) in result['stdout']
    assert len(calls) == 1
    assert sys.modules['data_input']._CG_FF == 'poisoned cache'


@pytest.mark.parametrize('missing', ['Molecule_FF/data_ff_Coarsening_gas','data_ff_UFF','Molecule_FF/UFF/CO2.def'])
def test_missing_cdft_asset_is_explicit_and_never_starts_child_or_submission(installation, missing):
    repo,installed,calls = installation
    (installed/missing).unlink()
    result = registry._exec_cdft({'action':'inputs','cif_dir':str(repo/'inputs'),'gas':'CO2'})
    assert result.get('blocked') is True and result.get('executed') is False
    assert result.get('submitted') is False and result.get('missing_dependencies')
    assert calls == []
    inspection = inspect_installation('run_cdft', {'action':'inputs','gas':'CO2'}, registry.get_config())
    assert inspection['ok'] is False
    assert any(check['path']==str(installed/missing) and not check['available'] for check in inspection['checks'])


def test_cdft_expand_cell_uses_configured_framework_asset_without_cached_module(installation, monkeypatch):
    repo,installed,calls = installation
    monkeypatch.setitem(sys.modules, 'data_input', SimpleNamespace(expand_cif_dir=lambda *a,**k:pytest.fail('cached module executed')))
    result = registry._exec_expand_cell({'cif_dir':str(repo/'inputs'),'output_dir':str(repo/'expanded')})
    assert result['expanded'] is True and result['asset_root'] == str(installed)
    assert len(calls) == 1


def test_cdft_inspection_and_executor_digest_cover_actual_configured_assets(installation):
    repo,installed,calls = installation
    before = executor_fingerprint('run_cdft', repo)
    inspection = inspect_installation('run_cdft', {'action':'inputs','gases':['CO2','N2']}, registry.get_config())
    assert inspection['ok'] is True and calls == []
    selected = {check.get('path') for check in inspection['checks']}
    assert str(installed/'Molecule_FF/data_ff_Coarsening_gas') in selected
    assert str(installed/'Molecule_FF/UFF/N2.def') in selected
    (installed/'Molecule_FF/UFF/N2.def').write_text('changed test-only asset\n')
    assert executor_fingerprint('run_cdft', repo) != before


@pytest.mark.parametrize('action', ['collect', 'submit'])
def test_cdft_existing_inputs_and_collection_do_not_require_unused_forcefield_assets(installation, monkeypatch, action):
    repo,installed,_calls = installation
    for relative in ['data_ff_UFF', 'Molecule_FF/data_ff_Coarsening_gas', 'Molecule_FF/UFF/CO2.def']:
        (installed/relative).unlink()
    observed = []
    inner = {'rows':[{'MOF':'fixture', 'CO2_henry_mol_L_atm':.25}]} if action=='collect' else {'submitted':True,'job_id':'42','status':'submitted','work_dir':str(repo/'owned/work')}
    monkeypatch.setattr('subprocess.run', lambda argv,**kwargs:observed.append(native_source(argv,kwargs)) or SimpleNamespace(
        returncode=0,stdout=json.dumps(inner),stderr=''))
    result = registry._exec_cdft({'action':action,'gas':'CO2','input_dir':str(repo/'existing-inputs'),
        'job_work_dir':str(repo/'owned/work'),'memory_mb':512})
    assert not result.get('blocked') and not result.get('failed')
    assert result['submitted']is(action=='submit')
    assert len(observed)==1 and '_cdft_data' not in observed[0]
    assert inspect_installation('run_cdft', {'action':action,'gas':'CO2'}, registry.get_config())['ok'] is True
