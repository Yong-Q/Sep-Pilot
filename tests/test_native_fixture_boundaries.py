"""Negative scientific/transport/configuration cases for the migrated fixtures."""
import hashlib
from pathlib import Path
from types import SimpleNamespace
from agents.result_validation import build_validation_dossier
from tests.native_protocol_cases import cdft_batch,gcmc_case,public_https_transport
from tests.test_result_validation import method_node
from tests.test_cdft_configured_assets import installation


def test_complete_public_adapter_closure_is_required_even_for_collection(installation):
    from agents.installations import inspect_installation
    from agents import registry
    repo,installed,calls=installation
    assert inspect_installation('run_cdft',{'action':'collect'},registry.get_config())['ok']
    helper=repo/'tools/cdft/cDFT_Initialization/cdft_scratch.py'
    helper.rename(helper.with_suffix('.retained-disabled'))
    refused=inspect_installation('run_cdft',{'action':'collect'},registry.get_config())
    assert not refused['ok']and any(c.get('path')==str(helper)and not c['available']for c in refused['checks'])
    assert calls==[]


def test_finite_cdft_csv_cannot_replace_original_material_inventory(tmp_path):
    root=tmp_path/'finite-only';root.mkdir()
    (root/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\nmaterial,1.2,1.3\n')
    node=method_node(root,'run_cdft',arguments={'action':'submit','gas':'N2'})
    dossier=build_validation_dossier(node,'unbound-inputs',tmp_path/'evidence')
    assert dossier['verdict']!='pass'and not dossier['method_facts']['cdft_native']['expected_materials_verified']


def test_native_gcmc_input_condition_mismatch_is_not_accepted_despite_finite_output(tmp_path):
    root=tmp_path/'native';gcmc_case(root)
    args={'gas':'CO2','temperature':310,'cycles':1000,'force_field':'UFF','guest_ff':'TraPPE',
        'pressure_start':1,'pressure_end':1,'n_pressure_points':1,'unit_cells':'2 2 2'}
    dossier=build_validation_dossier(method_node(root,'run_gcmc_isotherm',arguments=args),'different-temperature',tmp_path/'evidence')
    assert dossier['verdict']!='pass'and dossier['counts']=={'attempted':1,'succeeded':0,'failed':1}
    assert 'ExternalTemperature'in dossier['failed_items'][0]['reason']


def test_download_sha_mismatch_preserves_partial_evidence_without_publishing(tmp_path,monkeypatch):
    from tests.test_scientific_io_tools import _scope
    from agents.registry import _exec_download_scientific_file
    from agents.watch_context import clear_context
    root=_scope(tmp_path,monkeypatch);payload=b'controlled scientific source bytes'
    public_https_transport(monkeypatch,payload)
    target=root/'downloads/fresh.cif'
    try:result=_exec_download_scientific_file({'url':'https://example.org/file.cif','output_path':str(target),'expected_sha256':'0'*64})
    finally:clear_context()
    assert result['output_published']is False and not target.exists()
    assert 'SHA256 differs'in result['error']
    evidence=Path(result['partial_evidence_path'])
    assert evidence.is_file()and evidence.read_bytes()==payload
    assert not target.with_name('fresh.cif.part').exists()


def test_download_private_dns_is_refused_before_the_new_pool_boundary(tmp_path,monkeypatch):
    from tests.test_scientific_io_tools import _scope
    from agents.registry import _exec_download_scientific_file
    from agents.watch_context import clear_context
    root=_scope(tmp_path,monkeypatch);calls=public_https_transport(monkeypatch,b'not fetched',address='127.0.0.1')
    try:result=_exec_download_scientific_file({'url':'https://example.org/file.cif','output_path':str(root/'downloads/file.cif')})
    finally:clear_context()
    assert 'forbidden network address'in result['error']and not calls


def test_xtb_missing_explicit_configuration_cannot_reach_the_submit_spy(tmp_path,monkeypatch):
    import json
    from bimem_agent import paths
    from agents.config import AgentConfig
    from agents import registry
    runtime=tmp_path/'runtime.json';runtime.write_text(json.dumps({'software':{}}))
    monkeypatch.setattr(paths,'RUNTIME_CONFIG_PATH',runtime)
    monkeypatch.setattr(registry,'get_config',lambda:AgentConfig(project_root=tmp_path))
    calls=[];monkeypatch.setattr(registry.slurm,'submit_and_return',lambda **kw:calls.append(kw))
    result=registry.get_registry().execute_dict('run_xtb_optimize',{'input_path':'sample.xyz','output_dir':'fresh'})
    assert 'Configure software.xtb_executable'in result['error']and not calls
