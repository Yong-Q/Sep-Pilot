import copy
import hashlib
import json
from pathlib import Path

import pytest

from agents.forcefield_catalog import discover_forcefield, inspect_forcefield, parse_towhee_metadata
from agents.registry import _build_default_registry
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR

ROOT = Path(__file__).resolve().parents[1]


def test_project_assets_match_imported_source_checksums():
    catalog = json.loads((ROOT/'forcefields/towhee/catalog.json').read_text())
    assert len(catalog['entries']) == 84
    for entry in catalog['entries']:
        project = ROOT/'forcefields/towhee/ForceFields'/entry['file']
        assert hashlib.sha256(project.read_bytes()).hexdigest() == entry['sha256']
    assert (ROOT/'forcefields/towhee/UPSTREAM_LICENSE.gpl').is_file()


def test_opls_is_a_family_with_four_distinct_local_variants():
    result = discover_forcefield(ROOT,'OPLS','lammps')
    assert result['selection_status']=='ambiguous_variant'
    assert {e['name'] for e in result['candidates']} == {'OPLS-aa','OPLS-ua','OPLS-1996','OPLS-2001'}
    assert all(not e['ready_to_submit'] and e['engine_status']=='conversion_not_implemented' for e in result['candidates'])


@pytest.mark.parametrize('family', ['UFF','DREIDING','Amber','Charmm','COMPASS','TraPPE','TIP'])
def test_discovery_is_not_an_opls_special_case(family):
    result = discover_forcefield(ROOT,family)
    assert result['total_matches']>0
    assert all(Path(e['asset_path']).is_relative_to(ROOT/'forcefields/towhee') for e in result['candidates'])


def test_named_type_is_not_element_based_chemical_assignment():
    result = discover_forcefield(ROOT,'OPLS-aa','towhee',['CT','missing-mof-metal-type'])
    entry = result['candidates'][0]
    assert entry['missing_named_types']==['missing-mof-metal-type']
    assert not entry['ready_to_submit']
    assert 'chemical' in discover_forcefield(ROOT,'OPLS-aa','towhee',['CT'])['candidates'][0]['coverage_status']


def test_exact_atom_names_raw_units_and_base_charge_are_preserved():
    result = inspect_forcefield(ROOT,'OPLS-aa',['C k'],32)
    assert result['atom_types'] and result['missing_named_types']==[]
    assert any(a['names']['nonbonded']=='C k' for a in result['atom_types'])
    assert result['metadata']['mixing_rule']=='Geometric'
    assert result['metadata']['native_units']['energy_terms']=='kelvin'
    assert result['metadata']['nonbonded_type_count']==63
    assert 'not a complete' in result['metadata']['charge_warning']
    assert not result['ready_to_submit']


def test_catalog_assets_cannot_be_replaced_or_path_traversed(tmp_path):
    base=tmp_path/'forcefields/towhee/ForceFields';base.mkdir(parents=True)
    data=ROOT/'forcefields/towhee/ForceFields/towhee_ff_OPLS-aa'
    target=base/data.name;target.write_bytes(data.read_bytes())
    entry=parse_towhee_metadata(target)
    (base.parent/'catalog.json').write_text(json.dumps({'entries':[entry]}))
    target.write_text('mutated coefficients')
    with pytest.raises(ValueError,match='changed'):inspect_forcefield(tmp_path,'OPLS-aa')
    entry['file']='../../../../outside'
    (base.parent/'catalog.json').write_text(json.dumps({'entries':[entry]}))
    with pytest.raises(ValueError,match='escaped'):inspect_forcefield(tmp_path,'OPLS-aa')


def test_forcefield_discovery_is_exposed_to_main_without_compute_capability():
    names={fn.__name__ for fn in ORCHESTRATOR.functions}
    assert {'discover_forcefield','inspect_forcefield'} <= names
    registry=_build_default_registry()
    assert not registry.validate_params('discover_forcefield',{'query':'OPLS','target_engine':'lammps'})
    assert registry.validate_params('inspect_forcefield',{'forcefield_name':'OPLS-aa','max_types':100000})


def test_md_cannot_treat_towhee_opls_file_as_supported_native_opls_adapter():
    registry=_build_default_registry()
    issues=registry.validate_params('run_md_optimize',{'cif_path':'structure.cif','force_field':'OPLS-aa'})
    assert any('adapter' in issue for issue in issues)


@pytest.mark.parametrize('query, potential, mixing', [
    ('COMPASSv1','9-6','Sixth Power'), ('OPLS-aa','Lennard-Jones','Geometric'),
    ('UFF','UFF 12-6','Geometric')])
def test_different_potentials_and_mixing_are_not_flattened_into_generic_lj(query,potential,mixing):
    candidate=discover_forcefield(ROOT,query,'raspa')['candidates'][0]
    assert candidate['potential']==potential and candidate['mixing_rule']==mixing
    assert candidate['engine_status']=='conversion_not_implemented' and not candidate['ready_to_submit']


@pytest.mark.parametrize('name,version',[('ParkHijazi','14'),('Hoyt2003','5')])
def test_legacy_formats_are_found_but_never_falsely_parsed(name,version):
    candidate=discover_forcefield(ROOT,name,'towhee')['candidates'][0]
    assert candidate['format_version']==version and candidate['engine_status']=='unparsed_legacy_format_requires_review'
    assert 'nonbonded_type_count' not in candidate
    result=inspect_forcefield(ROOT,name)
    assert result['ok'] and result['supported'] is False and result['terminal_capability_result'] and 'atom_types' not in result


def test_fe_name_discovery_filters_without_guessing_or_assigning():
    result=inspect_forcefield(ROOT,'UFF',element='Fe')
    assert result['available_nonbonded_names']==['Fe3+2','Fe6+2']
    assert all(a['element']=='Fe' for a in result['atom_types'])
    assert 'not_assigned' in result['typing_status'] and not result['ready_to_submit']


def test_atom_type_pagination_is_complete_without_unbounded_page():
    values=[];offset=0
    while offset is not None:
        page=inspect_forcefield(ROOT,'UFF',max_types=17,offset=offset)
        assert len(page['atom_types'])<=17
        values.extend(a['type_number'] for a in page['atom_types'])
        offset=page['next_offset']
    assert len(values)==127 and len(set(values))==127


def test_query_namespaces_do_not_treat_bond_name_as_nonbonded_name():
    result=inspect_forcefield(ROOT,'OPLS-aa',['C'],name_namespace='nonbonded')
    assert result['matched_count']==0
    assert inspect_forcefield(ROOT,'OPLS-aa',['C'],name_namespace='bonded')['matched_count']>0


def test_charge_and_parsed_section_limits_are_explicit():
    result=inspect_forcefield(ROOT,'OPLS-aa',['C k'])
    metadata=result['metadata']
    assert metadata['bond_increment_count']==0
    assert metadata['parsed_sections']['bonded_coefficients_and_assignment'] is False
    assert 'zero Bond Increments' in metadata['reviewed_knowledge']['molecular_charge_assignment']
    assert 'not a benzene' in metadata['reviewed_knowledge']['atom_environments']['C k']


def test_water_discovery_and_reviewed_tip4p_charges_geometry():
    names={c['name'] for c in discover_forcefield(ROOT,'water')['candidates']}
    assert {'TIP3P','TIP4P','TIP5P','SPC-E'} <= names
    facts=inspect_forcefield(ROOT,'TIP4P')['metadata']['reviewed_knowledge']
    assert facts['site_charges_e']=={'H':.52,'O':0.0,'M':-1.04}
    assert facts['geometry_in_original_file']['O-H_angstrom']==.9572
    assert facts['geometry_in_original_file']['O-M_angstrom']==.15


def test_framework_charge_policy_is_separate_from_guest_forcefield():
    metadata=inspect_forcefield(ROOT,'TIP4P')['metadata']
    assert 'not forcefield' in metadata['framework_charge_policy']['default_source']
    assert 'does not overwrite' in metadata['framework_charge_policy']['guest_scope']


def test_budget_exhaustion_is_not_api_outage_and_preserves_goal(tmp_path):
    from agents.session import Session,ExecutionBudgetExceeded
    from agents.agent import Agent
    s=Session(config=AgentConfig(api_key='test',project_root=tmp_path))
    def bounded(agent):raise ExecutionBudgetExceeded('six requests consumed')
    s._call_api=bounded
    checkpoints=[];s._on_checkpoint=lambda state,reason:checkpoints.append((state,reason))
    answer=s.run_until_complete('只查UFF，不开算',agent=Agent(name='lead-orchestrator',functions=[]),verbose=False)
    assert '不是LLM API故障' in answer and '连续失败' not in answer
    assert not s.task_complete and s._waiting_for_user_input
    assert s.goal_contract.original_goal=='只查UFF，不开算'
    assert s._pending_user_interaction['tool']=='budget_decision'
    assert checkpoints[-1][1]=='execution_budget_exhausted'


def test_missing_family_does_not_fallback_to_opls_or_uff():
    result=discover_forcefield(ROOT,'SuperMOF-Quantum-9999','lammps')
    assert result['selection_status']=='not_found' and result['candidates']==[]


def test_all_target_engines_keep_unassigned_assets_not_ready():
    for engine in ('towhee','raspa','lammps','cdft'):
        assert not discover_forcefield(ROOT,'TIP4P',engine)['candidates'][0]['ready_to_submit']


def test_name_spaces_and_case_are_scientific_evidence_not_fuzzy_element_matching():
    found=inspect_forcefield(ROOT,'OPLS-aa',['C k'])
    missing=inspect_forcefield(ROOT,'OPLS-aa',['ck'])
    assert found['matched_count']>0 and missing['missing_named_types']==['ck']
