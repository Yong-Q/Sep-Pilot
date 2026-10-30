"""Approved zero coverage floors do not permit empty or fabricated results."""
import copy
import pytest
from tests.test_cdft_completion_policy import finish_fixture
from tests.test_parallel_workflow import factory
from agents.result_validation import cdft_completion_assessment


def quality():
    return {'expected_materials':['a'],'expected_materials_verified':True,
            'materials':{'a':{'output_valid':True,'technical_completion':True,'native_convergence':None}},
            'input_materials':['a'],'run_materials':['a'],'collected_csv_sha256':'observed',
            'collection_errors':[],'scientific_convergence':None}


def assess(floor=0,policy='best_effort',facts=None):
    return cdft_completion_assessment({'contract':{'arguments':{
        'completion_policy':policy,'min_success_ratio':floor}}},facts or quality(),[{'MOF':'a'}])


@pytest.mark.parametrize('floor',[0,0.0,0.1,1,1.0])
def test_schema_accepted_coverage_boundaries_pass_for_complete_real_results(floor):
    result=assess(floor)
    assert result['verdict']=='pass'
    assert result['integrity_errors']==[]
    assert result['required_success_ratio']==floor
    assert result['scientific_convergence'] is None


@pytest.mark.parametrize('floor',[-1,1.01,float('nan'),float('inf'),True,False,'0',[]])
def test_invalid_coverage_floor_stays_rejected(floor):
    result=assess(floor)
    assert result['verdict']=='invalid'
    assert 'invalid minimum success ratio' in result['integrity_errors']


@pytest.mark.parametrize('change',[
    {'materials':{'a':{'output_valid':False}}},
    {'materials':{'a':{'output_valid':True,'technical_completion':False}}},
    {'materials':{'a':{'output_valid':True,'native_convergence':False}}},
    {'input_materials':[]}, {'run_materials':[]},
    {'collected_csv_sha256':None}, {'collection_errors':['a/N2_density_mol_L: finite native mismatch']},
    {'expected_materials':[]}, {'expected_materials_verified':False}])
def test_zero_floor_never_accepts_no_success_or_integrity_error(change):
    facts=quality();facts.update(change)
    assert assess(facts=facts)['verdict']!='pass'


def test_strict_with_zero_floor_still_demands_every_material():
    facts=quality();facts['expected_materials']=['a','b']
    result=assess(policy='strict',facts=facts)
    assert result['required_success_ratio']==1.0 and result['verdict']=='recover'


def test_existing_prefinish_can_finish_under_approved_zero_without_contract_change(factory,tmp_path):
    rt=finish_fixture(factory,tmp_path,'best_effort',partial=False)
    old=rt.snapshot()['nodes']['collect'];contract=copy.deepcopy(old['contract'])
    contract['arguments']['min_success_ratio']=0
    rt.store.update(rt.workflow_id,'collect',old['token'],{'contract':contract})
    current=rt.snapshot()['nodes']['collect']
    dossier=rt._create_validation_dossier('collect',current,{},current['result'])
    rt.store.update(rt.workflow_id,'collect',old['token'],{'validation_dossier':dossier})
    rt.finish_node('collect',['ownerread'],'Observed native finite outputs are complete; scientific convergence remains unverified.')
    saved=rt.snapshot()['nodes']['collect']
    assert saved['status']=='succeeded'
    assert saved['contract']==contract
    assert saved['node_verification']['all_materials_valid'] is True
    assert saved['validation_dossier']['method_facts']['cdft_native']['scientific_convergence'] is None
