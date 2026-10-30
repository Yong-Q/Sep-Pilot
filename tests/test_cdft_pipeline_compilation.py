"""Mechanical split of unexecuted pipelines, without scientific execution."""
import copy
from pathlib import Path
import pytest
from agents.registry import _build_default_registry
from agents.workflow_compiler import compile_workflow_changes
from agents.workflow_patch import WorkflowContractError


def case(tmp_path):
    root=tmp_path/'runs/owner/conv'
    args={'action':'pipeline','cif_dir':str(root/'charged'),'gases':['CO2','N2'],'temperature':298,
        'bulk_densities':[0.04,0.04],'bulk_density_unit':'mol/L','framework_charge':True,
        'input_dir':str(root/'prepared'),'job_work_dir':str(root/'calculation'),'output':str(root/'results.csv'),
        'num_processes':4,'memory_mb':16384,'walltime':'72:00:00','partition':'compute','nodelist':'node25'}
    node={'step_id':'adsorb','agent':'adsorption','tool':'run_cdft','arguments':args,'depends_on':[],
          'expected_outputs':[str(root/'results.csv')],'resource_locks':[]}
    return root,node,[{'operation':'upsert','step_id':'adsorb','node':node}]


def compile_case(tmp_path,changes,**kwargs):
    return compile_workflow_changes(changes,_build_default_registry(),tmp_path,tmp_path/'runs/owner/conv',**kwargs)


def test_pipeline_becomes_preparation_and_same_identity_submit_with_exact_science(tmp_path):
    root,original,changes=case(tmp_path);before=copy.deepcopy(changes)
    out=compile_case(tmp_path,changes);nodes={c['step_id']:c['node'] for c in out.changes}
    assert changes==before and set(nodes)=={'adsorb__prepare','adsorb'}
    prep,submit=nodes['adsorb__prepare'],nodes['adsorb']
    assert prep['arguments']['action']=='inputs' and submit['arguments']['action']=='submit'
    for key in ('cif_dir','gases','temperature','bulk_densities','bulk_density_unit','framework_charge','input_dir','job_work_dir'):
        assert prep['arguments'][key]==submit['arguments'][key]==original['arguments'][key]
    for key in ('num_processes','memory_mb','walltime','partition','nodelist'):
        assert key not in prep['arguments'] and submit['arguments'][key]==original['arguments'][key]
    assert submit['depends_on']==['adsorb__prepare']
    assert submit['input_bindings']['input_dir']=={'step_id':'adsorb__prepare','output_index':0}
    assert prep['expected_outputs']==[{'kind':'directory','path':str(root/'prepared'),'pattern':'*.dat','min_count':1}]
    assert submit['expected_outputs']==[{'kind':'file','path':str(root/'results.csv')}]
    assert any(r.get('parameter')=='cdft_pipeline_expansion' and r.get('source')=='workflow_compiler' for r in out.defaults_applied)
    assert not (root/'prepared').exists()


def test_default_prepared_path_is_stable_owned_and_downstream_identity_is_unchanged(tmp_path):
    root,node,changes=case(tmp_path);node['arguments'].pop('input_dir')
    changes.append({'operation':'upsert','step_id':'inspect','node':{'step_id':'inspect','agent':'lead-orchestrator','tool':'inspect_path','arguments':{'path':str(root/'results.csv')},'depends_on':['adsorb'],'expected_outputs':[]}})
    a=compile_case(tmp_path,changes);b=compile_case(tmp_path,changes)
    assert a==b
    nodes={c['step_id']:c['node'] for c in a.changes};directory=Path(nodes['adsorb']['arguments']['input_dir'])
    assert directory.is_relative_to(root) and nodes['inspect']['depends_on']==['adsorb']


@pytest.mark.parametrize('evidence',[{'status':'running','token':'live'},{'status':'succeeded','phase':'finish'},
    {'status':'pending','attempt_id':'claimed'},{'status':'pending','job_ids':['123']}])
def test_existing_executed_pipeline_is_not_silently_replaced(tmp_path,evidence):
    root,node,changes=case(tmp_path)
    with pytest.raises(WorkflowContractError):compile_case(tmp_path,changes,existing_steps=[copy.deepcopy(node)],source_evidence={'adsorb':evidence})


def test_never_executed_pending_pipeline_can_be_compiled(tmp_path):
    root,node,changes=case(tmp_path)
    result=compile_case(tmp_path,changes,existing_steps=[{**copy.deepcopy(node),'status':'pending'}],source_evidence={'adsorb':{'status':'pending'}})
    assert len(result.changes)==2


@pytest.mark.parametrize('unsafe',['collision','cross_scope','existing_result'])
def test_preparation_cannot_collide_escape_or_overwrite_existing_inputs(tmp_path,unsafe):
    root,node,changes=case(tmp_path)
    if unsafe=='collision':changes.append({'operation':'upsert','step_id':'adsorb__prepare','node':{'step_id':'adsorb__prepare','tool':'inspect_path','arguments':{'path':str(root)},'agent':'lead-orchestrator','depends_on':[],'expected_outputs':[]}})
    elif unsafe=='cross_scope':node['arguments']['input_dir']=str(tmp_path/'other')
    else:
        directory=Path(node['arguments']['input_dir']);directory.mkdir(parents=True);(directory/'old.dat').write_text('immutable')
    with pytest.raises(WorkflowContractError):compile_case(tmp_path,changes)
    if unsafe=='existing_result':assert (directory/'old.dat').read_text()=='immutable'


def test_preparation_inherits_original_dependencies_and_input_bindings(tmp_path):
    root,node,changes=case(tmp_path)
    producer={'step_id':'source','agent':'lead-orchestrator','tool':'inspect_path','arguments':{'path':str(root/'charged')},'depends_on':[],
        'expected_outputs':[{'kind':'directory','path':str(root/'charged'),'pattern':'*.cif'}]}
    changes.insert(0,{'operation':'upsert','step_id':'source','node':producer})
    node['depends_on']=['source'];node['input_bindings']={'cif_dir':{'step_id':'source','output_index':0}}
    nodes={c['step_id']:c['node'] for c in compile_case(tmp_path,changes).changes}
    assert nodes['adsorb__prepare']['depends_on']==['source']
    assert nodes['adsorb__prepare']['input_bindings']==node['input_bindings']
    assert nodes['adsorb']['depends_on']==['source','adsorb__prepare']


def test_explicit_final_output_contract_is_preserved_without_an_output_argument(tmp_path):
    root,node,changes=case(tmp_path);node['arguments'].pop('output')
    nodes={c['step_id']:c['node'] for c in compile_case(tmp_path,changes).changes}
    assert nodes['adsorb']['arguments']['output']==str(root/'results.csv')


def test_pipeline_cannot_silently_drop_an_explicit_output_acceptance_floor(tmp_path):
    root,node,changes=case(tmp_path);node['expected_outputs'].append(str(root/'required-extra.json'))
    with pytest.raises(WorkflowContractError):compile_case(tmp_path,changes)


def test_prior_returned_result_without_a_job_is_still_an_executed_pipeline(tmp_path):
    root,node,changes=case(tmp_path)
    with pytest.raises(WorkflowContractError):compile_case(tmp_path,changes,source_evidence={'adsorb':{'status':'pending','result':{'n_inputs':49}}})
