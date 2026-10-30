"""Native GCMC cases, rather than scripts, define the output evidence."""
import copy
from pathlib import Path
from types import SimpleNamespace
import pytest

from agents.result_validation import build_validation_dossier
from agents.workflow_patch import complete_compute_contract
from agents.state_io import json_transaction
from tests.test_parallel_workflow import factory, node, eventually

ARGS={'gas':'CO2','temperature':298,'cycles':1000,'force_field':'UFF','guest_ff':'TraPPE',
      'pressure_start':1,'pressure_end':1,'n_pressure_points':1}
DATA='Number of unitcells [a]: 2\nNumber of unitcells [b]: 2\nNumber of unitcells [c]: 2\nSimulation finished\nAverage loading absolute [mol/kg framework] 2.75 +/- 0.03\n'


def native_case(root,pressure='1.00e+05',text=DATA):
    case=root/('P_'+pressure);out=case/'Output/System_0';out.mkdir(parents=True)
    (case/'simulation.input').write_text('SimulationType MonteCarlo\nNumberOfCycles 1000\nNumberOfInitializationCycles 2000\n'
        'Forcefield UFF\nUnitCells 2 2 2\nFrameworkName fixture\nExternalTemperature 298\nExternalPressure '+pressure+'\n'
        'Component 0 MoleculeName CO2\nMoleculeDefinition TraPPE\n')
    target=out/'output.data';target.write_text(text);return target


def test_dossier_counts_native_data_cases_and_never_submit_script(tmp_path):
    root=tmp_path/'run';data=native_case(root)
    (root/'submit.sh').write_text("grep 'Average loading absolute' output.data\necho 'Simulation finished'\n")
    current={'step_id':'gcmc','job_ids':['42'],'contract':{'tool':'run_gcmc_isotherm','arguments':dict(ARGS),
        'expected_outputs':[{'kind':'directory','path':str(tmp_path/'wrong'),'pattern':'*.data'}]},'result':{'work_dir':str(root)}}
    dossier=build_validation_dossier(current,'attempt',tmp_path/'evidence')
    assert dossier['counts']=={'attempted':1,'succeeded':1,'failed':0}
    assert dossier['verdict']=='pass' and dossier['successful_artifacts']==[str(data)]


@pytest.mark.parametrize('tool',['run_gcmc_isotherm','run_gcmc_batch'])
def test_compiler_uses_only_unambiguous_concrete_native_directory(tmp_path,tool):
    candidate={'tool':tool,'arguments':dict(ARGS),'expected_outputs':[str(tmp_path/'wrong.data')]}
    result=complete_compute_contract(candidate,'gcmc',tmp_path,tmp_path/'runs/u/c')
    expected=[{'kind':'directory','path':'P_1.00e+05/Output/System_0','pattern':'*.data','min_count':1}]
    assert result['expected_outputs']==(expected if tool=='run_gcmc_isotherm' else candidate['expected_outputs'])
    assert result['arguments']==candidate['arguments']


@pytest.mark.parametrize('change',['nan','incomplete','symlink','temperature','missing_pressure','extra_output'])
def test_native_collector_does_not_infer_success_from_script_or_other_cases(tmp_path,change):
    from agents.gcmc_results import inspect_gcmc_results
    root=tmp_path/'run';data=native_case(root);args=dict(ARGS)
    if change=='nan':data.write_text(DATA.replace('2.75','NaN'))
    if change=='incomplete':data.write_text(DATA.replace('Simulation finished','Still running'))
    if change=='symlink':data.rename(root/'foreign');data.symlink_to(root/'foreign')
    if change=='temperature':args['temperature']=310
    if change=='missing_pressure':args.update(pressure_end=2,n_pressure_points=2)
    if change=='extra_output':(data.parent/'second.data').write_text(DATA)
    result=inspect_gcmc_results(root,'run_gcmc_isotherm',args)
    assert not result['all_cases_passed']


def test_batch_counts_requested_cif_gas_cases_and_keeps_loading_units(tmp_path):
    from agents.gcmc_results import inspect_gcmc_results
    source=tmp_path/'cifs';source.mkdir();(source/'fixture.cif').write_text('CIF')
    root=tmp_path/'batch'
    for gas in ('CO2','N2'):
        created=native_case(tmp_path/gas)
        target=root/gas/'298K/fixture';target.parent.mkdir(parents=True)
        (tmp_path/gas/'P_1.00e+05').rename(target)
        text=(target/'simulation.input').read_text().replace('MoleculeName CO2','MoleculeName '+gas)
        (target/'simulation.input').write_text(text)
    result=inspect_gcmc_results(root,'run_gcmc_batch',{'gases':['CO2','N2'],'cif_dir':str(source),
        'temperature':298,'pressure':1,'cycles':1000,'force_field':'UFF'})
    assert result['all_cases_passed'] and result['counts']=={'attempted':2,'succeeded':2,'failed':0}
    assert result['cases'][0]['loadings']=={'mol/kg framework':2.75}
    assert result['statistical_convergence_established'] is False


@pytest.fixture
def completed_gcmc(factory,tmp_path):
    cid='gcmc-repair';scope=tmp_path/'runs/tester'/cid
    actual=scope/'gcmc/CO2/298K/fixture';cif=tmp_path/'fixture.cif';cif.write_text('data_fixture\n')
    arguments={**ARGS,'cif':str(cif)}
    contract=node('gcmc',tool='run_gcmc_isotherm',args=arguments,
        outputs=[{'kind':'directory','path':str(scope/'gcmc/Output/System_0'),'pattern':'*.data','min_count':1}])
    contract['agent']='adsorption';rt=factory([contract],cid=cid)
    calls=[];actual.mkdir(parents=True);script=actual/'submit.sh'
    script.write_text("grep 'Average loading absolute' output.data\n")
    def submit(params):
        calls.append(copy.deepcopy(params));return {'submitted':True,'job_id':'42','work_dir':str(actual),'script':str(script)}
    rt.main.registry.get('run_gcmc_isotherm').execute=submit
    def watched(job):
        current=rt.snapshot().get('nodes',{}).get('gcmc',{})
        return {'job_id':'42','username':'tester','conv_id':cid,'tool':'run_gcmc_isotherm',
            'attempt_id':current.get('attempt_id'),'recovery_key':current.get('recovery_key'),
            'work_dir':str(actual),'terminal':True,'failed':False,'state':'COMPLETED'}
    rt.job_watch=SimpleNamespace(get=watched,list=lambda:[])
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['gcmc']['status']=='waiting_jobs')
    output=native_case(actual)
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
    assert rt.snapshot()['nodes']['gcmc']['status']=='prefinish'
    return rt,output,calls


def test_owned_completed_gcmc_repairs_contract_but_does_not_finish_or_resubmit(completed_gcmc):
    rt,output,calls=completed_gcmc;before=rt.snapshot()['nodes']['gcmc']
    result=rt.revalidate_outputs('gcmc',[str(output)])
    current=rt.snapshot()['nodes']['gcmc']
    assert result['no_resubmission'] and current['status']=='prefinish' and len(calls)==1
    assert current['token']==before['token'] and current['job_ids']==['42']
    assert current['contract']['arguments']==before['contract']['arguments']
    assert current['machine_output_check'] is True and current['validation_dossier']['verdict']=='pass'
    assert current['output_contract_history'][-1]['before']==before['contract']['expected_outputs']
    assert current['contract']['expected_outputs']==[{'kind':'file','path':str(output)}]


@pytest.mark.parametrize('change',['paused','foreign_job','arbitrary_file','tampered_call','new_goal','new_token'])
def test_gcmc_contract_repair_rejects_scope_owner_and_receipt_changes(completed_gcmc,change,monkeypatch):
    rt,output,calls=completed_gcmc;old=copy.deepcopy(rt.snapshot()['nodes']['gcmc']['contract']);proposed=[str(output)]
    if change=='paused':rt.store.user_pause(rt.workflow_id,True)
    if change=='foreign_job':rt.job_watch=SimpleNamespace(list=lambda:[],get=lambda jid:{'username':'other','conv_id':rt.conv_id,'terminal':True,'state':'COMPLETED','failed':False})
    if change=='arbitrary_file':extra=output.parent/'not-native.txt';extra.write_text(DATA);proposed=[str(extra)]
    if change=='tampered_call':
        import json
        ref=rt.snapshot()['nodes']['gcmc']['result_ref'];p=Path(ref['evidence_path']);call=json.loads(p.read_text());call['params']['temperature']=310;p.write_text(json.dumps(call))
    if change in {'new_goal','new_token'}:
        original=rt._gcmc_result_repair_proof
        def race(*a,**k):
            proof=original(*a,**k)
            if change=='new_goal':rt.main.goal_contract.version+=1
            else:
                with json_transaction(rt.store.path) as data:data['workflows'][rt.workflow_id]['nodes']['gcmc']['token']='new-owner'
            return proof
        monkeypatch.setattr(rt,'_gcmc_result_repair_proof',race)
    with pytest.raises((ValueError,PermissionError)):rt.revalidate_outputs('gcmc',proposed)
    assert rt.snapshot()['nodes']['gcmc']['contract']==old and len(calls)==1


@pytest.mark.parametrize('change',['goal','data'])
def test_gcmc_commit_rechecks_after_native_evidence_reads(completed_gcmc,monkeypatch,change):
    rt,output,_calls=completed_gcmc;before=copy.deepcopy(rt.snapshot()['nodes']['gcmc']['contract']);reads=[]
    original=rt._gcmc_result_repair_proof
    def race(*a,**k):
        proof=original(*a,**k);reads.append(True)
        if len(reads)==2:
            if change=='goal':rt.main.goal_contract.version+=1
            else:output.write_text(output.read_text().replace('2.75','3.75'))
        return proof
    monkeypatch.setattr(rt,'_gcmc_result_repair_proof',race)
    with pytest.raises((ValueError,PermissionError)):rt.revalidate_outputs('gcmc',[str(output)])
    assert rt.snapshot()['nodes']['gcmc']['contract']==before


def test_gcmc_repair_does_not_cross_foreign_live_lease(completed_gcmc):
    rt,output,_calls=completed_gcmc
    with json_transaction(rt.store.path) as data:
        current=data['workflows'][rt.workflow_id]['nodes']['gcmc']
        data['leases']['foreign']={'workflow_id':'other','step_id':'other','resources':current['contract']['resources']}
    with pytest.raises((ValueError,PermissionError)):rt.revalidate_outputs('gcmc',[str(output)])


def test_contract_repair_never_reuses_an_old_verification_to_mark_success(completed_gcmc):
    rt,output,calls=completed_gcmc
    with json_transaction(rt.store.path) as data:
        data['workflows'][rt.workflow_id]['nodes']['gcmc']['node_verification']={'passed':False,'reason':'old artifact mismatch'}
    rt.revalidate_outputs('gcmc',[str(output)])
    assert rt.snapshot()['nodes']['gcmc']['status']=='prefinish' and len(calls)==1


def test_native_owner_api_can_derive_exact_artifacts_without_supplied_paths(completed_gcmc):
    rt,output,calls=completed_gcmc;token=rt.snapshot()['nodes']['gcmc']['token']
    result=rt.revalidate_outputs('gcmc',None,expected_token=token,control_valid=lambda:True)
    assert result['actual_outputs']==[{'kind':'file','path':str(output)}]
    assert rt.snapshot()['nodes']['gcmc']['status']=='prefinish' and len(calls)==1


def test_native_owner_control_cannot_be_revived_during_final_proof(completed_gcmc,monkeypatch):
    rt,output,_calls=completed_gcmc;token=rt.snapshot()['nodes']['gcmc']['token'];valid={'value':True};reads=[]
    original=rt._gcmc_result_repair_proof
    def race(*a,**k):
        proof=original(*a,**k);reads.append(True)
        if len(reads)==2:valid['value']=False
        return proof
    monkeypatch.setattr(rt,'_gcmc_result_repair_proof',race)
    with pytest.raises(ValueError):rt.revalidate_outputs('gcmc',None,expected_token=token,control_valid=lambda:valid['value'])
    assert rt.snapshot()['nodes']['gcmc']['contract']['expected_outputs'][0]['kind']=='directory'


def test_future_contract_preserves_explicit_data_count_and_valid_pattern(tmp_path):
    from agents.output_contract import normalize_output
    original={'kind':'directory','path':str(tmp_path/'wrong'),'pattern':'*.data','min_count':3}
    result=complete_compute_contract({'tool':'run_gcmc_isotherm','arguments':dict(ARGS),'expected_outputs':[original]},'x',tmp_path,tmp_path)
    assert result['expected_outputs'][0]['min_count']==3
    assert normalize_output(result['expected_outputs'][0])


def test_multiple_or_multi_pressure_requirements_are_preserved(tmp_path):
    outputs=[str(tmp_path/'one.data'),str(tmp_path/'report.csv')]
    candidate={'tool':'run_gcmc_isotherm','arguments':dict(ARGS),'expected_outputs':outputs}
    assert complete_compute_contract(candidate,'x',tmp_path,tmp_path)['expected_outputs']==outputs
    candidate['arguments'].update(n_pressure_points=2,pressure_end=2)
    candidate['expected_outputs']=[{'kind':'directory','path':str(tmp_path/'wrong'),'pattern':'*.data','min_count':2}]
    assert complete_compute_contract(candidate,'x',tmp_path,tmp_path)['expected_outputs']==candidate['expected_outputs']


def test_native_owner_control_rechecked_after_final_stat_io(completed_gcmc,monkeypatch):
    import agents.gcmc_results as native
    rt,_output,_calls=completed_gcmc;token=rt.snapshot()['nodes']['gcmc']['token'];valid={'value':True}
    original=native.proof_files_unchanged
    def pause(*a,**k):
        unchanged=original(*a,**k);valid['value']=False;return unchanged
    monkeypatch.setattr(native,'proof_files_unchanged',pause)
    with pytest.raises(ValueError):rt.revalidate_outputs('gcmc',None,expected_token=token,control_valid=lambda:valid['value'])
    assert rt.snapshot()['nodes']['gcmc']['contract']['expected_outputs'][0]['kind']=='directory'
