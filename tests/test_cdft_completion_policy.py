"""A batch's material coverage and CSV integrity have separate identities."""
import shutil
import pytest
from tests.test_cdft_result_facts import fixture


def dossier(tmp_path,policy=None,*,csv_wrong=False):
    from agents.result_validation import build_validation_dossier
    root,work,run=fixture(tmp_path)
    original=root/'original';original.mkdir()
    for name in ('a','b','missing'):(original/(name+'.dat')).write_bytes((work/'input/a.dat').read_bytes())
    shutil.copy2(work/'input/a.dat',work/'input/b.dat');shutil.copytree(run,work/'input_dir/b')
    (work/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\na,'+('999' if csv_wrong else '1.2')+',1.3\nb,1.2,1.3\nmissing,,\n')
    args={'action':'submit','input_dir':str(original),'gas':'N2'}
    if policy is not None:args['completion_policy']=policy
    args['min_success_ratio']=.1
    node={'contract':{'tool':'run_cdft','arguments':args,'expected_outputs':[str(work/'results.csv')]},'result':{'work_dir':str(work)}}
    return build_validation_dossier(node,'attempt',root/'evidence')


def test_default_partial_is_not_hidden_by_generic_sixty_percent_threshold(tmp_path):
    result=dossier(tmp_path)
    assert result['counts']=={'attempted':3,'succeeded':2,'failed':1}
    assert result['verdict']=='recover'
    assert result['method_facts']['cdft_completion']['completion_status']=='partial'
    assert result['method_facts']['cdft_native']['scientific_convergence'] is None


def test_strict_ignores_a_conflicting_lower_success_fraction(tmp_path):
    result=dossier(tmp_path,'strict')
    assert result['required_success_ratio']==1.0 and result['verdict']=='recover'
    assert result['counts']['attempted']==3


def test_explicit_best_effort_can_accept_only_a_declared_partial_result(tmp_path):
    result=dossier(tmp_path,'best_effort')
    assert result['verdict']=='pass'
    completion=result['method_facts']['cdft_completion']
    assert completion['completion_status']=='partial' and completion['all_materials_valid'] is False
    assert completion['successful_materials']==['a','b']
    assert [item['path'] for item in result['failed_items']]==['missing']
    assert result['counts']=={'attempted':3,'succeeded':2,'failed':1}


def test_native_csv_mismatch_cannot_be_accepted_by_best_effort_fraction(tmp_path):
    result=dossier(tmp_path,'best_effort',csv_wrong=True)
    assert result['verdict']!='pass'
    assert result['method_facts']['cdft_completion']['integrity_errors']
    assert result['counts']['attempted']==3


from tests.test_parallel_workflow import factory, node

def finish_fixture(factory,tmp_path,policy=None,partial=True):
    import json,time
    scope=tmp_path/'runs/tester/strict-cdft'
    args={'action':'collect','gas':'N2','input_dir':str(scope/'owned/original'),'job_work_dir':str(scope/'owned/batch')}
    if policy:args.update(completion_policy=policy,min_success_ratio=.5)
    contract=node('collect',tool='run_cdft',args=args,outputs=[str(scope/'owned/batch/results.csv')])
    rt=factory([contract],cid='strict-cdft');rt.start(1);token=rt.store.claim(rt.workflow_id,'collect')['token']
    root,work,run=fixture(rt.root)
    original=root/'original';original.mkdir()
    names=('a','missing') if partial else ('a',)
    for name in names:(original/(name+'.dat')).write_bytes((work/'input/a.dat').read_bytes())
    (work/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\na,1.2,1.3\n'+('missing,,\n' if partial else ''))
    rt.store.update(rt.workflow_id,'collect',token,{'status':'prefinish','phase':'prefinish','result':{'work_dir':str(work)},'tool_returned':True})
    current=rt.snapshot()['nodes']['collect']
    built=rt._create_validation_dossier('collect',current,{},current['result'])
    # Upstream owner/model labels are not the acceptance policy authority.
    built['verdict']='pass'
    rt.store.update(rt.workflow_id,'collect',token,{'validation_dossier':built})
    evidence=rt.root/'evidence';evidence.mkdir(exist_ok=True)
    (evidence/'ownerread.json').write_text(json.dumps({'call_id':'ownerread','time':time.time(),'tool':'read_file','failed':False,
        'params':{'path':str(work/'results.csv')},'result':{'path':str(work/'results.csv'),'content':'observed CSV'}}))
    return rt


def test_owner_prose_cannot_force_finish_an_incomplete_native_cdft_dossier(factory,tmp_path):
    rt=finish_fixture(factory,tmp_path)
    with pytest.raises(ValueError,match='cDFT.*completion'):
        rt.finish_node('collect',['ownerread'],'The owner declares every material successfully completed.')
    assert rt.snapshot()['nodes']['collect']['status']=='prefinish'


def test_explicit_best_effort_finish_preserves_failed_subset(factory,tmp_path):
    rt=finish_fixture(factory,tmp_path,'best_effort')
    rt.finish_node('collect',['ownerread'],'The owner accepts only material a, with missing material explicitly excluded.')
    current=rt.snapshot()['nodes']['collect'];verification=current['node_verification']
    assert current['status']=='succeeded'
    assert verification['completion_status']=='partial' and verification['all_materials_valid'] is False
    assert verification['successful_materials']==['a']
    assert [item['path'] for item in verification['failed_items']]==['missing']


def test_complete_native_table_can_finish_without_claiming_solver_convergence(factory,tmp_path):
    rt=finish_fixture(factory,tmp_path,partial=False)
    rt.finish_node('collect',['ownerread'],'All expected material tables are complete, with scientific convergence still unknown.')
    current=rt.snapshot()['nodes']['collect']
    assert current['status']=='succeeded' and current['node_verification']['all_materials_valid'] is True
    assert current['validation_dossier']['method_facts']['cdft_native']['scientific_convergence'] is None


def test_finish_rechecks_current_csv_bytes_against_the_owned_native_dossier(factory,tmp_path):
    rt=finish_fixture(factory,tmp_path,partial=False)
    path=__import__('pathlib').Path(rt.snapshot()['nodes']['collect']['result']['work_dir'])/'results.csv'
    path.write_text(path.read_text().replace('a,1.2,1.3','a,999,1.3'))
    with pytest.raises(ValueError,match='cDFT.*completion'):
        rt.finish_node('collect',['ownerread'],'The owner declares the modified CSV fully accepted.')
    assert rt.snapshot()['nodes']['collect']['status']=='prefinish'


def test_exit_failed_case_cannot_hide_finite_native_csv_tampering(tmp_path):
    import json,hashlib
    from agents.result_validation import build_validation_dossier
    root,work,run=fixture(tmp_path)
    original=root/'original';original.mkdir()
    for name in ('a','b'):(original/(name+'.dat')).write_bytes((work/'input/a.dat').read_bytes())
    shutil.copy2(work/'input/a.dat',work/'input/b.dat');shutil.copytree(run,work/'input_dir/b')
    failed=work/'input_dir/b';(failed/'solver.log').write_text('Termination status: 0\nmaximum norm for gradient: 1e-8\n');(failed/'stderr.log').write_text('native exit diagnostic\n')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (failed/'execution.json').write_text(json.dumps({'exit_code':7,'timed_out':False,'input_sha256':sha(failed/'input.dat'),
        'stdout_sha256':sha(failed/'solver.log'),'stderr_sha256':sha(failed/'stderr.log')}))
    (work/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\na,1.2,1.3\nb,999,1.3\n')
    result=build_validation_dossier({'contract':{'tool':'run_cdft','arguments':{'action':'collect','gas':'N2','input_dir':str(original),
        'completion_policy':'best_effort','min_success_ratio':.5}},'result':{'work_dir':str(work)}},'attempt',root/'evidence')
    assert result['method_facts']['cdft_native']['materials']['b']['technical_completion'] is False
    assert result['verdict']!='pass'
    assert result['method_facts']['cdft_completion']['integrity_errors']


def test_pause_during_final_artifact_io_cannot_finish_native_cdft(factory,tmp_path,monkeypatch):
    rt=finish_fixture(factory,tmp_path,partial=False)
    original=rt._artifacts
    def pause_after_artifacts(*args):
        result=original(*args);rt.store.user_pause(rt.workflow_id,True);return result
    monkeypatch.setattr(rt,'_artifacts',pause_after_artifacts)
    with pytest.raises(ValueError,match='completion commit.*superseded'):
        rt.finish_node('collect',['ownerread'],'All native data was reviewed before the user paused.')
    assert rt.snapshot()['user_paused'] is True and rt.snapshot()['nodes']['collect']['status']=='prefinish'


def test_original_input_change_during_final_artifact_io_cannot_finish(factory,tmp_path,monkeypatch):
    from pathlib import Path
    rt=finish_fixture(factory,tmp_path,partial=False)
    original=rt._artifacts
    def change_input(*args):
        result=original(*args)
        path=Path(rt.snapshot()['nodes']['collect']['contract']['arguments']['input_dir'])/'a.dat'
        path.write_bytes(path.read_bytes()+b'\n')
        return result
    monkeypatch.setattr(rt,'_artifacts',change_input)
    with pytest.raises(ValueError,match='completion commit.*superseded'):
        rt.finish_node('collect',['ownerread'],'The inputs were reviewed before their source bytes changed.')
    assert rt.snapshot()['nodes']['collect']['status']=='prefinish'
