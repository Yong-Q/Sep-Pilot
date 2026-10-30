"""Read existing owned typed cDFT evidence, never accept or rerun a batch."""
import json
from pathlib import Path
import pytest
from agents.state_io import json_transaction
from tests.test_cdft_partial_recovery import partial_cdft
from tests.test_parallel_workflow import factory


def test_native_cdft_inspection_returns_existing_partial_facts_without_state_write(partial_cdft):
    rt,work,calls,_=partial_cdft
    before=rt.store.path.read_bytes();ledger=(rt.root/'recovery_state.json').read_bytes()
    result=rt.inspect_workflow_result('cdft')
    assert result['status']=='inspected' and result['read_only'] and result['no_resubmission']
    facts=result['method_facts']['cdft_native']
    assert facts['complete_material_outputs'] is False
    assert facts['scientific_convergence'] is None
    assert result['scientific_acceptance_unchanged'] is True
    assert result['source_execution_call_id']==rt.snapshot()['nodes']['cdft']['result_ref']['call_id']
    assert rt.store.path.read_bytes()==before and (rt.root/'recovery_state.json').read_bytes()==ledger
    assert len(calls)==1


@pytest.mark.parametrize('change',['csv','output','input','new_log','dossier','symlink','token','job'])
def test_cdft_inspection_refuses_changed_native_evidence_or_authority(partial_cdft,change):
    rt,work,calls,_=partial_cdft;state=rt.snapshot();node=state['nodes']['cdft'];token=node['token']
    if change=='csv':(work/'results.csv').write_text('changed CSV')
    elif change=='output':(work/'input_dir/amber/output.dat').write_text('changed native output')
    elif change=='input':(Path(node['contract']['arguments']['input_dir'])/'amber.dat').write_text('changed recipe')
    elif change=='new_log':(work/'input_dir/amber/solver.log').write_text('new previously absent solver evidence')
    elif change=='dossier':
        path=Path(node['validation_dossier']['evidence_path']);raw=json.loads(path.read_text());raw['result']['job_ids']=['other'];path.write_text(json.dumps(raw))
    elif change=='symlink':
        path=work/'input_dir/amber/hr.dat';other=work/'other';other.write_bytes(path.read_bytes());path.unlink();path.symlink_to(other)
    elif change=='token':token='stale'
    else:
        old=rt.job_watch.get;rt.job_watch.get=lambda job:{**old(job),'username':'foreign'}
    before=rt.store.path.read_bytes()
    with pytest.raises((ValueError,PermissionError,OSError)):
        rt.inspect_workflow_result('cdft',expected_token=token)
    assert rt.store.path.read_bytes()==before and len(calls)==1


def test_cdft_inspection_rechecks_controls_after_native_reads(partial_cdft,monkeypatch):
    rt,work,calls,_=partial_cdft
    from agents.report_recovery import _OwnedFiles
    original=_OwnedFiles.read;valid={'yes':True}
    def race(self,path,**kwargs):
        result=original(self,path,**kwargs)
        if str(path).endswith('/hr.dat'):valid['yes']=False
        return result
    monkeypatch.setattr(_OwnedFiles,'read',race)
    with pytest.raises(ValueError):rt.inspect_workflow_result('cdft',control_valid=lambda:valid['yes'])
    assert len(calls)==1


def recorded_cdft_inspection(rt):
    from agents.state_io import compact_checkpoint_evidence,apply_evidence_references,write_checkpoint
    result=rt.inspect_workflow_result('cdft');session=rt.main;session._evidence_root=rt.root/'evidence'
    session.memory.record_tool_call('lead-orchestrator','inspect_workflow_result',{'step_id':'cdft'},json.dumps(result),failed=False)
    saved=session.export_state();refs=compact_checkpoint_evidence(saved,session._evidence_root)
    write_checkpoint(rt.root/'session_checkpoint.json',saved);apply_evidence_references(session.memory,refs)
    return session.memory.tool_call_log[-1]


def test_real_compact_cdft_inspection_is_a_review_source_without_claiming_convergence(partial_cdft):
    from agents.scientific_review import native_inspection_sources
    rt,work,calls,_=partial_cdft;call=recorded_cdft_inspection(rt)
    assert 'result' not in call and call['result_sha256']
    sources=native_inspection_sources([call],rt.snapshot(),rt.root/'evidence')
    assert len(sources)==1
    assert sources[0]['method_facts']['cdft_native']['scientific_convergence'] is None
    assert sources[0]['method_facts']['cdft_native']['complete_material_outputs'] is False


def test_cdft_review_source_rejects_changed_original_producer(partial_cdft):
    from agents.scientific_review import native_inspection_sources
    rt,work,calls,_=partial_cdft;call=recorded_cdft_inspection(rt)
    path=Path(rt.snapshot()['nodes']['cdft']['result_ref']['evidence_path']);path.write_text(path.read_text()+' ')
    assert native_inspection_sources([call],rt.snapshot(),rt.root/'evidence')==[]


def complete_inspected_cdft(partial_cdft):
    import shutil
    rt,work,calls,_=partial_cdft
    for name in ['output.dat','hr.dat']:shutil.copy2(work/'input_dir/amber'/name,work/'input_dir/cedar'/name)
    (work/'results.csv').write_text('MOF,N2_density_mol_L,N2_henry_mol_L_atm\namber,1.2,1.3\nbirch,1.2,1.3\ncedar,1.2,1.3\n')
    node=rt.snapshot()['nodes']['cdft'];token=node['token']
    rt.store.update(rt.workflow_id,'cdft',token,{'status':'prefinish','phase':'prefinish','validation_dossier':None})
    node=rt.snapshot()['nodes']['cdft'];dossier=rt._create_validation_dossier('cdft',node,{},node['result'])
    rt.store.update(rt.workflow_id,'cdft',token,{'validation_dossier':dossier})
    call=recorded_cdft_inspection(rt)
    return rt,work,calls,call


def test_actual_json_string_inspection_receipt_can_validate_its_owned_cdft_result(partial_cdft):
    rt,work,calls,call=complete_inspected_cdft(partial_cdft)
    evidence=json.loads(Path(call['evidence_path']).read_text())
    assert isinstance(evidence['result'],str)
    rt.finish_node('cdft',[call['call_id']],'All original material data is complete; solver convergence remains unknown.')
    state=rt.snapshot()['nodes']['cdft']
    assert state['status']=='succeeded' and state['node_verification']['all_materials_valid'] is True
    assert len(calls)==1


def test_real_inspection_call_tampering_cannot_finish_cdft(partial_cdft):
    rt,work,calls,call=complete_inspected_cdft(partial_cdft)
    path=Path(call['evidence_path']);raw=json.loads(path.read_text());result=json.loads(raw['result']);result['token']='foreign';raw['result']=json.dumps(result);path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):rt.finish_node('cdft',[call['call_id']],'The existing result must not be accepted using changed evidence.')
    assert rt.snapshot()['nodes']['cdft']['status']=='prefinish' and len(calls)==1


@pytest.mark.parametrize('changed',['inspection','producer'])
def test_inspection_proof_cannot_change_during_final_cdft_finish_io(partial_cdft,monkeypatch,changed):
    rt,work,calls,call=complete_inspected_cdft(partial_cdft)
    path=Path(call['evidence_path'] if changed=='inspection' else rt.snapshot()['nodes']['cdft']['result_ref']['evidence_path'])
    original=rt._artifacts
    def race(*a,**kw):
        result=original(*a,**kw);path.write_text(path.read_text()+' ');return result
    monkeypatch.setattr(rt,'_artifacts',race)
    with pytest.raises(ValueError):rt.finish_node('cdft',[call['call_id']],'Native numbers are complete, with unknown solver convergence.')
    assert rt.snapshot()['nodes']['cdft']['status']=='prefinish' and len(calls)==1


def carried_complete_cdft(partial_cdft):
    import copy
    from agents.native_resource_carry import prove_native_resource_carry
    rt,work,calls,_=partial_cdft
    rt,work,calls,_=complete_inspected_cdft(partial_cdft)
    state=rt.snapshot();node=state['nodes']['cdft'];actual=json.loads(Path(node['result_ref']['evidence_path']).read_text())['params']
    old=copy.deepcopy(node['contract']);old['arguments']=copy.deepcopy(actual);old['arguments']['memory_mb']=actual['memory_mb']-1
    rt.store.update(rt.workflow_id,'cdft',node['token'],{'contract':old,'effective_arguments':actual,
        'resource_allocation':{'budget_final':True,'memory_mb':actual['memory_mb'],'resource_review_id':actual.get('resource_review_id')}})
    incoming=copy.deepcopy(old);incoming['arguments']=copy.deepcopy(actual)
    goal=rt.main.goal_contract.to_dict();goal.update(approved_plan_version=2,approved_nodes=[incoming])
    from agents.goal_contract import GoalContract
    rt.main.goal_contract=GoalContract.from_dict(goal)
    state=rt.snapshot();proof=prove_native_resource_carry(state,'cdft',incoming,rt.root,goal,{'42':rt.job_watch.get('42')})
    with json_transaction(rt.store.path) as data:
        s=data['workflows'][rt.workflow_id];s.update(plan_version=2,goal_contract=goal)
        s['nodes']['cdft']['approved_resource_carry']=proof;s['nodes']['cdft']['validation_dossier']=None
    current=rt.snapshot()['nodes']['cdft'];dossier=rt._create_validation_dossier('cdft',current,{},current['result'])
    rt.store.update(rt.workflow_id,'cdft',node['token'],{'validation_dossier':dossier})
    call=recorded_cdft_inspection(rt)
    return rt,call


def test_real_carried_metadata_inspection_is_source_and_can_finish(partial_cdft):
    from agents.scientific_review import native_inspection_sources
    rt,call=carried_complete_cdft(partial_cdft)
    assert len(native_inspection_sources([call],rt.snapshot(),rt.root/'evidence'))==1
    rt.finish_node('cdft',[call['call_id']],'The original complete native results are verified; convergence remains unknown.')
    assert rt.snapshot()['nodes']['cdft']['status']=='succeeded'


def test_carried_metadata_inspection_refuses_missing_current_plan_proof(partial_cdft):
    from agents.scientific_review import native_inspection_sources
    rt,call=carried_complete_cdft(partial_cdft)
    with json_transaction(rt.store.path) as data:data['workflows'][rt.workflow_id]['nodes']['cdft'].pop('approved_resource_carry')
    assert native_inspection_sources([call],rt.snapshot(),rt.root/'evidence')==[]


def test_readonly_inspection_does_not_authorize_or_require_other_branch_rebasing(partial_cdft):
    import copy
    rt,work,calls,_=partial_cdft
    other={'step_id':'other','agent':'analyst','tool':'run_cdft','arguments':{'action':'submit','gas':'CO2'},
        'depends_on':[],'expected_outputs':[str(rt.root/'other/results.csv')]}
    rt.main.goal_contract.approved_nodes.append(copy.deepcopy(other))
    actual=copy.deepcopy(other);actual['expected_outputs']=[str(rt.root/'other/data/timestamp/results.csv')]
    with json_transaction(rt.store.path) as data:
        state=data['workflows'][rt.workflow_id];state['goal_contract']=rt.main.goal_contract.to_dict()
        state['nodes']['other']={'contract':actual,'status':'waiting_jobs','job_ids':['other-job']}
    before=rt.store.path.read_bytes()
    result=rt.inspect_workflow_result('cdft')
    assert result['status']=='inspected' and result['job_ids']==['42']
    assert rt.store.path.read_bytes()==before and len(calls)==1
    from agents.cdft_partial_recovery import prove_completed_attempt
    with pytest.raises(ValueError):
        prove_completed_attempt(rt.snapshot(),'cdft',rt.root,rt.main.goal_contract.to_dict(),{'42':rt.job_watch.get('42')})
