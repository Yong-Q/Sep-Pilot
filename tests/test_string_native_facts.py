"""Synthetic native String artifacts distinguish facts from assumptions."""
import copy
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def string_run(tmp_path):
    root = tmp_path/'owned-string'
    data = root/'string_input_CO2'
    native = data/'frame/dir1'; native.mkdir(parents=True)
    source = native/'input.dat'
    source.write_text('cutoff(A) FH_signal Mass(g/mol) Tempearture(K) Running_steps\n15 0 44 298 20\n#_of_points delta_frac delta_angle_degree\n2 0.001 1\n')
    output = native/'string_path.dat'; output.write_text('0 0 0 0 0 0 1\n1 0 0 0 0 0 2\n')
    log = native/'run.log'; log.write_text('info:\t1\t0\t424\t0.1\ninfo:\t0\t20\t424\t0.011737\n')
    manifest = data/'manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['structure','direction','input_dat']);writer.writeheader()
        writer.writerow({'structure':'frame','direction':1,'input_dat':str(source)})
    receipt={'schema_version':1,'success':True,'manifest_path':str(manifest),'manifest_sha256':sha(manifest),
        'executable_path':'/operator-installed/solver','results':[{'structure':'frame','direction':'1',
        'input_path':str(source),'input_sha256':sha(source),'output_path':str(output),'output_sha256':sha(output),
        'run_log_path':str(log),'run_log_sha256':sha(log),'point_count':2,'expected_point_count':2,
        'native_info':{'converged':False,'iterations':20,'expanded_atoms':424}}]}
    (data/'string_results.json').write_text(json.dumps(receipt))
    (root/'submit.sh').write_text('#!/bin/sh\n#SBATCH -t 00:15:00\n')
    return root


def inspect(root, **kwargs):
    from bimem_agent.result_inspector import inspect_string_run
    return inspect_string_run(root, **kwargs)


def test_legacy_string_facts_preserve_flags_and_keep_semantics_units_unknown(string_run):
    receipt=string_run/'string_input_CO2/string_results.json';before=receipt.read_bytes()
    facts=inspect(string_run, approved_walltime='00:15:00', approved_walltime_source='workflow_contract',
                  scheduler={'elapsed':'00:00:17'})
    assert facts['ok'] and facts['adapter_reported_success'] is True
    row=facts['rows'][0]
    assert row['column_count']==7 and row['point_count']==2
    assert row['native_info']=={'converged':False,'iterations':20,'expanded_atoms':424}
    assert row['declared_iteration_budget']==20
    assert facts['engine_reported_convergence'] is False
    assert facts['column_semantics']['status']=='unknown'
    assert facts['column_semantics']['columns'] is None
    assert facts['energy_unit'] is None
    assert facts['timing']['approved_walltime_seconds']==900
    assert facts['timing']['submitted_walltime_seconds']==900
    assert facts['timing']['scheduler_elapsed_seconds']==17
    assert facts['timing']['native_process_cpu_seconds'] is None
    assert facts['requires_scientific_review'] is True
    assert receipt.read_bytes()==before


@pytest.mark.parametrize('mutation', ['output_hash','manifest_hash','native_info','missing_log','nan','eight_columns','outside','symlink'])
def test_string_facts_reject_unbound_or_unsafe_artifacts(string_run,tmp_path,mutation):
    receipt=string_run/'string_input_CO2/string_results.json';data=json.loads(receipt.read_text());row=data['results'][0]
    output=Path(row['output_path'])
    if mutation=='output_hash': row['output_sha256']='0'*64
    if mutation=='manifest_hash': data['manifest_sha256']='0'*64
    if mutation=='native_info': row['native_info']['converged']=True
    if mutation=='missing_log': Path(row['run_log_path']).unlink()
    if mutation=='nan': output.write_text('0 0 0 0 0 0 nan\n1 0 0 0 0 0 2\n');row['output_sha256']=sha(output)
    if mutation=='eight_columns': output.write_text('0 0 0 0 0 0 1 2\n1 0 0 0 0 0 2 3\n');row['output_sha256']=sha(output)
    if mutation in {'outside','symlink'}:
        outside=tmp_path/'private.dat';outside.write_text('SYNTHETIC_OUTSIDE_SECRET')
        if mutation=='outside': row['output_path']=str(outside)
        else: output.unlink();output.symlink_to(outside)
    receipt.write_text(json.dumps(data));before=receipt.read_bytes()
    facts=inspect(string_run)
    assert not facts['ok'] and facts['errors']
    assert 'SYNTHETIC_OUTSIDE_SECRET' not in json.dumps(facts)
    assert receipt.read_bytes()==before


def test_string_budget_is_not_inferred_from_elapsed_or_native_log_cpu_time(string_run):
    facts=inspect(string_run,scheduler={'elapsed':'00:00:17'})
    assert facts['timing']['approved_walltime_seconds'] is None
    assert facts['timing']['submitted_walltime_seconds']==900
    assert facts['timing']['scheduler_elapsed_seconds']==17
    assert facts['timing']['adapter_native_wall_seconds'] is None


def test_string_receipt_protocol_is_declaration_not_binary_source_attestation(string_run):
    receipt=string_run/'string_input_CO2/string_results.json';data=json.loads(receipt.read_text())
    data['output_protocol']={'id':'tutrast_string_seven_columns_v1','column_count':7,
        'columns':['fractional_a','fractional_b','fractional_c','orientation_alpha','orientation_beta','orientation_gamma','potential'],
        'provenance':{'declared_by_adapter':True,'source_verified':True,'binary_source_binding':True},'energy_unit':'eV'}
    receipt.write_text(json.dumps(data))
    facts=inspect(string_run)
    semantics=facts['column_semantics']
    assert semantics['status']=='declared_by_adapter'
    assert len(semantics['columns'])==7 and 's' not in semantics['columns']
    assert semantics['provenance']=={'declared_by_adapter':True,'source_verified':False,'binary_source_binding':False}
    assert facts['energy_unit'] is None


@pytest.mark.parametrize('mutation',['malformed_row','untrusted_protocol','wrong_input_points','invalid_elapsed'])
def test_string_fact_boundaries_do_not_invent_protocol_or_timing(string_run,mutation):
    receipt=string_run/'string_input_CO2/string_results.json';data=json.loads(receipt.read_text())
    if mutation=='malformed_row': data['results']=['invalid']
    if mutation=='untrusted_protocol': data['output_protocol']={'id':'another-engine','column_count':7,'columns':['s','x','y','z','a','b','V']}
    if mutation=='wrong_input_points':
        source=Path(data['results'][0]['input_path']);source.write_text(source.read_text().replace('2 0.001 1','3 0.001 1'))
        data['results'][0]['input_sha256']=sha(source)
    receipt.write_text(json.dumps(data))
    facts=inspect(string_run,scheduler={'elapsed_seconds':float('nan'),'elapsed':'not-a-duration'})
    if mutation in {'malformed_row','wrong_input_points'}: assert not facts['ok']
    assert facts['column_semantics']['status']=='unknown'
    assert facts['timing']['scheduler_elapsed_seconds'] is None
    assert facts['timing']['approved_walltime_seconds'] is None


def test_string_typed_facts_reach_dossier_without_rewriting_old_receipt(string_run,tmp_path):
    from agents.result_validation import build_validation_dossier
    receipt=string_run/'string_input_CO2/string_results.json';before=receipt.read_bytes()
    node={'workflow_id':'owned','step_id':'string','plan_version':1,'contract':{'tool':'run_string_tst',
        'arguments':{'output_dir':str(string_run),'gas':'CO2','walltime':'00:15:00'},'expected_outputs':[str(receipt)]},
        'result':{'work_dir':str(string_run),'result_json_path':str(receipt)},'path_manifest':{'result_dir':str(string_run)}}
    dossier=build_validation_dossier(node,'original-attempt',tmp_path/'evidence')
    facts=dossier['method_facts']['string_native']
    assert facts['rows'][0]['native_info']['converged'] is False
    assert facts['timing']['approved_walltime_seconds']==900
    assert facts['timing']['scheduler_elapsed_seconds'] is None
    assert dossier['requires_agent_review'] is True and receipt.read_bytes()==before


def test_string_schema_upgrade_preserves_old_dossier_and_registered_inspection(string_run,tmp_path,monkeypatch):
    from agents import result_validation
    from bimem_agent import result_inspector
    receipt=string_run/'string_input_CO2/string_results.json'
    node={'workflow_id':'owned','step_id':'string','plan_version':1,'contract':{'tool':'run_string_tst',
        'arguments':{'output_dir':str(string_run),'gas':'CO2'},'expected_outputs':[str(receipt)]},
        'result':{'work_dir':str(string_run)},'path_manifest':{'result_dir':str(string_run)}}
    with monkeypatch.context() as old_schema:
        old_schema.setattr(result_validation,'VALIDATION_SCHEMA_VERSION',3)
        old=result_validation.build_validation_dossier(node,'attempt',tmp_path/'evidence')
    before=Path(old['evidence_path']).read_bytes()
    new=result_validation.build_validation_dossier(node,'attempt',tmp_path/'evidence')
    assert old['evidence_call_id']!=new['evidence_call_id'] and Path(old['evidence_path']).read_bytes()==before
    spec=string_run/'spec.json';spec.write_text('{}')
    monkeypatch.setattr(result_inspector,'load_run',lambda *a,**k:{'run_dir':str(string_run),'result_path':str(receipt),'spec_path':str(spec)})
    facts=result_inspector.inspect_run('owned')['string_native']
    assert facts['rows'][0]['native_info']['converged'] is False


@pytest.mark.parametrize('blocker', [None,'foreign_owner','different_workdir'])
def test_numeric_string_inspection_uses_owned_dispatch_budget_and_scheduler_elapsed(string_run,tmp_path,monkeypatch,blocker):
    from agents import registry, job_watch
    from agents.config import AgentConfig
    from agents.watch_context import set_context,clear_context
    record={'job_id':'42','tool':'run_string_tst','username':'alice','conv_id':'string',
        'work_dir':str(string_run),'recovery_key':'key','attempt_id':'attempt',
        'terminal':True,'failed':False,'state':'COMPLETED'}
    if blocker=='foreign_owner': record['username']='foreign'
    ledger=tmp_path/'runs/alice/string/recovery_state.json';ledger.parent.mkdir(parents=True)
    ledger.write_text(json.dumps({'key':{'tool':'run_string_tst','attempt_id':'attempt','job_ids':['42'],
        'params':{'walltime':'00:15:00','gas':'CO2'},'last_result':{'result_json_path':str(string_run/'string_input_CO2/string_results.json')}}}))
    monkeypatch.setattr(job_watch,'get_watch',lambda:SimpleNamespace(get=lambda _:record))
    monkeypatch.setattr(registry,'get_config',lambda:AgentConfig(api_key='test',project_root=tmp_path))
    monkeypatch.setattr(registry.slurm,'check_job_status',lambda *a,**kw:{'job_id':'42','status':'COMPLETED','terminal':True,'failed':False,'elapsed':'00:00:17'})
    set_context('alice','string','lead-orchestrator','')
    try:
        result=registry._exec_inspect_run({'run_id':'42','work_dir':str(tmp_path/'different') if blocker=='different_workdir' else str(string_run)})
    finally:
        clear_context()
    if blocker:
        assert result.get('blocked') and 'string_native' not in result
    else:
        facts=result['string_native']
        assert facts['timing']['approved_walltime_seconds']==900
        assert facts['timing']['approved_walltime_source']=='owned_dispatch_ledger'
        assert facts['timing']['scheduler_elapsed_seconds']==17
        assert facts['rows'][0]['native_info']['converged'] is False
