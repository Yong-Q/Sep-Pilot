"""Fresh charge mathematics may reuse only identical immutable evidence."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from agents.charge_contract import validate_framework_charges
from agents.parallel_workflow import _readonly_observed_artifacts
from tests.test_charge_contract import CIF
from tests.test_parallel_workflow import factory, node, eventually


@pytest.fixture
def batch(tmp_path):
    source=tmp_path/'charged';source.mkdir()
    files=[source/f'framework_{index}_pacmof.cif' for index in range(2)]
    for path in files:path.write_text(CIF)
    output=tmp_path/'charge_validation.json'
    first=validate_framework_charges(source,0.,'pacmof',1e-6,output)
    return source,files,output,first


def test_identical_report_reuse_revalidates_every_cif_and_never_writes(batch,monkeypatch):
    import ase.io.cif
    import agents.state_io
    source,files,output,_first=batch
    before=output.read_bytes();stat=output.stat();calls=[]
    original=ase.io.cif.parse_cif
    def parse(*args,**kwargs):calls.append(True);return original(*args,**kwargs)
    monkeypatch.setattr(ase.io.cif,'parse_cif',parse)
    monkeypatch.setattr(agents.state_io,'write_checkpoint',lambda *a,**k:pytest.fail('reuse cannot write checkpoint'))
    result=validate_framework_charges(source,0.,'pacmof',1e-6,output)
    assert len(calls)==len(files) and result['ok'] is True
    assert result['reused_existing_report'] is True
    assert result['report_sha256']==hashlib.sha256(before).hexdigest()
    assert output.read_bytes()==before and output.stat().st_mtime_ns==stat.st_mtime_ns and output.stat().st_ino==stat.st_ino
    assert not any(key in json.loads(before) for key in ['output_path','output_files','reused_existing_report','report_sha256'])


@pytest.mark.parametrize('change',['input_sha','source_model','tolerance','net_charge','missing_field','bad_json','wrong_type','nan','extra_field'])
def test_changed_inputs_parameters_or_invalid_existing_json_require_new_report(batch,change):
    source,files,output,_first=batch
    arguments={'expected_net_charge':0.,'charge_source':'pacmof','tolerance':1e-6}
    if change=='input_sha':files[0].write_text(CIF+'\n# changed input provenance\n')
    if change=='source_model':arguments['charge_source']='another-model'
    if change=='tolerance':arguments['tolerance']=1e-5
    if change=='net_charge':arguments['expected_net_charge']=.5
    if change in {'missing_field','wrong_type','extra_field'}:
        data=json.loads(output.read_text())
        if change=='missing_field':data.pop('results')
        if change=='wrong_type':data['ok']=1
        if change=='extra_field':data['unvalidated_extra']='unexpected'
        output.write_text(json.dumps(data))
    if change=='bad_json':output.write_text('{broken')
    if change=='nan':output.write_text('{"ok":NaN}')
    before=output.read_bytes()
    with pytest.raises(ValueError,match='fresh output_path'):
        validate_framework_charges(source,output_path=output,**arguments)
    assert output.read_bytes()==before


def test_identical_failed_scientific_report_stays_failed_on_reuse(tmp_path):
    source=tmp_path/'charged';source.mkdir();(source/'a_pacmof.cif').write_text(CIF.replace('1 0.0','1 0.25'))
    output=tmp_path/'report.json'
    first=validate_framework_charges(source,0.,'pacmof',output_path=output)
    second=validate_framework_charges(source,0.,'pacmof',output_path=output)
    assert first['ok'] is False and second['ok'] is False and second['reused_existing_report'] is True
    assert second['n_valid']==0 and second['results'][0]['cell_net_charge']==.25


def test_dag_fresh_validation_reuses_identical_old_report_without_freshness_failure(batch,factory):
    source,_files,output,_first=batch;before=output.read_bytes();mtime=output.stat().st_mtime_ns
    rt=factory([node('validate',tool='validate_framework_charges',args={'cif_path':str(source),
        'expected_net_charge':0.,'charge_source':'pacmof','output_path':str(output)},outputs=[str(output)])])
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['validate']['status'] in {'succeeded','failed','validation_failed'})
    current=rt.snapshot()['nodes']['validate']
    assert current['status']=='succeeded'
    assert current['result']['reused_existing_report'] is True
    assert output.read_bytes()==before and output.stat().st_mtime_ns==mtime


@pytest.mark.parametrize('change',['fake_flag_only','not_reused','not_readonly','input_modified','wrong_sha','wrong_path',
    'wrong_declared_path','two_outputs','changed_report','bad_counts','bool_counts',
    'different_model','different_expected_charge','different_tolerance'])
def test_validator_existing_artifact_exemption_is_exactly_bound_to_fresh_report(batch,tmp_path,change):
    source,_files,output,first=batch
    result=copy.deepcopy(first)
    result.update(reused_existing_report=True,report_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    contract=node('validate',tool='validate_framework_charges',args={'cif_path':str(source),
        'expected_net_charge':0.,'charge_source':'pacmof','output_path':str(output)},outputs=[str(output)])
    if change=='fake_flag_only':result={k:result[k] for k in ['ok','output_path','reused_existing_report','report_sha256']}
    if change=='not_reused':result['reused_existing_report']=False
    if change=='not_readonly':result['read_only_inputs']=False
    if change=='input_modified':result['input_modified']=True
    if change=='wrong_sha':result['report_sha256']='0'*64
    if change=='wrong_path':result['output_path']=str(tmp_path/'other.json')
    if change=='wrong_declared_path':contract['expected_outputs']=[str(tmp_path/'other.json')]
    if change=='two_outputs':contract['expected_outputs'].append(str(tmp_path/'other.json'))
    if change=='changed_report':
        data=json.loads(output.read_text());data['n_valid']=0;output.write_text(json.dumps(data))
        result['report_sha256']=hashlib.sha256(output.read_bytes()).hexdigest()
    if change=='bad_counts':result['n_valid']=0
    if change=='bool_counts':result['n_valid']=result['n_cifs']=True
    if change=='different_model':contract['arguments']['charge_source']='another-model'
    if change=='different_expected_charge':contract['arguments']['expected_net_charge']=.5
    if change=='different_tolerance':contract['arguments']['tolerance']=1e-5
    assert _readonly_observed_artifacts(contract,result,tmp_path)==set()


@pytest.mark.parametrize('same',[False,True])
def test_atomic_report_publication_never_replaces_concurrent_evidence(tmp_path,monkeypatch,same):
    import agents.charge_contract as charge
    source=tmp_path/'charged';source.mkdir();(source/'a_pacmof.cif').write_text(CIF)
    output=tmp_path/'report.json';captured={};original=charge.os.link
    def publish_concurrently(temporary,destination):
        raw=Path(temporary).read_bytes() if same else b'{"concurrent":"evidence"}'
        Path(destination).write_bytes(raw);captured['bytes']=raw;captured['stat']=Path(destination).stat()
        return original(temporary,destination)
    monkeypatch.setattr(charge.os,'link',publish_concurrently)
    if same:
        assert validate_framework_charges(source,0.,'pacmof',output_path=output)['reused_existing_report'] is True
    else:
        with pytest.raises(ValueError,match='fresh output_path'):
            validate_framework_charges(source,0.,'pacmof',output_path=output)
    assert output.read_bytes()==captured['bytes'] and output.stat().st_ino==captured['stat'].st_ino
    assert not list(tmp_path.glob('.charge-report-*'))
