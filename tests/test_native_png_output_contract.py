"""Actual OS output rejection and a shared worker contract before authoring."""
import copy
import hashlib
import json
from pathlib import Path
import pytest
from agents.plot_sandbox import plot_result_code
from agents.plot_sandbox_worker import native_png_output_contract,native_png_output_contract_sha256,validate_png_observation
from agents.figure_authoring import author_figures
from tests.fixture_figure_authoring import packet,SCRIPT


def test_worker_bounds_have_no_mutable_or_model_override_and_keep_original_limits():
    contract=native_png_output_contract()
    assert contract['image_count']=={'min':1,'max':8}
    assert contract['width_px']==contract['height_px']=={'min':1,'max':4096}
    assert contract['image_bytes']=={'min':1,'max':16*1024**2}
    digest=hashlib.sha256(json.dumps(contract,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    assert native_png_output_contract_sha256()==digest
    contract['width_px']['max']=99999
    assert native_png_output_contract()['width_px']['max']==4096
    observed={'regular':True,'suffix':'.png','bytes':12,'format':'PNG','width_px':4096,'height_px':1}
    assert validate_png_observation(observed)['passed']
    for field,value in [('regular',False),('suffix','.pdf'),('bytes',16*1024**2+1),('width_px',4097),('height_px',0),('format','JPEG')]:
        bad={**observed,field:value};assert not validate_png_observation(bad)['passed'],field


def run(tmp_path,script):
    root=tmp_path/'runs/owner/conversation';root.mkdir(parents=True)
    table=root/'source.csv';table.write_text('x,y\n1,2\n')
    return plot_result_code(script,[{'path':str(table)}],str(root/'attempt'),owned_root=root)


def test_real_worker_observes_actual_4320px_and_returns_allowed_bounds(tmp_path):
    result=run(tmp_path,"from PIL import Image\nImage.new('RGB',(4320,1860),'white').save('/output/oversized.png')\n")
    assert not result['ok']and result['error_kind']=='plot_output_contract'and result['exit_code']==0
    assert '4320x1860' in result['error']and '4096' in result['error']
    validation=result['output_validation'];checked=validation['checked_images'][0]
    assert checked['observed']['width_px']==4320 and checked['observed']['height_px']==1860
    assert checked['allowed']==native_png_output_contract()and not checked['passed']
    assert checked['scientific_acceptance']is False and result['artifacts']==[]


@pytest.mark.parametrize('bad',['count','auxiliary','symlink'])
def test_actual_worker_rejects_extra_files_and_nonregular_outputs(tmp_path,bad):
    script="from PIL import Image\nImage.new('RGB',(2,2),'white').save('/output/valid.png')\n"
    if bad=='count':script+="for i in range(8): Image.new('RGB',(1,1),'white').save('/output/extra'+str(i)+'.png')\n"
    elif bad=='auxiliary':script+="open('/output/extra.csv','w').write('not a PNG')\n"
    else:script+="import os\nos.symlink('/inputs/table0.csv','/output/linked.png')\n"
    result=run(tmp_path,script)
    assert not result['ok']and result['error_kind']=='plot_output_contract'and not result['artifacts']
    assert result['scientific_acceptance']is False
    if bad=='count':assert result['output_validation']['observed_image_count']==9


def test_actual_generated_script_can_read_same_readonly_contract_without_scientific_acceptance(tmp_path):
    script="""import json
from PIL import Image
contract=json.load(open('/inputs/output_contract.json'))
assert contract['width_px']['max']==4096 and contract['image_bytes']['max']==16777216
Image.new('RGB',(contract['width_px']['max'],1),'white').save('/output/boundary.png')
"""
    result=run(tmp_path,script)
    assert result['ok']and result['status']=='rendered_needs_review'
    assert result['output_contract']==native_png_output_contract()
    assert result['output_schema_sha256']==native_png_output_contract_sha256()
    assert result['visual_review']['status']=='pending'and result['scientific_acceptance']is False


def test_author_sees_contract_before_first_code_and_observed_bounds_on_correction(packet):
    calls=[];root=Path(packet['root'])
    def model(phase,payload):
        assert payload['sandbox_output_contract']==native_png_output_contract()
        if calls:
            first=payload['previous_attempts'][0]
            assert first['output_validation']['checked_images'][0]['observed']['width_px']==4320
            assert first['sandbox_output_contract']==native_png_output_contract()
        calls.append(payload)
        return {'plot':{'script':SCRIPT,'table_ids':['T1'],'rationale':'Actual table only.'},
            'actor':'analyst','call_id':'controlled-'+str(len(calls)),'model_response_id':'controlled-response'}
    def failed(arguments):
        return {'ok':False,'status':'execution_failed','exit_code':0,'error':'PNG dimensions invalid: observed 4320x1860',
            'output_validation':{'checked_images':[{'observed':{'width_px':4320,'height_px':1860},'allowed':native_png_output_contract(),'passed':False}]}}
    result=author_figures(packet,root/'figures',model_call=model,sandbox_call=failed,control_valid=lambda:True,max_attempts=2)
    assert len(calls)==2 and not result['ok']and result['scientific_acceptance']is False
