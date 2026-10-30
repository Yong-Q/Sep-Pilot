import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from agents.figure_visual_review import image_content,validate_review,capability_error
from agents.report_authoring import ReportCapabilityGap,_read,_digest
from agents.figure_authoring import author_figures
from agents.plot_sandbox import plot_result_code

SCRIPT="""import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
v=json.load(open('/inputs/J1.json'))['value']
f,a=plt.subplots();a.bar(['source'],[v]);a.set_ylabel('source unit');f.savefig('/output/plot.png')
"""
@pytest.fixture
def case(tmp_path):
    root=tmp_path/'runs'/'owner'/'conv';root.mkdir(parents=True)
    source=root/'data.json';source.write_text('{"value":0.75,"replicate_n":null}')
    _,fact=_read(root,source,{'bytes':0})
    packet={'root':str(root),'owner':{'username':'owner','conv_id':'conv'},'report_step':'fig',
        'report_token':'token','instructions':'Engineering visual-loop fixture','tables':[],'figures':[],
        'references':[],'file_facts':[fact],'sources':[{'id':'S1','step_id':'input','token':'s',
        'call_id':'call','source_sha256':'a'*64,'structured_data':[{'id':'J1','path':'data.json',
        'sha256':fact['sha256'],'untrusted_content':json.loads(source.read_text())}]}]}
    packet['packet_sha256']=_digest(packet)
    return root,packet

def reply(verified=True,issues=None):
    return {'actor':'figure-reviewer','call_id':'visual-native','model_response_id':'controlled',
        'visual':{'supported':True,'verified':verified,'issues':issues or[],'limitations':['Engineering fixture']}}

def test_visual_feedback_returns_exact_previous_script_to_same_os_sandbox(case):
    root,packet=case;calls=[]
    def model(phase,payload):
        calls.append(payload)
        if len(calls)==2:
            assert payload['previous_artifacts'][0]['sha256']==reviews[0][0]['sha256']
            assert payload['previous_script']==SCRIPT
            assert payload['previous_script_sha256']==hashlib.sha256(SCRIPT.encode()).hexdigest()
            assert payload['previous_attempts'][0]['issues'][0]['category']=='axis_clipping'
        return {'actor':'analyst','call_id':'plot-native','plot':{'script':SCRIPT,'input_ids':['J1'],
            'rationale':'Engineering','limitations':[]}}
    reviews=[]
    def visual(phase,payload):
        assert phase=='visual';blocks,facts=image_content(root,payload['artifacts'])
        assert blocks[1]['type']=='image'and blocks[1]['source']['type']=='base64'
        assert payload['source_contract']['sources'][0]['source_text']=='{"value":0.75,"replicate_n":null}'
        reviews.append(facts)
        result=reply(len(reviews)>1,[{'category':'axis_clipping','severity':'blocking','description':'Controlled negative first review'}]if len(reviews)==1 else[])
        result['image_receipts']=facts;return result
    result=author_figures(packet,root/'figures',model_call=model,visual_call=visual,
        sandbox_call=lambda p:plot_result_code(**p,owned_root=root),control_valid=lambda:True)
    assert result['ok']and result['visual_review']['verified']and len(result['attempts'])==2
    assert result['scientific_acceptance']is False
    assert result['sandbox_receipt']['sandbox']['os_isolated']
    assert (root/'figures/attempt-1/visual-review.json').is_file()

def test_visual_budget_exhaustion_is_partial_not_success(case):
    root,packet=case
    def model(*_):return {'actor':'analyst','call_id':'p','plot':{'script':SCRIPT,'input_ids':['J1'],'rationale':'x','limitations':[]}}
    result=author_figures(packet,root/'figures',model_call=model,visual_call=lambda *_:reply(False,[{'category':'layout','severity':'blocking','description':'Controlled explicit visual rejection'}]),
        sandbox_call=lambda p:plot_result_code(**p,owned_root=root),control_valid=lambda:True)
    assert result['ok']is False and len(result['attempts'])==3 and result['scientific_acceptance']is False
    assert result['partial']and result['partial_artifacts'][0]['kind']=='PNG'
    assert result['visual_review']['verified']is False

def test_blocking_issue_cannot_be_promoted_by_model_verified_true():
    assert validate_review(reply(True,[{'category':'units','severity':'blocking','description':'Mixed dimensions'}]))['verified']is False

@pytest.mark.parametrize('fault',['sha','bytes','outside','format'])
def test_visual_png_identity_is_native_not_author_claim(case,fault):
    from PIL import Image
    root,_=case;p=root/'actual.png';Image.new('RGB',(20,20)).save(p)
    artifact={'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size,'kind':'PNG'}
    if fault=='sha':artifact['sha256']='0'*64
    if fault=='bytes':artifact['bytes']+=1
    if fault=='outside':artifact['path']=str(root.parent/'escape.png')
    if fault=='format':artifact['kind']='SVG'
    with pytest.raises((ValueError,PermissionError,OSError)):image_content(root,[artifact])

def test_unsupported_vision_is_typed_and_never_success():
    r=reply();r['visual']['supported']=False;r['visual']['verified']=False
    with pytest.raises(ReportCapabilityGap)as error:validate_review(r)
    assert error.value.details['error_kind']=='figure_vision_capability_gap'
    class APIError(Exception):status_code=400
    assert capability_error(APIError('image input is unsupported'))
    assert not capability_error(APIError('invalid tool JSON'))


def test_private_visual_request_delivers_actual_image_content(case):
    from agents.report_authoring_session import NativeReportModelBridge
    from PIL import Image
    root,packet=case;p=root/'pixels.png';Image.new('RGB',(16,16),(230,0,0)).save(p)
    artifact={'path':str(p),'kind':'PNG','bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    bridge=NativeReportModelBridge(root,packet['owner'],packet['packet_sha256'],actor_factory=lambda _:None,
        checkpoint_paths={'plot':root/'p.json','visual':root/'v.json'},control_valid=lambda:True)
    class Actor:
        def _call_api(self,role):
            assert self._report_stage_request['messages'][0]['content'][2]['type']=='image'
            assert self.messages[0]['content'][2]['source']['media_type']=='image/png'
            return SimpleNamespace(id='controlled',content=[SimpleNamespace(type='tool_use',name='return_report_stage_data',input=reply()['visual']and{'visual':reply()['visual']})])
    role=SimpleNamespace(get_instructions=lambda _: 'private visual');actor=Actor();actor.context={}
    _,body,_=bridge._request_data(actor,role,'visual','visual',{'artifacts':[artifact]},packet,'figure-reviewer',[])
    assert body['visual']['supported']and bridge._last_image_receipts[0]['sha256']==artifact['sha256']


def test_optional_visual_commentary_cannot_erase_blocking_issues():
    r=reply(False,[{"category":"layout","severity":"blocking","description":"Confirmed overlap"}]);r["visual"].pop("limitations")
    assert validate_review(r)["verified"]is False


def test_shared_provider_budget_exhaustion_retains_rendered_partial_artifacts(case):
    from agents.report_authoring import ReportStageBudgetExceeded
    root,packet=case; calls=[]
    def model(*args):
        calls.append(1)
        if len(calls)>1:raise ReportStageBudgetExceeded('bounded report model stage budget exceeded')
        return {'actor':'analyst','call_id':'p','plot':{'script':SCRIPT,'input_ids':['J1'],'rationale':'x','limitations':[]}}
    result=author_figures(packet,root/'figures',model_call=model,visual_call=lambda *_:reply(False,[{'category':'layout','severity':'blocking','description':'Controlled explicit visual rejection'}]),
        sandbox_call=lambda p:plot_result_code(**p,owned_root=root),control_valid=lambda:True)
    assert result['ok']is False and result['partial']is True
    assert len(result['attempts'])==1 and len(calls)==2
    assert Path(result['partial_artifacts'][0]['path']).is_file()
    assert result['correction_diagnostics'][-1]['status']=='model_budget_exhausted'
    assert (root/'figures/manifest.json').is_file()
    assert result['executed']is True and result['manifest']in result['output_files']
    assert result['partial_artifacts'][0]['path']in result['output_files']


def test_invalid_source_selection_is_repaired_before_any_sandbox_execution(case):
    root,packet=case;calls=[];executions=[]
    def model(phase,payload):
        calls.append(payload)
        plan={'script':SCRIPT,'input_ids':['FOREIGN']if len(calls)==1 else['J1'],'rationale':'x','limitations':[]}
        if len(calls)==2:assert payload['previous_attempts'][0]['status']=='invalid_input_selection'
        return {'actor':'analyst','call_id':'p','plot':plan}
    def sandbox(params):
        executions.append(params)
        return plot_result_code(**params,owned_root=root)
    def visual(phase,payload):
        _,facts=image_content(root,payload['artifacts'])
        result=reply(True);result['image_receipts']=facts;return result
    result=author_figures(packet,root/'figures',model_call=model,visual_call=visual,
        sandbox_call=sandbox,control_valid=lambda:True)
    assert result['ok']and len(calls)==2 and len(executions)==1
    assert executions[0]['input_files'][0]['input_id']=='J1'


def test_bridge_budget_entry_is_typed_before_actor_or_provider(case):
    from agents.report_authoring_session import NativeReportModelBridge
    from agents.report_authoring import ReportStageBudgetExceeded
    root,packet=case
    bridge=NativeReportModelBridge(root,packet['owner'],packet['packet_sha256'],actor_factory=lambda _:pytest.fail('budget exhausted must not allocate actor'),
        checkpoint_paths={'plot':root/'p.json','visual':root/'v.json'},control_valid=lambda:True)
    bridge.model_responses=8
    with pytest.raises(ReportStageBudgetExceeded):bridge('plot',{'packet':packet})


def test_final_format_failure_consumes_typed_budget_without_extra_provider_call(case):
    from agents.report_authoring_session import NativeReportModelBridge
    from agents.report_authoring import ReportStageBudgetExceeded
    root,packet=case;calls=[]
    def provider(role):
        calls.append(1);raise json.JSONDecodeError('controlled malformed JSON','?',0)
    actor=SimpleNamespace(context={},_call_api=provider,memory=SimpleNamespace(record_tool_call=lambda *a,**k:None),_checkpoint=lambda _:None)
    role=SimpleNamespace(get_instructions=lambda _:'private role')
    bridge=NativeReportModelBridge(root,packet['owner'],packet['packet_sha256'],actor_factory=lambda _:None,
        checkpoint_paths={'plot':root/'p.json'},control_valid=lambda:True);bridge.model_responses=7
    with pytest.raises(ReportStageBudgetExceeded):bridge._request_data(actor,role,'plot','plot',{},packet,'analyst',[])
    assert calls==[1]and bridge.model_responses==8


def test_empty_second_selector_is_still_mutually_exclusive_without_execution(case):
    root,packet=case
    def model(*_):return {'actor':'analyst','call_id':'p','plot':{'script':SCRIPT,'input_ids':['J1'],'table_ids':[],'rationale':'x','limitations':[]}}
    result=author_figures(packet,root/'figures',model_call=model,sandbox_call=lambda _:pytest.fail('two selector fields cannot execute'),control_valid=lambda:True)
    assert result['ok']is False and result['executed']is False and not result['attempts']
    assert len(result['correction_diagnostics'])==3
    assert Path(result['manifest']).is_file()and result['output_files']==[result['manifest']]


def test_missing_required_plot_script_gets_schema_feedback_without_execution(case):
    from agents.report_authoring_session import NativeReportModelBridge
    from agents.figure_authoring import PLOT_SCHEMA
    root,packet=case;calls=[];recorded=[]
    class Actor:
        context={};_report_stage_return_schema=PLOT_SCHEMA
        memory=SimpleNamespace(record_tool_call=lambda *a,**k:recorded.append((a,k)))
        def _checkpoint(self,_):pass
        def _call_api(self,role):
            payload=json.loads(self.messages[-1]['content']);calls.append(payload)
            plan={'input_ids':['J1'],'rationale':'x','limitations':[]}
            if len(calls)==2:
                assert payload['request']['native_protocol_feedback']['error_kind']=='model_return_schema_invalid'
                assert 'script'in payload['request']['native_protocol_feedback']['message']
                plan['script']=SCRIPT
            return SimpleNamespace(id='controlled-schema-'+str(len(calls)),content=[SimpleNamespace(type='tool_use',name='return_report_stage_data',input={'plot':plan})])
    bridge=NativeReportModelBridge(root,packet['owner'],packet['packet_sha256'],actor_factory=lambda _:None,
        checkpoint_paths={'plot':root/'p.json'},control_valid=lambda:True)
    role=SimpleNamespace(get_instructions=lambda _:'private data only')
    response,body,_=bridge._request_data(Actor(),role,'plot','plot',{},packet,'analyst',[])
    assert len(calls)==2 and body['plot']['script']==SCRIPT and len(recorded)==1
    failed=json.loads(recorded[0][0][3])
    assert failed['error_kind']=='model_return_schema_invalid'and failed['rejected_data']['plot'].get('script')is None


def test_source_reported_is_internal_evidence_but_external_reported_needs_excerpt():
    from agents.report_authoring import render_validated_draft
    packet={'title':'Owned source','sources':[{'id':'S1','step_id':'source','call_id':'native','source_sha256':'a'*64,'conditions':{},'structured_data':[]}],
        'tables':[],'figures':[],'references':[]}
    block={'kind':'narrative','claim_type':'method','source_ids':['S1'],
        'text':'The source table retains the reported uptake values with method labels and units.'}
    assert 'reported uptake'in render_validated_draft(packet,{'blocks':[block]})
    block={**block,'claim_type':'external','text':'The reported external benchmark agrees with this result.'}
    with pytest.raises(ValueError,match='literature comparison lacks'):render_validated_draft(packet,{'blocks':[block]})
