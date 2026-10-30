"""Source-bound native rows and actual private visual-CP inheritance fixtures."""
import copy,hashlib,json
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image
from anthropic.types import TextBlock
from agents import report_authoring as report
from agents.report_native_inputs import append_thermo_tables,verified_visual_evidence
from agents.report_authoring_session import NativeReportModelBridge,ReportStageSession
from agents.config import AgentConfig
from agents.state_io import write_checkpoint,compact_checkpoint_evidence,apply_evidence_references
from tests.test_md_thermo_history import block

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fixture(tmp_path):
    root=tmp_path/'runs/owner/conv';root.mkdir(parents=True);owner={'username':'owner','conv_id':'conv'}
    log=root/'solver.log';raw=('LAMMPS engineering fixture\n'+block('minimization',steps=2,headers='Step Temp TotEng',rows='0 0 10\n2 0 3\n')+
        'SEPPILOT_STAGE production begin\n'+block(steps=20)+'SEPPILOT_STAGE production end\n').encode();log.write_bytes(raw)
    image=root/'figures/result.png';image.parent.mkdir();Image.new('RGB',(20,20),(60,90,120)).save(image)
    files={};nodes={}
    def producer(step,tool,args,result,artifacts,deps=()):
        cid=step+'call';path=root/'evidence'/(cid+'.json');path.parent.mkdir(exist_ok=True)
        call={'call_id':cid,'tool':tool,'params':args,'failed':False,'result':result};path.write_text(json.dumps(call))
        cp=root/'workers'/step/'session_checkpoint.json';cp.parent.mkdir(parents=True,exist_ok=True);cp.write_text(json.dumps({'owner':owner,'controlled_native_producer':step}))
        n={'status':'succeeded','token':step+'token','attempt_id':step+'attempt','execution_fingerprint':step+'fp',
            'contract':{'step_id':step,'tool':tool,'agent':'analyst'if tool!='generate_scientific_report'else'communicator','arguments':args,'depends_on':list(deps)},
            'result':result,'result_ref':{'call_id':cid,'evidence_path':str(path),'checkpoint_path':str(cp)},'artifacts':artifacts}
        nodes[step]=n;return n
    def identity(step):
        from agents.workflow_patch import execution_contract
        n=nodes[step];return {'step_id':step,'token':n['token'],'attempt_id':n['attempt_id'],'execution_fingerprint':n['execution_fingerprint'],
            'call_id':n['result_ref']['call_id'],'source_sha256':sha(n['result_ref']['evidence_path']),'contract_sha256':report._digest(execution_contract(n['contract']))}
    producer('raw','write_file',{'path':str(log),'content':raw.decode()},{'status':'ok'}, {str(log):{'size':len(raw)}})
    packet={'owner':owner,'root':str(root),'packet_sha256':'a'*64,'report_step':'figure','report_token':'figuretoken'}
    paths={phase:root/'workers'/hashlib.sha256(('report-stage:figure:figuretoken:'+phase).encode()).hexdigest()[:24]/'session_checkpoint.json'for phase in('plot','visual')}
    def factory(phase):
        a=ReportStageSession(config=AgentConfig(api_key='test-placeholder-not-a-credential',project_root=tmp_path));a._evidence_root=root/'evidence'
        def checkpoint(saved,reason):
            saved['owner']={**owner,'step_id':'figure','role':'report_'+phase};refs=compact_checkpoint_evidence(saved,root/'evidence');write_checkpoint(paths[phase],saved);apply_evidence_references(a.memory,refs)
        a._on_checkpoint=checkpoint
        def provider(role):
            body={'plot':{'script':'# controlled','input_ids':['L1'],'rationale':'fixture','limitations':[]}}if phase=='plot'else{'visual':{'supported':True,'verified':True,'issues':[],'limitations':[]}}
            return SimpleNamespace(id='controlled-'+phase,content=[TextBlock(type='text',text=json.dumps(body))])
        a._call_api=provider;return a
    b=NativeReportModelBridge(root,owner,packet['packet_sha256'],actor_factory=factory,checkpoint_paths=paths,control_valid=lambda:True)
    b('plot',{'packet':packet});image_ref={'path':str(image),'sha256':sha(image),'bytes':image.stat().st_size,'kind':'PNG'}
    from agents.figure_visual_review import VISUAL_SCHEMA,validate_review
    visual=validate_review(b('visual',{'packet':packet,'output_schema':VISUAL_SCHEMA,'artifacts':[image_ref]}));visual['status']='verified'
    result={'ok':True,'status':'rendered_needs_review','source_packet_sha256':packet['packet_sha256'],'source_steps':['raw'],'source_receipts':[identity('raw')],
        'artifacts':[image_ref],'visual_review':visual,'source_inputs':[{'id':'L1','format':'text','path':'solver.log','sha256':sha(log)}]}
    producer('figure','generate_result_figures',{'source_steps':['raw'],'output_dir':str(image.parent)},result,{str(image):{'size':image.stat().st_size}},['raw'])
    state={'username':'owner','conv_id':'conv','plan_version':1,'goal_contract':{'version':1},'nodes':nodes}
    def add_report(step,sources,mode='fragment'):
        nodes[step]={'status':'running','token':step+'token','contract':{'step_id':step,'tool':'generate_scientific_report','agent':'communicator','depends_on':sources,
            'arguments':{'source_steps':sources,'title':'Native engineering source','report_mode':mode,'output_path':str(root/(step+'.md'))}}}
    def attest(node,call,raw):
        cp=node['result_ref']['checkpoint_path'];assert sha(node['result_ref']['evidence_path'])==hashlib.sha256(raw).hexdigest()
        return {'checkpoint_verified':True,'checkpoint_path':cp,'checkpoint_sha256':sha(cp)}
    return root,log,image,state,attest,producer,identity,add_report


def authored(packet):
    blocks=[{'kind':'narrative','claim_type':'method','source_ids':[s['id']for s in packet['sources']],'text':'The native tables retain original quantities, units and unknown convergence.'}]
    blocks += [{'kind':'table','table_id':t['id']}for t in packet['tables']]
    blocks += [{'kind':'figure','figure_id':f['id']}for f in packet['figures']]
    draft={'blocks':blocks};text=report.render_validated_draft(packet,draft)
    return {'verified':True,'author_call_id':'controlled-author','review_call_id':'controlled-review','draft':draft,'draft_sha256':report._digest(draft),'packet_sha256':packet['packet_sha256'],'markdown':text}


def test_complete_native_rows_types_units_anchors_and_current_visual_are_rendered(tmp_path):
    root,log,image,state,attest,_,_,add=fixture(tmp_path);add('report',['figure'])
    packet=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert len(packet['tables'])==2 and sum(t['row_count']for t in packet['tables'])==4
    first=packet['tables'][0];assert first['format']=='native_thermo'and isinstance(first['rows'][0]['Temp'],float)
    assert first['rows'][0]['TotEng']==10 and first['rows'][1]['TotEng']==3
    assert first['sha256']==sha(log)and first['row_sources'][0]['line']==int(first['rows'][0]['native_row'])
    assert first['column_units']['TotEng']=='kcal/mol'and first['native_context']['scientific_convergence']=='unknown'
    assert packet['figures'][0]['visual_review']=='verified'
    text=authored(packet)['markdown'];assert 'independent visual review verified'in text and 'visual review remains pending'not in text
    assert '| 0.0 | 0.0 | 10.0 |'in text and '[native log]'in text and 'Native context:'in text
    result=report.publish_report(root,root/'report.md',packet,authored(packet))
    assert result['native_table_bindings']and result['figure_bindings'][0]['visual_evidence']['image_sha256']==sha(image)
    assert not list(root.rglob('*.csv'))


def test_cross_fragment_assembly_rechecks_original_visual_and_deduplicates_tables(tmp_path):
    root,log,image,state,attest,producer,identity,add=fixture(tmp_path)
    for step in('parta','partb'):
        add(step,['figure']);p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,step,attest=attest)
        result=report.publish_report(root,root/(step+'.md'),p,authored(p))
        producer(step,'generate_scientific_report',state['nodes'][step]['contract']['arguments'],result,{str(root/(step+'.md')):{'size':(root/(step+'.md')).stat().st_size}},['figure'])
    add('assembly',['parta','partb'],'assembly')
    p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'assembly',attest=attest)
    assert len(p['tables'])==2 and len(p['figures'])==1 and p['figures'][0]['visual_review']=='verified'
    assert len(p['tables'][0]['source_ids'])==2 and len(p['figures'][0]['source_ids'])==2
    state['nodes']['figure']['execution_fingerprint']='new-epoch'
    with pytest.raises(ValueError,match='stale source attempt'):report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'assembly',attest=attest)


@pytest.mark.parametrize('change',['review_checkpoint_digest','review_token','review_image_sha','review_model_id'])
def test_wrong_or_stale_visual_evidence_never_becomes_verified(tmp_path,change):
    root,log,image,state,attest,_,_,add=fixture(tmp_path);node=state['nodes']['figure'];review=node['result']['visual_review']
    cid=review['call_id'];ep=root/'evidence'/(cid+'.json');data=json.loads(ep.read_text());record=json.loads(data['result'])
    if change=='review_checkpoint_digest':data['result']=data['result']+' '
    elif change=='review_token':data['params']['report_token']='foreign-token'
    elif change=='review_image_sha':record['image_receipts'][0]['sha256']='0'*64;data['result']=json.dumps(record)
    else:record['model_response_id']='forged';data['result']=json.dumps(record)
    ep.write_text(json.dumps(data));_,fact=report._read(root,image,{'bytes':0})
    with pytest.raises(ValueError):verified_visual_evidence(root,state,'figure',fact,read=report._read,budget={'bytes':0},file_facts=[],cache={})


def test_author_self_declared_verified_or_missing_native_proof_remains_pending(tmp_path):
    root,log,image,state,attest,producer,_,add=fixture(tmp_path)
    producer('self','analyze_measurements',{}, {'visual_review':{'verified':True,'status':'verified'}},{str(image):{'size':image.stat().st_size}})
    add('report',['self']);p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'report',attest=attest)
    assert p['figures'][0]['visual_evidence']['status']=='pending'
    cp=root/'workers'/hashlib.sha256(b'report-stage:figure:figuretoken:visual').hexdigest()[:24]/'session_checkpoint.json';cp.rename(cp.with_suffix('.retained'))
    add('second',['figure']);p=report.build_result_packet(root,{'username':'owner','conv_id':'conv'},state,'second',attest=attest)
    assert p['figures'][0]['visual_review'].startswith('pending')


def test_native_rows_missing_cells_preserved_and_limits_not_raised(tmp_path):
    root=tmp_path;raw=('LAMMPS\n'+block(rows='0 300\n10 nan\n')).encode();fact={'path':'solver.log','sha256':hashlib.sha256(raw).hexdigest()};tables=[]
    append_thermo_tables(raw,fact,'S1',tables,max_rows=2)
    assert tables[0]['rows'][-1]['Temp']is None and tables[0]['native_context']['complete']is False
    with pytest.raises(ValueError,match='chunked'):append_thermo_tables(raw,fact,'S1',[],max_rows=1)
    assert tables[0]['native_context']['native_steps']==10 and tables[0]['row_count']==2
