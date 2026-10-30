"""Native .data custody through real workers, JSON sandbox and report stages."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import time
import pytest
from anthropic.types import TextBlock
from tests.test_parallel_workflow import factory,node,eventually
from tests.test_gcmc_result_contract import ARGS,native_case,DATA
from agents.report_authoring import build_result_packet,_read,sources_unchanged
from agents.report_recovery import _OwnedFiles
from agents.local_preparation_recovery import checkpoint_call_witness


def advance(rt,target,timeout=40):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        current=rt.snapshot()['nodes'][target]
        if current['status']in{'succeeded','failed','validation_failed','uncertain'}:return current
        rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}]);time.sleep(.03)
    raise AssertionError('native stage timed out: '+str(rt.snapshot()['nodes'][target]))


def setup_case(factory,tmp_path,monkeypatch,*,delivery=False,large=True,raw_plot=False):
    from agents import registry,workspace
    from agents.session import Session
    root=tmp_path/'runs/tester/gcmc-view';work=root/'gcmc';cif=tmp_path/'fixture.cif';cif.write_text('data_fixture\n')
    gc=node('gcmc',tool='run_gcmc_isotherm',args={**ARGS,'cif':str(cif)},outputs=[{
        'kind':'directory','path':str(work/'P_1.00e+05/Output/System_0'),'pattern':'*.data','min_count':1}]);gc['agent']='adsorption'
    nodes=[gc]
    if delivery:
        figure=node('fig',tool='generate_result_figures',args={'source_steps':['gcmc'],'instructions':'Plot native loading with original units and unknown convergence.',
            'output_dir':str(root/'figures'),'title':'Native GCMC engineering observation'},deps=['gcmc'],outputs=[{'kind':'directory','path':str(root/'figures'),'pattern':'*.png','min_count':1}])
        report=node('report',tool='generate_scientific_report',args={'source_steps':['fig'],'instructions':'Native loading and figure; short run and unknown statistical convergence.',
            'output_path':str(root/'report.md'),'title':'Native GCMC engineering observation','report_mode':'fragment','report_requirements':{'minimum_figures':1, 'independent_review': True}},deps=['fig'],outputs=[str(root/'report.md')]);report['agent']='communicator';nodes.extend([figure,report])
    rt=factory(nodes,cid='gcmc-view');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
    calls=[]
    def submit(args):
        calls.append(args);work.mkdir(parents=True);script=work/'submit.sh';script.write_text('# engineering scientific subprocess controlled\n')
        target=native_case(work)
        prefix='Number of unitcells [a]: 2\nNumber of unitcells [b]: 2\nNumber of unitcells [c]: 2\n'
        with target.open('w')as stream:
            stream.write(prefix)
            for i in range(5):
                stream.write('Block [ '+str(i)+' ]\n')
                if large:stream.write('engineering iteration filler\n'*70000)
            stream.write('Simulation finished\nAverage loading absolute [mol/kg framework] 2.75 +/- 0.03\n')
        return {'submitted':True,'job_id':'42','work_dir':str(work),'script':str(script)}
    rt.main.registry.get('run_gcmc_isotherm').execute=submit
    def watched(jid):
        n=rt.snapshot()['nodes']['gcmc'];return {'job_id':'42','username':rt.username,'conv_id':rt.conv_id,
            'tool':'run_gcmc_isotherm','attempt_id':n.get('attempt_id'),'recovery_key':n.get('recovery_key'),
            'work_dir':str(work),'terminal':True,'failed':False,'state':'COMPLETED'}
    rt.job_watch=SimpleNamespace(get=watched,list=lambda:[])
    def provider(self,role):
        assert not role.functions and not role.handoff_to
        content=self.messages[-1]['content'];payload=json.loads(content[0]['text']if isinstance(content,list)else content)
        phase=payload['stage'];request=payload['request']
        if phase=='plot':
            chosen=next(item for item in request['available_inputs']if item['format']=='json'and item.get('json_root_pointer')=='/public')
            script="import json\nimport matplotlib\nmatplotlib.use('Agg')\nimport matplotlib.pyplot as plt\nv=json.load(open("+repr(chosen['sandbox_path'])+"))['public']\nc=v['cases'][0]\nf,a=plt.subplots(figsize=(4,3));a.bar(['native'],[c['absolute_loading']['mol/kg framework']]);a.set_ylabel('Absolute loading (mol/kg framework)');f.tight_layout();f.savefig('/output/native.png')\n"
            original=[item for item in request['available_inputs'] if item.get('path','').endswith('.data')]
            assert len(original)==1 and original[0]['format']=='text' and not Path(original[0]['path']).is_absolute()
            assert original[0]['sha256']==hashlib.sha256((work/'P_1.00e+05/Output/System_0/output.data').read_bytes()).hexdigest()
            selected=[chosen['id']]
            if raw_plot:
                selected.append(original[0]['id'])
                script="with open("+repr(original[0]['sandbox_path'])+",'rb') as f:\n assert b'Average loading absolute' in f.read()\n"+script
            body={'plot':{'script':script,'input_ids':selected,'rationale':'Inspect source and native view.','limitations':['Statistical convergence unknown.']}}
        elif phase=='visual':
            assert isinstance(content,list)and content[2]['type']=='image'
            body={'visual':{'supported':True,'verified':True,'issues':[],'limitations':['Controlled engineering reviewer; no physical validation.']}}
        elif phase=='author':
            packet=request['packet'];table=packet['tables'][0];figure=packet['figures'][0]
            body={'draft':{'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'method','text':'原生数据保留载量和单位；这是工程链路观察。'},
                {'kind':'table','table_id':table['id']},{'kind':'figure','figure_id':figure['id'],'caption':'原生载量，统计收敛未知。'},
                {'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':'统计收敛未知；原生块索引不是独立样本数量。'}]}}
        elif phase=='review':body={'review':{'passed':True,'draft_sha256':request['draft_sha256'],'checked_blocks':[0,1,2,3],'blocking_issues':[],'notes':[]}}
        else:raise AssertionError('unexpected stage '+phase)
        return SimpleNamespace(id='controlled-native-'+phase,content=[TextBlock(type='text',text=json.dumps(body))])
    if delivery:monkeypatch.setattr(Session,'_call_api',provider)
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['gcmc']['status']=='waiting_jobs')
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}]);n=rt.snapshot()['nodes']['gcmc']
    assert n['status']=='prefinish' and n['validation_dossier']['verdict']=='pass'
    rt.finish_node('gcmc',[n['validation_dossier']['evidence_call_id']],'All native inputs and final loadings match; sampling convergence remains unknown.')
    assert rt.snapshot()['nodes']['gcmc']['status']=='succeeded'
    return rt,work/'P_1.00e+05/Output/System_0/output.data',calls


def packet(rt,*,mode='generate_scientific_report'):
    state=copy.deepcopy(rt.snapshot());state['nodes']['inspect-report']={'token':'host-context','status':'running','contract':{
        'step_id':'inspect-report','tool':mode,'agent':'communicator'if mode=='generate_scientific_report'else'analyst',
        'depends_on':['gcmc'],'arguments':{'source_steps':['gcmc'],'instructions':'Engineering native facts','title':'Native GCMC',
            **({'output_path':str(rt.root/'new-report.md')}if mode=='generate_scientific_report'else{'output_dir':str(rt.root/'new-figure')})}}}
    owner={'username':rt.username,'conv_id':rt.conv_id}
    def attest(source,call,raw):
        files=_OwnedFiles(rt.root,owner)
        try:
            path=source['result_ref']['checkpoint_path'];fact=checkpoint_call_witness(files,path,call,step_id=source['contract']['step_id'],require_result_hash=True)
            return {'checkpoint_verified':True,'checkpoint_path':path,'checkpoint_sha256':fact['sha256']}
        finally:files.close()
    return build_result_packet(rt.root,owner,state,'inspect-report',attest=attest)


def test_large_native_data_becomes_small_bound_public_json_without_raw_context(factory,tmp_path,monkeypatch):
    rt,raw,calls=setup_case(factory,tmp_path,monkeypatch);before=hashlib.sha256(raw.read_bytes()).hexdigest()
    assert raw.stat().st_size>8*1024**2
    p=packet(rt);doc=next(d for d in p['sources'][0]['structured_data']if d.get('metadata_schema')=='native_gcmc_report_view_v1')
    case=doc['untrusted_content']['cases'][0]
    assert case['absolute_loading']['mol/kg framework']==2.75
    assert case['temperature_K']==298 and case['pressure_Pa']==100000
    assert case['statistical_blocks']['distinct_index_count']==5
    assert not case['statistical_blocks']['is_independent_sample_count']
    assert case['scientific_convergence']=='unknown'
    assert 'job_ids'not in json.dumps(doc['untrusted_content'])
    assert len(json.dumps(p).encode())<384*1024
    assert (rt.root/doc['path']).stat().st_size<32*1024
    assert sources_unchanged(rt.root,p)
    assert hashlib.sha256(raw.read_bytes()).hexdigest()==before and len(calls)==1
    from agents.report_authoring import _fact_block
    _,binding,_=_fact_block(p,{'kind':'fact','source_id':'S1','document_id':doc['id'],'pointer':'/cases/0/temperature_K'})
    assert binding['pointer']=='/public/cases/0/temperature_K' and binding['value']==298


@pytest.mark.parametrize('change',['source_bytes_same_loading','input','extra_result','foreign_owner','producer_cp','original_dossier','token','private_envelope_public_forgery'])
def test_native_view_cannot_hide_changed_sources_or_attempts(factory,tmp_path,monkeypatch,change):
    from agents.gcmc_report_inputs import restore_gcmc_view
    rt,raw,calls=setup_case(factory,tmp_path,monkeypatch,large=False);p=packet(rt);doc=p['sources'][0]['structured_data'][0]
    state=rt.snapshot();source=state['nodes']['gcmc'];owner={'username':rt.username,'conv_id':rt.conv_id}
    if change=='source_bytes_same_loading':raw.write_bytes(raw.read_bytes()+b'changed native record\n')
    elif change=='input':(raw.parents[2]/'simulation.input').write_text('changed')
    elif change=='extra_result':(raw.parent/'other.data').write_bytes(raw.read_bytes())
    elif change=='foreign_owner':owner['username']='different-user'
    elif change=='producer_cp':Path(source['result_ref']['checkpoint_path']).write_text('{}')
    elif change=='original_dossier':Path(source['validation_dossier']['evidence_path']).write_text('{}')
    elif change=='token':source['token']='other-attempt'
    else:
        view=json.loads((rt.root/doc['path']).read_text());view['public']['cases'][0]['absolute_loading']['mol/kg framework']=999
        from agents.gcmc_report_inputs import digest
        forged=rt.root/'evidence'/('gcmc-view-'+digest(view)+'.json');forged.write_text(json.dumps(view));doc={**doc,'path':str(forged.relative_to(rt.root))}
    def attest(s,call,raw):
        files=_OwnedFiles(rt.root,{'username':rt.username,'conv_id':rt.conv_id})
        try:
            cp=s['result_ref']['checkpoint_path'];f=checkpoint_call_witness(files,cp,call,step_id='gcmc',require_result_hash=True)
            return {'checkpoint_verified':True,'checkpoint_path':cp,'checkpoint_sha256':f['sha256']}
        finally:files.close()
    path=rt.root/doc['path']
    with pytest.raises((ValueError,OSError,KeyError,TypeError,PermissionError)):
        restore_gcmc_view(rt.root,owner,state,path,path.read_bytes(),read=_read,budget={'bytes':0},file_facts=[],attest=attest)
    assert len(calls)==1
    if change in{'source_bytes_same_loading','input','extra_result'}:assert not sources_unchanged(rt.root,p)


def test_verified_native_json_and_raw_catalogue_enter_real_report(factory,tmp_path,monkeypatch):
    rt,raw,calls=setup_case(factory,tmp_path,monkeypatch,delivery=True)
    report=advance(rt,'report',timeout=55)
    assert report['status']=='succeeded',report
    fig=rt.snapshot()['nodes']['fig'];assert fig['status']=='succeeded'
    assert fig['result']['sandbox_receipt']['sandbox']['os_isolated']
    assert all(Path(x['path']).suffix=='.json' for x in fig['result']['source_inputs'])
    assert fig['result']['visual_review']['verified'] is True
    text=(rt.root/'report.md').read_text()
    assert '2.75' in text and 'mol/kg framework' in text and '统计收敛未知' in text
    assert report['result']['scientific_acceptance'] is False and len(calls)==1
    assert raw.stat().st_size>8*1024**2


def test_model_selects_original_large_source_in_real_sandbox_and_reports(factory,tmp_path,monkeypatch):
    rt,raw,calls=setup_case(factory,tmp_path,monkeypatch,delivery=True,raw_plot=True)
    report=advance(rt,'report',timeout=55)
    assert report['status']=='succeeded',report
    figure=rt.snapshot()['nodes']['fig']['result']
    assert figure['sandbox_receipt']['sandbox']['os_isolated'] is True
    originals=[item for item in figure['source_inputs']if str(item['path']).endswith('.data')]
    assert len(originals)==1 and originals[0]['sha256']==hashlib.sha256(raw.read_bytes()).hexdigest()
    assert '2.75' in (rt.root/'report.md').read_text() and report['result']['scientific_acceptance'] is False


def test_partial_native_view_keeps_failed_cases_and_does_not_promote_convergence(tmp_path):
    from agents.gcmc_results import inspect_gcmc_results
    from agents.gcmc_report_inputs import _public
    root=tmp_path/'actual';native_case(root)
    missing=root/'P_2.00e+05';native_case(tmp_path/'other',pressure='2.00e+05',text=DATA.replace('Simulation finished','Still running'))
    (tmp_path/'other/P_2.00e+05').rename(missing)
    native=inspect_gcmc_results(root,'run_gcmc_isotherm',{**ARGS,'pressure_end':2,'n_pressure_points':2})
    result=_public(native,root)
    assert result['counts']=={'attempted':2,'succeeded':1,'failed':1}
    assert [case['status']for case in result['cases']]==['valid','failed']
    assert result['cases'][1]['failure']['original_diagnostic_retained']
    assert result['scientific_convergence']=='unknown' and not result['statistical_convergence_established']


def test_native_source_stream_has_128mib_boundary_and_document_limit_stays_32mib(tmp_path):
    from agents.gcmc_report_inputs import stream_fact
    from agents.document_source_facts import stream_document_fact,MAX_DOCUMENT_BYTES
    root=tmp_path/'runs/user/limits';root.mkdir(parents=True);path=root/'sparse-native.data'
    with path.open('wb')as stream:stream.truncate(33*1024**2)
    assert MAX_DOCUMENT_BYTES==32*1024**2
    with pytest.raises(ValueError,match='bounded'):stream_document_fact(root,path,{})
    observed=stream_fact(root,path,{})
    assert observed['size']==33*1024**2
    with path.open('wb')as stream:stream.truncate(129*1024**2)
    with pytest.raises(ValueError,match='bounded'):stream_fact(root,path,{})


@pytest.mark.parametrize('change',['symlink','directory','empty','outside'])
def test_native_source_stream_rejects_unknown_or_unowned_shapes(tmp_path,change):
    from agents.gcmc_report_inputs import stream_fact
    root=tmp_path/'runs/user/limits';root.mkdir(parents=True);path=root/'source.data';path.write_text('native')
    if change=='symlink':actual=root/'original';path.rename(actual);path.symlink_to(actual)
    elif change=='directory':path=root
    elif change=='empty':path.write_bytes(b'')
    else:path=tmp_path/'foreign.data';path.write_text('foreign')
    with pytest.raises((ValueError,OSError,PermissionError)):stream_fact(root,path,{})


def test_gcmc_job_identifier_cannot_supply_scientific_narrative_number(factory,tmp_path,monkeypatch):
    from agents.report_authoring import render_validated_draft
    rt,raw,calls=setup_case(factory,tmp_path,monkeypatch,large=False);p=packet(rt)
    assert p['sources'][0]['job_ids']==['42']
    assert 'cif' not in p['sources'][0]['conditions']
    draft={'blocks':[{'kind':'narrative','source_ids':['S1'],'claim_type':'method','text':'The method loading input is 42 mol/kg.'},
        {'kind':'table','table_id':p['tables'][0]['id']},
        {'kind':'narrative','source_ids':['S1'],'claim_type':'limitation','text':'Convergence remains unknown.'}]}
    with pytest.raises(ValueError,match='unsupported numerical claim'):render_validated_draft(p,draft)
