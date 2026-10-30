"""Actual workflow workers/real Sessions, controlled provider and real sandbox."""
import json
import time
from pathlib import Path
from types import SimpleNamespace

from anthropic.types import TextBlock
import pytest

from agents.session import Session
from tests.test_parallel_workflow import factory,node


CSV='material,loading_mol_kg,method,scientific_convergence\nA,0.75,GCMC,unknown\nB,1.25,GCMC,unknown\n'
SCRIPT="""import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
with open('/inputs/table0.csv') as f: rows=list(csv.DictReader(f))
fig,ax=plt.subplots(figsize=(4,3))
ax.plot([r['material'] for r in rows],[float(r['loading_mol_kg']) for r in rows])
ax.set_ylabel('Loading (mol/kg)')
fig.savefig('/output/measured.png')
"""


def advance(rt,target,timeout=15):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        state=rt.snapshot()
        if state.get('nodes',{}).get(target,{}).get('status')in{'succeeded','failed','validation_failed','uncertain'}:
            return state['nodes'][target]
        rt.tick([]);time.sleep(.02)
    raise AssertionError('native workflow did not settle: '+str(rt.snapshot()['nodes'][target]))


def setup(factory,tmp_path,monkeypatch,kind,*,strict=True):
    root=tmp_path/'runs/tester'/(kind+'-route');table=root/'actual.csv'
    data=node('source',tool='write_file',args={'path':str(table),'content':CSV},outputs=[str(table)])
    if kind=='report':
        output=root/'reports/finished.md'
        target=node('delivery',tool='generate_scientific_report',args={'source_steps':['source'],
            'title':'Measured adsorption','instructions':'Write the actual measured results with table and limits.',
            'output_path':str(output),'report_mode':'fragment','report_requirements':{'independent_review':strict}},deps=['source'],outputs=[str(output)])
        target['agent']='communicator'
    else:
        output=root/'figures'
        target=node('delivery',tool='generate_result_figures',args={'source_steps':['source'],
            'title':'Measured adsorption','instructions':'Choose a clear plot of the actual measurements.',
            'output_dir':str(output)},deps=['source'],outputs=[str(output/'manifest.json')])
    rt=factory([data,target],cid=kind+'-route')
    from agents import registry,workspace
    monkeypatch.setattr(registry,'get_config',lambda:rt.main.config)
    monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
    calls=[]
    def provider(self,role):
        assert not role.functions and not role.handoff_to
        content=self.messages[-1]['content']
        payload=json.loads(content[0]['text']if isinstance(content,list)else content)
        phase=payload['stage'];request=payload['request'];calls.append((phase,role.name))
        if phase=='author':
            blocks=[{'kind':'table','table_id':'T1','columns':['material','loading_mol_kg','method','scientific_convergence']}]
            if request['packet'].get('independent_review_required',True):
                blocks=[{'kind':'narrative','text':'The source table retains the reported uptake values with method labels and units.',
                    'source_ids':['S1'],'claim_type':'method'}]+blocks+[{'kind':'narrative','text':'Statistical convergence remains unknown.',
                    'source_ids':['S1'],'claim_type':'limitation'}]
            else:blocks=[{'kind':'heading','text':'方法'}]+blocks
            body={'draft':{'title':'Measured adsorption','blocks':blocks}}
        elif phase=='review':
            body={'review':{'passed':True,'draft_sha256':request['draft_sha256'],
                            'checked_blocks':[0,1,2],'blocking_issues':[],'notes':[]}}
        elif phase=='visual':
            assert isinstance(content,list)and content[2]['type']=='image'
            assert content[2]['source']['media_type']=='image/png'
            body={'visual':{'supported':True,'verified':True,'issues':[],'limitations':['Controlled visual transport fixture']}}
        elif phase=='plot':body={'plot':{'script':SCRIPT,'table_ids':['T1'],
            'rationale':'Compare actual measurements with units.','limitations':['Convergence unknown.']}}
        else:raise AssertionError('unexpected model stage '+phase)
        return SimpleNamespace(id='actual-controlled-'+phase,
            content=[TextBlock(type='text',text=json.dumps(body))])
    monkeypatch.setattr(Session,'_call_api',provider)
    rt.start(1)
    return rt,output,calls


def test_native_worker_invokes_communicator_and_independent_reviewer(factory,tmp_path,monkeypatch):
    rt,output,calls=setup(factory,tmp_path,monkeypatch,'report')
    result=advance(rt,'delivery')
    assert result['status']=='succeeded',result
    assert calls==[('author','communicator'),('review','report-reviewer')]
    receipt=result['result'];assert receipt['status']=='authored_verified'
    assert '| B | 1.25 | GCMC | unknown |'in output.read_text()
    assert 'DAG 证据索引'not in output.read_text()and '```json'not in output.read_text()
    assert receipt['model_stage_receipts'][0]['call_ids']
    assert Path(receipt['model_stage_receipts'][0]['checkpoint_path']).is_file()
    assert len([n for n in rt.snapshot()['nodes'].values()if n.get('job_ids')])==0


def test_native_figure_node_writes_code_at_execution_then_real_os_sandbox(factory,tmp_path,monkeypatch):
    rt,output,calls=setup(factory,tmp_path,monkeypatch,'figure')
    result=advance(rt,'delivery',timeout=30)
    assert result['status']=='succeeded',result
    assert calls==[('plot','analyst'),('visual','figure-reviewer')]
    receipt=result['result'];assert receipt['status']=='rendered_needs_review'and receipt['ok']
    assert Path(receipt['artifacts'][0]['path']).is_file()
    assert (output/'attempt-1/script.py').read_text()==SCRIPT
    assert receipt['visual_review']['status']=='verified'
    assert receipt['visual_review']['image_receipts'][0]['sha256']==receipt['artifacts'][0]['sha256']
    assert receipt['scientific_acceptance']is False
    assert not result.get('job_ids')


def test_registry_cannot_fabricate_report_outside_current_native_worker(tmp_path):
    from agents.registry import _build_default_registry
    r=_build_default_registry()
    reply=r.execute_dict('generate_scientific_report',{'source_steps':['foreign'],
        'title':'Unowned','output_path':str(tmp_path/'report.md')})
    assert reply['error_kind']=='report_requires_native_context'and reply['executed']is False
    assert not(tmp_path/'report.md').exists()


def test_legacy_source_without_digest_gets_real_fresh_artifact_inspection(factory,tmp_path,monkeypatch):
    # The native runtime starts workers concurrently. Install the legacy
    # checkpoint fixture before letting a report take its immutable snapshot;
    # mutating a producer after packet capture correctly invalidates a report.
    import threading
    from agents import report_execution
    released=threading.Event();execute=report_execution.execute_context_report
    def after_fixture(arguments):
        assert released.wait(10),'legacy checkpoint fixture was not installed'
        return execute(arguments)
    monkeypatch.setattr(report_execution,'execute_context_report',after_fixture)
    rt,output,calls=setup(factory,tmp_path,monkeypatch,'report')
    source=advance(rt,'source');reference=source['result_ref'];checkpoint=Path(reference['checkpoint_path'])
    old=json.loads(checkpoint.read_text())
    row=next(c for c in old['memory']['tool_call_log']if c['call_id']==reference['call_id'])
    row.pop('result_sha256')
    checkpoint.write_text(json.dumps(old))
    released.set()
    completed=advance(rt,'delivery')
    assert completed['status']=='succeeded',completed
    receipt=completed['result'];inspections=[s for s in receipt['model_stage_receipts']if s['phase']=='source-source']
    assert len(inspections)==1 and inspections[0]['call_ids']
    native=json.loads((rt.root/'evidence'/(inspections[0]['call_ids'][0]+'.json')).read_text())
    assert native['tool']=='inspect_report_result_sources'and native['failed']is False
    facts=json.loads(native['result']);assert facts['artifacts'][0]['path']=='actual.csv'
    assert facts['producer_call_id']==reference['call_id']
    assert '| B | 1.25 | GCMC | unknown |'in output.read_text()


def test_figure_schema_requires_data_brief_not_planning_time_python_code():
    from agents.registry import _build_default_registry
    from agents.defns import ANALYST,COMMUNICATOR
    r=_build_default_registry()
    assert not r.validate_params('generate_result_figures',{'source_steps':['analysis'],
        'instructions':'Choose a meaningful figure from the results.','output_dir':'new-figures'})
    assert 'script'not in r.get('generate_result_figures').input_schema['properties']
    assert 'generate_result_figures'in ANALYST.function_map()and 'generate_result_figures'in COMMUNICATOR.function_map()


def test_native_worker_default_light_mode_uses_author_only(factory,tmp_path,monkeypatch):
    rt,output,calls=setup(factory,tmp_path,monkeypatch,'report',strict=False)
    result=advance(rt,'delivery')
    assert result['status']=='succeeded',result
    assert calls==[('author','communicator')]
    receipt=result['result'];assert receipt['review_mode']=='source_numeric'
    assert receipt['independent_review_performed'] is False and receipt['review_call_id'] is None
    assert receipt['scientific_acceptance'] is False and '| B | 1.25 | GCMC | unknown |' in output.read_text()
