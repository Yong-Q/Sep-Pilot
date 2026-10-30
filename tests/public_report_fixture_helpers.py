"""Owner-scoped report integration fixtures using real native model-stage receipts."""
import copy
import csv
import hashlib
import json
import time
from pathlib import Path
from types import SimpleNamespace
from anthropic.types import TextBlock
from agents.session import Session

MEASURED_CSV='material,measured_value,method,scientific_convergence\nsample,7,controlled_native,unknown\n'


def install_controlled_report_provider(runtime,monkeypatch):
    """Only provider transport is controlled; native packet/CP/render/review run."""
    from agents import registry,workspace
    monkeypatch.setattr(registry,'get_config',lambda:runtime.main.config)
    monkeypatch.setattr(workspace,'get_config',lambda:runtime.main.config)
    calls=[]
    def provider(self,role):
        assert not role.functions and not role.handoff_to
        content=self.messages[-1]['content']
        envelope=json.loads(content[0]['text'] if isinstance(content,list) else content)
        phase=envelope['stage'];request=envelope['request'];packet=request['packet']
        calls.append({'phase':phase,'actor':role.name,'step_id':packet['report_step'],
                      'packet_sha256':packet['packet_sha256']})
        if phase=='author':
            blocks=[{'kind':'narrative','text':'The measured value and source metadata are retained from the native files.',
                     'source_ids':[s['id']for s in packet['sources']],'claim_type':'method'}]
            # Assembly keeps actual upstream report titles, not a fabricated index.
            for source in packet['sources']:
                for document in source.get('documents',[]):
                    text=document['untrusted_content']
                    if text.startswith('# '):blocks.append({'kind':'heading','text':text.splitlines()[0][2:]})
            for table in packet['tables']:
                columns=table.get('display_columns')or table['columns']
                blocks.append({'kind':'table','table_id':table['id'],'columns':columns})
                for identity in ('material','sample_id','metric'):
                    if identity in table['columns'] and table['rows']:
                        selector={identity:table['rows'][0][identity]}
                        if (isinstance(selector[identity],str) and
                            sum(all(row[k]==v for k,v in selector.items())for row in table['rows'])==1):
                            blocks.append({'kind':'measurement','table_id':table['id'],
                                'row_selector':selector,'columns':columns[:8]})
                            break
            blocks.append({'kind':'narrative','text':'These controlled fixture results do not establish scientific convergence or predictive validity.',
                           'source_ids':[s['id']for s in packet['sources']],'claim_type':'limitation'})
            body={'draft':{'blocks':blocks}}
        elif phase=='review':
            body={'review':{'passed':True,'draft_sha256':request['draft_sha256'],
                'checked_blocks':list(range(len(request['draft']['blocks']))),'blocking_issues':[],
                'notes':['Controlled provider transport; native source and numerical binding remain enforced.']}}
        else:raise AssertionError('unexpected report-only provider stage '+phase)
        return SimpleNamespace(id='controlled-report-'+phase+'-'+str(len(calls)),
            content=[TextBlock(type='text',text=json.dumps(body))])
    monkeypatch.setattr(Session,'_call_api',provider)
    return calls


def advance_native(runtime,step_id,*,blocked=(),timeout=20):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        current=runtime.snapshot()['nodes'][step_id]
        if current['status'] in {'succeeded','failed','validation_failed','uncertain'}:return current
        runtime.tick(blocked_nodes=blocked);time.sleep(.01)
    raise AssertionError('native report fixture did not settle: '+str(runtime.snapshot()['nodes'][step_id]))


def assert_native_report_stages(runtime,node):
    receipt=node['result'];assert receipt['status']=='authored_verified'
    assert receipt['author_call_id'] and receipt['review_call_id']
    phases={row['phase']:row for row in receipt['model_stage_receipts']}
    for phase in ('author','review'):
        row=phases[phase];checkpoint=Path(row['checkpoint_path']);assert checkpoint.is_file() and row['call_ids']
        state=json.loads(checkpoint.read_text());assert state['owner']['conv_id']==runtime.conv_id
        assert state['owner']['role']=='report_'+phase
        for call_id in row['call_ids']:
            call=json.loads((runtime.root/'evidence'/(call_id+'.json')).read_text())
            assert call['tool']=='report_model_stage' and call['failed'] is False
            assert call['result_sha256']==hashlib.sha256(call['result'].encode()).hexdigest()
    return receipt


def controlled_safe_ml_product(arguments):
    """Deterministic safe RF serialization, existing fixed source labels only.

    Four TRAIN rows define the bounded tree. VALID/TEST labels remain out of
    that tree; the deliberately poor TEST score is calculated from original
    CSV rows, never replaced by a favourable invented metric or random labels.
    """
    from agents.ml_models import predict
    path=Path(arguments['data_csv']);rows=list(csv.DictReader(path.open()))
    train=rows[:4];test=rows[6:8]
    assert [(r['feature'],r['uptake'])for r in train]==[('1','2'),('2','4'),('3','6'),('4','8')]
    tree={'left':[1,-1,3,-1,5,-1,-1],'right':[2,-1,4,-1,6,-1,-1],
          'feature':[0,-2,0,-2,0,-2,-2],'threshold':[1.5,-2,2.5,-2,3.5,-2,-2],
          'value':[5,2,6,4,7,6,8]}
    model={'format':'native-rf-json-v1','features':['feature'],'target':'uptake','unit':'unspecified',
           'medians':[2.5],'trees':[tree,copy.deepcopy(tree)]}
    predictions=[predict(model,[float(row['feature'])])[0] for row in test]
    observed=[float(row['uptake'])for row in test];mean=sum(observed)/len(observed)
    r2=1-sum((a-b)**2 for a,b in zip(observed,predictions))/sum((a-mean)**2 for a in observed)
    output=Path(arguments['output_dir']);output.mkdir(parents=True)
    (output/'model.json').write_text(json.dumps(model))
    metadata={'status':'success','model_format':model['format'],'model_type':'RF',
        'metrics':{'held_out_r2':r2,'cv_r2_mean':None},'features_used':['feature'],'target':'uptake',
        'source_csv_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'train_rows':[1,2,3,4],'valid_rows':[5,6],'test_rows':[7,8],
        'synthetic_data_used':True,'labels_independently_verified':False,'scientific_acceptance':False}
    (output/'model_metadata.json').write_text(json.dumps(metadata))
    with (output/'metrics.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['metric','value','unit','status'])
        writer.writerow(['held_out_r2',r2,'dimensionless','controlled held-out fixture'])
        writer.writerow(['cv_r2_mean','','dimensionless','not evaluated'])
    with (output/'validation_predictions.csv').open('w')as stream:
        writer=csv.writer(stream);writer.writerow(['sample_id','observed','predicted','split'])
        writer.writerows([[str(index+7),row['uptake'],predictions[index],'TEST']for index,row in enumerate(test)])
    return {'status':'success','model_dir':str(output),'metrics':metadata['metrics'],
        'model_format':model['format'],'scientific_acceptance':False,
        'output_files':[str(output/name)for name in ['model.json','model_metadata.json','metrics.csv','validation_predictions.csv']]}
