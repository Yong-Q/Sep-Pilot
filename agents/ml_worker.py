"""Allocation-only fit/descriptor inference/acquisition; numeric JSON models."""
import csv,hashlib,io,json,math,os
from pathlib import Path
from .ml_dataset import read_owned,read_version,immutable_json,immutable_bytes,owner,training_signature


def predict(model,features):
    from .ml_models import predict as infer
    return infer(model,features)


def train(data,cpus,model_type='RF'):
    from .ml_models import train as fit
    return fit(data,cpus,model_type)


def _csv(root,path,headers,rows):
    stream=io.StringIO();writer=csv.DictWriter(stream,fieldnames=headers,lineterminator='\n',extrasaction='ignore')
    writer.writeheader();writer.writerows(rows)
    return immutable_bytes(root,path,stream.getvalue().encode())


def _candidates(root,request,data):
    spec=request.get('candidate_spec')
    if spec is None:
        eligible=set(data['split']['train'])-set(data['labels']);ids=request['candidate_ids']
        if len(ids)!=len(set(ids))or not ids or not set(ids)<=eligible:
            raise ValueError('acquisition consumes only unique unlabelled TRAIN candidates')
        return [dict(data['rows'][key],sample_id=key)for key in ids],[]
    if spec.get('features')!=data['features']or spec.get('format')not in{'features','cifs'}:
        raise ValueError('candidate descriptors differ from trained feature schema')
    for fact in spec['source_facts']:
        if read_owned(root,fact['snapshot_path'])[1]['sha256']!=fact['sha256']:
            raise ValueError('candidate input snapshot changed')
    rows=[];failures=[]
    for source in spec['rows']:
        row=dict(source)
        try:
            if spec['format']=='cifs':
                from .ml_cif_features import extract_features
                feature=extract_features(row['snapshot_path'],owned_root=root,expected_sha256=row['structure_sha256'])
                if feature.error:raise ValueError(feature.error_code+': '+feature.error)
                row['features']=feature.feature_values(data['features'])
                row['source']={**row['source'],'descriptor_version':feature.descriptor_version}
            if len(row['features'])!=len(data['features']):raise ValueError('candidate feature width mismatch')
            for value in row['features']:
                if value is not None and (isinstance(value,bool)or not isinstance(value,(int,float))or not math.isfinite(value)):
                    raise ValueError('candidate descriptor must be finite or missing')
            rows.append(row)
        except (OSError,ValueError,TypeError,KeyError)as error:
            failures.append({'sample_id':row.get('sample_id'),'cif':row.get('cif'),
                'error':str(error),'error_kind':'ml_candidate_features'})
    if not rows:raise ValueError('no valid candidate descriptors; '+str(failures[:4]))
    return rows,failures


def run_request(root,path,expected_sha):
    job=os.environ.get('SLURM_JOB_ID','')
    if not job.isdecimal():raise ValueError('ML computation requires a real Slurm allocation')
    raw,fact=read_owned(root,path)
    if fact['sha256']!=expected_sha:raise ValueError('native ML request bytes changed')
    request=json.loads(raw)
    if request.get('owner')!=owner(root):raise PermissionError('ML worker owner mismatch')
    from . import ml_dataset,ml_models
    implementations={'worker_sha256':Path(__file__),'dataset_adapter_sha256':Path(ml_dataset.__file__),
                     'model_adapter_sha256':Path(ml_models.__file__)}
    if request.get('cif_adapter_sha256'):
        from . import ml_cif_features
        implementations['cif_adapter_sha256']=Path(ml_cif_features.__file__)
    for key,source in implementations.items():
        if request.get(key)!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('ML implementation changed: '+key)
    version=request['dataset'];data=read_version(root,version['path'],version['sha256'])
    action=request['action'];output=Path(request['work_dir']);cpus=int(os.environ.get('SLURM_CPUS_PER_TASK','1'))
    if not 1<=cpus<=request['allocated_cpus']:raise ValueError('ML worker exceeds reviewed CPU allocation')
    artifacts={};facts={}
    if action=='train':
        model,facts=train(data,cpus,request.get('model_type','RF'));model['dataset_version']=version['sha256']
        model['training_signature']=training_signature(data)
        artifacts['model']=immutable_json(root,output/'model.json',model)
        importance=facts['feature_importance']
        artifacts['feature_importance']=_csv(root,output/'feature_importance.csv',
            ['feature','importance'],importance)
        valid=[]
        for key in facts['valid_ids']:
            valid.append({'sample_id':key,'observed':data['labels'][key]['value'],
                'predicted':predict(model,data['rows'][key]['features'])[0],'split':'VALID'})
        artifacts['validation_predictions']=_csv(root,output/'validation_predictions.csv',
            ['sample_id','observed','predicted','split'],valid)
        metadata={'status':'success',**facts,'features_used':data['features'],'target':data['target'],
            'target_unit':data['unit'],'conditions':data['conditions'],'n_samples':len(data['labels']),
            'n_source_rows':len(data['rows']),'data_sha256':data['source_facts']['features']['sha256'],
            'data_csv':data.get('source_declaration',{}).get('original_csv',{}).get('path')or data['source_facts']['features']['path'],
            'model_path':artifacts['model']['path'],'model_sha256':artifacts['model']['sha256'],
            'model_format':model['format'],'dataset_version':version['sha256'],
            'label_source_kind':data['label_source_kind'],'labels_independently_verified':False,
            'synthetic_data_used':data['synthetic_fixture'],'input_labels_declared_synthetic':data['synthetic_fixture'],
            'test_metrics_status':'not_evaluated_or_used_for_training_acquisition',
            'source_declaration':data.get('source_declaration')or{},'scientific_acceptance':False}
        artifacts['metadata']=immutable_json(root,output/'model_metadata.json',metadata)
    elif action in{'predict','select'}:
        raw,mfact=read_owned(root,request['model']['path'],limit=32*1024*1024)
        if mfact['sha256']!=request['model']['sha256']:raise ValueError('native ML model bytes changed')
        model=json.loads(raw);trained_ref=request.get('model_training_dataset')or version
        if(model.get('dataset_version')!=trained_ref['sha256']or model.get('features')!=data['features']
                or model.get('target')!=data['target']or model.get('unit')!=data['unit']):
            raise ValueError('model/dataset contract mismatch')
        if action=='select'and model.get('training_signature')not in(None,training_signature(data)):
            raise ValueError('current TRAIN labels/descriptors require retraining before acquisition')
        candidates,failures=_candidates(root,request,data);rows=[]
        for candidate in candidates:
            value,spread=predict(model,candidate['features'])
            rows.append({key:candidate.get(key)for key in('sample_id','structure_sha256','cif','cif_path')})
            rows[-1].update(predicted=value,ensemble_spread=spread,unit=data['unit'])
        ranked=rows
        if action=='select':
            count=request['n_select'];strategy=request.get('strategy','uncertainty')
            if type(count)is not int or not 1<=count<=len(request['candidate_ids']):raise ValueError('positive bounded n_select required')
            if strategy not in{'uncertainty','max_prediction'}:raise ValueError('supported native acquisition required')
            if strategy=='uncertainty'and model['format']!='native-rf-json-v1':
                raise ValueError('uncertainty selection requires an RF ensemble; no uncertainty is fabricated')
            ranked=sorted(rows,key=lambda row:(-row['ensemble_spread']if strategy=='uncertainty'else-row['predicted'],row['sample_id']))[:count]
        facts={'selected_ids':[row['sample_id']for row in ranked]if action=='select'else[],
            'n_candidates':len(candidates)+len(failures),'n_predicted':len(rows),'n_failures':len(failures),
            'requested_n_select':request.get('n_select')if action=='select'else None,
            'actual_n_selected':len(ranked)if action=='select'else None,
            'failures':failures,'test_labels_used':False,'synthetic_fixture':data['synthetic_fixture'],
            'uncertainty_interpretation':'ensemble_spread_proxy_not_calibrated'if model['format']=='native-rf-json-v1'else'not_available_for_this_model',
            'model_training_dataset_version':trained_ref['sha256']}
        if action=='select'and request.get('candidate_spec'):facts['candidate_rows']=candidates
        artifacts['selection'if action=='select'else'predictions']=immutable_json(root,output/(action+'.json'),
            {'rows':ranked,'facts':facts,'dataset_version':version['sha256']})
        csv_path=Path(request.get('output_csv')or output/('selection.csv'if action=='select'else'predictions.csv'))
        artifacts['csv']=_csv(root,csv_path,['sample_id','structure_sha256','cif','cif_path','predicted','ensemble_spread','unit'],ranked)
    else:raise ValueError('unsupported ML worker action')
    read_version(root,version['path'],version['sha256'])
    if read_owned(root,path)[1]['sha256']!=expected_sha:raise ValueError('ML request changed while computing')
    if action in{'predict','select'}and read_owned(root,request['model']['path'],limit=32*1024*1024)[1]['sha256']!=request['model']['sha256']:
        raise ValueError('ML model changed while computing')
    for key,source in implementations.items():
        if request[key]!=hashlib.sha256(source.read_bytes()).hexdigest():raise ValueError('ML implementation changed while computing')
    result={'format':'native-ml-result-v1','status':'completed','action':action,'owner':owner(root),'job_id':job,
        'attempt_id':request['attempt_id'],'request_sha256':expected_sha,'dataset_version':version['sha256'],
        'artifacts':artifacts,'facts':facts,'scientific_acceptance':False}
    return immutable_json(root,output/'worker_result.json',result)


if __name__=='__main__':
    import sys
    try:result=run_request(Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]);print(json.dumps(result))
    except Exception as error:
        print(json.dumps({'status':'failed','error_kind':'native_ml_worker_failure','error':str(error)[:2000]}),flush=True)
        raise
