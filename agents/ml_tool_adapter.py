"""Owned source snapshots, reviewed native Slurm jobs and numeric model metadata.

No backend model fitting or arbitrary object/joblib deserialization is allowed.
"""
import csv,io,json,random
from pathlib import Path
from .ml_runtime import MLRuntime
from .ml_dataset import read_owned,table,digest,read_version,immutable_bytes,owner
from .ml_inputs import snapshot_source,candidate_spec,numeric
from .watch_context import get_context
from .state_io import json_transaction


def _new_csv(root,path,headers,rows):
    stream=io.StringIO();w=csv.writer(stream,lineterminator='\n');w.writerow(headers);w.writerows(rows)
    return immutable_bytes(root,path,stream.getvalue().encode(),reuse_identical=True)


def _legacy_dataset(root,p,config):
    staged,fact=snapshot_source(root,p['data_csv'],config,kind='csv')
    rows,fields,_=table(root,staged);target=p.get('target','uptake')
    if target not in fields:raise ValueError('CSV target column is missing; labels are not generated')
    features=p.get('features')
    if features is None:
        features=[]
        for field in fields:
            if field in{target,'sample_id','structure_sha256','unit','units'}:continue
            try:
                for row in rows:numeric(row[field])
            except ValueError:continue
            features.append(field)
    if(not isinstance(features,list)or not features or target in features or len(features)!=len(set(features))
        or not set(features)<=set(fields)):raise ValueError('explicit unique non-target numeric feature columns required')
    samples=[];ids=set();structures=set()
    for index,row in enumerate(rows):
        sample=row.get('sample_id')or'row-'+str(index)
        if sample in ids:raise ValueError('duplicate source sample IDs')
        x=[numeric(row[f])for f in features]
        y=numeric(row[target])
        # Descriptor equality does not imply two structures are identical.
        identity=row.get('structure_sha256')or digest({'source_sha256':fact['sha256'],'sample_id':sample})
        if identity in structures:raise ValueError('duplicate structure identity requires an explicit replicate policy')
        samples.append((sample,identity,x,y,index+2));ids.add(sample);structures.add(identity)
    revealed=[s[0]for s in samples if s[3]is not None]
    if len(revealed)<8:raise ValueError('at least eight source-labelled CSV rows are required')
    random.Random(42).shuffle(revealed);hold=max(2,len(revealed)//5)
    split=p.get('split')or{'test':revealed[:hold],'valid':revealed[hold:2*hold],
        'train':revealed[2*hold:]+[s[0]for s in samples if s[3]is None]}
    kind='synthetic_fixture'if p.get('synthetic_fixture')is True else'user_supplied_unverified'
    schema={'features':features,'target':target,'unit':p.get('target_unit')or'unspecified',
        'conditions':p.get('conditions')or{'scope':'user-supplied CSV; experimental method unverified'},'split':split,
        'label_source_kind':kind,'source_declaration':{'original_csv':fact,
            'label_source':p.get('label_source')or'not supplied; input label provenance is unverified',
            'source_rows':{s[0]:s[4]for s in samples},'descriptor_provenance':'explicit CSV columns; scientific suitability unverified'}}
    directory=root/'ml_inputs'/digest({'source':fact['sha256'],'schema':schema})
    feature=directory/'features.csv';label=directory/'labels.csv'
    _new_csv(root,feature,['sample_id','structure_sha256',*features],[[sid,identity,*[''if v is None else v for v in x]]for sid,identity,x,y,index in samples])
    _new_csv(root,label,['sample_id',target],[[sid,y]for sid,identity,x,y,index in samples if y is not None])
    return feature,label,schema,fact


def _runtime(config):
    from .workspace import validate_scope_component
    ctx=get_context();user=ctx.get('username');conv=ctx.get('conv_id');validate_scope_component(user);validate_scope_component(conv)
    root=Path(config.project_root)/'runs'/user/conv;root.mkdir(parents=True,exist_ok=True)
    return MLRuntime(root,Path(config.project_root)/'data/state/native_ml_loops.json')


def _find(rt,p,action=None):
    matches=[]
    for key,loop in rt.all_loops().items():
        for attempt,record in loop.get('jobs',{}).items():
            if action is not None and record['action']!=action:continue
            match=(str(record.get('submission',{}).get('job_id'))==str(p['job_id'])if p.get('job_id')else
                Path(record['work_dir'])==Path(p.get('model_dir')or p.get('work_dir')or'/missing').absolute())
            if match:matches.append((key,attempt))
    if len(matches)!=1:raise ValueError('exact owned native ML job/model reference required; legacy joblib needs CSV retraining')
    rt.key,attempt=matches[0];return attempt


def _inspect(rt,attempt):
    record=rt.snapshot()['jobs'][attempt]
    if rt.watch is None:
        from .job_watch import get_watch
        watch=get_watch()
    else:watch=rt.watch
    job=watch.get(str(record.get('submission',{}).get('job_id')))or{}
    # A waiting observation is never an accepted model or a resubmission hint.
    if job.get('terminal')is not True:
        if any(job.get(k)!=v for k,v in rt.owner.items())or job.get('attempt_id')!=attempt:
            raise ValueError('owned native scheduler/watch identity is missing')
        return {'status':'waiting','ready':False,'job_id':str(job.get('job_id')),
            'scheduler_state':job.get('state','UNKNOWN'),'attempt_id':attempt,
            'retry_authorized':False,'scientific_acceptance':False}
    return rt.collect(attempt)


def _initialize(rt,p,config):
    requested=p.get('dataset_manifest')or p.get('data_csv')
    if not requested and not rt.snapshot():raise ValueError('source CSV labels or dataset manifest required')
    staged=fact=None
    if requested:staged,fact=snapshot_source(rt.root,requested,config,kind='json'if p.get('dataset_manifest')else'csv')
    snapshot=rt.snapshot()
    if snapshot:
        data=read_version(rt.root,snapshot['head']['path'],snapshot['head']['sha256'])
        original=snapshot.get('initial_source')or data['source_facts']['features']
        same=fact is None or all(fact[k]==original[k]for k in('path','sha256','size'))
        contract=all(key not in p or p[key]==actual for key,actual in(
            ('target',data['target']),('features',data['features']),('target_unit',data['unit']),('conditions',data['conditions'])))
        if same and contract:return
        if 'initial_source'not in snapshot:raise ValueError('requested ML contract differs from the immutable dataset head')
    if fact is None:raise ValueError('source-bound ML data are missing')
    rt.scope_dataset({'source':fact['sha256'],'target':p.get('target','uptake'),'features':p.get('features'),
        'unit':p.get('target_unit'),'conditions':p.get('conditions'),'split':p.get('split'),
        'synthetic_fixture':p.get('synthetic_fixture',False)})
    if rt.snapshot():return
    if p.get('dataset_manifest'):
        manifest=json.loads(staged.read_bytes());parent=Path(requested).parent
        def stage_input(value):
            path=Path(value);return snapshot_source(rt.root,path if path.is_absolute()else parent/path,config,kind='csv')[0]
        feature=stage_input(manifest['feature_csv']);labels=stage_input(manifest['label_csv']);schema=manifest['schema']
    else:feature,labels,schema,fact=_legacy_dataset(rt.root,p,config)
    rt.initialize(feature,labels,schema)
    with json_transaction(rt.control)as state:state['loops'][rt.key]['initial_source']=fact


def _features(rt,p,config):
    from .ml_cif_features import extract_features
    from dataclasses import asdict
    from .workspace import tool_scope_issues
    source=p.get('cif_path')or p.get('cif_dir')
    if not source or bool(p.get('cif_path'))==bool(p.get('cif_dir')):raise ValueError('exactly one cif_path or cif_dir required')
    if tool_scope_issues({'cif_path':source},**rt.owner,project_root=config.project_root):raise PermissionError('CIF input is outside allowed session/shared scope')
    if type(p.get('n_threads',4))is not int or not 1<=p.get('n_threads',4)<=32:raise ValueError('n_threads must be in [1,32]')
    paths=[Path(source)]if p.get('cif_path')else sorted(Path(source).glob('*.cif'))
    if not 1<=len(paths)<=128:raise ValueError('lightweight descriptor request supports 1–128 CIF files')
    rows=[];failures=[];full=[]
    for path in paths:
        staged,fact=snapshot_source(rt.root,path,config,kind='cif');f=extract_features(staged,owned_root=rt.root,expected_sha256=fact['sha256'])
        if f.error:failures.append({'cif':path.name,'error':f.error,'error_kind':f.error_code,'source_sha256':fact['sha256']});continue
        full.append({**asdict(f),'input_path':str(path),'input_sha256':fact['sha256']})
        row=f.to_row(include_provenance=True);row['cif']=path.name;row['source_path']=str(path);rows.append(row)
    if not rows:raise ValueError('no valid CIF descriptors: '+str(failures[:4]))
    if p.get('cif_path')and not p.get('output_csv'):return {**full[0],'status':'success','scientific_acceptance':False}
    output=Path(p.get('output_csv')or rt.root/'ml_features'/('features-'+digest([r['source_sha256']for r in rows])+'.csv'))
    headers=list(rows[0]);artifact=_new_csv(rt.root,output,headers,[[r.get(k,'')for k in headers]for r in rows])
    return {'status':'completed','output_csv':str(output),'output_files':[str(output)],'artifact':artifact,
        'features':full,'n_inputs':len(paths),'n_features':len(rows),'n_success':len(rows),'n_failures':len(failures),'failures':failures,'scientific_acceptance':False}


def execute_ml(action,p,config):
    try:
        p=dict(p)
        if p.get('manifest_path'):
            if p.get('dataset_manifest'):raise ValueError('choose one dataset manifest alias')
            p['dataset_manifest']=p.pop('manifest_path')
        from .workspace import resolve_project_path
        for name in('data_csv','feature_csv','candidate_csv','dataset_manifest','output_dir','output_csv','model_dir','labels_csv','work_dir','cif_dir','cif_path'):
            if p.get(name):p[name]=str(resolve_project_path(p[name],config.project_root))
        rt=_runtime(config)
        if action=='train':
            model_type=str(p.get('model_type','GBR')).upper()
            if model_type not in{'GBR','RF','RIDGE'}:raise ValueError('model_type must be GBR, RF, or Ridge')
            _initialize(rt,p,config)
            return rt.submit('train',output_dir=p.get('output_dir'),model_type=model_type)
        if action in{'select','predict','importance'}:
            if sum(bool(p.get(k))for k in('data_csv','feature_csv','candidate_csv','cif_path','cif_dir'))>1:raise ValueError('choose exactly one candidate CSV or CIF input')
            attempt=_find(rt,p,'train');observed=_inspect(rt,attempt)
            if observed.get('status')!='verified':return observed
            produced=rt.snapshot()['jobs'][attempt]
            if action=='importance':
                artifact=produced['result']['artifacts']['metadata'];raw,fact=read_owned(rt.root,artifact['path'])
                if fact['sha256']!=artifact['sha256']:raise ValueError('verified model metadata changed')
                metadata=json.loads(raw)
                return {'status':'success','model_type':metadata['model_type'],'feature_importance':metadata['feature_importance'],
                    'feature_importance_interpretation':metadata['feature_importance_interpretation'],'metadata':artifact,
                    'read_only':True,'scientific_acceptance':False}
            ref=produced['dataset']if action=='predict'else rt.snapshot()['head']
            data=read_version(rt.root,ref['path'],ref['sha256'])
            spec=candidate_spec(rt.root,p,config,data,rt.snapshot().get('pending',{}),selection=action=='select')
            if action=='predict'and spec is None:raise ValueError('prediction requires explicit CIF or feature CSV input')
            return rt.submit(action,model_attempt=attempt,n_select=p.get('n_select',5),strategy=p.get('strategy','uncertainty'),
                output_dir=p.get('output_dir'),output_csv=p.get('output_csv'),candidate_spec=spec)
        if action=='inspect':return _inspect(rt,_find(rt,p))
        if action=='append':
            attempt=_find(rt,{'job_id':p.get('selection_job_id'),'work_dir':p.get('work_dir')},'select')
            observed=_inspect(rt,attempt)
            if observed.get('status')!='verified':return observed
            snapshot=rt.snapshot();data=read_version(rt.root,snapshot['head']['path'],snapshot['head']['sha256'])
            labels=p.get('labels_csv')or p.get('data_csv')
            if not labels:raise ValueError('externally supplied label CSV is required')
            staged,_=snapshot_source(rt.root,labels,config,kind='csv')
            return rt.append_labels(attempt,staged,expected_version=p.get('expected_dataset_version')or snapshot['head']['sha256'],
                source_kind=p.get('label_source_kind')or data['label_source_kind'])
        if action=='features':return _features(rt,p,config)
        raise ValueError('unsupported native ML tool action')
    except (OSError,ValueError,KeyError,TypeError,PermissionError)as error:
        return {'error':str(error),'error_kind':'native_ml_contract','executed':False,
            'scientific_acceptance':False,'labels_generated':False}
