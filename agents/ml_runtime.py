"""Native versioned ML loop controller; no training or model deserialization here."""
import copy,hashlib,json,os,shlex,sys,uuid
from pathlib import Path
from .ml_dataset import freeze_dataset,read_version,read_owned,immutable_json,append_external_labels,owner,training_signature,extend_candidates,digest
from .state_io import json_transaction
from .watch_context import get_context

TOOLS={'train':'ml_train','predict':'ml_predict','select':'ml_active_learning'}

class MLRuntime:
    """Constructed by native API bindings, never with model-provided control paths."""
    def __init__(self,root,control_path,*,submitter=None,watch=None,python=None):
        self.root=Path(root).absolute();self.owner=owner(self.root);self.control=Path(control_path).absolute()
        # Execution journals live in native state, beyond user task output paths.
        if self.control.is_relative_to(self.root):raise PermissionError('ML control journal must be outside tool-writable task scope')
        self.control.parent.mkdir(parents=True,exist_ok=True)
        if submitter is None:
            from .slurm import submit_job
            submitter=submit_job
        self.submitter=submitter;self.watch=watch;self.python=python or sys.executable
        self.key=self.owner['username']+'/'+self.owner['conv_id']

    def snapshot(self):
        return copy.deepcopy(self.all_loops().get(self.key,{}))

    def all_loops(self):
        if not self.control.exists():return{}
        data=json.loads(self.control.read_text())
        if not isinstance(data,dict):raise ValueError('native ML journal is corrupt')
        return {key:value for key,value in data.get('loops',{}).items()if value.get('owner')==self.owner}

    def scope_dataset(self,identity):
        self.key=self.owner['username']+'/'+self.owner['conv_id']+'/'+digest(identity)[:24]

    def initialize(self,feature_csv,label_csv,schema):
        version=freeze_dataset(self.root,feature_csv,label_csv,schema)
        with json_transaction(self.control)as data:
            loops=data.setdefault('loops',{});current=loops.get(self.key)
            if current and current['head']!=version:raise ValueError('ML loop already has an immutable dataset head; append verified source labels instead')
            loops.setdefault(self.key,{'owner':self.owner,'head':version,'jobs':{},'pending':{}})
        return version

    def _context(self,action):
        ctx=get_context();a=ctx.get('resource_allocation')or{}
        if any(ctx.get(k)!=self.owner[k]for k in self.owner):raise PermissionError('ML submit owner mismatch')
        if ctx.get('tool_name')!=TOOLS[action]or not ctx.get('attempt_id')or not ctx.get('recovery_key')or not ctx.get('recovery_path'):
            raise ValueError('ML submission requires the existing native recovery attempt binding')
        if(not a.get('resource_review_id')or type(a.get('cpus'))is not int or not 1<=a['cpus']<=256
            or type(a.get('memory_mb'))is not int or a['memory_mb']<64
            or a.get('nodelist')=='node06'):
            raise ValueError('ML computation requires a reviewed finite Slurm allocation excluding node06')
        from .placement_budget import project_policy,validate_memory_budget
        validate_memory_budget(project_policy(self.root.parents[2]),cpus=a['cpus'],memory_mb=a['memory_mb'],
            nodelist=a.get('nodelist',''),partition=a.get('partition',''))
        return ctx,a

    def submit(self,action,*,model_attempt=None,n_select=None,strategy='uncertainty',output_dir=None,model_type='RF',candidate_spec=None,output_csv=None):
        if action not in TOOLS:raise ValueError('unsupported native ML action')
        ctx,a=self._context(action);attempt=ctx['attempt_id']
        from . import ml_worker,ml_dataset,ml_models
        with json_transaction(self.control)as state:
            loop=state.get('loops',{}).get(self.key)
            if not loop:raise ValueError('source-bound ML labels/dataset are missing')
            previous=loop['jobs'].get(attempt)
            if previous:
                if previous['action']!=action:raise ValueError('native ML attempt cannot change its action')
                return copy.deepcopy(previous.get('submission')or{'submitted':False,'uncertain':True,'reason':'ML attempt submission is already reserved; reconcile its native scheduler receipt before retry'})
            version=copy.deepcopy(loop['head']);dataset=read_version(self.root,version['path'],version['sha256'])
            candidates=sorted(set(dataset['split']['train'])-set(dataset['labels'])-set(loop['pending']))
            model=None
            if action!='train':
                produced=loop['jobs'].get(model_attempt,{})
                if produced.get('status')!='verified'or produced.get('action')!='train':
                    raise ValueError('model must come from a verified native training job for the current immutable dataset')
                trained=read_version(self.root,produced['dataset']['path'],produced['dataset']['sha256'])
                if action=='select'and training_signature(trained)!=training_signature(dataset):
                    raise ValueError('TRAIN labels/descriptors changed; retrain before another acquisition')
                if action=='predict':version=copy.deepcopy(produced['dataset']);dataset=trained
                model=copy.deepcopy(produced['result']['artifacts']['model'])
                if candidate_spec is not None:candidates=[row['sample_id']for row in candidate_spec['rows']]
                if not candidates and candidate_spec is None:raise ValueError('no unlabelled unclaimed TRAIN candidates remain')
                if action=='select'and strategy=='uncertainty'and produced['result']['facts'].get('model_type')!='RF':
                    raise ValueError('uncertainty acquisition requires an RF ensemble; use max_prediction for GBR/Ridge')
            if action=='select'and(type(n_select)is not int or not 1<=n_select<=len(candidates)):
                raise ValueError('positive n_select within eligible candidate budget required')
            work=Path(output_dir)if output_dir else self.root/'ml_jobs'/uuid.uuid4().hex
            work=work if work.is_absolute()else self.root/work
            if not work.is_relative_to(self.root)or'..'in work.parts:raise PermissionError('ML job output escapes owner')
            work.parent.mkdir(parents=True,exist_ok=True)
            if work.parent.resolve()!=work.parent:raise PermissionError('linked ML job output parent')
            work.mkdir(exist_ok=True)
            if work.resolve()!=work:raise PermissionError('linked ML output directory')
            if any((work/name).exists()for name in ('request.json','worker_result.json','model.json','model_metadata.json')):
                raise ValueError('ML output contains prior results; choose a fresh output_dir')
            if output_csv:
                output_csv=Path(os.path.abspath(output_csv))
                if not output_csv.is_relative_to(self.root)or output_csv.exists():
                    raise ValueError('ML output_csv must be a new file in the current session')

            request={'format':'native-ml-request-v1','owner':self.owner,'action':action,'dataset':version,
                'attempt_id':attempt,'recovery_key':ctx['recovery_key'],'work_dir':str(work),'allocated_cpus':a['cpus'],
                'worker_sha256':hashlib.sha256(Path(ml_worker.__file__).read_bytes()).hexdigest(),
                'model_adapter_sha256':hashlib.sha256(Path(ml_models.__file__).read_bytes()).hexdigest(),
                'model_type':model_type,
                'dataset_adapter_sha256':hashlib.sha256(Path(ml_dataset.__file__).read_bytes()).hexdigest()}
            if model:request.update(model=model,candidate_ids=candidates,n_select=n_select,strategy=strategy,
                model_training_dataset=produced['dataset'])
            if candidate_spec is not None:
                request['candidate_spec']=candidate_spec
                if candidate_spec.get('format')=='cifs':
                    from . import ml_cif_features
                    request['cif_adapter_sha256']=hashlib.sha256(Path(ml_cif_features.__file__).read_bytes()).hexdigest()
            if output_csv:request['output_csv']=str(output_csv)
            frozen=immutable_json(self.root,work/'request.json',request)
            loop['jobs'][attempt]={'action':action,'status':'submitting','dataset':version,'request':frozen,'work_dir':str(work),
                'owner':self.owner,'attempt_id':attempt,'recovery_key':ctx['recovery_key'],'candidate_ids':candidates if action=='select'else[]}
            if action=='select':loop['pending'].update({key:attempt for key in candidates})
        package=Path(__file__).resolve().parents[1]
        command=("export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1\n"
            +'export PYTHONDONTWRITEBYTECODE=1 PYTHONPATH='+shlex.quote(str(package))+'\n'
            +shlex.quote(str(self.python))+' -B -m agents.ml_worker '+shlex.quote(str(self.root))+' '
            +shlex.quote(frozen['path'])+' '+shlex.quote(frozen['sha256']))
        # The common helper applies the exact reviewed allocation and registers
        # the scheduler job with the same RecoveryGate/JobWatch attempt.
        result=self.submitter(job_name='native_ml_'+action,command=command,work_dir=str(work),mode='local',
            cpus_per_task=a['cpus'],memory_mb=a['memory_mb'],partition=a.get('partition','compute'),nodelist=a.get('nodelist',''),
            ntasks=1,walltime='00:20:00',extra_directives=['--exclude=node06'])
        response={**result,'work_dir':str(work),'dataset_version':version['sha256'],'request':frozen,
            'expected_outputs':[str(work/'worker_result.json')],'scientific_acceptance':False}
        if result.get('submitted')and not(str(result.get('job_id','')).isdecimal()):
            response.update(submitted=False,uncertain=True,reason='scheduler outcome requires a native job identity before acceptance')
        with json_transaction(self.control)as state:
            record=state['loops'][self.key]['jobs'][attempt]
            record.update(submission=copy.deepcopy(response),status='submitted'if response.get('submitted')else'uncertain')
        return response

    def collect(self,attempt):
        observed=self.snapshot();record=observed.get('jobs',{}).get(attempt)
        if not record or record.get('status')not in {'submitted','verified'}:raise ValueError('no submitted native ML attempt to inspect')
        if self.watch is None:
            from .job_watch import get_watch
            watch=get_watch()
        else:watch=self.watch
        job_id=str(record['submission']['job_id']);job=watch.get(job_id)or{}
        if(any(job.get(k)!=v for k,v in self.owner.items())or job.get('attempt_id')!=attempt
            or job.get('recovery_key')!=record['recovery_key']or job.get('tool')!=TOOLS[record['action']]
            or Path(job.get('work_dir','')).absolute()!=Path(record['work_dir'])
            or job.get('terminal')is not True or job.get('failed')or job.get('state')!='COMPLETED'):
            raise ValueError('ML result requires the exact owned successful native scheduler/watch receipt')
        raw,fact=read_owned(self.root,Path(record['work_dir'])/'worker_result.json');result=json.loads(raw)
        expected={'format':'native-ml-result-v1','status':'completed','action':record['action'],'owner':self.owner,'job_id':job_id,
            'attempt_id':attempt,'request_sha256':record['request']['sha256'],'dataset_version':record['dataset']['sha256']}
        if any(result.get(k)!=v for k,v in expected.items()):raise ValueError('ML result identity differs from native request')
        request_raw,request_fact=read_owned(self.root,record['request']['path'])
        if request_fact['sha256']!=record['request']['sha256']:raise ValueError('ML submitted request bytes changed')
        request=json.loads(request_raw)
        read_version(self.root,record['dataset']['path'],record['dataset']['sha256'])
        for artifact in result.get('artifacts',{}).values():
            _,af=read_owned(self.root,artifact['path'],limit=32*1024*1024)
            owned_output=(Path(artifact['path']).parent==Path(record['work_dir'])or artifact['path']==request.get('output_csv'))
            if af['sha256']!=artifact['sha256']or not owned_output:raise ValueError('ML output bytes or namespace changed')
        if record['action']=='train'and 'model'not in result.get('artifacts',{}):raise ValueError('training model artifact is absent')
        selected=result.get('facts',{}).get('selected_ids',[])
        if record['action']=='select'and(not selected or len(selected)!=len(set(selected))or not set(selected)<=set(record['candidate_ids'])):
            raise ValueError('selection introduced unclaimed or duplicated samples')
        selection_version=record.get('selection_dataset')or record['dataset']
        if record['action']=='select'and request.get('candidate_spec')and record.get('status')!='verified':
            rows=result.get('facts',{}).get('candidate_rows')
            if not isinstance(rows,list)or not rows or not set(selected)<={row.get('sample_id')for row in rows}:
                raise ValueError('selection lacks native source descriptors for its selected candidates')
            selection_version=extend_candidates(self.root,record['dataset'],rows,
                {'job_id':job_id,'request_sha256':record['request']['sha256'],'worker_result_sha256':fact['sha256']})
        with json_transaction(self.control)as state:
            loop=state['loops'][self.key];current=loop['jobs'][attempt]
            accepted_head=record.get('selection_dataset')or record['dataset']
            if (record['action']=='select'and loop['head']!=accepted_head
                    or current['request']!=record['request']):raise ValueError('ML dataset head or attempt changed before result adoption')
            if current.get('status')=='verified'and current.get('result')!=result:raise ValueError('previously verified ML receipt changed')
            current.update(status='verified',result=copy.deepcopy(result),result_source=fact)
            if record['action']=='select':
                loop['head']=selection_version;current['selection_dataset']=selection_version
                for key in record['candidate_ids']:
                    if loop['pending'].get(key)==attempt and key not in selected:loop['pending'].pop(key)
        return {'status':'verified','attempt_id':attempt,'job_id':job_id,'result':result,'result_source':fact,'scientific_acceptance':False}

    def append_labels(self,selection_attempt,label_csv,*,expected_version,source_kind):
        snapshot=self.snapshot();selection=snapshot.get('jobs',{}).get(selection_attempt,{})
        if(snapshot.get('head',{}).get('sha256')!=expected_version or selection.get('status')!='verified'
            or selection.get('action')!='select'or (selection.get('selection_dataset')or selection.get('dataset'))!=snapshot['head']):
            raise ValueError('external label append requires a current verified native selection ticket')
        # Revalidate the saved immutable selection receipt, not model-supplied IDs.
        _,fact=read_owned(self.root,selection['result_source']['path'])
        if fact['sha256']!=selection['result_source']['sha256']:raise ValueError('verified selection receipt changed')
        ids=selection['result']['facts']['selected_ids']
        if any(snapshot['pending'].get(key)!=selection_attempt for key in ids):raise ValueError('candidate reservation changed')
        new=append_external_labels(self.root,snapshot['head'],label_csv,selected_ids=ids,source_kind=source_kind)
        with json_transaction(self.control)as state:
            loop=state['loops'][self.key]
            if loop['head']!=snapshot['head']or any(loop['pending'].get(key)!=selection_attempt for key in ids):raise ValueError('ML dataset head changed during external label append')
            loop['head']=new;loop['jobs'][selection_attempt]['labels_appended_to']=new
            for key in ids:loop['pending'].pop(key,None)
        return {'status':'labels_appended','dataset':new,'source_bound_not_independently_verified':True,'synthetic_fixture':source_kind=='synthetic_fixture'}
