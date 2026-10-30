"""Immutable source-bound ML data versions. Source declarations are not truth proofs."""
import copy,csv,hashlib,io,json,math,os,re,errno,time,uuid
from pathlib import Path
from .report_recovery import _OwnedFiles
from .workload_context import _read_regular

KINDS={'external_measurement','native_calculation','synthetic_fixture','user_supplied_unverified'}

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def owner(root):return {'username':Path(root).parent.name,'conv_id':Path(root).name}

def _read_owned_once(root,path,limit):
    files=_OwnedFiles(root,owner(root))
    try:
        p=files.owned(path);_,fact=files.read(p)
        if fact['size']>limit:raise ValueError('bounded ML data limit exceeded')
        raw,sha=_read_regular(p,limit=limit)
        if sha!=fact['sha256']:raise ValueError('ML source changed during observation')
        return raw,{k:fact[k]for k in ('path','sha256','size')}
    finally:files.close()

def read_owned(root,path,*,limit=8*1024*1024):
    # Shared filesystems may briefly return EAGAIN after an allocation publishes
    # a completed result. Retry the exact readonly proof, never its computation.
    for attempt in range(4):
        try:return _read_owned_once(root,path,limit)
        except OSError as error:
            if error.errno!=errno.EAGAIN or attempt==3:raise
            time.sleep(.025*(2**attempt))


def immutable_json(root,path,value):
    return immutable_bytes(root,path,canonical(value),reuse_identical=True)


def immutable_bytes(root,path,raw,*,reuse_identical=False):
    """Publish through owned directory descriptors; never follow output links."""
    files=_OwnedFiles(root,owner(root));path=files.owned(path)
    parent=os.dup(files.fd);temporary='.ml-stage-'+uuid.uuid4().hex;created=False;published=False
    try:
        parts=path.relative_to(files.root).parts
        for part in parts[:-1]:
            try:os.mkdir(part,0o700,dir_fd=parent)
            except FileExistsError:pass
            child=os.open(part,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW,dir_fd=parent)
            os.close(parent);parent=child
        fd=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=parent);created=True
        with os.fdopen(fd,'wb')as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
        try:os.link(temporary,parts[-1],src_dir_fd=parent,dst_dir_fd=parent,follow_symlinks=False)
        except FileExistsError:
            if not reuse_identical:raise ValueError('ML output already exists; choose a fresh output path')from None
            old,_=read_owned(root,path,limit=64*1024*1024)
            if old!=raw:raise ValueError('immutable ML output already exists with different bytes')
        os.fsync(parent);published=True
        return {'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}
    finally:
        if created and published:os.unlink(temporary,dir_fd=parent)
        os.close(parent);files.close()


def table(root,path):
    raw,fact=read_owned(root,path)
    reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
    fields=reader.fieldnames or []
    if not fields or len(fields)!=len(set(fields)):raise ValueError('unique CSV headers required')
    rows=list(reader)
    if not rows or len(rows)>100000:raise ValueError('bounded nonempty ML table required')
    if any(None in row for row in rows):raise ValueError('CSV row shape mismatch')
    return rows,fields,fact

def number(value):
    try:n=float(value)
    except (ValueError,TypeError):raise ValueError('finite numeric ML value required')from None
    if not math.isfinite(n):raise ValueError('finite numeric ML value required')
    return n

def _labels(root,path,target,allowed,source_kind):
    rows,fields,fact=table(root,path)
    if 'sample_id'not in fields or target not in fields:raise ValueError('label sample_id and target columns required')
    result={}
    for index,row in enumerate(rows,2):
        key=row['sample_id']
        if key not in allowed or key in result:raise ValueError('unowned or duplicate labelled sample')
        result[key]={'value':number(row[target]),'source':{**fact,'row':index,'column':target,'declared_kind':source_kind,'independently_verified':False}}
    return result,fact

def _publish(root,data):
    version=digest(data)
    return immutable_json(root,Path(root)/'ml_versions'/(version+'.json'),data)

def freeze_dataset(root,feature_csv,label_csv,schema):
    root=Path(root).absolute();schema=copy.deepcopy(schema)
    kind=schema.get('label_source_kind')
    if kind not in KINDS:raise ValueError('an explicit external/native/fixture label source declaration is required')
    if not label_csv:raise ValueError('real external labels are missing; no labels are generated')
    features=schema.get('features');target=schema.get('target');unit=schema.get('unit');split=schema.get('split')
    if(not isinstance(features,list)or not features or len(features)!=len(set(features))or len(features)>256
        or any(not isinstance(f,str)or not f for f in features)or not isinstance(target,str)or not target
        or target in features or any(f in {'sample_id','structure_sha256'}for f in features)
        or not isinstance(unit,str)or not unit or not isinstance(schema.get('conditions'),dict)):
        raise ValueError('explicit descriptors excluding target, target/unit and conditions required')
    if not isinstance(split,dict)or set(split)!={'train','valid','test'}:raise ValueError('fixed TRAIN/VALID/TEST split required')
    ids=[]
    for name in ('train','valid','test'):
        items=split[name]
        if(not isinstance(items,list)or not items or any(not isinstance(x,str)or not x for x in items)
            or len(items)!=len(set(items))):raise ValueError('nonempty unique split IDs required')
        ids.extend(items)
    if len(ids)!=len(set(ids)):raise ValueError('TRAIN/VALID/TEST IDs overlap')
    rows,fields,feature_fact=table(root,feature_csv)
    if not {'sample_id','structure_sha256',*features}.issubset(fields):raise ValueError('descriptor columns are absent')
    data={};structures=set()
    for row in rows:
        key=row['sample_id'];structure=row['structure_sha256']
        if key in data or not re.fullmatch('[0-9a-f]{64}',structure)or structure in structures:
            raise ValueError('duplicate/aliased structure or invalid structure identity')
        structures.add(structure)
        data[key]={'structure_sha256':structure,'features':[None if row[f]==''else number(row[f])for f in features]}
    if set(data)!=set(ids):raise ValueError('descriptor IDs differ from fixed split membership')
    labels,label_fact=_labels(root,label_csv,target,set(ids),kind)
    if not(set(labels)&set(split['train'])):raise ValueError('no revealed TRAIN labels; labels are never guessed')
    return _publish(root,{'format':'native-ml-dataset-v1','features':features,'target':target,'unit':unit,
        'conditions':schema['conditions'],'split':split,'rows':data,'labels':labels,'label_source_kind':kind,
        'labels_independently_verified':False,'source_facts':{'features':feature_fact,'labels':label_fact},
        'source_declaration':schema.get('source_declaration')or{},'test_labels_used_for_selection':False,'synthetic_fixture':kind=='synthetic_fixture'})

def read_version(root,path,expected_sha):
    raw,fact=read_owned(root,path,limit=64*1024*1024)
    if fact['sha256']!=expected_sha:raise ValueError('immutable ML dataset bytes changed')
    data=json.loads(raw)
    if not isinstance(data,dict)or data.get('format')!='native-ml-dataset-v1':raise ValueError('unsupported ML dataset version')
    return data


def training_signature(data):
    """Only revealed TRAIN content affects a model's reuse suitability."""
    return digest({'features':data['features'],'target':data['target'],'unit':data['unit'],
        'conditions':data['conditions'],'train':{key:{'features':data['rows'][key]['features'],
            'label':data['labels'][key]['value']}for key in data['split']['train']if key in data['labels']}})


def extend_candidates(root,version,rows,source):
    """Adopt verified worker descriptors as unlabelled TRAIN pool entries."""
    data=read_version(root,version['path'],version['sha256']);changed=False
    structures={value['structure_sha256']:key for key,value in data['rows'].items()}
    for row in rows:
        key=row['sample_id'];identity=row['structure_sha256'];values=row['features']
        if(not isinstance(key,str)or not key or not re.fullmatch('[a-f0-9]{64}',str(identity))
                or len(values)!=len(data['features'])):raise ValueError('invalid native candidate descriptors')
        values=[None if value is None else number(value)for value in values]
        if key in data['rows']:
            if data['rows'][key]['structure_sha256']!=identity or data['rows'][key]['features']!=values:
                raise ValueError('candidate identity differs from its immutable descriptors')
            continue
        if identity in structures:raise ValueError('candidate aliases an already registered structure')
        data['rows'][key]={'structure_sha256':identity,'features':values,'candidate_source':row.get('source')or{}}
        data['split']['train'].append(key);structures[identity]=key;changed=True
    if not changed:return version
    data['parent_version']=version['sha256'];data.setdefault('candidate_batches',[]).append(source)
    return _publish(root,data)

def append_external_labels(root,version,label_csv,*,selected_ids,source_kind):
    data=read_version(root,version['path'],version['sha256'])
    if source_kind not in KINDS or source_kind!=data['label_source_kind']:raise ValueError('label source kind changed; review the new contract first')
    selected=set(selected_ids)
    if(len(selected)!=len(selected_ids)or not selected or not selected<=set(data['split']['train'])
        or selected&set(data['labels'])):raise ValueError('only selected unlabelled TRAIN samples can receive new labels')
    labels,fact=_labels(root,label_csv,data['target'],selected,source_kind)
    if set(labels)!=selected:raise ValueError('complete selected label set required; missing labels remain pending')
    data['labels'].update(labels);data['parent_version']=version['sha256']
    data.setdefault('label_batches',[]).append(fact)
    return _publish(root,data)
