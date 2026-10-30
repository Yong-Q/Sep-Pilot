"""Declared CSV/CIF inputs frozen once into the current owner's ML workspace."""
import csv,hashlib,io,json,math,os,re
from pathlib import Path
from .ml_dataset import immutable_bytes,read_owned,table,digest,owner

MISSING={'','nan','na','n/a','null','none','?'}


def numeric(value,*,missing=True):
    text=''if value is None else str(value).strip()
    if missing and text.lower()in MISSING:return None
    if text.lower()in{'true','false'}:return float(text.lower()=='true')
    try:result=float(text)
    except ValueError:raise ValueError('feature/label is not numeric')from None
    if not math.isfinite(result):
        if missing:return None
        raise ValueError('finite label value required')
    return result


def snapshot_source(root,value,config,*,kind):
    from .workspace import resolve_project_path,tool_scope_issues
    from .raspa_assets import _read
    path=resolve_project_path(value,config.project_root)
    key='cif_path'if kind=='cif'else'data_csv'
    issues=tool_scope_issues({key:str(path)},**owner(root),project_root=config.project_root)
    if issues:raise PermissionError('ML input is outside allowed session/shared dataset scope')
    suffix={ 'cif':'.cif','csv':'.csv','json':'.json'}[kind]
    if path.suffix.lower()!=suffix:raise ValueError('declared ML input must use '+suffix+' format')
    raw,fact=_read(path)
    if len(raw)>8*1024*1024:raise ValueError('ML source exceeds bounded input size')
    if kind=='csv':
        reader=csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))
        if not reader.fieldnames or len(set(reader.fieldnames))!=len(reader.fieldnames):raise ValueError('unique CSV columns required')
    elif kind=='json':
        content=json.loads(raw)
        if not isinstance(content,dict)or not {'feature_csv','label_csv','schema'}<=set(content):
            raise ValueError('ML dataset manifest must declare feature_csv, label_csv and schema')
    elif b'_cell_length_'not in raw.lower() or b'_atom_site_'not in raw.lower():
        raise ValueError('CIF input has no declared cell/atom-site data')
    destination=Path(root)/'ml_inputs'/'source_snapshots'/(fact['sha256']+suffix)
    immutable_bytes(root,destination,raw,reuse_identical=True)
    return destination,{**fact,'snapshot_path':str(destination)}


def candidate_spec(root,p,config,data,pending,*,selection):
    """Only declared features are copied. Candidate target labels are ignored."""
    sources=[];rows=[];kind=None
    csv_path=p.get('data_csv')or p.get('feature_csv')or p.get('candidate_csv')
    if csv_path and (p.get('cif_path')or p.get('cif_dir')):raise ValueError('choose one candidate CSV or CIF source')
    if p.get('cif_path')and p.get('cif_dir'):raise ValueError('choose cif_path or cif_dir, not both')
    if csv_path:
        source,fact=snapshot_source(root,csv_path,config,kind='csv');sources.append(fact)
        values,fields,_=table(root,source)
        if not set(data['features'])<=set(fields):raise ValueError('candidate CSV lacks exact trained feature columns')
        for index,row in enumerate(values):
            key=row.get('sample_id')or'candidate-'+digest({'source':fact['sha256'],'row':index})[:24]
            values=[numeric(row[column])for column in data['features']]
            structure=row.get('structure_sha256')or row.get('source_sha256')
            if not structure and key in data['rows']and values==data['rows'][key]['features']:
                structure=data['rows'][key]['structure_sha256']
            structure=structure or digest({'source':fact['sha256'],'sample_id':key})
            if not re.fullmatch('[a-f0-9]{64}',structure):raise ValueError('candidate structure identity is invalid')
            rows.append({'sample_id':key,'structure_sha256':structure,
                'features':values,
                'source':{'path':fact['path'],'sha256':fact['sha256'],'row':index+2}})
        kind='features'
    elif p.get('cif_path')or p.get('cif_dir'):
        from .workspace import resolve_project_path,tool_scope_issues
        source=resolve_project_path(p.get('cif_path')or p['cif_dir'],config.project_root)
        if tool_scope_issues({'cif_path':str(source)},**owner(root),project_root=config.project_root):
            raise PermissionError('candidate CIF scope is not readable')
        files=[source]if p.get('cif_path')else sorted(path for path in source.iterdir()if path.suffix.lower()=='.cif')
        if not 1<=len(files)<=4096:raise ValueError('bounded nonempty CIF input is required')
        for index,path in enumerate(files):
            staged,fact=snapshot_source(root,path,config,kind='cif');sources.append(fact)
            rows.append({'sample_id':'cif-'+fact['sha256'][:24]+('-'+str(index)if not selection else''),
                'structure_sha256':fact['sha256'],'cif':path.name,'cif_path':str(path),
                'snapshot_path':str(staged),'source':fact})
        kind='cifs'
    else:return None
    by_structure={row['structure_sha256']:key for key,row in data['rows'].items()}
    accepted=[];excluded=[];seen_ids=set();seen_structures=set()
    for row in rows:
        key=by_structure.get(row['structure_sha256'])
        if selection and key is not None:
            if key not in data['split']['train']or key in data['labels']:
                excluded.append({'sample_id':row['sample_id'],'reason':'known_labelled_or_heldout_structure'});continue
            row['sample_id']=key
        if selection and (row['sample_id']in pending or row['structure_sha256']in seen_structures):
            excluded.append({'sample_id':row['sample_id'],'reason':'pending_or_duplicate_structure'});continue
        if row['sample_id']in seen_ids:raise ValueError('candidate sample IDs are duplicated')
        if selection and row['sample_id']in data['rows']and data['rows'][row['sample_id']]['structure_sha256']!=row['structure_sha256']:
            raise ValueError('candidate sample ID conflicts with a different registered structure')
        seen_ids.add(row['sample_id']);seen_structures.add(row['structure_sha256']);accepted.append(row)
    if not accepted:raise ValueError('no eligible candidate inputs remain after known/heldout/pending exclusion')
    return {'format':kind,'rows':accepted,'source_facts':sources,'excluded':excluded,
        'features':data['features'],'source_kind':'caller_features_unverified'if kind=='features'else'native_cif_descriptor',
        'test_labels_used':False}
