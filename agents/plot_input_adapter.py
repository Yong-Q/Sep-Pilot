"""Read-only raw CSV/JSON/text descriptors for the existing OS plot sandbox.

This module does not run model code, create numeric CSVs, infer measurements or
normalize source values. The original bytes are mounted read-only for Python.
"""
import hashlib,json,math,re
from pathlib import Path

FORMATS={'csv':'.csv','json':'.json','text':'.log','png':'.png'}
TEXT_SUFFIXES={'.log','.out','.txt','.dat'}
ID=re.compile(r'[A-Za-z][A-Za-z0-9_-]{0,31}\Z')


def _authenticated_raw_inputs(packet,document):
    """Reveal bounded original sources only from a verified native envelope."""
    candidates=document.get('raw_input_sources') or []
    if document.get('metadata_schema')=='native_gcmc_report_view_v1':
        from .report_authoring import _read,_relative
        from .recovery import result_object
        raw,fact=_read(packet['root'],document['path'],{'bytes':0})
        value=json.loads(raw)
        if (fact['sha256']!=document.get('sha256') or value.get('schema')!=document['metadata_schema']
                or value.get('public')!=document.get('untrusted_content')
                or (value.get('binding') or {}).get('owner')!=packet.get('owner')):
            raise ValueError('native source envelope changed before sandbox catalogue')
        binding=value.get('binding') or {};call_id=binding.get('producer_call_id')
        if not isinstance(call_id,str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}',call_id):
            raise ValueError('native source producer identity is invalid')
        call_raw,call_fact=_read(packet['root'],Path(packet['root'])/'evidence'/(call_id+'.json'),{'bytes':0})
        call=json.loads(call_raw);producer=result_object(call.get('result'))
        from .gcmc_results import TOOLS
        if (call_fact['sha256']!=binding.get('producer_sha256') or call.get('call_id')!=call_id
                or call.get('failed') is not False or call.get('tool') not in TOOLS):
            raise ValueError('native source producer evidence changed')
        work=_relative(packet['root'],producer.get('work_dir',''))
        candidates=[{'path':x.get('path'),'sha256':x.get('sha256'),'bytes':x.get('size'),'format':'text'}
            for x in binding.get('original_sources',[])
            if isinstance(x,dict) and Path(str(x.get('path',''))).suffix.lower()=='.data'
            and Path(str(x.get('path',''))).is_relative_to(work)]
    return candidates


def preview(raw,limit=6000):
    text=raw.decode('utf-8-sig')
    if '\x00'in text:raise ValueError('plot text input contains NUL bytes')
    if len(text)<=limit:return {'text':text,'truncated':False}
    half=limit//2
    return {'text':text[:half]+'\n[bounded preview; original file remains complete]\n'+text[-half:],
        'truncated':True,'original_characters':len(text)}


def text_format_hints(raw):
    """Index syntax/section markers only; never synthesize numerical tables."""
    headers=[];stages=[];styles=[]
    for number,line in enumerate(raw.decode('utf-8-sig').splitlines(),1):
        tokens=line.split()
        if tokens and tokens[0]in{'Step','Temp'}and 2<=len(tokens)<=64 and all(re.fullmatch(r'[A-Za-z_][A-Za-z0-9_\[\]()./-]*',token)for token in tokens):
            if len(headers)<32:headers.append({'line':number,'text':line[:500]})
        if re.fullmatch(r'\s*SEPPILOT_STAGE\s+\S+\s+(?:begin|end)\s*',line):
            if len(stages)<32:stages.append({'line':number,'text':line[:500]})
        if re.match(r'\s*Unit style\s*:',line)and len(styles)<16:styles.append({'line':number,'text':line[:500]})
    return {'column_header_lines':headers,'native_stage_marker_lines':stages,'unit_style_lines':styles,
        'scope':'Line-indexed native syntax only. Read all values from the original sandbox input; markers are not convergence evidence.'}


def describe(raw,descriptor,index):
    fmt=descriptor.get('format')
    suffix=Path(descriptor['path']).suffix.lower()
    if fmt is None:fmt='json'if suffix=='.json'else'csv'if suffix=='.csv'else'png'if suffix=='.png'else'text'if suffix in TEXT_SUFFIXES else None
    if fmt not in FORMATS:raise ValueError('sandbox input must be CSV, JSON, PNG or controlled UTF-8 text')
    name=descriptor.get('input_id')or'input'+str(index)
    if not isinstance(name,str)or not ID.fullmatch(name)or name.lower()in{'audit','guidance_index'}or name.lower().startswith('guidance'):
        raise ValueError('plot input ID must be a unique safe literal name')
    filename=name+FORMATS[fmt]
    units=descriptor.get('units')or{}
    if not isinstance(units,dict)or any(not isinstance(k,str)or not isinstance(v,str)or not 0<len(v)<=80 for k,v in units.items()):
        raise ValueError('plot input units must use bounded source-declared string labels')
    if fmt=='png':
        import io
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as image:
            if image.format!='PNG':raise ValueError('image input must be PNG')
            image.verify()
        input_preview={'kind':'image','format':'PNG'}
    else:input_preview=preview(raw)
    facts={'input_id':name,'format':fmt,'source_path':descriptor['path'],
        'sandbox_path':'/inputs/'+filename,'sha256':hashlib.sha256(raw).hexdigest(),
        'bytes':len(raw),'preview':input_preview,'scientific_verification':False,
        'units':units,'method_column':descriptor.get('method_column')}
    if fmt=='json':
        value=json.loads(raw)
        facts['json_root_type']=type(value).__name__
        if isinstance(value,dict):facts['json_top_level_keys']=list(value)[:128]
    elif fmt=='text':
        text=raw.decode('utf-8-sig')
        facts['text_lines']=len(text.splitlines())
        styles=sorted(set(re.findall(r'^\s*Unit style\s*:\s*(\w+)',text,re.M)))
        if styles:facts['native_lammps_unit_styles']=styles
        facts['native_lammps_log']='LAMMPS'in text and bool(re.search(r'^\s*(?:Step|Temp)\s',text,re.M))
        facts['format_hints']=text_format_hints(raw)
        try:
            native=native_thermo_input(raw,name)
            if native:facts['native_structure']=native[2]
        except ValueError as error:facts['native_structure_gap']=str(error)
    return filename,facts


def catalogue(packet):
    """All choices come from already owned/native-attested packet artifacts."""
    result=[];seen=set()
    for table in packet.get('tables',[]):
        # Native JSON/text tables are report views, not CSV byte streams.
        # Their complete typed original input is offered below instead.
        if Path(table['path']).suffix.lower()!='.csv':continue
        result.append({'id':table['id'],'format':'csv','path':table['path'],'sha256':table['sha256'],
            'source_id':table['source_id'],'columns':table['columns'],
            'sandbox_path':'/inputs/'+table['id']+'.csv'})
        seen.add(table['path'])
    for source in packet.get('sources',[]):
        for fmt,key in(('json','structured_data'),('text','text_documents')):
            for document in source.get(key,[]):
                if document.get('role')=='provenance':continue
                if document['path']in seen:continue
                name=document.get('id')or('L'+str(1+sum(item['format']=='text'for item in result)))
                entry={'id':name,'format':fmt,'path':document['path'],'sha256':document['sha256'],
                    'source_id':source['id'],'sandbox_path':'/inputs/'+name+FORMATS[fmt]}
                if 'preview'in document:entry['preview']=document['preview']
                elif fmt=='json':
                    value=document['untrusted_content']
                    if document.get('json_root_pointer')=='/public':
                        # The mounted file remains its original envelope. Show
                        # the real public subtree location, not a flattened
                        # preview that would make sandbox code use wrong keys.
                        value={'schema':document.get('metadata_schema'),'role':document.get('role'),'public':value}
                        entry['json_root_pointer']='/public'
                    entry['preview']=preview(json.dumps(value,ensure_ascii=False,default=str).encode())
                if document.get('format_hints'):entry['format_hints']=document['format_hints']
                if document.get('native_string'):entry['native_string']=document['native_string']
                if document.get('native_thermo'):
                    summary=document['native_thermo']
                    entry['native_structure']={'format':summary['format'],'sandbox_path':'/inputs/'+name+'.thermo.json',
                        'summary':summary,'source_sha256':document['sha256'],
                        'schema':'Independent blocks with tables.headers and row.values; consume this native parsed JSON rather than re-guessing raw log segmentation. Raw log remains available for audit.'}
                result.append(entry);seen.add(document['path'])
                for raw in _authenticated_raw_inputs(packet,document):
                    if not isinstance(raw,dict):continue
                    path=raw.get('path')
                    matches=[fact for fact in packet.get('file_facts',[]) if fact.get('path')==path
                        and fact.get('sha256')==raw.get('sha256') and fact.get('size')==raw.get('bytes')]
                    if (not isinstance(path,str) or Path(path).is_absolute() or '..' in Path(path).parts
                            or raw.get('format')!='text' or type(raw.get('bytes')) is not int
                            or not 0<raw['bytes']<=64*1024**2 or not matches or path in seen):
                        continue
                    used={item['id']for item in result};number=1
                    while 'D'+str(number) in used:number+=1
                    identifier='D'+str(number)
                    result.append({'id':identifier,'format':'text','path':path,'sha256':raw['sha256'],
                        'bytes':raw['bytes'],'source_id':source['id'],'sandbox_path':'/inputs/'+identifier+'.log',
                        'description':'Complete owned original data; inspect in the sandbox. No numeric summary is substituted.'})
                    seen.add(path)
    if len(result)>64:raise ValueError('plot input catalogue exceeds bounded budget')
    return result


def native_thermo_input(raw, input_id):
    """Create a deterministic data-only JSON view, never a handwritten CSV."""
    if b'LAMMPS'not in raw or not re.search(rb'(?m)^\s*(?:Step|Temp|Time)\s',raw):return None
    from .md_result_facts import parse_md_thermo,md_thermo_summary
    history=parse_md_thermo(raw)
    if not history['blocks']:return None
    encoded=json.dumps(history,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
    if len(encoded)>8*1024*1024:raise ValueError('native structured thermo input exceeds byte bound')
    name=input_id+'.thermo.json'
    facts={'format':history['format'],'source_sha256':history['source_sha256'],
        'parser_sha256':history['parser_sha256'],'sha256':hashlib.sha256(encoded).hexdigest(),
        'bytes':len(encoded),'sandbox_path':'/inputs/'+name,'summary':md_thermo_summary(history),
        'schema':'blocks[] are independent executions; tables[].headers define row.values positions; each row/source has original line+SHA. Read native_steps only from Loop summaries; do not join blocks or infer convergence.'}
    return name,encoded,facts
