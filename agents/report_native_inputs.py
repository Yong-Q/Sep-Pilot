"""Native report tables and independently witnessed PNG review provenance.

No model-transcribed values, CSV fabrication, new computation or author-granted
visual acceptance. Every inherited review is re-resolved from current producers.
"""
import hashlib,json,re
from pathlib import Path


def append_thermo_tables(raw,fact,source_id,tables,*,max_rows):
    if b'LAMMPS'not in raw or not re.search(rb'(?m)^\s*(?:Step|Temp|Time)\s',raw):return None
    from .md_result_facts import parse_md_thermo,md_thermo_summary
    history=parse_md_thermo(raw)
    if history['source_sha256']!=fact['sha256']:raise ValueError('native thermo report source SHA differs')
    for block in history['blocks']:
        for native in block['tables']:
            if len(native['rows'])>max_rows:raise ValueError('table needs explicit chunked reporting: use the data sandbox to process the owned source in bounded chunks and supply a traceable task-relevant result; no rows were silently discarded')
            headers=native['headers']
            if len(headers)!=len(set(headers))or'native_row'in headers:raise ValueError('ambiguous native thermo report headers')
            if not native['rows']:continue
            identity={'format':history['format'],'path':fact['path'],'source_sha256':fact['sha256'],
                'parser_sha256':history['parser_sha256'],'block_id':block['id'],'table_id':native['id'],
                'header_source':native['header_source']}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True,allow_nan=False).encode()).hexdigest()
            previous=next((t for t in tables if t.get('native_identity_sha256')==key),None)
            if previous:
                previous.setdefault('source_ids',[previous['source_id']])
                if source_id not in previous['source_ids']:previous['source_ids'].append(source_id)
                continue
            rows=[];anchors=[]
            for row in native['rows']:
                if len(row['values'])!=len(headers):raise ValueError('native thermo report row width differs')
                rows.append({**dict(zip(headers,row['values'])),'native_row':str(row['source']['line'])})
                anchors.append(row['source'])
            context={'block_id':block['id'],'table_id':native['id'],'kind':block['kind'],
                'stage':block['stage'],'native_steps':block['native_steps'],'unit_style':native['unit_style'],
                'scientific_convergence':block['convergence'],'complete':block['complete'],
                'errors':block['errors'],'stopping_criterion':(block.get('native_minimization')or{}).get('stopping_criterion')}
            tables.append({'id':'T'+str(len(tables)+1),'source_id':source_id,'source_ids':[source_id],
                'format':'native_thermo','path':fact['path'],'sha256':fact['sha256'],
                'parser_sha256':history['parser_sha256'],'native_identity_sha256':key,
                'native_source':identity,'native_context':context,'row_sources':anchors,
                'columns':[*headers,'native_row'],'display_columns':list(headers),
                'provenance_columns':['native_row'],'row_identity_columns':['native_row'],
                'column_units':native['column_units'],'rows':rows,'row_count':len(rows),
                'complete':not native['rejected_rows'],'units':'as declared by the native header and Unit style; normalization unknown'})
    return md_thermo_summary(history)


def _reviewed_publication_source(root, contract, result, artifact, image_fact, *, read, budget, file_facts):
    """Reopen the original PNG behind a content-addressed native publication."""
    from .report_authoring import _relative
    reference = artifact.get('review_source')
    if reference is None and 'publication_schema' not in artifact:
        return image_fact
    directory = (contract.get('arguments') or {}).get('output_dir')
    if (artifact.get('publication_schema') != 'reviewed-png-publication-v1'
            or not isinstance(reference, dict) or not directory
            or str(_relative(root, result.get('output_dir', ''))) != str(_relative(root, directory))
            or Path(image_fact['path']).parent != _relative(root, directory)
            or Path(image_fact['path']).name != 'figure-' + image_fact['sha256'] + '.png'
            or reference.get('sha256') != image_fact['sha256']
            or reference.get('bytes') != image_fact['size']):
        raise ValueError('published PNG lacks its exact native content identity')
    original = str(_relative(root, reference.get('path', '')))
    if original == image_fact['path']:
        raise ValueError('published PNG source must be the independently reviewed attempt')
    for group in (result.get('attempt_artifacts') or [],
                  (result.get('sandbox_receipt') or {}).get('artifacts') or []):
        matches = [item for item in group if isinstance(item, dict) and item.get('kind') == 'PNG'
            and str(_relative(root, item.get('path', ''))) == original
            and item.get('sha256') == reference['sha256'] and item.get('bytes') == reference['bytes']]
        if len(matches) != 1:
            raise ValueError('published PNG is not bound to the native sandbox original')
    raw, fact = read(root, root / original, budget)
    if fact['sha256'] != image_fact['sha256'] or fact['size'] != image_fact['size']:
        raise ValueError('published PNG differs from its independently reviewed source')
    file_facts.append(fact)
    return fact


def verified_visual_evidence(root,state,start_step,image_fact,*,read,budget,file_facts,cache):
    """Only native producer+current private reviewer CP may verify this PNG."""
    from .report_authoring import _relative,_result
    from .report_recovery import _OwnedFiles
    from .local_preparation_recovery import checkpoint_call_witness
    from .figure_visual_review import validate_review
    owner={key:state[key]for key in('username','conv_id')};root=Path(root)
    pending=[start_step];seen=set()
    while pending:
        step=pending.pop()
        if step in seen:continue
        seen.add(step)
        if len(seen)>64:raise ValueError('visual provenance ancestry exceeds bound')
        node=state['nodes'].get(step,{})
        if node.get('status')!='succeeded':raise ValueError('visual producer ancestry is stale')
        contract=node.get('contract')or{};result=_result(node.get('result'))
        if contract.get('tool')in{'generate_scientific_report','generate_result_figures'}:
            pending.extend(result.get('source_steps')or[])
        if contract.get('tool')!='generate_result_figures':continue
        matches=[a for a in result.get('artifacts',[])if isinstance(a,dict)and
            str(_relative(root,a.get('path','')))==image_fact['path']and a.get('sha256')==image_fact['sha256']]
        if not matches:continue
        if len(matches)!=1:raise ValueError('ambiguous native figure publication')
        review=result.get('visual_review')or{}
        if review.get('verified')is not True:return {'status':'pending','reason':'producer has no verified visual decision'}
        cid=review.get('call_id');packet_sha=result.get('source_packet_sha256')
        if not isinstance(cid,str)or not re.fullmatch('[a-zA-Z0-9_-]{1,128}',cid):return {'status':'pending','reason':'native visual call unavailable'}
        key=(step,node.get('token'),cid,image_fact['path'],image_fact['sha256'],image_fact['size'])
        if key in cache:return cache[key]
        reviewed_fact=_reviewed_publication_source(root,contract,result,matches[0],image_fact,
            read=read,budget=budget,file_facts=file_facts)
        path=root/'evidence'/(cid+'.json')
        checkpoint=root/'workers'/hashlib.sha256(('report-stage:'+step+':'+str(node.get('token'))+':visual').encode()).hexdigest()[:24]/'session_checkpoint.json'
        if not path.is_file()or not checkpoint.is_file():return {'status':'pending','reason':'native visual evidence/checkpoint unavailable'}
        raw,ef=read(root,path,budget);call=json.loads(raw)
        cpraw,cpf=read(root,checkpoint,budget);cp=json.loads(cpraw)
        expected={'phase':'visual','packet_sha256':packet_sha,'report_step':step,'report_token':node.get('token')}
        if (call.get('tool')!='report_model_stage'or call.get('agent')!='figure-reviewer'or call.get('failed')is not False or call.get('params')!=expected
                or call.get('call_id')!=cid or (cp.get('owner')or{}).get('role')!='report_visual'):
            raise ValueError('stale native visual producer/reviewer identity')
        files=_OwnedFiles(root,owner)
        try:checkpoint_call_witness(files,checkpoint,call,step_id=step,require_result_hash=True)
        finally:files.close()
        record=_result(call.get('result'))
        if (record.get('native_stage')!='report_visual'or record.get('packet_sha256')!=packet_sha
                or record.get('model_response_id')!=review.get('model_response_id')):
            raise ValueError('native visual response differs from current producer')
        decision=validate_review({'visual':record.get('visual'),'actor':'figure-reviewer','call_id':cid,
            'model_response_id':record.get('model_response_id'),'image_receipts':record.get('image_receipts',[])})
        exact=[a for a in decision['image_receipts']if a.get('path')==reviewed_fact['path']and
            a.get('sha256')==reviewed_fact['sha256']and a.get('bytes')==reviewed_fact['size']]
        if decision['verified']is not True or len(exact)!=1:
            raise ValueError('native visual verdict is stale or belongs to another PNG')
        author_notes=result.get('plot_limitations') or []
        review_notes=decision.get('limitations') or []
        if (not isinstance(author_notes,list) or len(author_notes)>32
                or any(not isinstance(note,str) or len(note)>2000 for note in author_notes)
                or not isinstance(review_notes,list) or len(review_notes)>16
                or any(not isinstance(note,str) or len(note)>2000 for note in review_notes)):
            raise ValueError('native figure limitations exceed bounded text contract')
        file_facts.extend([ef,cpf])
        proof={'status':'verified','scientific_acceptance':False,'producer_step':step,
            'producer_token':node.get('token'),'producer_attempt_id':node.get('attempt_id'),
            'producer_execution_fingerprint':node.get('execution_fingerprint'),
            'producer_call_id':(node.get('result_ref')or{}).get('call_id'),
            'review_call_id':cid,'review_model_response_id':record['model_response_id'],
            'review_evidence_path':ef['path'],'review_evidence_sha256':ef['sha256'],
            'review_checkpoint_path':cpf['path'],'review_checkpoint_sha256':cpf['sha256'],
            'packet_sha256':packet_sha,'image_path':image_fact['path'],'image_sha256':image_fact['sha256'],
            'author_limitations':author_notes,'review_limitations':review_notes,
            'reviewed_image_path':reviewed_fact['path'],
            'native_publication_byte_identity_verified':reviewed_fact['path']!=image_fact['path']}
        cache[key]=proof;return proof
    return {'status':'pending','reason':'no current independently witnessed figure producer'}
