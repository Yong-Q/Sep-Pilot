"""Bounded key-only guidance for the exact packet-authorized JSON view.

These hints never resolve a missing field, read an original envelope, supply a
source value, or permit a different JSON Pointer. The strict renderer remains
responsible for every actual scalar and its original file/hash binding.
"""
import json
import math
from collections import deque

_SCHEMAS=frozenset({'native_report_source_metadata_v1','native_gcmc_report_view_v1'})


def validate_fact_projection(document):
    base=document.get('json_root_pointer','')
    if base not in {'','/public'} or base and document.get('metadata_schema') not in _SCHEMAS:
        raise ValueError('unknown native fact projection root')
    return base


def _key(value):
    return isinstance(value,str) and len(value)<=96 and all(c.isprintable() for c in value)


def _escape(value):
    return value.replace('~','~0').replace('/','~1')


def scalar_pointer_examples(document,limit=48):
    """Index bounded accessible scalar names breadth first, never values."""
    validate_fact_projection(document)
    result=[];visited=0;pending=deque([(document['untrusted_content'],'',0)])
    while pending and len(result)<limit and visited<512:
        value,pointer,depth=pending.popleft();visited+=1
        if depth>16:continue
        if isinstance(value,dict):
            for index,(key,item) in enumerate(value.items()):
                if index>=128 or len(pending)>=512:break
                if _key(key):pending.append((item,pointer+'/'+_escape(key),depth+1))
        elif isinstance(value,list):
            for index,item in enumerate(value[:128]):
                if len(pending)>=512:break
                pending.append((item,pointer+'/'+str(index),depth+1))
        elif ((not isinstance(value,float) or math.isfinite(value)) and len(pointer)<=1024
              and len(json.dumps(value,ensure_ascii=False,allow_nan=False))<=2000):
            result.append(pointer)
    return result


def fact_pointer_catalogue(packet):
    rows=[];used=0
    for source in packet.get('sources',[]):
        for document in source.get('structured_data',[]):
            if len(rows)>=8:return rows
            pointers=scalar_pointer_examples(document)
            row={'document_id':document.get('id'),'json_root_pointer':validate_fact_projection(document),
                 'relative_scalar_pointer_examples':[]}
            for pointer in pointers:
                if used+len(pointer)>4096:break
                row['relative_scalar_pointer_examples'].append(pointer);used+=len(pointer)
            if not row['relative_scalar_pointer_examples']:continue
            rows.append(row)
    return rows


def missing_fact_pointer_diagnostic(document,parent):
    """Only sibling key/index names from the already authorized public view."""
    base=validate_fact_projection(document)
    hint={'document_id':str(document.get('id',''))[:64],
          'pointer_scope':'relative_to_untrusted_content',
          'original_file_view_root':base or '/',
          'action':'Select an exact existing scalar. An absent field must remain absent or be stated qualitatively as unknown; never guess or substitute a similarly named field.'}
    if isinstance(parent,dict):
        keys=[]
        for index,key in enumerate(parent):
            if index>=128 or len(keys)>=12:break
            if _key(key):keys.append(key)
        hint['available_keys']=keys
    elif isinstance(parent,list):hint['available_indices']=[str(index) for index in range(min(len(parent),12))]
    else:hint['available_keys']=[]
    return json.dumps(hint,ensure_ascii=False,separators=(',',':'))
