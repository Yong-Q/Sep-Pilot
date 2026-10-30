"""Evidence-backed control receipts in model requests; no history mutation.

This is presentation only. It neither validates scientific completion nor grants
permissions. Full contracts remain in view; bulky file bookkeeping and duplicate
copies remain retrievable from the exact owned native receipt.
"""
import copy
import hashlib
import json
import re
from pathlib import Path

CONTROL_TOOLS=frozenset({'propose_workflow_patch','apply_workflow_patch','task_line_query'})
SEARCH_TOOLS=frozenset({'grep_search'})
_FILE_FIELDS=frozenset({'path','name','size','bytes','sha256','mtime','modified_at','exists','is_file','is_dir','kind'})


def _encoded(value):return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def _without_reference(value):
    result=copy.deepcopy(value);result.pop('evidence_ref',None);return result


def _owned_record(root,owner,tool,args,result):
    from .report_recovery import _OwnedFiles
    from .recovery import result_object
    ref=result.get('evidence_ref')or{};cid=ref.get('call_id')
    if not isinstance(cid,str)or not re.fullmatch('[A-Za-z0-9_-]{1,128}',cid):return None
    files=None
    try:
        files=_OwnedFiles(Path(root).parent,owner)
        path=Path(root)/(cid+'.json');saved,fact=files.read(path,as_json=True)
        if (saved.get('call_id')!=cid or saved.get('tool')!=tool or saved.get('failed')
                or _encoded(saved.get('params'))!=_encoded(args)
                or _encoded(_without_reference(result_object(saved.get('result'))))!=_encoded(_without_reference(result))):return None
        return {'call_id':cid,'path':str(path),'sha256':fact['sha256'],
            'result_sha256':hashlib.sha256(str(saved.get('result','')).encode()).hexdigest()}
    except (OSError,ValueError,KeyError,TypeError):return None
    finally:
        if files is not None:files.close()


def _metadata_reference(value):
    encoded=_encoded(value)
    return {'model_view_metadata_reference':True,'count':len(value),
            'sha256':hashlib.sha256(encoded.encode()).hexdigest(),'sample':copy.deepcopy(value[:2]),
            'full_metadata_tool':'task_line_query',
            'note':'File bookkeeping only; full original remains in this receipt evidence, not a scientific acceptance proof.'}


def _file_metadata(value):
    return isinstance(value,list)and all(isinstance(item,str)or(isinstance(item,dict)
        and set(item)<=_FILE_FIELDS and isinstance(item.get('path')or item.get('name'),str))for item in value)


def _bounded_result(value):
    """Project large receipt content only; evidence remains the authority."""
    if isinstance(value,str) and len(value)>2000:
        return value[:2000]+' … [receipt text truncated in model view; full original in evidence reference]'
    if isinstance(value,list):
        shown=[_bounded_result(item) for item in value[:30]]
        if len(value)>30:
            shown.append({'model_view_omitted_items':len(value)-30,'full_list_in_evidence':True})
        return shown
    if isinstance(value,dict):return {key:_bounded_result(item) for key,item in value.items()}
    return value


def _project(value):
    if isinstance(value,list):return [_project(v)for v in value]
    if not isinstance(value,dict):return value
    result={key:_project(item)for key,item in value.items()}
    if isinstance(value.get('step_id'),str)and isinstance(value.get('tool'),str)and isinstance(value.get('arguments'),dict):
        # Never shorten arguments, expected_outputs, dependencies, bindings,
        # locks, resource budgets, result values, errors or scientific facts.
        for field in ('output_files',):
            items=value.get(field)
            if _file_metadata(items)and len(_encoded(items))>2000:result[field]=_metadata_reference(items)
        manifest=value.get('path_manifest')
        if isinstance(manifest,dict):
            for field in ('input_paths','output_files','files'):
                items=manifest.get(field)
                if _file_metadata(items)and len(_encoded(items))>2000:
                    result['path_manifest'][field]=_metadata_reference(items)
    return result


def project_control_history(messages,*,evidence_root,owner,minimum_characters=12000,trusted_calls=None):
    """Return an independent request copy, retaining lawful tool IDs/arguments.

    Unknown/mismatched evidence fails open *to the original full transcript*,
    never to a compressed claim. No file is read for ordinary chat or small
    results. Reads are owned/NOFOLLOW/bounded using the existing native reader.
    """
    from .request_context_view import project_control_result
    view=copy.deepcopy(messages);before=len(json.dumps(messages,ensure_ascii=False,default=str));count=0;checked=0
    known={call.get('call_id'):call for call in trusted_calls or[]if isinstance(call,dict)}
    for index in range(len(view)-1,0,-1):
        message=view[index]
        if index==0 or message.get('role')!='user' or not isinstance(message.get('content'),list):continue
        previous=view[index-1]
        if previous.get('role')!='assistant' or not isinstance(previous.get('content'),list):continue
        calls={b.get('id'):b for b in previous['content']if isinstance(b,dict)and b.get('type')=='tool_use'}
        for block in message['content']:
            if not isinstance(block,dict)or block.get('type')!='tool_result' or block.get('is_error'):continue
            call=calls.get(block.get('tool_use_id'));content=block.get('content')
            if not call or not isinstance(call.get('name'),str) or not isinstance(content,str)or len(content)<minimum_characters or not evidence_root:continue
            try:obj=json.loads(content)
            except (ValueError,TypeError):continue
            if not isinstance(obj,dict)or obj.get('error')or obj.get('blocked')or obj.get('ok')is False:continue
            if checked>=8:continue
            checked+=1
            proof=_owned_record(evidence_root,owner,call['name'],call.get('input'),obj)
            if proof is None:continue
            if trusted_calls is not None:
                witness=known.get(proof['call_id'])
                if not witness or witness.get('failed')is not False or witness.get('tool')!=call['name']or witness.get('params')!=call.get('input'):
                    continue
                if witness.get('result_sha256')!=proof['result_sha256']:continue
            if call['name'] in SEARCH_TOOLS:
                results=obj.get('results')
                if not isinstance(results,list) or not all(isinstance(line,str) for line in results):continue
                projected={key:copy.deepcopy(value) for key,value in obj.items() if key!='results'}
                projected['results']=[line if len(line)<=700 else line[:700]+' … [matching line truncated in model view]' for line in results[:20]]
                projected['model_view_truncated']=(len(results)>20 or any(len(line)>700 for line in results))
                projected['model_view_note']='Search result excerpts only; the original owned tool receipt remains unchanged and can be inspected by its evidence reference. Truncated text is not evidence of absence.'
            elif call['name'] in CONTROL_TOOLS:
                projected=project_control_result(obj,proof)
            else:
                projected=_bounded_result(obj)
                projected['model_view_truncated']=True
                projected['model_view_note']='Bounded tool-result view only; exact original receipt and data remain available through the owned evidence reference. Do not treat omitted content as absent.'
            patch=projected.get('patch')
            if (isinstance(patch,dict)and isinstance(projected.get('proposed_nodes'),list)
                    and patch.get('proposed_nodes')==projected['proposed_nodes']):
                patch['proposed_nodes']={'model_view_reference':True,'same_as':'$.proposed_nodes',
                    'count':len(projected['proposed_nodes']),
                    'note':'Identical contracts appear once in top-level proposed_nodes; original native receipt is unchanged.'}
            projected['request_view_provenance']={**proof,'original_characters':len(content),
                'full_result_tool':'read_file','not_authorization':True}
            encoded=json.dumps(projected,ensure_ascii=False,separators=(',',':'))
            if len(encoded)<len(content):block['content']=encoded;count+=1
    return view,{'projected_receipts':count,'checked_receipts':checked,'characters_before':before,
                 'characters_after':len(json.dumps(view,ensure_ascii=False,default=str))}


def drop_empty_assistant_messages(messages):
    """Remove empty assistant turns from the outbound view only.

    Empty user messages and any tool-result adjacency are ambiguous: refuse
    the request instead of silently discarding user input or tool evidence.
    Persisted conversation history is not modified.
    """
    if not isinstance(messages,list):raise ValueError('provider messages must be a list')
    cleaned=[]
    for index,message in enumerate(messages):
        if not isinstance(message,dict):raise ValueError('provider message must be an object')
        content=message.get('content')
        if not isinstance(content,(str,list)):raise ValueError('provider message content has unsupported shape')
        empty=(isinstance(content,str) and not content.strip()) or isinstance(content,list) and not content
        if not empty:
            cleaned.append(message)
            continue
        if message.get('role')!='assistant':raise ValueError('empty non-assistant message cannot be silently discarded')
        next_message=messages[index+1] if index+1<len(messages) else None
        next_content=next_message.get('content') if isinstance(next_message,dict) else None
        if isinstance(next_content,list) and any(isinstance(block,dict) and block.get('type')=='tool_result' for block in next_content):
            raise ValueError('empty assistant message precedes tool results; preserve evidence for repair')
    if not cleaned:raise ValueError('provider request has no messages after projection')
    return cleaned
