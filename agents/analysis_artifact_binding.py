"""Read-only source binding for an owned sandbox data artifact.

Validates provenance and byte identity, not correctness of derived science.
"""
from pathlib import Path
import copy
from .report_recovery import _OwnedFiles,_same
from .local_preparation_recovery import checkpoint_call_witness
from .recovery import result_object
from .output_contract import output_path
import json
import math


def pointer_tokens(pointer):
    if not isinstance(pointer,str) or not pointer.startswith('/') or len(pointer)>256:
        raise ValueError('bounded non-root JSON Pointer required')
    result=[]
    for part in pointer[1:].split('/'):
        if not part or '~' in part.replace('~0','').replace('~1',''):
            raise ValueError('invalid JSON Pointer escape or empty segment')
        result.append(part.replace('~1','/').replace('~0','~'))
    return result


def declared_scalar_schema(schema,pointer,value_type):
    current=schema
    for token in pointer_tokens(pointer):
        if not isinstance(current,dict):raise ValueError('JSON pointer has no declared source schema')
        if current.get('type')=='object':
            if token not in (current.get('required') or []) or token not in (current.get('properties') or {}):
                raise ValueError('JSON pointer must select a required declared source property')
            current=current['properties'][token]
        elif current.get('type')=='array' and token.isdigit() and token==str(int(token)):
            if type(current.get('minItems')) is not int or int(token)>=current['minItems']:
                raise ValueError('JSON array pointer must be guaranteed by minItems')
            current=current.get('items')
        else:raise ValueError('JSON pointer does not identify a declared scalar')
    if not isinstance(current,dict) or current.get('type')!=value_type or value_type not in {'number','integer','string','boolean'}:
        raise ValueError('JSON pointer scalar type differs from its source schema')
    return current


def _pointer_value(value,pointer):
    for token in pointer_tokens(pointer):
        if isinstance(value,dict) and token in value:value=value[token]
        elif isinstance(value,list) and token.isdigit() and token==str(int(token)) and int(token)<len(value):value=value[int(token)]
        else:raise ValueError('sandbox JSON pointer is missing')
    return value


def _scalar_type(value,kind):
    return ((kind=='boolean' and type(value)is bool) or (kind=='integer' and type(value)is int)
        or (kind=='number' and type(value) in (int,float) and math.isfinite(value))
        or (kind=='string' and isinstance(value,str) and bool(value)))


def resolve_analysis_input_bindings(root,state,consumer,arguments,registry,goal):
    root=Path(root).absolute();owner={'username':root.parent.name,'conv_id':root.name}
    if (state.get('username'),state.get('conv_id'))!=(owner['username'],owner['conv_id']):raise PermissionError('sandbox workflow owner changed')
    bindings=consumer.get('input_bindings') or {};proofs=[];files=None;resolved=copy.deepcopy(arguments)
    ancestors=set();pending=list(consumer.get('depends_on') or [])
    while pending:
        step=pending.pop()
        if step in ancestors:continue
        ancestors.add(step)
        if len(ancestors)>256:raise ValueError('sandbox dependency ancestry exceeds bound')
        pending.extend(((state.get('nodes') or {}).get(step) or {}).get('contract',{}).get('depends_on') or [])
    try:
        for target,binding in bindings.items():
            if not isinstance(binding,dict):raise ValueError('invalid sandbox data binding')
            step=binding.get('step_id');node=(state.get('nodes') or {}).get(step) or {};contract=node.get('contract') or {}
            if contract.get('tool')!='run_analysis_sandbox':continue
            index=binding.get('output_index');outputs=contract.get('expected_outputs') or []
            scalar='json_pointer' in binding
            if (set(binding)!=({'step_id','output_index','json_pointer','value_type'} if scalar else {'step_id','output_index'})
                    or type(index)is not int or not 0<=index<len(outputs)
                    or step not in ancestors or node.get('status')!='succeeded'
                    or node.get('phase')!='finish' or not node.get('token')):
                raise ValueError('sandbox data producer is not the current completed dependency')
            ref=node.get('result_ref') or {};cid=ref.get('call_id')
            if not isinstance(cid,str) or not cid.isalnum() or ref.get('evidence_path')!=str(root/'evidence'/(cid+'.json')):
                raise ValueError('sandbox producer call identity is unavailable')
            if files is None:files=_OwnedFiles(root,owner)
            call,producer=files.read(ref['evidence_path'],as_json=True)
            result=result_object(call.get('result'));args=node.get('effective_arguments') or contract.get('arguments') or {}
            if (call.get('call_id')!=cid or call.get('tool')!='run_analysis_sandbox' or call.get('failed') is not False
                    or not _same(call.get('params'),args) or not _same(result,result_object(node.get('result')))
                    or result.get('ok') is not True or result.get('status')!='data_ready_needs_review'
                    or result.get('owner')!=owner or result.get('scientific_acceptance') is not False
                    or result.get('producer')!={'step_id':step,'runtime_token':node['token']}
                    or (result.get('sandbox') or {}).get('os_isolated') is not True):
                raise ValueError('sandbox producer receipt or ownership changed')
            checkpoint=checkpoint_call_witness(files,ref.get('checkpoint_path'),call,step_id=step,require_result_hash=True)
            expected=output_path(outputs[index],Path(args['output_dir']))
            if not scalar and Path(arguments.get(target,'')).absolute()!=expected:raise ValueError('sandbox consumer parameter differs from its approved artifact')
            records=[item for item in result.get('artifacts',[]) if item.get('path')==str(expected)]
            if len(records)!=1 or records[0].get('scientific_acceptance') is not False:
                raise ValueError('sandbox artifact is not uniquely bound to producer receipt')
            data,artifact=files.read(expected,as_json=scalar)
            if artifact['sha256']!=records[0].get('sha256') or artifact['size']!=records[0].get('bytes'):
                raise ValueError('sandbox artifact bytes changed after production')
            publication=result.get('publication') or {};commit_sha=None
            if publication.get('protocol')=='nfs-link-commit-v1':
                marker,commit=files.read(Path(args['output_dir'])/'artifacts'/'.publication.json',as_json=True)
                expected_records=[{k:item.get(k)for k in ('name','sha256','bytes')}for item in result.get('artifacts',[])]
                if (commit['sha256']!=publication.get('commit_sha256') or not isinstance(marker,dict)
                        or marker.get('schema_version')!=1 or marker.get('protocol')!='nfs-link-commit-v1'
                        or marker.get('artifacts')!=expected_records):
                    raise ValueError('sandbox publication commit or complete artifact membership changed')
                if not 1<=len(expected_records)<=8 or len({item['name']for item in expected_records})!=len(expected_records):
                    raise ValueError('sandbox publication artifact membership is invalid')
                for item in expected_records:
                    name=item.get('name')
                    if not isinstance(name,str) or Path(name).name!=name:
                        raise ValueError('sandbox publication artifact name is invalid')
                    _,fact=files.read(Path(args['output_dir'])/'artifacts'/name)
                    if fact['sha256']!=item.get('sha256') or fact['size']!=item.get('bytes'):
                        raise ValueError('sandbox committed artifact changed')
                commit_sha=commit['sha256']
            elif publication and publication.get('protocol')!='atomic-rename-noreplace-v1':
                raise ValueError('unknown sandbox publication protocol')
            inputs=result.get('inputs') or []
            if not 1<=len(inputs)<=8:raise ValueError('sandbox producer input provenance is absent')
            original=[]
            for item in inputs:
                if not isinstance(item,dict) or not item.get('source_path'):raise ValueError('sandbox source identity is absent')
                _,fact=files.read(item['source_path'])
                if fact['sha256']!=item.get('sha256') or fact['size']!=item.get('bytes'):
                    raise ValueError('sandbox original input changed')
                original.append({'path':fact['path'],'sha256':fact['sha256']})
            proof={'parameter':target,'source_step':step,'producer_call_id':cid,
                'producer_sha256':producer['sha256'],'checkpoint_sha256':checkpoint['sha256'],
                'artifact_sha256':artifact['sha256'],'original_inputs':original,
                'scientific_acceptance':False}
            if commit_sha:proof['publication_commit_sha256']=commit_sha
            if scalar:
                if target in resolved:raise ValueError('sandbox scalar cannot override an explicit parameter')
                declared=[item for item in result.get('declared_outputs',[]) if item.get('name')==expected.name and item.get('format')=='json']
                if len(declared)!=1:raise ValueError('sandbox scalar source declaration is absent')
                kind=binding['value_type'];declared_scalar_schema(declared[0].get('json_schema'),binding['json_pointer'],kind)
                value=_pointer_value(data,binding['json_pointer'])
                if not _scalar_type(value,kind):raise ValueError('sandbox scalar value has the wrong type or is nonfinite')
                tool=registry.get(consumer.get('tool',''));schema=(tool.input_schema.get('properties') or {}).get(target) if tool else None
                if not isinstance(schema,dict):raise ValueError('sandbox scalar target schema is unavailable')
                from jsonschema import Draft7Validator
                if list(Draft7Validator(schema).iter_errors(value)):raise ValueError('sandbox scalar violates the target parameter schema')
                resolved[target]=value;proof.update(json_pointer=binding['json_pointer'],value_type=kind,value=value)
            proofs.append(proof)
        if proofs:
            issues=registry.validate_params(consumer.get('tool',''),resolved)
            if issues:raise ValueError('resolved sandbox inputs violate target tool schema: '+str(issues[:4]))
            from .workflow_patch import execution_contract
            from .goal_contract import GoalContract
            from .parallel_workflow import _approval_digest
            current=goal.to_dict();approved=[item for item in current.get('approved_nodes',[])if item.get('step_id')==consumer.get('step_id')]
            if (len(approved)!=1 or execution_contract(approved[0])!=execution_contract(consumer)
                    or _approval_digest(state.get('goal_contract') or {})!=_approval_digest(current)
                    or state.get('plan_version')!=current.get('approved_plan_version')):
                raise ValueError('sandbox consumer is not the exact currently approved node')
            allowed,reason=GoalContract.from_dict(current).guard_tool_call(consumer.get('tool',''),resolved,enforce_approved_node=False)
            if not allowed:raise ValueError('resolved sandbox input violates approved scientific scope: '+str(reason))
        return resolved,proofs
    finally:
        if files is not None:files.close()
