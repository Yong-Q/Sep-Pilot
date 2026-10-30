"""Lossless references for request-only orchestration bookkeeping.

Native state and scientific call results are never changed. Full executable
arguments/outputs/bindings, current pending changes and validation remain data.
References identify the owned native receipt or fresh scoped runtime tool.
"""
import copy
import hashlib
import json
from pathlib import Path

CONTRACT_FIELDS=frozenset({'agent','tool','arguments','effective_arguments','input_bindings',
    'depends_on','expected_outputs','resource_locks'})
ACTIVE=frozenset({'pending','running','submitted','waiting_jobs','waiting_prerequisite',
    'prefinish','failed','validation_failed','uncertain','needs_resources'})
CONTROLLERS=frozenset({'lead-orchestrator','operations'})
CONDITION_KEYS=frozenset({'gas','gases','temperature','temperature_K','pressure','pressure_Pa','pressures',
    'p_min','p_max','method','model_type','target','target_unit','unit','units','features','cycles',
    'initialization_cycles','production_cycles','ensemble','directions','n_steps','num_processes',
    'cpus_per_task','timestep','cutoff','unit_cells','forcefield','framework_forcefield',
    'adsorbate_forcefield','mixing_rule','tolerance','gradient_tolerance','max_iterations','grid_spacing'})


def encoded(value):
    return json.dumps(value,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':'))


def digest(value):return hashlib.sha256(encoded(value).encode()).hexdigest()


def contract_digest(node):
    from .workflow_patch import execution_contract
    return digest(execution_contract(node))


def _reference(value,proof,pointer,*,runtime=None):
    reference={'model_view_reference':True,'kind':'orchestration_metadata',
        'sha256':digest(value),'count':len(value)if isinstance(value,(dict,list))else None,
        'not_authorization':True,'source_call_id':proof['call_id']}
    if runtime is None:
        reference.update(json_pointer=pointer,full_metadata_tool='read_file')
    else:
        reference.update(full_metadata_tool='lifecycle_state',native_runtime={**runtime,'field_path':pointer},
            note='Fresh scoped runtime metadata; compare its hash before using a later version.')
    return reference


def project_node(node,proof,pointer,*,runtime=None,minimum=700):
    """Only explicit bookkeeping fields are replaced; unknown facts stay full."""
    if not isinstance(node,dict):return copy.deepcopy(node)
    result=copy.deepcopy(node)
    if (not isinstance(node.get('step_id'),str)or not node['step_id']
            or not isinstance(node.get('tool'),str)or not isinstance(node.get('arguments'),dict)):return result
    completed=node.get('status')in{'completed','succeeded','superseded'}
    if completed:
        # Completed contracts are recoverable from the owned full receipt (or
        # the current scoped task_line_query step). They need not be repeated
        # beside every current runtime view and historical proposal.
        for field in CONTRACT_FIELDS-{'agent','tool','depends_on'}:result.pop(field,None)
        args=node['arguments']
        conditions={key:copy.deepcopy(value)for key,value in args.items()if key in CONDITION_KEYS and(
            isinstance(value,(int,float,bool,type(None)))
            or isinstance(value,str)and len(value)<=256
            or isinstance(value,list)and len(value)<=16 and all(isinstance(item,(str,int,float,bool,type(None)))for item in value))}
        result['completed_contract_reference']={**_reference(node,proof,pointer,runtime=runtime),
            'step_id':node.get('step_id'),'full_contract_tool':'task_line_query',
            'query':{'step_id':node.get('step_id')},'condition_summary':conditions,
            'contract_sha256':contract_digest(node),'hash_kind':'execution_contract_v1'}
        # Existing source results/validation/errors are deliberately retained.
    for name in('output_files','output_contract_history'):
        value=node.get(name)
        if isinstance(value,(list,dict))and len(encoded(value))>minimum:
            result[name]=_reference(value,proof,pointer+'/'+name,runtime=runtime)
    manifest=node.get('path_manifest')
    if isinstance(manifest,dict)and len(encoded(manifest))>minimum:
        # Preserve actual directories and exact output contracts without a
        # duplicate recursive inventory at every representation of the node.
        kept={key:copy.deepcopy(value)for key,value in manifest.items()
              if key in({'calculation_dir','result_dir','checkpoint_path','evidence_path'}if completed else
                        {'calculation_dir','result_dir','checkpoint_path','evidence_path','expected_outputs','input_bindings'})}
        inventory={key:value for key,value in manifest.items()if key not in kept}
        if inventory:
            kept['bookkeeping_reference']=_reference(manifest,proof,pointer+'/path_manifest',runtime=runtime)
        result['path_manifest']=kept
    return result


def project_control_result(value,proof):
    """Current pending changes are deliberately preserved, including arguments."""
    protected=CONTRACT_FIELDS|{'result','results','scientific_results','facts','validation','node_verification','changes'}
    def walk(item,pointer):
        if isinstance(item,list):return[walk(child,pointer+'/'+str(index))for index,child in enumerate(item)]
        if not isinstance(item,dict):return copy.deepcopy(item)
        shaped=project_node(item,proof,pointer)
        for key,child in item.items():
            if key in protected or key in{'output_files','path_manifest','output_contract_history'}:continue
            escaped=key.replace('~','~0').replace('/','~1')
            shaped[key]=walk(child,pointer+'/'+escaped)
        return shaped
    result=walk(value,'')
    patch=result.get('patch')
    if isinstance(patch,dict)and isinstance(value['patch'].get('proposed_nodes'),list):
        original=value['patch']['proposed_nodes']
        if original==value.get('proposed_nodes'):
            patch['proposed_nodes']={'model_view_reference':True,'same_as':'/proposed_nodes',
                'count':len(original),'sha256':digest(original),'not_authorization':True}
        else:
            patch['proposed_nodes']=[project_node(node,proof,'/patch/proposed_nodes/'+str(i))for i,node in enumerate(original)]
    return result


def project_runtime_context(summary,state,*,owner,evidence_root,agent_name):
    """Scope heavy native node details while preserving global graph headers."""
    result=copy.deepcopy(summary)
    if (not isinstance(state,dict)or any(state.get(k)!=v for k,v in owner.items())
            or not evidence_root):return result
    from .request_projection import _owned_record
    from .recovery import result_object
    nodes=state.get('nodes',{});active={key for key,node in nodes.items()if node.get('status')in ACTIVE}
    own={key for key,node in nodes.items()if node.get('contract',{}).get('agent')==agent_name}
    relevant=set(nodes)if agent_name in CONTROLLERS else(own&active or own)
    pending=list(relevant)
    while pending:
        for parent in nodes.get(pending.pop(),{}).get('contract',{}).get('depends_on',[]):
            if parent not in relevant:relevant.add(parent);pending.append(parent)
    # runtime_summary's ordinary size cap must not discard a relevant active
    # executable contract. Keep the full typed contract from native state.
    for key in active&relevant:
        if key not in result.setdefault('nodes',{}):
            raw=nodes[key];result['nodes'][key]={'status':raw.get('status'),
                'agent':raw.get('contract',{}).get('agent'),'tool':raw.get('contract',{}).get('tool'),
                'depends_on':raw.get('contract',{}).get('depends_on',[]),'result_ref':raw.get('result_ref'),
                'job_ids':raw.get('job_ids',[]),'error':raw.get('error')}
    for key,node in result.get('nodes',{}).items():
        raw=nodes.get(key,{})
        if key in relevant and raw.get('status')in ACTIVE:
            node['contract']={name:copy.deepcopy(raw.get('contract',{}).get(name))for name in CONTRACT_FIELDS if name in raw.get('contract',{})}
            if isinstance(raw.get('effective_arguments'),dict):node['effective_arguments']=copy.deepcopy(raw['effective_arguments'])
            if isinstance(raw.get('resource_allocation'),dict):node['resource_allocation']=copy.deepcopy(raw['resource_allocation'])
            if raw.get('status')in{'failed','validation_failed','uncertain'}:
                for field in('error','validation','node_verification','validation_dossier','runtime_input_repair'):
                    if field in raw:node[field]=copy.deepcopy(raw[field])
                if 'result'in raw:node['failure_result']=copy.deepcopy(raw['result'])
        ref=raw.get('result_ref')or{};call_id=ref.get('call_id')
        if not call_id or ref.get('evidence_path')!=str(Path(evidence_root)/(call_id+'.json')):continue
        wire=result_object(raw.get('result'));wire=copy.deepcopy(wire)if isinstance(wire,dict)else None
        if wire is None:continue
        wire['evidence_ref']={'call_id':call_id}
        contract=raw.get('contract',{})
        proof=_owned_record(evidence_root,owner,contract.get('tool'),
            raw.get('effective_arguments')or contract.get('arguments',{}),wire)
        if proof is None:continue
        location={'plan_version':state.get('plan_version'),'step_id':key,'token':raw.get('token')}
        projected=project_node({**contract,'path_manifest':raw.get('path_manifest'),
            'status':raw.get('status'),'step_id':key,'output_files':raw.get('output_files',[])},proof,'/nodes/'+key,runtime=location)
        if raw.get('status')in{'completed','succeeded','superseded'}:
            node.pop('path_manifest',None)
            complete=projected['completed_contract_reference']
            node['completed_contract_reference']={'tool':'task_line_query','step_id':key,
                'plan_version':state.get('plan_version'),'token':raw.get('token'),
                'contract_sha256':contract_digest(contract),'hash_kind':'execution_contract_v1',
                'conditions':complete['condition_summary'],'not_authorization':True}
        else:
            node['path_manifest']=projected.get('path_manifest')
            node['request_view_provenance']={**proof,'not_authorization':True}
        if isinstance(node.get('result_ref'),dict):node['result_ref']['native_evidence_sha256']=proof['sha256']
    result['request_view_scope']={'role':agent_name,'relevant_nodes':sorted(relevant),
        'active_contracts_complete':True,'not_authorization':True}
    return result
