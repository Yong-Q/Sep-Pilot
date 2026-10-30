"""Discover a proved forward metadata proposal; never apply, retry or accept it."""
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

_ACTION = 'inspect_forward_metadata_alignment'
_SOURCE_CACHE = {}


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def source_epoch():
    """Bind this capability to loaded proof/control bodies that match source."""
    from .executor_identity import executor_identity
    from .workflow_metadata_alignment import prove_forward_transition, session_transition_proof
    from .native_resource_carry import prove_native_resource_carry, proof_matches
    from .parallel_workflow import ParallelWorkflow, _valid_execution_approval
    from .workflow_user_control import workflow_control_guard
    from .workflow_patch import execution_contract
    from .session import Session
    functions=(prove_forward_transition, session_transition_proof,
        prove_native_resource_carry, proof_matches, ParallelWorkflow._validate_transition_reuse,
        ParallelWorkflow.start, ParallelWorkflow.recover_notifications,
        Session._propose_workflow_patch, Session._apply_workflow_patch,
        Session.resume_lifecycle_event, Session._get_error_branch_summary,
        _valid_execution_approval, workflow_control_guard, execution_contract,
        source_epoch, _digest, _control, _candidate, _key, _fresh_jobs,
        notify_metadata_capability, metadata_event_current, orient_metadata_event,
        metadata_application_authority, metadata_execution_approval)
    service=sys.modules.get('api')
    if service is not None:
        functions+=tuple(getattr(service,name) for name in
            ('_lifecycle_controls_changed','_dispatch_conversation_lifecycle'))
    signature=[]
    for function in functions:
        source=Path(inspect.getsourcefile(function)).resolve(strict=True);st=source.stat()
        signature.append((id(function.__code__),str(source),st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns))
    key=tuple(signature)
    if _SOURCE_CACHE.get('key')==key:return copy.deepcopy(_SOURCE_CACHE['value'])
    bodies=[executor_identity(function) for function in functions]
    if any(body.get('verified')is not True or body.get('matches')is not True for body in bodies):
        raise ValueError('metadata capability requires current matching loaded proof bodies')
    epoch=_digest([{k:body[k] for k in ('qualified_name','loaded_code_sha256','source_code_sha256','source_sha256')} for body in bodies])
    value={'epoch':epoch,'loaded_bodies':bodies,'scope':'named proof/control bodies; not scientific validation'}
    _SOURCE_CACHE.update(key=key,value=copy.deepcopy(value))
    return value


def _control(main,owner):
    getter=getattr(main,'_native_workflow_control',None)
    if callable(getter):
        current=getter()
        if (not isinstance(current,dict) or current.get('owner')!=owner
                or type(current.get('epoch'))is not int or current['epoch']<0
                or current.get('actor_bound')is not True or current.get('interrupted')is not False):
            raise ValueError('metadata capability native control revoked')
        return {'owner':list(owner),'epoch':current['epoch']}
    if main._interrupt_requested():raise ValueError('metadata capability owner turn revoked')
    return {'owner':list(owner),'legacy_uid':getattr(main,'_api_turn_id',None)}


def _candidate(runtime,state,step,*,line=None):
    """Only a new approved same-science plan and receipt-only discrepancy."""
    from .workflow_patch import execution_contract
    from .parallel_workflow import _valid_execution_approval
    main=runtime.main;goal=copy.deepcopy(main.goal_contract.to_dict());saved=state.get('goal_contract')or{}
    owner=(runtime.username,runtime.conv_id);node=state.get('nodes',{}).get(step,{})
    line=runtime._line()if line is None else line;approval=copy.deepcopy(main.context.get('approved_execution_proof'))
    if ((state.get('username'),state.get('conv_id'))!=owner or state.get('user_paused')
            or state.get('status')not in {'active','needs_user'} or state.get('pause_kind')not in {None,'failure'}
            or main._pending_workflow_patch or main._awaiting_plan_approval
            or (main._pending_user_interaction or {}).get('tool')not in {None,'lifecycle_wait'}
            or goal.get('execution_authorized')is not True
            or type(state.get('plan_version'))is not int or type(goal.get('approved_plan_version'))is not int
            or goal['approved_plan_version']<=state['plan_version']
            or not _valid_execution_approval(approval,goal,line,*owner)
            or any(saved.get(k)!=goal.get(k) for k in ('version','method','gases','parameters','execution_mode','execution_authorized'))
            or node.get('status')!='prefinish' or node.get('tool_returned')is not True
            or node.get('jobs_confirmed_terminal')is not True or not node.get('job_ids') or not node.get('token')
            or any(n.get('status')in {'running','waiting_jobs','uncertain','waiting_prerequisite'} for n in state.get('nodes',{}).values())):
        raise ValueError('no current owned terminal receipt-only metadata capability')
    if any(future is not None and not future.done() for future in runtime.futures.values()):
        raise ValueError('native worker still active')
    incoming=copy.deepcopy(goal.get('approved_nodes')or[])
    by_id={item['step_id']:item for item in incoming}
    if len(by_id)!=len(incoming) or set(by_id)!=set(state['nodes']):raise ValueError('approved graph changed')
    visible={item['step_id']:item for item in line.get('steps',[]) if 'expected_outputs'in item and item.get('status')!='superseded' and not item.get('branch_parent')}
    if set(visible)!=set(by_id) or any(execution_contract(by_id[k])!=execution_contract(visible[k]) for k in by_id):
        raise ValueError('current approved line differs from checkpoint approval')
    for key,item in by_id.items():
        if key!=step and execution_contract(item)!=execution_contract(state['nodes'][key]['contract']):
            raise ValueError('other approved execution contracts changed')
    requested=by_id[step];actual=node.get('effective_arguments')
    if not isinstance(actual,dict):raise ValueError('executed arguments missing')
    args=requested.get('arguments')or{}
    changed={key for key in set(args)|set(actual) if args.get(key)!=actual.get(key)}
    canonical=execution_contract(node['contract']);aligned=execution_contract(requested);aligned['arguments']=copy.deepcopy(actual)
    oldargs=canonical.pop('arguments');newargs=aligned.pop('arguments')
    if (changed!={'resource_review_id'} or not actual.get('resource_review_id')
            or canonical!=aligned
            or {k:v for k,v in oldargs.items()if k!='resource_review_id'}!={k:v for k,v in newargs.items()if k!='resource_review_id'}):
        raise ValueError('proposal changes science, outputs or actual allocation quantities')
    candidate=copy.deepcopy(requested);candidate['arguments']=copy.deepcopy(actual)
    proposed=[candidate if item['step_id']==step else item for item in incoming]
    identity={'owner':list(owner),'step_id':step,'token':node['token'],'attempt_id':node.get('attempt_id'),
        'job_ids':node['job_ids'],'goal_sha256':_digest(goal),'runtime_plan':state['plan_version'],
        'runtime_goal_sha256':_digest(saved),'node_sha256':_digest(node),
        'line_sha256':_digest(line),'approval_sha256':_digest(approval),'proposed_nodes_sha256':_digest(proposed),
        'proposed_contracts_sha256':_digest({n['step_id']:execution_contract(n)for n in proposed})}
    return {'goal':goal,'proposed_nodes':proposed,'candidate':candidate,'identity':identity,
        'control':_control(main,owner),'line':line,'approval':approval}


def _key(captured,epoch):
    return _digest({'epoch':epoch,'token':captured['identity']['token'],
        'owner':captured['identity']['owner'],'approval':captured['identity']['goal_sha256']})


def _fresh_jobs(runtime,node):
    jobs={str(j):runtime.job_watch.get(str(j))or{} for j in node['job_ids']} if runtime.job_watch else {}
    return all((job.get('username'),job.get('conv_id'))==(runtime.username,runtime.conv_id)
        and str(job.get('job_id'))==jid and job.get('attempt_id')==node.get('attempt_id')
        and job.get('tool',node['contract']['tool'])==node['contract']['tool']
        and job.get('terminal')is True and job.get('failed')is False and job.get('state')=='COMPLETED'
        for jid,job in jobs.items()) and len(jobs)==len(node['job_ids'])


def notify_metadata_capability(runtime,observed_state=None):
    """Native lifecycle notice once per code epoch/token/current approval."""
    from .workflow_metadata_alignment import prove_forward_transition
    from .native_resource_carry import proof_matches
    from .workflow_user_control import workflow_control_guard
    from .state_io import json_transaction
    state=runtime.snapshot()if observed_state is None else observed_state
    main=runtime.main;control=workflow_control_guard(main,runtime.username,runtime.conv_id)
    if not control() or main._interrupt_requested():return []
    created=[]
    for step,node in state.get('nodes',{}).items():
        if node.get('status')!='prefinish':continue
        try:
            captured=_candidate(runtime,state,step);epoch=source_epoch();key=_key(captured,epoch['epoch'])
            event_id='runtime:'+runtime.workflow_id+':'+node['token']+':metadata-ready:'+key
            if event_id in state.get('outbox',{}):continue
            proof=prove_forward_transition(runtime,captured['proposed_nodes'],captured['goal']['approved_plan_version']+1,[step])
            if step not in proof.get('covered_terminal_nodes',[]):continue
            fresh=_candidate(runtime,runtime.snapshot(),step)
            if (not control() or main._interrupt_requested() or runtime.main is not main
                    or fresh!=captured or source_epoch()!=epoch):continue
            forward=copy.deepcopy(captured['goal']);forward.update(approved_plan_version=captured['goal']['approved_plan_version']+1,approved_nodes=captured['proposed_nodes'])
            if (not _fresh_jobs(runtime,runtime.snapshot()['nodes'][step])
                    or any(not proof_matches(runtime.snapshot(),key,next(n for n in captured['proposed_nodes']if n['step_id']==key),runtime.root,forward,value)
                        for key,value in proof.get('proofs',{}).items())):continue
            recovery={'action':_ACTION,'tool':'propose_workflow_patch','arguments':{
                'base_version':captured['goal']['approved_plan_version'],
                'reason':'Align only receipt metadata to the original proved terminal executed allocation; preserve science and the original job.',
                'changes':[{'operation':'upsert','step_id':step,'node':captured['candidate']}]},
                'candidate_verified':True,'alignment_proof':proof,'source_epoch':epoch,
                'binding':captured['identity'],'native_control':captured['control'],
                'retry_granted':False,'applied':False,'scientific_acceptance':False,
                'instruction':'先读取lifecycle_state核对本事件及当前原生状态，再由Main调用本事件公开propose_workflow_patch入口检查精确候选；后续按普通批准/execute_workflow/原生验收流程推进。旧ACTIVE技术拒绝仅属于历史提案，不能证明本事件的当前已验证元数据候选仍不可用；不要只把旧拒绝委派给无写权限operations。这里没有批准、自动apply、重试许可、科学成功或新计算，不向用户重复索要已有条件。'}
            payload={'step_id':step,'correlation_token':node['token'],'attempt_id':node.get('attempt_id'),
                'requires_user':False,'runtime_id':runtime.workflow_id,'workflow_id':main._current_line_id,
                'origin_goal_version':captured['goal']['version'],'plan_version':state['plan_version'],'recovery':recovery}
            with json_transaction(runtime.store.path)as data:
                current=data.get('workflows',{}).get(runtime.workflow_id,{})
                if (not control() or runtime.main is not main or main._interrupt_requested()
                        or _candidate(runtime,current,step,line=captured['line'])!=captured or source_epoch()!=epoch):continue
                outbox=current.setdefault('outbox',{})
                if event_id in outbox:continue
                outbox[event_id]={'event_id':event_id,'kind':'worker_recovery_ready','payload':copy.deepcopy(payload),'sent':False}
            runtime.flush_events();created.append(event_id)
        except (ValueError,PermissionError,OSError,TypeError,KeyError,AttributeError):continue
    return created


def metadata_event_current(runtime,event):
    """Fresh source, owner, controls, approval, token, jobs and native files."""
    from .native_resource_carry import proof_matches
    recovery=(event.get('payload')or{}).get('recovery')or{}
    if recovery.get('action')!=_ACTION:return True
    try:
        state=runtime.snapshot();step=event['payload']['step_id'];captured=_candidate(runtime,state,step)
        epoch=source_epoch();node=state['nodes'][step]
        expected='runtime:'+runtime.workflow_id+':'+node['token']+':metadata-ready:'+_key(captured,epoch['epoch'])
        stored=state.get('outbox',{}).get(expected,{})
        proof=recovery['alignment_proof'];forward=copy.deepcopy(captured['goal'])
        forward.update(approved_plan_version=captured['goal']['approved_plan_version']+1,approved_nodes=captured['proposed_nodes'])
        return bool(event.get('kind')=='worker_recovery_ready' and event.get('event_id')==expected
            and stored.get('kind')==event.get('kind') and stored.get('payload')==event.get('payload')
            and recovery.get('binding')==captured['identity'] and recovery.get('native_control')==captured['control']
            and recovery.get('source_epoch')==epoch and _fresh_jobs(runtime,node)
            and recovery.get('retry_granted')is False and recovery.get('applied')is False
            and recovery.get('scientific_acceptance')is False and proof.get('no_dispatch')is True
            and step in proof.get('covered_terminal_nodes',[])
            and all(proof_matches(state,key,next(n for n in captured['proposed_nodes']if n['step_id']==key),
                runtime.root,forward,value) for key,value in proof.get('proofs',{}).items()))
    except (ValueError,PermissionError,OSError,TypeError,KeyError,AttributeError,StopIteration):return False


def orient_metadata_event(session,event):
    """Reprove on Main admission and label only the old ACTIVE technical block."""
    recovery=(event.get('payload')or{}).get('recovery')or{}
    if recovery.get('action')!=_ACTION:return None
    from .workflow_metadata_alignment import session_transition_proof
    from .workflow_patch import execution_contract
    from .watch_context import get_context
    owner=get_context();scope=(owner.get('username'),owner.get('conv_id'))
    state=session._runtime_snapshot()if callable(session._runtime_snapshot)else{}
    stored=state.get('outbox',{}).get(event.get('event_id'),{})
    if (stored.get('kind')!=event.get('kind') or stored.get('payload')!=event.get('payload')
            or recovery.get('binding',{}).get('node_sha256')!=_digest(state.get('nodes',{}).get(event.get('payload',{}).get('step_id'),{}))):
        raise ValueError('metadata recovery notice is not the current native outbox event')
    if (recovery.get('source_epoch')!=source_epoch() or recovery.get('native_control')!=_control(session,scope)
            or recovery.get('binding',{}).get('goal_sha256')!=_digest(session.goal_contract.to_dict())):
        raise ValueError('metadata recovery notice superseded before Main admission')
    arguments=recovery['arguments'];step=event['payload']['step_id']
    proposed=copy.deepcopy(session.goal_contract.approved_nodes)
    replacement=arguments['changes'][0]
    if replacement.get('operation')!='upsert' or replacement.get('step_id')!=step:raise ValueError('metadata proposal identity differs')
    proposed=[copy.deepcopy(replacement['node']) if n['step_id']==step else n for n in proposed]
    proof=session_transition_proof(session,proposed,[step],session.goal_contract.approved_plan_version+1)
    if step not in proof.get('covered_terminal_nodes',[]):raise ValueError('metadata capability no longer proved')
    branches=[]
    for bid,branch in session._error_branches.items():
        params=branch.get('failed_arguments')or{}
        if (branch.get('status')=='open' and branch.get('failed_step')=='propose_workflow_patch'
                and branch.get('error')=='affected runtime node still owns running/unknown resources; reconcile before patching'
                and type(params.get('base_version'))is int and params['base_version']<arguments['base_version']
                and any(c.get('step_id')==step for c in params.get('changes',[]))):branches.append(bid)
    from .task_line import get_store
    inherited_line=get_store().get_line(session._current_line_id,username=scope[0],conv_id=scope[1])or{}
    if _digest(inherited_line)!=recovery['binding']['line_sha256']:raise ValueError('metadata original approval line changed')
    return {'event_id':event['event_id'],'step_id':step,'source_epoch':recovery['source_epoch']['epoch'],
        'current_approval_sha256':recovery['binding']['goal_sha256'],'native_control':recovery['native_control'],
        'historical_active_branches':branches,'instruction':recovery['instruction'],
        'native_event':copy.deepcopy(event),'inherited_goal':copy.deepcopy(session.goal_contract.to_dict()),
        'inherited_approval':copy.deepcopy(session.context.get('approved_execution_proof')),'inherited_line':inherited_line,
        'retry_granted':False,'scientific_acceptance':False}


def metadata_application_authority(session,patch,plan_version):
    """A model-selected exact metadata patch may inherit its real prior approval.

    Lifecycle provenance stays lifecycle. This is no new science permission,
    and cannot authorize any other patch or replenish a scientific retry budget.
    """
    from .workflow_metadata_alignment import session_transition_proof
    from .workflow_patch import execution_contract
    from .parallel_workflow import _valid_execution_approval
    from .watch_context import get_context
    from .workflow_user_control import workflow_control_guard
    window=session.context.get('native_metadata_capability_window')or{}
    event=window.get('native_event')or{};recovery=(event.get('payload')or{}).get('recovery')or{}
    owner=get_context();scope=(owner.get('username'),owner.get('conv_id'));source=session._operation_control_source()or{}
    prior=window.get('inherited_approval');goal=window.get('inherited_goal')or{};line=window.get('inherited_line')or{}
    if (getattr(session.current_agent,'name','')!='lead-orchestrator' or not patch
            or source.get('origin')!='lifecycle' or source.get('turn_id')!=event.get('event_id')
            or source.get('owner')!={'username':scope[0],'conv_id':scope[1]}
            or source.get('goal_version')!=goal.get('version')
            or not isinstance(prior,dict)or prior.get('kind')not in {'main_model_applied_plan','explicit_user_parameter_patch'}
            or not _valid_execution_approval(prior,goal,line,*scope)
            or recovery.get('source_epoch')!=source_epoch()
            or recovery.get('native_control')!=_control(session,scope)
            or patch.get('base_version')!=goal.get('approved_plan_version')
            or patch.get('new_version')!=plan_version or plan_version!=goal['approved_plan_version']+1
            or _digest({n['step_id']:execution_contract(n)for n in patch.get('proposed_nodes',[])})!=recovery.get('binding',{}).get('proposed_contracts_sha256')
            or len(patch.get('changes',[]))!=1
            or patch['changes'][0].get('operation')!='upsert'
            or patch['changes'][0].get('step_id')!=recovery['arguments']['changes'][0]['step_id']
            or execution_contract(patch['changes'][0].get('node',{}))!=execution_contract(recovery['arguments']['changes'][0]['node'])):
        raise PermissionError('lifecycle may apply only its current proved metadata candidate under the unchanged prior user approval')
    expected=copy.deepcopy(goal);expected['pending_plan_version']=plan_version
    if session.goal_contract.to_dict()!=expected:
        raise PermissionError('lifecycle metadata proposal changed the original authorized task')
    state=session._runtime_snapshot()if session._runtime_snapshot else{}
    stored=state.get('outbox',{}).get(event.get('event_id'),{})
    step=event.get('payload',{}).get('step_id')
    if (stored.get('kind')!=event.get('kind')or stored.get('payload')!=event.get('payload')
            or _digest(state.get('nodes',{}).get(step,{}))!=recovery['binding']['node_sha256']
            or _digest(state.get('goal_contract'))!=recovery['binding']['runtime_goal_sha256']
            or state.get('plan_version')!=recovery['binding']['runtime_plan'] or state.get('user_paused')):
        raise PermissionError('lifecycle metadata attempt, receipt or native state changed')
    proof=session_transition_proof(session,patch['proposed_nodes'],patch['affected_nodes'],plan_version)
    if step not in proof.get('covered_terminal_nodes',[]):raise PermissionError('current metadata application proof incomplete')
    receipt={'schema_version':1,'kind':'inherited_terminal_receipt_metadata','event_id':event['event_id'],
        'step_id':step,'source_epoch':recovery['source_epoch'],'native_control':recovery['native_control'],
        'original_goal':copy.deepcopy(goal),'original_approval':copy.deepcopy(prior),
        'original_line':{k:copy.deepcopy(line.get(k))for k in ('username','conv_id','plan_version','plan_history')},
        'proposed_nodes_sha256':_digest(patch['proposed_nodes']),'alignment_proof':proof,
        'no_dispatch':True,'retry_granted':False,'scientific_acceptance':False}
    control=workflow_control_guard(session,*scope);uid=getattr(session,'_api_turn_id',None)
    def valid():
        # Safe inside TaskLine/WorkflowStore transactions: no ledger or native
        # store reads here. Their existing identity/proof CAS remains in force.
        current=session.goal_contract.to_dict();post=copy.deepcopy(expected)
        post.update(approved_nodes=copy.deepcopy(patch['proposed_nodes']),approved_plan_version=plan_version,
            pending_plan_version=plan_version,execution_authorized=True)
        comparable=lambda value:{**value,'approved_nodes':[execution_contract(n)for n in value.get('approved_nodes',[])]}
        return bool(control()and not session._interrupt_requested()and getattr(session,'_api_turn_id',None)==uid
            and getattr(session.current_agent,'name','')=='lead-orchestrator'
            and session.context.get('native_metadata_capability_window')is window
            and session._operation_control_source()==source
            and source_epoch()==receipt['source_epoch']
            and (current==expected or comparable(current)==comparable(post)))
    if not valid():raise PermissionError('lifecycle metadata control changed before application')
    return receipt,valid


def metadata_execution_approval(proof,goal,line,username,conv_id):
    """Pure provenance check, also safe within the existing native store CAS."""
    from .parallel_workflow import _valid_execution_approval, _approval_digest
    from .workflow_patch import execution_contract
    from .native_resource_carry import _hash, _goal_scope
    try:
        record=proof['metadata_continuation'];prior=record['original_approval'];original=record['original_goal']
        source=proof['control_source'];owner={'username':username,'conv_id':conv_id}
        if (proof.get('schema_version')!=1 or proof.get('kind')!='native_metadata_continuation'
                or proof.get('owner')!=owner or proof.get('goal_version')!=goal.get('version')
                or proof.get('plan_version')!=goal.get('approved_plan_version')
                or proof.get('base_plan_version')!=original.get('approved_plan_version')
                or proof['plan_version']!=proof['base_plan_version']+1
                or proof.get('approval_sha256')!=_approval_digest(goal)
                or (line.get('username'),line.get('conv_id'))!=(username,conv_id)
                or line.get('plan_version')!=proof['plan_version']
                or source.get('origin')!='lifecycle' or source.get('owner')!=owner
                or source.get('goal_version')!=goal.get('version')or source.get('turn_id')!=record.get('event_id')
                or prior.get('kind')not in {'main_model_applied_plan','explicit_user_parameter_patch'}
                or not _valid_execution_approval(prior,original,record['original_line'],username,conv_id)
                or record.get('kind')!='inherited_terminal_receipt_metadata' or record.get('no_dispatch')is not True
                or record.get('retry_granted')is not False or record.get('scientific_acceptance')is not False
                or any(original.get(k)!=goal.get(k)for k in ('version','method','gases','parameters','execution_mode','execution_authorized'))):return False
        before={n['step_id']:execution_contract(n)for n in original['approved_nodes']}
        after={n['step_id']:execution_contract(n)for n in goal['approved_nodes']}
        if set(before)!=set(after)or not 0<len(before)<=512:return False
        step=record['step_id']
        if step not in before or any(before[k]!=after[k]for k in before if k!=step):return False
        left=copy.deepcopy(before[step]);right=copy.deepcopy(after[step]);a=left.pop('arguments');b=right.pop('arguments')
        if left!=right or {k for k in set(a)|set(b)if a.get(k)!=b.get(k)}!={'resource_review_id'}:return False
        alignment=record['alignment_proof'];native=alignment['proofs'][step]
        return bool(alignment.get('no_dispatch')is True and alignment.get('covered_terminal_nodes')==[step]
            and alignment.get('proposed_plan_version')==proof['plan_version']
            and native.get('owner')==owner and native.get('step_id')==step
            and native.get('incoming_contract')==after[step] and native.get('goal_sha256')==_hash(_goal_scope(goal))
            and native.get('proof_sha256')==_hash({k:v for k,v in native.items()if k!='proof_sha256'}))
    except (ValueError,TypeError,KeyError,AttributeError):return False
