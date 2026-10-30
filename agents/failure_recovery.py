"""Persistent failure incidents and bounded role stages, never retry authority.

Native observation and stage/outbox updates share the WorkflowStore transaction.
A diagnostic capability change wakes a *read-only* role even when an old
scientific proposal is waiting. It cannot clear that proposal, pause or goal.
"""
import copy
from functools import lru_cache
import hashlib
import json
import math
import os
import uuid
from pathlib import Path
import re
import time
import types

from .state_io import json_transaction
from .workflow_patch import execution_contract
from .recovery import result_object

FAILURE_STATES = frozenset({'failed','validation_failed'})
PHASE_ROLES = {'diagnosis_required':'diagnostician','repair_required':'repair_planner',
    'verification_required':'repair_verifier','resource_review_required':'resource_reviewer',
    'execution_ready':'operations','validating':'scientific_owner'}


def _digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def _code(code):
    if isinstance(code,types.CodeType):
        return [code.co_code.hex(),list(code.co_names),list(code.co_varnames),[_code(v) for v in code.co_consts]]
    if isinstance(code,(tuple,list)):return [_code(v) for v in code]
    if isinstance(code,(set,frozenset)):return sorted((_code(v) for v in code),key=repr)
    if isinstance(code,bytes):return ['bytes',code.hex()]
    if code is None or type(code) in (str,int,float,bool):return code
    return type(code).__name__


@lru_cache(maxsize=32)
def _revision(codes, tool):
    return _digest({'schema':1,'tool':tool,'implementations':[_code(code)for code in codes]})


def diagnostic_revision(tool):
    """Hash loaded diagnostic capability, not unrelated live source files.

    No filesystem-wide search, scientific execution or provider call occurs.
    Reloading a genuinely changed inspector creates a new diagnostic epoch.
    """
    from . import registry,result_validation,output_contract,installations
    functions=[registry._exec_diagnose_job,installations.inspect_installation,
        result_validation._profile_results,output_contract.artifact_is_valid]
    if tool=='run_cdft':
        from .cdft_failure_facts import inspect_failed_cdft_attempt
        from .cdft_result_facts import inspect_cdft_batch_quality
        functions.extend([inspect_failed_cdft_attempt,inspect_cdft_batch_quality])
    if tool=='generate_scientific_report':
        # A repaired report fact adapter or reviewer changes diagnostic
        # capability even when scheduler diagnostics are byte-for-byte equal.
        # This only wakes an owned read-only incident; it grants no retry.
        from . import report_authoring,report_source_metadata,literature_report_bridge
        functions.extend([report_authoring.build_result_packet,
            report_authoring.render_validated_draft,report_authoring.author_report,
            report_authoring._source_material_labels,report_authoring._mask_source_material_labels,
            report_authoring._nonquantitative_literature_denial,
            report_source_metadata.cdft_document,report_source_metadata.metadata_tables,
            literature_report_bridge.excerpt_block])
    functions.extend([FailureRecovery._source,FailureRecovery.run_diagnosis])
    return _revision(tuple(f.__code__ for f in functions),tool)


def _node_identity(state,step):
    node=state.get('nodes',{}).get(step,{})
    return {key:copy.deepcopy(node.get(key)) for key in ('contract','status','token','attempt_id',
        'recovery_key','job_ids','tool_returned','dispatch_phase','started_at','effective_arguments',
        'result_ref','result','validation_dossier')}


def _same_approved_goal_with_continuations(saved, goal):
    """Read-only diagnostics may accept only proved same-version resumes."""
    if not isinstance(saved,dict) or not isinstance(goal,dict):return False
    from .report_recovery import _same
    try:
        if _same(saved,goal):return True
        if not _same({k:v for k,v in saved.items()if k!='directives'},
                     {k:v for k,v in goal.items()if k!='directives'}):return False
    except (ValueError,TypeError):return False
    old=saved.get('directives');new=goal.get('directives')
    if not isinstance(old,list)or not isinstance(new,list)or len(new)<=len(old) or new[:len(old)]!=old:
        return False
    from .goal_contract import scope_preserving_continuation
    return all(isinstance(text,str) and scope_preserving_continuation(text,goal.get('approved_nodes')or[])
               for text in new[len(old):])


class FailureRecovery:
    """Native-only coordinator. Public tools cannot supply verdicts or raw state."""
    def __init__(self,runtime):
        self.runtime=runtime
        self.owner={'username':runtime.username,'conv_id':runtime.conv_id}

    def _authority(self,state,step):
        rt=self.runtime;goal=rt.main.goal_contract.to_dict();node=state.get('nodes',{}).get(step,{})
        approved=[n for n in goal.get('approved_nodes',[]) if n.get('step_id')==step]
        token=node.get('token');future=rt.futures.get(step)
        return (state.get('username')==rt.username and state.get('conv_id')==rt.conv_id
            and not state.get('user_paused') and state.get('status')in {'active','needs_user'}
            and _same_approved_goal_with_continuations(state.get('goal_contract'),goal) and goal.get('execution_authorized')is True
            and state.get('plan_version')==goal.get('approved_plan_version') and len(approved)==1
            and execution_contract(approved[0])==execution_contract(node.get('contract',{}))
            and node.get('status')in FAILURE_STATES and node.get('tool_returned')is True
            and node.get('dispatch_phase')=='entered_tool'
            and isinstance(token,str) and re.fullmatch('[0-9a-f]{32}',token)
            and (future is None or future.done()))

    def _source(self,state,step):
        from .report_recovery import _OwnedFiles
        from .local_preparation_recovery import checkpoint_call_witness
        rt=self.runtime;node=state['nodes'][step];ref=node.get('result_ref')or{}
        cid=ref.get('call_id');args=node.get('effective_arguments')or node['contract'].get('arguments',{})
        if not isinstance(cid,str)or not re.fullmatch('[A-Za-z0-9_-]{1,128}',cid):
            raise ValueError('failure has no owned native producer')
        files=_OwnedFiles(rt.root,self.owner)
        try:
            expected=files.root/'evidence'/(cid+'.json')
            if ref.get('evidence_path')!=str(expected):raise ValueError('failure producer is not owned')
            call,fact=files.read(expected,as_json=True)
            stamp=call.get('time')
            if (call.get('call_id')!=cid or call.get('tool')!=node['contract'].get('tool')
                    or _digest(call.get('params'))!=_digest(args)
                    or _digest(result_object(call.get('result')))!=_digest(result_object(node.get('result')))
                    or type(stamp)not in (int,float)or not math.isfinite(stamp)
                    or stamp<node.get('started_at',float('inf')) or stamp>time.time()+1):
                raise ValueError('failure return does not belong to this native attempt')
            witness=checkpoint_call_witness(files,ref.get('checkpoint_path'),call,step_id=step)
            jobs=[]
            for value in node.get('job_ids')or[]:
                job=rt.job_watch.get(str(value)) if rt.job_watch else None
                if (not isinstance(job,dict) or (job.get('username'),job.get('conv_id'))!=(rt.username,rt.conv_id)
                        or str(job.get('job_id'))!=str(value)or job.get('attempt_id')!=node.get('attempt_id')
                        or job.get('recovery_key')!=node.get('recovery_key')or job.get('tool')!=node['contract'].get('tool')
                        or job.get('terminal')is not True):raise ValueError('failure job is not this owned terminal attempt')
                jobs.append({key:job.get(key)for key in ('job_id','attempt_id','recovery_key','state','terminal','failed')})
            if len(jobs)>8:raise ValueError('failure diagnostic job bound exceeded')
            dossier=node.get('validation_dossier')or{}
            return {'call_id':cid,'evidence_path':str(expected),'call_sha256':fact['sha256'],
                'checkpoint_path':witness['path'],'checkpoint_sha256':witness['sha256'],
                'jobs':jobs,'dossier':{key:dossier.get(key)for key in ('evidence_call_id','verdict','schema_version')}}
        finally:files.close()

    def observe(self,step_id,*,observed_state=None):
        """Persist one current diagnostic epoch and event, without changing a node."""
        rt=self.runtime;state=observed_state if observed_state is not None else rt.snapshot()
        if not self._authority(state,step_id):return None
        try:
            revision=diagnostic_revision(state['nodes'][step_id]['contract']['tool'])
            signature=_digest(_node_identity(state,step_id))
            # Negative-only prefilter: native evidence is immutable per call ID.
            # Later stage claims still authenticate and rehash their witnesses.
            for existing in (state.get('failure_recovery')or{}).get('incidents',{}).values():
                if (existing.get('step_id')==step_id and existing.get('token')==state['nodes'][step_id].get('token')
                        and existing.get('diagnostic_revision')==revision
                        and existing.get('observation_signature')==signature):return None
            proof=self._source(state,step_id)
        except (OSError,ValueError,KeyError,TypeError,PermissionError,AttributeError):return None
        node=state['nodes'][step_id];identity=_node_identity(state,step_id)
        incident_id=_digest({'owner':self.owner,'workflow_id':rt.workflow_id,'step_id':step_id,
            'token':node['token'],'attempt_id':node.get('attempt_id'),
            'goal_version':state['goal_contract']['version'],'plan_version':state['plan_version'],
            'contract':execution_contract(node['contract'])})[:32]
        epoch=_digest({'diagnostic_revision':revision,'source':proof})
        previous=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id)
        if previous and previous.get('epoch')==epoch:return None
        with json_transaction(rt.store.path) as data:
            current=data.get('workflows',{}).get(rt.workflow_id,{})
            if (not self._authority(current,step_id)or _node_identity(current,step_id)!=identity
                    or current.get('goal_contract')!=state.get('goal_contract')):return None
            # Bounded native receipts are hashed again, never stat-only.
            try:again=self._source(current,step_id)
            except (OSError,ValueError,KeyError,TypeError,PermissionError):return None
            if again!=proof or not self._authority(current,step_id):return None
            book=current.setdefault('failure_recovery',{'schema_version':1,'incidents':{}})
            records=book['incidents'];old=records.get(incident_id)
            if old and old.get('epoch')==epoch:return None
            history=copy.deepcopy(old.get('history',[]))if old else []
            if old:history.append({key:copy.deepcopy(old.get(key))for key in
                ('epoch','diagnostic_revision','phase','revision','evidence','rounds','last_result')})
            incident={'incident_id':incident_id,'owner':dict(self.owner),'workflow_id':rt.workflow_id,
                'step_id':step_id,'token':node['token'],'attempt_id':node.get('attempt_id'),
                'job_ids':list(map(str,node.get('job_ids')or[])),'goal_version':state['goal_contract']['version'],
                'plan_version':state['plan_version'],'contract_sha256':_digest(execution_contract(node['contract'])),
                'phase':'diagnosis_required','diagnostic_revision':revision,'epoch':epoch,'observation_signature':signature,
                'revision':(old or{}).get('revision',0)+1,'event_revision':(old or{}).get('revision',0)+1,'evidence':proof,'history':history,
                'read_only':True,'rounds':{},'created_at':(old or{}).get('created_at',time.time()),'updated_at':time.time()}
            event_id='failure:'+rt.workflow_id+':'+incident_id+':diagnosis:'+epoch
            calls=[{'tool':'diagnose_job','arguments':{'job_id':job}}for job in incident['job_ids']]
            if not calls:calls=[{'tool':'inspect_runtime','arguments':{'tool_name':node['contract']['tool'],
                'arguments':copy.deepcopy(node.get('effective_arguments')or node['contract'].get('arguments',{}))}}]
            payload={'step_id':step_id,'runtime_id':rt.workflow_id,'workflow_id':rt.main._current_line_id,
                'correlation_token':node['token'],'attempt_id':node.get('attempt_id'),
                'origin_goal_version':incident['goal_version'],'plan_version':incident['plan_version'],
                'requires_user':False,'failure_recovery':{'incident_id':incident_id,'epoch':epoch,
                    'revision':incident['revision'],'stage':'diagnosis','role':'diagnostician',
                    'diagnostic_calls':calls,'evidence':copy.deepcopy(proof),'retry_granted':False,
                    'preserve_pending_scientific_question':True,'preserve_scientific_goal':True}}
            records[incident_id]=incident
            current.setdefault('outbox',{}).setdefault(event_id,{'event_id':event_id,
                'kind':'worker_failure_recovery','payload':payload,'sent':False})
            return {'incident_id':incident_id,'event_id':event_id,'revision':incident['revision']}

    def _incident_current(self,state,incident):
        step=incident.get('step_id');node=state.get('nodes',{}).get(step,{})
        return (self._authority(state,step) and incident.get('owner')==self.owner
            and incident.get('workflow_id')==self.runtime.workflow_id
            and incident.get('token')==node.get('token')and incident.get('attempt_id')==node.get('attempt_id')
            and incident.get('goal_version')==state.get('goal_contract',{}).get('version')
            and incident.get('plan_version')==state.get('plan_version')
            and incident.get('contract_sha256')==_digest(execution_contract(node.get('contract',{}))))

    def current_event(self,event,*,observed_state=None):
        """Pure event/phase CAS against one native snapshot; never clears a question."""
        state=self.runtime.snapshot()if observed_state is None else observed_state
        payload=event.get('payload')or{};stage=payload.get('failure_recovery')or{}
        incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(stage.get('incident_id'),{})
        phases={'diagnosis':{'diagnosis_required','diagnosing'},'repair':{'repair_required','repairing'},
                'patch':{'repair_execution_required'},'verification':{'verification_required','verifying'},'resource_review':{'resource_review_required'},
                'execution':{'execution_ready'},'validation':{'validating'},'blocked':{'blocked'}}
        return bool(event.get('kind')=='worker_failure_recovery'and incident and self._incident_current(state,incident)
            and payload.get('runtime_id')==self.runtime.workflow_id
            and payload.get('step_id')==incident['step_id']
            and payload.get('correlation_token')==incident['token']
            and payload.get('attempt_id')==incident.get('attempt_id')
            and stage.get('epoch')==incident.get('epoch')
            and stage.get('revision')==incident.get('event_revision',incident.get('revision'))
            and incident.get('phase')in phases.get(stage.get('stage'),set()))

    def _stage_event(self,state,incident,stage,role,*,evidence_call_ids=()):
        rt=self.runtime;revision=incident['revision'];incident['event_revision']=revision
        event_id='failure:'+rt.workflow_id+':'+incident['incident_id']+':'+stage+':'+str(revision)+':'+incident['epoch']
        payload={'step_id':incident['step_id'],'runtime_id':rt.workflow_id,'workflow_id':rt.main._current_line_id,
            'correlation_token':incident['token'],'attempt_id':incident.get('attempt_id'),
            'origin_goal_version':incident['goal_version'],'plan_version':incident['plan_version'],'requires_user':False,
            'failure_recovery':{'incident_id':incident['incident_id'],'epoch':incident['epoch'],'revision':revision,
                'stage':stage,'role':role,'evidence_call_ids':list(evidence_call_ids),'retry_granted':False,
                'preserve_pending_scientific_question':True,'preserve_scientific_goal':True}}
        state.setdefault('outbox',{}).setdefault(event_id,{'event_id':event_id,'kind':'worker_failure_recovery',
            'payload':payload,'sent':False})
        return event_id

    def run_diagnosis(self,actor,incident_id,expected_revision,*,checkpoint_path,control_valid=None):
        """Claim and execute exact native readonly diagnostics, then persist repair work.

        actor/checkpoint_path/control_valid are internal API bindings, never model
        parameters. Success returns phase, next event_id and actual evidence IDs.
        Stale controls/claims or pause raise ValueError/PermissionError. Native
        exceptions leave a bounded diagnosis retry (max two per evidence epoch);
        no job, node success, retry permit or scientific question is altered.
        """
        from .watch_context import get_context,set_context
        from .report_recovery import _OwnedFiles
        from .local_preparation_recovery import checkpoint_call_witness
        rt=self.runtime
        if (getattr(actor,'_evidence_root',None)!=rt.root/'evidence'
                or Path(actor.config.project_root)!=Path(rt.main.config.project_root)
                or not callable(getattr(actor,'_on_checkpoint',None))):
            raise ValueError('diagnosis requires an owned native Session checkpoint binding')
        def control():
            if control_valid is not None and not control_valid():raise ValueError('failure recovery control was superseded')
        lease=uuid.uuid4().hex
        with json_transaction(rt.store.path) as data:
            state=data.get('workflows',{}).get(rt.workflow_id,{})
            incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
            if (not incident or not self._incident_current(state,incident)
                    or incident.get('event_revision',incident.get('revision'))!=expected_revision
                    or incident.get('phase')!='diagnosis_required'):
                raise ValueError('diagnostic stage is stale, already claimed, or no longer needed')
            if incident.get('rounds',{}).get('diagnosis',0)>=2:
                raise ValueError('bounded native diagnostic attempts exhausted; new evidence or verified repair required')
            control()
            incident.setdefault('rounds',{})['diagnosis']=incident.get('rounds',{}).get('diagnosis',0)+1
            incident.update(phase='diagnosing',revision=incident['revision']+1,
                stage_lease={'lease_id':lease,'role':'diagnostician','owner_pid':os.getpid(),'claimed_at':time.time(),'read_only':True})
            ticket=copy.deepcopy(incident);node=copy.deepcopy(state['nodes'][incident['step_id']])
        commands=[{'tool':'diagnose_job','arguments':{'job_id':job}}for job in ticket['job_ids']]
        if not commands:commands=[{'tool':'inspect_runtime','arguments':{'tool_name':node['contract']['tool'],
            'arguments':copy.deepcopy(node.get('effective_arguments')or node['contract'].get('arguments',{}))}}]
        previous=get_context();records=[];native_facts=[]
        try:
            set_context(rt.username,rt.conv_id,'supervisor',rt.main._current_line_id)
            for command in commands:
                control()
                raw=actor.registry.execute_dict(command['tool'],command['arguments'])
                raw=raw if isinstance(raw,dict)else {'result':raw}
                # `failed` in diagnose_job describes the JOB, not failed execution
                # of this readonly diagnostic. Never misclassify it as tool error.
                actor.memory.record_tool_call('supervisor',command['tool'],command['arguments'],
                    json.dumps(raw,ensure_ascii=False,allow_nan=False),failed=bool(raw.get('error')or raw.get('blocked')or raw.get('isError')))
                call_id=actor.memory.tool_call_log[-1]['call_id']
                actor._checkpoint('failure_recovery_diagnosis')
                files=_OwnedFiles(rt.root,self.owner)
                try:
                    call,fact=files.read(rt.root/'evidence'/(call_id+'.json'),as_json=True)
                    witness=checkpoint_call_witness(files,checkpoint_path,call,require_result_hash=True)
                    live=[r for r in actor.memory.tool_call_log if r.get('call_id')==call_id]
                    if (len(live)!=1 or live[0].get('result_sha256')!=hashlib.sha256(str(call.get('result','')).encode()).hexdigest()
                            or call.get('time',0)<ticket['stage_lease']['claimed_at']
                            or call.get('tool')!=command['tool']or _digest(call.get('params'))!=_digest(command['arguments'])
                            or _digest(result_object(call.get('result')))!=_digest(raw)):
                        raise ValueError('native diagnostic call does not match its actual Session witness')
                    records.append({'call_id':call_id,'tool':command['tool'],'params':copy.deepcopy(command['arguments']),
                        'evidence_path':fact['path'],'sha256':fact['sha256'],'checkpoint_path':witness['path'],
                        'result_sha256':live[0]['result_sha256'],'failed':call.get('failed')is True})
                    facts=raw.get('native_failure_facts')
                    if isinstance(facts,dict):native_facts.append({key:facts.get(key)for key in ('kind','proof_sha256','counts','advisor')})
                finally:files.close()
            with json_transaction(rt.store.path) as data:
                state=data.get('workflows',{}).get(rt.workflow_id,{})
                incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
                if (not incident or not self._incident_current(state,incident)
                        or incident.get('epoch')!=ticket['epoch']or (incident.get('stage_lease')or{}).get('lease_id')!=lease
                        or incident.get('phase')!='diagnosing'):
                    raise ValueError('diagnostic stage changed before native evidence commit')
                # Re-read actual diagnostic evidence after checkpoint compaction.
                files=_OwnedFiles(rt.root,self.owner)
                try:
                    for record in records:
                        call,fact=files.read(record['evidence_path'],as_json=True)
                        if fact['sha256']!=record['sha256']:raise ValueError('diagnostic bytes changed before stage commit')
                        checkpoint_call_witness(files,record['checkpoint_path'],call,require_result_hash=True)
                finally:files.close()
                control()
                if not self._incident_current(state,incident):raise ValueError('diagnostic authority changed during evidence IO')
                incident['phase']='blocked'if any(r['failed']for r in records)else 'repair_required'
                incident['revision']+=1;incident.pop('stage_lease',None)
                incident['last_result']={'evidence':records,'native_failure_facts':native_facts,
                    'retry_granted':False,'scientific_acceptance':False,'goal_unchanged':True}
                incident['updated_at']=time.time()
                phase='blocked'if incident['phase']=='blocked'else 'repair'
                event_id=self._stage_event(state,incident,phase,'main_chat'if phase=='blocked'else 'repair_planner',
                    evidence_call_ids=[r['call_id']for r in records])
            rt.flush_events()
            return {'phase':incident['phase'],'incident_id':incident_id,'event_id':event_id,
                'evidence_call_ids':[r['call_id']for r in records],'retry_granted':False,'read_only':True}
        except Exception as error:
            with json_transaction(rt.store.path) as data:
                state=data.get('workflows',{}).get(rt.workflow_id,{})
                incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
                if (incident.get('stage_lease')or{}).get('lease_id')==lease:
                    incident.setdefault('history',[]).append({'stage':'diagnosis','lease_id':lease,
                        'status':'failed_or_superseded','error_type':type(error).__name__,'evidence_call_ids':[r['call_id']for r in records]})
                    incident['phase']='diagnosis_required'if incident.get('rounds',{}).get('diagnosis',0)<2 else 'blocked'
                    incident.pop('stage_lease',None)
                    incident['revision']+=1
                    if incident['phase']=='blocked':
                        incident['blocked_reason']='bounded_native_diagnosis_exhausted'
                        self._stage_event(state,incident,'blocked','main_chat',evidence_call_ids=[r['call_id']for r in records])
            raise
        finally:set_context(**previous)

    def recover_stage_leases(self,*,observed_state=None):
        """Recover expired READONLY diagnosis/planning leases, never execution.

        A stale reader may finish, but its old lease cannot commit afterwards.
        The epoch's stage budget is retained, including across host/restart.
        """
        rt=self.runtime;state=rt.snapshot()if observed_state is None else observed_state
        if state.get('user_paused'):return []
        stages={'diagnosing':('diagnostician','diagnosis','diagnosis_required'),
            'repairing':('repair_planner','repair','repair_required')}
        now=time.time();candidates=[]
        for iid,incident in (state.get('failure_recovery')or{}).get('incidents',{}).items():
            lease=incident.get('stage_lease')or{};stamp=lease.get('claimed_at')
            stage=stages.get(incident.get('phase'))
            if (stage and lease.get('read_only')is True and lease.get('role')==stage[0]
                    and type(stamp)in (int,float)and math.isfinite(stamp)
                    and now-stamp>=120 and self._incident_current(state,incident)):
                candidates.append((iid,incident.get('phase'),copy.deepcopy(lease)))
        if not candidates:return []
        recovered=[]
        with json_transaction(rt.store.path)as data:
            current=data.get('workflows',{}).get(rt.workflow_id,{})
            for iid,phase,expected_lease in candidates:
                incident=(current.get('failure_recovery')or{}).get('incidents',{}).get(iid,{})
                lease=incident.get('stage_lease')or{}
                role,stage,required=stages[phase]
                if (not self._incident_current(current,incident)or incident.get('phase')!=phase
                        or lease!=expected_lease or lease.get('read_only')is not True
                        or lease.get('role')!=role):continue
                incident.setdefault('history',[]).append({'stage':stage,'status':'expired_readonly_lease',
                    'lease':copy.deepcopy(lease),'no_scientific_retry':True})
                incident.pop('stage_lease',None);incident['revision']+=1
                incident['phase']=required if incident.get('rounds',{}).get(stage,0)<2 else 'blocked'
                event_stage=stage if incident['phase']==required else 'blocked'
                if event_stage=='blocked':incident['blocked_reason']='bounded_native_'+stage+'_exhausted'
                self._stage_event(current,incident,event_stage,role if event_stage==stage else 'main_chat')
                recovered.append(iid)
        return recovered

    @staticmethod
    def _repair_permission_code(error):
        # Only these fixed internal messages are recorded. Never store raw
        # provider exceptions, paths, prompts, credentials or user content.
        if not isinstance(error, PermissionError):return None
        return {
            'repair planning cannot submit, finish, approve retry or mutate workflow':'repair_tool_not_allowed',
            'recovery root does not match its owned session scope':'recovery_owner_mismatch',
            'recovery evidence path escapes its owned session scope':'recovery_evidence_outside_scope',
            'missing recovery owner scope':'recovery_owner_missing',
            'native evidence call path does not match owned identity':'recovery_call_identity_mismatch',
        }.get(str(error), 'permission_denied_cause_unclassified')

    def run_repair(self,actor,incident_id,expected_revision,*,checkpoint_path,stage_runner=None,control_valid=None):
        """Run a bounded *planning* role and persist a real patch execution request.

        A proposal/model verdict or passing regression is not an applied repair.
        Actual patch execution, independent verification, resource review and
        existing retry/finish gates remain separate native stages.
        """
        from .watch_context import get_context,set_context
        from .report_recovery import _OwnedFiles
        from .local_preparation_recovery import checkpoint_call_witness
        rt=self.runtime
        if (actor is rt.main or getattr(actor,'_evidence_root',None)!=rt.root/'evidence'
                or Path(actor.config.project_root)!=Path(rt.main.config.project_root)
                or not callable(getattr(actor,'_on_checkpoint',None))):
            raise ValueError('repair planning requires an isolated owned native Session')
        runner=stage_runner if stage_runner is not None else getattr(actor,'run_failure_repair_stage',None)
        if not callable(runner):raise ValueError('native repair stage runner is unavailable')
        def control():
            if control_valid is not None and not control_valid():raise ValueError('repair planning control was superseded')
        lease=uuid.uuid4().hex
        with json_transaction(rt.store.path)as data:
            state=data.get('workflows',{}).get(rt.workflow_id,{})
            incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
            if (not incident or not self._incident_current(state,incident)
                    or incident.get('event_revision')!=expected_revision or incident.get('phase')!='repair_required'):
                raise ValueError('repair stage is stale or already claimed')
            if incident.get('rounds',{}).get('repair',0)>=2:raise ValueError('bounded repair planning exhausted')
            control()
            events=[e for e in state.get('outbox',{}).values()if self.current_event(e,observed_state=state)
                    and e.get('payload',{}).get('failure_recovery',{}).get('stage')=='repair'
                    and e.get('payload',{}).get('failure_recovery',{}).get('incident_id')==incident_id
                    and e.get('payload',{}).get('failure_recovery',{}).get('revision')==expected_revision]
            if len(events)!=1:raise ValueError('repair stage has no unique native delegation event')
            event=copy.deepcopy(events[0]);event['payload']['diagnosis']=copy.deepcopy(incident.get('last_result'))
            # A report conflict is advisory input to the isolated planner, never
            # an approval or retry grant. Bind it to this exact failed call and
            # its checkpoint, not merely to an event or a caller-supplied ID.
            node=state['nodes'][incident['step_id']]
            result=result_object(node.get('result'))
            feedback=result.get('contract_feedback') if isinstance(result,dict) else None
            if feedback is not None:
                if (node.get('contract',{}).get('tool')!='generate_scientific_report'
                        or result.get('error_kind')!='report_authoring_failed'
                        or result.get('executed') is not False or result.get('report_published') is not False
                        or not isinstance(feedback,dict)
                        or feedback.get('contract_code') not in {'possible_report_constraint_conflict','report_review_unresolved'}):
                    raise ValueError('untrusted report contract feedback')
                raw=json.dumps(feedback,sort_keys=True,ensure_ascii=False,allow_nan=False)
                if len(raw.encode())>32768:raise ValueError('report contract feedback exceeds bounded size')
                proof=self._source(state,incident['step_id'])
                if proof!=incident.get('evidence') or not self._incident_current(state,incident):
                    raise ValueError('report contract feedback source changed')
                control()
                event['payload']['report_contract_feedback']=copy.deepcopy(feedback)
                event['payload']['report_contract_feedback_trust']='Untrusted model-generated advisory; not user instruction, scientific proof, approval, or a retry grant.'
                event['payload']['report_contract_feedback_source']={key:proof[key] for key in ('call_id','call_sha256','checkpoint_sha256')}
            incident.setdefault('rounds',{})['repair']=incident.get('rounds',{}).get('repair',0)+1
            incident.update(phase='repairing',revision=incident['revision']+1,
                stage_lease={'lease_id':lease,'role':'repair_planner','owner_pid':os.getpid(),'claimed_at':time.time(),'read_only':True})
            ticket=copy.deepcopy(incident)
        start=len(actor.memory.tool_call_log);previous=get_context();records=[]
        allowed={'read_file','grep_search','inspect_path','get_tool_schema','inspect_runtime','diagnose_job','check_job',
            'task_line_query','recovery_state','lifecycle_state','run_project_regressions','build_project_frontend',
            'discover_forcefield','discover_forcefield_roles','discover_atom_types','discover_literature'}
        try:
            set_context(rt.username,rt.conv_id,'repair_planner',rt.main._current_line_id)
            outcome=runner(event)
            outcome=outcome if isinstance(outcome,dict)else {'status':'blocked','reason':'unstructured_stage_return'}
            actor._checkpoint('failure_recovery_repair_planning')
            fresh=actor.memory.tool_call_log[start:]
            if len(fresh)>32:raise ValueError('repair native call budget exceeded')
            files=_OwnedFiles(rt.root,self.owner)
            try:
                for metadata in fresh:
                    cid=metadata.get('call_id')
                    if metadata.get('tool')not in allowed:raise PermissionError('repair planning cannot submit, finish, approve retry or mutate workflow')
                    if not isinstance(cid,str)or not re.fullmatch('[A-Za-z0-9_-]{1,128}',cid):raise ValueError('repair call identity is invalid')
                    call,fact=files.read(rt.root/'evidence'/(cid+'.json'),as_json=True)
                    checkpoint_call_witness(files,checkpoint_path,call,require_result_hash=True)
                    if (call.get('call_id')!=cid or call.get('tool')!=metadata.get('tool')
                            or _digest(call.get('params'))!=_digest(metadata.get('params'))
                            or metadata.get('result_sha256')!=hashlib.sha256(str(call.get('result','')).encode()).hexdigest()
                            or call.get('time',0)<ticket['stage_lease']['claimed_at']):
                        raise ValueError('repair source does not match its actual Session call')
                    records.append({'call_id':cid,'tool':call['tool'],'params':copy.deepcopy(call.get('params')),
                        'evidence_path':fact['path'],'sha256':fact['sha256'],'checkpoint_path':str(checkpoint_path),
                        'result_sha256':metadata['result_sha256'],'failed':call.get('failed')is True})
            finally:files.close()
            raw_proposal=outcome.get('proposal')or{};request=raw_proposal.get('request')if isinstance(raw_proposal,dict)else None
            proposed=(outcome.get('status')=='proposed'and any(not r['failed']for r in records)
                and isinstance(request,str)and bool(request.strip())and len(request)<=6000)
            proposal={'target_role':'patcher','request':request,'execution_required':True}if proposed else None
            with json_transaction(rt.store.path)as data:
                state=data.get('workflows',{}).get(rt.workflow_id,{})
                incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
                if (not incident or not self._incident_current(state,incident)or incident.get('epoch')!=ticket['epoch']
                        or (incident.get('stage_lease')or{}).get('lease_id')!=lease or incident.get('phase')!='repairing'):
                    raise ValueError('repair delegation changed before commit')
                files=_OwnedFiles(rt.root,self.owner)
                try:
                    for record in records:
                        call,fact=files.read(record['evidence_path'],as_json=True)
                        if fact['sha256']!=record['sha256']:raise ValueError('repair evidence changed before stage commit')
                        checkpoint_call_witness(files,checkpoint_path,call,require_result_hash=True)
                finally:files.close()
                control()
                if not self._incident_current(state,incident):raise ValueError('repair authority changed during evidence IO')
                incident.update(phase='repair_execution_required'if proposed else 'blocked',revision=incident['revision']+1)
                incident.pop('stage_lease',None)
                incident['repair_plan']={'planning_only':True,'proposal':proposal,'evidence':records,
                    'repair_applied':False,'repair_verified':False,'retry_granted':False,
                    'blocked_reason':None if proposed else 'no_verified_repair_execution_proposal'}
                next_stage='patch'if proposed else 'blocked'
                event_id=self._stage_event(state,incident,next_stage,'patcher'if proposed else 'main_chat',
                    evidence_call_ids=[r['call_id']for r in records])
            rt.flush_events()
            return {'phase':incident['phase'],'incident_id':incident_id,'event_id':event_id,
                'evidence_call_ids':[r['call_id']for r in records],'planning_only':True,
                'repair_applied':False,'repair_verified':False,'retry_granted':False}
        except Exception as error:
            with json_transaction(rt.store.path)as data:
                state=data.get('workflows',{}).get(rt.workflow_id,{})
                incident=(state.get('failure_recovery')or{}).get('incidents',{}).get(incident_id,{})
                if (incident.get('stage_lease')or{}).get('lease_id')==lease:
                    failed={'stage':'repair','status':'failed_or_unknown',
                        'error_type':type(error).__name__,'evidence_call_ids':[r['call_id']for r in records]}
                    code=self._repair_permission_code(error)
                    if code is not None:failed['permission_code']=code
                    incident.setdefault('history',[]).append(failed)
                    incident.update(phase='blocked',revision=incident['revision']+1,blocked_reason='repair_planning_failed_or_unknown')
                    incident.pop('stage_lease',None)
                    self._stage_event(state,incident,'blocked','main_chat',evidence_call_ids=[r['call_id']for r in records])
            raise
        finally:set_context(**previous)
