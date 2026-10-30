"""Durable scientific-owner review lane, independent of unrelated main repairs.

The lane never submits or retries work. A model can request acceptance; only the
existing runtime finish guard can commit it after actual native evidence checks.
"""
import copy
import hashlib
import inspect as python_inspect
import json
import os
from pathlib import Path
import socket
import time
import uuid

from .state_io import json_transaction
from .workflow_patch import execution_contract
from .owner_review_diagnostics import ReviewFailure, failure_record


_KIND = 'worker_scientific_owner_review'
_FIELDS = ('token','attempt_id','job_ids','contract','result_ref','result','validation_dossier',
           'tool_returned','jobs_confirmed_terminal','started_at')


def _review_control_projection(state,step):
    """Raw exact native facts only; no results/history or acceptance decision.

    Identity hashes include the complete original protected result/dossier, so
    replacing science while a model streams invalidates the same native witness.
    External goal, interrupt, futures and legacy credential proofs stay live.
    """
    fields=('username','conv_id','status','user_paused','plan_version','goal_contract')
    raw=state.get('nodes',{}).get(step,{})
    node={key:raw.get(key) for key in ('token','status','contract','job_ids',
        'tool_returned','jobs_confirmed_terminal')}
    node['_review_identity_sha256']=_digest({key:raw.get(key) for key in _FIELDS})
    records={}
    for key,record in state.get('scientific_owner_reviews',{}).items():
        if (record.get('binding')or{}).get('step_id')!=step:continue
        event=record.get('event')or{}
        records[key]={'phase':record.get('phase'),'binding':record.get('binding'),
            'event_sha256':_digest({name:event.get(name) for name in ('event_id','kind','payload')}),
            'identity_sha256':_digest(record.get('identity'))}
    return {**{key:state.get(key) for key in fields},'nodes':{step:node},
            'scientific_owner_reviews':records}

def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _same_event(left,right):
    return all(left.get(key)==right.get(key) for key in ('event_id','kind','payload'))


def _identity(node):
    return {key:copy.deepcopy(node.get(key)) for key in _FIELDS}


def _process():
    return {'hostname':socket.gethostname(),'pid':os.getpid(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'starttime':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}


class WorkflowReviewStage:
    """Native controller; actor and control callback are trusted API bindings."""
    def __init__(self, runtime):
        self.runtime=runtime

    def _authority(self,state,step):
        rt=self.runtime;main=rt.main;goal=main.goal_contract.to_dict()
        from .workflow_user_control import workflow_control_current
        node=state.get('nodes',{}).get(step,{})
        contract=node.get('contract',{})
        supported=(contract.get('tool') in {'run_gcmc_isotherm','run_gcmc_batch'}
            or contract.get('tool')=='run_cdft' and contract.get('arguments',{}).get('action')=='submit')
        if not supported:return False
        future=rt.futures.get(step)
        if future is not None and not future.done():return False
        approved=[item for item in goal.get('approved_nodes',[]) if item.get('step_id')==step]
        if (state.get('username')!=rt.username or state.get('conv_id')!=rt.conv_id
                or state.get('status') not in {'active','needs_user'} or state.get('user_paused')
                or not workflow_control_current(main,rt.username,rt.conv_id) or main._pending_workflow_patch or main._awaiting_plan_approval
                or state.get('goal_contract')!=goal or not goal.get('execution_authorized')
                or state.get('plan_version')!=goal.get('approved_plan_version') or len(approved)!=1
                or node.get('status')!='prefinish' or not node.get('token')
                or node.get('tool_returned') is not True or node.get('jobs_confirmed_terminal') is not True
                or not node.get('job_ids')):
            return False
        if execution_contract(approved[0])!=execution_contract(node.get('contract',{})):
            # The same current-plan native carry comparator used by activation.
            from .native_resource_carry import view_contract_matches
            if not view_contract_matches(state,step,approved[0]):return False
        return True

    def _binding(self,state,step):
        rt=self.runtime;node=state['nodes'][step];ref=node.get('result_ref')or{}
        dossier=node.get('validation_dossier')or{}
        paths=[ref.get('evidence_path'),dossier.get('evidence_path')]
        paths.extend((node.get('artifacts')or{}).keys())
        owned=[]
        for value in paths:
            if not isinstance(value,str):continue
            path=Path(value)
            if path.is_absolute() and path.is_relative_to(rt.root) and '..' not in path.parts:
                owned.append(str(path))
        return {'username':rt.username,'conv_id':rt.conv_id,'workflow_id':rt.workflow_id,
            'goal_version':state['goal_contract']['version'],'plan_version':state['plan_version'],
            'step_id':step,'token':node['token'],'attempt_id':node.get('attempt_id'),
            'contract_sha256':_digest(node['contract']),'job_ids':list(node['job_ids']),
            'producer_call_id':ref.get('call_id'),'dossier_call_id':dossier.get('evidence_call_id'),
            'allowed_read_paths':sorted(set(owned))}

    def observe(self,step_id,*,observed_state=None):
        """Create one durable wake per exact native attempt/evidence revision."""
        rt=self.runtime;state=observed_state if observed_state is not None else rt.snapshot()
        if not self._authority(state,step_id):return None
        identity=_identity(state['nodes'][step_id]);binding=self._binding(state,step_id)
        key=_digest({'binding':binding,'identity':identity})
        if key in (state.get('scientific_owner_reviews')or{}):return None
        with json_transaction(rt.store.path) as data:
            current=data.get('workflows',{}).get(rt.workflow_id,{})
            if not self._authority(current,step_id) or _identity(current['nodes'][step_id])!=identity:return None
            book=current.setdefault('scientific_owner_reviews',{})
            if key in book:return None
            event={'event_id':f'scientific-review:{rt.workflow_id}:{key}', 'kind':_KIND,
                'time':time.time(),'payload':{'step_id':step_id,'runtime_id':rt.workflow_id,
                    'origin_goal_version':binding['goal_version'],'plan_version':binding['plan_version'],
                    'review_binding':binding,'review_contract':copy.deepcopy(current['nodes'][step_id]['contract']),
                    'validation_dossier':copy.deepcopy(current['nodes'][step_id].get('validation_dossier')),'requires_user':False}}
            book[key]={'phase':'required','rounds':0,'identity':identity,'binding':binding,
                'event':event,'created_at':time.time(),'scientific_acceptance':False,'acceptance_origin':None,
                'history':[{'transition':'required','at':time.time(),'round':0}]}
            current.setdefault('outbox',{})[event['event_id']]={**copy.deepcopy(event),'sent':False}
            return copy.deepcopy(event)

    def _record(self,state,event):
        if event.get('kind')!=_KIND:return None
        binding=(event.get('payload')or{}).get('review_binding')or{}
        step=binding.get('step_id')
        if not self._authority(state,step):return None
        for record in (state.get('scientific_owner_reviews')or{}).values():
            if (record.get('event',{}).get('event_id')==event.get('event_id')
                    and _same_event(record.get('event',{}),event) and record.get('binding')==binding
                    and record.get('identity')==_identity(state['nodes'][step])
                    and record.get('phase') in {'required','retryable','reviewing'}):return record
        return None

    def current_event(self,event):
        project=getattr(self.runtime,'project_snapshot',None)
        if not callable(project):
            return bool(self._record(self.runtime.snapshot(),event))
        binding=(event.get('payload')or{}).get('review_binding')or{}
        step=binding.get('step_id')
        state=project(('owner-review-control-v1',step),
            lambda current:_review_control_projection(current,step))
        # Legacy allocation carry also verifies live external credential bytes.
        # Do not memoize that proof behind the workflow file's identity.
        approved=[n for n in self.runtime.main.goal_contract.to_dict().get('approved_nodes',[])
                  if n.get('step_id')==step]
        node=state.get('nodes',{}).get(step,{})
        if (len(approved)==1 and execution_contract(approved[0])!=execution_contract(node.get('contract',{}))):
            return bool(self._record(self.runtime.snapshot(),event))
        if not self._authority(state,step):return False
        event_identity=_digest({key:event.get(key) for key in ('event_id','kind','payload')})
        for record in state.get('scientific_owner_reviews',{}).values():
            if (record.get('event_sha256')==event_identity and record.get('binding')==binding
                    and record.get('identity_sha256')==node.get('_review_identity_sha256')
                    and record.get('phase') in {'required','retryable','reviewing'}):return True
        return False

    def _verify_request(self,actor,checkpoint_path,request,binding,started):
        from .report_recovery import _OwnedFiles
        from .local_preparation_recovery import checkpoint_call_witness
        from .recovery import result_object
        rt=self.runtime;step=binding['step_id'];ids=request.get('evidence_call_ids')
        if (request.get('step_id')!=step or not isinstance(ids,list) or not 1<=len(ids)<=8
                or len(set(ids))!=len(ids) or not isinstance(request.get('conclusion'),str)
                or not 10<=len(request['conclusion'])<=16000):
            raise ReviewFailure('evidence_invalid')
        files=_OwnedFiles(rt.root,{'username':rt.username,'conv_id':rt.conv_id})
        verified=[];native=[];evidence_facts={}
        try:
            for cid in ids:
                if not isinstance(cid,str) or not cid.isalnum():raise ReviewFailure('evidence_invalid')
                calls=[call for call in actor.memory.tool_call_log if call.get('call_id')==cid]
                if len(calls)!=1:raise ReviewFailure('evidence_invalid')
                memory=calls[0];call,fact=files.read(rt.root/'evidence'/(cid+'.json'),as_json=True)
                fields=('call_id','time','agent','tool','params','failed')
                if (any(memory.get(k)!=call.get(k) for k in fields)
                        or memory.get('result_sha256')!=hashlib.sha256(str(call.get('result','')).encode()).hexdigest()
                        or memory.get('evidence_path')!=fact['path'] or call.get('time',0)<started
                        or call.get('failed') is not False or call.get('tool')!='inspect_workflow_result'
                        or call.get('params')!={'step_id':step}):
                    raise ReviewFailure('evidence_invalid')
                checkpoint_fact=checkpoint_call_witness(files,checkpoint_path,call,step_id=step,require_result_hash=True)
                evidence_facts[fact['path']]=fact
                evidence_facts[checkpoint_fact['path']]=checkpoint_fact
                raw=result_object(call.get('result'))
                expected={'status':'inspected','workflow_id':rt.workflow_id,'step_id':step,
                    'token':binding['token'],'attempt_id':binding['attempt_id'],
                    'job_ids':binding['job_ids'],'plan_version':binding['plan_version'],
                    'goal_version':binding['goal_version'],'owner':[rt.username,rt.conv_id],
                    'source_execution_call_id':binding['producer_call_id']}
                if any(raw.get(key)!=value for key,value in expected.items()):
                    raise ReviewFailure('evidence_invalid')
                if _digest(raw.get('contract'))!=binding['contract_sha256']:
                    raise ReviewFailure('evidence_invalid')
                fresh=rt.inspect_workflow_result(step,expected_token=binding['token'])
                if _digest(fresh)!=_digest(raw):
                    raise ReviewFailure('sources_changed')
                node=rt.snapshot()['nodes'][step];reference=node.get('result_ref')or{}
                producer,producer_fact=files.read(reference['evidence_path'],as_json=True)
                if producer_fact['sha256']!=raw.get('source_execution_sha256'):
                    raise ReviewFailure('sources_changed')
                producer_checkpoint=checkpoint_call_witness(files,reference['checkpoint_path'],producer,step_id=step)
                dossier=node.get('validation_dossier')or{}
                _,dossier_fact=files.read(dossier['evidence_path'])
                for bound in (producer_fact,producer_checkpoint,dossier_fact):evidence_facts[bound['path']]=bound
                verified.append(copy.deepcopy(memory));native.append(raw)
            from .scientific_review import native_inspection_sources
            sources=native_inspection_sources(verified,rt.snapshot(),rt.root/'evidence')
            if len(sources)!=len(verified):raise ReviewFailure('sources_changed')
            if len(evidence_facts)>32 or sum(item['size'] for item in evidence_facts.values())>64*1024*1024:
                raise ReviewFailure('evidence_invalid')
            return verified,native,list(evidence_facts.values())
        finally:files.close()

    def run(self,actor,event,*,checkpoint_path,stage_runner=None,control_valid=None):
        """Bounded review, actual-call adoption, then existing native finish CAS."""
        rt=self.runtime;main=rt.main
        from .workflow_user_control import workflow_control_guard
        user_control=workflow_control_guard(main,rt.username,rt.conv_id)
        goal=copy.deepcopy(main.goal_contract.to_dict());lease=uuid.uuid4().hex
        checkpoint_path=Path(checkpoint_path)
        if (actor is main or getattr(actor,'_evidence_root',None)!=rt.root/'evidence'
                or not callable(getattr(actor,'_on_checkpoint',None))
                or not callable(getattr(main,'_on_checkpoint',None))
                or Path(actor.config.project_root)!=Path(main.config.project_root)
                or not checkpoint_path.is_absolute() or not checkpoint_path.is_relative_to(rt.root)):
            raise ReviewFailure('identity_mismatch')
        def control():
            if rt.main is not main:raise ReviewFailure('controls_changed')
            if main.goal_contract.to_dict()!=goal:raise ReviewFailure('goal_changed')
            if not user_control() or control_valid is not None and control_valid() is not True:
                raise ReviewFailure('controls_changed')
        with json_transaction(rt.store.path) as data:
            state=data.get('workflows',{}).get(rt.workflow_id,{})
            record=self._record(state,event);control()
            if (not record or record['phase']=='reviewing' or record['rounds']>=2
                    or any(item.get('phase')=='reviewing' for item in (state.get('scientific_owner_reviews')or{}).values())):
                raise ReviewFailure('review_claim_unavailable')
            record.update(phase='reviewing',rounds=record['rounds']+1,
                lease={'id':lease,'process':_process(),'claimed_at':time.time()})
            self._history(record,'reviewing',lease_id=lease)
            ticket=copy.deepcopy(record)
        binding=ticket['binding'];step=binding['step_id'];started=ticket['lease']['claimed_at']
        previous_interrupt=getattr(actor,'_on_interrupt_requested',None)
        actor._on_interrupt_requested=lambda: self._cancelled(control,event)
        def inspect(target):
            control()
            if target!=step or not self.current_event(event):raise ReviewFailure('attempt_superseded')
            return rt.inspect_workflow_result(step,expected_token=binding['token'],control_valid=lambda:not self._cancelled(control,event))
        actor._on_workflow_result_inspect=inspect
        def stage_valid():
            control()
            if not self.current_event(event):raise ReviewFailure('attempt_superseded')
            return True
        try:
            control()
            result=(stage_runner(actor,event) if stage_runner else actor.run_workflow_review_stage(event,control_valid=stage_valid))
            control()
            if not self.current_event(event):raise ReviewFailure('attempt_superseded')
            request=result.get('finish_request') if isinstance(result,dict) else None
            if not isinstance(request,dict):
                self._complete(event,lease,'blocked',{'reason':'scientific owner did not request acceptance','review':result})
                return {'status':'blocked','scientific_acceptance':False,'not_resubmitted':True}
            calls,native,source_facts=self._verify_request(actor,checkpoint_path,request,binding,started)
            from .owner_review_sources import capture_review_sources,review_sources_unchanged
            sealed_sources=capture_review_sources(rt,binding,native,source_facts)
            control()
            if not self.current_event(event):raise ReviewFailure('attempt_superseded')
            existing={call.get('call_id'):call for call in main.memory.tool_call_log}
            for call in calls:
                if call['call_id'] in existing and existing[call['call_id']]!=call:
                    raise ReviewFailure('evidence_invalid')
            # Preserve the actual IDs; do not fabricate a second execution record.
            main.memory.tool_call_log.extend(copy.deepcopy(call) for call in calls if call['call_id'] not in existing)
            checkpoint=main._checkpoint
            try:parameters=python_inspect.signature(checkpoint).parameters
            except (ValueError,TypeError):parameters={}
            if 'sync_chain' in parameters:
                checkpoint('scientific_owner_review_evidence',sync_chain=False)
            else:
                # Existing deployed Sessions retain mandatory synchronous durability.
                # Never catch an I/O TypeError and replay a partially written call.
                checkpoint('scientific_owner_review_evidence')
            control()
            if not self.current_event(event):raise ReviewFailure('attempt_superseded')
            node=rt.snapshot()['nodes'][step]
            ids=request['evidence_call_ids']
            if node['contract'].get('tool') in {'run_gcmc_batch','run_gcmc_isotherm'}:
                # GCMC's finish guard consumes its internal native dossier; the
                # public inspection and its actual producer remain lane audit.
                ids=[node.get('validation_dossier',{}).get('evidence_call_id')]
            def reviewed_sources_current():
                return review_sources_unchanged(rt.root,sealed_sources) and self._control_ok(control)
            if not reviewed_sources_current():raise ReviewFailure('sources_changed')
            review_evidence={'schema_version':1,'binding':copy.deepcopy(binding),
                'inspection_call_ids':[call['call_id'] for call in calls],
                'source_files':[{key:fact[key] for key in ('path','sha256','size')} for fact in source_facts],
                'native_fact_sha256':sorted({_digest(value) for value in native}),
                'sealed_sources_sha256':sealed_sources['sha256']}
            review_evidence['sha256']=_digest(review_evidence)
            receipt=rt.finish_node(step,ids,request['conclusion'],control_valid=lambda:self._control_ok(control),
                review_valid=reviewed_sources_current,review_evidence=review_evidence)
            committed=rt.snapshot()['nodes'][step]
            if receipt.get('phase')!='finish' or committed.get('status')!='succeeded':
                raise ReviewFailure('finish_not_committed')
            if (committed.get('node_verification')or{}).get('scientific_owner_review')!=review_evidence:
                # An idempotent receipt may belong to a main-role finish that
                # won the final race. Native completion is valid; attribution
                # to this isolated review is not.
                raise ReviewFailure('attempt_superseded')
            outcome={'status':'accepted','acceptance_origin':'accepted_by_owner','scientific_acceptance':True,'step_id':step,'evidence_call_ids':[call['call_id']for call in calls],
                'finish_receipt':receipt,'review_evidence_sha256':review_evidence['sha256'],'not_resubmitted':True}
            self._complete(event,lease,'accepted',outcome)
            return outcome
        except BaseException as error:
            self._complete(event,lease,'retryable',failure_record(error))
            raise
        finally:actor._on_interrupt_requested=previous_interrupt

    @staticmethod
    def _control_ok(control):
        try:control();return True
        except (ValueError,PermissionError):return False

    def _cancelled(self,control,event):
        try:control();return not self.current_event(event)
        except (ValueError,PermissionError):return True

    @staticmethod
    def _history(record,transition,**facts):
        entry={'transition':transition,'at':time.time(),'round':record.get('rounds',0),**copy.deepcopy(facts)}
        record.setdefault('history',[]).append(entry)

    def _complete(self,event,lease,phase,result):
        rt=self.runtime
        with json_transaction(rt.store.path) as data:
            state=data.get('workflows',{}).get(rt.workflow_id,{})
            for record in (state.get('scientific_owner_reviews')or{}).values():
                if not _same_event(record.get('event',{}),event):continue
                owns_lease=(record.get('lease')or{}).get('id')==lease
                claimed_before=any(h.get('lease_id')==lease and h.get('transition')=='reviewing' for h in record.get('history',[]))
                if not owns_lease and not claimed_before:continue
                if phase=='retryable':
                    record.setdefault('first_failure',copy.deepcopy(result))
                    record['last_failure']=copy.deepcopy(result)
                    self._history(record,'review_failed',lease_id=lease,failure=result)
                else:self._history(record,'accepted_by_owner' if phase=='accepted' else phase,lease_id=lease)
                # External completion can recover a still-running lane. A late
                # error stays visible but cannot rewrite that completion origin.
                if not owns_lease:return
                record.update(phase='blocked' if phase=='retryable' and record['rounds']>=2 else phase,
                    result=copy.deepcopy(result),updated_at=time.time())
                record.pop('lease',None)
                if phase=='accepted':
                    record.update(scientific_acceptance=True,acceptance_origin='accepted_by_owner')
                if record['phase']=='blocked':
                    eid=record['event']['event_id']+':blocked'
                    state.setdefault('outbox',{}).setdefault(eid,{'event_id':eid,
                        'kind':'worker_scientific_review_blocked','sent':False,
                        'payload':{'step_id':record['binding']['step_id'],'runtime_id':rt.workflow_id,
                            'origin_goal_version':record['binding']['goal_version'],
                            'plan_version':record['binding']['plan_version'],'review_binding':record['binding'],
                            'reason':'Scientific owner review requires evidence-backed diagnosis; no acceptance granted.',
                            'review_result':copy.deepcopy(result),'scientific_acceptance':False,'not_resubmitted':True}})
                return

    @staticmethod
    def _committed(state,record):
        node=state.get('nodes',{}).get(record['binding']['step_id'],{})
        return (node.get('status')=='succeeded' and bool(node.get('node_verification'))
            and all(node.get(field)==record['identity'].get(field)
                for field in ('token','attempt_id','contract','job_ids','result_ref')))

    def recover(self,*,observed_state=None):
        """Recover a proven-dead local process; never steal a live/foreign lease."""
        rt=self.runtime;state=observed_state if observed_state is not None else rt.snapshot()
        candidates=[];here=_process()
        for key,record in (state.get('scientific_owner_reviews')or{}).items():
            if record.get('phase') not in {'reviewing','retryable'}:continue
            if self._committed(state,record):
                candidates.append((key,copy.deepcopy(record)));continue
            if record.get('phase')!='reviewing':continue
            process=(record.get('lease')or{}).get('process')or{}
            if process.get('hostname')!=here['hostname']:continue
            pid=process.get('pid')
            if type(pid)is not int or pid<=0:continue
            dead=process.get('boot_id')!=here['boot_id']
            if not dead:
                try:start=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[19]
                except FileNotFoundError:dead=True
                except (PermissionError,OSError,IndexError):continue
                else:dead=start!=process.get('starttime')
            if dead:candidates.append((key,copy.deepcopy(record)))
        if not candidates:return []
        restored=[]
        with json_transaction(rt.store.path) as data:
            current=data.get('workflows',{}).get(rt.workflow_id,{})
            for key,observed in candidates:
                record=(current.get('scientific_owner_reviews')or{}).get(key)
                if record!=observed:continue
                step=record['binding']['step_id'];node=current.get('nodes',{}).get(step,{})
                if self._committed(current,record):
                    proof=copy.deepcopy((node.get('node_verification')or{}).get('scientific_owner_review')or{})
                    proof_digest=proof.pop('sha256',None)
                    owner_committed=(bool(proof) and proof.get('binding')==record['binding'] and proof_digest==_digest(proof))
                    origin='accepted_by_owner' if owner_committed else 'recovered_external_finish'
                    if 'result'in record:record.setdefault('review_result_before_recovery',copy.deepcopy(record['result']))
                    record.update(phase='accepted' if owner_committed else 'recovered_external_finish',
                        acceptance_origin=origin,scientific_acceptance=owner_committed,native_workflow_completed=True,
                        result={'status':'accepted' if owner_committed else 'recovered_external_finish',
                            'acceptance_origin':origin,'scientific_acceptance':owner_committed,
                            'recovered_native_finish':True,'not_resubmitted':True})
                    self._history(record,'recovered_owner_finish' if owner_committed else 'recovered_external_finish')
                else:
                    record['phase']='retryable' if record['rounds']<2 else 'blocked'
                    self._history(record,'dead_owner_recovered')
                record.pop('lease',None);restored.append(record['event']['event_id'])
        return restored
