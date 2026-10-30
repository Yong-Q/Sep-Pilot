"""Server-only worker capability for model-authored scientific report tools."""
from contextlib import contextmanager
import copy
import hashlib
import json
from pathlib import Path
import threading

_bound=threading.local()


def _hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


@contextmanager
def report_execution_context(runtime,worker,step_id,token):
    """Only the native worker binds this; there is no tool-schema callback."""
    prior=getattr(_bound,'value',None)
    original_main=runtime.main
    goal=copy.deepcopy(original_main.goal_contract.to_dict())
    from .workflow_user_control import workflow_control_guard
    user_control=workflow_control_guard(original_main,runtime.username,runtime.conv_id)
    state=runtime.snapshot();node=copy.deepcopy(state.get('nodes',{}).get(step_id,{}))
    # Include original producers behind figure/report intermediates. An
    # unchanged fragment cannot hide a replaced original attempt while the
    # author is waiting for its model response.
    ancestors=set();pending=list(node.get('contract',{}).get('depends_on',[]))
    while pending:
        key=pending.pop()
        if key in ancestors:continue
        if key not in state.get('nodes',{})or key==step_id or len(ancestors)>=512:
            raise ValueError('report dependency provenance is missing or excessive')
        ancestors.add(key)
        pending.extend(state['nodes'][key].get('contract',{}).get('depends_on',[]))
    source_identity={key:{name:copy.deepcopy(state['nodes'][key].get(name))for name in
        ('token','attempt_id','job_ids','result_ref','contract','execution_fingerprint','status')}
        for key in ancestors}
    if (node.get('token')!=token or node.get('contract',{}).get('tool')not in{'generate_scientific_report','generate_result_figures'}
            or node.get('status')!='running' or node.get('dispatch_phase')!='entered_tool'
            or worker is original_main or Path(worker._evidence_root)!=runtime.root/'evidence'):
        raise PermissionError('report capability requires the current native report worker')
    control_ids=tuple(sorted(ancestors|{step_id}))
    def project_control(current):
        fields=('username','conv_id','status','user_paused','plan_version','goal_contract')
        identities=('token','attempt_id','job_ids','result_ref','contract','execution_fingerprint','status')
        return {**{key:current.get(key) for key in fields},'nodes':{
            key:{name:current.get('nodes',{}).get(key,{}).get(name) for name in identities}
            for key in control_ids}}
    def valid():
        if (runtime.main is not original_main or original_main.goal_contract.to_dict()!=goal
                or not user_control()):return False
        project=getattr(runtime,'project_snapshot',None)
        current=(project(('report-control-v1',control_ids),project_control)
                 if callable(project) else project_control(runtime.snapshot()))
        active=current.get('nodes',{}).get(step_id,{})
        if (current.get('username'),current.get('conv_id'))!=(runtime.username,runtime.conv_id):return False
        if (current.get('status')!='active' or current.get('user_paused')
                or current.get('plan_version')!=state.get('plan_version')
                or current.get('goal_contract')!=state.get('goal_contract')
                or active.get('token')!=token or active.get('status')!='running'
                or active.get('contract')!=node.get('contract')):return False
        for key,identity in source_identity.items():
            source=current.get('nodes',{}).get(key,{})
            if {name:source.get(name)for name in identity}!=identity:return False
        return True
    if not valid():raise ValueError('report worker control was superseded')
    _bound.value={'runtime':runtime,'worker':worker,'step_id':step_id,'token':token,
        'state':state,'arguments':copy.deepcopy(node.get('effective_arguments')or node['contract']['arguments']),
        'control_valid':valid}
    try:yield
    finally:_bound.value=prior


def execute_context_report(arguments):
    """Called by registry only with a current native worker context."""
    capability=getattr(_bound,'value',None)
    if not capability:
        return {'error':'Model-authored reports require a current owned workflow report node',
            'error_kind':'report_requires_native_context','executed':False,
            'next_action':'Compile a communicator report node with completed source dependencies.'}
    from .report_authoring import build_result_packet,author_report,publish_report,ReportCapabilityGap,ReportReviewUnresolved,_read
    from .report_authoring_session import NativeReportModelBridge,ReportStageSession
    from .session import Session
    from .goal_contract import GoalContract
    from .report_recovery import _OwnedFiles
    from .local_preparation_recovery import checkpoint_call_witness
    from .state_io import write_checkpoint,compact_checkpoint_evidence,apply_evidence_references
    from .result_guidance import read_result_guidance
    runtime=capability['runtime'];worker=capability['worker'];valid=capability['control_valid']
    if arguments!=capability['arguments']or not valid():
        return {'error':'Report arguments or control differ from the approved worker',
                'error_kind':'report_control_changed','executed':False}
    owner={'username':runtime.username,'conv_id':runtime.conv_id}
    actors={};paths={}
    def factory(phase):
        if phase in actors:return actors[phase]
        actor=ReportStageSession(config=worker.config,registry=worker.registry)
        actor.client=actor.client.with_options(timeout=60,max_retries=0)
        actor._evidence_root=runtime.root/'evidence'
        # The packet and native control callback already bind every source and
        # owner. A private data-only actor must not load the full TaskLine or
        # receive unrelated scheduler/recovery envelopes from its parent.
        actor._current_line_id=None
        # Drafting is a data-reading capability, not a copy of the parent's
        # mutable execution grant or large conversation/maintenance history.
        actor.goal_contract=GoalContract.from_dict({'version':worker.goal_contract.version,
            'execution_authorized':False,'requires_plan_approval':False})
        directory=runtime.root/'workers'/hashlib.sha256(
            ('report-stage:'+capability['step_id']+':'+capability['token']+':'+phase).encode()).hexdigest()[:24]
        path=directory/'session_checkpoint.json';paths[phase]=path
        def checkpoint(saved,reason):
            if not valid():raise ValueError('report model stage control changed before checkpoint')
            saved['owner']={**owner,'step_id':capability['step_id'],'role':'report_'+phase}
            refs=compact_checkpoint_evidence(saved,runtime.root/'evidence')
            write_checkpoint(path,saved);apply_evidence_references(actor.memory,refs)
        actor._on_checkpoint=checkpoint
        actor._on_interrupt_requested=lambda:not valid()
        actors[phase]=actor
        return actor
    def attest(source,call,raw):
        files=_OwnedFiles(runtime.root,owner)
        try:
            path=source['result_ref']['checkpoint_path']
            quantity={}
            if source.get('contract',{}).get('tool')=='run_pore_analysis':
                adapter=Path(worker.config.project_root)/'tools/zeopp.py'
                if adapter.is_symlink()or not adapter.is_file()or adapter.stat().st_size>1024*1024:
                    raise ValueError('native pore quantity contract is unavailable')
                quantity={'source':'native Zeo adapter StructuralProperties and command contract',
                    'adapter':'tools/zeopp.py','adapter_sha256':hashlib.sha256(adapter.read_bytes()).hexdigest(),
                    'fields':{'di_A':'largest included sphere diameter (LCD), angstrom',
                        'df_A':'largest free sphere diameter (PLD), angstrom',
                        'dif_A':'largest included sphere along the free-sphere path, angstrom'},
                    'pore_diameters_are_not_sampling_probe_radius':True,
                    'sampling_probe_scope':'Configured probe/channel radius belongs to surface/volume sampling; do not reinterpret df as a probe-selected adsorbate pore window without separate evidence.'}
            try:
                witness=checkpoint_call_witness(files,path,call,
                    step_id=source['contract']['step_id'],require_result_hash=True)
                return {'checkpoint_verified':True,'checkpoint_sha256':witness['sha256'],
                    'checkpoint_path':path,'producer_result_digest_verified':True,'quantity_contract':quantity}
            except ValueError as error:
                # A legacy producer's full result may already have been compacted
                # without a digest. Its metadata witness alone is not promoted.
                # Perform and journal a fresh exact-artifact read instead.
                if 'lacks the trusted native result digest'not in str(error):raise
                witness=checkpoint_call_witness(files,path,call,step_id=source['contract']['step_id'])
                inspector=factory('source-'+source['contract']['step_id']);facts=[];budget={'bytes':0}
                for name,fact in (source.get('artifacts')or{}).items():
                    if not isinstance(fact,dict)or fact.get('files')is not None:continue
                    if Path(name).suffix.lower()not in{'.csv','.md','.json','.png'}:continue
                    _,fresh=_read(runtime.root,name,budget)
                    if (fresh['size']!=fact.get('size')or fresh['mtime_ns']!=fact.get('mtime_ns')):
                        raise ValueError('completed source artifact changed before fresh report inspection')
                    facts.append(fresh)
                if not facts:raise ValueError('legacy source requires a fresh real output inspection')
                record={'status':'inspected','read_only':True,'owner':owner,
                    'workflow_id':runtime.workflow_id,'source_step':source['contract']['step_id'],
                    'source_token':source.get('token'),'source_attempt':source.get('attempt_id'),
                    'producer_call_id':call['call_id'],'producer_sha256':hashlib.sha256(raw).hexdigest(),
                    'producer_checkpoint_sha256':witness['sha256'],'artifacts':facts,
                    'scientific_acceptance':False,'no_resubmission':True}
                inspector.memory.record_tool_call('communicator','inspect_report_result_sources',
                    {'source_step':source['contract']['step_id'],'producer_call_id':call['call_id']},
                    json.dumps(record,ensure_ascii=False,allow_nan=False),failed=False)
                cid=inspector.memory.tool_call_log[-1]['call_id']
                inspector._checkpoint('fresh_report_source_inspection')
                fresh_call,fresh_fact=files.read(runtime.root/'evidence'/(cid+'.json'),as_json=True)
                fresh_witness=checkpoint_call_witness(files,paths['source-'+source['contract']['step_id']],
                    fresh_call,step_id=capability['step_id'],require_result_hash=True)
                return {'checkpoint_verified':True,'checkpoint_sha256':witness['sha256'],
                    'checkpoint_path':path,'producer_result_digest_verified':False,
                    'quantity_contract':quantity,
                    'fresh_artifact_inspection_verified':True,'fresh_inspection_call_id':cid,
                    'fresh_inspection_evidence':fresh_fact,'fresh_inspection_checkpoint':fresh_witness}
        finally:files.close()
    try:
        native_job_watch=getattr(runtime,'job_watch',None)
        packet=build_result_packet(runtime.root,owner,capability['state'],capability['step_id'],
            attest=attest,control_valid=valid,
            job_lookup=native_job_watch.get if native_job_watch else None)
        if capability['state']['nodes'][capability['step_id']]['contract']['tool']=='generate_result_figures':
            from .figure_authoring import author_figures
            factory('plot');factory('visual')
            bridge=NativeReportModelBridge(runtime.root,owner,packet['packet_sha256'],
                actor_factory=factory,checkpoint_paths={key:paths[key]for key in('plot','visual')},
                control_valid=valid,guidance_reader=read_result_guidance)
            def sandbox(params):
                if not valid():raise ValueError('figure control changed before sandbox')
                result=worker.registry.execute_dict('plot_result_code',params)
                worker.memory.record_tool_call('analyst','plot_result_code',params,
                    json.dumps(result,ensure_ascii=False,default=str),failed=result.get('ok')is not True)
                worker._checkpoint('after_report_figure_sandbox')
                return result
            return author_figures(packet,arguments['output_dir'],model_call=bridge,
                sandbox_call=sandbox,control_valid=valid,visual_call=bridge)
        phases=('author','review') if packet.get('independent_review_required',True) else ('author',)
        for phase in phases:factory(phase)
        bridge=NativeReportModelBridge(runtime.root,owner,packet['packet_sha256'],
            actor_factory=factory,checkpoint_paths={key:paths[key]for key in phases},
            control_valid=valid,guidance_reader=read_result_guidance)
        authored=author_report(packet,model_call=bridge,control_valid=valid)
        if not valid():raise ValueError('report control changed before immutable publication')
        receipt=publish_report(runtime.root,arguments['output_path'],packet,authored,control_valid=valid)
        receipt['model_stage_receipts']=[{'phase':phase,'checkpoint_path':str(paths[phase]),
            'call_ids':[call['call_id']for call in actor.memory.tool_call_log]}
            for phase,actor in actors.items()]
        return receipt
    except ReportCapabilityGap as error:
        return {'error':str(error),**error.details}
    except ReportReviewUnresolved as error:
        return {'error':str(error), 'error_kind':'report_authoring_failed',
            'executed':False, 'report_published':False, 'scientific_acceptance':False,
            'contract_feedback':error.details}
    except (OSError,ValueError,PermissionError,KeyError,TypeError)as error:
        return {'error':str(error),'error_kind':'report_authoring_failed','executed':False,
            'report_published':False,'scientific_acceptance':False}
