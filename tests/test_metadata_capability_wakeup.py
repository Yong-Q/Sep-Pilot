"""Native proofs and API lifecycle admission; no model, retry or science calls."""
import asyncio
import copy
import json
import threading
from types import SimpleNamespace
import pytest
from agents.state_io import json_transaction
from agents.workflow_user_control import bind_api_workflow_control,advance_user_control
from agents.metadata_recovery import notify_metadata_capability,metadata_event_current,source_epoch
from tests.test_native_metadata_forward import forward_case
from tests.test_active_resource_carry import native_case
from tests.test_parallel_workflow import factory


def prepare_api(rt):
    from agents.parallel_workflow import _approval_digest
    from agents.watch_context import set_context
    main=rt.main
    # Narrow the existing native carry fixture to this feature's receipt-only
    # domain: quantities are already agreed, as in the manufactured receipt-only scene.
    with json_transaction(rt.store.path)as data:
        state=data['workflows'][rt.workflow_id];state.update(status='needs_user',pause_kind='failure')
        node=state['nodes']['n2'];node['contract']['arguments']['memory_mb']=node['effective_arguments']['memory_mb']
        for item in state['goal_contract']['approved_nodes']:
            if item['step_id']=='n2':item['arguments']['memory_mb']=node['effective_arguments']['memory_mb']
    main.context['approved_execution_proof']={'schema_version':1,'kind':'main_model_applied_plan',
        'owner':{'username':rt.username,'conv_id':rt.conv_id},'goal_version':main.goal_contract.version,
        'plan_version':main.goal_contract.approved_plan_version,'base_plan_version':1,
        'approval_sha256':_approval_digest(main.goal_contract.to_dict()),'control_source':copy.deepcopy(main._control_source)}
    main._api_turn_id='native-original-turn'
    conv=SimpleNamespace(username=rt.username,conv_id=rt.conv_id,session=main,current_uid=main._api_turn_id,
        interrupt_requested=False,interrupt_message='',is_processing=False,last_progress_at=0,current_status='',
        current_agent='lead-orchestrator',execution_logs=[])
    bind_api_workflow_control(conv,main)
    main._on_interrupt_requested=lambda:conv.interrupt_requested or conv.session is not main or conv.current_uid!=main._api_turn_id
    main._runtime_snapshot=rt.snapshot
    main._lifecycle_store=rt.lifecycle
    main._on_checkpoint=lambda *a,**k:None
    set_context(rt.username,rt.conv_id,'lead-orchestrator',main._current_line_id)
    return conv


def notices(rt):
    return [event for event in rt.lifecycle.snapshot()['events'].values()
        if (event.get('payload',{}).get('recovery')or{}).get('action')=='inspect_forward_metadata_alignment']


def test_native_epoch_notice_once_preserves_every_execution_and_budget_fact(forward_case):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt)
    before=copy.deepcopy(rt.snapshot());line=copy.deepcopy(rt._line());goal=rt.main.goal_contract.to_dict()
    ledger=(rt.root/'recovery_state.json').read_bytes();history=copy.deepcopy(rt.main.memory.tool_call_log)
    assert notify_metadata_capability(rt)
    events=notices(rt);assert len(events)==1
    event=events[0];recovery=event['payload']['recovery']
    assert recovery['arguments']['base_version']==2
    assert recovery['arguments']['changes'][0]['node']['arguments']==aligned['arguments']
    assert recovery['candidate_verified']is True and recovery['alignment_proof']['no_dispatch']is True
    assert recovery['retry_granted']is False and recovery['applied']is False and recovery['scientific_acceptance']is False
    assert source_epoch()==recovery['source_epoch'] and metadata_event_current(rt,event)
    assert notify_metadata_capability(rt)==[] and len(notices(rt))==1
    after=copy.deepcopy(rt.snapshot())
    if 'outbox'in before:after['outbox']=before['outbox']
    else:after.pop('outbox',None)
    assert after==before and rt._line()==line and rt.main.goal_contract.to_dict()==goal
    assert (rt.root/'recovery_state.json').read_bytes()==ledger and rt.main.memory.tool_call_log==history and reviews==[]


@pytest.mark.parametrize('change',['paused','uid','owner','actor','epoch','goal','running','uncertain',
    'foreign_job','nonterminal','new_science','unapproved','pending_user','pending_patch','line','ledger','producer','unknown_loaded'])
def test_no_notice_or_forward_permission_for_unknown_or_superseded_facts(forward_case,monkeypatch,change):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt);node=rt.snapshot()['nodes']['n2']
    if change=='paused':conv.interrupt_requested=True
    elif change=='uid':conv.current_uid='new-user-turn'
    elif change=='owner':conv.username='foreign'
    elif change=='actor':conv.session=object()
    elif change=='epoch':conv._workflow_user_control_epoch='invalid'
    elif change=='goal':rt.main.goal_contract.version+=1
    elif change in {'running','uncertain'}:rt.store.update(rt.workflow_id,'n2',node['token'],{'status':change})
    elif change=='foreign_job':jobs['77']['username']='foreign'
    elif change=='nonterminal':jobs['77'].update(terminal=False,state='RUNNING')
    elif change=='new_science':rt.main.goal_contract.approved_nodes[0]['arguments']['gas']='CO2'
    elif change=='unapproved':rt.main.context.pop('approved_execution_proof')
    elif change=='pending_user':rt.main._pending_user_interaction={'tool':'user_decision','prompt':'new science condition'}
    elif change=='pending_patch':rt.main._pending_workflow_patch={'new_version':3}
    elif change=='line':rt.task_lines.apply_workflow_patch(rt.main._current_line_id,[],2,3,'different current approval')
    elif change=='ledger':
        path=rt.root/'recovery_state.json';data=json.loads(path.read_text());data['recovery']['status']='uncertain';path.write_text(json.dumps(data))
    elif change=='producer':
        from pathlib import Path
        Path(node['result_ref']['evidence_path']).write_text('{}')
    else:monkeypatch.setattr('agents.parallel_workflow.ParallelWorkflow.start',lambda *a,**k:None)
    before=copy.deepcopy(rt.snapshot());ledger=(rt.root/'recovery_state.json').read_bytes()
    assert notify_metadata_capability(rt)==[] and notices(rt)==[]
    assert rt.snapshot()==before and (rt.root/'recovery_state.json').read_bytes()==ledger and reviews==[]


@pytest.mark.parametrize('change',['pause','uid','actor','epoch','goal','job','producer'])
def test_inflight_proof_has_owner_control_cas_before_any_outbox_write(forward_case,monkeypatch,change):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt)
    import agents.workflow_metadata_alignment as alignment
    original=alignment.prove_forward_transition
    # Invalidation is injected after the real proof, with source identity held
    # to the true prepatch epoch solely to reach the publication CAS boundary.
    epoch=source_epoch();monkeypatch.setattr('agents.metadata_recovery.source_epoch',lambda:epoch)
    def proved(*a,**k):
        result=original(*a,**k)
        if change=='pause':conv.interrupt_requested=True
        elif change=='uid':conv.current_uid='new-user-turn'
        elif change=='actor':conv.session=object()
        elif change=='epoch':advance_user_control(conv)
        elif change=='goal':rt.main.goal_contract.version+=1
        elif change=='job':jobs['77'].update(terminal=False,state='RUNNING')
        else:
            from pathlib import Path
            Path(rt.snapshot()['nodes']['n2']['result_ref']['evidence_path']).write_text('{}')
        return result
    monkeypatch.setattr(alignment,'prove_forward_transition',proved)
    assert notify_metadata_capability(rt)==[] and notices(rt)==[]


@pytest.mark.parametrize('change',['epoch','goal','approval','job','source','producer','token','pause','actor'])
def test_durable_event_cannot_be_replayed_against_new_controls_or_proof(forward_case,monkeypatch,change):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt);assert notify_metadata_capability(rt)
    event=notices(rt)[0]
    if change=='epoch':advance_user_control(conv)
    elif change=='goal':rt.main.goal_contract.version+=1
    elif change=='approval':rt.main.goal_contract.approved_plan_version+=1
    elif change=='job':jobs['77']['terminal']=False
    elif change=='source':monkeypatch.setattr('agents.metadata_recovery.source_epoch',lambda:{'epoch':'changed'})
    elif change=='producer':
        from pathlib import Path
        Path(rt.snapshot()['nodes']['n2']['result_ref']['evidence_path']).write_text('{}')
    elif change=='token':
        n=rt.snapshot()['nodes']['n2'];rt.store.update(rt.workflow_id,'n2',n['token'],{'token':'different-native-attempt'})
    elif change=='pause':conv.interrupt_requested=True
    elif change=='actor':conv.session=object()
    assert not metadata_event_current(rt,event)


def test_main_admission_only_annotates_old_active_diagnostic_and_keeps_history_budget(forward_case,monkeypatch):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt);main=rt.main
    branch={'status':'open','failed_step':'propose_workflow_patch','error':'affected runtime node still owns running/unknown resources; reconcile before patching',
        'diagnosis':'old technical rejection','resolution':None,'fix_attempts':[],
        'failed_arguments':{'base_version':1,'changes':[{'operation':'upsert','step_id':'n2','node':aligned}]}}
    main._error_branches={'old-active':copy.deepcopy(branch),'science':{**copy.deepcopy(branch),'failed_step':'finish_workflow_node','error':'scientific convergence not verified'}}
    before=copy.deepcopy(main._error_branches);ledger=(rt.root/'recovery_state.json').read_bytes()
    assert notify_metadata_capability(rt);event=notices(rt)[0]
    called=[]
    def loop(agent):
        window=main.context['native_metadata_capability_window'];called.append(window)
        assert window['historical_active_branches']==['old-active']
        summary=main._get_error_branch_summary()
        assert '历史ACTIVE拒绝不证明新候选不可用' in summary
        assert 'scientific convergence not verified' in summary
        assert main._error_branches==before
        return 'native main observed current capability'
    monkeypatch.setattr(main,'_execute_loop',loop)
    assert main.resume_lifecycle_event(event)=='native main observed current capability'
    assert called and 'native_metadata_capability_window'not in main.context
    assert main._error_branches==before and (rt.root/'recovery_state.json').read_bytes()==ledger
    assert main.goal_contract.approved_plan_version==2 and rt.snapshot()['plan_version']==1 and reviews==[]


def test_actual_api_lifecycle_reserves_native_main_without_human_query(forward_case,monkeypatch):
    import api
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt)
    assert notify_metadata_capability(rt);event=notices(rt)[0]
    monkeypatch.setattr(api,'_workflow_runtimes',{(conv.username,conv.conv_id):rt})
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda _:rt)
    monkeypatch.setattr(api,'_watch',rt.job_watch)
    monkeypatch.setattr(rt,'tick',lambda *a,**k:None)
    monkeypatch.setattr(api,'_PROCESS_STARTED_AT',0)
    scheduled=[];monkeypatch.setattr(api,'_schedule_lifecycle',scheduled.append)
    async def scenario():
        await api._dispatch_conversation_lifecycle(conv.username,conv,rt.lifecycle)
        assert conv.is_processing and conv.current_uid.startswith('lifecycle:') and len(scheduled)==1
        state=rt.lifecycle.snapshot()['events'][event['event_id']]
        assert state['supervisor']=='delivered' and state['main_chat']=='processing'
        assert state['supervisor_receipt']['recovery_capability_verified']is True
        # API reservation changes its own UID while the native epoch stays the
        # same. It must not revoke the proved advisory lifecycle event.
        assert not await api._lifecycle_controls_changed(conv,event,rt.main.goal_contract.version)
        for coro in scheduled:coro.close()
    try:asyncio.run(scenario())
    finally:
        for coro in scheduled:coro.close()


@pytest.mark.parametrize('change',['pause','uid','epoch','actor'])
def test_api_waiting_claim_rejects_replaced_owner_before_scheduling(forward_case,monkeypatch,change):
    import api
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt);assert notify_metadata_capability(rt);event=notices(rt)[0]
    monkeypatch.setattr(api,'_workflow_runtimes',{(conv.username,conv.conv_id):rt})
    entered,release=threading.Event(),threading.Event();original=rt.lifecycle.claim
    def blocked(*a,**k):entered.set();assert release.wait(3);return original(*a,**k)
    monkeypatch.setattr(rt.lifecycle,'claim',blocked)
    scheduled=[];monkeypatch.setattr(api,'_schedule_lifecycle',scheduled.append)
    async def scenario():
        task=asyncio.create_task(api._reserve_and_schedule_lifecycle_main(conv,event,rt.lifecycle))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            if change=='pause':conv.interrupt_requested=True
            elif change=='uid':conv.current_uid='new-user-turn';advance_user_control(conv)
            elif change=='epoch':advance_user_control(conv)
            else:conv.session=SimpleNamespace(goal_contract=rt.main.goal_contract,_runtime_snapshot=rt.snapshot)
            release.set();assert not await task
            assert scheduled==[]
            if change=='uid':assert conv.current_uid=='new-user-turn'
        finally:release.set();await task
    asyncio.run(scenario())


def test_api_lifecycle_model_selected_metadata_patch_inherits_prior_user_approval_without_new_user_origin(forward_case,monkeypatch):
    import api
    from agents.parallel_workflow import _valid_execution_approval
    from agents.defns import ORCHESTRATOR
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt)
    main=rt.main;main.current_agent=ORCHESTRATOR
    original=copy.deepcopy(rt.snapshot()['nodes']['n2']);ledger=(rt.root/'recovery_state.json').read_bytes()
    prior=copy.deepcopy(main.context['approved_execution_proof'])
    starts=[]
    def start(version,**options):
        starts.append(version)
        return rt.start(version,**options)
    main._on_workflow_start=start
    assert notify_metadata_capability(rt);event=notices(rt)[0]
    monkeypatch.setattr(api,'_workflow_runtimes',{(conv.username,conv.conv_id):rt})
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda _:rt)
    monkeypatch.setattr(api,'_watch',rt.job_watch)
    monkeypatch.setattr(rt,'tick',lambda *a,**k:None)
    monkeypatch.setattr(api,'_PROCESS_STARTED_AT',0)
    selected=[]
    def model_selects_event_candidate(agent):
        assert main._control_source['origin']=='lifecycle'
        args=copy.deepcopy(event['payload']['recovery']['arguments'])
        main._propose_workflow_patch(**args)
        version=main._pending_workflow_patch['new_version']
        result=main._apply_workflow_patch(version)
        assert result['applied'] and result['ok'],result
        selected.append(result)
        return 'Model selected the currently proved native metadata proposal and applied it.'
    monkeypatch.setattr(main,'_execute_loop',model_selects_event_candidate)
    async def native_turn(username,query,cid,lifecycle_event=None):
        assert lifecycle_event is not None and not query.startswith('new user')
        main._api_turn_id=conv.current_uid
        return await api._durable_io(api._owned_conversation_io,conv,main.resume_lifecycle_event,lifecycle_event)
    monkeypatch.setattr(api,'run_agent_background',native_turn)
    scheduled=[];monkeypatch.setattr(api,'_schedule_lifecycle',scheduled.append)
    async def scenario():
        await api._dispatch_conversation_lifecycle(conv.username,conv,rt.lifecycle)
        assert len(scheduled)==1
        await scheduled[0]
    asyncio.run(scenario())
    assert selected and starts==[3] and rt.snapshot()['plan_version']==3
    proof=main.context['approved_execution_proof']
    assert proof['kind']=='native_metadata_continuation' and proof['control_source']['origin']=='lifecycle'
    assert proof['metadata_continuation']['original_approval']==prior
    assert proof['metadata_continuation']['original_approval']['control_source']['origin']=='user'
    assert _valid_execution_approval(proof,main.goal_contract.to_dict(),rt._line(),rt.username,rt.conv_id)
    after=rt.snapshot()['nodes']['n2']
    for key in ('token','attempt_id','job_ids','result_ref','result','effective_arguments','resource_allocation'):
        assert after.get(key)==original.get(key),key
    assert after['status']=='prefinish' and (rt.root/'recovery_state.json').read_bytes()==ledger and reviews==[]


@pytest.mark.parametrize('change',['science','receipt','different_step','pause','uid'])
def test_lifecycle_metadata_window_never_authorizes_a_different_operation(forward_case,monkeypatch,change):
    rt,aligned,jobs,reviews=forward_case;conv=prepare_api(rt);main=rt.main
    assert notify_metadata_capability(rt);event=notices(rt)[0]
    before=copy.deepcopy(rt.snapshot()['nodes']);ledger=(rt.root/'recovery_state.json').read_bytes()
    def choose(agent):
        args=copy.deepcopy(event['payload']['recovery']['arguments'])
        if change=='science':args['changes'][0]['node']['arguments']['gas']='CO2'
        elif change=='receipt':args['changes'][0]['node']['arguments']['resource_review_id']='unproved-new-receipt'
        elif change=='different_step':args['changes'][0]['step_id']='new-computation';args['changes'][0]['node']['step_id']='new-computation'
        try:main._propose_workflow_patch(**args)
        except (ValueError,PermissionError):return 'Rejected before application.'
        if change=='pause':conv.interrupt_requested=True
        elif change=='uid':conv.current_uid='replacement-user-turn'
        with pytest.raises((ValueError,PermissionError)):
            main._apply_workflow_patch(main._pending_workflow_patch['new_version'])
        return 'Rejected at native application.'
    monkeypatch.setattr(main,'_execute_loop',choose)
    main.resume_lifecycle_event(event)
    assert main._control_source['origin']=='lifecycle'
    assert main.goal_contract.approved_plan_version==2 and rt.snapshot()['nodes']==before
    assert (rt.root/'recovery_state.json').read_bytes()==ledger and reviews==[]
