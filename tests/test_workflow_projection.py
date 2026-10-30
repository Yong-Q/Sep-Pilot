"""The API view contains current contracts, not repeated native/history payloads."""
import asyncio
import json
import threading
from types import SimpleNamespace

import pytest


def test_thin_projection_preserves_science_dependencies_paths_and_refs_without_history():
    from agents.workflow_projection import thin_workflow
    huge='native output'*100000
    contract={'step_id':'calc','tool':'run_string_tst','agent':'adsorption',
              'arguments':{'gas':'CO2','temperature':298,'running_steps':20},'depends_on':['input'],
              'expected_outputs':[{'kind':'file','path':'string_results.json'}]}
    raw={'scope':{'username':'a','conv_id':'one'},'goal_contract':{'version':2,'parameters':{'temperature_K':298},'approved_nodes':[contract]},
         'parallel_runtime':{'username':'a','conv_id':'one','status':'active','nodes':{'calc':{'contract':contract,'status':'succeeded','job_ids':['123'],
            'path_manifest':{'result_dir':'/scope/results','evidence_path':'/scope/evidence'},'evidence_refs':[{'call_id':'call1','path':'/scope/evidence'}],
            'result':{'stdout':huge},'evidence_call':{'raw':huge},'attempt_history':[huge]}}},
         'lifecycle':{'agents':{'main_chat':{'agent':'lead','status':'active'}},'events':{'old':{'payload':huge}}},
         'lines':[{'line_id':'line','steps':[{**contract,'status':'succeeded','result':huge}], 'history':[huge]}],
         'memory_files':{'chain':'/scope/orchestration_chain.json'},'recent_operations':[{'call_id':'c','tool':'read_file','params':{'text':huge},'evidence_path':'/scope/c'}],
         'outbox':{'a':huge},'history':[huge]}
    result=thin_workflow(raw)
    assert len(json.dumps(result))<12000
    node=result['parallel_runtime']['nodes']['calc']
    assert node['contract']==contract
    assert node['job_ids']==['123'] and node['evidence_refs'][0]['call_id']=='call1'
    assert node['path_manifest']['result_dir']=='/scope/results'
    assert result['goal_contract']['parameters']=={'temperature_K':298}
    assert result['memory_files']==raw['memory_files']
    assert 'result' not in node and 'history' not in result and 'events' not in result['lifecycle']
    assert 'params' not in result['recent_operations'][0]


def test_singleflight_global_bound_cancel_waiter_and_cached_mutation_isolation():
    from agents.workflow_projection import ProjectionCache
    cache=ProjectionCache();entered=threading.Event();release=threading.Event();calls=[];active=0;peak=0
    def build(scope):
        def run():
            nonlocal active,peak
            active+=1;peak=max(peak,active);calls.append(scope);entered.set();assert release.wait(3)
            active-=1;return {'scope':scope,'nodes':[]}
        return run
    async def scenario():
        old=asyncio.create_task(cache.get(('a','one'),lambda:1,build('one')))
        assert await asyncio.to_thread(entered.wait,2)
        old.cancel()
        with pytest.raises(asyncio.CancelledError):await old
        same=[asyncio.create_task(cache.get(('a','one'),lambda:1,build('one'))) for _ in range(4)]
        other=asyncio.create_task(cache.get(('a','two'),lambda:1,build('two')))
        await asyncio.sleep(.03)
        assert calls==['one']
        release.set();first,*_=await asyncio.gather(*same,other)
        first['nodes'].append('caller mutation')
        again=await cache.get(('a','one'),lambda:1,build('one'))
        assert again['nodes']==[]
        assert calls==['one','two'] and peak==1
    asyncio.run(scenario());cache.close()


def test_changed_control_during_build_is_not_returned_or_cached_and_scope_isolated():
    from agents.workflow_projection import ProjectionCache
    cache=ProjectionCache();entered=threading.Event();release=threading.Event();control=[1];calls=[]
    def build():
        value=control[0];calls.append(value)
        if value==1:entered.set();assert release.wait(3)
        return {'version':value}
    async def scenario():
        task=asyncio.create_task(cache.get(('a','one'),lambda:control[0],build))
        assert await asyncio.to_thread(entered.wait,2)
        control[0]=2;release.set()
        assert await task=={'version':2}
        assert calls==[1,2]
        assert await cache.get(('b','one'),lambda:2,lambda:{'version':'other owner'})=={'version':'other owner'}
    asyncio.run(scenario());cache.close()


def test_cache_byte_limit_and_oversize_and_corruption_never_cached():
    from agents.workflow_projection import ProjectionCache
    cache=ProjectionCache(max_bytes=100,max_item_bytes=60);calls=[]
    async def scenario():
        for i in range(12):
            await cache.get(('owner',str(i)),lambda:1,lambda:{'text':'x'*25})
            assert cache.cached_bytes<=100
        def big():calls.append('big');return {'text':'x'*100}
        await cache.get(('o','big'),lambda:1,big);await cache.get(('o','big'),lambda:1,big)
        assert calls==['big','big'] and cache.cached_bytes<=100
        def corrupt():calls.append('bad');raise ValueError('corrupt source')
        for _ in range(2):
            with pytest.raises(ValueError,match='corrupt'):await cache.get(('o','bad'),lambda:1,corrupt)
        assert calls[-2:]==['bad','bad']
    asyncio.run(scenario());cache.close()


def test_actual_asgi_concurrent_workflow_reads_build_once_without_starving_default_io(monkeypatch):
    import api
    import httpx
    from concurrent.futures import ThreadPoolExecutor
    conv=SimpleNamespace(username='owner',conv_id='case',session=None,current_uid='u',
                         is_processing=False,interrupt_requested=False,interrupt_message='',current_agent='lead',current_status='ready')
    user=SimpleNamespace(get_conversation=lambda _:conv)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:user))
    def ensure_readonly_session(current,*,publish=True,_expected_source=None):
        assert current is conv and publish is False
        assert _expected_source == api._session_admission_source(conv)
        return None
    monkeypatch.setattr(api,'_ensure_session',ensure_readonly_session)
    calls=[];entered=threading.Event();release=threading.Event()
    def build(*a,**kw):
        calls.append(1);entered.set();assert release.wait(3)
        return {'scope':{'username':'owner','conv_id':'case'},'history':['x'*2000000],
                'goal_contract':{'version':1},'parallel_runtime':{'nodes':{}}}
    monkeypatch.setattr(api,'_workflow_state',build)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as client:
            tasks=[asyncio.create_task(client.get('/api/conversations/case/workflow')) for _ in range(8)]
            try:
                for _ in range(100):
                    if entered.is_set():break
                    await asyncio.sleep(.005)
                assert entered.is_set()
                # Projection waiters/builders cannot occupy the default pool
                # used by checkpoint/user durable I/O.
                assert await asyncio.wait_for(asyncio.to_thread(lambda:'durable IO free'),.2)=='durable IO free'
            finally:
                release.set();responses=await asyncio.gather(*tasks)
            assert calls==[1]
            assert all(r.status_code==200 and len(r.content)<5000 for r in responses)
            again=await client.get('/api/conversations/case/workflow')
            assert again.status_code==200 and calls==[1]
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()


def test_signature_tracks_scope_session_pause_goal_broker_and_both_chain_files(tmp_path,monkeypatch):
    from agents.workflow_projection import projection_signature
    from agents.config import AgentConfig
    import agents.workspace
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr(agents.workspace,'get_config',lambda:config)
    goal=SimpleNamespace(version=1,parameters={'temperature':298},approved_nodes=[])
    session=SimpleNamespace(config=config,goal_contract=goal,_current_line_id='line')
    conv=SimpleNamespace(username='owner',conv_id='case',session=session,current_uid='u',interrupt_requested=False)
    previous=projection_signature(conv,1)
    for change in (lambda:setattr(conv,'interrupt_requested',True),lambda:setattr(goal,'version',2),
                   lambda:goal.parameters.update(temperature=300),lambda:setattr(conv,'current_uid','new'),
                   lambda:setattr(conv,'session',SimpleNamespace(**vars(session)))):
        change();current=projection_signature(conv,1);assert current!=previous;previous=current
    assert projection_signature(conv,2)!=previous
    root=tmp_path/'runs/owner/case';root.mkdir(parents=True)
    for name in ('orchestration_chain.json','orchestration_current.json','lifecycle.json','session_checkpoint.json'):
        (root/name).write_text('{}');current=projection_signature(conv,1)
        assert current!=previous;previous=current
    conv.username='different-owner';assert projection_signature(conv,1)!=previous


def test_thin_dto_memory_does_not_copy_large_native_history():
    from agents.workflow_projection import thin_workflow
    import tracemalloc
    # Allocate the synthetic preserved evidence before measuring the view, so
    # the assertion measures duplication rather than ownership of the input.
    history='x'*(24*1024*1024)
    raw={'parallel_runtime':{'nodes':{'n':{'status':'running','result':history,'attempt_history':[history]}}},
         'lifecycle':{'events':{'old':{'payload':history}}},'history':[history]}
    tracemalloc.start()
    try:
        result=thin_workflow(raw);encoded=json.dumps(result)
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert len(encoded)<5000 and peak<1024*1024


def test_signature_does_not_serialize_unrelated_pending_evidence(monkeypatch):
    from agents.workflow_projection import projection_signature
    class Evidence:
        def __str__(self):raise AssertionError('UI signature serialized historical evidence')
    goal=SimpleNamespace(version=1,parameters={},approved_nodes=[])
    session=SimpleNamespace(goal_contract=goal,_pending_user_interaction={'tool':'confirm','raw_evidence':Evidence()})
    conv=SimpleNamespace(username='owner',conv_id='case',session=session)
    assert projection_signature(conv,1)


def test_actual_asgi_failed_projection_is_not_cached(monkeypatch):
    import api
    import httpx
    conv=SimpleNamespace(username='owner',conv_id='broken',session=None,current_uid='u',is_processing=False)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:SimpleNamespace(get_conversation=lambda _:conv)))
    def ensure_readonly_session(current,*,publish=True,_expected_source=None):
        assert current is conv and publish is False
        assert _expected_source == api._session_admission_source(conv)
        return None
    monkeypatch.setattr(api,'_ensure_session',ensure_readonly_session)
    calls=[]
    def corrupt(*a,**kw):calls.append(1);raise ValueError('corrupt persisted graph')
    monkeypatch.setattr(api,'_workflow_state',corrupt)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app,raise_app_exceptions=False),base_url='http://test') as client:
            for _ in range(2):assert (await client.get('/api/conversations/broken/workflow')).status_code==500
            assert calls==[1,1]
            monkeypatch.setattr(api,'_workflow_state',lambda *a,**kw:{'scope':{'username':'owner','conv_id':'broken'}})
            response=await client.get('/api/conversations/broken/workflow')
            assert response.status_code==200 and response.json()['scope']['conv_id']=='broken'
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()


def test_foreign_runtime_cannot_leak_when_no_taskline_exists(monkeypatch):
    import api
    import agents.task_line
    session=SimpleNamespace(goal_contract=SimpleNamespace(to_dict=lambda:{}),_error_branches={},_awaiting_plan_approval=False,
        _current_line_id='line',_runtime_snapshot=lambda:{'username':'other','conv_id':'private','nodes':{}},
        context={},memory=SimpleNamespace(tool_call_log=[]),recovery_gate=SimpleNamespace(snapshot=lambda:{}),
        _lifecycle_store=None,_pending_user_interaction=None,_pending_workflow_patch=None,_evidence_root=None)
    conv=SimpleNamespace(username='owner',conv_id='case',session=session,current_agent='lead',current_status='ready',is_processing=False)
    monkeypatch.setattr(agents.task_line,'get_store',lambda:SimpleNamespace(get_line=lambda *a,**kw:None,get_by_conv=lambda *a,**kw:[]))
    with pytest.raises(PermissionError,match='scope'):
        api._workflow_state(conv,ensure_session=False)


def test_corrupt_taskline_cannot_be_cached_as_an_empty_success(monkeypatch):
    import api
    import agents.task_line
    session=SimpleNamespace(goal_contract=SimpleNamespace(to_dict=lambda:{}),_error_branches={},_awaiting_plan_approval=False,
        _current_line_id='line',_runtime_snapshot=None,context={},memory=SimpleNamespace(tool_call_log=[]),
        recovery_gate=SimpleNamespace(snapshot=lambda:{}),_lifecycle_store=None,
        _pending_user_interaction=None,_pending_workflow_patch=None,_evidence_root=None)
    conv=SimpleNamespace(username='owner',conv_id='case',session=session,current_agent='lead',current_status='ready',is_processing=False)
    def corrupt(*a,**kw):raise ValueError('corrupt taskline')
    monkeypatch.setattr(agents.task_line,'get_store',lambda:SimpleNamespace(get_line=corrupt))
    with pytest.raises(ValueError,match='corrupt taskline'):
        api._workflow_state(conv,ensure_session=False)


def test_latest_answer_outer_controls_are_read_after_the_coalesced_build(monkeypatch):
    import api
    import httpx
    goal=SimpleNamespace(version=1,parameters={},approved_nodes=[])
    conv=SimpleNamespace(username='owner',conv_id='control',session=SimpleNamespace(goal_contract=goal),current_uid='old',
        is_processing=True,interrupt_requested=False,interrupt_message='',current_agent='lead',current_status='running',
        current_answer='',current_reasoning='',execution_logs=[],partial_results=[],queued_redirect='')
    monkeypatch.setattr(api,'_get_conv',lambda *a:conv)
    monkeypatch.setattr(api,'_ensure_session',lambda _:None)
    entered=threading.Event();release=threading.Event()
    def build(*a,**kw):
        version=goal.version
        if version==1:entered.set();assert release.wait(3)
        return {'scope':{'username':'owner','conv_id':'control'},'goal_contract':{'version':version}}
    monkeypatch.setattr(api,'_workflow_state',build)
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as client:
            task=asyncio.create_task(client.get('/api/latest_answer?conv_id=control'))
            try:
                assert await asyncio.to_thread(entered.wait,2)
                conv.current_uid='new';conv.is_processing=False;conv.interrupt_requested=True;goal.version=2
            finally:release.set()
            response=await task;data=response.json()
            assert response.status_code==200
            assert data['uid']=='new' and data['done'] is True and data['interrupt_requested'] is True
            assert data['workflow']['goal_contract']['version']==2
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()


def test_replacement_session_waiter_cannot_receive_the_previous_producers_payload():
    from agents.workflow_projection import ProjectionCache
    cache=ProjectionCache();entered=threading.Event();release=threading.Event()
    def old():entered.set();assert release.wait(3);return {'session':'old'}
    async def scenario():
        first=asyncio.create_task(cache.get(('owner','case'),lambda:'old session',old))
        assert await asyncio.to_thread(entered.wait,2)
        replacement=asyncio.create_task(cache.get(('owner','case'),lambda:'new session',lambda:{'session':'new'}))
        await asyncio.sleep(.01);release.set()
        await first
        assert await replacement=={'session':'new'}
    asyncio.run(scenario());cache.close()


@pytest.mark.parametrize('mismatch',[False,True])
def test_real_workflow_view_keeps_binding_lock_contracts_and_runtime_verification(monkeypatch,mismatch):
    import api
    import agents.task_line
    contract={'step_id':'calc','agent':'adsorption','tool':'run_string_tst','arguments':{'gas':'CO2'},
        'depends_on':['producer'],'expected_outputs':['result.json'],'input_bindings':{'cif_dir':{'step_id':'producer','output_index':0}},
        'resource_locks':['path:/fixtures/cifs'],'resources':[{'key':'path:/fixtures/cifs','mode':'read'}]}
    actual={**contract,'input_bindings':{'cif_dir':{'step_id':'different' if mismatch else 'producer','output_index':0}}}
    verification={'evidence_call_ids':['v1'],'conclusion':'interface smoke; not convergence','verified_at':123,'owner':'adsorption'}
    runtime={'username':'owner','conv_id':'bindings','plan_version':1,'status':'completed','nodes':{'calc':{
        'contract':actual,'status':'succeeded','job_ids':['job-fixture'],'node_verification':verification,
        'resolved_arguments':{'gas':'CO2','output_dir':'/fixtures/new-results'},
        'runtime_input_bindings':[{'producer':'producer','path':'/fixtures/cifs'}],'runtime_binding_fingerprint':'sha'}}}
    line={'line_id':'line','username':'owner','conv_id':'bindings','plan_version':1,'steps':[{**contract,'status':'completed'}]}
    goal=SimpleNamespace(version=1,parameters={},approved_nodes=[contract],to_dict=lambda:{'version':1,'approved_nodes':[contract]})
    session=SimpleNamespace(goal_contract=goal,_error_branches={},_awaiting_plan_approval=False,_current_line_id='line',
        _runtime_snapshot=lambda:runtime,context={},memory=SimpleNamespace(tool_call_log=[]),recovery_gate=SimpleNamespace(snapshot=lambda:{}),
        _lifecycle_store=None,_pending_user_interaction=None,_pending_workflow_patch=None,_evidence_root=None)
    conv=SimpleNamespace(username='owner',conv_id='bindings',session=session,current_agent='lead',current_status='ready',is_processing=False)
    monkeypatch.setattr(api,'_ensure_session',lambda _:None)
    monkeypatch.setattr(agents.task_line,'get_store',lambda:SimpleNamespace(get_line=lambda *a,**kw:line))
    view=asyncio.run(api._workflow_state_async(conv));step=view['lines'][0]['steps'][0]
    assert step['contract_matches_execution'] is (not mismatch)
    assert step['status']==('pending' if mismatch else 'succeeded')
    assert step['input_bindings']==contract['input_bindings'] and step['resource_locks']==contract['resource_locks']
    assert step['resources']==contract['resources']
    node=view['parallel_runtime']['nodes']['calc']
    assert node['node_verification']==verification
    assert node['resolved_arguments']['output_dir']=='/fixtures/new-results'
    assert node['runtime_binding_fingerprint']=='sha' and node['runtime_input_bindings']


def test_actual_workflow_builder_reads_each_native_source_once_for_concurrent_gets(tmp_path,monkeypatch):
    import api
    import httpx
    import agents.task_line
    from agents.lifecycle import LifecycleStore
    store=LifecycleStore(tmp_path/'large-mailbox.json')
    store.path.write_text(json.dumps({'agents':{'main_chat':{'agent':'lead','status':'active'}},
        'events':{'old':{'event_id':'old','kind':'historical','payload':{'native':'x'*2000000}}}}))
    counts={'mailbox':0,'runtime':0,'line':0}
    snapshot=store.snapshot
    def mailbox():counts['mailbox']+=1;return snapshot()
    monkeypatch.setattr(store,'snapshot',mailbox)
    contract={'step_id':'calc','agent':'analyst','tool':'inspect_path','arguments':{'path':'/fixtures'},'expected_outputs':['/fixtures']}
    def runtime():
        counts['runtime']+=1
        return {'username':'owner','conv_id':'native','status':'active','nodes':{'calc':{'contract':contract,'status':'running','result':{'native':'x'*2000000}}}}
    def line(*a,**kw):
        counts['line']+=1
        return {'line_id':'line','username':'owner','conv_id':'native','steps':[contract]}
    monkeypatch.setattr(agents.task_line,'get_store',lambda:SimpleNamespace(get_line=line))
    session=SimpleNamespace(goal_contract=SimpleNamespace(version=1,approved_nodes=[contract],to_dict=lambda:{'approved_nodes':[contract]}),
        _error_branches={},_awaiting_plan_approval=False,_current_line_id='line',_runtime_snapshot=runtime,context={},
        memory=SimpleNamespace(tool_call_log=[]),recovery_gate=SimpleNamespace(snapshot=lambda:{}),_lifecycle_store=store,
        _pending_user_interaction=None,_pending_workflow_patch=None,_evidence_root=None)
    conv=SimpleNamespace(username='owner',conv_id='native',session=session,current_agent='lead',current_status='ready',is_processing=False)
    monkeypatch.setattr(api,'_ensure_session',lambda _:None)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:SimpleNamespace(get_conversation=lambda _:conv)))
    api.app.dependency_overrides[api._get_user]=lambda:'owner'
    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app),base_url='http://test') as client:
            replies=await asyncio.gather(*(client.get('/api/conversations/native/workflow') for _ in range(8)))
            assert all(r.status_code==200 and len(r.content)<10000 for r in replies)
            assert (await client.get('/api/conversations/native/workflow')).status_code==200
            assert counts=={'mailbox':1,'runtime':1,'line':1}
            assert replies[0].json()['lifecycle']['agents']['main_chat']['agent']=='lead'
    try:asyncio.run(scenario())
    finally:api.app.dependency_overrides.clear()


def test_lifecycle_heartbeat_sidecar_invalidates_the_projection(tmp_path):
    from agents.workflow_projection import projection_signature
    mailbox=tmp_path/'selected-mailbox.json'
    session=SimpleNamespace(goal_contract=SimpleNamespace(version=1,parameters={},approved_nodes=[]),
        _lifecycle_store=SimpleNamespace(path=mailbox))
    conv=SimpleNamespace(username='owner',conv_id='heartbeat',session=session)
    before=projection_signature(conv,1)
    (tmp_path/'selected-mailbox.json.heartbeat.json').write_text('{"main_chat":{"service_heartbeat":123}}')
    assert projection_signature(conv,1)!=before
