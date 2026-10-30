"""Independent-review races: final writers, ownership and real session binding."""
import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from agents.lifecycle import LifecycleStore
from tests.test_api_storage_responsiveness import conversation, ready_event


@pytest.mark.parametrize('cancel_count', [1, 2])
def test_final_user_writer_must_stop_before_cancel_releases_conversation_lease(tmp_path, monkeypatch, cancel_count):
    import api
    conv = conversation(LifecycleStore(tmp_path / 'mailbox.json'))
    conv.is_processing = True
    conv.session._persist_chain = lambda _: None
    entered, release, finished = threading.Event(), threading.Event(), threading.Event()
    def persist(*a, **kw):
        entered.set()
        assert release.wait(3)
        finished.set()
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(persist_conversations=persist))
    monkeypatch.setattr(api, '_workflow_events', SimpleNamespace(publish=lambda *a: None))
    lock = threading.Lock();lock.acquire()
    async def owner():
        try:await api._finish_api_turn(conv, 'original', 'owner')
        finally:lock.release()
    async def scenario():
        task = asyncio.create_task(owner())
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            for _ in range(cancel_count):
                task.cancel();await asyncio.sleep(.02)
            acquired = lock.acquire(False)
            if acquired:lock.release()
            assert not acquired, 'lease released while the original final writer was still running'
            assert not finished.is_set()
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(scenario())
    assert finished.is_set() and not lock.locked()


def test_old_lifecycle_io_error_does_not_clear_new_user_busy_state(tmp_path, monkeypatch):
    import api
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event_id = ready_event(store);store.claim(event_id, 'main_chat')
    conv = conversation(store)
    async def old_run(*a, **kw):
        conv.current_uid = 'new-user-turn';conv.is_processing = True
        conv.session._api_turn_id = 'new-user-turn';conv.interrupt_requested = True
        raise OSError('synthetic old-turn chain I/O failure')
    monkeypatch.setattr(api, 'run_agent_background', old_run)
    asyncio.run(api._continue_lifecycle_event(conv, store.snapshot()['events'][event_id]))
    assert conv.current_uid == 'new-user-turn' and conv.is_processing is True
    assert conv.session._api_turn_id == 'new-user-turn' and conv.interrupt_requested is True
    assert store.snapshot()['events'][event_id]['main_chat'] == 'retry'


@pytest.mark.parametrize('control', ['uid', 'pause', 'goal'])
def test_reserved_lifecycle_rechecks_owner_when_scheduled_coroutine_actually_starts(tmp_path, monkeypatch, control):
    import api
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event_id = ready_event(store);event = store.snapshot()['events'][event_id]
    conv = conversation(store);queued=[];runs=[]
    monkeypatch.setattr(api, '_schedule_lifecycle', queued.append)
    async def run(*a, **kw):runs.append(conv.current_uid)
    monkeypatch.setattr(api, 'run_agent_background', run)
    async def scenario():
        assert await api._reserve_and_schedule_lifecycle_main(conv, event, store)
        if control == 'uid':conv.current_uid = 'new-user-turn'
        if control == 'pause':conv.interrupt_requested = True
        if control == 'goal':conv.session.goal_contract.version += 1
        for coro in queued:await coro
    asyncio.run(scenario())
    assert runs == []
    assert store.snapshot()['events'][event_id]['main_chat'] == 'retry'
    if control == 'uid':assert conv.current_uid == 'new-user-turn' and conv.is_processing is True
    else:assert conv.is_processing is False and conv.current_uid == 'original'
    if control == 'pause':assert conv.interrupt_requested is True


def bind_real_session(tmp_path, monkeypatch):
    import api
    from auth import ConversationState
    from agents.config import AgentConfig
    from agents.session import Session
    import agents.config
    import agents.workspace
    config = AgentConfig(api_key='test', project_root=tmp_path)
    monkeypatch.setattr(agents.config, 'get_config', lambda: config)
    monkeypatch.setattr(agents.workspace, 'get_config', lambda: config)
    conv = ConversationState(conv_id='bound-case', username='bound-owner')
    conv.session = Session(config=config)
    api._ensure_session(conv)
    return conv, config


def test_hot_real_session_does_not_block_health_by_rebinding_taskline(tmp_path, monkeypatch):
    import api
    from agents.task_line import get_store
    conv, config = bind_real_session(tmp_path, monkeypatch)
    entered, release = threading.Event(), threading.Event()
    original = get_store().begin_line
    def blocked(*a, **kw):
        entered.set();assert release.wait(3)
        return original(*a, **kw)
    monkeypatch.setattr(get_store(), 'begin_line', blocked)
    def release_later():
        if entered.wait(2):time.sleep(.6)
        release.set()
    timer=threading.Thread(target=release_later);timer.start()
    async def scenario():
        work=asyncio.create_task(api._workflow_state_async(conv))
        try:
            await asyncio.sleep(.01)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
                response=await client.get('/health')
            assert response.status_code==200 and not release.is_set()
            view=await work
            assert view['scope']=={'username':'bound-owner','conv_id':'bound-case'}
            assert not entered.is_set(), 'already bound Session performed a second begin_line transaction'
        finally:release.set();await work
    try:asyncio.run(scenario())
    finally:release.set();entered.set();timer.join(3)


def test_real_session_binding_cache_invalidates_on_session_replacement(tmp_path, monkeypatch):
    import api
    from agents.session import Session
    from agents.task_line import get_store
    conv, config = bind_real_session(tmp_path, monkeypatch)
    calls=[];original=get_store().begin_line
    monkeypatch.setattr(get_store(), 'begin_line', lambda *a, **kw:calls.append(1) or original(*a, **kw))
    replacement=Session(config=config);conv.session=replacement
    assert api._ensure_session(conv) is replacement
    assert api._ensure_session(conv) is replacement
    assert calls == [1]
    assert replacement._on_checkpoint is not None


def test_recovery_capability_receipt_is_diagnostic_not_scientific_acceptance(tmp_path, monkeypatch):
    import api
    store=LifecycleStore(tmp_path/'mailbox.json');conv=conversation(store)
    conv.session._pending_user_interaction={'tool':'user_decision'}
    event=store.emit('worker_recovery_ready', {'step_id':'n','recovery':{
        'tool':'inspect_runtime','arguments':{'step_id':'n'}}})
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', 0)
    asyncio.run(api._dispatch_conversation_lifecycle('owner',conv,store))
    receipt=store.snapshot()['events'][event]['supervisor_receipt']
    assert receipt['recovery_capability_verified'] is True
    assert receipt['diagnostic'] is True
    assert 'scientific_review' not in receipt
    assert receipt['suggested_changes'][0]['tool']=='inspect_runtime'
    assert receipt['suggested_changes'][0]['arguments']=={'step_id':'n'}


def test_managed_restart_defers_while_a_final_writer_is_still_pending(monkeypatch):
    import api
    import agents._restart
    class Pending:
        def done(self):return False
    requested=[]
    monkeypatch.setattr(api, '_api_durable_writes', {Pending()})
    monkeypatch.setattr(api, '_workflow_runtimes', {})
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(list_users=lambda: []))
    monkeypatch.setattr(agents._restart, 'request', lambda: requested.append(True))
    async def no_delay(*a):pass
    monkeypatch.setattr(api.asyncio, 'sleep', no_delay)
    asyncio.run(api._perform_backend_restart())
    assert requested == [True]


@pytest.mark.parametrize('boundary,control', [('persist','uid'), ('persist','pause'), ('lease','uid'), ('worker','uid'), ('model_error','uid')])
def test_superseded_or_paused_turn_cannot_start_model_after_async_wait(tmp_path, monkeypatch, boundary, control):
    import api
    import agents._restart
    import agents.workspace
    from concurrent.futures import ThreadPoolExecutor
    conv=conversation(LifecycleStore(tmp_path/'mailbox.json'))
    conv.messages_history=[];conv.partial_results=[];conv.conversation_active=False
    conv.current_reasoning='';conv.current_answer=''
    conv.session.current_agent=SimpleNamespace(name='lead-orchestrator')
    conv.session._persist_chain=lambda *a:None
    model_calls=[]
    entered,release=threading.Event(),threading.Event()
    def start(*a,**kw):
        model_calls.append(True)
        if boundary=='model_error':
            entered.set();assert release.wait(3)
            raise OSError('synthetic old-provider failure after ownership handoff')
        return 'offline reply'
    conv.session.start=start
    count=[]
    def persist(*a,**kw):
        count.append(1)
        if boundary=='persist' and len(count)==1:entered.set();assert release.wait(3)
    class Lock:
        def __init__(self):self.inner=threading.Lock()
        def acquire(self, *a, **kw):
            if boundary=='lease':entered.set();assert release.wait(3)
            return self.inner.acquire(*a,**kw)
        def release(self):self.inner.release()
    lock=Lock()
    class GatedExecutor(ThreadPoolExecutor):
        def __init__(self,*a,**kw):
            self.model_pool=kw.get('max_workers')==api.MAX_THREAD_WORKERS
            super().__init__(*a,**kw)
        def submit(self,fn,*a,**kw):
            if boundary=='worker' and self.model_pool:
                def gated():
                    entered.set();assert release.wait(3)
                    return fn(*a,**kw)
                return super().submit(gated)
            return super().submit(fn,*a,**kw)
    monkeypatch.setattr(api.concurrent.futures,'ThreadPoolExecutor',GatedExecutor)
    user=SimpleNamespace(get_conversation=lambda _:conv)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:user,persist_conversations=persist))
    monkeypatch.setattr(api,'_ensure_session',lambda _:conv.session)
    monkeypatch.setattr(api,'_get_conv_lock',lambda *a:lock)
    monkeypatch.setattr(api,'_execution_status',lambda _:'done')
    monkeypatch.setattr(api,'_append_conversation_log',lambda *a:None)
    monkeypatch.setattr(api,'_workflow_events',SimpleNamespace(publish=lambda *a:None))
    monkeypatch.setattr(agents._restart,'consume',lambda:False)
    monkeypatch.setattr(agents.workspace,'session_root',lambda *a:tmp_path/'absent-session')
    async def scenario():
        work=asyncio.create_task(api.run_agent_background('owner','offline','case'))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            if control=='uid':
                conv.current_uid='new-user-turn';conv.is_processing=True
                conv.current_answer='new answer';conv.current_status='new status'
            else:conv.interrupt_requested=True
        finally:release.set();await work
    asyncio.run(scenario())
    assert model_calls==([True] if boundary=='model_error' else [])
    if control=='uid':
        assert conv.current_uid=='new-user-turn' and conv.is_processing is True
        assert conv.current_answer=='new answer' and conv.current_status=='new status'
    else:assert conv.interrupt_requested is True and conv.is_processing is False


def test_cancelled_conversation_lock_acquire_does_not_leave_an_orphan_lease(tmp_path):
    import api
    from agents.state_io import ConversationLock
    lock=ConversationLock(tmp_path/'conversation.lock')
    assert lock.acquire(False)
    async def scenario():
        waiting=asyncio.create_task(api._acquire_conversation_lease(lock,timeout=1))
        await asyncio.sleep(.02);waiting.cancel();await asyncio.sleep(.02)
        lock.release()
        with pytest.raises(asyncio.CancelledError):await waiting
        assert lock.acquire(False), 'cancelled acquire left a lock with no owner'
        lock.release()
    asyncio.run(scenario())


def test_interrupt_hook_stops_old_turn_even_after_redirect_clears_freeze_flag(tmp_path, monkeypatch):
    conv, _ = bind_real_session(tmp_path, monkeypatch)
    conv.interrupt_requested=False;conv.interrupt_message=''
    conv.current_uid='old-turn';conv.session._api_turn_id='old-turn'
    assert not conv.session._on_interrupt_requested()
    conv.current_uid='new-user-turn'
    assert conv.session._on_interrupt_requested(), 'old model must stop after its API ownership was superseded'
    conv.interrupt_requested=True
    conv.session._on_interrupt_clear()
    assert conv.interrupt_requested is True, 'old actor cannot clear the new owner pause'
    conv.interrupt_requested=False
    conv.session._api_turn_id='new-user-turn'
    assert not conv.session._on_interrupt_requested()


@pytest.mark.parametrize('takeover', [False, True])
def test_initial_user_persist_cancel_cleans_only_its_own_busy_reservation(tmp_path, monkeypatch, takeover):
    import api
    conv=conversation(LifecycleStore(tmp_path/'mailbox.json'))
    conv.messages_history=[];conv.partial_results=[];conv.conversation_active=False
    conv.current_reasoning='';conv.current_answer=''
    entered,release,finished=threading.Event(),threading.Event(),threading.Event()
    persisted=[]
    def persist(*a,**kw):
        if not persisted:
            entered.set();assert release.wait(3)
            finished.set()
        persisted.append((conv.current_uid,conv.is_processing))
    user=SimpleNamespace(get_conversation=lambda _:conv)
    monkeypatch.setattr(api,'session_manager',SimpleNamespace(get_user=lambda _:user,persist_conversations=persist))
    monkeypatch.setattr(api,'_ensure_session',lambda _:conv.session)
    monkeypatch.setattr(api,'_get_conv_lock',lambda *a:pytest.fail('cancelled initial persistence must not reach a model lease'))
    async def scenario():
        task=asyncio.create_task(api.run_agent_background('owner','offline','case'))
        assert await asyncio.to_thread(entered.wait,2)
        conv.interrupt_requested=True
        if takeover:conv.current_uid='new-user-turn'
        task.cancel();release.set()
        with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(scenario())
    assert finished.is_set()
    assert conv.interrupt_requested is True
    if takeover:
        assert conv.current_uid=='new-user-turn' and conv.is_processing is True
        assert len(persisted)==1
    else:
        assert conv.current_uid=='original' and conv.is_processing is False
        assert persisted[-1]==('original',False) and len(persisted)==2
