"""Blocked durable storage must not occupy the API loop or duplicate a turn."""
import asyncio
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from agents.lifecycle import LifecycleStore


def conversation(store):
    session = SimpleNamespace(_lifecycle_store=store, _runtime_snapshot=None,
        _awaiting_plan_approval=False, _pending_user_interaction=None,
        goal_contract=SimpleNamespace(version=1), _current_line_id='line')
    return SimpleNamespace(session=session, username='owner', conv_id='case',
        is_processing=False, current_uid='original', current_agent='lead-orchestrator',
        last_progress_at=0, interrupt_requested=False, interrupt_message='', current_status='', title='fixture', execution_logs=[])


def ready_event(store):
    event = store.emit('job_completed', {'origin_goal_version': 1, 'step_id': 'n', 'correlation_token': 'old-token'})
    store.receipt(event, 'supervisor', 'delivered', {'next_action': 'advance_dependencies'})
    return event


@pytest.mark.parametrize('boundary', ['recover_claims', 'heartbeat', 'pending', 'runtime_snapshot',
                                       'jobs', 'receipt', 'claim', 'persist', 'workflow_projection'])
def test_health_responds_while_storage_operation_is_blocked(tmp_path, monkeypatch, boundary):
    import api
    store = LifecycleStore(tmp_path / 'mailbox.json')
    conv = conversation(store)
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', 0)
    scheduled = []
    monkeypatch.setattr(api, '_schedule_lifecycle', lambda coro: scheduled.append(coro.close()))
    entered, release = threading.Event(), threading.Event()
    blocked_threads = []
    loop_thread = threading.get_ident()
    def wrap(original):
        def blocked(*args, **kwargs):
            blocked_threads.append(threading.get_ident())
            entered.set()
            assert release.wait(3), 'fixture release timeout'
            return original(*args, **kwargs)
        return blocked
    if boundary in {'recover_claims', 'heartbeat', 'pending'}:
        monkeypatch.setattr(store, boundary, wrap(getattr(store, boundary)))
    elif boundary == 'runtime_snapshot':
        conv.session._runtime_snapshot = wrap(lambda: {})
    elif boundary == 'jobs':
        conv.session._runtime_snapshot = lambda: {'status': 'active', 'nodes': {'n': {}}}
        monkeypatch.setattr(api, '_ensure_workflow_runtime', lambda _: SimpleNamespace(tick=lambda *a: None))
        monkeypatch.setattr(api, '_watch', SimpleNamespace(list=wrap(lambda: [])))
    elif boundary in {'receipt', 'claim'}:
        ready_event(store)
        if boundary == 'receipt':
            conv.session._pending_user_interaction = {'tool': 'user_decision'}
            monkeypatch.setattr(store, 'receipt', wrap(store.receipt))
        else:
            monkeypatch.setattr(store, 'claim', wrap(store.claim))
    else:
        user = SimpleNamespace(get_conversation=lambda _: conv)
        monkeypatch.setattr(api, 'session_manager', SimpleNamespace(get_user=lambda _: user,
            persist_conversations=wrap(lambda *a, **kw: None) if boundary == 'persist' else lambda *a, **kw: None))
        if boundary == 'workflow_projection':
            monkeypatch.setattr(api, '_ensure_session', lambda _: None)
            monkeypatch.setattr(api, '_workflow_state', wrap(lambda *a, **kw: {}))
    def release_later():
        if entered.wait(3):
            time.sleep(.6)
        release.set()
    timer = threading.Thread(target=release_later)
    timer.start()
    async def scenario():
        work = asyncio.create_task(api.rename_conversation('case', {'title': 'renamed'}, username='owner')
            if boundary == 'persist' else api.get_workflow('case', username='owner')
            if boundary == 'workflow_projection' else api._dispatch_conversation_lifecycle('owner', conv, store))
        try:
            deadline = time.monotonic() + 2
            while not entered.is_set() and time.monotonic() < deadline:
                await asyncio.sleep(.001)
            assert entered.is_set()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
                response = await client.get('/health')
            assert response.status_code == 200
            assert not release.is_set(), 'health waited for the blocked storage operation'
            assert blocked_threads == [value for value in blocked_threads if value != loop_thread]
        finally:
            release.set()
            await work
    try:
        asyncio.run(scenario())
    finally:
        release.set(); timer.join(3)


@pytest.mark.parametrize('control', ['pause', 'runtime_pause', 'superseded', 'new_goal', 'new_uid', 'cancel', 'other_dispatch'])
def test_pending_durable_main_claim_keeps_loop_reservation_and_preserves_controls(tmp_path, monkeypatch, control):
    import api
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event_id = ready_event(store)
    conv = conversation(store)
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', 0)
    scheduled = []
    def schedule(coro):
        scheduled.append((threading.get_ident(), conv.current_uid));coro.close()
    monkeypatch.setattr(api, '_schedule_lifecycle', schedule)
    entered, release = threading.Event(), threading.Event()
    original = store.claim
    def blocked_claim(*a):
        entered.set()
        assert release.wait(3)
        return original(*a)
    monkeypatch.setattr(store, 'claim', blocked_claim)
    async def scenario():
        work = asyncio.create_task(api._dispatch_conversation_lifecycle('owner', conv, store))
        try:
            assert await asyncio.to_thread(entered.wait, 2)
            assert conv.is_processing is True
            assert conv.current_uid != 'original'
            if control == 'pause':conv.interrupt_requested = True
            if control == 'runtime_pause':conv.session._runtime_snapshot = lambda: {'user_paused': True}
            if control == 'superseded':conv.session._runtime_snapshot = lambda: {'nodes': {'n': {'token': 'new-token'}}}
            if control == 'new_goal':conv.session.goal_contract.version += 1
            if control == 'new_uid':conv.current_uid = 'new-user-turn'
            if control == 'cancel':work.cancel()
            if control == 'other_dispatch':
                await api._dispatch_conversation_lifecycle('owner', conv, store)
            release.set()
            if control == 'cancel':
                with pytest.raises(asyncio.CancelledError):await work
            else:await work
        finally:
            release.set()
            if not work.done():await work
    asyncio.run(scenario())
    current = store.snapshot()
    if control == 'other_dispatch':
        assert len(scheduled) == 1
        assert scheduled[0][0] == threading.get_ident()
        assert conv.is_processing is True
        assert current['events'][event_id]['main_chat'] == 'processing'
    else:
        assert scheduled == []
        assert current['events'][event_id]['main_chat'] == 'retry'
        assert current['agents']['main_chat'].get('busy_event_id') is None
        if control == 'new_uid':assert conv.current_uid == 'new-user-turn' and conv.is_processing is True
        else:assert conv.current_uid == 'original' and conv.is_processing is False
        if control == 'pause':assert conv.interrupt_requested is True


def test_cancellation_after_claim_commit_releases_only_its_durable_claim(tmp_path, monkeypatch):
    import api
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event_id = ready_event(store)
    conv = conversation(store)
    entered, release = threading.Event(), threading.Event()
    original = api._claim_lifecycle_receiver
    def blocked_snapshot():
        entered.set()
        assert release.wait(3)
        return {}
    async def claimed_then_snapshot(*args):
        result = await original(*args)
        conv.session._runtime_snapshot = blocked_snapshot
        return result
    monkeypatch.setattr(api, '_claim_lifecycle_receiver', claimed_then_snapshot)
    monkeypatch.setattr(api, '_schedule_lifecycle', lambda coro: pytest.fail('cancelled event must not schedule'))
    async def scenario():
        task = asyncio.create_task(api._reserve_and_schedule_lifecycle_main(conv, store.snapshot()['events'][event_id], store))
        assert await asyncio.to_thread(entered.wait, 2)
        assert store.snapshot()['events'][event_id]['main_chat'] == 'processing'
        task.cancel(); release.set()
        with pytest.raises(asyncio.CancelledError):await task
    asyncio.run(scenario())
    after = store.snapshot()
    assert after['events'][event_id]['main_chat'] == 'retry'
    assert after['agents']['main_chat'].get('busy_event_id') is None
    assert conv.is_processing is False and conv.current_uid == 'original'


@pytest.mark.parametrize('superseded', [False, True, 'cancel'])
def test_turn_end_chain_storage_keeps_loop_responsive_and_new_owner_intact(tmp_path, monkeypatch, superseded):
    import api
    import agents._restart
    import agents.workspace
    store = LifecycleStore(tmp_path / 'mailbox.json')
    conv = conversation(store)
    conv.messages_history = [];conv.partial_results = [];conv.conversation_active = False
    conv.current_reasoning = '';conv.current_answer = ''
    conv.session.current_agent = SimpleNamespace(name='lead-orchestrator')
    conv.session.start = lambda *a, **kw: 'offline reply'
    loop_thread = threading.get_ident()
    entered, release = threading.Event(), threading.Event()
    def persist_chain(reason):
        assert reason == 'api_turn_end'
        entered.set()
        assert release.wait(3)
    conv.session._persist_chain = persist_chain
    user = SimpleNamespace(get_conversation=lambda _: conv)
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(get_user=lambda _: user,
        persist_conversations=lambda *a, **kw: None))
    monkeypatch.setattr(api, '_ensure_session', lambda _: conv.session)
    lock = threading.Lock()
    monkeypatch.setattr(api, '_get_conv_lock', lambda *a: lock)
    monkeypatch.setattr(api, '_execution_status', lambda _: 'done')
    monkeypatch.setattr(api, '_append_conversation_log', lambda *a: None)
    monkeypatch.setattr(api, '_workflow_events', SimpleNamespace(publish=lambda *a:
        pytest.fail('control publish moved off loop') if threading.get_ident() != loop_thread else None))
    monkeypatch.setattr(agents._restart, 'consume', lambda: False)
    monkeypatch.setattr(agents.workspace, 'session_root', lambda *a: tmp_path / 'absent-session')
    def release_later():
        if entered.wait(3):time.sleep(.6)
        release.set()
    timer = threading.Thread(target=release_later);timer.start()
    async def scenario():
        task = asyncio.create_task(api.run_agent_background('owner', 'offline question', 'case'))
        try:
            deadline = time.monotonic() + 2
            while not entered.is_set() and time.monotonic() < deadline:await asyncio.sleep(.001)
            assert entered.is_set()
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=api.app), base_url='http://test') as client:
                response = await client.get('/health')
            assert response.status_code == 200 and not release.is_set()
            assert conv.is_processing is True
            if superseded is True:
                conv.current_uid = 'new-owner';conv.session._api_turn_id = 'new-owner'
                conv.interrupt_requested = True
            if superseded == 'cancel':task.cancel()
        finally:
            release.set()
            if superseded == 'cancel':
                with pytest.raises(asyncio.CancelledError):await task
            else:await task
    try:asyncio.run(scenario())
    finally:release.set();timer.join(3)
    assert lock.acquire(blocking=False), 'turn cancellation leaked the conversation lease'
    lock.release()
    if superseded is True:
        assert conv.current_uid == 'new-owner' and conv.is_processing is True
        assert conv.session._api_turn_id == 'new-owner' and conv.interrupt_requested is True
    else:
        assert conv.is_processing is False and conv.session._api_turn_id is None
