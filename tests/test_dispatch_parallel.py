"""Behavioral tests for conversation-level lifecycle scheduling."""
import asyncio
from types import SimpleNamespace
import pytest


def install_conversations(monkeypatch, api, names):
    conversations = {name: SimpleNamespace(conv_id=name,
        session=SimpleNamespace(_lifecycle_store=object())) for name in names}
    manager = SimpleNamespace(list_users=lambda: ['owner'],
        get_user=lambda _: SimpleNamespace(conversations=conversations))
    monkeypatch.setattr(api, 'session_manager', manager)
    return conversations


def test_slow_mailbox_does_not_block_other_conversation(monkeypatch):
    import api
    install_conversations(monkeypatch, api, ['slow', 'fast'])
    async def scenario():
        slow_started = asyncio.Event(); fast_seen = asyncio.Event(); release = asyncio.Event()
        async def dispatch(username, conv, store):
            if conv.conv_id == 'slow':
                slow_started.set(); await release.wait()
            else:
                fast_seen.set()
        monkeypatch.setattr(api, '_dispatch_conversation_lifecycle', dispatch)
        sweep = asyncio.create_task(api._dispatch_lifecycle_once())
        await asyncio.wait_for(slow_started.wait(), 1)
        await asyncio.wait_for(fast_seen.wait(), 1)
        assert not sweep.done()
        release.set(); await asyncio.wait_for(sweep, 1)
        assert not api._lifecycle_dispatch_inflight
    asyncio.run(scenario())


def test_overlapping_sweeps_do_not_run_one_conversation_twice(monkeypatch):
    import api
    install_conversations(monkeypatch, api, ['same'])
    async def scenario():
        entered = asyncio.Event(); release = asyncio.Event(); calls = []
        async def dispatch(username, conv, store):
            calls.append(conv.conv_id); entered.set(); await release.wait()
        monkeypatch.setattr(api, '_dispatch_conversation_lifecycle', dispatch)
        first = asyncio.create_task(api._dispatch_lifecycle_once())
        await asyncio.wait_for(entered.wait(), 1)
        await asyncio.wait_for(api._dispatch_lifecycle_once(), 1)
        assert calls == ['same']
        release.set(); await asyncio.wait_for(first, 1)
        assert not api._lifecycle_dispatch_inflight
    asyncio.run(scenario())


def test_sweep_cancellation_waits_for_children_to_exit(monkeypatch):
    import api
    install_conversations(monkeypatch, api, ['one', 'two'])
    async def scenario():
        entered = 0; all_entered = asyncio.Event(); exited = []
        async def dispatch(username, conv, store):
            nonlocal entered
            entered += 1
            if entered == 2: all_entered.set()
            try: await asyncio.Event().wait()
            finally: exited.append(conv.conv_id)
        monkeypatch.setattr(api, '_dispatch_conversation_lifecycle', dispatch)
        sweep = asyncio.create_task(api._dispatch_lifecycle_once())
        await asyncio.wait_for(all_entered.wait(), 1); sweep.cancel()
        with pytest.raises(asyncio.CancelledError): await sweep
        assert sorted(exited) == ['one', 'two']
        assert not api._lifecycle_dispatch_inflight
    asyncio.run(scenario())


def test_real_slow_mailbox_lock_does_not_delay_other_mailbox(tmp_path, monkeypatch):
    import fcntl
    import time
    import api
    from agents.lifecycle import LifecycleStore
    from agents.state_io import json_transaction
    slow = LifecycleStore(tmp_path / 'slow' / 'lifecycle.json')
    fast = LifecycleStore(tmp_path / 'fast' / 'lifecycle.json')
    slow.initialize(); fast.initialize()
    conversations = {}
    for name, store in [('slow', slow), ('fast', fast)]:
        session = SimpleNamespace(_lifecycle_store=store, _runtime_snapshot=None)
        conversations[name] = SimpleNamespace(conv_id=name, username='owner', session=session,
            interrupt_requested=True, interrupt_message='', is_processing=False)
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(list_users=lambda: ['owner'],
        get_user=lambda _: SimpleNamespace(conversations=conversations)))
    # Exercise the real JSON transaction lock used by mailbox writers while a
    # separate descriptor holds the slow conversation's lock.
    def slow_recovery():
        with json_transaction(slow.path):pass
    monkeypatch.setattr(slow, 'recover_claims', slow_recovery)
    async def scenario():
        fast_done = asyncio.Event(); original = fast.heartbeat
        def fast_heartbeat():
            original(); loop.call_soon_threadsafe(fast_done.set)
        monkeypatch.setattr(fast, 'heartbeat', fast_heartbeat)
        with open(str(slow.path) + '.lock', 'a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            started = time.monotonic(); sweep = asyncio.create_task(api._dispatch_lifecycle_once())
            try:
                await asyncio.wait_for(fast_done.wait(), 2)
                assert time.monotonic() - started < 2
                assert not sweep.done()
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)
            await asyncio.wait_for(sweep, 5)
        assert not api._lifecycle_dispatch_inflight
    loop = asyncio.new_event_loop()
    try:loop.run_until_complete(scenario())
    finally:loop.close()
