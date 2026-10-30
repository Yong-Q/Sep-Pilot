"""A broken workflow must not starve another conversation's dispatcher."""
import asyncio
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('failure_stage', ['mailbox', 'runtime'])
def test_dispatcher_continues_other_conversations_after_failure(tmp_path, monkeypatch, failure_stage, capsys):
    import api
    from agents.lifecycle import LifecycleStore

    ticked = []
    conversations = {}
    for name in ['broken', 'healthy']:
        store = LifecycleStore(tmp_path / name / 'lifecycle.json')
        session = SimpleNamespace(_lifecycle_store=store,
            _runtime_snapshot=lambda: {'status': 'active', 'nodes': {'test': {}}})
        conversations[name] = SimpleNamespace(session=session, conv_id=name, username='owner',
            interrupt_requested=True, interrupt_message='', execution_logs=[])
    if failure_stage == 'mailbox':
        def fail_mailbox():
            raise ValueError('invalid persisted mailbox')
        monkeypatch.setattr(conversations['broken'].session._lifecycle_store, 'recover_claims', fail_mailbox)

    def runtime(conv):
        def tick(jobs, paused, blocked_nodes=(), pause_revision=None):
            assert blocked_nodes == ()
            assert type(pause_revision) is int and pause_revision >= 0
            from agents.watch_context import get_context
            scope=get_context()
            assert scope['username']==conv.username and scope['conv_id']==conv.conv_id
            assert paused is conv.interrupt_requested
            if conv.conv_id == 'broken' and failure_stage == 'runtime':
                raise ValueError('native output receipt differs from the approved session path')
            ticked.append(conv.conv_id)
        return SimpleNamespace(tick=tick)

    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(
        list_users=lambda: ['owner'], get_user=lambda _: SimpleNamespace(conversations=conversations)))
    monkeypatch.setattr(api, '_watch', SimpleNamespace(list=lambda: []))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', runtime)
    asyncio.run(api._dispatch_lifecycle_once())
    assert ticked == ['healthy']
    assert conversations['healthy'].interrupt_requested  # isolation never clears controls
    assert 'broken' in capsys.readouterr().out


def test_dispatcher_does_not_swallow_service_cancellation(tmp_path, monkeypatch):
    import api
    from agents.lifecycle import LifecycleStore
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    def cancelled():
        raise asyncio.CancelledError()
    monkeypatch.setattr(store, 'recover_claims', cancelled)
    conv = SimpleNamespace(session=SimpleNamespace(_lifecycle_store=store), conv_id='cancelled')
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(
        list_users=lambda: ['owner'], get_user=lambda _: SimpleNamespace(conversations={'cancelled': conv})))
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(api._dispatch_lifecycle_once())
