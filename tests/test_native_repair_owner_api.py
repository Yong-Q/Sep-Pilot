"""Native lifecycle tick keeps owner scope across reused API worker threads.

Synthetic conversations and scheduler records only. No provider, scientific
executor, authentication account or production workflow is loaded by fixtures.
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agents.lifecycle import LifecycleStore
from agents.watch_context import clear_context, get_context, set_context


@pytest.mark.parametrize('existing_scope', [False, True])
def test_native_lifecycle_tick_binds_owner_and_restores_reused_thread(tmp_path, monkeypatch, existing_scope):
    import api

    class AfterNativeTick(Exception):
        pass

    seen = []
    jobs = [
        {'username': 'demo-owner-a', 'conv_id': 'case-a', 'job_id': '42'},
        {'username': 'demo-owner-b', 'conv_id': 'case-b', 'job_id': '43'},
    ]
    conversations = [SimpleNamespace(username=item['username'], conv_id=item['conv_id'],
        session=SimpleNamespace(_runtime_snapshot=lambda: {'user_pause_revision': 7},
            _current_line_id='line-' + item['conv_id']), interrupt_requested=False)
        for item in jobs]

    class TickBridge:
        def __init__(self, conversation):
            self.conversation = conversation

        def tick(self, current_jobs, interrupted, events, revision):
            current = get_context()
            conversation = self.conversation
            assert (current['username'], current['conv_id']) == (conversation.username, conversation.conv_id)
            assert current['agent_name'] == 'api_state'
            assert current['line_id'] == 'line-' + conversation.conv_id
            assert current_jobs == [item for item in jobs if item['username'] == conversation.username]
            assert interrupted is False and events == () and revision == 7
            seen.append((current['username'], current['conv_id']))
            raise AfterNativeTick()

    monkeypatch.setattr(api, '_watch', SimpleNamespace(list=lambda: jobs))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', TickBridge)
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    store.initialize()

    async def dispatch_with_one_reused_worker():
        asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(max_workers=1))
        if existing_scope:
            await asyncio.to_thread(set_context, 'demo-prior-owner', 'prior-case', 'prior-agent',
                'prior-line', tool_name='read_file', recovery_key='prior-key',
                attempt_id='prior-attempt', recovery_path='prior-path', step_id='prior-step',
                resource_allocation={'cpus': 2, 'memory_mb': 1024})
        else:
            await asyncio.to_thread(clear_context)
        before = await asyncio.to_thread(get_context)
        assert bool(before) is existing_scope
        try:
            for conversation in conversations:
                with pytest.raises(AfterNativeTick):
                    await api._dispatch_conversation_lifecycle(conversation.username, conversation, store)
                assert await asyncio.to_thread(get_context) == before
        finally:
            await asyncio.to_thread(clear_context)

    main_thread_scope = get_context()
    asyncio.run(dispatch_with_one_reused_worker())
    assert seen == [('demo-owner-a', 'case-a'), ('demo-owner-b', 'case-b')]
    assert get_context() == main_thread_scope


def test_owned_api_io_normal_return_restores_exact_empty_scope():
    import api

    clear_context()
    conversation = SimpleNamespace(username='demo-owner', conv_id='case',
        session=SimpleNamespace(_current_line_id='approved-line'))
    def operation(value):
        assert get_context()['username'] == 'demo-owner'
        assert get_context()['conv_id'] == 'case'
        return value
    assert api._owned_conversation_io(conversation, operation, 'observed') == 'observed'
    assert get_context() == {}
