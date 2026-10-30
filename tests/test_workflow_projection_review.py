"""Independent observable UI contracts for the thin projection."""
import json
import asyncio
import threading

import pytest

from agents.workflow_projection import thin_workflow


def test_thin_graph_keeps_retained_revision_metadata():
    raw = {'graph_projection': {'source': 'persisted_chain', 'version': 3,
        'revision': 19, 'retained': True,
        'nodes': [{'step_id': 'complete', 'status': 'succeeded'}]}}
    graph = thin_workflow(raw)['graph_projection']
    assert graph.get('retained') is True
    assert graph.get('revision') == 19
    assert graph['nodes'][0]['status'] == 'succeeded'


def test_thin_lifecycle_preserves_stale_heartbeat_and_processing_action():
    raw = {'lifecycle': {'agents': {
        'main_chat': {'agent': 'lead-orchestrator', 'service_heartbeat': 1000,
                      'last_action': 'processing', 'busy_event_id': None},
        'supervisor': {'agent': 'supervisor', 'service_heartbeat': 1005,
                       'last_action': 'restored', 'read_only': True},
    }}}
    agents = thin_workflow(raw)['lifecycle']['agents']
    assert agents['main_chat'].get('service_heartbeat') == 1000
    assert agents['main_chat'].get('last_action') == 'processing'
    assert agents['supervisor'].get('service_heartbeat') == 1005
    assert agents['supervisor'].get('last_action') == 'restored'


def test_legacy_live_report_preview_has_same_bound_as_native_preview():
    text = 'measured finite path; convergence not established.\n' * 100000
    raw = {'parallel_runtime': {'nodes': {}, 'live_reports': [{
        'step_id': 'report', 'role': 'final', 'status': 'succeeded',
        'report_file': '/owned/full-scientific-report.md', 'content': text,
    }]}}
    view = thin_workflow(raw)
    report = view['parallel_runtime']['live_reports'][0]
    assert report['content'] == text[:24000]
    assert report['report_file'] == '/owned/full-scientific-report.md'
    assert len(json.dumps(view)) < 30000
    assert raw['parallel_runtime']['live_reports'][0]['content'] == text


def test_repeated_producer_cancellation_does_not_release_permit_before_native_reader_exits():
    from agents.workflow_projection import ProjectionCache
    cache = ProjectionCache()
    entered, release = threading.Event(), threading.Event()
    submitted = []
    real_submit = cache._executor.submit
    def tracked_submit(function, *args, **kwargs):
        submitted.append(function)
        return real_submit(function, *args, **kwargs)
    cache._executor.submit = tracked_submit
    def first_builder():
        entered.set()
        assert release.wait(3)
        return {'scope': 'first'}
    async def scenario():
        first = asyncio.create_task(cache.get(('owner', 'one'), lambda: 1, first_builder))
        assert await asyncio.to_thread(entered.wait, 2)
        producer = cache._flights[('owner', 'one')]
        producer.cancel()
        await asyncio.sleep(.01)
        producer.cancel()
        other = asyncio.create_task(cache.get(('owner', 'two'), lambda: 1, lambda: {'scope': 'second'}))
        try:
            await asyncio.sleep(.03)
            # This asserts submissions to the real native executor, not just
            # the executor's own single-worker concurrency guarantee.
            assert len(submitted) == 1
        finally:
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await first
            assert await other == {'scope': 'second'}
        assert len(submitted) == 2
    try:
        asyncio.run(scenario())
    finally:
        release.set()
        cache.close()
