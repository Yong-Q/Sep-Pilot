"""A completed stale flight must not starve the HTTP event loop."""
import subprocess
import sys
import textwrap


def test_completed_stale_flight_retires_before_its_callback_runs():
    # Isolate the regression: the old loop never yields, so an asyncio timeout
    # inside that same loop cannot stop it or report the failed health check.
    program = textwrap.dedent('''
        import asyncio, json
        from agents.workflow_projection import ProjectionCache, _StaleProjection

        async def scenario(exception):
            cache = ProjectionCache()
            scope = ('owner', 'conversation')
            cache._loop_state()
            async def completed():
                if exception:
                    raise _StaleProjection()
                return ('old', json.dumps({'revision': 'old'}).encode())
            task = asyncio.create_task(completed())
            try:
                await task
            except _StaleProjection:
                pass
            # A request may run before an already queued cleanup callback.
            cache._flights[scope] = task
            def cleanup():
                if cache._flights.get(scope) is task:
                    cache._flights.pop(scope)
            asyncio.get_running_loop().call_soon(cleanup)
            builds = []
            try:
                result = await cache.get(scope, lambda: 'new',
                    lambda: builds.append(1) or {'revision': 'new'})
                assert result == {'revision': 'new'}
                assert builds == [1]
            finally:
                cache.close()
        asyncio.run(scenario(False))
        asyncio.run(scenario(True))
    ''')
    result = subprocess.run([sys.executable, '-c', program], capture_output=True,
                            text=True, timeout=4)
    assert result.returncode == 0, result.stderr
