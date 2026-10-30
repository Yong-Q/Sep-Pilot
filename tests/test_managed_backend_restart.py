import asyncio
from types import SimpleNamespace


def test_managed_restart_signals_existing_compute_supervisor(tmp_path, monkeypatch):
    import api
    marker = tmp_path / 'restart.json'
    monkeypatch.setenv('BIMEM_MANAGED_RESTART_FILE', str(marker))
    monkeypatch.setattr(api, 'session_manager', SimpleNamespace(list_users=lambda: []))
    monkeypatch.setattr(api, '_workflow_runtimes', {})
    async def no_sleep(_):
        pass
    monkeypatch.setattr(api.asyncio, 'sleep', no_sleep)
    monkeypatch.setattr(api.subprocess, 'Popen', lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError('managed backend must not spawn an unallocated replacement')))
    asyncio.run(api._perform_backend_restart())
    assert marker.exists()
