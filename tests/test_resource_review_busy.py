from types import SimpleNamespace


def test_existing_reviewer_is_coordination_wait_not_failed_review(tmp_path, monkeypatch):
    import api
    monkeypatch.setattr('agents.workspace.session_root', lambda *args: tmp_path)
    class BusyLock:
        def __init__(self, path): pass
        def acquire(self, blocking=False): return False
    monkeypatch.setattr('agents.state_io.ConversationLock', BusyLock)
    monkeypatch.setattr('agents.resource_review.run_resource_review',
                        lambda *args: (_ for _ in ()).throw(AssertionError('must not start duplicate reviewer')))
    conv = SimpleNamespace(conv_id='c', username='u')
    receipt = api._review_resources(conv, {'kind': 'before_submission'})
    assert receipt['status'] == 'review_busy'
    assert receipt['review_in_progress'] is True and receipt['retryable'] is True
    assert 0 < receipt['retry_after_seconds'] <= 30
