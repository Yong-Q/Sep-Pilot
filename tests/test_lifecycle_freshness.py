from agents.lifecycle import superseded_runtime_event


def test_old_attempt_event_cannot_reenter_current_workflow():
    runtime = {'nodes': {'generate': {'token': 'current', 'attempt_id': 'new-attempt',
                                     'recovery_key': 'same-task', 'status': 'failed'}}}
    assert superseded_runtime_event({'kind': 'worker_uncertain', 'payload': {
        'step_id': 'generate', 'correlation_token': 'old'}}, runtime)
    assert superseded_runtime_event({'kind': 'job_failed', 'payload': {'job': {
        'recovery_key': 'same-task', 'attempt_id': 'old-attempt'}}}, runtime)
    assert not superseded_runtime_event({'kind': 'worker_failed', 'payload': {
        'step_id': 'generate', 'correlation_token': 'current'}}, runtime)


def test_legacy_runtime_event_uses_native_event_id_attempt_token():
    runtime = {'nodes': {'generate': {'token': 'b' * 32, 'status': 'failed'}}}
    assert superseded_runtime_event({'event_id': 'runtime:workflow:' + 'a' * 32 + ':failed',
        'kind': 'worker_failed', 'payload': {'step_id': 'generate', 'runtime_id': 'workflow'}}, runtime)


def test_completed_state_supersedes_old_phase_of_same_attempt():
    runtime = {'nodes': {'generate': {'token': 'current', 'status': 'succeeded'}}}
    assert superseded_runtime_event({'kind': 'worker_failed', 'payload': {
        'step_id': 'generate', 'correlation_token': 'current'}}, runtime)
    assert not superseded_runtime_event({'kind': 'worker_completed', 'payload': {
        'step_id': 'generate', 'correlation_token': 'current'}}, runtime)
    assert not superseded_runtime_event({'kind': 'user_paused', 'payload': {}}, runtime)
    assert not superseded_runtime_event({'kind': 'worker_failed', 'payload': {'step_id': 'unknown'}}, runtime)
