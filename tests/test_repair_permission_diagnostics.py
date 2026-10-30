from agents.failure_recovery import FailureRecovery

def test_fixed_permission_reasons_only():
    classify=FailureRecovery._repair_permission_code
    assert classify(PermissionError('repair planning cannot submit, finish, approve retry or mutate workflow'))=='repair_tool_not_allowed'
    assert classify(PermissionError('recovery evidence path escapes its owned session scope'))=='recovery_evidence_outside_scope'
    assert classify(PermissionError('private path or arbitrary user text'))=='permission_denied_cause_unclassified'
    assert classify(ValueError('permission denied')) is None

def test_permission_error_is_scoped_and_does_not_grant_retry(tmp_path):
    from tests.test_repair_event_isolation import setup
    import pytest
    machine,store,actor,root=setup(tmp_path)
    def runner(_):raise PermissionError('recovery evidence path escapes its owned session scope')
    with pytest.raises(PermissionError):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=runner)
    state=store.snapshot('workflow');incident=state['failure_recovery']['incidents']['incident-one']
    assert incident['phase']=='blocked' and incident['history'][-1]['permission_code']=='recovery_evidence_outside_scope'
    assert state['nodes']['one']['status']=='failed' and 'retry_granted' not in incident
    assert state['failure_recovery']['incidents']['incident-two']['phase']=='repair_required'


def test_arbitrary_permission_message_is_not_persisted(tmp_path):
    from tests.test_repair_event_isolation import setup
    import pytest,json
    machine,store,actor,root=setup(tmp_path)
    def runner(_):raise PermissionError('private path or arbitrary user text')
    with pytest.raises(PermissionError):
        machine.run_repair(actor(),'incident-one',4,checkpoint_path=root/'one.json',stage_runner=runner)
    history=store.snapshot('workflow')['failure_recovery']['incidents']['incident-one']['history']
    assert history[-1]['permission_code']=='permission_denied_cause_unclassified'
    assert 'private path or arbitrary user text' not in json.dumps(history)
