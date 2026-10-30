"""Guard failures report current facts, without granting control or retries."""
import copy
import json
from types import SimpleNamespace

import pytest

from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.session import Session
from agents.workflow_patch import WorkflowContractError


def fixture(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._evidence_root = tmp_path / 'runs/u/c/evidence'
    session.goal_contract = GoalContract.from_dict({'version': 4, 'approved_plan_version': 9,
        'execution_authorized': True, 'execution_mode': 'workflow'})
    session._checkpoint = lambda *_: None
    session._emit_lifecycle_event = lambda *_: None
    runtime = {'username': 'u', 'conv_id': 'c', 'plan_version': 7, 'status': 'active',
               'goal_contract': {'version': 4}, 'nodes': {'validate': {
                   'status': 'running', 'token': 'attemptA', 'job_ids': [],
                   'contract': {'tool': 'validate_framework_charges', 'arguments': {'data': 'x'*100000}}}}}
    session._runtime_snapshot = lambda: runtime
    return session, runtime


def test_activation_failure_reports_current_version_and_waiting_work_not_generic_replan(tmp_path):
    s, runtime = fixture(tmp_path)
    error = WorkflowContractError('approved plan has an existing live owner', {'cause': 'owner_alive'})
    def blocked(*_): raise error
    s._on_workflow_start = blocked
    before = copy.deepcopy(runtime)
    receipt = s._activate_approved_workflow()
    context = receipt['current_workflow']
    assert context['approved_plan_version'] == 9 and context['runtime_plan_version'] == 7
    assert context['plan_activation_pending'] is True
    assert context['waiting_on_current_attempts'] == ['validate']
    assert context['nodes'][0]['token'] == 'attemptA'
    assert context['diagnostic_only'] is True
    assert receipt['scheduled'] is False and receipt['legacy_fallback_allowed'] is False
    assert receipt['guard_details']['cause'] == 'owner_alive'
    assert receipt['next_action'] != 'reconcile_legacy_dispatch_or_propose_workflow_patch'
    assert runtime == before and len(json.dumps(context)) < 5000


def test_fresh_failure_context_changes_epoch_when_old_wait_condition_finishes(tmp_path):
    s, runtime = fixture(tmp_path)
    def blocked(*_): raise ValueError('same prior rejection')
    s._on_workflow_start = blocked
    first = s._activate_approved_workflow()['current_workflow']
    runtime['nodes']['validate']['status'] = 'succeeded'
    second = s._activate_approved_workflow()['current_workflow']
    assert first['state_epoch'] != second['state_epoch']
    assert second['waiting_on_current_attempts'] == []


def test_finish_error_keeps_original_exception_and_exposes_only_current_dossier(tmp_path):
    s, runtime = fixture(tmp_path)
    from agents.parallel_workflow import WorkflowStore
    runtime['plan_version'] = 9
    contract = {'tool': 'read_file', 'arguments': {'path': 'owned.csv'}}
    dossier = {'workflow_id': WorkflowStore.identity('u', 'c'), 'step_id': 'validate',
        'plan_version': 9, 'attempt_id': 'attemptA', 'contract': contract,
        'evidence_call_id': 'currentProof', 'evidence_path': str(s._evidence_root / 'currentProof.json')}
    runtime['nodes']['validate'].update(status='prefinish', contract=contract, validation_dossier=dossier)
    original = ValueError('verification must use current successful inspection')
    def blocked(*_): raise original
    s._on_workflow_finish = blocked
    with pytest.raises(ValueError) as failure:
        s._finish_workflow_node('validate', ['oldProof'], 'Inspect the current result.')
    assert failure.value is original
    context = failure.value.details['current_workflow']
    assert context['current_dossier'] == {'call_id': 'currentProof', 'evidence_path': dossier['evidence_path']}
    assert 'retry_allowed' not in context and 'may_finish' not in context
    runtime['nodes']['validate']['validation_dossier']['attempt_id'] = 'oldAttempt'
    with pytest.raises(ValueError) as failure:
        s._finish_workflow_node('validate', ['oldProof'], 'Keep the original guard.')
    assert 'current_dossier' not in failure.value.details['current_workflow']


@pytest.mark.parametrize('problem', ['foreign', 'unavailable'])
def test_foreign_or_unavailable_snapshots_do_not_expose_facts_or_mask_original_error(tmp_path, problem):
    s, runtime = fixture(tmp_path)
    if problem == 'foreign': runtime['username'] = 'other'
    else:
        def unavailable(): raise OSError('unavailable checkpoint')
        s._runtime_snapshot = unavailable
    original = ValueError('original ownership guard')
    def blocked(*_): raise original
    s._on_workflow_finish = blocked
    with pytest.raises(ValueError) as failure:
        s._finish_workflow_node('validate', [], 'Must not read another owner.')
    assert failure.value is original
    assert 'current_workflow' not in getattr(original, 'details', {})


def test_successful_activation_does_not_add_snapshot_io(tmp_path):
    s, _ = fixture(tmp_path)
    s._on_workflow_start = lambda _: {'status': 'scheduled', 'scheduled': True}
    def unexpected(): pytest.fail('success path must not read additional workflow state')
    s._runtime_snapshot = unexpected
    assert s._activate_approved_workflow()['scheduled'] is True


def test_failure_context_caps_node_rows_and_job_ids_without_copying_payloads(tmp_path):
    s, runtime = fixture(tmp_path)
    from agents.workflow_feedback import current_workflow_feedback
    runtime['nodes'] = {str(i): {'status': 'waiting_jobs', 'token': str(i),
        'job_ids': [str(n) for n in range(1000)], 'result': 'x' * 100000}
        for i in range(200)}
    context = current_workflow_feedback(s)
    assert context['waiting_attempt_count'] == 200 and len(context['waiting_on_current_attempts']) == 32
    assert context['nodes_truncated'] is True and len(context['nodes']) == 32
    assert len(json.dumps(context)) < 16000


def test_unavailable_snapshot_cannot_retain_a_prior_current_context(tmp_path):
    s, runtime = fixture(tmp_path)
    original = ValueError('unchanged original rejection')
    original.details = {'error': str(original), 'diagnostic_code': 'native_guard',
                        'current_workflow': {'scope': {'username': 'other'}}}
    def blocked(*_): raise original
    s._on_workflow_finish = blocked
    runtime['username'] = 'other'
    with pytest.raises(ValueError) as failure:
        s._finish_workflow_node('validate', [], 'Keep scope isolation.')
    assert 'current_workflow' not in failure.value.details
    assert failure.value.details['diagnostic_code'] == 'native_guard'
