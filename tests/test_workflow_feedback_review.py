"""Independent current-evidence and original-guard diagnostic boundaries."""
import pytest

from agents.parallel_workflow import WorkflowStore
from tests.test_workflow_guard_feedback import fixture


def test_nonmapping_guard_details_cannot_replace_the_original_finish_exception(tmp_path):
    session, _runtime = fixture(tmp_path)
    original = ValueError('original native verification guard')
    original.details = 'nonmapping adapter diagnostics'
    def blocked(*args):
        raise original
    session._on_workflow_finish = blocked
    with pytest.raises(ValueError) as failure:
        session._finish_workflow_node('validate', [], 'Preserve the real native guard.')
    assert failure.value is original


def test_dossier_with_an_older_scheduler_identity_is_not_presented_as_current(tmp_path):
    from agents.workflow_feedback import current_workflow_feedback
    session, runtime = fixture(tmp_path)
    runtime['plan_version'] = 9
    contract = {'tool': 'read_file', 'arguments': {'path': 'owned.csv'}}
    dossier = {'workflow_id': WorkflowStore.identity('u', 'c'), 'step_id': 'validate',
        'plan_version': 9, 'attempt_id': 'attemptA', 'contract': contract,
        'job_ids': ['previous_job'], 'evidence_call_id': 'olderJobProof',
        'evidence_path': str(session._evidence_root / 'olderJobProof.json')}
    runtime['nodes']['validate'].update(status='prefinish', contract=contract,
        job_ids=['current_job'], validation_dossier=dossier)
    context = current_workflow_feedback(session, 'validate')
    assert context['nodes'][0]['job_ids'] == ['current_job']
    assert 'current_dossier' not in context
