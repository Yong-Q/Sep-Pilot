from agents.config import AgentConfig
from agents.session import Session


def test_incomplete_workflow_review_does_not_prohibit_authorized_repair(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._execute_loop = lambda _: s.messages[-1]['content']
    message = s.resume_lifecycle_event({'event_id': 'worker-failed', 'kind': 'worker_failed',
        'payload': {'step_id': 'generate'}, 'supervisor_receipt': {
            'next_action': 'diagnose_and_fix', 'scientific_review': {'passed': False, 'issues': ['downstream results not ready']}}})
    assert 'new_submission_allowed' not in message


def test_actual_scientific_review_still_protects_against_unapproved_recompute(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._execute_loop = lambda _: s.messages[-1]['content']
    message = s.resume_lifecycle_event({'event_id': 'review-failed', 'kind': 'scientific_review_failed',
        'payload': {}, 'supervisor_receipt': {'scientific_review': {'passed': False, 'issues': ['unsupported claim']}}})
    assert 'new_submission_allowed' in message
