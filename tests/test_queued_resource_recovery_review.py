"""Independent metadata-order boundary for same-job resource recovery."""
import copy

from agents.lifecycle import obsolete_internal_question_for_recovery
from tests.test_queued_resource_question_recovery import queued_case


def test_queued_proposal_cannot_clear_question_from_a_goal_older_than_the_runtime(tmp_path):
    session, event, runtime, _assessment, _path, _jobs, checkpoints = queued_case(tmp_path)
    runtime['goal_contract']['version'] = 7  # Current main goal is version 6.
    before = copy.deepcopy(session._pending_user_interaction)
    assert not obsolete_internal_question_for_recovery(session, event, runtime)
    assert session._pending_user_interaction == before
    assert session._waiting_for_user_input and checkpoints == []
