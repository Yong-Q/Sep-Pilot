from agents.session import Session


def test_explicit_continue_selects_saved_contract_patch_for_replay():
    session = Session()
    session.goal_contract.version = 2
    session._current_line_id = "conversation-1"
    session._error_branches = {
        "error_branch_1": {
            "branch_id": "error_branch_1",
            "failed_step": "propose_workflow_patch",
            "category": "contract_validation",
            "status": "open",
            "failed_arguments": {
                "base_version": 0,
                "reason": "replay the complete saved plan",
                "changes": [{"operation": "upsert", "step_id": "generate"}],
            },
        }
    }

    candidate = session._initial_patch_recovery_candidate("继续执行")

    assert candidate["base_version"] == 0
    assert candidate["changes"][0]["step_id"] == "generate"


def test_patch_recovery_does_not_replay_for_unrelated_messages():
    session = Session()
    session._error_branches = {
        "error_branch_1": {
            "failed_step": "propose_workflow_patch",
            "category": "contract_validation",
            "status": "open",
            "failed_arguments": {"base_version": 0, "changes": []},
        }
    }

    assert session._initial_patch_recovery_candidate("先介绍一下方法") is None
