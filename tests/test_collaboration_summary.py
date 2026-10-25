from agents.parallel_workflow import runtime_summary


def test_runtime_summary_exposes_branch_relationships_for_main_agent():
    state = {
        "status": "active", "plan_version": 2,
        "nodes": {
            "charge": {"status": "succeeded", "contract": {"agent": "analyst", "tool": "run_pacman_charge", "depends_on": []}, "result_ref": {"call_id": "a"}},
            "pore": {"status": "succeeded", "contract": {"agent": "adsorption", "tool": "run_pore_analysis", "depends_on": []}, "result_ref": {"call_id": "b"}},
            "join": {"status": "pending", "contract": {"agent": "analyst", "tool": "ml_template_plan", "depends_on": ["charge", "pore"]}},
        },
    }
    summary = runtime_summary(state)
    assert summary["nodes"]["join"]["depends_on"] == ["charge", "pore"]
    assert summary["nodes"]["join"]["upstream_receipts"] == {"charge": "a", "pore": "b"}
    assert summary["collaboration_mode"] == "parallel_dag"


def test_collaboration_mode_detects_branches_after_shared_preparation():
    state = {"status": "active", "plan_version": 1, "nodes": {
        "prepare": {"status": "succeeded", "contract": {"depends_on": []}},
        "gas_a": {"status": "pending", "contract": {"depends_on": ["prepare"]}},
        "gas_b": {"status": "pending", "contract": {"depends_on": ["prepare"]}},
    }}
    assert runtime_summary(state)["collaboration_mode"] == "parallel_dag"
