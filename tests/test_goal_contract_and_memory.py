"""Regression coverage for evaluator4 goal drift and durable memory."""
from pathlib import Path
import json

from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.goal_contract import GoalContract, extract_gases
from agents.registry import get_registry
from agents import registry as registry_module
from agents.session import Session
from agents.task_line import TaskLineStore
import auth


EVALUATOR4_GOAL = (
    "用PORMAKE生成42个u-HOF结构，用cDFT计算298K的Kr和Xe Henry系数，"
    "拓扑用utk,hxg,bto,cds,cdt,srs,eta。直接开始执行。"
)


def test_evaluator4_goal_contract_blocks_gas_and_method_drift():
    goal = GoalContract.from_user_message(EVALUATOR4_GOAL)
    assert goal.gases == ["Kr", "Xe"]
    assert goal.method == "CDFT"
    assert goal.parameters["temperature_K"] == 298.0
    assert goal.parameters["n_structures"] == 42
    assert goal.parameters["topologies"][:3] == ["utk", "hxg", "bto"]
    assert goal.execution_authorized

    ok, reason = goal.guard_tool_call(
        "generate_structure",
        {
            "material_type": "HOF", "n_structures": 42, "max_atoms": 1500,
            "topologies": ["utk", "hxg", "bto", "cds", "cdt", "srs", "eta"],
        },
    )
    assert ok, reason
    assert not goal.guard_tool_call(
        "generate_structure",
        {"material_type": "HOF", "n_structures": 42, "max_atoms": 1500, "topologies": ["dia"]},
    )[0]

    ok, _ = goal.guard_tool_call(
        "run_cdft", {"action": "pipeline", "gases": ["Kr", "Xe"], "temperature": 298}
    )
    assert ok

    ok, reason = goal.guard_tool_call(
        "run_gcmc_batch", {"cif_dir": "charged", "gases": ["CO2", "SO2"], "temperature": 298}
    )
    assert not ok
    assert "CO2" in reason and "SO2" in reason

    ok, reason = goal.guard_tool_call(
        "handoff_to_adsorption", {"task": "改做CO2/SO2 GCMC计算"}
    )
    assert ok  # Delegation prose is interpreted by the model, not keyword-gated.
    assert not goal.guard_tool_call('run_gcmc_batch', {'gases':['CO2','SO2']})[0]


def test_explicit_user_method_change_is_versioned_but_keeps_gases():
    goal = GoalContract.from_user_message(EVALUATOR4_GOAL)
    assert goal.apply_user_message("方案A，改用run_henry_chain逐个计算Kr和Xe")
    assert goal.version == 2
    assert goal.method == "HENRY"
    assert goal.gases == ["Kr", "Xe"]
    assert goal.guard_tool_call(
        "run_henry_chain", {"material": "u-HOF-1", "gas": "Kr", "temperature": 298}
    )[0]


def test_followup_constraints_update_without_erasing_original_target():
    goal = GoalContract.from_user_message(
        "筛选u-HOF用于Kr/Xe分离，用cDFT计算298K Henry系数"
    )
    goal.apply_user_message(
        "方案1，修改脚本支持指定拓扑。每个拓扑6个，共42个。原子数限制1500以内。直接执行。"
    )
    assert goal.original_goal.startswith("筛选u-HOF")
    assert goal.gases == ["Kr", "Xe"]
    assert goal.method == "CDFT"
    assert goal.parameters["n_structures"] == 42
    assert goal.parameters["max_atoms"] == 1500
    assert goal.execution_authorized


def test_natural_autonomous_research_mandate_authorizes_initial_dag_execution():
    goal = GoalContract.from_user_message(
        "这是用户已授权的自主科研模式，建立完整DAG并自主推进到最终报告"
    )
    assert goal.execution_mode == "workflow"
    assert goal.execution_authorized
    assert not goal.requires_plan_approval

    pending = GoalContract.from_user_message("建立完整DAG，计算后生成报告")
    assert not pending.execution_authorized
    assert pending.apply_user_message("agent自己解决并推进完成，不再重复确认")
    assert pending.execution_authorized


def test_hydrocarbon_mixture_and_framework_name_are_parsed_without_gas_collision():
    assert extract_gases("HKUST-1 中 C2H4/C2H6 分离") == {"C2H4", "C2H6"}
    assert extract_gases("Co-MOF-74 中 CO2/CH4 吸附") == {"CO2", "CH4"}
    goal = GoalContract.from_user_message(
        "用 GCMC 计算 MFI 中 C2H4/C2H6 在 298K、0.1-10 bar 的等温线"
    )
    assert goal.gases == ["C2H4", "C2H6"]
    assert goal.parameters["material"].upper() == "MFI"


def test_negated_drift_gases_are_removed_from_active_contract():
    goal = GoalContract.from_user_message(EVALUATOR4_GOAL)
    goal.apply_user_message("改算CO2/SO2 GCMC")
    assert goal.gases == ["CO2", "SO2"]
    goal.apply_user_message(
        "停！原始任务是Kr/Xe分离，不是SO2/CO2。请立刻停止所有SO2/CO2相关作业，"
        "回到正确方向：用cDFT计算Kr和Xe Henry系数（298K）。"
    )
    assert goal.gases == ["Kr", "Xe"]
    assert goal.method == "CDFT"


def test_multistep_plan_requires_approval_unless_direct_execution():
    pending = GoalContract.from_user_message(
        "先生成MOF，然后赋电荷，再用cDFT计算Kr/Xe并分析选择性"
    )
    assert pending.requires_plan_approval
    assert not pending.execution_authorized
    assert not pending.guard_tool_call(
        "run_cdft", {"action":"pipeline", "gases":["Kr","Xe"]}
    )[0]
    pending.propose_plan()
    pending.apply_user_message("确认执行")
    pending.approve_pending_plan()
    assert pending.execution_authorized
    assert pending.approved_plan_version == 1


def test_session_checkpoint_restores_goal_and_agent_memory():
    config = AgentConfig(api_key="unit-test")
    session = Session(config=config)
    session.current_agent = ORCHESTRATOR
    session.messages = [{"role": "user", "content": EVALUATOR4_GOAL}]
    session.goal_contract = GoalContract.from_user_message(EVALUATOR4_GOAL)
    session.memory.record_tool_call(
        "analyst", "run_cdft", {"gases": ["Kr", "Xe"]}, '{"job_id":"42"}'
    )

    restored = Session(config=config)
    assert restored.import_state(session.export_state())
    assert restored.goal_contract.gases == ["Kr", "Xe"]
    assert restored.goal_contract.method == "CDFT"
    assert restored.memory.agent_memories["analyst"]["tool_call_log"][0]["params"] == {
        "gases": ["Kr", "Xe"]
    }


def test_newer_transcript_directive_overrides_stale_checkpoint():
    config = AgentConfig(api_key="unit-test")
    stale = Session(config=config)
    stale.current_agent = ORCHESTRATOR
    stale.messages = [
        {"role": "user", "content": EVALUATOR4_GOAL},
        {"role": "assistant", "content": "old"},
        {"role": "user", "content": "方案A，改用run_henry_chain计算Kr/Xe"},
    ]
    stale.goal_contract = GoalContract.from_user_message(EVALUATOR4_GOAL)
    stale.goal_contract.apply_user_message("方案A，改用run_henry_chain计算Kr/Xe")
    assert stale.goal_contract.method == "HENRY"

    restored = Session(config=config)
    assert restored.import_state(stale.export_state())
    merged = restored.reconcile_transcript(stale.messages + [{
        "role": "user",
        "content": "停！不是SO2/CO2。回到正确方向：用cDFT计算Kr和Xe Henry系数（298K）。",
    }])
    assert merged == 1
    assert restored.goal_contract.method == "CDFT"
    assert restored.goal_contract.gases == ["Kr", "Xe"]


def test_registry_rejects_unscientific_parameters_before_executor():
    registry = get_registry()
    issues = registry.validate_params(
        "run_gcmc_isotherm",
        {
            "cif": "sample.cif", "gas": "Kr", "temperature": -1,
            "pressure_start": 10, "pressure_end": 1,
        },
    )
    assert any("temperature" in issue for issue in issues)
    assert any("pressure_start" in issue for issue in issues)


def test_gcmc_schema_separates_electrostatics_from_charge_assignment():
    registry = get_registry()
    schema = registry.get("run_gcmc_isotherm").input_schema
    assert "electrostatics_method" in schema["properties"]
    assert "charge_method" not in schema["properties"]
    assert schema["properties"]["electrostatics_method"]["enum"] == ["Ewald", "Wolf"]
    # Persisted old workflow nodes remain executable through a compatibility alias.
    assert not registry.validate_params("run_gcmc_isotherm", {
        "cif": "sample.cif", "gas": "CO2", "temperature": 298,
        "pressure_start": 0.1, "pressure_end": 10, "charge_method": "Ewald",
    })


def test_uncharged_gcmc_routes_to_pacmof_instead_of_solver_switch(tmp_path: Path, monkeypatch):
    cif = tmp_path / "HKUST-1.cif"
    cif.write_text("data_HKUST\n_cell_length_a 10\n")
    monkeypatch.setattr(registry_module.slurm, "_find_charged_cif", lambda path: path)
    monkeypatch.setattr(
        registry_module.slurm, "submit_gcmc_isotherm",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("uncharged CIF must not submit")),
    )
    result = registry_module._exec_gcmc_isotherm({
        "cif": str(cif), "gas": "C2H4", "temperature": 298,
        "pressure_start": 0.1, "pressure_end": 10,
        "electrostatics_method": "Ewald",
    })
    assert result["error_code"] == "FRAMEWORK_CHARGES_REQUIRED"
    assert result["next_action"]["tool"] == "run_pacman_charge"
    assert result["next_action"]["arguments"]["method"] == "pacmof"
    assert "不是电荷赋值" in result["explanation"]


def test_taskline_persists_structured_dependencies(tmp_path: Path):
    store = TaskLineStore(str(tmp_path / "taskline.json"))
    store.begin_line("c1", username="u", conv_id="c1", title="plan v1")
    store.upsert_step(
        "c1", "step_1", tool="run_pacman_charge", agent="analyst",
        depends_on=[], status="pending", plan_version=1,
    )
    store.upsert_step(
        "c1", "step_2", tool="run_cdft", agent="analyst",
        depends_on=["step_1"], status="pending", plan_version=1,
        arguments={"gases": ["Kr", "Xe"]},
    )
    restored = TaskLineStore(str(tmp_path / "taskline.json"))
    line = restored.get_line("c1")
    assert line["steps"][1]["depends_on"] == ["step_1"]
    assert line["steps"][1]["arguments"] == {"gases": ["Kr", "Xe"]}


def test_concurrent_managers_merge_different_conversations(tmp_path: Path, monkeypatch):
    user_db = tmp_path / "users.json"
    token_db = tmp_path / "tokens.json"
    monkeypatch.setattr(auth, "USER_DB", user_db)
    monkeypatch.setattr(auth, "TOKEN_DB", token_db)
    user_db.write_text(json.dumps({"u": {"display_name": "u", "conversations": {}}}))

    m1 = auth.UserSessionManager()
    m2 = auth.UserSessionManager()
    u1 = m1.get_user("u")
    u2 = m2.get_user("u")
    c1 = u1.new_conversation()
    c2 = u2.new_conversation()
    c1.title = "from worker 1"
    c2.title = "from worker 2"

    m1.persist_conversations("u", conv_id=c1.conv_id)
    m2.persist_conversations("u", conv_id=c2.conv_id)
    stored = json.loads(user_db.read_text())["u"]["conversations"]
    assert stored[c1.conv_id]["title"] == "from worker 1"
    assert stored[c2.conv_id]["title"] == "from worker 2"


def test_structure_generator_uses_local_topology_aware_script(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_submit(**kwargs):
        captured.update(kwargs)
        return {"submitted": False, "work_dir": kwargs["work_dir"]}

    monkeypatch.setattr(registry_module.slurm, "submit_and_return", fake_submit)
    result = registry_module._exec_structure_gen({
        "material_type": "u-HOF",
        "n_structures": 42,
        "max_atoms": 1500,
        "topologies": ["utk", "hxg", "bto", "cds", "cdt", "srs", "eta"],
        "structures_per_topology": 6,
        "output_dir": str(tmp_path / "structures"),
    })
    command = captured["command"]
    assert "pormake_generate_topo.py" in command
    assert "/home/user/.conda/envs/pormake/bin/python" in command
    assert "conda activate" not in command
    for topology in ("utk", "hxg", "bto", "cds", "cdt", "srs", "eta"):
        assert f"--topo {topology}" in command
    assert command.count("--n 6") == 7
    assert result["n_structures"] == 42


def test_agent_config_loads_workspace_tool_paths():
    config = AgentConfig(api_key="unit-test")
    assert config.conda_path == "/opt/conda/miniconda/3-python3.9.13/etc/profile.d/conda.sh"
    assert config.pormake_python == "/home/user/.conda/envs/pormake/bin/python"
    assert config.raspa_path == "/home/user/RASPA2/simulations"


def test_live_registry_has_no_retired_conda_initialization_path():
    source = Path(registry_module.__file__).read_text()
    assert "/home/user/.conda/etc/profile.d/conda.sh" not in source
