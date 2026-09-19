import copy

import pytest

from agents.goal_contract import GoalContract
from agents.session import Session
from auth import ConversationState
from api import _workflow_state
from agents.defns import ORCHESTRATOR


def test_research_draft_is_persisted_without_execution_and_shows_real_edges():
    session = Session()
    session.current_agent = ORCHESTRATOR
    writes = []
    session._on_checkpoint = lambda state, reason: writes.append((state, reason))
    nodes = [{'step_id': f's{i}', 'description': f'stage {i}', 'agent':'analyst',
              'depends_on': [] if i == 0 else [f's{i-1}'], 'missing_parameters':['temperature'] if i == 2 else []}
             for i in range(5)]
    session._record_workflow_draft(nodes, completion_criteria='Validated final report from all five stages')
    assert writes[-1][0]['context']['workflow_draft']['nodes'][4]['depends_on'] == ['s3']
    conv = ConversationState(conv_id='draft-only-regression', username='123')
    conv.session = session
    graph = _workflow_state(conv)['graph_projection']
    assert graph['source'] == 'workflow_draft' and len(graph['nodes']) == 5
    assert graph['executable'] is False
    assert session.goal_contract.approved_nodes == []
    with pytest.raises(ValueError, match='cycle'):
        session._record_workflow_draft([{**nodes[0], 'depends_on':['s0']}], completion_criteria='Final report')
    assert len(session.context['workflow_draft']['nodes']) == 5


CASES = [
    "生成20个COF并做CO2/N2 cDFT，298K，0.1-10 bar，10点",
    "生成10个MOF后做CO2吸附",
    "对已有CIF做孔径分析然后计算吸附",
    "构建5个HOF并计算Xe/Kr选择性",
    "PACMOF赋电荷后做GCMC",
    "检查力场后计算CO2等温线",
    "两种气体并行计算最后汇总",
    "结构生成失败时修补后继续",
    "计算完成后验收结果再画图",
    "生成结构并做cDFT压力扫描",
    "批量20个材料并行筛选",
    "先优化结构再做吸附模拟",
    "构建COF，计算CO2和N2并比较",
    "读取输入，生成参数，提交计算并验收",
    "创建材料、赋电荷、模拟、汇总报告",
]


def _contracts(index):
    return [
        {"step_id": f"s{index}_generate", "agent": "harness-maintainer",
         "tool": "generate_structure", "arguments": {"material_type": "COF", "n_structures": 20,
         "output_dir": f"runs/123/case-{index}/structures"}, "depends_on": [],
         "expected_outputs": [{"kind": "directory", "path": f"runs/123/case-{index}/structures",
                               "pattern": "*.cif", "min_count": 20}], "resources": []},
        {"step_id": f"s{index}_calculate", "agent": "analyst", "tool": "run_cdft",
         "arguments": {"action": "pipeline", "temperature": 298},
         "depends_on": [f"s{index}_generate"],
         "expected_outputs": [{"kind": "file", "path": f"runs/123/case-{index}/results.csv"}],
         "resources": []},
    ]


@pytest.mark.parametrize("index,prompt", enumerate(CASES))
def test_user123_session_graph_is_scoped_and_required(index, prompt):
    session = Session()
    session._current_line_id = f"case-{index}"
    session._on_workflow_start = lambda *_: None
    session.goal_contract = GoalContract.from_user_message(prompt)
    assert "CHAIN_REQUIRED" in session._chain_required_block("generate_structure")
    assert session._chain_required_block("task_line_query") == ""

    session.goal_contract.approved_nodes = _contracts(index)
    session.goal_contract.approved_plan_version = 1
    assert session._chain_required_block("generate_structure") == ""
    assert "approved DAG owns delegation" in session._chain_required_block("handoff_to_harness")

    conv = ConversationState(conv_id=f"case-{index}", username="123")
    conv.session = session
    view = _workflow_state(conv)
    assert view["scope"] == {"username": "123", "conv_id": f"case-{index}"}
    graph = view["goal_contract"]["approved_nodes"]
    assert [node["step_id"] for node in graph] == [f"s{index}_generate", f"s{index}_calculate"]
    assert graph[1]["depends_on"] == [graph[0]["step_id"]]


def test_fifteen_user123_graphs_never_cross_session_scope():
    graphs = []
    for index in range(15):
        session = Session()
        session.goal_contract = GoalContract.from_user_message(CASES[index])
        session.goal_contract.approved_nodes = copy.deepcopy(_contracts(index))
        conv = ConversationState(conv_id=f"case-{index}", username="123")
        conv.session = session
        graphs.append(_workflow_state(conv))
    for index, graph in enumerate(graphs):
        nodes=graph["goal_contract"]["approved_nodes"]
        roots={node["arguments"].get("output_dir") for node in nodes if node["arguments"].get("output_dir")}
        assert roots == {f"runs/123/case-{index}/structures"}
        assert graph["scope"] == {"username":"123","conv_id":f"case-{index}"}


def test_interrupted_complex_goal_exposes_honest_planning_state_not_empty_dag():
    session = Session()
    session._current_line_id = "interrupted-before-plan"
    session.goal_contract = GoalContract.from_user_message(CASES[0])
    conv = ConversationState(conv_id="interrupted-before-plan", username="123")
    conv.session = session
    conv.interrupt_requested = True

    view = _workflow_state(conv)

    assert view["graph_projection"]["source"] == "workflow_planning_state"
    assert view["graph_projection"]["executable"] is False
    assert view["graph_projection"]["nodes"] == []
    assert view["graph_projection"]["status"] == "paused"
    assert session.goal_contract.approved_nodes == []
    assert session.goal_contract.approved_plan_version == 0
