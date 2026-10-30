"""Independent parser checks: continuation differs from new scientific scope."""
import copy

import pytest
from agents.goal_contract import GoalContract


def original():
    goal = GoalContract.from_user_message(
        '请直接执行 /project/runs/owner/case/input/original.cif 的经典cDFT甲烷短测，298 K，生成图表报告。')
    goal.execution_authorized = True
    goal.execution_mode = 'workflow'
    goal.approved_plan_version = 2
    goal.approved_nodes = [{'step_id': 'submit', 'tool': 'run_cdft', 'agent': 'analyst',
        'arguments': {'action': 'submit', 'cif_path': '/project/runs/owner/case/input/original.cif',
                      'gas': 'CH4', 'temperature': 298},
        'depends_on': [], 'expected_outputs': []}]
    return goal


def test_plain_preserving_continuation_does_not_create_new_execution_epoch():
    goal = original(); version = goal.version; nodes = copy.deepcopy(goal.approved_nodes)
    goal.apply_user_message('继续原来的经典cDFT短测任务，保持原先指定的科学参数和计算上限，完成计算与图表报告。')
    assert goal.version == version
    assert goal.approved_nodes == nodes


@pytest.mark.parametrize('instruction', [
    '改为计算 /project/runs/owner/case/input/replacement.cif 的经典cDFT甲烷吸附。',
    '继续经典cDFT，但温度改成308 K。',
    '继续经典cDFT，但换成另一组MOF结构，重新计算。',
    '继续经典cDFT，但力场改为UFF。',
    '不要提交，只读检查当前经典cDFT输入。',
])
def test_explicit_scientific_or_authority_change_is_not_metadata(instruction):
    goal = original(); version = goal.version
    assert goal.apply_user_message(instruction) is True
    assert goal.version > version
