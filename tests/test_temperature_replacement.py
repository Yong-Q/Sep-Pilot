import importlib.util
import sys
from pathlib import Path
import pytest

from agents import goal_contract as module

@pytest.mark.parametrize('text, expected', [
    ('温度从298 K改为308 K，继续计算', 308.0),
    ('change temperature from 298 K to 308 K', 308.0),
    ('从 298K 调整为 308K，另算 318 K', [308.0, 318.0]),
    ('分别计算298 K和308 K', [298.0, 308.0]),
    ('温度298 K到308 K', [298.0, 308.0]),
])
def test_temperature_replacement(text, expected):
    assert module.extract_parameters(text)['temperature_K'] == expected

def test_goal_change_keeps_only_target():
    goal = module.GoalContract.from_user_message('用 GCMC 计算 CO2 在 298 K 下的吸附')
    goal.apply_user_message('温度从298 K改为308 K，继续计算')
    assert goal.parameters['temperature_K'] == 308.0
    assert goal.guard_tool_call('run_gcmc_batch', {'temperature': 308}, enforce_approved_node=False)[0]
    assert not goal.guard_tool_call('run_gcmc_batch', {'temperature': 298}, enforce_approved_node=False)[0]

@pytest.mark.parametrize('text', ['不要从298 K改为308 K', 'do not change from 298 K to 308 K'])
def test_negated_replacement_does_not_change_goal(text):
    goal = module.GoalContract.from_user_message('用 GCMC 计算 CO2 在 298 K 下的吸附')
    goal.apply_user_message(text)
    assert goal.parameters['temperature_K'] == 298.0
