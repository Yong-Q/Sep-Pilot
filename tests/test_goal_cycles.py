"""Explicit MC budgets survive user updates and cannot be bypassed by a DAG."""
import copy
import json

import pytest

from agents.goal_contract import GoalContract


@pytest.mark.parametrize('message,expected', [
    ('GCMC测试计算按1000 cycles做功能验收', 1000),
    ('Run GCMC using 2,000 cycles', 2000),
    ('Henry Monte Carlo cycles = 2500', 2500),
    ('GCMC不要50000 cycles，改成1200 cycles', 1200),
    ('GCMC cycles改成3200', 3200),
])
def test_explicit_mc_cycle_budget_is_typed(message, expected):
    goal = GoalContract.from_user_message(message)
    assert goal.parameters['cycles'] == expected
    assert isinstance(goal.parameters['cycles'], int)


@pytest.mark.parametrize('message', ['String迭代1000 steps', 'LAMMPS MD运行1000 cycles',
    'Python string repeat 1000 cycles', 'GCMC 1000 cycles是什么意思？'])
def test_unrelated_steps_and_questions_do_not_pin_mc_budget(message):
    assert 'cycles' not in GoalContract.from_user_message(message).parameters


def test_followup_updates_budget_without_approval_changing_it_back():
    goal = GoalContract.from_user_message('GCMC用50000 cycles计算')
    assert goal.apply_user_message('这次测试按1000 cycles做功能验收')
    assert goal.parameters['cycles'] == 1000
    goal.apply_user_message('确认执行')
    assert goal.parameters['cycles'] == 1000


@pytest.mark.parametrize('count', [-5, 0])
def test_invalid_user_budget_never_falls_back_to_default(count):
    goal = GoalContract.from_user_message(f'GCMC cycles={count}')
    assert goal.parameters['cycles'] == count
    assert not goal.guard_tool_call('run_gcmc_batch', {'cycles': 50000})[0]


def test_md_followup_does_not_replace_existing_mc_cycles():
    goal = GoalContract.from_user_message('GCMC按2500 cycles运行')
    goal.apply_user_message('MD用1000 steps优化结构')
    assert goal.parameters['cycles'] == 2500


def test_legacy_restore_does_not_take_cycle_questions_as_user_edits():
    saved = GoalContract.from_user_message('GCMC按50000 cycles运行').to_dict()
    saved['parameters'].pop('cycles')
    saved['directives'] += ['这次按1000 cycles做功能验收', 'GCMC 2000 cycles是什么意思？']
    assert GoalContract.from_dict(saved).parameters['cycles'] == 1000


@pytest.mark.parametrize('tool', ['run_gcmc_batch', 'run_gcmc_isotherm'])
@pytest.mark.parametrize('actual', [None, 50000, True, '1000', 1000.0, -1, 0])
def test_old_approved_arguments_cannot_override_new_explicit_cycles(tool, actual):
    goal = GoalContract.from_user_message('GCMC测试按1000 cycles做功能验收')
    arguments = {'cycles': actual} if actual is not None else {}
    goal.approved_nodes = [{'step_id': 'calc', 'tool': tool, 'arguments': arguments}]
    goal.approved_plan_version = 1
    goal.execution_authorized = True
    allowed, reason = goal.guard_tool_call(tool, arguments)
    assert not allowed and 'cycles' in reason


@pytest.mark.parametrize('tool', ['run_gcmc_batch', 'run_gcmc_isotherm'])
def test_exact_requested_cycles_are_allowed(tool):
    goal = GoalContract.from_user_message('GCMC 1000 cycles')
    assert goal.guard_tool_call(tool, {'cycles': 1000})[0]


def test_structured_handoff_cannot_hide_a_stale_budget():
    goal = GoalContract.from_user_message('GCMC 1000 cycles')
    allowed, reason = goal.guard_tool_call('handoff_to_adsorption', {'context': json.dumps({
        'node': {'tool': 'run_gcmc_batch', 'arguments': {'cycles': 50000}}})})
    assert not allowed and 'cycles' in reason


@pytest.mark.parametrize('tool', ['generate_structure', 'run_pacman_charge', 'run_md_optimize', 'run_string_tst'])
def test_mc_budget_does_not_constrain_prerequisites_or_other_methods(tool):
    goal = GoalContract.from_user_message('GCMC 1000 cycles')
    goal.method = ''
    assert goal.guard_tool_call(tool, {'steps': 50000})[0]


@pytest.mark.parametrize('tool', ['run_henry', 'run_henry_chain', 'run_isotherm_chain'])
def test_routes_without_cycle_parameter_do_not_silently_use_their_defaults(tool):
    goal = GoalContract.from_user_message('GCMC 1000 cycles')
    assert not goal.guard_tool_call(tool, {'cycles': 1000})[0]


def test_restore_backfills_last_real_directive_without_changing_version():
    saved = GoalContract.from_user_message('GCMC 50000 cycles').to_dict()
    saved['parameters'].pop('cycles', None)
    saved['directives'] += ['这次按1000 cycles做功能验收', '[生命周期事件]\n500 cycles', '确认执行']
    saved['version'] = 7
    restored = GoalContract.from_dict(copy.deepcopy(saved))
    assert restored.parameters['cycles'] == 1000
    assert restored.version == saved['version']
    assert restored.directives == saved['directives']
