"""A saved node identity cannot supersede the user's current science scope."""
import copy
import json

import pytest

from agents.goal_contract import GoalContract


def approved_goal(tool, arguments, *, method='', gases=(), parameters=None):
    return GoalContract(method=method, gases=list(gases), parameters=parameters or {},
        execution_authorized=True, approved_plan_version=1,
        approved_nodes=[{'step_id': 'old-approved', 'tool': tool, 'arguments': copy.deepcopy(arguments)}])


@pytest.mark.parametrize('tool,method', [('run_gcmc_batch', 'GCMC'), ('run_gcmc_isotherm', 'GCMC'),
                                       ('run_cdft', 'CDFT'), ('run_henry', 'HENRY')])
def test_old_exact_approved_temperature_cannot_override_current_user_constraint(tool, method):
    arguments = {'gas': 'CO2', 'temperature': 298}
    if tool == 'run_cdft': arguments['action'] = 'pipeline'
    goal = approved_goal(tool, arguments, method=method, gases=['CO2'], parameters={'temperature_K': 310})
    allowed, reason = goal.guard_tool_call(tool, arguments)
    assert not allowed and '温度' in reason


@pytest.mark.parametrize('tool,method', [('run_gcmc_batch', 'GCMC'), ('run_cdft', 'CDFT'),
                                       ('run_henry', 'HENRY')])
def test_old_exact_approved_gas_cannot_override_current_user_constraint(tool, method):
    arguments = {'gases': ['CO2'], 'temperature': 298}
    if tool == 'run_cdft': arguments['action'] = 'pipeline'
    goal = approved_goal(tool, arguments, method=method, gases=['Kr'], parameters={'temperature_K': 298})
    allowed, reason = goal.guard_tool_call(tool, arguments)
    assert not allowed and 'CO2' in reason and 'Kr' in reason


@pytest.mark.parametrize('field,old,current', [('n_structures', 50, 20), ('max_atoms', 1500, 1000)])
def test_old_exact_approved_generation_constraint_does_not_override_user_update(field, old, current):
    arguments = {'material_type': 'MOF', field: old}
    goal = approved_goal('generate_structure', arguments, parameters={field: current})
    allowed, reason = goal.guard_tool_call('generate_structure', arguments)
    assert not allowed and field in reason


def test_old_exact_approved_topology_and_method_are_still_scientific_constraints():
    arguments = {'material_type': 'MOF', 'topologies': ['dia']}
    goal = approved_goal('generate_structure', arguments, parameters={'topologies': ['pcu']})
    assert not goal.guard_tool_call('generate_structure', arguments)[0]
    arguments = {'gases': ['CO2'], 'temperature': 298}
    goal = approved_goal('run_gcmc_batch', arguments, method='CDFT', gases=['CO2'])
    assert not goal.guard_tool_call('run_gcmc_batch', arguments)[0]


@pytest.mark.parametrize('tool,arguments,method,parameters', [
    ('run_gcmc_batch', {'gases': ['CO2'], 'temperature': 310}, 'GCMC', {'temperature_K': 310}),
    ('run_cdft', {'gas': 'Kr', 'temperature': 298}, 'CDFT', {'temperature_K': [298, 310]}),
    ('generate_structure', {'material_type': 'MOF', 'n_structures': 20}, '', {'n_structures': 20}),
])
def test_unchanged_approved_node_matching_current_science_stays_authorized(tool, arguments, method, parameters):
    gases = ['Kr'] if arguments.get('gas') == 'Kr' else ['CO2']
    goal = approved_goal(tool, arguments, method=method, gases=gases, parameters=parameters)
    assert goal.guard_tool_call(tool, arguments)[0]


def test_multi_route_science_and_unaffected_prerequisites_remain_valid():
    nodes = [
        {'tool': 'run_cdft', 'arguments': {'gas': 'CO2', 'temperature': 310}},
        {'tool': 'run_gcmc_batch', 'arguments': {'gases': ['CO2'], 'temperature': 310, 'cycles': 1000}},
        {'tool': 'generate_structure', 'arguments': {'material_type': 'MOF', 'n_structures': 20}},
        {'tool': 'run_pacman_charge', 'arguments': {'cif_dir': 'generated', 'method': 'pacmof'}},
    ]
    goal = GoalContract(method='MULTI', gases=['CO2'], execution_authorized=True,
        approved_plan_version=2, approved_nodes=copy.deepcopy(nodes),
        parameters={'temperature_K': 310, 'cycles': 1000, 'n_structures': 20})
    for node in nodes:
        assert goal.guard_tool_call(node['tool'], node['arguments'])[0]
    assert goal.guard_tool_call('read_file', {'path': 'original-result.csv'})[0]


@pytest.mark.parametrize('tool,arguments,parameters', [
    ('run_cdft', {'gas': 'CO2', 'temperature': 298}, {'temperature_K': 310}),
    ('generate_structure', {'material_type': 'MOF', 'n_structures': 50}, {'n_structures': 20}),
])
def test_structured_handoff_uses_actual_node_science_not_the_outer_context(tool, arguments, parameters):
    goal = GoalContract(gases=['CO2'], parameters=parameters, execution_authorized=True)
    allowed, reason = goal.guard_tool_call('handoff_to_adsorption', {
        'context': json.dumps({'node': {'tool': tool, 'arguments': arguments}})})
    assert not allowed and reason


def test_new_matching_science_requires_a_patch_to_replace_an_old_approved_node():
    original = {'gases': ['CO2'], 'temperature': 298}
    goal = approved_goal('run_gcmc_batch', original, method='GCMC', gases=['CO2'],
                         parameters={'temperature_K': 310})
    revised = {**original, 'temperature': 310}
    assert not goal.guard_tool_call('run_gcmc_batch', revised)[0]
    assert goal.guard_tool_call('run_gcmc_batch', revised, enforce_approved_node=False)[0]
