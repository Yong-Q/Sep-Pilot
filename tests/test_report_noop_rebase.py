"""No-op approved plan revisions cannot strand verified presentation history."""
import copy
import pytest
from agents.parallel_workflow import _approval_digest
from agents.presentation_transition import report_only_history_context
from tests.test_report_only_presentation_history import fixture


def rebased():
    state, goal, line, approval = fixture()
    goal['approved_nodes'] = copy.deepcopy(state['goal_contract']['approved_nodes'])
    goal['approved_plan_version'] = 8
    goal['directives'].append('继续原任务，保留已有计算结果和图表，不重复计算，完成报告。')
    state['pause_kind'] = 'input_changed'
    state['pause_reason'] = 'cached inputs/tool sources/artifacts changed; approve revised plan'
    line['plan_version'] = 8
    line['plan_history'] = [
        {'version': 6, 'steps': copy.deepcopy(state['goal_contract']['approved_nodes'])},
        {'version': 7, 'steps': copy.deepcopy(state['goal_contract']['approved_nodes'])},
    ]
    approval['base_plan_version'] = 7
    approval['plan_version'] = 8
    approval['approval_sha256'] = _approval_digest(goal)
    return state, goal, line, approval


def test_owned_unchanged_graph_rebases_without_new_execution_authority():
    state, goal, line, approval = rebased()
    before = copy.deepcopy((state, goal, line, approval))
    result = report_only_history_context(state, goal, line, approval, owner={'username': 'owner', 'conv_id': 'conv'})
    assert result['kind'] == 'approved_report_only_history'
    assert result['retained_steps'] == ['figure'] and result['changed_report_steps'] == []
    assert result['from_plan_version'] == 6 and result['to_plan_version'] == 8
    assert result['no_execution_authority'] and result['no_scientific_change']
    assert (state, goal, line, approval) == before


def test_completed_preparation_is_marked_for_native_proof_not_reexecution():
    state, goal, line, approval = rebased()
    analysis = {'step_id': 'analysis', 'agent': 'analyst', 'tool': 'analyze_gcmc_screening',
                'arguments': {'work_dir': 'source'}, 'depends_on': ['science'], 'expected_outputs': ['result.csv']}
    state['nodes']['analysis'] = {'contract': copy.deepcopy(analysis), 'status': 'succeeded', 'phase': 'finish', 'token': 'analysis-token'}
    state['goal_contract']['approved_nodes'].append(copy.deepcopy(analysis))
    goal['approved_nodes'].append(copy.deepcopy(analysis))
    for version in line['plan_history']:
        version['steps'].append(copy.deepcopy(analysis))
    approval['approval_sha256'] = _approval_digest(goal)
    result = report_only_history_context(state, goal, line, approval, owner={'username': 'owner', 'conv_id': 'conv'})
    assert result['retained_steps'] == ['analysis', 'figure']
    assert result['no_execution_authority']


def test_unknown_completed_local_step_fails_closed_before_reexecution():
    state, goal, line, approval = rebased()
    opaque = {'step_id': 'opaque', 'agent': 'analyst', 'tool': 'custom_local_tool',
              'arguments': {}, 'depends_on': [], 'expected_outputs': ['result.csv']}
    state['nodes']['opaque'] = {'contract': copy.deepcopy(opaque), 'status': 'succeeded', 'phase': 'finish', 'token': 'opaque-token'}
    state['goal_contract']['approved_nodes'].append(copy.deepcopy(opaque))
    goal['approved_nodes'].append(copy.deepcopy(opaque))
    for version in line['plan_history']:
        version['steps'].append(copy.deepcopy(opaque))
    approval['approval_sha256'] = _approval_digest(goal)
    assert report_only_history_context(state, goal, line, approval, owner={'username': 'owner', 'conv_id': 'conv'}) is None


@pytest.mark.parametrize('change', [
    'science', 'figure', 'report', 'history_contract', 'history_missing', 'history_duplicate',
    'user_change', 'directive_missing', 'user_paused', 'pause_reason', 'foreign',
    'unsigned', 'old_goal', 'gap_too_large', 'wrong_base', 'old_runtime_contract', 'stop', 'read_only',
])
def test_rebase_denies_changed_science_presentation_history_or_authority(change):
    state, goal, line, approval = rebased()
    if change == 'science': goal['approved_nodes'][0]['arguments']['gas'] = 'CO2'
    elif change == 'figure': goal['approved_nodes'][1]['arguments']['instructions'] = 'redraw'
    elif change == 'report': goal['approved_nodes'][2]['arguments']['instructions'] = 'different'
    elif change == 'history_contract': line['plan_history'][1]['steps'][0]['arguments']['gas'] = 'CO2'
    elif change == 'history_missing': line['plan_history'].pop()
    elif change == 'history_duplicate': line['plan_history'].append(copy.deepcopy(line['plan_history'][1]))
    elif change == 'user_change': goal['directives'][-1] = '继续原任务但换用 CO2。'
    elif change == 'directive_missing': goal['directives'].pop()
    elif change == 'user_paused': state['user_paused'] = True
    elif change == 'pause_reason': state['pause_reason'] = 'unrelated safety stop'
    elif change == 'foreign': approval['owner']['username'] = 'other'
    elif change == 'unsigned': approval['approval_sha256'] = 'bad'
    elif change == 'old_goal': goal['active_goal'] = 'different goal'
    elif change == 'gap_too_large': goal['approved_plan_version'] = line['plan_version'] = approval['plan_version'] = 16
    elif change == 'wrong_base': approval['base_plan_version'] = 5
    elif change == 'old_runtime_contract': state['nodes']['science']['contract']['arguments']['gas'] = 'CO2'
    elif change == 'stop': goal['directives'][-1] = '不要继续原任务。'
    elif change == 'read_only': goal['directives'][-1] = '继续原任务但不要执行。'
    if change != 'unsigned': approval['approval_sha256'] = _approval_digest(goal)
    assert report_only_history_context(state, goal, line, approval, owner={'username': 'owner', 'conv_id': 'conv'}) is None
