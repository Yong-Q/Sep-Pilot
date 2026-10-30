"""Report presentation repairs keep their approved scientific evidence scope."""
import copy

import pytest

from agents.workflow_patch import operational_patch


def report():
    return {'step_id': 'summary', 'tool': 'generate_scientific_report', 'agent': 'communicator',
        'arguments': {'source_steps': ['analysis'], 'report_mode': 'fragment',
            'output_path': 'reports/old.md', 'title': 'Results',
            'instructions': 'Summarize the source evidence.'},
        'expected_outputs': ['reports/old.md'], 'depends_on': ['analysis']}


def test_report_wording_and_new_output_location_are_operational():
    before = report()
    after = copy.deepcopy(before)
    after['arguments'].update(title='Verified results',
        instructions='Separate measured results from literature statements.',
        output_path='reports/new.md')
    after['expected_outputs'] = ['reports/new.md']
    assert operational_patch([before], [{'operation': 'upsert', 'step_id': 'summary', 'node': after}])


@pytest.mark.parametrize('key,value', [
    ('source_steps', ['different_analysis']), ('report_mode', 'assembled'),
    ('temperature', 400), ('gases', ['different_gas']),
])
def test_report_presentation_permission_does_not_change_evidence_or_science(key, value):
    before = report()
    after = copy.deepcopy(before)
    after['arguments'][key] = value
    assert not operational_patch([before], [{'operation': 'upsert', 'step_id': 'summary', 'node': after}])


def test_instructions_of_a_scientific_tool_remain_protected():
    before = report()
    before['tool'] = 'run_gcmc_batch'
    after = copy.deepcopy(before)
    after['arguments']['instructions'] = 'Change how this computation runs.'
    assert not operational_patch([before], [{'operation': 'upsert', 'step_id': 'summary', 'node': after}])


@pytest.mark.parametrize('action,field,allowed', [
    ('inputs', 'input_dir', True), ('inputs', 'cif_dir', False),
    ('inputs', 'temperature', False), ('submit', 'input_dir', False),
])
def test_cdft_path_role_depends_on_which_action_owns_the_directory(action, field, allowed):
    before = {'step_id': 'cdft', 'agent': 'analyst', 'tool': 'run_cdft',
        'arguments': {'action': action, 'cif_dir': 'frameworks', 'input_dir': 'old-inputs',
            'gas': 'CO2', 'temperature': 298}, 'expected_outputs': ['old-output.dat']}
    after = copy.deepcopy(before)
    after['arguments'][field] = 310 if field == 'temperature' else 'new-directory'
    after['expected_outputs'] = ['new-output.dat']
    assert operational_patch([before], [{'operation': 'upsert', 'step_id': 'cdft', 'node': after}]) is allowed
