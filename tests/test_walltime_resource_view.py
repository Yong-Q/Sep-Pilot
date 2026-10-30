"""Time-limit metadata is distinct from explicit effective allocation claims."""
import copy
import pytest
from agents.native_resource_carry import view_contract_status
from agents.workflow_view import authoritative_line, workflow_completion
from tests.test_resource_view_truth import fixture


def walltime_only_case():
    line, state = fixture()
    node = state['nodes']['submit']
    for arguments in (line['steps'][0]['arguments'], node['contract']['arguments'], node['effective_arguments']):
        for name in ('memory_mb', 'num_processes', 'resource_review_id'):
            arguments.pop(name)
        arguments['walltime'] = '00:05:00'
    return line, state


def test_walltime_only_does_not_invent_allocation_inconsistency_or_resource_verification():
    line, state = walltime_only_case()
    before = copy.deepcopy((line, state))
    step = authoritative_line(line, state)['steps'][0]
    assert step['effective_resource_claimed_fields'] == []
    assert step['effective_resource_receipt_status'] == 'not_recorded'
    assert step['effective_walltime_status'] == 'execution_record_unknown'
    assert step['done'] is True
    assert workflow_completion(line, state, line['steps'], 1)['ok'] is True
    assert (line, state) == before


@pytest.mark.parametrize('change', ['changed', 'recorded_mismatch', 'recorded_match'])
def test_walltime_records_and_changes_have_explicit_meaning(change):
    line, state = walltime_only_case()
    node = state['nodes']['submit']
    if change == 'changed':
        node['effective_arguments']['walltime'] = '00:06:00'
    else:
        node['resource_allocation']['walltime'] = '00:05:00' if change == 'recorded_match' else '00:06:00'
    step = authoritative_line(line, state)['steps'][0]
    assert step['effective_walltime_status'] == {
        'changed': 'differs_from_approved_contract', 'recorded_mismatch': 'inconsistent',
        'recorded_match': 'matches_recorded_allocation'}[change]
    assert step['done'] is (change == 'recorded_match')
    assert workflow_completion(line, state, line['steps'], 1)['ok'] is (change == 'recorded_match')


@pytest.mark.parametrize('dimension', ['memory_mb', 'num_processes', 'cpus_per_task', 'partition', 'nodelist'])
@pytest.mark.parametrize('missing', ['budget', 'evidence', 'effective_review', 'allocation_review', 'dimension'])
def test_genuine_allocation_claims_still_require_exact_review(dimension, missing):
    line, state = walltime_only_case()
    node = state['nodes']['submit']
    mapping = {'memory_mb': 'memory_mb', 'num_processes': 'cpus', 'cpus_per_task': 'cpus',
               'partition': 'partition', 'nodelist': 'nodelist'}
    value = {'memory_mb': 1536, 'num_processes': 1, 'cpus_per_task': 1,
             'partition': 'compute', 'nodelist': 'node10'}[dimension]
    node['resource_allocation'][mapping[dimension]] = value
    for arguments in (line['steps'][0]['arguments'], node['contract']['arguments'], node['effective_arguments']):
        arguments.update({dimension: value, 'resource_review_id': 'new'})
    if missing == 'budget': node['resource_allocation']['budget_final'] = False
    elif missing == 'evidence': node['resource_allocation'].pop('evidence_call_ids')
    elif missing == 'effective_review': node['effective_arguments'].pop('resource_review_id')
    elif missing == 'allocation_review': node['resource_allocation'].pop('resource_review_id')
    else: node['resource_allocation'].pop(mapping[dimension])
    step = authoritative_line(line, state)['steps'][0]
    assert step['execution_acceptance_status'] == 'effective_resources_unverified'
    assert step['done'] is False
    assert workflow_completion(line, state, line['steps'], 1)['ok'] is False


def test_gpu_dimension_tamper_remains_rejected():
    line, state = walltime_only_case()
    state['nodes']['submit']['effective_arguments']['gpus'] = 2
    step = authoritative_line(line, state)['steps'][0]
    assert step['execution_acceptance_status'] == 'effective_science_changed'
    assert step['done'] is False
