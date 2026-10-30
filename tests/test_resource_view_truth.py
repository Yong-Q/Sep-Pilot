import copy
import pytest
from agents.workflow_view import authoritative_line, line_digest, workflow_completion


def fixture():
    arguments = {'gas': 'CH4', 'temperature': 298, 'input_dir': '/owned/input',
                 'memory_mb': 1536, 'num_processes': 1, 'resource_review_id': 'new'}
    contract = {'step_id': 'submit', 'tool': 'run_cdft', 'agent': 'adsorption',
                'arguments': arguments, 'depends_on': [], 'expected_outputs': ['/owned/results.csv']}
    line = {'username': 'u', 'conv_id': 'c', 'plan_version': 1,
            'steps': [copy.deepcopy(contract)]}
    node = {'contract': copy.deepcopy(contract), 'status': 'succeeded', 'phase': 'finish',
            'job_ids': ['123'], 'node_verification': {'ok': True}, 'effective_arguments': copy.deepcopy(arguments),
            'resource_allocation': {'memory_mb': 1536, 'workers': 1, 'cpus': 1,
                'resource_review_id': 'new', 'budget_final': True, 'evidence_call_ids': ['review']}}
    state = {'username': 'u', 'conv_id': 'c', 'plan_version': 1, 'nodes': {'submit': node}}
    return line, state


def test_receipt_identity_is_not_scientific_drift_or_new_success():
    line, state = fixture()
    line['steps'][0]['arguments']['resource_review_id'] = 'old'
    before = copy.deepcopy((line, state))
    view = authoritative_line(line, state)
    step = view['steps'][0]
    assert step['science_contract_status'] == 'matches'
    assert step['resource_contract_status'] == 'receipt_identity_changed'
    assert step['status'] == 'succeeded' and step['job_ids'] == ['123']
    assert not step['done'] and not step['contract_matches_execution']
    assert step['approved_arguments']['resource_review_id'] == 'old'
    assert step['canonical_arguments']['resource_review_id'] == 'new'
    assert step['effective_arguments']['resource_review_id'] == 'new'
    assert 'not scientific drift' in line_digest(view)
    assert not workflow_completion(line, state, line['steps'], 1)['ok']
    assert (line, state) == before


@pytest.mark.parametrize('key,value', [('gas', 'CO2'), ('temperature', 350), ('input_dir', '/other')])
def test_scientific_edits_reject_old_success(key, value):
    line, state = fixture()
    line['steps'][0]['arguments'][key] = value
    step = authoritative_line(line, state)['steps'][0]
    assert step['science_contract_status'] == 'changed'
    assert not step['done'] and step['status'] == 'pending'
    assert not workflow_completion(line, state, line['steps'], 1)['ok']


@pytest.mark.parametrize('key,value', [('memory_mb', 9999), ('num_processes', 4)])
def test_resource_edits_still_require_approval(key, value):
    line, state = fixture()
    line['steps'][0]['arguments'][key] = value
    step = authoritative_line(line, state)['steps'][0]
    assert step['science_contract_status'] == 'matches'
    assert step['resource_contract_status'] == 'allocation_changed'
    assert key in step['resource_contract_changed_fields']
    assert not step['done'] and not step['contract_matches_execution']
    assert not workflow_completion(line, state, line['steps'], 1)['ok']


@pytest.mark.parametrize('key,value', [('gas', 'CO2'), ('temperature', 350), ('input_dir', '/other'),
                                      ('memory_mb', 9999), ('num_processes', 4),
                                      ('resource_review_id', 'unreviewed')])
def test_effective_tampering_is_visible_and_never_done(key, value):
    line, state = fixture()
    state['nodes']['submit']['effective_arguments'][key] = value
    step = authoritative_line(line, state)['steps'][0]
    assert not step['done']
    assert not workflow_completion(line, state, line['steps'], 1)['ok']
    if key in {'gas', 'temperature', 'input_dir'}:
        assert step['science_contract_status'] == 'changed'
        assert step['execution_acceptance_status'] == 'effective_science_changed'
    else:
        assert step['effective_resource_receipt_status'] == 'inconsistent'
        assert step['execution_acceptance_status'] == 'effective_resources_unverified'


def test_reviewed_effective_receipt_rotation_does_not_change_science():
    line, state = fixture()
    state['nodes']['submit']['effective_arguments']['resource_review_id'] = 'newer'
    state['nodes']['submit']['resource_allocation']['resource_review_id'] = 'newer'
    view = authoritative_line(line, state)
    step = view['steps'][0]
    assert step['science_contract_status'] == 'matches'
    assert step['effective_resource_receipt_status'] == 'recorded_review'
    assert step['effective_resource_changed_fields'] == ['resource_review_id']
    assert step['done']
    assert 'Resource receipt refresh, no science replan' in line_digest(view)


def test_review_uses_native_cpu_mapping_not_worker_count():
    line, state = fixture()
    node = state['nodes']['submit']
    line['steps'][0]['arguments']['num_processes'] = 4
    node['contract']['arguments']['num_processes'] = 4
    node['effective_arguments']['num_processes'] = 4
    node['resource_allocation']['cpus'] = 4
    assert node['resource_allocation']['workers'] == 1
    assert authoritative_line(line, state)['steps'][0]['done']


def test_unreviewed_budget_is_not_presented_as_success():
    line, state = fixture()
    state['nodes']['submit']['resource_allocation']['budget_final'] = False
    step = authoritative_line(line, state)['steps'][0]
    assert not step['done']
    assert step['effective_resource_receipt_status'] == 'unreviewed'
