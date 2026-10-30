"""Schema-valid report edits preserve native compiler parent ownership."""
import copy

import jsonschema
import pytest

from agents.registry import _build_default_registry
from agents.workflow_compiler import expand_parallel_report_nodes
from agents.workflow_patch import WorkflowContractError


PUBLIC_FIELDS = {'step_id', 'description', 'agent', 'tool', 'arguments', 'input_bindings',
                 'depends_on', 'expected_outputs', 'resource_locks'}


def public_node(node):
    return {key: copy.deepcopy(value) for key, value in node.items() if key in PUBLIC_FIELDS}


def initial(tmp_path):
    root = tmp_path / 'owned-session'
    report = {'step_id': 'report', 'agent': 'communicator', 'tool': 'generate_scientific_report',
        'arguments': {'source_steps': ['measured', 'figure'], 'output_path': str(root / 'report.md'),
                      'title': 'Measured results', 'instructions': 'Use actual evidence and disclose unknown values.'},
        'depends_on': ['measured', 'figure'], 'expected_outputs': [str(root / 'report.md')]}
    changes = expand_parallel_report_nodes([{'operation': 'upsert', 'step_id': 'report', 'node': report}],
        root, target_plan_version=1)
    return root, [change['node'] for change in changes]


def edits(existing, suffix='corrected'):
    changes = []
    for prior in existing:
        node = public_node(prior)
        node['arguments']['instructions'] = 'Keep original measurements; disclose unknown values: ' + suffix
        if node['arguments'].get('report_mode') == 'fragment':
            path = node['arguments']['output_path'] + '.' + suffix
            node['arguments']['output_path'] = path
            node['expected_outputs'] = [path]
        changes.append({'operation': 'upsert', 'step_id': node['step_id'], 'node': node})
    return changes


def validate_public_schema(changes):
    tool = _build_default_registry().get('propose_workflow_patch')
    jsonschema.validate({'base_version': 1, 'reason': 'Revise only evidence-grounded presentation.',
                         'changes': changes}, tool.input_schema)


def assert_compiler_associations(changes):
    nodes = {change['step_id']: change['node'] for change in changes}
    parent = nodes['report']
    assert parent['report_role'] == 'assembly'
    assert parent['arguments']['source_steps'] == parent['depends_on']
    assert parent['depends_on'] == ['report__fragment__measured', 'report__fragment__figure']
    for source in ('measured', 'figure'):
        fragment = nodes['report__fragment__' + source]
        assert fragment['report_role'] == 'fragment'
        assert fragment['report_parent'] == 'report' and fragment['report_source'] == source
        assert fragment['arguments']['report_mode'] == 'fragment'
        assert fragment['arguments']['source_steps'] == fragment['depends_on'] == [source]


def test_schema_valid_assembly_and_same_source_fragment_edits_can_repeat(tmp_path):
    root, existing = initial(tmp_path)
    for suffix in ('first', 'second'):
        changes = edits(existing, suffix)
        validate_public_schema(changes)
        assert all('report_role' not in change['node'] for change in changes)
        before = copy.deepcopy((existing, changes))
        result = expand_parallel_report_nodes(changes, root, existing, target_plan_version=2)
        assert_compiler_associations(result)
        assert (existing, changes) == before
        existing = [change['node'] for change in result]


def legacy_parent_loss(tmp_path):
    root, existing = initial(tmp_path)
    original_parent = copy.deepcopy(existing[-1])
    existing[-1] = public_node(original_parent)
    existing[-1]['arguments']['instructions'] = 'Previous schema-valid edit lost internal role.'
    runtime = {'report': {'contract': copy.deepcopy(existing[-1]),
        'attempt_history': [{'contract': original_parent, 'status': 'pending'}]}}
    return root, existing, runtime


def test_old_parent_role_loss_recovers_only_original_native_association(tmp_path):
    root, existing, runtime = legacy_parent_loss(tmp_path)
    changes = edits(existing)
    validate_public_schema(changes)
    before = copy.deepcopy((existing, changes, runtime))
    result = expand_parallel_report_nodes(changes, root, existing, source_evidence=runtime)
    assert_compiler_associations(result)
    assert (existing, changes, runtime) == before
    for change, planned in zip(result, changes):
        assert change['node']['arguments'] == planned['node']['arguments']
        assert change['node']['expected_outputs'] == planned['node']['expected_outputs']


@pytest.mark.parametrize('change', ['no_history', 'missing_runtime', 'native_contract', 'history_parent',
    'history_agent', 'history_tool', 'history_source', 'history_dependencies', 'history_role', 'current_role'])
def test_unproved_or_foreign_parent_history_cannot_inherit_fragment_ownership(tmp_path, change):
    root, existing, runtime = legacy_parent_loss(tmp_path)
    historical = runtime['report']['attempt_history'][0]['contract']
    if change == 'no_history': runtime['report']['attempt_history'] = []
    elif change == 'missing_runtime': runtime = {}
    elif change == 'native_contract': runtime['report']['contract']['arguments']['instructions'] = 'different execution'
    elif change == 'history_parent': historical['step_id'] = 'foreign-report'
    elif change == 'history_agent': historical['agent'] = 'foreign-agent'
    elif change == 'history_tool': historical['tool'] = 'read_file'
    elif change == 'history_source': historical['arguments']['source_steps'] = ['foreign-fragment', 'report__fragment__figure']
    elif change == 'history_dependencies': historical['depends_on'] = ['foreign-fragment', 'report__fragment__figure']
    elif change == 'history_role': historical['report_role'] = 'fragment'
    else: existing[-1]['report_role'] = 'foreign-role'
    with pytest.raises(WorkflowContractError, match='owned fragment'):
        expand_parallel_report_nodes(edits(existing), root, existing, source_evidence=runtime)


@pytest.mark.parametrize('change', ['source', 'dependency', 'fragment_parent', 'fragment_agent'])
def test_current_fragment_source_owner_or_agent_cannot_change_during_metadata_restore(tmp_path, change):
    root, existing, runtime = legacy_parent_loss(tmp_path)
    changes = edits(existing)
    node = changes[0]['node']
    if change == 'source': node['arguments']['source_steps'] = ['foreign-source']
    elif change == 'dependency': node['depends_on'] = ['foreign-source']
    elif change == 'fragment_parent': node['report_parent'] = 'foreign-report'
    else: node['agent'] = 'foreign-agent'
    with pytest.raises(WorkflowContractError, match='owned report fragment'):
        expand_parallel_report_nodes(changes, root, existing, source_evidence=runtime)


def test_similar_new_fragment_name_never_gains_existing_parent_ownership(tmp_path):
    root, existing, runtime = legacy_parent_loss(tmp_path)
    changes = edits(existing)
    node = changes[0]['node']
    changes[0]['step_id'] = node['step_id'] = 'report__fragment__foreign'
    node['arguments']['source_steps'] = node['depends_on'] = ['foreign']
    parent = changes[-1]['node']
    parent['arguments']['source_steps'][0] = parent['depends_on'][0] = node['step_id']
    with pytest.raises(WorkflowContractError, match='owned fragment'):
        expand_parallel_report_nodes(changes, root, existing, source_evidence=runtime)


def test_explicit_foreign_assembly_role_is_rejected_after_fragment_check(tmp_path):
    root, existing = initial(tmp_path)
    changes = edits(existing)
    changes[-1]['node']['report_role'] = 'fragment'
    with pytest.raises(WorkflowContractError, match='compiler-owned role'):
        expand_parallel_report_nodes(changes, root, existing)
