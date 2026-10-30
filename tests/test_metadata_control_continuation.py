import copy
import json

import pytest

from agents.goal_contract import GoalContract, scope_preserving_continuation
from agents.parallel_workflow import WorkflowStore
from agents.report_recovery import _context
from agents.workflow_execution_identity import metadata_only_goal_revision, reconciled_origin_version


PRESERVE = ('先不采用移除阅读步骤的方案。保留已经下载并读取的原始文献，不要重复下载，'
            '沿用原任务继续完成原文核验和有页码出处的中文报告；保留阅读记录，'
            '已有读取成功的内容可以复用，无法查证的结论明确披露。')


def fixture(tmp_path):
    root = tmp_path / 'tester' / 'case'
    root.mkdir(parents=True)
    node = {'step_id': 'read_source', 'agent': 'analyst', 'tool': 'read_scientific_document',
            'arguments': {'path': str(root / 'source.pdf')}, 'depends_on': [],
            'expected_outputs': [{'kind': 'file', 'path': str(root / 'source.pdf')}]}
    goal = GoalContract.from_dict({'schema_version': 4, 'version': 2, 'method': '', 'gases': [], 'parameters': {},
        'original_goal': '读取原始文献并交付可核验的报告。', 'active_goal': '读取原始文献并交付可核验的报告。',
        'directives': ['读取原始文献并交付可核验的报告。'], 'execution_mode': 'workflow',
        'execution_authorized': True, 'requires_plan_approval': True,
        'approved_plan_version': 1, 'pending_plan_version': 1, 'approved_nodes': [node]})
    state = {'username': 'tester', 'conv_id': 'case', 'plan_version': 1, 'goal_contract': goal.to_dict(),
        'nodes': {'read_source': {'contract': copy.deepcopy(node), 'token': 'original-token',
            'origin_goal_version': 2, 'execution_plan_version': 1, 'job_ids': [],
            'status': 'validation_failed'}}}
    return root, goal, state


def old_parser_revision(goal, text=PRESERVE):
    after = copy.deepcopy(goal.to_dict())
    after['version'] += 1
    after['active_goal'] = text
    after['directives'].append(text)
    return after


def test_refusing_a_proposed_removal_preserves_existing_typed_goal(tmp_path):
    root, goal, state = fixture(tmp_path)
    before = copy.deepcopy(goal.to_dict())
    assert scope_preserving_continuation(PRESERVE, goal.approved_nodes)
    assert goal.apply_user_message(PRESERVE) is False
    assert goal.version == 2
    assert goal.active_goal == PRESERVE
    for name in ('approved_nodes', 'approved_plan_version', 'parameters', 'method', 'gases',
                 'execution_mode', 'execution_authorized'):
        assert goal.to_dict()[name] == before[name]


@pytest.mark.parametrize('suffix', [
    '把温度改为308 K。', '改用新势能方法。', '增加另一种材料。',
    'continue the existing task and remove a source.',
    '继续原任务，但换用另一个输入other.pdf。',
])
def test_positive_scope_changes_are_not_hidden_by_rejected_proposal(tmp_path, suffix):
    root, goal, state = fixture(tmp_path)
    text = '先不采用移除步骤的方案。沿用原任务继续。' + suffix
    assert not scope_preserving_continuation(text, goal.approved_nodes)
    assert not metadata_only_goal_revision(goal.to_dict(), old_parser_revision(goal, text))


def test_rejected_clause_does_not_hide_a_following_unseparated_change(tmp_path):
    root, goal, state = fixture(tmp_path)
    assert not scope_preserving_continuation('拒绝这个方案改用新方法，继续原任务。', goal.approved_nodes)


def test_new_source_inside_rejected_clause_stays_conservative(tmp_path):
    root, goal, state = fixture(tmp_path)
    assert not scope_preserving_continuation('先不接受other.pdf的移除方案。继续原任务。', goal.approved_nodes)


@pytest.mark.parametrize('field,value', [
    ('parameters', {'temperature_K': 308}), ('gases', ['N2']), ('method', 'GCMC'),
    ('execution_authorized', False), ('execution_mode', 'read_only'), ('approved_plan_version', 2),
])
def test_metadata_proof_rejects_every_protected_scope_change(tmp_path, field, value):
    root, goal, state = fixture(tmp_path)
    after = old_parser_revision(goal)
    after[field] = value
    assert not metadata_only_goal_revision(goal.to_dict(), after)


def test_native_reconciliation_preserves_original_node_identity_and_enables_owned_context(tmp_path):
    root, goal, state = fixture(tmp_path)
    after = old_parser_revision(goal)
    before_node = copy.deepcopy(state['nodes']['read_source'])
    with pytest.raises(ValueError, match='goal/plan changed'):
        _context(state, 'read_source', root, after)
    path = tmp_path / 'workflows.json'
    path.write_text(json.dumps({'workflows': {'native-workflow': state}, 'leases': {}}))
    store = WorkflowStore(path)
    assert store.reconcile_goal('native-workflow', after, owner=('tester', 'case'), require_unpaused=True)
    updated = store.snapshot('native-workflow')
    assert updated['nodes']['read_source'] == before_node
    assert updated['goal_contract'] == after
    assert len(updated['goal_reconciliation_history']) == 1
    context = _context(updated, 'read_source', root, after)
    context[3].close()
    assert context[0]['origin_goal_version'] == 2
    assert context[0]['token'] == 'original-token'
    assert context[-1]['goal_version'] == 3
    assert context[-1]['no_resubmission'] is True


@pytest.mark.parametrize('history', [
    [], [{'from_version': 2, 'to_version': 3, 'metadata_only_verified': False,
          'basis': 'identical protected scope and approved node contracts'}],
    [{'from_version': 2, 'to_version': 3, 'metadata_only_verified': True, 'basis': 'model says resume'}],
    [{'from_version': 1, 'to_version': 3, 'metadata_only_verified': True,
      'basis': 'identical protected scope and approved node contracts'}],
    [{'from_version': 2, 'to_version': True, 'metadata_only_verified': True,
      'basis': 'identical protected scope and approved node contracts'}],
])
def test_origin_cannot_cross_unproved_metadata_revision(tmp_path, history):
    root, goal, state = fixture(tmp_path)
    after = old_parser_revision(goal)
    state['goal_contract'] = after
    state['goal_reconciliation_history'] = history
    assert reconciled_origin_version(state, 2, after) is None
    with pytest.raises(ValueError, match='dispatch goal/plan'):
        _context(state, 'read_source', root, after)


def test_changed_contract_cannot_borrow_metadata_history(tmp_path):
    root, goal, state = fixture(tmp_path)
    after = old_parser_revision(goal)
    after['approved_nodes'][0]['arguments']['path'] = str(root / 'another.pdf')
    path = tmp_path / 'workflows.json'
    path.write_text(json.dumps({'workflows': {'native-workflow': state}, 'leases': {}}))
    assert WorkflowStore(path).reconcile_goal('native-workflow', after, owner=('tester', 'case')) is False



def test_rejected_numbered_proposal_can_continue_exact_original_scope(tmp_path):
    _,goal,_=fixture(tmp_path)
    text=('不批准当前第4版变更的方案，也不同意删除阅读步骤。保留原始材料，'
          '继续原要求的报告，无法查证的结论明确写未知。')
    assert scope_preserving_continuation(text,goal.approved_nodes)
    assert metadata_only_goal_revision(goal.to_dict(),old_parser_revision(goal,text))


@pytest.mark.parametrize('text',[
    '不批准第4版改成308 K的方案，继续原任务。',
    '不批准第4版读取other.pdf的方案，继续原任务。',
    '不批准第4版方案，继续原任务并删除阅读步骤。',
    '不批准第4版方案，继续原要求并改换材料。',
    '不批准第4版方案，继续原要求并新增计算。',
    '拒绝第4版方案改用新方法，继续原要求。',
    '不批准第4版方案改用新材料方案，继续原任务。',
    '不同意删除阅读步骤改换材料步骤，继续原任务。',
])
def test_rejected_numbered_proposal_cannot_mask_a_new_scientific_scope(tmp_path,text):
    _,goal,_=fixture(tmp_path)
    assert not scope_preserving_continuation(text,goal.approved_nodes)
    assert not metadata_only_goal_revision(goal.to_dict(),old_parser_revision(goal,text))
