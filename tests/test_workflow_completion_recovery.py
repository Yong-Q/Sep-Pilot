"""Supplementary guards for the real SDK long-conversation scenario."""
import copy
from types import SimpleNamespace

import pytest

import auth
from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.session import Session
from agents.workflow_view import restore_approved_nodes, workflow_completion


def fixture():
    goal = GoalContract.from_user_message('用cDFT计算Xe/Kr，298K')
    goal.execution_authorized = True
    goal.approved_plan_version = 8
    node = {'step_id': 'mix', 'agent': 'analyst', 'tool': 'run_cdft',
            'arguments': {'action': 'submit', 'gases': ['Xe', 'Kr']}, 'depends_on': [], 'expected_outputs': ['results.csv']}
    goal.approved_nodes = [node]
    line = {'username': 'u', 'conv_id': 'c', 'plan_version': 8, 'steps': [
        {**copy.deepcopy(node), 'status': 'completed', 'plan_version': 8},
        {'step_id': 'old_error', 'status': 'failed', 'branch_parent': 'grep_search'},
        {'step_id': 'run_bash', 'status': 'failed'}]}
    runtime = {'username': 'u', 'conv_id': 'c', 'plan_version': 8, 'goal_contract': goal.to_dict(),
               'status': 'completed', 'nodes': {'mix': {'contract': copy.deepcopy(node), 'status': 'succeeded',
                   'node_verification': {'evidence_call_ids': ['verified']}}}}
    return goal, line, runtime


def test_completed_executor_ignores_history_even_when_approved_ids_are_missing():
    goal, line, runtime = fixture()
    assert workflow_completion(line, runtime, [], 8)['ok']
    assert line['steps'][1]['status'] == 'failed'  # No evidence was erased.


@pytest.mark.parametrize('status', ['prefinish', 'failed', 'uncertain', 'pending'])
def test_unfinished_actual_node_is_not_hidden_by_completed_task_line(status):
    goal, line, runtime = fixture()
    runtime['nodes']['mix']['status'] = status
    assert not workflow_completion(line, runtime, goal.approved_nodes, 8)['ok']


def test_missing_or_unverified_contract_fails_closed():
    goal, line, runtime = fixture()
    runtime['nodes']['mix'].pop('node_verification')
    assert not workflow_completion(line, runtime, goal.approved_nodes, 8)['ok']
    runtime['nodes'].clear()
    assert not workflow_completion(line, runtime, goal.approved_nodes, 8)['ok']
    assert not workflow_completion({'steps': line['steps'][1:]}, {}, [], 8)['ok']
    assert not workflow_completion(line, {**runtime, 'plan_version': 9}, [], 8)['ok']


def test_approval_recovery_is_scoped_versioned_and_does_not_grant_execution():
    goal, line, runtime = fixture()
    goal.approved_nodes = []
    goal.execution_authorized = False
    recovered = restore_approved_nodes(goal, runtime, {'username': 'u', 'conv_id': 'c'})
    assert recovered[0]['step_id'] == 'mix'
    assert not goal.execution_authorized
    with pytest.raises(PermissionError): restore_approved_nodes(goal, runtime, {'username': 'other', 'conv_id': 'c'})
    goal.parameters['temperature_K'] = 308
    assert not restore_approved_nodes(goal, runtime, {'username': 'u', 'conv_id': 'c'})
    goal.parameters['temperature_K'] = 298
    runtime['plan_version'] = 9
    assert not restore_approved_nodes(goal, runtime, {'username': 'u', 'conv_id': 'c'})


def test_legacy_transcript_reconstruction_retains_identical_science_approval(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    goal, _, _ = fixture()
    session.goal_contract = goal
    session.reconcile_transcript([{'role': 'user', 'content': goal.original_goal},
                                  {'role': 'user', 'content': '继续推进整个项目'}])
    assert session.goal_contract.approved_nodes == goal.approved_nodes
    assert session.goal_contract.approved_plan_version == 8


def test_conversation_serialization_retains_processed_boundary_and_owner():
    conv = auth.ConversationState(username='u', conv_id='c')
    conv.session_state = {'transcript_message_count': 2}
    conv.messages_history = [{'role': 'user', 'content': 'old'}, {'role': 'assistant', 'content': 'old'},
                             {'role': 'user', 'content': 'unconsumed redirect'}]
    conv.session = SimpleNamespace(export_state=lambda: {'checkpoint_at': 100, 'goal_contract': {}})
    saved = auth._conv_to_dict(conv)['session_state']
    assert saved['transcript_message_count'] == 2
    assert saved['owner'] == {'username': 'u', 'conv_id': 'c'}


def test_repeat_finish_returns_original_proof_without_dispatch():
    from agents.parallel_workflow import ParallelWorkflow
    proof = {'evidence_call_ids': ['oldproof'], 'conclusion': 'Original verified receipt'}
    runtime = SimpleNamespace(snapshot=lambda: {'nodes': {'mix': {
        'status': 'succeeded', 'node_verification': proof}}})
    receipt = ParallelWorkflow.finish_node(runtime, 'mix', [], 'Not a new verification')
    assert receipt['already_finished'] and receipt['verification'] == proof
    assert receipt['verification'] is not proof


def test_readonly_success_uses_executor_output_validation_not_a_scientific_verification():
    goal, line, runtime = fixture()
    contract = {'step_id': 'summary', 'agent': 'analyst', 'tool': 'read_file',
                'arguments': {'path': 'results.csv'}, 'depends_on': [], 'expected_outputs': ['results.csv']}
    runtime['nodes'] = {'summary': {'status': 'succeeded', 'contract': contract}}
    assert workflow_completion(line, runtime, [contract], 8)['ok']


def test_old_framework_gate_prompts_are_recomputed_after_restore(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    state = session.export_state()
    state['messages'] = [{'role': 'user', 'content': 'Continue my existing research'},
        {'role': 'user', 'content': '[系统·结构化完成门禁] old_error=failed'}]
    assert session.import_state(state)
    assert session.messages == [state['messages'][0]]


def test_legacy_pending_delivery_still_blocks_four_receipts_but_error_history_does_not():
    line = {'steps': [{'step_id': 'deliver', 'tool': 'task_line_update', 'status': 'pending'},
                      {'step_id': 'old', 'status': 'failed', 'branch_parent': 'read_file'}]}
    assert workflow_completion(line, {}, [], 0)['blockers'] == [{'step_id': 'deliver', 'status': 'pending'}]
    line['steps'][0]['status'] = 'completed'
    assert workflow_completion(line, {}, [], 0)['ok']


def test_changed_science_cannot_complete_using_an_old_runtime_with_missing_approval(tmp_path, monkeypatch):
    from agents import task_line
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    goal, line, runtime = fixture()
    goal.approved_nodes = []
    goal.parameters['temperature_K'] = 308
    session.goal_contract = goal
    session._current_line_id = 'c'
    session._runtime_snapshot = lambda: runtime
    monkeypatch.setattr(task_line, 'get_store', lambda: SimpleNamespace(get_line=lambda _: line))
    state = session._workflow_completion_state()
    assert not state['ok'] and state['reason'] == 'approved_contract_missing'


def test_valid_delivery_does_not_require_programmed_words_or_fake_numbers(tmp_path, monkeypatch):
    from agents.defns import ORCHESTRATOR
    from agents import task_line
    from anthropic.types import TextBlock
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    goal, line, runtime = fixture()
    session.goal_contract = goal
    session._current_line_id = 'c'
    session._runtime_snapshot = lambda: runtime
    session._pending_jobs_for_conv = lambda: []
    session.messages = [{'role': 'user', 'content': goal.original_goal}]
    session.memory.record_tool_call('analyst', 'run_cdft', {'action': 'submit'}, '{"kind":"MOCKfixture"}')
    monkeypatch.setattr(task_line, 'get_store', lambda: SimpleNamespace(get_line=lambda _: line))
    # Deliberately no mandatory conclusion words, units, numbers or tables.
    answer = 'This is an explicit MOCK neutral process artifact, not a physical adsorption value. ' * 8
    responses = iter([SimpleNamespace(content=[TextBlock(type='text', text=answer)])])
    session._call_api = lambda _: next(responses)
    assert session._execute_loop(ORCHESTRATOR) == answer
    assert session.task_complete


def test_reply_to_budget_question_is_not_saved_as_a_scientific_method(tmp_path):
    from agents.defns import ORCHESTRATOR
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session.goal_contract = GoalContract.from_user_message('用cDFT计算Xe/Kr，298K')
    session.current_agent = ORCHESTRATOR
    session._waiting_for_user_input = True
    session._pending_user_interaction = {'tool': 'budget_decision', 'params': {}, 'prompt': 'Budget reached'}
    session._execute_loop = lambda *a, **k: 'Existing record delivered'
    answer = '只告诉我结果文件在哪里，不要重做。'
    assert session.reply(answer, verbose=False) == 'Existing record delivered'
    assert session.memory.user_preferences['budget_decision'] == {'answer': answer}
    assert session.goal_contract.method == 'CDFT'
