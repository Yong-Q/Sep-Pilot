import copy
from types import SimpleNamespace

import pytest

from agents.workflow_view import authoritative_line, bind_result_handoff
from agents.reflection_progress import ReflectionProgress


def fixture():
    contract = {'agent': 'analyst', 'tool': 'read_file', 'arguments': {'path': 'a.cif'},
        'depends_on': [], 'expected_outputs': ['a.cif']}
    line = {'username': 'u', 'conv_id': 'c', 'plan_version': 1,
        'steps': [{'step_id': 'a', **contract, 'status': 'completed', 'done': True, 'job_ids': ['old']}]}
    runtime = {'username': 'u', 'conv_id': 'c', 'plan_version': 1, 'status': 'active',
        'nodes': {'a': {'contract': contract, 'status': 'prefinish', 'phase': 'prefinish', 'job_ids': ['actual']}}}
    return line, runtime


def test_unverified_runtime_result_cannot_be_completed_by_task_line():
    line, runtime = fixture()
    before = copy.deepcopy(line)
    node = authoritative_line(line, runtime)['steps'][0]
    assert node['status'] == 'prefinish' and not node['done'] and node['job_ids'] == ['actual']
    assert node['reported_state']['done'] and line == before


def test_wrong_scope_cannot_share_runtime_memory():
    line, runtime = fixture()
    runtime['username'] = 'other'
    with pytest.raises(PermissionError): authoritative_line(line, runtime)


def test_version_mismatch_is_visible_and_old_success_not_a_new_contract():
    line, runtime = fixture()
    line['plan_version'] = 2
    line['steps'][0]['arguments'] = {'path': 'new.cif'}
    runtime['nodes']['a']['status'] = 'succeeded'
    view = authoritative_line(line, runtime)
    assert view['execution_plan_version'] == 1
    assert not view['steps'][0]['contract_matches_execution']
    assert view['steps'][0]['status'] == 'pending'
    assert view['steps'][0]['job_ids'] == []
    assert view['steps'][0]['previous_execution']['status'] == 'succeeded'


def test_production_binding_preserves_archived_job_parameter():
    session = SimpleNamespace()
    calls = []
    runtime = SimpleNamespace(revalidate_outputs=lambda *a, **k: calls.append((a, k)) or {'ok': True}, finish_node=lambda *a: None)
    bind_result_handoff(session, lambda: runtime)
    session._on_workflow_revalidate('gen', ['original.cif'], completed_job_id='77316')
    assert calls == [(('gen', ['original.cif']), {'completed_job_id': '77316'})]


def test_failed_calls_and_fresh_ids_do_not_count_as_progress():
    progress = ReflectionProgress()
    for i in range(4):
        receipt = progress.observe('write_file', {'path': 'a'}, {'error': 'permission denied', 'evidence_ref': {'call_id': str(i)}}, {'status': 'failed'})
        assert not receipt['progress']
    assert receipt['stalled_calls'] == 4 and receipt['last_problem']['actual_result']['error'] == 'permission denied'


def test_changed_schema_repair_progresses_but_identical_invalid_call_stalls():
    progress = ReflectionProgress()
    result = {'error': 'parameter validation failed', 'error_kind': 'schema_validation',
              'missing_parameters': ['output_path']}
    assert progress.observe('generate_scientific_report', {'title': 'R'}, result, {})['progress']
    assert not progress.observe('generate_scientific_report', {'title': 'R'}, result, {})['progress']
    assert progress.observe(
        'generate_scientific_report', {'title': 'R', 'source_steps': ['collect']}, result, {}
    )['progress']


def test_identical_success_query_is_not_new_evidence_but_changed_result_is():
    progress = ReflectionProgress()
    assert progress.observe('read_file', {'path': 'a'}, {'content': 'old', 'time': 1}, {})['progress']
    assert not progress.observe('read_file', {'path': 'a'}, {'content': 'old', 'time': 2}, {})['progress']
    assert progress.observe('read_file', {'path': 'a'}, {'content': 'fixed'}, {})['progress']
    assert progress.observe('read_file', {'path': 'a'}, {'content': 'fixed'}, {'status': 'succeeded'})['progress']


def test_file_success_does_not_resolve_another_files_error(tmp_path):
    from agents.config import AgentConfig
    from agents.session import Session
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    first = session._create_error_branch('read_file', 'failed a')
    second = session._create_error_branch('read_file', 'failed b')
    session._error_branches[first]['failed_arguments'] = {'path': 'a'}
    session._error_branches[second]['failed_arguments'] = {'path': 'b'}
    session._resolve_error_branches_for_tool('read_file', '{"content":"actual a"}', {'path': 'a'})
    assert session._error_branches[first]['status'] == 'resolved'
    assert session._error_branches[second]['status'] == 'open'


def test_corrected_contract_call_resolves_prior_parameter_branch(tmp_path):
    from agents.config import AgentConfig
    from agents.session import Session
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    branch = session._create_error_branch(
        'generate_scientific_report', 'missing output_path', category='contract_validation'
    )
    session._error_branches[branch]['failed_arguments'] = {'title': 'R'}
    session._resolve_error_branches_for_tool(
        'generate_scientific_report', '{"status":"success"}',
        {'title': 'R', 'output_path': 'report.md'},
    )
    assert session._error_branches[branch]['status'] == 'resolved'


def test_changed_contract_failures_share_counter_and_auto_handoff_to_patcher(tmp_path):
    from agents.config import AgentConfig
    from agents.defns import ORCHESTRATOR
    from agents.session import Session
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session.current_agent = ORCHESTRATOR
    for index in range(3):
        arguments = {'title': 'Report', 'attempt': index}
        result = {'error': 'parameter validation failed', 'error_kind': 'schema_validation',
                  'issues': ['missing required parameter(s): output_path']}
        encoded = __import__('json').dumps(result)
        session.memory.record_tool_call(
            'lead-orchestrator', 'generate_scientific_report', arguments, encoded, failed=True,
        )
        session.memory.tool_call_log[-1]['workflow_step_id'] = 'report'
        session._handle_failures(
            [{'type': 'tool_result', 'tool_use_id': f'call-{index}',
              'content': __import__('json').dumps(result)}],
            [{'type': 'tool_use', 'id': f'call-{index}',
              'name': 'generate_scientific_report', 'input': arguments}],
            'lead-orchestrator',
        )
    matching = {key: value for key, value in session._failure_retries.items()
                if key.startswith('contract:')}
    assert list(matching.values()) == [3]
    assert session._pending_handoff.name == 'patcher'
    assert '原DAG继续' in session.context['automatic_handoff_task']
