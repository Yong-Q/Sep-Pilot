from tests.native_child_capture import native_source
"""Bounded barriers and fake scheduler receipts, never real submissions/LLMs."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import (ParallelWorkflow, WorkflowStore, conflicts,
                                      resources_for, dependency_runtime_arguments,
                                      runtime_summary)
from agents.registry import _build_default_registry
from agents.recovery import RecoveryGate, input_fingerprint
from agents.session import Session
from agents.result_validation import VALIDATION_SCHEMA_VERSION
from agents.state_io import json_transaction,write_checkpoint
from agents.task_line import TaskLineStore


def test_model_gate_defaults_to_four_parallel_calls(monkeypatch):
    from agents import model_gate
    monkeypatch.delenv('BIMEM_MODEL_CALL_CONCURRENCY', raising=False)
    assert model_gate._configured_concurrency() == 4


def test_find_cif_is_a_readonly_runtime_query():
    from agents.parallel_workflow import READ_ONLY
    assert 'find_cif' in READ_ONLY


def test_configuration_recovery_preserves_failed_attempt_and_completed_branches(tmp_path):
    path = tmp_path / 'workflow.json'
    failed = {'status': 'failed', 'token': 'old', 'attempt_id': 'attempt',
              'tool_returned': True, 'job_ids': [], 'error': 'old preflight',
              'contract': {'tool': 'generate_structure'}}
    path.write_text(json.dumps({'workflows': {'w': {'status': 'needs_user', 'pause_kind': 'failure',
        'nodes': {'generate': failed, 'literature': {'status': 'succeeded'}}}}, 'leases': {}}))
    store = WorkflowStore(path)
    assert store.requeue_configuration_preflight('w', 'generate', 'old', {
        'kind': 'verified_configuration_preflight', 'attempt_id': 'attempt', 'no_job_submitted': True})
    state = store.snapshot('w')
    assert state['status'] == 'active'
    assert state['nodes']['generate']['status'] == 'pending'
    assert state['nodes']['generate']['attempt_history'][0]['error'] == 'old preflight'
    assert state['nodes']['literature']['status'] == 'succeeded'
    assert not store.requeue_configuration_preflight('w', 'generate', 'old', {})


def test_verified_retry_requeues_only_matching_failed_attempt_and_keeps_history(tmp_path):
    path = tmp_path / 'workflow.json'
    failed = {'status': 'failed', 'token': 'old', 'attempt_id': 'attempt', 'recovery_key': 'key',
              'tool_returned': True, 'job_ids': ['42'], 'error': 'real job failure',
              'configuration_recovery': {'attempt_id': 'earlier-preflight'},
              'contract': {'tool': 'generate_structure'}}
    path.write_text(json.dumps({'workflows': {'w': {'status': 'needs_user', 'pause_kind': 'failure',
        'nodes': {'generate': failed, 'literature': {'status': 'succeeded'}}}}, 'leases': {}}))
    store = WorkflowStore(path)
    permit = {'key': 'key', 'status': 'failed', 'attempt_id': 'attempt', 'approved_attempt_id': 'attempt',
              'approved_fingerprint': 'verified', 'attempts': 1}
    assert not store.requeue_verified_retry('w', 'generate', 'wrong', permit)
    assert store.requeue_verified_retry('w', 'generate', 'old', permit)
    state = store.snapshot('w')
    assert state['nodes']['generate']['attempt_history'][0]['job_ids'] == ['42']
    assert state['nodes']['generate']['status'] == 'pending'
    assert 'configuration_recovery' not in state['nodes']['generate']
    assert state['nodes']['generate']['attempt_history'][0]['configuration_recovery']['attempt_id'] == 'earlier-preflight'
    assert state['nodes']['literature']['status'] == 'succeeded'
    assert state['status'] == 'active'
    assert not store.requeue_verified_retry('w', 'generate', 'old', permit)


@pytest.mark.parametrize('blocker', [None, 'token', 'permit_attempt', 'proof_attempt', 'error',
                                    'tool_returned', 'job_ids', 'attempt_id', 'live_lease', 'user_pause',
                                    'missing_token', 'missing_permit', 'missing_proof', 'verified_attempt', 'dispatch_phase'])
def test_stale_configuration_preclaim_migration_is_exact_and_single_use(tmp_path, blocker):
    failed = {'status': 'failed', 'token': 'current', 'attempt_id': None, 'recovery_key': 'key',
              'dispatch_phase': 'preflight', 'tool_returned': False, 'job_ids': [],
              'error': 'configuration recovery requires an unexecuted generator preflight failure',
              'configuration_recovery': {'attempt_id': 'older-preflight'},
              'verified_retry': {'attempt_id': 'real-failure'},
              'contract': {'tool': 'generate_structure'}}
    permit = {'key': 'key', 'status': 'failed', 'attempt_id': 'real-failure',
              'approved_attempt_id': 'real-failure', 'approved_fingerprint': 'verified', 'attempts': 2}
    if blocker == 'permit_attempt': permit['approved_attempt_id'] = 'other'
    if blocker == 'missing_token': failed['token'] = None
    if blocker == 'missing_permit': permit['approved_fingerprint'] = None
    if blocker == 'missing_proof': failed.pop('configuration_recovery')
    if blocker == 'verified_attempt': failed['verified_retry']['attempt_id'] = 'other'
    if blocker == 'dispatch_phase': failed['dispatch_phase'] = 'entered_tool'
    if blocker == 'proof_attempt': failed['configuration_recovery']['attempt_id'] = 'real-failure'
    if blocker == 'error': failed['error'] = 'unrelated preclaim failure'
    if blocker == 'tool_returned': failed['tool_returned'] = True
    if blocker == 'job_ids': failed['job_ids'] = ['live-job']
    if blocker == 'attempt_id': failed['attempt_id'] = 'new-claim'
    workflow = {'status': 'needs_user', 'pause_kind': 'failure',
                'user_paused': blocker == 'user_pause', 'nodes': {'generate': failed}}
    leases = {'current': {'workflow_id': 'w', 'step_id': 'generate'}} if blocker == 'live_lease' else {}
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps({'workflows': {'w': workflow}, 'leases': leases}))
    store = WorkflowStore(path)
    before = store.snapshot()
    assert store.requeue_verified_retry('w', 'generate', 'wrong' if blocker == 'token' else 'current', permit) is (blocker is None)
    if blocker:
        assert store.snapshot() == before
    else:
        recovered = store.snapshot('w')['nodes']['generate']
        assert recovered['status'] == 'pending'
        assert recovered['verified_retry']['attempt_id'] == 'real-failure'
        assert 'configuration_recovery' not in recovered
        assert recovered['attempt_history'][0]['error'] == failed['error']
        assert not store.requeue_verified_retry('w', 'generate', 'current', permit)


@pytest.mark.parametrize('blocker', ['user_pause', 'live_lease'])
def test_verified_retry_preserves_user_pause_and_live_lease(tmp_path, blocker):
    failed = {'status': 'failed', 'token': 'old', 'attempt_id': 'attempt',
              'recovery_key': 'key', 'job_ids': ['42'], 'contract': {'tool': 'generate_structure'}}
    workflow = {'status': 'needs_user', 'pause_kind': 'failure',
                'user_paused': blocker == 'user_pause', 'nodes': {'generate': failed}}
    leases = {'old': {'workflow_id': 'w', 'step_id': 'generate', 'owner_pid': os.getpid()}} if blocker == 'live_lease' else {}
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps({'workflows': {'w': workflow}, 'leases': leases}))
    store = WorkflowStore(path)
    before = store.snapshot()
    permit = {'key': 'key', 'status': 'failed', 'attempt_id': 'attempt',
              'approved_attempt_id': 'attempt', 'approved_fingerprint': 'verified', 'attempts': 1}

    assert not store.requeue_verified_retry('w', 'generate', 'old', permit)
    assert store.snapshot() == before


def test_legacy_compatible_reset_restores_only_affected_completed_branch(tmp_path):
    state_path = tmp_path / 'parallel.json'
    store = WorkflowStore(state_path)
    workflow_id = 'legacy-compatible-reset'
    charged = tmp_path / 'charged'
    charged.mkdir()
    charged_cif = charged / 'a_pacmof.cif'
    charged_cif.write_text('charged structure\n')
    result_csv = tmp_path / 'results.csv'
    result_csv.write_text('name,value\na,1\n')
    unrelated_file = tmp_path / 'unrelated.txt'
    unrelated_file.write_text('old result\n')
    original = {
        'step_id': 'charge', 'agent': 'analyst', 'tool': 'run_pacman_charge',
        'arguments': {'cif_dir': str(tmp_path / 'cifs'), 'method': 'pacmof',
                      'output_dir': str(charged)},
        'depends_on': [], 'expected_outputs': [{
            'kind': 'directory', 'path': str(charged),
            'pattern': '*_pacmof.cif', 'min_count': 3,
        }], 'resource_locks': [],
    }
    revised = copy.deepcopy(original)
    revised['arguments']['completion_policy'] = 'best_effort'
    revised['expected_outputs'][0]['min_count'] = 1
    child = node('collect', deps=['charge'], args={
        'path': str(charged), 'resource_review_id': 'new-review',
    }, outputs=[str(result_csv)])
    old_child = copy.deepcopy(child)
    old_child['arguments']['resource_review_id'] = 'old-review'
    unrelated = node('unrelated', outputs=[str(unrelated_file)])

    def prior(contract, artifact, *, submission=False):
        value = {
            'contract': copy.deepcopy(contract), 'status': 'succeeded',
            'artifacts': {str(artifact): True}, 'path_manifest': {
                'output_files': [str(artifact)],
            },
        }
        if submission:
            value.update(job_ids=['78241'], node_verification={'decision': 'pass'})
        return value

    current_token = 'stale-current-lease'
    payload = {
        'workflows': {workflow_id: {
            'username': 'tester', 'conv_id': 'c1', 'plan_version': 2,
            'goal_contract': {'version': 2}, 'status': 'active',
            'nodes': {
                'charge': {'contract': revised, 'status': 'pending',
                           'token': current_token,
                           'attempt_history': [prior(original, charged_cif, submission=True)]},
                'collect': {'contract': child, 'status': 'pending',
                            'attempt_history': [prior(old_child, result_csv)]},
                'unrelated': {'contract': unrelated, 'status': 'pending',
                              'attempt_history': [prior(unrelated, unrelated_file)]},
            },
        }},
        'leases': {current_token: {'workflow_id': workflow_id, 'step_id': 'charge',
                                   'resources': [], 'owner_pid': 1}},
    }
    state_path.write_text(json.dumps(payload))

    restored = store.restore_compatible_completed_branches(workflow_id)
    state = store.snapshot(workflow_id)

    assert restored == ['charge', 'collect']
    assert state['nodes']['charge']['status'] == 'succeeded'
    assert state['nodes']['charge']['contract'] == revised
    assert state['nodes']['collect']['status'] == 'succeeded'
    assert state['nodes']['unrelated']['status'] == 'pending'
    assert current_token not in store.snapshot().get('leases', {})


@pytest.mark.parametrize('protection', ['running', 'waiting_jobs', 'waiting_prerequisite', 'prefinish', 'uncertain', 'user_paused', 'user_change', 'input_changed', 'changed_artifact', 'other_failure'])
def test_compatible_restoration_preserves_execution_and_recovery_boundaries(tmp_path, protection):
    artifact = tmp_path / 'result.txt'
    artifact.write_text('verified result')
    from agents.output_contract import artifact_fact
    original = node('restored', args={}, outputs=[str(artifact)])
    revised = copy.deepcopy(original)
    revised['arguments']['completion_policy'] = 'best_effort'
    prior = {'status': 'succeeded', 'contract': original,
             'artifacts': {str(artifact): artifact_fact(str(artifact), tmp_path)}}
    current = {'status': 'pending', 'contract': revised, 'attempt_history': [prior]}
    workflow = {'status': 'needs_user', 'pause_kind': 'failure', 'pause_reason': 'repair required',
                'nodes': {'restored': current}}
    if protection in {'running', 'waiting_jobs', 'waiting_prerequisite', 'prefinish', 'uncertain'}:
        current.update(status=protection, token='live-token')
    elif protection == 'user_paused':
        workflow['user_paused'] = True
    elif protection in {'user_change', 'input_changed'}:
        workflow['pause_kind'] = protection
    elif protection == 'changed_artifact':
        artifact.write_text('replaced unverified output with different contents')
    else:
        workflow['nodes']['failed'] = {'status': 'validation_failed', 'contract': node('failed')}
    path = tmp_path / 'workflow.json'
    payload = {'workflows': {'w': workflow}, 'leases': {'live-token': {'step_id': 'restored'}}}
    path.write_text(json.dumps(payload))
    store = WorkflowStore(path)

    restored = store.restore_compatible_completed_branches('w')

    if protection == 'other_failure':
        assert restored == ['restored']
        assert store.snapshot('w')['status'] == 'needs_user'
        assert store.snapshot('w')['pause_reason'] == 'repair required'
    else:
        assert restored == []
        assert store.snapshot() == payload


@pytest.mark.parametrize('join_migrated', [False, True])
def test_compatible_restoration_does_not_restore_join_with_failed_dependency(tmp_path, join_migrated):
    artifact = tmp_path / 'result.txt'
    artifact.write_text('verified result')
    original = node('restored', outputs=[str(artifact)])
    revised = copy.deepcopy(original)
    revised['arguments']['completion_policy'] = 'best_effort'
    child = node('join', deps=['restored', 'failed'], outputs=[str(artifact)])
    old_child = copy.deepcopy(child)
    if join_migrated:
        child['arguments']['completion_policy'] = 'best_effort'
    def prior(contract):
        return {'status': 'succeeded', 'contract': contract, 'artifacts': {str(artifact): True}}
    states = {
        'restored': {'status': 'pending', 'contract': revised, 'attempt_history': [prior(original)]},
        'join': {'status': 'pending', 'contract': child, 'attempt_history': [prior(old_child)]},
        'failed': {'status': 'failed', 'contract': node('failed')},
    }
    path = tmp_path / 'workflow.json'
    path.write_text(json.dumps({'workflows': {'w': {'status': 'needs_user', 'nodes': states}}}))
    store = WorkflowStore(path)

    assert store.restore_compatible_completed_branches('w') == ['restored']
    assert store.snapshot('w')['nodes']['join']['status'] == 'pending'
    assert store.snapshot('w')['status'] == 'needs_user'


def test_branch_failure_keeps_independent_ready_work_and_blocks_descendants(factory):
    rt = factory([node('bad'), node('dependent', deps=['bad']), node('independent')])
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'bad')
    rt.store.update(rt.workflow_id, 'bad', ticket['token'], {'status': 'failed'}, release=True)
    rt.store.pause(rt.workflow_id, 'failure', step_id='bad')
    assert rt.snapshot()['status'] == 'active'
    assert rt.store.claim(rt.workflow_id, 'dependent') is None
    independent = rt.store.claim(rt.workflow_id, 'independent')
    assert independent
    rt.store.update(rt.workflow_id, 'independent', independent['token'], {'status': 'succeeded'}, release=True)


def stopped_branch_dag(factory, status='needs_resources', pause_kind='resource_review'):
    rt = factory([node('gen'), node('charge', deps=['gen']),
        node('qc', deps=['charge']), node('pore', deps=['qc']),
        node('report', deps=['pore'])])
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        state = data['workflows'][rt.workflow_id]
        for key in ('gen', 'charge'):
            state['nodes'][key].update(status='succeeded', phase='finish', result={'content': key})
        state['nodes']['pore'].update(status=status, token='old-pore', dispatch_phase='preflight',
            tool_returned=False, job_ids=[], resource_review_attempts=3)
        state.update(status='needs_user', pause_kind=pause_kind, pause_reason='old branch-local failure')
    return rt


@pytest.mark.parametrize('status', ['failed', 'validation_failed', 'needs_resources'])
@pytest.mark.parametrize('pause_kind', [None, 'failure', 'resource_review'])
def test_restart_dispatches_independent_qc_despite_stopped_downstream_branch(factory, status, pause_kind):
    rt = stopped_branch_dag(factory, status, pause_kind)
    calls = []
    rt.main.registry.get('read_file').execute = lambda args: calls.append(args['path']) or {'content': 'qc checked'}
    before = rt.snapshot()
    rt.main._pending_workflow_patch = {'affected_nodes': ['pore', 'report'], 'plan_version': 2}
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['qc']['status'] == 'succeeded')
    rt.tick()
    state = rt.snapshot()
    assert calls == ['qc']
    assert state['nodes']['pore'] == before['nodes']['pore']
    assert state['nodes']['report']['status'] == 'pending'
    assert all(state['nodes'][key] == before['nodes'][key] for key in ('gen', 'charge'))


@pytest.mark.parametrize('blocker', ['user_pause', 'user_change', 'input_changed', 'uncertain',
    'owner', 'goal_scope', 'unapproved', 'line_scope', 'affected_qc', 'lease', 'awaiting_approval'])
def test_branch_resume_retains_approval_pause_and_lease_guards(factory, blocker):
    rt = stopped_branch_dag(factory)
    calls = []
    rt.main.registry.get('read_file').execute = lambda args: calls.append(args['path']) or {'content': 'qc checked'}
    with json_transaction(rt.store.path) as data:
        state = data['workflows'][rt.workflow_id]
        if blocker == 'user_pause': state['user_paused'] = True
        if blocker in {'user_change', 'input_changed'}: state['pause_kind'] = blocker
        if blocker == 'uncertain': state['nodes']['pore']['status'] = 'uncertain'
        if blocker == 'owner': state['username'] = 'other'
        if blocker == 'lease':
            resources = state['nodes']['qc']['contract']['resources']
            data.setdefault('leases', {})['other-owner'] = {'workflow_id': 'other', 'step_id': 'other',
                'resources': [{**resource, 'mode': 'exclusive'} for resource in resources]}
    if blocker == 'goal_scope': rt.main.goal_contract.parameters['temperature'] = 310
    if blocker == 'unapproved': rt.main.goal_contract.execution_authorized = False
    if blocker == 'line_scope':
        rt.main.goal_contract.approved_nodes[2]['arguments']['path'] = 'different-qc'
    if blocker == 'affected_qc':
        rt.main._pending_workflow_patch = {'affected_nodes': ['qc'], 'plan_version': 2}
    if blocker == 'awaiting_approval': rt.main._awaiting_plan_approval = True
    rt.tick()
    assert calls == []
    assert rt.snapshot()['nodes']['qc']['status'] == 'pending'


@pytest.mark.parametrize('status', ['failed', 'validation_failed', 'needs_resources'])
@pytest.mark.parametrize('revised', [False, True])
def test_activate_and_start_report_partial_when_independent_ready_branch_remains(factory, status, revised):
    rt = factory([node('bad'), node('independent'), node('dependent', deps=['bad'])])
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        state = data['workflows'][rt.workflow_id]
        state['nodes']['bad'].update(status=status, token='failed-token')
        state.update(status='needs_user', pause_kind='resource_review')
    if revised:
        rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
            [{'operation': 'upsert', 'step_id': 'independent', 'node': node('independent')}], 1, 2, 'approved revision')
        rt.main.goal_contract.approved_plan_version = 2
    receipt = rt.start(2 if revised else 1)
    assert receipt['scheduled'] is True
    assert receipt['status'] == 'partial'
    assert receipt['blocked_nodes'] == ['bad', 'dependent']
    assert rt.snapshot()['status'] == 'active'
    assert rt.store.claim(rt.workflow_id, 'bad') is None
    assert rt.store.claim(rt.workflow_id, 'dependent') is None
    assert rt.store.claim(rt.workflow_id, 'independent')


def test_start_partial_preserves_live_branch_without_claiming_failed_nodes_recovered(factory):
    rt = factory([node('bad'), node('live')])
    rt.start(1)
    live = rt.store.claim(rt.workflow_id, 'live')
    with json_transaction(rt.store.path) as data:
        state = data['workflows'][rt.workflow_id]
        state['nodes']['bad'].update(status='validation_failed', token='old-bad')
        state.update(status='needs_user', pause_kind='failure')
    receipt = rt.start(1)
    assert receipt['status'] == 'partial' and receipt['scheduled'] is True
    assert receipt['blocked_nodes'] == ['bad']
    assert rt.snapshot()['nodes']['live'] == live
    assert rt.snapshot()['nodes']['bad']['status'] == 'validation_failed'


def test_parallel_report_fragment_is_claimable_while_other_compute_runs(factory, tmp_path):
    fragment_path = tmp_path / 'runs' / 'tester' / 'c1' / 'reports' / 'fragments' / 'a.md'
    report = node('report__fragment__a', tool='generate_scientific_report', args={
        'source_steps': ['a'], 'output_path': str(fragment_path),
        'title': 'A', 'report_mode': 'fragment',
    }, deps=['a'], outputs=[str(fragment_path)])
    report.update(agent='communicator', report_role='fragment', report_parent='report')
    rt = factory([node('a'), node('b'), report])
    rt.start(1)
    a = rt.store.claim(rt.workflow_id, 'a')
    b = rt.store.claim(rt.workflow_id, 'b')
    assert a and b
    rt.store.update(rt.workflow_id, 'a', a['token'], {
        'status': 'succeeded', 'execution_fingerprint': 'attempt-a',
    }, release=True)

    fragment = rt.store.claim(rt.workflow_id, 'report__fragment__a')

    assert fragment
    assert fragment['contract']['report_role'] == 'fragment'
    assert rt.snapshot()['nodes']['b']['status'] == 'running'
    assert b['token'] in rt.store.snapshot().get('leases', {})


def test_parallel_report_assembles_current_fragments_and_rejects_stale_source(factory, tmp_path, monkeypatch):
    from tests.public_report_fixture_helpers import (MEASURED_CSV, install_controlled_report_provider,
        advance_native, assert_native_report_stages)
    root = tmp_path / 'runs' / 'tester' / 'c1'
    source = root / 'source.csv'
    fragment_path = root / 'reports' / 'fragments' / 'source.md'
    final_path = root / 'reports' / 'final.md'
    fragment = node('final__fragment__source', tool='generate_scientific_report', args={
        'source_steps': ['source'], 'output_path': str(fragment_path),
        'title': 'Source result', 'report_mode': 'fragment',
    'report_requirements': {'independent_review': True}}, deps=['source'], outputs=[str(fragment_path)])
    fragment.update(agent='communicator', report_role='fragment', report_parent='final')
    final = node('final', tool='generate_scientific_report', args={
        'source_steps': ['final__fragment__source'], 'output_path': str(final_path),
        'title': 'Live report', 'report_mode': 'assembly',
    'report_requirements': {'independent_review': True}}, deps=['final__fragment__source'], outputs=[str(final_path)])
    final.update(agent='communicator', report_role='assembly')
    data = node('source', tool='write_file', args={'path': str(source), 'content': MEASURED_CSV}, outputs=[str(source)])
    rt = factory([data, fragment, final])
    calls = install_controlled_report_provider(rt, monkeypatch)
    rt.start(1)
    assert advance_native(rt, 'source', blocked=['final'])['status'] == 'succeeded'
    completed = advance_native(rt, 'final__fragment__source', blocked=['final'])
    assert completed['status'] == 'succeeded', completed.get('error')
    fragment_result = assert_native_report_stages(rt, completed)
    assert fragment_result['source_step'] == 'source'
    assert fragment_result['source_attempt_fingerprint']
    assert fragment_result['source_evidence_call_id'] == rt.snapshot()['nodes']['source']['result_ref']['call_id']
    assert 'measured value' in fragment_path.read_text()
    assert '| sample | 7 | controlled_native | unknown |' in fragment_path.read_text().replace(r'\_', '_')
    quantity = next(binding for binding in fragment_result['measurement_bindings'] if binding['column'] == 'measured_value')
    assert quantity['source_value'] == '7'
    assert quantity['source_sha256'] == __import__('hashlib').sha256(source.read_bytes()).hexdigest()
    completed_final = advance_native(rt, 'final')
    assert completed_final['status'] == 'succeeded', completed_final.get('error')
    final_result = assert_native_report_stages(rt, completed_final)
    # Native authored reports replaced the old JSON-index fragment manifest;
    # exact source receipts retain stronger token/call/content-digest lineage.
    manifest = final_result['source_receipts']
    assert manifest[0]['step_id'] == 'final__fragment__source'
    assert manifest[0]['token'] == completed['token']
    assert manifest[0]['call_id'] == completed['result_ref']['call_id']
    assert manifest[0]['source_sha256'] == __import__('hashlib').sha256(Path(completed['result_ref']['evidence_path']).read_bytes()).hexdigest()
    assert 'Source result' in final_path.read_text()
    assert final_result['report_preview'].startswith('# Live report')
    assert final_result['report_preview'] == final_path.read_text()[:24000]
    live = runtime_summary(rt.snapshot())['live_reports']
    assert live[0]['role'] == 'assembly'
    assert live[0]['content'].startswith('# Live report')
    assert {'author', 'review'} == {call['phase'] for call in calls}
    before = {path: path.stat().st_mtime_ns for path in (fragment_path, final_path)}
    assert rt.start(1)['status'] in {'scheduled', 'already_active'}
    rt.tick()
    assert rt.snapshot()['status'] == 'completed'
    assert {path: path.stat().st_mtime_ns for path in (fragment_path, final_path)} == before



def test_parallel_report_rejects_fragment_from_stale_source_attempt(factory, tmp_path, monkeypatch):
    from tests.public_report_fixture_helpers import (MEASURED_CSV, install_controlled_report_provider,
        advance_native, assert_native_report_stages)
    root = tmp_path / 'runs' / 'tester' / 'stale'
    source = root / 'stale-source.csv'
    fragment_path = root / 'reports' / 'fragments' / 'source.md'
    final_path = root / 'reports' / 'final.md'
    fragment = node('final__fragment__source', tool='generate_scientific_report', args={
        'source_steps': ['source'], 'output_path': str(fragment_path),
        'title': 'Source result', 'report_mode': 'fragment',
    'report_requirements': {'independent_review': True}}, deps=['source'], outputs=[str(fragment_path)])
    fragment.update(agent='communicator', report_role='fragment', report_parent='final')
    final = node('final', tool='generate_scientific_report', args={
        'source_steps': ['final__fragment__source'], 'output_path': str(final_path),
        'title': 'Live report', 'report_mode': 'assembly',
    'report_requirements': {'independent_review': True}}, deps=['final__fragment__source'], outputs=[str(final_path)])
    final.update(agent='communicator', report_role='assembly')
    data = node('source', tool='write_file', args={'path': str(source), 'content': MEASURED_CSV}, outputs=[str(source)])
    rt = factory([data, fragment, final], cid='stale')
    install_controlled_report_provider(rt, monkeypatch)
    rt.start(1)
    assert advance_native(rt, 'source', blocked=['final'])['status'] == 'succeeded'
    completed = advance_native(rt, 'final__fragment__source', blocked=['final'])
    assert completed['status'] == 'succeeded', completed.get('error')
    receipt = assert_native_report_stages(rt, completed)
    original = receipt['source_attempt_fingerprint']
    assert original == rt.snapshot()['nodes']['source']['execution_fingerprint']
    with json_transaction(rt.store.path) as data:
        data['workflows'][rt.workflow_id]['nodes']['source']['execution_fingerprint'] = 'new-attempt'
    failed = advance_native(rt, 'final')
    assert failed['status'] == 'failed'
    assert 'stale source attempt' in failed['error']
    assert rt.snapshot()['nodes']['final__fragment__source']['result']['source_attempt_fingerprint'] == original
    assert not final_path.exists()


def test_ml_branch_finishes_and_is_added_to_final_report(factory, tmp_path, monkeypatch):
    from tests.public_report_fixture_helpers import (MEASURED_CSV, install_controlled_report_provider,
        controlled_safe_ml_product, assert_native_report_stages)
    root = tmp_path / 'runs' / 'tester' / 'ml-report'
    data_csv = root / 'training.csv'
    simulation = root / 'simulation.csv'
    model_dir = root / 'ml'
    sim_fragment = root / 'reports' / 'fragments' / 'simulation.md'
    ml_fragment = root / 'reports' / 'fragments' / 'ml.md'
    final_path = root / 'reports' / 'final.md'
    nodes = [
        node('simulation', tool='write_file', args={'path': str(simulation), 'content': MEASURED_CSV}, outputs=[str(simulation)]),
        {**node('train_ml', tool='ml_train', args={
            'data_csv': str(data_csv), 'target': 'uptake', 'model_type': 'RF',
            'output_dir': str(model_dir), 'synthetic_fixture': True,
        }, outputs=[{'kind': 'directory', 'path': str(model_dir), 'pattern': '*', 'min_count': 2}]), 'agent': 'analyst'},
        {**node('final__fragment__simulation', tool='generate_scientific_report', args={
            'source_steps': ['simulation'], 'output_path': str(sim_fragment),
            'title': 'Simulation', 'report_mode': 'fragment',
        'report_requirements': {'independent_review': True}}, deps=['simulation'], outputs=[str(sim_fragment)]),
         'agent': 'communicator', 'report_role': 'fragment', 'report_parent': 'final'},
        {**node('final__fragment__train_ml', tool='generate_scientific_report', args={
            'source_steps': ['train_ml'], 'output_path': str(ml_fragment),
            'title': 'Machine learning', 'report_mode': 'fragment',
        'report_requirements': {'independent_review': True}}, deps=['train_ml'], outputs=[str(ml_fragment)]),
         'agent': 'communicator', 'report_role': 'fragment', 'report_parent': 'final'},
        {**node('final', tool='generate_scientific_report', args={
            'source_steps': ['final__fragment__simulation', 'final__fragment__train_ml'],
            'output_path': str(final_path), 'title': 'Combined report', 'report_mode': 'assembly',
        'report_requirements': {'independent_review': True}}, deps=['final__fragment__simulation', 'final__fragment__train_ml'], outputs=[str(final_path)]),
         'agent': 'communicator', 'report_role': 'assembly'},
    ]
    rt = factory(nodes, cid='ml-report')
    # Fixed original labels and explicit deterministic split; no generated or
    # random candidate labels, no joblib, no backend estimator fitting.
    data_csv.write_text('feature,uptake\n1,2\n2,4\n3,6\n4,8\n5,10\n6,12\n7,14\n8,16\n')
    install_controlled_report_provider(rt, monkeypatch)
    rt.main.registry.get('ml_train').execute = controlled_safe_ml_product
    rt.start(1)
    verified_ml = False
    for _ in range(1000):
        rt.tick()
        current = rt.snapshot()['nodes']['train_ml']
        if current['status'] == 'prefinish' and not verified_ml:
            metadata = json.loads((model_dir / 'model_metadata.json').read_text())
            assert metadata['metrics']['held_out_r2'] == -49.0
            assert metadata['train_rows'] == [1, 2, 3, 4] and metadata['test_rows'] == [7, 8]
            assert not (model_dir / 'model.joblib').exists()
            dossier = current['validation_dossier']
            assert Path(dossier['evidence_path']).exists()
            rt.finish_node('train_ml', [dossier['evidence_call_id']],
                'Native safe JSON model, original fixed labels, deterministic split and actual metric output were checked; predictive validity remains unverified.')
            verified_ml = True
        if rt.snapshot()['status'] == 'completed': break
        time.sleep(.01)
    assert rt.snapshot()['status'] == 'completed', {
        key: (value['status'], value.get('error')) for key, value in rt.snapshot()['nodes'].items()}
    content = final_path.read_text()
    assert 'Machine learning' in content and 'held_out_r2' in content.replace(r'\_', '_')
    assert '-49.0' in content
    assert rt.snapshot()['nodes']['train_ml']['status'] == 'succeeded'
    ml_node = rt.snapshot()['nodes']['final__fragment__train_ml']
    assert ml_node['status'] == 'succeeded'
    receipt = assert_native_report_stages(rt, ml_node)
    metric = next(binding for binding in receipt['measurement_bindings'] if binding['row_selector'] == {'metric': 'held_out_r2'} and binding['column'] == 'value')
    assert float(metric['source_value']) == metadata['metrics']['held_out_r2']
    assert metric['source_sha256'] == __import__('hashlib').sha256((model_dir / 'metrics.csv').read_bytes()).hexdigest()
    assert rt.snapshot()['nodes']['final']['result']['report_preview'] == content[:24000]


def test_pending_patch_does_not_persist_user_pause(factory):
    rt = factory([node('a')])
    rt.start(1)
    rt.main._pending_workflow_patch = {'affected_nodes': ['a']}
    rt.tick(blocked_nodes=['a'])
    assert not rt.snapshot()['user_paused']
    assert rt.snapshot()['nodes']['a']['status'] == 'pending'


def test_start_and_tick_requires_a_real_root_claim(factory):
    rt = factory([node('a', path='/shared')])
    blocker = factory([node('blocker', tool='write_file',
                            args={'path': '/shared', 'content': 'held'})], cid='c2')
    blocker.start(1)
    assert blocker.store.claim(blocker.workflow_id, 'blocker')

    blocked = rt.start_and_tick(1)

    assert blocked['scheduled'] is False
    assert blocked['status'] == 'waiting_dispatch'
    assert 'resource lease conflict' in blocked['blocked_nodes']['a'][0]


def test_start_and_tick_reports_dispatch_only_after_root_claim(factory):
    rt = factory([node('a')])
    gate = threading.Event()
    rt.main.registry.get('read_file').execute = lambda args: gate.wait(5) or {'content': 'done'}

    receipt = rt.start_and_tick(1)

    assert receipt['scheduled'] is True
    assert receipt['dispatch_started'] is True
    assert receipt['root_nodes']['a']['token'] is True
    gate.set()


@pytest.mark.parametrize('leased', [False, True])
def test_partial_dispatch_receipt_does_not_reuse_completed_root_token(factory, leased):
    rt = factory([node('done'), node('bad'), node('qc', deps=['done']), node('report', deps=['bad'])])
    calls = []
    rt.main.registry.get('read_file').execute = lambda args: calls.append(args['path']) or {'content': 'checked'}
    rt.start(1)
    rt.tick(blocked_nodes=['bad', 'qc', 'report'])
    eventually(lambda: rt.snapshot()['nodes']['done']['status'] == 'succeeded')
    with json_transaction(rt.store.path) as data:
        state = data['workflows'][rt.workflow_id]
        state['nodes']['bad'].update(status='failed', token='failed-token')
        state.update(status='needs_user', pause_kind='failure')
        if leased:
            data.setdefault('leases', {})['lease-other'] = {'workflow_id': 'other', 'step_id': 'writer',
                'resources': [{**resource, 'mode': 'exclusive'}
                    for resource in state['nodes']['qc']['contract']['resources']]}
    receipt = rt.start_and_tick(1)
    if leased:
        assert receipt['scheduled'] is False
        assert receipt['dispatch_started'] is False
        assert calls == ['done']
    else:
        assert receipt['status'] == 'partial' and receipt['scheduled'] is True
        assert receipt['dispatch_started'] is True
        assert 'qc' in receipt['dispatched_nodes']
        assert 'done' not in receipt['dispatched_nodes']
        eventually(lambda: calls == ['done', 'qc'])


def test_restarting_same_plan_does_not_pause_a_live_scheduler_node(factory, tmp_path):
    inputs = tmp_path / 'input'; work = tmp_path / 'work'
    inputs.mkdir()
    rt = factory([node('compute', tool='run_cdft', args={
        'action': 'pipeline', 'cif_dir': str(inputs), 'gas': 'CO2',
        'temperature': 298, 'job_work_dir': str(work),
    }, outputs=[str(work / 'results.csv')])], cid='same-live-plan')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'compute')
    rt.store.update(rt.workflow_id, 'compute', ticket['token'], {
        'status': 'waiting_jobs', 'job_ids': ['42'], 'tool_returned': True,
    })
    rt.store.pause(rt.workflow_id, 'stale cache check', kind='input_changed')

    receipt = rt.start(1)

    assert receipt['status'] == 'already_active'
    assert rt.snapshot()['status'] == 'active'
    assert rt.snapshot()['nodes']['compute']['job_ids'] == ['42']


def test_tick_clears_stale_cache_pause_while_node_is_in_prefinish(factory):
    rt = factory([node('inspect')], cid='prefinish-cache-pause')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'inspect')
    rt.store.update(rt.workflow_id, 'inspect', ticket['token'], {'status': 'prefinish'})
    rt.store.pause(rt.workflow_id, 'stale cache check', kind='input_changed')

    rt.tick()

    assert rt.snapshot()['status'] == 'active'
    assert rt.snapshot()['nodes']['inspect']['status'] == 'prefinish'


def test_tick_backfills_dossier_for_existing_prefinish_node(factory, tmp_path):
    from tests.fixture_native_dossier import prepare_native_result,write_native_result
    work=tmp_path/'runs/tester/prefinish-dossier-backfill/native'
    cif=work/'framework.cif';output=work/'Output/System_0/existing-native-output'
    rt = factory([node('calc', tool='run_henry', args={
        'cif': str(cif), 'gas': 'Kr', 'temperature': 298,
    }, outputs=[str(output)])], cid='prefinish-dossier-backfill')
    actual_cif,actual_output=prepare_native_result(work,filename='existing-native-output')
    assert (actual_cif,actual_output)==(cif,output)
    write_native_result(output,1.25)
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'calc')
    receipt = {'submitted': True, 'job_id': '499', 'work_dir': str(work)}
    manifest = {'calculation_dir': str(work), 'result_dir': str(work),
                'input_paths': [str(cif),str(work/'simulation.input')], 'expected_outputs': [str(output)],
                'output_files': [str(output)], 'checkpoint_path': None,
                'evidence_path': None, 'updated_at': time.time()}
    rt.store.update(rt.workflow_id, 'calc', ticket['token'], {
        'status': 'prefinish', 'phase': 'prefinish', 'job_ids': ['499'],
        'result': receipt, 'path_manifest': manifest,
        'recovery_key': 'owned-recovery',
        'validation_dossier': {'verdict': 'pass'},  # older compact schema
    })
    recovery_key = rt.snapshot()['nodes']['calc'].get('recovery_key')
    rt.main._pending_user_interaction = {'tool': 'user_decision', 'params': {
        'decision_id': 'stale-internal-question', 'related_nodes': ['calc'],
        'recovery_key': recovery_key,
    }}
    rt.main._waiting_for_user_input = True

    rt.tick(paused=True)

    current = rt.snapshot()['nodes']['calc']
    assert current['status'] == 'prefinish'
    assert current['validation_dossier']['verdict'] == 'pass'
    assert current['validation_dossier']['job_ids'] == ['499']
    assert rt.main._pending_user_interaction is None
    assert rt.main._waiting_for_user_input is False
    dossier_events = [event for event in rt.snapshot()['outbox'].values()
                      if event['kind'] == 'worker_validation_dossier_ready']
    assert len(dossier_events) == 1
    assert dossier_events[0]['payload']['validation_dossier']['evidence_call_id'] == \
        current['validation_dossier']['evidence_call_id']
    rt.main._pending_user_interaction = {'tool': 'user_decision', 'params': {
        'decision_id': 'stale-after-schema-upgrade', 'related_nodes': ['calc'],
        'recovery_key': 'owned-recovery',
    }}
    rt.main._waiting_for_user_input = True
    rt.tick(paused=True)
    assert rt.main._pending_user_interaction is None


def test_goal_metadata_reconciles_but_science_does_not(factory):
    rt = factory([node('a')])
    rt.start(1)
    rt.main.goal_contract.version += 1
    assert rt.store.reconcile_goal(rt.workflow_id, rt.main.goal_contract.to_dict())
    rt.main.goal_contract.parameters['temperature_K'] = 999
    assert not rt.store.reconcile_goal(rt.workflow_id, rt.main.goal_contract.to_dict())


def test_resource_review_transient_failures_retry_without_submission_or_user_veto(factory, tmp_path):
    work = tmp_path / 'resource-work'
    work.mkdir()
    rt = factory([node('compute', tool='run_cdft', args={
        'action': 'pipeline', 'cif_dir': str(tmp_path / 'inputs'), 'gas': 'Kr',
        'temperature': 298, 'job_work_dir': str(work)}, outputs=[str(work / 'result.csv')])])
    rt.start(1)
    rt.main._on_resource_review = lambda request: {'status': 'review_retry', 'reason': 'temporary provider failure'}
    rt.main.registry.get('run_cdft').execute = lambda args: pytest.fail('must not submit without verified resources')
    for attempt in range(1, 4):
        rt.tick()
        eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'needs_resources')
        for future in list(rt.futures.values()): future.result(timeout=5)
        current = rt.snapshot()['nodes']['compute']
        assert current['resource_review_attempts'] == attempt
        assert not rt.snapshot()['user_paused']
        assert not rt.store.snapshot().get('leases')
        if attempt < 3:
            assert current['resource_retry_at']
            rt.store.update(rt.workflow_id, 'compute', current['token'], {'resource_retry_at': time.time() - 1})
        else:
            assert current['resource_retry_at'] is None
    events = list(rt.snapshot()['outbox'].values())
    reviews = [event for event in events if event['kind'] == 'worker_resource_review_required']
    assert len(reviews) == 3
    assert all(not event['payload']['requires_user'] for event in reviews)
    assert reviews[-1]['payload']['automatic_retry'] is False


def exhausted_resource_review(factory, tmp_path, fingerprint='old-evaluator'):
    work = tmp_path / 'resource-work'
    work.mkdir()
    contract = node('compute', tool='run_cdft', args={'action': 'pipeline',
        'cif_dir': str(tmp_path / 'inputs'), 'gas': 'Kr', 'temperature': 298,
        'job_work_dir': str(work)}, outputs=[str(work / 'result.csv')])
    rt = factory([contract])
    rt.start(1)
    claim = rt.store.claim(rt.workflow_id, 'compute')
    receipt = {'status': 'review_retry', 'retryable': True, 'reason': 'fixture provider unavailable'}
    rt.store.update(rt.workflow_id, 'compute', claim['token'], {
        'status': 'needs_resources', 'resource_review': receipt,
        'resource_review_attempts': 3, 'resource_review_fingerprint': fingerprint,
        'resource_retry_at': None, 'resource_review_history': [
            {'review_id': f'prior{i}', 'attempt': i, 'fingerprint': fingerprint, 'receipt': receipt}
            for i in range(1, 4)]}, release=True)
    rt.store.pause(rt.workflow_id, 'resource review exhausted', kind='resource_review', step_id='compute')
    return rt


@pytest.mark.parametrize('exhausted', [False, True])
def test_resource_review_busy_wait_does_not_consume_evaluation_or_source_recovery(factory, tmp_path, monkeypatch, exhausted):
    rt = exhausted_resource_review(factory, tmp_path) if exhausted else factory([
        node('compute', tool='run_henry', args={'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298},
             outputs=[str(tmp_path / 'output')])])
    if not exhausted:
        rt.start(1)
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: 'fixed-evaluator')
    reviews, science = [], []
    busy = {'status': 'review_busy', 'review_in_progress': True, 'retryable': True, 'retry_after_seconds': 5}
    ready = {'status': 'ready', 'resource_review_id': 'actual-ready-review', 'resource_allocations': {}}
    rt.main._on_resource_review = lambda request: reviews.append(request) or (busy if len(reviews) <= 2 else ready)
    tool = rt.snapshot()['nodes']['compute']['contract']['tool']
    rt.main.registry.get(tool).execute = lambda args: science.append(args) or {'submitted': True, 'job_id': 'busy-safe-job'}
    expected = 3 if exhausted else 0
    for index in range(2):
        started = time.time()
        rt.tick()
        eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'needs_resources')
        current = rt.snapshot()['nodes']['compute']
        assert len(reviews) == index + 1 and science == []
        assert current.get('resource_review_attempts', 0) == expected
        assert len(current.get('resource_review_history', [])) == expected
        assert current['resource_review_waits'] == index + 1
        assert current['resource_retry_at'] >= started + 4
        assert current['resource_retry_at'] < time.time() + 61
        assert current.get('resource_review_epoch_pending') == ('fixed-evaluator' if exhausted else None)
        assert not any(lease['workflow_id'] == rt.workflow_id for lease in rt.store.snapshot().get('leases', {}).values())
        for future in rt.futures.values(): future.result(timeout=5)
        rt.store.update(rt.workflow_id, 'compute', current['token'], {'resource_retry_at': time.time() - 1})
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'waiting_jobs')
    assert len(reviews) == 3 and len(science) == 1
    current = rt.snapshot()['nodes']['compute']
    assert current['resource_review_attempts'] == expected + 1
    assert len(current['resource_review_history']) == expected + 1
    assert current.get('resource_review_epoch_pending') is None


@pytest.mark.parametrize('legacy', [False, True])
def test_new_evaluator_retries_exhausted_preflight_once_and_still_requires_ready(factory, tmp_path, monkeypatch, legacy):
    rt = exhausted_resource_review(factory, tmp_path, None if legacy else 'old-evaluator')
    before = rt.snapshot()['nodes']['compute']
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: 'fixed-evaluator', raising=False)
    reviews, dispatched = [], []
    def review(request):
        reviews.append(request)
        return {'status': 'ready', 'resource_review_id': 'new-verified-review',
                'resource_allocations': {'compute': {'cpus': 1, 'memory_mb': 512}}}
    rt.main._on_resource_review = review
    rt.main.registry.get('run_cdft').execute = lambda args: dispatched.append(args) or {'submitted': True, 'job_id': '70301'}
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'waiting_jobs')
    for future in list(rt.futures.values()):future.result(timeout=5)
    for _ in range(3):rt.tick()
    current = rt.snapshot()['nodes']['compute']
    assert len(reviews) == 1 and len(dispatched) == 1
    assert current['contract'] == before['contract']
    assert current['resource_review_attempts'] == 4
    assert current['resource_review_history'][:3] == before['resource_review_history']
    assert current['resource_review_history'][-1]['receipt']['status'] == 'ready'
    assert current['resource_review_fingerprint'] == 'fixed-evaluator'
    assert current['job_ids'] == ['70301']
    ledger = RecoveryGate(path=rt.root / 'recovery_state.json').snapshot()
    assert len(ledger) == 1 and next(iter(ledger.values()))['attempts'] == 1


def test_resource_evaluator_source_epochs_are_bounded_without_resetting_history(factory, tmp_path, monkeypatch):
    rt = exhausted_resource_review(factory, tmp_path)
    source = ['old-evaluator']
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: source[0], raising=False)
    reviews = []
    rt.main._on_resource_review = lambda req: reviews.append(req) or {
        'status': 'review_retry', 'retryable': True, 'reason': 'still unavailable'}
    rt.main.registry.get('run_cdft').execute = lambda args: pytest.fail('unreviewed scientific tool must not execute')
    rt.tick(); rt.tick()
    assert reviews == []
    for index, value in enumerate(['new-evaluator', 'newer-evaluator'], 4):
        source[0] = value
        rt.tick()
        eventually(lambda: rt.snapshot()['nodes']['compute']['resource_review_attempts'] == index)
        for future in list(rt.futures.values()):future.result(timeout=5)
        for _ in range(3):rt.tick()
        assert len(reviews) == index - 3
        assert rt.snapshot()['nodes']['compute']['resource_retry_at'] is None
        assert not rt.store.snapshot().get('leases')
    assert len(rt.snapshot()['nodes']['compute']['resource_review_history']) == 5


@pytest.mark.parametrize('filename', ['resource_policy.py', 'resource_review.py', 'node_inventory.py', 'slurm.py'])
def test_resource_evaluator_fingerprint_includes_native_review_sources(factory, monkeypatch, filename):
    from agents.parallel_workflow import resource_evaluator_fingerprint
    rt = factory([node('a')])
    before = resource_evaluator_fingerprint(rt.main)
    original = Path.read_bytes
    monkeypatch.setattr(Path, 'read_bytes', lambda path: original(path) + (b'\n# source fix' if path.name == filename else b''))
    assert resource_evaluator_fingerprint(rt.main) != before


def test_resource_evaluator_fingerprint_tracks_selected_policy_not_live_inventory(factory, tmp_path):
    from agents.parallel_workflow import resource_evaluator_fingerprint
    rt = factory([node('a')])
    env = tmp_path / 'env'
    env.mkdir()
    policy = env / 'node_policy.json'
    policy.write_text('{"eligible_nodes":["node05"]}')
    before = resource_evaluator_fingerprint(rt.main)
    (env / 'node_inventory.json').write_text('{"observed_at":123}')
    assert resource_evaluator_fingerprint(rt.main) == before
    policy.write_text('{"eligible_nodes":["node04","node05"]}')
    assert resource_evaluator_fingerprint(rt.main) != before


@pytest.mark.parametrize('blocker', ['paused', 'user_paused', 'pending_patch', 'awaiting_approval',
    'unauthorized', 'scope_changed', 'job_ids', 'attempt_id', 'recovery_key', 'entered_tool',
    'tool_returned', 'lease', 'nonretryable', 'uncertain'])
def test_exhausted_resource_review_recovery_preserves_dispatch_safety(factory, tmp_path, monkeypatch, blocker):
    rt = exhausted_resource_review(factory, tmp_path)
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: 'fixed-evaluator', raising=False)
    reviews = []
    rt.main._on_resource_review = lambda req: reviews.append(req) or {'status': 'ready'}
    if blocker == 'pending_patch':rt.main._pending_workflow_patch = {'affected_nodes': ['compute']}
    if blocker == 'awaiting_approval':rt.main._awaiting_plan_approval = True
    if blocker == 'unauthorized':rt.main.goal_contract.execution_authorized = False
    if blocker == 'scope_changed':rt.main.goal_contract.parameters['temperature_K'] = 310
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id];current = workflow['nodes']['compute']
        if blocker == 'user_paused':workflow['user_paused'] = True
        if blocker == 'job_ids':current['job_ids'] = ['70301']
        if blocker == 'attempt_id':current['attempt_id'] = 'actual-dispatch'
        if blocker == 'recovery_key':current['recovery_key'] = 'unresolved-dispatch'
        if blocker == 'entered_tool':current['dispatch_phase'] = 'entered_tool'
        if blocker == 'tool_returned':current['tool_returned'] = True
        if blocker == 'nonretryable':current['resource_review']['retryable'] = False
        if blocker == 'uncertain':current['status'] = 'uncertain'
        if blocker == 'lease':data.setdefault('leases', {})[current['token']] = {
            'workflow_id': rt.workflow_id, 'step_id': 'compute', 'resources': current['contract']['resources']}
    rt.tick(paused=blocker == 'paused')
    assert reviews == []
    assert rt.snapshot()['nodes']['compute']['resource_review_attempts'] == 3


def test_changing_output_path_does_not_clear_resource_review_budget(factory, tmp_path, monkeypatch):
    rt = exhausted_resource_review(factory, tmp_path)
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: 'old-evaluator', raising=False)
    reviews = []
    rt.main._on_resource_review = lambda req: reviews.append(req) or {
        'status': 'review_retry', 'retryable': True, 'reason': 'same old evaluator'}
    revised = copy.deepcopy(rt.main.goal_contract.approved_nodes[0])
    revised['arguments']['job_work_dir'] = str(tmp_path / 'new-resource-work')
    revised['expected_outputs'] = [str(tmp_path / 'new-resource-work/result.csv')]
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'compute', 'node': revised}], 1, 2, 'authorized location edit')
    rt.main.goal_contract.approved_nodes = [revised]
    rt.main.goal_contract.approved_plan_version = 2
    rt.start(2)
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'needs_resources')
    assert reviews == []
    assert rt.snapshot()['nodes']['compute']['resource_review_attempts'] == 3


def test_pause_arriving_during_recovered_review_prevents_scientific_dispatch(factory, tmp_path, monkeypatch):
    rt = exhausted_resource_review(factory, tmp_path)
    monkeypatch.setattr('agents.parallel_workflow.resource_evaluator_fingerprint', lambda main: 'fixed-evaluator')
    dispatched = []
    def review(request):
        rt.store.user_pause(rt.workflow_id)
        return {'status': 'ready', 'resource_review_id': 'new-ready-review',
                'resource_allocations': {'compute': {'cpus': 1, 'memory_mb': 512}}}
    rt.main._on_resource_review = review
    rt.main.registry.get('run_cdft').execute = lambda args: dispatched.append(args) or {'submitted': True, 'job_id': '70301'}
    rt.tick()
    for future in list(rt.futures.values()):future.result(timeout=5)
    current = rt.snapshot()['nodes']['compute']
    assert rt.snapshot()['user_paused'] is True
    assert current['status'] == 'pending'
    assert current['resource_review']['status'] == 'ready'
    assert dispatched == []
    assert RecoveryGate(path=rt.root / 'recovery_state.json').snapshot() == {}
    assert not rt.store.snapshot().get('leases')


@pytest.mark.parametrize('saved_version,refreshes', [(2, 1), (3, 0)])
def test_prefinish_dossier_refresh_uses_current_native_schema_without_overwriting_old_evidence(
        factory, tmp_path, monkeypatch, saved_version, refreshes):
    from agents import result_validation
    monkeypatch.setattr(result_validation, 'VALIDATION_SCHEMA_VERSION', 3)
    rt = factory([node('calc', tool='run_henry', args={'cif': str(tmp_path / 'fixture.cif'), 'gas': 'Kr', 'temperature': 298},
                       outputs=[str(tmp_path / 'result.dat')])])
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'calc')
    rt.root.joinpath('evidence').mkdir(exist_ok=True)
    old = rt.root / 'evidence' / 'old-dossier.json'
    old.write_text('immutable earlier validation evidence')
    rt.store.update(rt.workflow_id, 'calc', ticket['token'], {
        'status': 'prefinish', 'phase': 'prefinish', 'job_ids': ['70999'],
        'validation_dossier': {'schema_version': saved_version, 'successful_artifacts': [],
            'evidence_call_id': 'old-dossier', 'evidence_path': str(old)}})
    calls = []
    def rebuild(step_id, current, manifest, result):
        calls.append(step_id)
        fresh = rt.root / 'evidence' / 'new-dossier.json'
        fresh.write_text('new version validation evidence')
        return {'schema_version': 3, 'successful_artifacts': [],
                'evidence_call_id': 'new-dossier', 'evidence_path': str(fresh)}
    monkeypatch.setattr(rt, '_create_validation_dossier', rebuild)
    rt.tick()
    assert len(calls) == refreshes
    assert rt.snapshot()['nodes']['calc']['validation_dossier']['schema_version'] == 3
    assert old.read_text() == 'immutable earlier validation evidence'


def test_reviewed_resources_reach_tool_and_failed_dispatch_releases_taskline_lease(factory, tmp_path):
    output = tmp_path / 'generated'
    contract = node('generate_cof', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 2, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 2}])
    contract['agent'] = 'harness'
    rt = factory([contract])
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-1',
        'resource_allocations': {'generate_cof': {
            'memory_mb': 8192, 'cpus': 4, 'nodelist': 'node03', 'partition': 'debug',
        }},
    }
    calls = []
    rt.main.registry.get('generate_structure').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled generator failure after dispatch entry',
    }
    rt.start(1)
    rt.tick()
    eventually(lambda: bool(calls))
    eventually(lambda: rt.snapshot()['nodes']['generate_cof']['status'] == 'failed')
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    assert calls[0]['memory_mb'] == 8192
    assert calls[0]['cpus_per_task'] == 4
    line_node = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert line_node['resource_allocation']['nodelist'] == 'node03'
    assert line_node['effective_arguments']['partition'] == 'debug'
    assert line_node['status'] == 'failed'
    assert line_node['validation']['runtime_token'] is None
    assert line_node['validation']['resource_leases'] == []
    assert not rt.store.snapshot().get('leases')


def test_retry_reuses_persisted_resource_review_without_calling_model_again(factory, tmp_path):
    work = tmp_path / 'reviewed-retry'
    inputs = tmp_path / 'charged'
    inputs.mkdir()
    contract = node('compute', tool='run_cdft', args={
        'action': 'pipeline', 'cif_dir': str(inputs), 'gas': 'CO2',
        'temperature': 298, 'job_work_dir': str(work),
    }, outputs=[str(work / 'result.csv')])
    rt = factory([contract], cid='reviewed-retry')
    rt.start(1)
    state_path = rt.store.path
    with json_transaction(state_path) as data:
        current = data['workflows'][rt.workflow_id]['nodes']['compute']
        current['resource_review'] = {
            'status': 'ready', 'resource_review_id': 'review-existing',
            'evidence_call_ids': ['resource-proof'],
        }
        current['resource_allocation'] = {
            'resource_review_id': 'review-existing', 'memory_mb': 8192,
            'cpus': 8, 'nodelist': 'node15', 'partition': 'compute',
            'evidence_call_ids': ['resource-proof'],
        }
        current['effective_arguments'] = {
            **current['contract']['arguments'], 'resource_review_id': 'review-existing',
            'memory_mb': 8192, 'num_processes': 8,
            'nodelist': 'node15', 'partition': 'compute',
        }
    rt.main._on_resource_review = lambda request: pytest.fail('a verified retry must not call the model reviewer again')
    calls = []
    rt.main.registry.get('run_cdft').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled failure after proving dispatch arguments',
    }

    rt.tick()
    eventually(lambda: bool(calls))
    for future in list(rt.futures.values()):
        future.result(timeout=5)

    assert calls[0]['resource_review_id'] == 'review-existing'
    assert calls[0]['memory_mb'] == 8192
    assert calls[0]['num_processes'] == 8


def test_provisional_resource_receipt_is_never_reused_for_dispatch():
    from agents.parallel_workflow import persisted_resource_review
    state = {'contract': {'tool': 'run_cdft', 'arguments': {'action': 'submit'}},
             'effective_arguments': {'action': 'submit', 'resource_review_id': 'preview'},
             'resource_allocation': {'resource_review_id': 'preview', 'budget_final': False,
                                     'evidence_call_ids': ['planning-evidence']}}
    assert persisted_resource_review(state, 'compute') is None


@pytest.mark.parametrize('final', [True, False])
def test_dispatch_budget_uses_resolved_inputs_and_final_allocation(factory, tmp_path, final):
    approved = tmp_path / 'prepared'
    actual = approved / 'timestamp'
    actual.mkdir(parents=True)
    (actual / 'a.dat').write_text('prepared input')
    work = tmp_path / 'batch'
    rt = factory([
        node('prepare', tool='run_cdft', args={'action': 'inputs', 'gas': 'CO2', 'temperature': 298,
             'cif_dir': str(tmp_path / 'charged'), 'input_dir': str(approved)},
             outputs=[{'kind': 'directory', 'path': str(actual), 'pattern': '*.dat', 'min_count': 1}]),
        node('compute', tool='run_cdft', args={'action': 'submit', 'input_dir': str(approved),
             'job_work_dir': str(work), 'gas': 'CO2', 'temperature': 298,
             'num_processes': 1, 'memory_mb': 16384}, deps=['prepare'],
             outputs=[str(work / 'results.csv')]),
    ], cid='resolved-budget')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'prepare')
    rt.store.update(rt.workflow_id, 'prepare', ticket['token'], {
        'status': 'succeeded', 'phase': 'finish', 'result': {'input_dir': str(actual)}}, release=True)
    reviews, calls = [], []
    def review(request):
        reviews.append(copy.deepcopy(request))
        return {'status': 'ready', 'resource_review_id': 'final-review',
                'resource_allocations': {'compute': {'budget_final': final,
                    'cpus': 8, 'memory_mb': 8704, 'nodelist': 'node25', 'partition': 'long'}}}
    rt.main._on_resource_review = review
    rt.main.registry.get('run_cdft').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled failure after verifying resource propagation'}
    rt.tick()
    eventually(lambda: bool(reviews))
    eventually(lambda: rt.snapshot()['nodes']['compute']['status'] in {'failed', 'needs_resources'})
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    assert reviews[0]['review_phase'] == 'dispatch'
    assert reviews[0]['nodes'][0]['arguments']['input_dir'] == str(actual)
    if final:
        assert len(calls) == 1
        assert calls[0]['input_dir'] == str(actual)
        assert calls[0]['num_processes'] == 8
        assert calls[0]['memory_mb'] == 8704
        assert calls[0]['nodelist'] == 'node25'
    else:
        assert not calls
        assert rt.snapshot()['nodes']['compute']['status'] == 'needs_resources'


@pytest.mark.parametrize('memory_mb,cpus', [(8192, 2), (4096, 4), (8192, 4)])
@pytest.mark.parametrize('stalled_preclaim', [False, True])
def test_verified_retry_dispatch_uses_new_resources_instead_of_old_receipt(factory, tmp_path, memory_mb, cpus, stalled_preclaim):
    """Two real worker dispatches, with fake tool failure and native retry ledger."""
    from agents.recovery import RecoveryGate, input_fingerprint
    output = tmp_path / 'runs/tester/resource-repair/cdft'
    contract = node('generate', tool='run_cdft', args={
        'action': 'pipeline', 'gas': 'CO2', 'temperature': 298,
        'cif_dir': str(tmp_path / 'runs/tester/resource-repair/inputs'),
        'job_work_dir': str(output)}, outputs=[str(output / 'result.csv')])
    rt = factory([contract], cid='resource-repair')
    reviews, dispatches = [], []

    def review(request):
        reviews.append(copy.deepcopy(request))
        index = len(reviews)
        assert index <= 2, 'the verified retry must need exactly one fresh resource review'
        return {'status': 'ready', 'resource_review_id': f'review-{index}',
                'resource_allocations': {'generate': {
                    'resource_review_id': f'review-{index}', 'evidence_call_ids': [f'proof-{index}'],
                    'memory_mb': 4096 if index == 1 else memory_mb,
                    'cpus': 2 if index == 1 else cpus,
                    'nodelist': 'node03', 'partition': 'debug'}}}

    def dispatch(arguments):
        dispatches.append(copy.deepcopy(arguments))
        return {'error': 'controlled failure after worker entered the scientific adapter'}

    rt.main.registry.get('run_cdft').execute = dispatch
    rt.start(1)
    rt.main._on_resource_review = review
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['generate']['status'] == 'failed')
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    failed = rt.snapshot()['nodes']['generate']
    with json_transaction(rt.store.path) as data:
        data['workflows'][rt.workflow_id]['nodes']['generate']['configuration_recovery'] = {
            'kind': 'verified_configuration_preflight', 'attempt_id': 'older-preflight',
            'no_job_submitted': True}
    assert failed['resource_allocation']['evidence_call_ids'] == ['proof-1']
    from agents.parallel_workflow import persisted_resource_review
    assert persisted_resource_review(failed, 'generate')['resource_review_id'] == 'review-1'
    assert dispatches[0]['memory_mb'] == 4096 and dispatches[0]['num_processes'] == 2
    corrected = {**dispatches[0], 'memory_mb': memory_mb, 'num_processes': cpus}
    gate = RecoveryGate(path=rt.root / 'recovery_state.json')
    gate.approve(failed['recovery_key'], input_fingerprint(corrected, tmp_path),
                 {'fix_summary': 'Correct the reviewed CPU and memory budget'}, arguments=corrected)
    if stalled_preclaim:
        # Persist the exact old-runtime failure after requeue but before claim.
        # No scheduler attempt or ledger claim was made by that failed worker.
        with json_transaction(rt.store.path) as data:
            current = data['workflows'][rt.workflow_id]['nodes']['generate']
            current.setdefault('attempt_history', []).append(copy.deepcopy(failed))
            current.update(attempt_id=None, dispatch_phase='preflight', tool_returned=False,
                           job_ids=[], verified_retry={'attempt_id': failed['attempt_id']},
                           error='configuration recovery requires an unexecuted generator preflight failure')

    rt.tick()
    eventually(lambda: len(dispatches) == 2 or rt.snapshot()['nodes']['generate']['status'] == 'failed')
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    assert len(reviews) == 2, rt.snapshot()['nodes']['generate']
    assert len(dispatches) == 2, rt.snapshot()['nodes']['generate']
    assert dispatches[1]['memory_mb'] == memory_mb
    assert dispatches[1]['num_processes'] == cpus
    assert dispatches[1]['resource_review_id'] == 'review-2'
    assert gate.snapshot()[failed['recovery_key']]['attempts'] == 2
    assert rt.snapshot()['nodes']['generate']['attempt_history'][0]['attempt_id'] == failed['attempt_id']
    assert 'configuration_recovery' not in rt.snapshot()['nodes']['generate']
    rt.tick()
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    assert len(dispatches) == 2, 'the consumed retry permit must never submit a third attempt'


def test_resource_projection_failure_does_not_block_approved_tool(factory, tmp_path):
    output = tmp_path / 'generated-projection-fallback'
    contract = node('generate_cof', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 1, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 1}])
    contract['agent'] = 'harness'
    rt = factory([contract], cid='projection-fallback')
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-fallback',
        'resource_allocations': {'generate_cof': {
            'memory_mb': 4096, 'cpus': 2, 'nodelist': 'node03', 'partition': 'debug',
        }},
    }
    original = rt.task_lines.upsert_step
    def projection_write(*args, **kwargs):
        if 'resource_allocation' in kwargs:
            raise TypeError('simulated older TaskLine projection schema')
        return original(*args, **kwargs)
    rt.task_lines.upsert_step = projection_write
    calls = []
    rt.main.registry.get('generate_structure').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled failure proves tool entry',
    }
    rt.start(1)
    rt.tick()
    eventually(lambda: bool(calls))
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    state = rt.snapshot()['nodes']['generate_cof']
    assert state['resource_allocation']['memory_mb'] == 4096
    assert 'older TaskLine projection schema' in state['metadata_sync_error']
    assert calls[0]['memory_mb'] == 4096 and calls[0]['cpus_per_task'] == 2


def node(key, path=None, deps=(), locks=(), tool='read_file', args=None, outputs=()):
    return {'step_id': key, 'agent': 'adsorption' if tool == 'run_henry' else 'analyst', 'tool': tool,
            'arguments': args or {'path': path or key}, 'depends_on': list(deps),
            'expected_outputs': list(outputs), 'resource_locks': list(locks)}


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.01)
    assert predicate(), 'condition did not become true before deadline'


def verify_prefinish(rt,step_id):
    """Unit proof fixture; real model decisions are covered by SDK handoff."""
    state=rt.snapshot()['nodes'][step_id]
    assert state['status']=='prefinish' and state.get('phase')=='prefinish'
    paths=list(state.get('artifacts',{}))
    assert paths and all(state['artifacts'].values())
    call_id='verification'+''.join(character for character in step_id if character.isalnum())
    write_checkpoint(rt.root/'evidence'/f'{call_id}.json',{'call_id':call_id,'time':time.time(),'tool':'read_file',
        'params':{'path':paths[0]},'result':'{"ok":true,"content":"actual verification fixture"}','failed':False})
    rt.finish_node(step_id,[call_id],'Test verifier checked fresh native output against the original contract.')


@pytest.fixture
def factory(tmp_path):
    runtimes = []
    def build(nodes, cid='c1', username='tester'):
        root = tmp_path / 'runs' / username / cid
        root.mkdir(parents=True)
        main = Session(config=AgentConfig(api_key='test', project_root=tmp_path), registry=_build_default_registry())
        main._current_line_id = cid
        main.goal_contract = GoalContract.from_user_message('hello')
        main.goal_contract.execution_authorized = True
        main.goal_contract.approved_plan_version = 1
        main.goal_contract.approved_nodes = copy.deepcopy(nodes)
        lines = TaskLineStore(str(tmp_path / (cid + '-lines.json')))
        line = lines.begin_line(cid, username=username, conv_id=cid)
        main._current_line_id = line['line_id']
        lines.apply_workflow_patch(line['line_id'], [{'operation': 'upsert', 'step_id': n['step_id'], 'node': n} for n in nodes], 0, 1, 'test approved DAG')
        mailbox = LifecycleStore(root / 'lifecycle.json')
        mailbox.initialize()
        runtime = ParallelWorkflow(main, root, WorkflowStore(tmp_path / 'data/state/parallel_workflows.json'), mailbox, lines, username, cid)
        runtime.main._runtime_snapshot = runtime.snapshot
        runtimes.append(runtime)
        return runtime
    yield build
    for runtime in runtimes:
        for future in runtime.futures.values():
            if not future.done(): future.result(timeout=6)
        runtime.shutdown()


def approved_activation_upgrade(factory, tmp_path, native_floor=False, missing_binding=False):
    """Persisted completed receipts plus one formally approved pending edit."""
    root = tmp_path / 'runs/tester/activation-upgrade'
    generated, charged = root / 'generated', root / 'charged'
    generate = node('generate', tool='generate_structure', args={'material_type': 'MOF',
        'n_structures': 1, 'output_dir': str(generated)}, outputs=[{
            'kind': 'directory', 'path': str(generated), 'pattern': '*.cif', 'min_count': 1}])
    generate['agent'] = 'harness'
    charge = node('charge', tool='run_pacman_charge', args={'cif_dir': str(generated),
        'method': 'pacmof', 'output_dir': str(charged)}, deps=['generate'], outputs=[{
            'kind': 'directory', 'path': str(charged), 'pattern': '*.cif',
            'min_count': 30 if native_floor else 1}])
    charge['input_bindings'] = {'cif_dir': {'step_id': 'generate', 'output_index': 0}}
    rt = factory([generate, charge, node('rank', path='old-rank-input', deps=['charge'])], cid='activation-upgrade')
    rt.start(1)
    for directory in (generated, charged):
        directory.mkdir()
        (directory / 'fixture.cif').write_text('verified fixture artifact\n')
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id]
        for step_id, job_id in [('generate', '501'), ('charge', '502')]:
            current = workflow['nodes'][step_id]
            if step_id == 'charge' and native_floor:
                current['contract']['expected_outputs'][0]['min_count'] = 1
                current['batch_success_policy'] = {'mode': 'best_effort', 'decision_owner': 'responsible_agent',
                    'applied_once': True, 'outputs': [{'index': 0, 'declared_target_count': 30,
                                                       'artifact_presence_floor': 1}]}
            current.update(status='succeeded', phase='finish', token=step_id + '-token',
                job_ids=[job_id], attempt_id=step_id + '-attempt', node_verification={'passed': True},
                effective_arguments=copy.deepcopy(current['contract']['arguments']),
                result={'output_dir': str(generated if step_id == 'generate' else charged)},
                execution_fingerprint='earlier-runtime-source')
            current['artifacts'] = rt._artifacts(current['contract'], current['result'])
        if missing_binding:
            workflow['nodes']['charge']['contract'].pop('input_bindings')
        workflow.update(status='needs_user', pause_kind='failure', pause_reason='old activation failed')
    rt.job_watch = SimpleNamespace(list=lambda: [], get=lambda _: {
        'username': 'tester', 'conv_id': rt.conv_id, 'terminal': True,
        'failed': False, 'state': 'COMPLETED'})
    revised = node('rank', path='approved-new-rank-input', deps=['charge'])
    operations = [{'operation': 'upsert', 'step_id': 'rank', 'node': revised}]
    if native_floor:
        charge['expected_outputs'][0]['min_count'] = 1
        operations.append({'operation': 'upsert', 'step_id': 'charge', 'node': charge})
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id, operations, 1, 2, 'approved pending repair')
    rt.main.goal_contract.approved_plan_version = 2
    rt.main.goal_contract.approved_nodes = [generate, charge, revised]
    rt.main.registry.get('generate_structure').execute = lambda _: pytest.fail('completed generation must not resubmit')
    rt.main.registry.get('run_pacman_charge').execute = lambda _: pytest.fail('completed charge must not resubmit')
    return rt


@pytest.mark.parametrize('native_floor,missing_binding', [(False, False), (True, False), (True, True)])
def test_tick_recovers_persisted_approved_upgrade_after_restart_once(factory, tmp_path, native_floor, missing_binding):
    rt = approved_activation_upgrade(factory, tmp_path, native_floor, missing_binding)
    before = rt.snapshot()
    invoked = []
    rt.main.registry.get('read_file').execute = lambda args: invoked.append(args['path']) or {'ok': True}
    # A new executor instance has no ephemeral memory of the failed activation.
    restarted = ParallelWorkflow(rt.main, rt.root, rt.store, rt.lifecycle, rt.task_lines,
                                 rt.username, rt.conv_id, job_watch=rt.job_watch)
    try:
        restarted.tick()
        eventually(lambda: restarted.snapshot()['nodes']['rank']['status'] == 'succeeded')
        for _ in range(3): restarted.tick()
        after = restarted.snapshot()
        assert after['plan_version'] == 2
        assert invoked == ['approved-new-rank-input']
        for step_id in ('generate', 'charge'):
            for field in ('token', 'attempt_id', 'job_ids', 'artifacts', 'node_verification', 'result'):
                assert after['nodes'][step_id][field] == before['nodes'][step_id][field]
        attempts = restarted.store.snapshot()['activation_recoveries'][rt.workflow_id]
        assert len(attempts) == 1
        assert next(iter(attempts.values()))['status'] == 'activated'
    finally:
        restarted.pool.shutdown(wait=True)


@pytest.mark.parametrize('blocker', ['paused_argument', 'user_paused', 'pending_patch', 'awaiting_approval',
                                   'unauthorized', 'scope_changed', 'goal_version', 'runtime_owner',
                                   'line_owner', 'line_version', 'line_contract', 'uncertain',
                                   'goal_scope_change', 'missing_floor_proof'])
def test_approved_activation_recovery_respects_durable_authority(factory, tmp_path, monkeypatch, blocker):
    rt = approved_activation_upgrade(factory, tmp_path, native_floor=blocker == 'missing_floor_proof')
    if blocker == 'pending_patch': rt.main._pending_workflow_patch = {'affected_nodes': ['rank']}
    if blocker == 'awaiting_approval': rt.main._awaiting_plan_approval = True
    if blocker == 'unauthorized': rt.main.goal_contract.execution_authorized = False
    if blocker == 'scope_changed': rt.main.goal_contract.parameters['temperature'] = 301
    if blocker == 'goal_version': rt.main.goal_contract.version += 1
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id]
        if blocker == 'user_paused': workflow['user_paused'] = True
        if blocker == 'runtime_owner': workflow['username'] = 'different-user'
        if blocker == 'goal_scope_change': workflow['pause_kind'] = 'goal_scope_change'
        if blocker == 'uncertain': workflow['nodes']['rank']['status'] = 'uncertain'
        if blocker == 'missing_floor_proof': workflow['nodes']['charge'].pop('batch_success_policy')
    if blocker.startswith('line_'):
        with rt.task_lines._process_guard():
            line = rt.task_lines._lines[rt.main._current_line_id]
            if blocker == 'line_owner': line['username'] = 'different-user'
            if blocker == 'line_version': line['plan_version'] = 3
            if blocker == 'line_contract': line['steps'][-1]['arguments']['path'] = 'unapproved-input'
            rt.task_lines._save()
    starts = []
    monkeypatch.setattr(rt, 'start', lambda *args, **kwargs: starts.append(args) or {'status': 'scheduled'})
    rt.tick(paused=blocker == 'paused_argument')
    assert starts == []
    assert rt.snapshot()['plan_version'] == 1


def test_failed_native_activation_is_bounded_across_restart_and_source_change(factory, tmp_path, monkeypatch):
    rt = approved_activation_upgrade(factory, tmp_path)
    source = ['source-v1']
    monkeypatch.setattr('agents.parallel_workflow.activation_framework_fingerprint', lambda: source[0])
    calls = []
    def blocked(*args, **kwargs):
        calls.append(args[0])
        raise ValueError('fixture guard refused activation')
    monkeypatch.setattr(rt, 'start', blocked)
    rt.tick(); rt.tick(); rt.tick()
    assert calls == [2]
    restarted = ParallelWorkflow(rt.main, rt.root, rt.store, rt.lifecycle, rt.task_lines,
                                 rt.username, rt.conv_id, job_watch=rt.job_watch)
    try:
        monkeypatch.setattr(restarted, 'start', blocked)
        restarted.tick()
        assert calls == [2]
        source[0] = 'source-v2'
        restarted.tick(); restarted.tick()
        assert calls == [2, 2]
        attempts = rt.store.snapshot()['activation_recoveries'][rt.workflow_id]
        assert len(attempts) == 2
        assert all(item['status'] == 'blocked' for item in attempts.values())
        events = [e for e in rt.lifecycle.snapshot()['events'].values() if e['kind'] == 'activation_blocked']
        assert len(events) == 2
    finally:
        restarted.shutdown()


def test_native_activation_does_not_clear_a_concurrent_user_pause(factory, tmp_path, monkeypatch):
    rt = approved_activation_upgrade(factory, tmp_path)
    original = rt.store.activate
    def pause_before_commit(*args, **kwargs):
        rt.store.user_pause(rt.workflow_id)
        return original(*args, **kwargs)
    monkeypatch.setattr(rt.store, 'activate', pause_before_commit)
    rt.tick()
    assert rt.snapshot()['user_paused'] is True
    assert rt.snapshot()['plan_version'] == 1
    attempts = rt.store.snapshot()['activation_recoveries'][rt.workflow_id]
    assert len(attempts) == 1
    assert next(iter(attempts.values()))['status'] == 'blocked'


def test_native_activation_keeps_an_owned_running_scheduler_branch(factory, tmp_path):
    rt = approved_activation_upgrade(factory, tmp_path)
    gate = RecoveryGate(path=rt.root / 'recovery_state.json')
    charge = rt.snapshot()['nodes']['charge']
    key = gate.key(charge['contract']['tool'], charge['contract']['arguments'],
                   rt.main.goal_contract.version, node='charge', project_root=tmp_path)
    attempt, denial = gate.claim(key, charge['contract']['tool'], charge['contract']['arguments'],
                                input_fingerprint(charge['contract']['arguments'], tmp_path))
    assert attempt and not denial
    gate.outcome(key, attempt, {'job_id': '502', 'submitted': True})
    with json_transaction(rt.store.path) as data:
        live = data['workflows'][rt.workflow_id]['nodes']['charge']
        live.update(status='waiting_jobs', phase=None, recovery_key=key, attempt_id=attempt,
                    result={'job_id': '502', 'submitted': True}, node_verification=None)
        data.setdefault('leases', {})[live['token']] = {'workflow_id': rt.workflow_id, 'step_id': 'charge',
                                        'resources': live['contract']['resources']}
    rt.job_watch = SimpleNamespace(list=lambda: [], get=lambda job_id: {
        'username': 'tester', 'conv_id': rt.conv_id,
        'terminal': job_id != '502', 'failed': False,
        'state': 'RUNNING' if job_id == '502' else 'COMPLETED'})
    before = rt.snapshot()['nodes']['charge']
    rt.tick(); rt.tick()
    after = rt.snapshot()['nodes']['charge']
    assert rt.snapshot()['plan_version'] == 2
    for field in ('token', 'attempt_id', 'job_ids', 'recovery_key', 'result', 'status'):
        assert after[field] == before[field]
    assert after['status'] == 'waiting_jobs'
    assert after['token'] in rt.store.snapshot()['leases']
    assert rt.snapshot()['nodes']['rank']['status'] == 'pending'


def test_new_formal_approval_allows_one_native_activation_after_a_block(factory, tmp_path, monkeypatch):
    rt = approved_activation_upgrade(factory, tmp_path)
    starts = []
    def refuse(version, **kwargs):
        starts.append(version)
        raise ValueError('guard remains blocked')
    monkeypatch.setattr(rt, 'start', refuse)
    rt.tick(); rt.tick()
    revised = node('rank', path='newly-approved-rank-input', deps=['charge'])
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'rank', 'node': revised}], 2, 3, 'new formal approval')
    rt.main.goal_contract.approved_nodes[-1] = revised
    rt.main.goal_contract.approved_plan_version = 3
    rt.tick(); rt.tick()
    assert starts == [2, 3]
    assert len(rt.store.snapshot()['activation_recoveries'][rt.workflow_id]) == 2


@pytest.mark.parametrize('invalid', ['target', 'floor', 'strict', 'verification', 'pattern'])
def test_native_batch_floor_compatibility_requires_exact_kernel_proof(factory, tmp_path, invalid):
    rt = approved_activation_upgrade(factory, tmp_path, native_floor=True)
    with json_transaction(rt.store.path) as data:
        current = data['workflows'][rt.workflow_id]['nodes']['charge']
        if invalid == 'target': current['batch_success_policy']['outputs'][0]['declared_target_count'] = 29
        if invalid == 'floor': current['batch_success_policy']['outputs'][0]['artifact_presence_floor'] = 2
        if invalid == 'strict': current['batch_success_policy']['mode'] = 'strict'
        if invalid == 'verification': current['node_verification'] = None
        if invalid == 'pattern': current['contract']['expected_outputs'][0]['pattern'] = '*.dat'
    assert rt._approved_activation_candidate(rt.snapshot()) is None


@pytest.mark.parametrize('saved_method,approved_version,expected', [('CDFT', 1, True),
    ('CDFT', 0, False), ('VASP', 1, False)])
def test_activation_uses_existing_approved_multi_method_normalization(factory, tmp_path,
                                                                    saved_method, approved_version, expected):
    rt = approved_activation_upgrade(factory, tmp_path)
    extra = [node('cdf', tool='run_cdft', args={'action': 'inputs'}, deps=['rank']),
             node('gcm', tool='run_gcmc_isotherm', args={'gas': 'N2'}, deps=['rank'])]
    with json_transaction(rt.store.path) as data:
        old = data['workflows'][rt.workflow_id]['goal_contract']
        old.update(method=saved_method, approved_plan_version=approved_version)
        old['approved_nodes'].extend(copy.deepcopy(extra))
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': n['step_id'], 'node': n} for n in extra], 2, 3, 'approved multi-method graph')
    rt.main.goal_contract.approved_nodes.extend(copy.deepcopy(extra))
    rt.main.goal_contract.method = 'MULTI'
    rt.main.goal_contract.approved_plan_version = 3
    assert (rt._approved_activation_candidate(rt.snapshot()) is not None) is expected


def formal_current_approval(rt):
    from agents.parallel_workflow import _approval_digest
    owner = {'username': rt.username, 'conv_id': rt.conv_id}
    goal = rt.main.goal_contract.to_dict()
    return {'schema_version': 1, 'kind': 'main_model_applied_plan', 'owner': owner,
            'goal_version': goal['version'], 'plan_version': goal['approved_plan_version'],
            'base_plan_version': goal['approved_plan_version'] - 1,
            'approval_sha256': _approval_digest(goal), 'approved_at': time.time(),
            'control_source': {'origin': 'user', 'owner': owner, 'turn_id': 'authenticated-user-turn'}}


@pytest.mark.parametrize('missing_binding', [False, True])
@pytest.mark.parametrize('proof_kind', ['main_model_applied_plan', 'explicit_user_parameter_patch'])
def test_formally_approved_new_goal_reuses_unaffected_completed_results(factory, tmp_path, missing_binding, proof_kind):
    rt = approved_activation_upgrade(factory, tmp_path, native_floor=True, missing_binding=missing_binding)
    before = rt.snapshot()
    rt.main.goal_contract.version += 1
    rt.main.goal_contract.parameters['cycles'] = 1000
    revised = node('rank', path='new-goal-approved-rank-input', deps=['charge'])
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'rank', 'node': revised}], 2, 3, 'formal new-goal approval')
    rt.main.goal_contract.approved_nodes[-1] = revised
    rt.main.goal_contract.approved_plan_version = 3
    rt.main.context['approved_execution_proof'] = formal_current_approval(rt)
    rt.main.context['approved_execution_proof']['kind'] = proof_kind
    assert rt._recover_approved_activation(rt.snapshot()) is False
    after = rt.snapshot()
    assert after['plan_version'] == 3
    assert after['goal_contract']['version'] == rt.main.goal_contract.version
    assert after['goal_contract']['parameters']['cycles'] == 1000
    for step_id in ('generate', 'charge'):
        for field in ('token', 'attempt_id', 'job_ids', 'artifacts', 'node_verification', 'result'):
            assert after['nodes'][step_id][field] == before['nodes'][step_id][field]
    assert after['nodes']['charge']['contract']['input_bindings'] == {
        'cif_dir': {'step_id': 'generate', 'output_index': 0}}
    assert after['nodes']['rank']['status'] == 'pending'


@pytest.mark.parametrize('invalid', ['missing', 'goal_version', 'plan_version', 'hash', 'owner',
                                   'system_source', 'empty_turn', 'guard_rejects'])
def test_new_goal_activation_requires_current_formal_approval_and_guard(factory, tmp_path, monkeypatch, invalid):
    rt = approved_activation_upgrade(factory, tmp_path)
    rt.main.goal_contract.version += 1
    rt.main.goal_contract.parameters['cycles'] = 1000
    revised = node('rank', path='formally-updated-rank-input', deps=['charge'])
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'rank', 'node': revised}], 2, 3, 'formal current goal')
    rt.main.goal_contract.approved_plan_version = 3
    rt.main.goal_contract.approved_nodes[-1] = revised
    proof = formal_current_approval(rt)
    if invalid == 'goal_version': proof['goal_version'] -= 1
    if invalid == 'plan_version': proof['plan_version'] -= 1
    if invalid == 'hash': proof['approval_sha256'] = 'different'
    if invalid == 'owner': proof['owner'] = {'username': 'other', 'conv_id': rt.conv_id}
    if invalid == 'system_source': proof['control_source']['origin'] = 'system'
    if invalid == 'empty_turn': proof['control_source']['turn_id'] = ''
    if invalid != 'missing': rt.main.context['approved_execution_proof'] = proof
    if invalid == 'guard_rejects':
        monkeypatch.setattr(rt.main.goal_contract, 'guard_tool_call', lambda *args, **kwargs: (False, 'new user parameter is not reflected'))
    assert rt._approved_activation_candidate(rt.snapshot()) is None


def test_new_user_parameter_cannot_reactivate_an_old_approval(factory, tmp_path):
    rt = approved_activation_upgrade(factory, tmp_path)
    old_proof = formal_current_approval(rt)
    rt.main.goal_contract.version += 1
    rt.main.goal_contract.parameters['cycles'] = 1000
    rt.main.context['approved_execution_proof'] = old_proof
    assert rt._approved_activation_candidate(rt.snapshot()) is None
    assert rt.snapshot()['plan_version'] == 1


def test_readonly_find_cif_can_deliver_an_existing_unchanged_input(factory, tmp_path):
    original = tmp_path / 'existing.cif'
    original.write_text('data_existing\n_cell_length_a 10\n')
    os.utime(original, (time.time() - 3600, time.time() - 3600))
    before = original.stat().st_mtime_ns
    rt = factory([node('verify_input', tool='find_cif', args={'name': str(original)}, outputs=[str(original)])])
    rt.main.registry.get('find_cif').execute = lambda _: {
        'ok': True, 'status': 'found', 'read_only': True, 'search_complete': True,
        'candidates': [str(original)], 'cif_path': str(original)}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['verify_input']['status'] in {'succeeded', 'validation_failed'})
    result = rt.snapshot()['nodes']['verify_input']
    assert result['status'] == 'succeeded'
    assert result['artifacts'][str(original)] == result['output_baseline'][str(original)]
    assert original.stat().st_mtime_ns == before


@pytest.mark.parametrize('case', ['read_other', 'read_receipt_mismatch', 'find_other', 'find_text_hint',
                                 'inspect_unlisted', 'inspect_receipt_mismatch', 'inspect_text_hint',
                                 'schema_path_annotation'])
def test_readonly_receipt_cannot_validate_an_unobserved_existing_artifact(factory, tmp_path, case):
    directory = tmp_path / 'inputs'
    directory.mkdir()
    first, other = directory / 'first.cif', directory / 'other.cif'
    first.write_text('data_first\n'); other.write_text('data_other\n')
    for path in (first, other):os.utime(path, (time.time() - 3600, time.time() - 3600))
    if case.startswith('read_'):
        tool, arguments = 'read_file', {'path': str(first)}
        result = {'path': str(other if case == 'read_receipt_mismatch' else first), 'content': 'read fixture'}
    elif case.startswith('find_'):
        tool, arguments = 'find_cif', {'name': str(first)}
        result = {'ok': True, 'status': 'found', 'read_only': True, 'search_complete': True,
                  'candidates': [str(first)], 'cif_path': str(first)}
        if case == 'find_text_hint':result['message'] = 'Unverified suggested path: ' + str(other)
    elif case.startswith('inspect_'):
        tool, arguments = 'inspect_path', {'path': str(directory)}
        result = {'path': str(other if case == 'inspect_receipt_mismatch' else directory), 'exists': True,
                  'kind': 'directory', 'entries': [{'name': first.name, 'path': str(first), 'kind': 'file'}]}
        if case == 'inspect_text_hint':result['preview'] = [str(other)]
    else:
        tool, arguments = 'get_tool_schema', {'tool_name': 'read_file'}
        result = {'name': 'read_file', 'input_schema': {'description': str(other)}, 'path': str(other)}
    rt = factory([node('inspect', tool=tool, args=arguments, outputs=[str(other)])])
    rt.main.registry.get(tool).execute = lambda _: result
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['inspect']['status'] in {'succeeded', 'validation_failed'})
    assert rt.snapshot()['nodes']['inspect']['status'] == 'validation_failed'


@pytest.mark.parametrize('case', ['read_exact', 'inspect_exact_file', 'inspect_exact_directory', 'inspect_listed_child'])
def test_readonly_receipt_validates_only_its_exact_native_observed_paths(factory, tmp_path, case):
    directory = tmp_path / 'inputs'
    directory.mkdir()
    first = directory / 'first.cif'
    first.write_text('data_first\n')
    os.utime(first, (time.time() - 3600, time.time() - 3600))
    if case == 'read_exact':
        tool, arguments = 'read_file', {'path': str(first)}
        result = {'path': str(first), 'content': 'data_first\n'}
        expected = str(first)
    elif case == 'inspect_exact_file':
        tool, arguments = 'inspect_path', {'path': str(first)}
        result = {'path': str(first), 'exists': True, 'kind': 'file'}
        expected = str(first)
    else:
        tool, arguments = 'inspect_path', {'path': str(directory)}
        result = {'path': str(directory), 'exists': True, 'kind': 'directory',
                  'entries': [{'name': first.name, 'path': str(first), 'kind': 'file'}]}
        expected = ({'kind': 'directory', 'path': str(directory), 'pattern': '*.cif', 'min_count': 1}
                    if case == 'inspect_exact_directory' else str(first))
    rt = factory([node('inspect', tool=tool, args=arguments, outputs=[expected])])
    rt.main.registry.get(tool).execute = lambda _: result
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['inspect']['status'] in {'succeeded', 'validation_failed'})
    assert rt.snapshot()['nodes']['inspect']['status'] == 'succeeded'


@pytest.mark.parametrize('missing_binding', [False, True])
@pytest.mark.parametrize('blocker', [None, 'receipt_owner', 'artifact_changed', 'runtime_owner', 'input_proof_changed'])
def test_new_plan_preserves_verified_executor_results_when_line_still_pending(factory, tmp_path, missing_binding, blocker):
    root = tmp_path / 'runs/tester/verified-plan-upgrade'
    generated, charged = root / 'generated', root / 'charged'
    generate = node('generate', tool='generate_structure', args={'material_type': 'MOF',
        'n_structures': 1, 'output_dir': str(generated)},
        outputs=[{'kind': 'directory', 'path': str(generated), 'pattern': '*.cif', 'min_count': 1}])
    generate['agent'] = 'harness'
    charge = node('charge', tool='run_pacman_charge', args={'cif_dir': str(generated),
        'method': 'pacmof', 'output_dir': str(charged)}, deps=['generate'],
        outputs=[{'kind': 'directory', 'path': str(charged), 'pattern': '*.cif', 'min_count': 1}])
    charge['input_bindings'] = {'cif_dir': {'step_id': 'generate', 'output_index': 0}}
    rt = factory([generate, charge, node('rank', deps=['charge'])], cid='verified-plan-upgrade')
    rt.start(1)
    for directory in (generated, charged):
        directory.mkdir()
        (directory / 'a.cif').write_text('real artifact fixture\n')
    with json_transaction(rt.store.path) as data:
        w = data['workflows'][rt.workflow_id]
        for step_id, job_id in [('generate', '101'), ('charge', '102')]:
            current = w['nodes'][step_id]
            current.update(status='succeeded', phase='finish', token=step_id + '-token',
                job_ids=[job_id], attempt_id=step_id + '-attempt', node_verification={'passed': True},
                effective_arguments=copy.deepcopy(current['contract']['arguments']),
                result={'output_dir': str(generated if step_id == 'generate' else charged)},
                execution_fingerprint='earlier-runtime-source')
            current['artifacts'] = rt._artifacts(current['contract'], current['result'])
        if missing_binding: w['nodes']['charge']['contract'].pop('input_bindings')
        if blocker == 'input_proof_changed':
            w['nodes']['charge']['effective_arguments']['cif_dir'] = str(root / 'different-input')
        if blocker == 'runtime_owner': w['username'] = 'someone-else'
    rt.job_watch = SimpleNamespace(list=lambda: [], get=lambda job_id: {
        'username': 'someone-else' if blocker == 'receipt_owner' else 'tester',
        'conv_id': rt.conv_id, 'terminal': True, 'failed': False, 'state': 'COMPLETED'})
    revised = node('rank', path='corrected-rank-input', deps=['charge'])
    rt.task_lines.apply_workflow_patch(rt.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'rank', 'node': revised}], 1, 2, 'approved pending operation repair')
    rt.main.goal_contract.approved_plan_version = 2
    rt.main.goal_contract.approved_nodes[-1] = revised
    assert not rt._line()['steps'][0]['done']
    if blocker == 'artifact_changed':
        (generated / 'a.cif').write_text('changed artifact fixture\n')
    before = rt.snapshot()
    rt.main.registry.get('generate_structure').execute = lambda _: pytest.fail('completed generation must not resubmit')
    rt.main.registry.get('run_pacman_charge').execute = lambda _: pytest.fail('completed charge must not resubmit')
    if blocker and (blocker != 'input_proof_changed' or missing_binding):
        with pytest.raises((ValueError, PermissionError)):
            rt.start(2)
        assert rt.snapshot()['plan_version'] == 1
        for step_id in ('generate', 'charge'):
            assert rt.snapshot()['nodes'][step_id]['status'] == 'succeeded'
        return
    assert rt.start(2)['status'] == 'scheduled'
    after = rt.snapshot()
    assert after['plan_version'] == 2
    for step_id in ('generate', 'charge'):
        for field in ('status', 'phase', 'token', 'attempt_id', 'job_ids', 'result', 'node_verification', 'artifacts'):
            assert after['nodes'][step_id][field] == before['nodes'][step_id][field]
    assert after['nodes']['charge']['contract']['input_bindings'] == charge['input_bindings']
    assert after['nodes']['rank']['status'] == 'pending'
    assert after['nodes']['rank']['contract']['arguments'] == revised['arguments']


def test_start_preserves_approved_input_bindings_in_runtime_contract(factory, tmp_path):
    source = tmp_path / 'source.txt'
    first = node('producer', path=str(source), outputs=[str(source)])
    consumer = node('consumer', path=str(source), deps=['producer'])
    consumer['input_bindings'] = {'path': {'step_id': 'producer', 'output_index': 0}}
    rt = factory([first, consumer], cid='preserve-bindings')
    rt.start(1)
    assert rt.snapshot()['nodes']['consumer']['contract']['input_bindings'] == consumer['input_bindings']


@pytest.mark.parametrize('upgrade', ['start', 'tick'])
def test_missing_binding_upgrade_preserves_completed_and_prefinish_branches(factory, tmp_path, upgrade):
    source = tmp_path / 'produced.txt'
    producer = node('producer', path=str(source), outputs=[str(source)])
    consumer = node('consumer', path=str(source), deps=['producer'])
    consumer['input_bindings'] = {'path': {'step_id': 'producer', 'output_index': 0}}
    rt = factory([producer, consumer, node('literature')], cid='binding-upgrade')
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        nodes = data['workflows'][rt.workflow_id]['nodes']
        nodes['consumer']['contract'].pop('input_bindings')
        nodes['producer'].update(status='prefinish', token='generation-token', job_ids=['completed-generation'],
            result={'state': 'COMPLETED'}, validation_dossier={
                'schema_version': VALIDATION_SCHEMA_VERSION, 'successful_artifacts': []})
        nodes['literature'].update(status='succeeded', result={'content': 'verified literature'}, phase='finish')
    before = rt.snapshot()
    rt.main.registry.get('read_file').execute = lambda _: pytest.fail('metadata repair must not dispatch work')
    if upgrade == 'start':
        assert rt.start(1)['resubmitted'] is False
    else:
        rt.tick(blocked_nodes=['producer', 'consumer', 'literature'])
    after = rt.snapshot()
    assert after['nodes']['producer'] == before['nodes']['producer']
    assert after['nodes']['literature'] == before['nodes']['literature']
    restored = after['nodes']['consumer']
    assert restored['status'] == 'pending' and not restored.get('attempt_id')
    assert restored['contract']['input_bindings'] == consumer['input_bindings']
    assert restored['contract_metadata_history'][0]['no_dispatch']
    assert len(restored['contract_metadata_history']) == 1
    assert rt.store.restore_missing_input_bindings(rt.workflow_id, rt.main.goal_contract.to_dict(),
        rt._line(), rt.username, rt.conv_id) == []
    from agents.workflow_view import authoritative_line
    visible = authoritative_line(rt._line(), after)
    assert next(n for n in visible['steps'] if n['step_id'] == 'consumer')['contract_matches_execution']


@pytest.mark.parametrize('status', ['waiting_jobs', 'prefinish', 'succeeded', 'needs_resources'])
@pytest.mark.parametrize('proof_valid', [True, False])
def test_binding_metadata_upgrade_preserves_executed_and_resource_waiting_facts(factory, tmp_path, status, proof_valid):
    source = tmp_path / 'structures'
    source.mkdir()
    consumer = node('consumer', args={'path': str(source)}, tool='inspect_path', deps=['producer'])
    consumer['input_bindings'] = {'path': {'step_id': 'producer', 'output_index': 0}}
    rt = factory([node('producer', path=str(source), outputs=[{'kind': 'directory', 'path': str(source)}]),
                  consumer], cid='claimed-binding-upgrade')
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        w = data['workflows'][rt.workflow_id]
        current = w['nodes']['consumer']
        current['contract'].pop('input_bindings')
        current.update(status=status, token='current-token', artifacts={'original': {'exists': True}},
                       phase='prefinish' if status == 'prefinish' else None,
                       dispatch_phase='preflight' if status == 'needs_resources' else 'entered_tool')
        if status != 'needs_resources':
            current.update(effective_arguments=copy.deepcopy(consumer['arguments']),
                           attempt_id='actual-attempt', job_ids=['existing-job'], tool_returned=True)
            if not proof_valid:
                current['effective_arguments']['path'] = str(tmp_path / 'different-input')
        elif not proof_valid:
            current['dispatch_phase'] = 'entered_tool'
        data.setdefault('leases', {})['current-token'] = {'workflow_id': rt.workflow_id, 'step_id': 'consumer',
                                                       'resources': [{'path': str(source), 'mode': 'shared'}]}
    before = rt.store.snapshot()
    restored = rt.store.restore_missing_input_bindings(rt.workflow_id, rt.main.goal_contract.to_dict(),
        rt._line(), rt.username, rt.conv_id, tmp_path, rt.root)
    if not proof_valid:
        assert restored == []
        assert rt.store.snapshot() == before
        return
    assert restored == ['consumer']
    after = rt.store.snapshot()
    expected = copy.deepcopy(before)
    fixed = after['workflows'][rt.workflow_id]['nodes']['consumer']
    expected_node = expected['workflows'][rt.workflow_id]['nodes']['consumer']
    expected_node['contract']['input_bindings'] = consumer['input_bindings']
    expected_node['contract_metadata_history'] = fixed['contract_metadata_history']
    assert after == expected, 'only the missing declaration and its proof may change'
    proof = fixed['contract_metadata_history'][-1]
    assert proof['binding_proof'][0]['resolved_path'] == str(source)
    assert proof['binding_proof'][0]['basis'] == ('unexecuted_preflight_contract' if status == 'needs_resources' else 'persisted_effective_arguments')


@pytest.mark.parametrize('blocker', ['owner', 'line_owner', 'goal_version', 'plan_version', 'line_version',
    'approval_version', 'goal_science', 'approval_binding', 'line_binding', 'line_science', 'runtime_science',
    'live_node', 'completed_node', 'attempt', 'old_token', 'tool_returned', 'jobs', 'live_lease', 'user_pause'])
def test_missing_binding_upgrade_rejects_unowned_stale_or_executed_contracts(factory, tmp_path, blocker):
    source = tmp_path / 'source.txt'
    consumer = node('consumer', path=str(source))
    consumer['input_bindings'] = {'path': {'step_id': 'producer', 'output_index': 0}}
    rt = factory([node('producer', path=str(source), outputs=[str(source)]),
                  {**consumer, 'depends_on': ['producer']}], cid='binding-reject')
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id]
        current = workflow['nodes']['consumer']
        current['contract'].pop('input_bindings')
        if blocker == 'owner': workflow['username'] = 'other'
        if blocker == 'goal_version': workflow['goal_contract']['version'] += 1
        if blocker == 'plan_version': workflow['plan_version'] += 1
        if blocker == 'goal_science': workflow['goal_contract']['gases'] = ['different-gas']
        if blocker == 'approval_binding': workflow['goal_contract']['approved_nodes'][1]['input_bindings'] = {}
        if blocker == 'runtime_science': current['contract']['arguments']['path'] = 'changed'
        if blocker == 'live_node': current['status'] = 'running'
        if blocker == 'completed_node': current['status'] = 'succeeded'
        if blocker == 'attempt': current['attempt_id'] = 'prior-attempt'
        if blocker == 'old_token': current['token'] = 'old-token'
        if blocker == 'tool_returned': current['tool_returned'] = True
        if blocker == 'jobs': current['job_ids'] = ['existing-job']
        if blocker == 'live_lease': data.setdefault('leases', {})['lease'] = {'workflow_id': rt.workflow_id, 'step_id': 'consumer'}
        if blocker == 'user_pause': workflow['user_paused'] = True
    goal, line = rt.main.goal_contract.to_dict(), rt._line()
    if blocker == 'line_owner': line['username'] = 'other'
    if blocker == 'line_version': line['plan_version'] += 1
    if blocker == 'approval_version': goal['approved_plan_version'] += 1
    if blocker == 'line_binding': line['steps'][1]['input_bindings'] = {}
    if blocker == 'line_science': line['steps'][1]['arguments']['path'] = 'changed'
    before = rt.store.snapshot()
    assert rt.store.restore_missing_input_bindings(rt.workflow_id, goal, line, rt.username, rt.conv_id) == []
    assert rt.store.snapshot() == before


def test_independent_branches_really_overlap_and_join_waits(factory):
    rt = factory([node('a'), node('b'), node('join', deps=['a', 'b'])])
    entered, release = threading.Barrier(3), threading.Event()
    executed = []
    def tool(args):
        executed.append(args['path'])
        if args['path'] != 'join':
            entered.wait(timeout=4)
            assert release.wait(4)
        return {'content': args['path']}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1)
    rt.tick()
    try:
        entered.wait(timeout=4)  # A/B and this thread must reach the barrier together.
        assert set(executed) == {'a', 'b'}
        assert rt.snapshot()['nodes']['join']['status'] == 'pending'
    finally: release.set()
    eventually(lambda: all(rt.snapshot()['nodes'][n]['status'] == 'succeeded' for n in ('a', 'b')))
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    join = rt.snapshot()['nodes']['join']
    saved = json.loads(Path(join['checkpoint_path']).read_text())
    assert set(saved['context']['dependency_results']) == {'a', 'b'}
    assert len({n['checkpoint_path'] for n in rt.snapshot()['nodes'].values()}) == 3
    assert rt.main.messages == [] and rt.main.memory.tool_call_log == []


def test_fixed_pormake_configuration_resumes_original_node_without_model(factory, tmp_path):
    output = tmp_path / 'runs/tester/c1/structures'
    contract = node('generate', tool='generate_structure', args={
        'material_type': 'MOF', 'n_structures': 2, 'output_dir': str(output)},
        outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 2}])
    contract['agent'] = 'harness-maintainer'
    runtime = factory([contract])
    missing = tmp_path / 'missing-python'
    runtime.main.config._tools_config = {'pormake': {'python': str(missing)}}
    calls = []
    def generator(arguments):
        calls.append(arguments)
        if len(calls) == 1:
            return {'blocked': True, 'executed': False,
                    'error': f'configured pormake interpreter is unavailable: {missing}'}
        return {'submitted': True, 'job_id': '91001', 'work_dir': str(output)}
    runtime.main.registry.get('generate_structure').execute = generator
    runtime.start(1)
    runtime.tick()
    eventually(lambda: runtime.snapshot()['nodes']['generate']['status'] == 'failed')
    executable = tmp_path / 'python'
    executable.write_text('#!/bin/sh\nexit 0\n')
    executable.chmod(0o700)
    runtime.main.config._tools_config['pormake']['python'] = str(executable)
    runtime.tick()
    eventually(lambda: runtime.snapshot()['nodes']['generate']['status'] == 'waiting_jobs')
    state = runtime.snapshot()['nodes']['generate']
    assert len(calls) == 2
    assert state['attempt_history'][0]['status'] == 'failed'
    assert state['job_ids'] == ['91001']
    assert runtime.main.memory.tool_call_log == []


def test_shared_resource_serializes_across_conversations_not_everything(factory):
    a = factory([node('a', locks=['global:same-output'])], 'a')
    b = factory([node('b', locks=['global:same-output']), node('unrelated')], 'b')
    entered, release = threading.Event(), threading.Event()
    def hold(args):
        entered.set()
        assert release.wait(4)
        return {'content': 'first'}
    a.main.registry.get('read_file').execute = hold
    b.main.registry.get('read_file').execute = lambda p: {'content': p['path']}
    a.start(1); b.start(1)
    a.tick()
    try:
        assert entered.wait(4)
        b.tick()
        eventually(lambda: b.snapshot()['nodes']['unrelated']['status'] == 'succeeded')
        assert b.snapshot()['nodes']['b']['status'] == 'pending'
    finally: release.set()
    eventually(lambda: a.snapshot()['status'] == 'completed')
    b.tick()
    eventually(lambda: b.snapshot()['status'] == 'completed')


@pytest.mark.parametrize('mode_a,mode_b,expected', [('read','read',False),('write','read',True),('read','write',True),('write','write',True)])
def test_parent_child_paths_and_shared_reads(mode_a, mode_b, expected):
    assert conflicts({'key':'path:/project/output','mode':mode_a}, {'key':'path:/project/output/result.csv','mode':mode_b}) == expected
    assert not conflicts({'key':'path:/project/output','mode':mode_a}, {'key':'path:/project/output-other','mode':mode_b})


def process_claim(path, wid, event, queue):
    event.wait(5)
    queue.put(bool(WorkflowStore(path).claim(wid, 'a')))


def test_multiple_processes_only_one_claims_node_and_all_resources(factory):
    rt = factory([node('a', locks=['r1', 'r2'])])
    rt.start(1)
    ctx = mp.get_context('fork')
    event, queue = ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=process_claim, args=(rt.store.path, rt.workflow_id, event, queue)) for _ in range(4)]
    for process in processes: process.start()
    event.set()
    assert sum(queue.get(timeout=6) for _ in processes) == 1
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0 and not process.is_alive()
    leases = rt.store.snapshot()['leases']
    assert len(leases) == 1 and len(next(iter(leases.values()))['resources']) == 3
    queue.close(); queue.join_thread()


def test_running_messages_notify_main_supervisor_and_worker_receipts(factory):
    rt = factory([node('a')])
    entered, release = threading.Event(), threading.Event()
    def tool(args):
        entered.set(); assert release.wait(4)
        return {'content': 'actual result'}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1); rt.tick()
    try:
        assert entered.wait(4)
        message = rt.send('a', 'Please retain input provenance', 'comment', 'same-id')
        assert rt.send('a', message['text'], 'comment', 'same-id') == message
        event = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_user_message')
        assert event['main_chat'] == event['supervisor'] == 'pending'
        assert event['payload']['message']['text'] == message['text']
    finally: release.set()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    state = rt.snapshot()['nodes']['a']
    assert state['messages']['same-id']['receipt'] == 'seen_at_tool_boundary'
    saved = json.loads(Path(state['checkpoint_path']).read_text())
    assert saved['context']['received_messages']['same-id']['text'] == message['text']


def test_change_request_does_not_mutate_running_parameters_or_start_descendant(factory):
    rt = factory([node('a'), node('next', deps=['a'])])
    entered, release = threading.Event(), threading.Event()
    seen = []
    def tool(args):
        seen.append(copy.deepcopy(args)); entered.set(); assert release.wait(4)
        return {'content':'old approved params'}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1); rt.tick()
    try:
        assert entered.wait(4)
        assert rt.send('a', 'change input to other.cif', 'change')['receipt'] == 'awaiting_main_chat_approval'
    finally: release.set()
    eventually(lambda: rt.snapshot()['nodes']['a']['status'] == 'succeeded')
    rt.tick()
    assert rt.snapshot()['status'] == 'needs_user'
    assert rt.snapshot()['nodes']['next']['status'] == 'pending'
    assert seen == [{'path':'a'}]


def test_submitted_job_keeps_lease_unknown_cannot_release_then_fresh_output_unlocks(factory, tmp_path):
    output = tmp_path / 'fresh.csv'
    rt = factory([node('calc', tool='run_henry', args={'cif':'x.cif','gas':'Kr','temperature':298}, outputs=[str(output)])])
    rt.main.registry.get('run_henry').execute = lambda p: {'submitted':True,'job_id':'42','work_dir':str(tmp_path)}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    assert len(rt.store.snapshot()['leases']) == 1
    rt.tick([{'job_id':'42','state':'UNKNOWN','terminal':True,'failed':False}])
    assert rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs'
    assert len(rt.store.snapshot()['leases']) == 1
    output.write_text('actual new output')
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
    assert rt.snapshot()['status']=='active'  # exit alone never finishes the task
    verify_prefinish(rt,'calc')
    assert rt.snapshot()['status'] == 'completed'
    assert not rt.store.snapshot()['leases']


def test_prefinish_creates_one_validation_dossier_and_emits_it(factory, tmp_path):
    from tests.fixture_native_dossier import prepare_native_result,write_native_result
    work=tmp_path/'runs/tester/one-pass-dossier/native'
    cif=work/'framework.cif';output=work/'Output/System_0/native-output.any-name'
    rt = factory([node('calc', tool='run_henry', args={
        'cif': str(cif), 'gas': 'Kr', 'temperature': 298,
    }, outputs=[str(output)])], cid='one-pass-dossier')
    actual_cif,actual_output=prepare_native_result(work,filename='native-output.any-name')
    assert (actual_cif,actual_output)==(cif,output)
    rt.main.registry.get('run_henry').execute = lambda params: {
        'submitted': True, 'job_id': '420', 'work_dir': str(work),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    write_native_result(output,1.25)

    rt.tick([{'job_id': '420', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])

    current = rt.snapshot()['nodes']['calc']
    assert current['status'] == 'prefinish'
    dossier = current['validation_dossier']
    assert dossier['verdict'] == 'pass'
    assert dossier['counts']['succeeded'] >= 1
    assert Path(dossier['evidence_path']).exists()
    events = [event for event in rt.snapshot()['outbox'].values()
              if event['kind'] == 'worker_prefinish']
    assert len(events) == 1
    assert events[0]['payload']['validation_dossier']['evidence_call_id'] == dossier['evidence_call_id']
    rt.finish_node('calc', [dossier['evidence_call_id']],
                   'One-pass dossier verified the real native output and scheduler-owned result root.')
    assert rt.snapshot()['nodes']['calc']['status'] == 'succeeded'


def test_collector_exception_preserves_prefinish_without_resubmission(factory, tmp_path, monkeypatch):
    output = tmp_path / 'native-output'
    calls = []
    rt = factory([node('calc', tool='run_henry', args={
        'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298,
    }, outputs=[str(output)])], cid='dossier-exception')
    rt.main.registry.get('run_henry').execute = lambda params: calls.append(params) or {
        'submitted': True, 'job_id': '421', 'work_dir': str(tmp_path),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    output.write_text('native calculation output\n')
    monkeypatch.setattr('agents.result_validation.build_validation_dossier',
                        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('parser exploded')))

    rt.tick([{'job_id': '421', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])

    current = rt.snapshot()['nodes']['calc']
    assert current['status'] == 'prefinish'
    assert current['validation_dossier']['verdict'] == 'invalid'
    assert 'parser exploded' in current['validation_dossier']['reasons'][0]
    assert len(calls) == 1


def test_dossier_from_another_attempt_cannot_finish_current_node(factory, tmp_path):
    output = tmp_path / 'native-output'
    rt = factory([node('calc', tool='run_henry', args={
        'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298,
    }, outputs=[str(output)])], cid='dossier-identity')
    rt.main.registry.get('run_henry').execute = lambda params: {
        'submitted': True, 'job_id': '422', 'work_dir': str(tmp_path),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    output.write_text('Simulation finished\nAverage loading absolute [mol/kg framework] 1.5\n')
    rt.tick([{'job_id': '422', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])
    dossier = rt.snapshot()['nodes']['calc']['validation_dossier']
    foreign_id = 'foreignvalidation'
    foreign = json.loads(Path(dossier['evidence_path']).read_text())
    foreign['call_id'] = foreign_id
    foreign['result']['evidence_call_id'] = foreign_id
    foreign['result']['attempt_id'] = 'some-other-attempt'
    write_checkpoint(rt.root / 'evidence' / f'{foreign_id}.json', foreign)

    with pytest.raises(ValueError, match='identity does not match'):
        rt.finish_node('calc', [foreign_id], 'This dossier belongs to the wrong execution attempt.')
    assert rt.snapshot()['nodes']['calc']['status'] == 'prefinish'


def test_agent_repairs_stale_output_contract_from_current_dossier(factory, tmp_path):
    from tests.fixture_native_dossier import prepare_native_result,write_native_result
    work=tmp_path/'runs/tester/dossier-contract-repair/native'
    cif=work/'framework.cif';output=work/'Output/System_0/native-output'
    stale=work/'intermediate-output-that-does-not-exist'
    rt = factory([node('calc', tool='run_henry', args={
        'cif': str(cif), 'gas': 'Kr', 'temperature': 298,
    }, outputs=[str(output), str(stale)])], cid='dossier-contract-repair')
    actual_cif,actual_output=prepare_native_result(work,filename='native-output')
    assert (actual_cif,actual_output)==(cif,output)
    calls = []
    rt.main.registry.get('run_henry').execute = lambda params: calls.append(params) or {
        'submitted': True, 'job_id': '423', 'work_dir': str(work),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    write_native_result(output,1.5)
    rt.tick([{'job_id': '423', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])
    current = rt.snapshot()['nodes']['calc']
    assert current['validation_dossier']['verdict'] == 'pass'
    assert current['machine_output_check'] is False

    repaired = rt.revalidate_outputs('calc', [str(output)])

    assert repaired['status'] == 'prefinish'
    current = rt.snapshot()['nodes']['calc']
    assert current['contract']['expected_outputs'] == [
        {'kind': 'file', 'path': str(output)}]
    assert repaired['evidence_call_id'] == current['validation_dossier']['evidence_call_id']
    rt.finish_node('calc', [repaired['evidence_call_id']],
                   'Agent repaired the stale contract from current-attempt dossier evidence.')
    assert rt.snapshot()['nodes']['calc']['status'] == 'succeeded'
    assert len(calls) == 1


@pytest.mark.parametrize('tool,arguments,filename,content', [
    ('run_henry', {'cif': 'input.cif', 'gas': 'Kr', 'temperature': 298},
     'raspa-native.custom',
     'Simulation finished\nAverage loading absolute [mol/kg framework] 2.75 +/- 0.03\n'),
    ('run_pacman_charge', {'cif_path': 'input.cif', 'method': 'pacmof'},
     'charged-framework.custom',
     'data_charged\nloop_\n_atom_site_label\n_atom_site_charge\nC1 -0.125\nH1 0.125\n'),
    ('run_md_optimize', {'cif_path': 'input.cif', 'mode': 'single'},
     'engine-screen.custom',
     'Step Temp TotEng\n0 298 -1234.5\nLoop time of 4.2 on 8 procs for 1000 steps\n'),
    ('run_vasp', {'action': 'submit'},
     'electronic-native.custom',
     'reached required accuracy - stopping structural energy minimisation\n'
     'free energy TOTEN = -432.125 eV\nGeneral timing and accounting informations for this job\n'),
])
def test_native_calculations_use_one_dossier_then_finish(
        factory, tmp_path, tool, arguments, filename, content):
    cid = 'native-' + tool
    work = tmp_path / 'runs' / 'tester' / cid / 'result-root'
    source = tmp_path / 'input.cif'
    source.write_text('data_input\n_cell_length_a 10\n_atom_site_label C1\n')
    arguments = dict(arguments)
    if 'cif' in arguments:
        arguments['cif'] = str(source)
    if 'cif_path' in arguments:
        arguments['cif_path'] = str(source)
    if tool in {'run_pacman_charge', 'run_md_optimize'}:
        arguments['output_dir'] = str(work)
    if tool == 'run_vasp':
        arguments['work_dir'] = str(work)
    output = work / filename
    contract = node('calc', tool=tool, args=arguments, outputs=[str(output)])
    if tool == 'run_md_optimize':
        contract['agent'] = 'adsorption'
    rt = factory([contract], cid=cid)
    work.mkdir()
    calls = []
    rt.main.registry.get(tool).execute = lambda params: calls.append(params) or {
        'submitted': True, 'job_id': '510', 'work_dir': str(work),
        'output_dir': str(work),
    }

    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    output.write_text(content)
    rt.tick([{'job_id': '510', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])

    current = rt.snapshot()['nodes']['calc']
    dossier = current['validation_dossier']
    assert current['status'] == 'prefinish'
    assert dossier['verdict'] == 'pass'
    assert dossier['requires_agent_review'] is True
    assert dossier['diagnostic_excerpts']
    rt.finish_node('calc', [dossier['evidence_call_id']],
                   'Scientific owner accepted the native facts and bounded source excerpts.')
    assert rt.snapshot()['nodes']['calc']['status'] == 'succeeded'
    assert len(calls) == 1


@pytest.mark.parametrize('tool,arguments,content', [
    ('run_henry', {'cif': 'input.cif', 'gas': 'Kr', 'temperature': 298},
     'Simulation finished\nAverage loading absolute [mol/kg framework] NaN\n'),
    ('run_pacman_charge', {'cif_path': 'input.cif', 'method': 'pacmof'},
     'data_bad\nloop_\n_atom_site_label\n_atom_site_charge\nC1 NaN\n'),
    ('run_md_optimize', {'cif_path': 'input.cif', 'mode': 'single'},
     'Step Temp TotEng\n0 298 NaN\n'),
    ('run_vasp', {'action': 'submit'},
     'free energy TOTEN = NaN eV\nGeneral timing and accounting informations for this job\n'),
])
def test_native_calculation_anomalies_stay_prefinish_for_agent_recovery(
        factory, tmp_path, tool, arguments, content):
    cid = 'abnormal-' + tool
    work = tmp_path / 'runs' / 'tester' / cid / 'result-root'
    source = tmp_path / 'input.cif'
    source.write_text('data_input\n_cell_length_a 10\n_atom_site_label C1\n')
    arguments = dict(arguments)
    if 'cif' in arguments:
        arguments['cif'] = str(source)
    if 'cif_path' in arguments:
        arguments['cif_path'] = str(source)
    if tool in {'run_pacman_charge', 'run_md_optimize'}:
        arguments['output_dir'] = str(work)
    if tool == 'run_vasp':
        arguments['work_dir'] = str(work)
    output = work / 'arbitrary-program-output'
    contract = node('calc', tool=tool, args=arguments, outputs=[str(output)])
    if tool == 'run_md_optimize':
        contract['agent'] = 'adsorption'
    rt = factory([contract], cid=cid)
    work.mkdir()
    calls = []
    rt.main.registry.get(tool).execute = lambda params: calls.append(params) or {
        'submitted': True, 'job_id': '511', 'work_dir': str(work),
        'output_dir': str(work),
    }

    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    output.write_text(content)
    rt.tick([{'job_id': '511', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])

    current = rt.snapshot()['nodes']['calc']
    assert current['status'] == 'prefinish'
    assert current['validation_dossier']['verdict'] == 'recover'
    assert current['validation_dossier']['failed_items']
    assert len(calls) == 1


def test_generator_can_revalidate_real_split_output_directories(factory, tmp_path):
    output = tmp_path / 'runs' / 'tester' / 'c1' / 'generated'
    contract = node('gen', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 2, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 2}])
    contract['agent'] = 'harness'
    rt = factory([contract])
    rt.job_watch = SimpleNamespace(
        list=lambda: [],
        get=lambda job_id: {
            'username': 'tester', 'conv_id': 'c1', 'terminal': True,
            'failed': False, 'state': 'COMPLETED', 'work_dir': str(output),
        },
    )
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-generator',
        'resource_allocations': {'gen': {'memory_mb': 4096, 'cpus': 2}},
    }
    rt.main.registry.get('generate_structure').execute = lambda args: {
        'submitted': True, 'job_id': '42', 'output_dir': str(output),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['gen']['status'] != 'running')
    current = rt.snapshot()['nodes']['gen']
    assert current['status'] == 'waiting_jobs', (current['status'], current.get('error'))
    (output / 'large').mkdir(parents=True)
    (output / 'small').mkdir()
    (output / 'large' / 'a.cif').write_text('data_a')
    (output / 'small' / 'b.cif').write_text('data_b')
    rt.tick([{'job_id': '42', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])
    assert rt.snapshot()['nodes']['gen']['status'] == 'prefinish'

    result = rt.revalidate_outputs('gen', [
        {'kind': 'directory', 'path': str(output / 'large'), 'pattern': '*.cif', 'min_count': 1},
        {'kind': 'directory', 'path': str(output / 'small'), 'pattern': '*.cif', 'min_count': 1},
    ], completed_job_id='42')

    assert result['ok'] and result['verified_files'] == 2
    assert rt.snapshot()['nodes']['gen']['status'] == 'prefinish'
    verify_prefinish(rt, 'gen')
    assert rt.snapshot()['nodes']['gen']['status'] == 'succeeded'


@pytest.mark.parametrize('preexisting', [False, True])
def test_missing_or_old_output_blocks_dependency_but_allows_repair(factory, tmp_path, preexisting):
    output = tmp_path / 'old.csv'
    if preexisting:
        output.write_text('old task')
        os.utime(output, (1,1))
    rt = factory([node('a', outputs=[str(output)]), node('next', deps=['a'])])
    rt.main.registry.get('read_file').execute = lambda p: {'content':'no new output'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    assert rt.snapshot()['nodes']['a']['status'] == 'validation_failed'
    assert not rt.store.snapshot()['leases']
    rt.tick()
    assert rt.snapshot()['nodes']['next']['status'] == 'pending'
    eventually(lambda: any(e['kind'] == 'worker_validation_failed' for e in rt.lifecycle.pending()))
    event = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_validation_failed')
    assert event['payload']['requires_user'] is False
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    # Simulate the state-to-outbox crash gap, then reconstruct the same repair
    # notification. A restart must not turn a technical validation failure into
    # a new scientific decision for the user.
    with json_transaction(rt.store.path) as data:
        data['workflows'][rt.workflow_id]['outbox'] = {}
    with json_transaction(rt.lifecycle.path) as data:
        data['events'] = {}
    rt.recover_notifications()
    replay = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_validation_failed')
    assert replay['payload']['requires_user'] is False


def test_real_error_preserved_and_no_automatic_retry(factory):
    rt = factory([node('a'), node('next', deps=['a'])])
    calls = []
    rt.main.registry.get('read_file').execute = lambda p: calls.append(p) or {'error':'real CIF parser failure'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    for _ in range(3): rt.tick()
    assert len(calls) == 1 and rt.snapshot()['nodes']['next']['status'] == 'pending'
    event = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_failed')
    assert event['payload']['error'] == 'real CIF parser failure'
    assert event['payload']['evidence_call']['call_id']
    assert Path(event['payload']['evidence_call']['evidence_path']).is_file()


def test_dependency_runtime_arguments_refines_only_existing_descendant(tmp_path):
    approved = tmp_path / 'cdft'; actual = approved / 'data' / '20260920_124153'
    actual.mkdir(parents=True)
    node_state = {'contract': {'tool': 'run_cdft', 'arguments': {
        'action': 'collect', 'job_work_dir': str(approved)},
        'depends_on': ['pipeline']}}
    states = {'pipeline': {'status': 'succeeded', 'result': {'work_dir': str(actual)}}}

    arguments, bindings = dependency_runtime_arguments(node_state, states, tmp_path)

    assert arguments['job_work_dir'] == str(actual)
    assert bindings == [{'parameter': 'job_work_dir', 'source_step': 'pipeline',
                         'receipt_field': 'work_dir', 'approved_root': str(approved),
                         'resolved_path': str(actual)}]
    foreign = tmp_path.parent / 'foreign-runtime-binding'
    foreign.mkdir(exist_ok=True)
    states['pipeline']['result']['work_dir'] = str(foreign)
    arguments, bindings = dependency_runtime_arguments(node_state, states, tmp_path)
    assert arguments['job_work_dir'] == str(approved) and bindings == []


def test_tick_materializes_dependency_receipt_before_downstream_dispatch(factory, tmp_path):
    approved = tmp_path / 'cdft'
    actual = approved / 'data' / 'timestamp'
    actual.mkdir(parents=True)
    output = approved / 'results.csv'
    rt = factory([
        node('pipeline', tool='run_cdft', args={
            'action': 'pipeline', 'cif_dir': str(tmp_path), 'gas': 'CO2',
            'temperature': 298, 'job_work_dir': str(approved),
        }, outputs=[str(actual / 'results.csv')]),
        node('collect', tool='run_cdft', args={
            'action': 'collect', 'job_work_dir': str(approved),
            'output': str(output),
        }, deps=['pipeline'], outputs=[str(output)]),
    ], cid='materialized-runtime-binding')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'pipeline')
    rt.store.update(rt.workflow_id, 'pipeline', ticket['token'], {
        'status': 'succeeded', 'phase': 'finish',
        'result': {'work_dir': str(actual)},
    }, release=True)

    rt.tick(blocked_nodes=('collect',))

    collect = rt.snapshot()['nodes']['collect']
    assert collect['resolved_arguments']['job_work_dir'] == str(actual)
    assert collect['runtime_input_bindings'][0]['source_step'] == 'pipeline'
    assert collect['runtime_binding_fingerprint']

    actual.rmdir()
    rt.tick(blocked_nodes=('collect',))
    collect = rt.snapshot()['nodes']['collect']
    assert 'resolved_arguments' not in collect
    assert collect['contract']['arguments']['job_work_dir'] == str(approved)


def test_agent_runtime_input_repair_resumes_failed_node_with_upstream_receipt(factory, tmp_path):
    approved = tmp_path / 'cdft'; actual = approved / 'data' / 'timestamp'
    actual.mkdir(parents=True)
    output = approved / 'results.csv'
    rt = factory([
        node('pipeline', tool='run_cdft', args={'action': 'pipeline',
             'cif_dir': str(tmp_path), 'gas': 'CO2', 'temperature': 298,
             'job_work_dir': str(approved)}, outputs=[str(actual / 'results.csv')]),
        node('collect', tool='run_cdft', args={'action': 'collect',
             'job_work_dir': str(approved), 'output': str(output)},
             deps=['pipeline'], outputs=[str(output)]),
    ], cid='agent-runtime-repair')
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id]
        workflow['nodes']['pipeline'].update(status='succeeded', phase='finish',
                                             result={'work_dir': str(actual)})
        workflow['nodes']['collect'].update(status='failed', error='collection failed')
        workflow.update(status='needs_user', pause_kind='failure', pause_reason='internal path')
    calls = []
    def collect(arguments):
        calls.append(copy.deepcopy(arguments))
        output.write_text('MOF,CO2_henry_mol_L_atm\na,1.0\n')
        return {'submitted': False, 'rows': [{'MOF': 'a', 'CO2_henry_mol_L_atm': 1.0}],
                'output_csv': str(output)}
    rt.main.registry.get('run_cdft').execute = collect

    receipt = rt.repair_runtime_inputs('collect')

    assert receipt['ok'] and receipt['requires_user'] is False
    eventually(lambda: bool(calls))
    assert calls[0]['job_work_dir'] == str(actual)
    eventually(lambda: rt.snapshot()['nodes']['collect']['status'] == 'succeeded')
    assert rt.snapshot()['status'] == 'completed'
    assert rt.snapshot()['nodes']['collect']['runtime_input_bindings'][0]['source_step'] == 'pipeline'


def test_main_agent_sdk_can_call_runtime_input_repair_tool(factory, tmp_path):
    from agents.defns import ORCHESTRATOR
    from anthropic.types import TextBlock, ToolUseBlock

    approved = tmp_path / 'cdft'; actual = approved / 'data' / 'timestamp'
    actual.mkdir(parents=True)
    output = approved / 'results.csv'
    rt = factory([
        node('pipeline', tool='run_cdft', args={'action': 'pipeline',
             'cif_dir': str(tmp_path), 'gas': 'CO2', 'temperature': 298,
             'job_work_dir': str(approved)}, outputs=[str(actual / 'results.csv')]),
        node('collect', tool='run_cdft', args={'action': 'collect',
             'job_work_dir': str(approved), 'output': str(output)},
             deps=['pipeline'], outputs=[str(output)]),
    ], cid='sdk-runtime-repair')
    rt.start(1)
    with json_transaction(rt.store.path) as data:
        workflow = data['workflows'][rt.workflow_id]
        workflow['nodes']['pipeline'].update(status='succeeded', phase='finish',
                                             result={'work_dir': str(actual)})
        workflow['nodes']['collect'].update(status='failed', error='collection failed')
        workflow.update(status='needs_user', pause_kind='failure', pause_reason='internal path')
    calls = []
    def collect(arguments):
        calls.append(copy.deepcopy(arguments))
        output.write_text('MOF,CO2_henry_mol_L_atm\na,1.0\n')
        return {'submitted': False, 'rows': [{'MOF': 'a', 'CO2_henry_mol_L_atm': 1.0}],
                'output_csv': str(output)}
    rt.main.registry.get('run_cdft').execute = collect
    rt.main.current_agent = ORCHESTRATOR
    rt.main._on_workflow_runtime_repair = rt.repair_runtime_inputs
    rt.main.messages = [{'role': 'user', 'content': '[生命周期事件] worker_failed'}]
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='repair',
            name='repair_workflow_runtime_inputs', input={'step_id': 'collect'})]),
        SimpleNamespace(content=[TextBlock(type='text', text='Runtime input repaired; continuing.')]),
    ])
    rt.main._call_api = lambda agent: next(responses)

    rt.main._execute_loop(ORCHESTRATOR)

    eventually(lambda: bool(calls))
    assert calls[0]['job_work_dir'] == str(actual)
    assert any(call['tool'] == 'repair_workflow_runtime_inputs' and not call['failed']
               for call in rt.main.memory.tool_call_log)


def test_dead_readonly_worker_restored_without_killing_main_role(factory):
    rt = factory([node('a')])
    rt.main.registry.get('read_file').execute = lambda p: {'content':'restored'}
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {'owner_pid':99999999})
    with json_transaction(rt.store.path) as data:
        data['leases'][ticket['token']]['owner_pid'] = 99999999
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    rt.lifecycle.heartbeat()
    assert rt.lifecycle.snapshot()['agents']['main_chat']['lifetime']
    assert rt.lifecycle.snapshot()['agents']['supervisor']['lifetime']


def test_dead_mutating_worker_unknown_keeps_lease_and_never_redispatches(factory):
    rt = factory([node('a', tool='write_file', args={'path':'x','content':'new'})])
    rt.main.registry.get('write_file').execute = lambda p: pytest.fail('unknown mutation must not redispatch')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {'owner_pid':99999999,'dispatch_phase':'entered_tool'})
    rt.tick(); rt.tick()
    assert rt.snapshot()['status'] == 'needs_user'
    assert rt.snapshot()['nodes']['a']['status'] == 'uncertain'
    assert len(rt.store.snapshot()['leases']) == 1


def test_dead_preflight_writer_is_failed_not_a_permanently_unreconcilable_unknown(factory):
    rt=factory([node('a',tool='write_file',args={'path':'x','content':'new'})]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'owner_pid':99999999})
    rt.tick()
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert rt.snapshot()['status']=='needs_user' and not rt.store.snapshot()['leases']


def test_storage_failure_blocks_dispatch(factory, monkeypatch):
    rt = factory([node('a')])
    rt.start(1)
    rt.main.registry.get('read_file').execute = lambda p: pytest.fail('no dispatch without checkpoint')
    monkeypatch.setattr('agents.parallel_workflow.write_checkpoint', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    assert rt.snapshot()['nodes']['a']['status'] in {'failed','uncertain'}


def test_lifetime_heartbeat_busy_stall_warning_is_idempotent_and_mailbox_survives(factory):
    rt = factory([node('a')])
    rt.lifecycle.emit('task', {}, event_id='busy')
    assert rt.lifecycle.claim('busy','main_chat')
    with json_transaction(rt.lifecycle.path) as data:
        data['events']['busy']['main_chat_claimed_at'] = time.time()-400
    rt.lifecycle.heartbeat(); rt.lifecycle.heartbeat()
    state = rt.lifecycle.snapshot()
    assert state['agents']['main_chat']['busy_event_id'] == 'busy'
    assert len([e for e in state['events'].values() if e['kind']=='role_unresponsive']) == 1
    restored = LifecycleStore(rt.lifecycle.path)
    restored.recover_claims()
    assert restored.snapshot()['events']['busy']['main_chat'] == 'processing'  # live owner is never stolen


def test_wrong_version_unapproved_and_legacy_jobs_do_not_run(factory):
    rt = factory([node('a')])
    with pytest.raises(ValueError, match='exact approved'): rt.start(2)
    rt.main.goal_contract.execution_authorized = False
    with pytest.raises(ValueError, match='approval'): rt.start(1)
    rt.main.goal_contract.execution_authorized = True
    rt.task_lines.upsert_step('c1','a',job_ids=['old'],status='submitted')
    with pytest.raises(ValueError, match='legacy'): rt.start(1)


def test_busy_main_api_still_accepts_targeted_worker_messages(factory, monkeypatch):
    pytest.importorskip('fastapi')
    import api
    from fastapi.testclient import TestClient
    rt = factory([node('a')]); rt.start(1)
    conv = SimpleNamespace(is_processing=True, interrupt_requested=False)
    monkeypatch.setattr(api.session_manager, 'get_user', lambda u: SimpleNamespace(get_conversation=lambda cid: conv if cid=='c1' else None))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', lambda c: rt)
    api.app.dependency_overrides[api._get_user] = lambda: 'tester'
    try:
        client = TestClient(api.app)
        response = client.post('/api/conversations/c1/workflow/nodes/a/messages',json={'text':'status please','kind':'status'})
        assert response.status_code == 200 and response.json()['receipt'] == 'status_returned'
        assert client.post('/api/conversations/c1/workflow/execute',json={'plan_version':1}).status_code == 409
        assert client.post('/api/conversations/other/workflow/nodes/a/messages',json={'text':'x'}).status_code == 404
        assert any(e['kind']=='worker_user_message' for e in rt.lifecycle.pending())
    finally: api.app.dependency_overrides.pop(api._get_user,None)


def test_message_outbox_recovers_crash_between_state_and_mailbox(factory, monkeypatch):
    rt = factory([node('a')]); rt.start(1)
    emit = rt.lifecycle.emit
    monkeypatch.setattr(rt.lifecycle, 'emit', lambda *a, **kw: (_ for _ in ()).throw(OSError('mailbox unavailable')))
    with pytest.raises(OSError): rt.send('a','retain this message','comment','durable-message')
    assert rt.snapshot()['nodes']['a']['messages']['durable-message']['text'] == 'retain this message'
    assert any(not e['sent'] for e in rt.snapshot()['outbox'].values())
    monkeypatch.setattr(rt.lifecycle, 'emit', emit)
    rt.flush_events(); rt.flush_events()
    events = [e for e in rt.lifecycle.pending() if e['kind']=='worker_user_message']
    assert len(events) == 1 and all(e['sent'] for e in rt.snapshot()['outbox'].values())


def test_helper_error_after_job_acceptance_never_releases_live_resource(factory):
    rt = factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    calls=[]
    rt.main.registry.get('run_henry').execute = lambda p: calls.append(p) or {'submitted':True,'job_id':'42','error':'helper failed after scheduler accepted'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['a']['status']=='uncertain')
    assert rt.store.snapshot()['leases']
    rt.tick([{'job_id':'42','state':'RUNNING','terminal':False,'failed':False}])
    assert rt.store.snapshot()['leases'] and len(calls)==1
    rt.tick([{'job_id':'42','state':'FAILED','terminal':True,'failed':True}])
    assert rt.snapshot()['nodes']['a']['status']=='failed' and not rt.store.snapshot()['leases']
    assert len(calls)==1


def test_restart_waiting_job_does_not_call_tool_again(factory, tmp_path):
    output=tmp_path/'result.csv'
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=[str(output)])])
    calls=[]
    rt.main.registry.get('run_henry').execute=lambda p:calls.append(p) or {'submitted':True,'job_id':'42'}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    restored=ParallelWorkflow(rt.main,rt.root,rt.store,rt.lifecycle,rt.task_lines,'tester','c1')
    try:
        restored.tick([{'job_id':'42','state':'RUNNING','terminal':False,'failed':False}])
        assert len(calls)==1 and restored.snapshot()['nodes']['a']['status']=='waiting_jobs'
        output.write_text('new real artifact')
        restored.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
        verify_prefinish(restored,'a')
        assert restored.snapshot()['status']=='completed' and len(calls)==1
    finally:restored.shutdown()


def test_patch_preserves_verified_unaffected_branch_and_repairs_failed_node(factory):
    rt=factory([node('a'),node('b',deps=['a'])])
    calls=[]
    entered,release=threading.Event(),threading.Event()
    def execute(p):
        calls.append(p['path'])
        if p['path']=='a':
            entered.set();assert release.wait(4)
        return {'content':'ok'}
    rt.main.registry.get('read_file').execute=execute
    rt.start(1);rt.tick()
    try:
        assert entered.wait(4)
        rt.store.pause(rt.workflow_id,'repair remaining branch')
    finally:release.set()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='succeeded')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    replacement=node('b',path='fixed',deps=['a'])
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'upsert','step_id':'b','node':replacement}],1,2,'approved node repair')
    rt.main.goal_contract.approved_nodes=line['steps']
    rt.main.goal_contract.approved_plan_version=2
    rt.main.goal_contract.version+=1
    rt.start(2);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')
    assert calls==['a','fixed']


def test_new_plan_with_identical_failed_node_does_not_retry(factory):
    rt = factory([node('a')])
    calls = []
    def fail(params):
        calls.append(params)
        return {'error': 'Fixture failure requires a concrete repair'}
    rt.main.registry.get('read_file').execute = fail
    rt.start(1)
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['a']['status'] == 'failed')
    original = rt.main.goal_contract.approved_nodes[0]
    line = rt.task_lines.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': 'a', 'node': original}], 1, 2, 'identical patch must not act as retry')
    rt.main.goal_contract.approved_nodes = line['steps']
    rt.main.goal_contract.approved_plan_version = 2
    receipt = rt.start(2)
    rt.tick()
    assert receipt['status'] == 'needs_recovery' and not receipt['scheduled']
    assert rt.snapshot()['nodes']['a']['status'] == 'failed' and len(calls) == 1


def test_resource_contract_includes_absolute_output_even_when_not_in_tool_args(tmp_path):
    outputs=resources_for(node('a',outputs=[str(tmp_path/'shared.csv')]),tmp_path,tmp_path/'conv')
    assert {'key':'path:'+str(tmp_path/'shared.csv'),'mode':'write'} in outputs


def test_long_workflow_digest_is_bounded_without_losing_global_counts():
    from agents.parallel_workflow import runtime_summary
    state={'status':'active','plan_version':1,'nodes':{str(i):{'status':'succeeded' if i else 'running',
           'messages':{'long':{'text':'x'*100000}},'error':'e'*10000} for i in range(1000)}}
    summary=runtime_summary(state)
    assert summary['counts']=={'running':1,'succeeded':999}
    assert len(summary['nodes'])==24 and summary['omitted_nodes']==976
    assert len(json.dumps(summary))<18000


def test_api_dispatcher_runs_workers_and_delivers_evidence_when_busy_main_returns(factory, monkeypatch):
    pytest.importorskip('fastapi')
    import asyncio
    import api
    from anthropic.types import TextBlock
    rt=factory([node('a')])
    rt.main._lifecycle_store=rt.lifecycle
    rt.main._evidence_root=rt.root/'evidence'
    rt.main.registry.get('read_file').execute=lambda p:{'content':'actual worker evidence'}
    rt.main._call_api=lambda agent:SimpleNamespace(content=[TextBlock(type='text',text='Read validation results are available.')])
    conv=SimpleNamespace(session=rt.main,username='tester',conv_id='c1',is_processing=True,
        interrupt_requested=False,interrupt_message='',last_progress_at=0,current_uid='busy',current_status='busy',execution_logs=[])
    monkeypatch.setattr(api.session_manager,'list_users',lambda:['tester'])
    monkeypatch.setattr(api.session_manager,'get_user',lambda u:SimpleNamespace(conversations={'c1':conv}))
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda c:rt)
    monkeypatch.setattr(api._watch,'list',lambda:[])
    monkeypatch.setattr(api,'_lifecycle_tasks',set())
    async def supervise(c,event):
        rt.lifecycle.receipt(event['event_id'],'supervisor','delivered',{'next_action':'advance_dependencies','reason':'test read-only receipt'})
    async def run_main(username,query,cid,lifecycle_event=None):
        rt.main.resume_lifecycle_event(lifecycle_event)
        c=conv;c.current_status='delivered';c.is_processing=False
    monkeypatch.setattr(api,'_supervise_lifecycle_event',supervise)
    monkeypatch.setattr(api,'run_agent_background',run_main)
    async def dispatch():
        await api._dispatch_lifecycle_once()
        if api._lifecycle_tasks:await asyncio.gather(*list(api._lifecycle_tasks))
    async def scenario():
        rt.start(1)
        await dispatch()
        for _ in range(200):
            if rt.snapshot()['status']=='completed':break
            await asyncio.sleep(.01)
        assert rt.snapshot()['status']=='completed'
        await dispatch();await dispatch()
        event=next(e for e in rt.lifecycle.pending() if e['kind']=='worker_completed')
        assert event['main_chat']=='pending'  # busy main does not lose or prematurely ACK it
        conv.is_processing=False
        await dispatch();await dispatch()
        assert rt.lifecycle.snapshot()['events'][event['event_id']]['main_chat']=='delivered'
        assert rt.main.memory.tool_call_log[0]['call_id']==event['payload']['evidence_call']['call_id']
        assert rt.main.memory.tool_call_log[0]['evidence_path']
    asyncio.run(scenario())


def test_shared_worker_evidence_deduplicates_and_keeps_original_agent_partition(factory):
    rt=factory([node('a')])
    rt.main._execute_loop=lambda agent:'received'
    call={'call_id':'shared-once','agent':'analyst','tool':'read_file','params':{'path':'x'},'result':'actual evidence'}
    event={'kind':'worker_completed','payload':{'evidence_call':call},'supervisor_receipt':{'evidence_calls':[call]}}
    rt.main.resume_lifecycle_event(event)
    rt.main.resume_lifecycle_event(event)
    assert len(rt.main.memory.tool_call_log)==1
    assert len(rt.main.memory.agent_memories['analyst']['tool_call_log'])==1
    assert not rt.main.memory.agent_memories.get('supervisor',{}).get('tool_call_log')


def test_user_freeze_survives_tick_and_explicit_start_resumes(factory):
    rt=factory([node('a')])
    rt.main.registry.get('read_file').execute=lambda p:{'content':'safe resumption'}
    rt.start(1)
    rt.store.user_pause(rt.workflow_id)
    assert rt.store.claim(rt.workflow_id,'a') is None
    rt.tick(paused=False)
    assert rt.snapshot()['nodes']['a']['status']=='pending'
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')


def test_deleting_main_cannot_orphan_live_owner_but_can_retire_unstarted_plan(factory):
    rt=factory([node('a')]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    with pytest.raises(ValueError,match='running/unknown'):rt.store.retire(rt.workflow_id)
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'status':'pending'},release=True)
    rt.store.retire(rt.workflow_id)
    assert rt.store.claim(rt.workflow_id,'a') is None
    assert rt.snapshot()['status']=='retired_by_user'


def test_global_admission_budget_is_not_multiplied_by_api_processes(factory):
    rt=factory([node('a'),node('b'),node('c')]);rt.start(1)
    store=WorkflowStore(rt.store.path,max_running=2)
    assert store.claim(rt.workflow_id,'a')
    assert store.claim(rt.workflow_id,'b')
    other_process_view=WorkflowStore(rt.store.path,max_running=2)
    assert other_process_view.claim(rt.workflow_id,'c') is None


def test_worker_initialization_failure_keeps_main_alive_and_does_not_hold_phantom_writer(factory, monkeypatch):
    rt=factory([node('a')]);rt.start(1)
    def broken_session(*a,**kw):raise RuntimeError('worker Session initialization failed')
    monkeypatch.setattr('agents.session.Session',broken_session)
    rt.tick()
    eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert not rt.store.snapshot()['leases']
    assert rt.main.goal_contract.original_goal=='hello'


def test_actual_main_loop_can_start_dag_after_old_handoff_threshold(factory, monkeypatch):
    from agents.agent import Agent
    from anthropic.types import ToolUseBlock,TextBlock
    rt=factory([node('a')])
    monkeypatch.setattr('agents.task_line.get_store',lambda:rt.task_lines)
    for _ in range(5):rt.main.memory.record_tool_call('lead-orchestrator','task_line_query',{},'{"ok":true}')
    def execute_workflow(**kwargs):raise AssertionError('Session must use its runtime callback')
    agent=Agent(name='lead-orchestrator',functions=[execute_workflow])
    rt.main.messages=[{'role':'user','content':'hello'}]
    rt.main._on_workflow_start=rt.start
    responses=iter([SimpleNamespace(content=[ToolUseBlock(type='tool_use',id='start',name='execute_workflow',input={'plan_version':1})]),
        SimpleNamespace(content=[TextBlock(type='text',text='Runtime has started.')])])
    rt.main._call_api=lambda agent:next(responses)
    rt.main._execute_loop(agent)
    assert rt.snapshot()['status']=='active' and not rt.main.task_complete
    assert rt.main.memory.tool_call_log[-1]['tool']=='execute_workflow'
    assert [n['tool'] for n in rt.task_lines.get_line('c1')['steps']]==['read_file']


def test_unknown_local_writer_has_user_negotiated_repair_exit_not_permanent_stall(factory,tmp_path):
    from agents.defns import ORCHESTRATOR
    target=tmp_path/'partial.txt';target.write_text('partial old value')
    rt=factory([node('a',tool='write_file',args={'path':str(target),'content':'correct value'})]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'owner_pid':99999999,'dispatch_phase':'entered_tool'})
    rt.tick()
    with pytest.raises(ValueError,match='genuine user'):rt.resolve_local_write('a','invented','invented')
    rt.main.current_agent=ORCHESTRATOR
    rt.main._request_user_decision('确认原写入已停止，是否允许检查后修补？','write_file stopped after crash; inspect original path',['a'])
    decision_id=rt.main._pending_user_interaction['params']['decision_id']
    rt.main._execute_loop=lambda agent,**kwargs:'main chat received genuine user approval'
    rt.main.reply('确认原写入已停止，同意修补',verbose=False)
    params={'path':str(target)}
    result=rt.main.registry.execute_dict('read_file',params)
    rt.main.memory.record_tool_call('lead-orchestrator','read_file',params,json.dumps(result))
    receipt=rt.resolve_local_write('a',decision_id,rt.main.memory.tool_call_log[-1]['call_id'])
    assert receipt['lease_released'] and not receipt['resubmitted']
    assert rt.snapshot()['nodes']['a']['status']=='failed' and rt.snapshot()['status']=='needs_user'
    assert not rt.store.snapshot()['leases'] and target.read_text()=='partial old value'


def test_two_scientific_submissions_with_disjoint_workdirs_really_overlap(factory,tmp_path):
    entered,release=threading.Barrier(3),threading.Event()
    nodes=[]
    scope=tmp_path/'runs/tester/c1'
    recipes={}
    for gas in ('Kr','Xe'):
        work=scope/gas
        recipes[gas]=_parallel_native_input(tmp_path,scope,gas)
        nodes.append(node(gas,tool='run_cdft',args={'action':'pipeline','cif_dir':str(tmp_path/'inputs'),
            'gas':gas,'temperature':298,'job_work_dir':str(work),'input_dir':str(recipes[gas][0])},outputs=[str(work/'results.csv')]))
    rt=factory(nodes)
    for gas in ('Kr','Xe'):
        (scope/gas).mkdir()
        recipes[gas][0].mkdir()
        (recipes[gas][0]/'a.dat').write_bytes((recipes[gas][1]/'input/a.dat').read_bytes())
    seen=[]
    def compute(args):
        _parallel_native_result(recipes[args['gas']][1],Path(args['job_work_dir']),args['gas'])
        seen.append(args['gas']);entered.wait(timeout=4);assert release.wait(4)
        return {'submitted':True,'job_id':'42' if args['gas']=='Kr' else '43','work_dir':args['job_work_dir']}
    rt.main.registry.get('run_cdft').execute=compute
    rt.start(1);rt.tick()
    try:
        entered.wait(timeout=4)
        assert set(seen)=={'Kr','Xe'}
        assert len(rt.store.snapshot()['leases'])==2
    finally:release.set()
    eventually(lambda:all(n['status']=='waiting_jobs' for n in rt.snapshot()['nodes'].values()))
    rt.tick([{'job_id':jid,'state':'COMPLETED','terminal':True,'failed':False} for jid in ('42','43')])
    for gas in ('Kr','Xe'):verify_prefinish(rt,gas)
    assert rt.snapshot()['status']=='completed' and not rt.store.snapshot()['leases']


def test_native_scheduler_receipt_binds_generic_output_csv_without_resubmission(factory, tmp_path):
    """A completed native job may repair only its approved session CSV location."""
    actual = tmp_path / 'runs' / 'tester' / 'c1' / 'pore_results.csv'
    approved = str(actual)
    # This reproduces a legacy contract whose project-relative output was later
    # interpreted relative to the tool work directory.
    legacy_output = 'runs/tester/c1/pore_results.csv'
    contract = node('pore', tool='run_pore_analysis', args={
        'cif_dir': str(tmp_path / 'runs' / 'tester' / 'c1' / 'subset_cifs'),
        'output_csv': approved,
    }, outputs=[legacy_output])
    rt = factory([contract])
    rt.start(1)
    claimed = rt.store.claim(rt.workflow_id, 'pore')
    assert claimed
    token = claimed['token']
    receipt = {'submitted': True, 'job_id': '77744', 'output_csv': approved,
               'work_dir': str(actual.parent / 'pore')}
    rt.store.update(rt.workflow_id, 'pore', token, {
        'status': 'prefinish', 'job_ids': ['77744'], 'jobs_confirmed_terminal': True, 'result': receipt,
        'result_ref': {'call_id': 'nativecall'},
        'evidence_call': {'call_id': 'nativecall', 'result': receipt},
    })

    assert rt._bind_native_result_path('pore', token, receipt)
    repaired = rt.snapshot()['nodes']['pore']
    assert repaired['contract']['expected_outputs'] == [{'kind': 'file', 'path': approved}]
    assert repaired['output_contract_history'][-1]['job_id'] == '77744'
    assert repaired['output_contract_history'][-1]['no_resubmission'] is True
    assert repaired['job_ids'] == ['77744']
    line_step = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert line_step['expected_outputs'] == [{'kind': 'file', 'path': approved}]

    # Including the exact execution receipt alongside real validation is a
    # harmless model habit. It must not poison the finish request, and it must
    # not itself be stored as scientific validation.
    actual.write_text('material,diameter\nMOF,12.3\n')
    evidence = rt.root / 'evidence'
    evidence.mkdir(exist_ok=True)
    write_checkpoint(evidence / 'nativecall.json', {'call_id': 'nativecall', 'time': time.time(),
        'tool': 'run_pore_analysis', 'params': contract['arguments'], 'result': receipt, 'failed': False})
    write_checkpoint(evidence / 'readcall.json', {'call_id': 'readcall', 'time': time.time(),
        'tool': 'read_file', 'params': {'path': approved}, 'result': {'content': actual.read_text()}, 'failed': False})
    settled = []
    rt._settle = lambda *args: settled.append(args)
    rt.finish_node('pore', ['nativecall', 'readcall'], 'CSV contains the expected real pore-analysis row.')
    verification = rt.snapshot()['nodes']['pore']['node_verification']
    assert verification['evidence_call_ids'] == ['readcall']
    assert settled and settled[0][0] == 'pore'


def test_native_scheduler_receipt_binds_output_directory_and_uses_best_effort_count(factory, tmp_path):
    actual = tmp_path / 'runs' / 'tester' / 'c1' / 'charged_cifs'
    contract = node('charge', tool='run_pacman_charge', args={
        'cif_dir': str(tmp_path / 'runs' / 'tester' / 'c1' / 'subset_cifs'),
        'method': 'pacmof', 'output_dir': str(actual),
    }, outputs=[{'kind': 'directory', 'path': 'runs/tester/c1/charged_cifs',
                 'pattern': '*_pacmof.cif', 'min_count': 6}])
    rt = factory([contract])
    actual.mkdir()
    rt.start(1)
    claimed = rt.store.claim(rt.workflow_id, 'charge')
    receipt = {'submitted': True, 'job_id': '77749', 'output_dir': str(actual),
               'work_dir': str(actual)}
    rt.store.update(rt.workflow_id, 'charge', claimed['token'], {
        'status': 'prefinish', 'job_ids': ['77749'], 'result': receipt,
        'result_ref': {'call_id': 'chargecall'},
        'evidence_call': {'call_id': 'chargecall', 'result': receipt},
    })
    assert rt._bind_native_result_path('charge', claimed['token'], receipt)
    rt._apply_batch_success_threshold('charge', claimed['token'])
    rt._apply_batch_success_threshold('charge', claimed['token'])
    expected = rt.snapshot()['nodes']['charge']['contract']['expected_outputs']
    assert expected == [{'kind': 'directory', 'path': str(actual),
                         'pattern': '*_pacmof.cif', 'min_count': 1}]
    assert rt.snapshot()['nodes']['charge']['batch_success_policy']['outputs'][0] == {
        'index': 0, 'declared_target_count': 6, 'artifact_presence_floor': 1}
    assert rt.snapshot()['nodes']['charge']['job_ids'] == ['77749']


def test_repaired_parent_invalidates_identical_child_contract_not_just_changed_arguments(factory):
    rt=factory([node('a'),node('child',deps=['a'])])
    calls=[]
    rt.main.registry.get('read_file').execute=lambda p:calls.append(p['path']) or {'content':'validated'}
    rt.start(1)
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='completed':break
        time.sleep(.01)
    assert rt.snapshot()['status']=='completed'
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    replacement=node('a',path='repaired-parent')
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'upsert','step_id':'a','node':replacement}],1,2,'user approved parent repair')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['nodes']['child']['status']=='pending'  # same args/dependency ID, different parent generation
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='completed':break
        time.sleep(.01)
    assert calls.count('child')==2 and rt.snapshot()['status']=='completed'


def test_same_client_message_id_on_different_nodes_does_not_hide_second_notification(factory):
    rt=factory([node('a'),node('b')]);rt.start(1)
    rt.send('a','A status','status','client-id')
    rt.send('b','B status','status','client-id')
    events=[e for e in rt.lifecycle.pending() if e['kind']=='worker_user_message']
    assert len(events)==2
    assert {e['payload']['message']['step_id'] for e in events}=={'a','b'}


def test_api_does_not_label_idle_main_as_completed_while_branches_work(factory):
    pytest.importorskip('fastapi')
    import api
    rt=factory([node('a')]);rt.start(1)
    assert api._execution_status(rt.main)=='编排分支运行中（主chat在岗）'
    rt.store.user_pause(rt.workflow_id)
    assert api._execution_status(rt.main)=='编排已暂停（主chat在岗）'
    rt.store.pause(rt.workflow_id,'needs repair')
    assert api._execution_status(rt.main)=='等待用户协商编排'


def test_approved_removal_of_failed_branch_with_only_cached_results_finishes(factory):
    rt=factory([node('a'),node('b',deps=['a'])])
    rt.main.registry.get('read_file').execute=lambda p:{'content':'ok'} if p['path']=='a' else {'error':'known branch failure'}
    rt.start(1)
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='needs_user':break
        time.sleep(.01)
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'remove','step_id':'b'}],1,2,'user approved removing failed branch')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['status']=='completed'


def test_pretty_helper_json_is_authoritative_not_only_last_brace():
    from agents.recovery import result_object,failure_reason
    result={'submitted':True,'stdout':'helper progress\n'+json.dumps({'submitted':False,'error':'scheduler refused'},indent=2)}
    assert failure_reason(result)=='scheduler refused'
    result['stdout']=json.dumps({'submitted':True,'job_id':'42'},indent=2)
    assert result_object(result)['job_id']=='42'


def test_missing_receipt_enters_user_reconciliation_not_empty_wait(factory):
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'stdout':'helper returned but no scheduler receipt'}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='uncertain' and rt.store.snapshot()['leases']
    eventually(lambda:any(e['kind']=='worker_uncertain' for e in rt.lifecycle.pending()))
    event=next(e for e in rt.lifecycle.pending() if e['kind']=='worker_uncertain')
    assert event['payload']['evidence_call']['call_id']


def test_real_cdft_executor_scopes_preparation_and_submission_to_explicit_directory(tmp_path,monkeypatch):
    monkeypatch.setattr('agents.registry._cdft_adapter_bootstrap', lambda *args: ('', []))
    from agents.registry import _exec_cdft
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.registry.get_config',lambda:config)
    monkeypatch.setattr('agents.workspace.session_dir',lambda kind:tmp_path/'unused-default')
    captured=[]
    def fake_run(command,**kwargs):
        captured.append((native_source(command, kwargs),kwargs['cwd']))
        return SimpleNamespace(returncode=0,stdout='{"submitted":true,"job_id":"42"}',stderr='')
    monkeypatch.setattr('subprocess.run',fake_run)
    work=tmp_path/'isolated'
    _exec_cdft({'action':'pipeline','cif_dir':'cifs','gas':'Kr','temperature':298,'job_work_dir':str(work),'memory_mb':512})
    assert captured and all(cwd==str(work) for code,cwd in captured)
    generated='\n'.join(code for code,cwd in captured)
    assert str(work/'inputs') in generated and str(work/'data') in generated
    assert str(tmp_path/'cifs') in generated
    resource=resources_for(node('x',tool='run_cdft',args={'action':'pipeline','input_dir':'shared'}),tmp_path,tmp_path/'run')
    assert {'key':'path:'+str(tmp_path/'shared'),'mode':'write'} in resource


def test_runtime_registers_authoritative_receipt_when_helper_did_not(factory,tmp_path):
    from agents.job_watch import JobWatch
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    rt.job_watch=JobWatch(watch_file=str(tmp_path/'watch.json'))
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'42','work_dir':str(tmp_path)}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    record=rt.job_watch.get('42')
    assert record['username']=='tester' and record['conv_id']=='c1' and record['tool']=='run_henry'


def test_crash_after_transition_or_message_before_outbox_is_recovered_once(factory):
    rt=factory([node('a')]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'status':'failed','error':'crashed before event enqueue'},release=True)
    rt.store.pause(rt.workflow_id,'failure')
    rt.store.send(rt.workflow_id,'a','message before enqueue','comment','lost-gap')
    rt.tick();rt.tick()
    events=rt.lifecycle.pending()
    assert len([e for e in events if e['kind']=='worker_failed'])==1
    assert len([e for e in events if e['kind']=='worker_user_message'])==1


def test_recovery_sweep_emits_new_agent_action_after_old_failure_was_delivered(factory, tmp_path):
    produced = tmp_path / 'approved' / 'attempt-1'
    produced.mkdir(parents=True)
    output = tmp_path / 'approved' / 'results.csv'
    rt = factory([
        node('upstream', tool='run_cdft', args={
            'action': 'pipeline', 'job_work_dir': str(tmp_path / 'approved'),
            'cif_dir': str(tmp_path), 'gas': 'CO2', 'temperature': 298,
        }, outputs=[str(produced / 'results.csv')]),
        node('downstream', tool='run_cdft', args={
            'action': 'collect', 'job_work_dir': str(tmp_path / 'approved'),
            'output': str(output),
        }, deps=['upstream'], outputs=[str(output)]),
    ], cid='recoverable-event')
    rt.start(1)
    up = rt.store.claim(rt.workflow_id, 'upstream')
    rt.store.update(rt.workflow_id, 'upstream', up['token'], {
        'status': 'succeeded', 'result': {'work_dir': str(produced)},
    }, release=True)
    down = rt.store.claim(rt.workflow_id, 'downstream')
    rt.store.update(rt.workflow_id, 'downstream', down['token'], {
        'status': 'failed', 'error': 'declared root has no material rows',
        'effective_arguments': {'input_dir': str(tmp_path / 'approved')},
    }, release=True)
    old_id = f"runtime:{rt.workflow_id}:{down['token']}:failed"
    rt._emit('worker_failed', {'step_id': 'downstream'}, down['token'] + ':failed')
    rt.lifecycle.receipt(old_id, 'supervisor', 'delivered', {'next_action': 'diagnose_and_fix'})
    rt.lifecycle.receipt(old_id, 'main_chat', 'delivered', {'status': 'handled'})

    rt.recover_notifications()

    events = [event for event in rt.lifecycle.pending()
              if event['kind'] == 'worker_recovery_ready']
    assert len(events) == 1
    assert events[0]['payload']['step_id'] == 'downstream'
    assert events[0]['payload']['recovery']['tool'] == 'repair_workflow_runtime_inputs'
    assert events[0]['payload']['requires_user'] is False
    rt.recover_notifications()
    assert len([event for event in rt.lifecycle.snapshot()['events'].values()
                if event['kind'] == 'worker_recovery_ready']) == 1


def test_failed_preflight_runtime_repairs_ghost_running_taskline_without_user_cleanup(factory):
    rt = factory([node('a')], cid='preflight-ghost')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.task_lines.upsert_step(rt.main._current_line_id, 'a', status='running', validation={
        'runtime_token': ticket['token'], 'resource_leases': ticket['contract']['resources'],
    })
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {
        'status': 'failed', 'error': 'projection interface mismatch',
        'dispatch_phase': 'preflight', 'tool_returned': False, 'job_ids': [],
    }, release=True)
    rt.store.pause(rt.workflow_id, 'preflight framework failure', step_id='a')
    rt.recover_notifications()
    projected = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert projected['status'] == 'failed'
    assert projected['validation']['runtime_token'] is None
    assert projected['validation']['resource_leases'] == []
    assert projected['validation']['dispatch_not_entered'] is True


def test_new_attempt_resets_terminal_flags_and_keeps_old_facts_as_history(factory):
    rt=factory([node('a')]);rt.start(1)
    old=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',old['token'],{'status':'pending','tool_returned':True,
        'jobs_confirmed_terminal':True,'job_ids':['old-job'],'result_ref':{'call_id':'old-call'},'error':'old'},release=True)
    new=rt.store.claim(rt.workflow_id,'a')
    assert new['token']!=old['token'] and not new['tool_returned'] and not new['jobs_confirmed_terminal']
    assert new['job_ids']==[] and 'result_ref' not in new and 'error' not in new
    assert new['attempt_history'][0]['result_ref']=={'call_id':'old-call'}
    assert not rt.store.update(rt.workflow_id,'a',old['token'],{'status':'failed'},release=True)
    assert rt.store.snapshot()['leases']


def test_old_prerequisite_terminal_flag_cannot_release_new_live_job(factory):
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])]);rt.start(1)
    old=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',old['token'],{'status':'pending','jobs_confirmed_terminal':True},release=True)
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'43','error':'helper failed after new job accepted'}
    rt.tick();eventually(lambda:rt.snapshot()['nodes']['a']['status']=='uncertain')
    assert not rt.snapshot()['nodes']['a']['jobs_confirmed_terminal'] and rt.store.snapshot()['leases']


@pytest.mark.parametrize('user_change',[False,True])
def test_verified_recovery_finishes_unless_user_requested_contract_change(factory,tmp_path,user_change):
    output=tmp_path/'result.csv'
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=[str(output)])])
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'42'}
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    state=rt.snapshot()['nodes']['a']
    rt.store.update(rt.workflow_id,'a',state['token'],{'status':'uncertain'})
    rt.store.pause(rt.workflow_id,'dispatch reconciliation',kind='reconciliation')
    if user_change:rt.send('a','change temperature after this call','change')
    output.write_text('fresh verified result')
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
    verify_prefinish(rt,'a')
    assert rt.snapshot()['nodes']['a']['status']=='succeeded'
    assert rt.snapshot()['status']==('needs_user' if user_change else 'completed')


def test_relative_outputs_use_same_actual_anchor_and_do_not_serialize_disjoint_cdft(factory,tmp_path):
    entered,release=threading.Barrier(3),threading.Event()
    nodes=[]
    scope=tmp_path/'runs/tester/c1'
    recipes={}
    for name in ('a','b'):
        work=scope/name
        recipes[name]=_parallel_native_input(tmp_path,scope,name)
        nodes.append(node(name,tool='run_cdft',args={'action':'pipeline','cif_dir':str(tmp_path/'cifs'),
            'gas':'Kr','temperature':298,'job_work_dir':str(work),'input_dir':str(recipes[name][0])},outputs=['results.csv']))
    rt=factory(nodes)
    for name in ('a','b'):
        (scope/name).mkdir()
        recipes[name][0].mkdir()
        (recipes[name][0]/'a.dat').write_bytes((recipes[name][1]/'input/a.dat').read_bytes())
    def compute(args):
        key=Path(args['job_work_dir']).name
        _parallel_native_result(recipes[key][1],Path(args['job_work_dir']),args['gas'])
        entered.wait(timeout=4);assert release.wait(4)
        return {'submitted':True,'job_id':'42' if args['job_work_dir'].endswith('a') else '43','work_dir':args['job_work_dir']}
    rt.main.registry.get('run_cdft').execute=compute
    rt.start(1);rt.tick()
    try:entered.wait(timeout=4)
    finally:release.set()
    eventually(lambda:all(n['status']=='waiting_jobs' for n in rt.snapshot()['nodes'].values()))
    # Unowned decoys must never substitute for the source-bound native CSV.
    for name in ('a','b'):
        (tmp_path/name).mkdir()
        (tmp_path/name/'results.csv').write_text('fresh')
    rt.tick([{'job_id':jid,'state':'COMPLETED','terminal':True,'failed':False} for jid in ('42','43')])
    for name in ('a','b'):verify_prefinish(rt,name)
    for name in ('a','b'):
        assert str(scope/name/'results.csv') in rt.snapshot()['nodes'][name]['artifacts']
        assert str(tmp_path/name/'results.csv') not in rt.snapshot()['nodes'][name]['artifacts']
        assert (tmp_path/name/'results.csv').read_text()=='fresh'
    assert rt.snapshot()['status']=='completed'


@pytest.mark.parametrize('change',['input','executor','solver_version','output'])
def test_cache_rechecks_input_code_version_and_artifacts_before_reuse(factory,tmp_path,change):
    source=tmp_path/'input.cif';source.write_text('initial input')
    output=tmp_path/'result.csv'
    rt=factory([node('a',path=str(source),outputs=[str(output)])])
    calls=[]
    def tool(params):
        calls.append(params);output.write_text(source.read_text());return {'content':'inspected'}
    rt.main.registry.get('read_file').execute=tool
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['status']=='completed')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    if change=='input':source.write_text('changed input')
    elif change=='executor':
        def replacement(params):return tool(params)
        rt.main.registry.get('read_file').execute=replacement
    elif change=='solver_version':rt.main.registry.get('read_file').version='new-validated-solver'
    else:output.write_text('externally modified output')
    line=rt.task_lines.apply_workflow_patch('c1',[],1,2,'new approved version; unchanged argument paths')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['nodes']['a']['status']=='pending'
    rt.tick();eventually(lambda:rt.snapshot()['status']=='completed')
    assert len(calls)==2


def test_directory_artifact_cache_survives_json_checkpoint(factory, tmp_path):
    folder = tmp_path / 'dataset'
    folder.mkdir()
    result = folder / 'sample.dat'
    rt = factory([node('a', outputs=[{'kind': 'directory', 'path': str(folder), 'pattern': '*.dat', 'min_count': 1}])])
    calls = []
    def inspect(params):
        calls.append(params)
        result.write_text('neutral fixture, not a scientific calculation')
        return {'content': 'fixture inspected'}
    rt.main.registry.get('read_file').execute = inspect
    rt.start(1)
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    line = rt.task_lines.apply_workflow_patch('c1', [], 1, 2, 'preserve the unchanged dataset')
    rt.main.goal_contract.approved_nodes = line['steps']
    rt.main.goal_contract.approved_plan_version = 2
    rt.main.goal_contract.version += 1
    rt.start(2)
    assert rt.snapshot()['nodes']['a']['status'] == 'succeeded'
    assert len(calls) == 1


def test_same_conversation_node_message_ids_in_different_users_do_not_share_state(factory):
    a=factory([node('same',locks=['workspace'])],'same-cid',username='alice')
    b=factory([node('same',locks=['workspace'])],'same-cid',username='bob')
    a.start(1);b.start(1)
    assert a.workflow_id!=b.workflow_id and a.main._current_line_id!=b.main._current_line_id
    assert a.store.claim(a.workflow_id,'same') and b.store.claim(b.workflow_id,'same')  # logical names are local
    a.send('same','Alice only','comment','same-message-id')
    b.send('same','Bob only','change','same-message-id')
    assert a.snapshot()['status']=='active' and b.snapshot()['status']=='needs_user'
    assert a.snapshot()['nodes']['same']['messages']['same-message-id']['text']=='Alice only'
    assert b.snapshot()['nodes']['same']['messages']['same-message-id']['text']=='Bob only'
    assert {e['payload']['message']['text'] for e in a.lifecycle.pending() if e['kind']=='worker_user_message'}=={'Alice only'}
    assert a.task_lines.get_by_conv('same-cid',username='alice')[0]['username']=='alice'
    assert b.task_lines.get_by_conv('same-cid',username='bob')[0]['username']=='bob'


def test_user_and_session_goals_memories_checkpoints_and_global_resources(factory):
    runtimes=[factory([node('same')],'one',username='alice'),factory([node('same')],'two',username='alice'),
              factory([node('same')],'one',username='bob')]
    for index,rt in enumerate(runtimes):
        rt.main.memory.record_user_preference('private',index)
        rt.main.registry.get('read_file').execute=lambda p:{'content':'own worker'}
        rt.start(1);rt.tick()
    for rt in runtimes:eventually(lambda:rt.snapshot()['status']=='completed')
    files=[Path(rt.snapshot()['nodes']['same']['checkpoint_path']) for rt in runtimes]
    assert len(set(files))==3
    for index,(rt,path) in enumerate(zip(runtimes,files)):
        assert rt.main.memory.user_preferences['private']==index
        assert rt.main.memory.tool_call_log==[]
        assert json.loads(path.read_text())['owner']=={'username':rt.username,'conv_id':rt.conv_id,'step_id':'same'}
    # Explicit global resources remain global; private scope must not hide a real conflict.
    a=factory([node('a',locks=['global:shared-device'])],'global-a',username='alice')
    b=factory([node('b',locks=['global:shared-device'])],'global-b',username='bob')
    a.start(1);b.start(1)
    assert a.store.claim(a.workflow_id,'a') and b.store.claim(b.workflow_id,'b') is None


def test_taskline_tools_never_enumerate_or_update_other_user_or_session(factory,monkeypatch):
    from agents.registry import _exec_task_line_query,_exec_task_line_update
    a=factory([node('a',path='Alice-private')],'same',username='alice')
    b=factory([node('b',path='Bob-private')],'same',username='bob')
    other=factory([node('c',path='Bob-other-session')],'other',username='bob')
    # One shared store contains both users' task lines, including a duplicate CID.
    monkeypatch.setattr('agents.task_line.get_store',lambda:b.task_lines)
    ctx={'username':'bob','conv_id':'same','line_id':b.main._current_line_id}
    monkeypatch.setattr('agents.watch_context.get_context',lambda:ctx)
    own=_exec_task_line_query({})
    assert own['ok'] and own['steps'][0]['arguments']['path']=='Bob-private'
    ambiguous=_exec_task_line_query({'line_id':'same'})
    assert 'Alice-private' not in json.dumps(ambiguous)
    assert not _exec_task_line_query({'conv_id':'other'})['ok']
    assert not _exec_task_line_update({'line_id':a.main._current_line_id,'step_id':'a','status':'failed'})['ok']
    ctx.clear()
    assert not _exec_task_line_query({})['ok']
    assert not _exec_task_line_update({'line_id':'same','step_id':'a','status':'failed'})['ok']


def test_api_jobs_admin_and_unknown_conversation_are_scoped(tmp_path,monkeypatch):
    pytest.importorskip('fastapi')
    import api,auth
    from fastapi.testclient import TestClient
    user=SimpleNamespace(conversations={'same':SimpleNamespace()},current_conv_id='same',get_conversation=lambda cid:None)
    monkeypatch.setattr(api.session_manager,'get_user',lambda username:user)
    monkeypatch.setattr(api._watch,'list',lambda:[{'job_id':'A','username':'alice','conv_id':'same'},
        {'job_id':'B','username':'bob','conv_id':'same'},{'job_id':'B2','username':'bob','conv_id':'other'}])
    monkeypatch.setattr(auth,'is_admin',lambda username:False)
    api.app.dependency_overrides[api._get_user]=lambda:'bob'
    try:
        client=TestClient(api.app)
        assert {j['job_id'] for j in client.get('/api/jobs').json()['jobs']}=={'B','B2'}
        assert [j['job_id'] for j in client.get('/api/jobs?conv_id=same').json()['jobs']]==['B']
        assert client.get('/api/admin/conversations').status_code==403
        assert client.get('/api/latest_answer?conv_id=foreign').status_code==404
    finally:api.app.dependency_overrides.pop(api._get_user,None)


@pytest.mark.parametrize('name',['..','.','../alice','/absolute','alice/c1','alice\\c1'])
def test_private_workspace_cannot_escape_its_user_session_scope(name,monkeypatch,tmp_path):
    from agents.workspace import session_root
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    with pytest.raises(ValueError):session_root('cid',name)
    with pytest.raises(ValueError):session_root(name,'alice')


def test_private_scope_cannot_symlink_into_another_user(monkeypatch,tmp_path):
    from agents.workspace import session_root
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    (tmp_path/'runs/bob/cid').mkdir(parents=True)
    (tmp_path/'runs/alice').symlink_to(tmp_path/'runs/bob',target_is_directory=True)
    with pytest.raises(ValueError,match='alias'):session_root('cid','alice')


def test_worker_checkpoint_foreign_owner_fails_before_tool(factory,tmp_path):
    import hashlib
    rt=factory([node('a')]);rt.start(1)
    path=rt.root/'workers'/hashlib.sha256(b'a').hexdigest()[:24]/'session_checkpoint.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'owner':{'username':'foreign','conv_id':'same','step_id':'a'}}))
    rt.main.registry.get('read_file').execute=lambda p:pytest.fail('foreign checkpoint must never dispatch')
    rt.tick();eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert 'different user/session' in rt.snapshot()['nodes']['a']['error']


def test_signup_cannot_use_path_identity_or_grant_admin(tmp_path,monkeypatch):
    import auth
    monkeypatch.setattr(auth,'USER_DB',tmp_path/'users.json')
    monkeypatch.delenv('BIMEM_ADMIN_USERS',raising=False)
    assert not auth.register_user('../alice','password')['ok']
    assert auth.register_user('alice','password')['ok']
    assert not auth.is_admin('alice')


@pytest.mark.parametrize('target',['runs/alice/same/input.cif','runs/bob/other/input.cif','tokens.json','users.json','data/state/parallel_workflows.json','logs/api_v33.log'])
def test_native_tool_scope_rejects_other_private_sessions_and_global_stores(tmp_path,target,monkeypatch):
    from agents.workspace import tool_scope_issues
    from agents.registry import _exec_restart_backend
    assert tool_scope_issues({'path':str(tmp_path/target)},'bob','same',tmp_path)
    assert not tool_scope_issues({'path':str(tmp_path/'runs/bob/same/input.cif')},'bob','same',tmp_path)
    assert not tool_scope_issues({'cif_dir':str(tmp_path/'shared-cifs')},'bob','same',tmp_path)
    monkeypatch.setattr('agents.watch_context.get_context',lambda:{'username':'bob','conv_id':'same'})
    monkeypatch.setattr('auth.is_admin',lambda name:False)
    assert _exec_restart_backend({})['blocked']


def test_compiled_node_cannot_write_into_other_user_session(factory,tmp_path):
    target=tmp_path/'runs/alice/one/file.txt'
    rt=factory([node('a',tool='write_file',args={'path':str(target),'content':'bad cross-user write'})],'one',username='bob')
    with pytest.raises(ValueError,match='private user/session'):rt.start(1)
    assert not target.exists()


def test_default_grep_is_private_and_repo_search_excludes_global_memories(monkeypatch,tmp_path):
    from agents.registry import _exec_grep_search
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    monkeypatch.setattr('agents.registry.get_config',lambda:config)
    monkeypatch.setattr('agents.watch_context.get_context',lambda:{'username':'bob','conv_id':'same'})
    calls=[]
    monkeypatch.setattr('subprocess.run',lambda cmd,**kw:calls.append(cmd) or SimpleNamespace(stdout='',returncode=0))
    _exec_grep_search({'pattern':'secret'})
    assert calls[-1][-1]==str(tmp_path/'runs/bob/same')
    _exec_grep_search({'pattern':'source','path':str(tmp_path)})
    assert '--exclude-dir=runs' in calls[-1] and '--exclude=tokens.json' in calls[-1]


def test_admin_restart_defers_when_another_user_session_is_busy(monkeypatch):
    pytest.importorskip('fastapi')
    import asyncio,api
    original_sleep=asyncio.sleep
    monkeypatch.setattr(api.asyncio,'sleep',lambda duration:original_sleep(0))
    monkeypatch.setattr(api.session_manager,'list_users',lambda:['bob'])
    monkeypatch.setattr(api.session_manager,'get_user',lambda u:SimpleNamespace(conversations={'busy':SimpleNamespace(is_processing=True)}))
    requested=[]
    monkeypatch.setattr('agents._restart.request',lambda:requested.append(True))
    monkeypatch.setattr(api.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not restart another user mid-turn'))
    asyncio.run(api._perform_backend_restart())
    assert requested==[True]


def test_same_approved_version_cannot_silently_return_stale_cache(factory,tmp_path):
    path=tmp_path/'input.cif';path.write_text('old')
    rt=factory([node('a',path=str(path))]);rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    path.write_text('new dataset under same path')
    with pytest.raises(ValueError,match='revised plan'):rt.start(1)


@pytest.mark.parametrize('store_kind',['users','task_lines'])
def test_kernel_lock_failure_never_falls_back_to_unlocked_shared_write(tmp_path,monkeypatch,store_kind):
    import fcntl,auth
    path=tmp_path/'shared.json';path.write_text('{}')
    def unavailable(*args):raise OSError('kernel lock unavailable')
    monkeypatch.setattr(fcntl,'flock',unavailable)
    if store_kind=='users':
        with pytest.raises(OSError):
            with auth._file_guard(path):
                pytest.fail('must not enter an unlocked user-state transaction')
    else:
        store=TaskLineStore(str(path))
        with pytest.raises(OSError):store.begin_line('c1',username='alice',conv_id='c1')
    assert path.read_text()=='{}'


def _parallel_native_input(tmp_path, scope, gas):
    """Native-format engineering fixture, no solver/scientific execution."""
    from tests.test_cdft_result_facts import fixture as native_case
    _, template, _ = native_case(tmp_path / ('native-template-' + gas))
    original = scope / ('original-' + gas)
    return original, template


def _parallel_native_result(template, work, gas):
    import shutil
    for directory in ('input', 'input_dir'):
        shutil.copytree(template / directory, work / directory)
    (work / 'results.csv').write_text(
        f'MOF,{gas}_density_mol_L,{gas}_henry_mol_L_atm\na,1.2,1.3\n')
