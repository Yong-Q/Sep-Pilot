"""Source-bound failed report retries never publish, accept, or resubmit science."""
import copy
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
import pytest
from agents import report_bounded_retry as retry
from agents.config import AgentConfig
from agents.failure_recovery import _digest, diagnostic_revision
from agents.parallel_workflow import WorkflowStore
from agents.state_io import json_transaction
from agents.watch_context import set_context
from agents.workflow_patch import execution_contract


def native_call(root, call_id, step, tool, args, result, failed):
    now = time.time(); evidence = root / 'evidence' / (call_id + '.json'); evidence.parent.mkdir(exist_ok=True)
    text = json.dumps(result, ensure_ascii=False)
    call = {'call_id': call_id, 'time': now, 'agent': 'communicator' if failed else 'analyst',
            'tool': tool, 'params': copy.deepcopy(args), 'result': text, 'failed': failed}
    evidence.write_text(json.dumps(call))
    checkpoint = root / 'workers' / step / 'session_checkpoint.json'; checkpoint.parent.mkdir(parents=True, exist_ok=True)
    record = {k: call[k] for k in ('call_id', 'time', 'agent', 'tool', 'params', 'failed')}
    record.update(evidence_path=str(evidence), result_preview=text[:500], result_sha256=hashlib.sha256(text.encode()).hexdigest())
    checkpoint.write_text(json.dumps({'owner': {'username': 'owner', 'conv_id': 'case', 'step_id': step},
        'checkpoint_at': now + .01, 'memory': {'tool_call_log': [record]}}))
    return call, {'call_id': call_id, 'evidence_path': str(evidence), 'checkpoint_path': str(checkpoint)}


@pytest.fixture
def candidate(tmp_path):
    root = tmp_path / 'runs' / 'owner' / 'case'; root.mkdir(parents=True)
    csv = root / 'source.csv'; csv.write_text('material,value\nfixture,2.0\n')
    source_contract = {'step_id': 'source', 'agent': 'analyst', 'tool': 'controlled_analysis',
        'arguments': {'output_csv': str(csv)}, 'depends_on': [], 'expected_outputs': [str(csv)]}
    source_result = {'ok': True, 'output_csv': str(csv), 'output_files': [str(csv)]}
    source_call, source_ref = native_call(root, 'sourcecall', 'source', source_contract['tool'],
                                         source_contract['arguments'], source_result, False)
    report_contract = {'step_id': 'report', 'agent': 'communicator', 'tool': 'generate_scientific_report',
        'arguments': {'source_steps': ['source'], 'output_path': str(root / 'report.md'), 'title': 'Controlled report'},
        'depends_on': ['source'], 'expected_outputs': [str(root / 'report.md')]}
    failure = {'error': 'controlled rejected draft', 'error_kind': 'report_authoring_failed',
               'executed': False, 'report_published': False, 'scientific_acceptance': False}
    call, ref = native_call(root, 'failedcall', 'report', report_contract['tool'], report_contract['arguments'], failure, True)
    goal = {'version': 1, 'approved_plan_version': 1, 'execution_authorized': True, 'execution_mode': 'workflow',
            'method': 'GCMC', 'gases': ['CO2'], 'parameters': {}, 'directives': ['original'],
            'approved_nodes': [copy.deepcopy(source_contract), copy.deepcopy(report_contract)]}
    source = {'contract': source_contract, 'status': 'succeeded', 'phase': 'finish', 'token': 'a' * 32,
              'attempt_id': None, 'job_ids': [], 'result': source_result, 'result_ref': source_ref,
              'effective_arguments': source_contract['arguments'], 'execution_fingerprint': 'source-fingerprint',
              'artifacts': {str(csv): {'size': csv.stat().st_size}}}
    node = {'contract': report_contract, 'status': 'failed', 'token': 'b' * 32, 'owner_pid': 999999999,
            'attempt_id': None, 'job_ids': [], 'origin_goal_version': 1, 'execution_plan_version': 1,
            'dispatch_phase': 'entered_tool', 'tool_returned': True, 'started_at': call['time'] - 1,
            'effective_arguments': report_contract['arguments'], 'result': failure, 'result_ref': ref}
    incident = {'incident_id': 'incident', 'owner': {'username': 'owner', 'conv_id': 'case'},
        'workflow_id': 'workflow', 'step_id': 'report', 'token': node['token'], 'attempt_id': None,
        'goal_version': 1, 'plan_version': 1, 'contract_sha256': _digest(execution_contract(report_contract)),
        'phase': 'blocked', 'diagnostic_revision': diagnostic_revision('generate_scientific_report'),
        'history': [{'diagnostic_revision': '0' * 64}]}
    event = {'event_id': 'failed-event', 'kind': 'worker_failed', 'payload': {'runtime_id': 'workflow',
        'workflow_id': 'case', 'step_id': 'report', 'correlation_token': node['token'], 'attempt_id': None,
        'plan_version': 1, 'origin_goal_version': 1, 'result_ref': copy.deepcopy(ref), 'evidence_call': {'call_id': 'failedcall'}}}
    state = {'username': 'owner', 'conv_id': 'case', 'plan_version': 1, 'status': 'needs_user', 'pause_kind': 'failure',
             'goal_contract': copy.deepcopy(goal), 'nodes': {'source': source, 'report': node},
             'outbox': {'failed-event': event}, 'failure_recovery': {'incidents': {'incident': incident}}}
    class Goal:
        def to_dict(self): return copy.deepcopy(goal)
    project = Path(__file__).resolve().parents[1]
    actor = SimpleNamespace(goal_contract=Goal(), config=AgentConfig(api_key='controlled', project_root=project),
        _pending_workflow_patch=None, _awaiting_plan_approval=False, _pending_user_interaction=None, _current_line_id='case')
    path = tmp_path / 'workflow.json'; path.write_text(json.dumps({'workflows': {'workflow': state}, 'leases': {}}))
    store = WorkflowStore(path)
    runtime = SimpleNamespace(username='owner', conv_id='case', workflow_id='workflow', root=root,
        main=actor, futures={}, job_watch=None, store=store)
    runtime.snapshot = lambda: store.snapshot('workflow')
    set_context('owner', 'case', 'lead-orchestrator', 'case')
    return runtime, goal, csv, ref


def capture(runtime):
    return retry.capture_report_retry(runtime, runtime.snapshot(), 'report', control_valid=lambda: True, return_packet=True)


def commit(runtime, goal, proof, packet, control=lambda: True):
    return runtime.store.requeue_bounded_report('workflow', 'report', 'b' * 32, goal,
        {'username': 'owner', 'conv_id': 'case'}, proof,
        lambda state: retry.capture_report_retry(runtime, state, 'report', control_valid=control, prepared_packet=packet))


def test_real_native_proof_requeues_only_report_and_keeps_failure(candidate):
    runtime, goal, _, _ = candidate; before = runtime.snapshot(); proof, packet = capture(runtime)
    assert proof['prior_implementation_identity'] == 'unknown' and proof['fix_effectiveness'] == 'unknown'
    assert proof['no_scientific_resubmission'] and proof['report_not_published']
    import agents.report_authoring as report
    import sys
    builder_calls = []
    def profile(frame, event, arg):
        if event == 'call' and frame.f_code is report.build_result_packet.__code__: builder_calls.append(True)
    evidence_before = set((runtime.root / 'evidence').iterdir())
    sys.setprofile(profile)
    try: receipt = commit(runtime, goal, proof, packet)
    finally: sys.setprofile(None)
    assert not builder_calls
    after = runtime.snapshot(); node = after['nodes']['report']
    assert receipt['resubmitted'] is False and node['status'] == 'pending' and not node.get('token')
    assert node['attempt_history'][-1]['result'] == before['nodes']['report']['result']
    assert node['report_retry_epochs'][0]['capability_revision'] == proof['capability_revision']
    assert after['nodes']['source'] == before['nodes']['source'] and not (runtime.root / 'report.md').exists()
    assert set((runtime.root / 'evidence').iterdir()) == evidence_before


@pytest.mark.parametrize('change', ['source_bytes', 'target_exists', 'failure_receipt', 'token', 'pause',
                                   'goal', 'lease', 'owner_alive', 'control', 'event_token', 'event_duplicate',
                                   'event_call', 'event_plan', 'not_returned', 'foreign_owner', 'same_epoch',
                                   'budget_exhausted', 'pending_question', 'target_symlink', 'event_ref',
                                   'event_attempt', 'prior_epoch_missing', 'already_published',
                                   'historical_epoch', 'historical_budget'])
def test_native_retry_revalidates_scope_and_never_overwrites(candidate, change):
    runtime, goal, csv, ref = candidate; proof, packet = capture(runtime)
    if change == 'source_bytes': csv.write_text('material,value\nfixture,3.0\n')
    elif change == 'target_exists': (runtime.root / 'report.md').write_text('keep existing')
    elif change == 'target_symlink': (runtime.root / 'report.md').symlink_to(csv)
    elif change == 'failure_receipt': Path(ref['evidence_path']).write_text('{}')
    elif change == 'goal': goal['parameters']['temperature_K'] = 308
    elif change == 'pending_question': runtime.main._pending_user_interaction = {'tool': 'ask_user'}
    with json_transaction(runtime.store.path) as data:
        state = data['workflows']['workflow']; node = state['nodes']['report']
        if change == 'token': node['token'] = 'c' * 32
        elif change == 'pause': state['user_paused'] = True
        elif change == 'owner_alive': node['owner_pid'] = os.getpid()
        elif change == 'lease': data['leases']['lease'] = {'workflow_id': 'workflow', 'step_id': 'report', 'resources': []}
        elif change == 'event_token': state['outbox']['failed-event']['payload']['correlation_token'] = 'c' * 32
        elif change == 'event_duplicate': state['outbox']['duplicate'] = copy.deepcopy(state['outbox']['failed-event'])
        elif change == 'event_call': state['outbox']['failed-event']['payload']['evidence_call']['call_id'] = 'foreign'
        elif change == 'event_plan': state['outbox']['failed-event']['payload']['plan_version'] = 9
        elif change == 'event_ref': state['outbox']['failed-event']['payload']['result_ref']['call_id'] = 'foreign'
        elif change == 'event_attempt': state['outbox']['failed-event']['payload']['attempt_id'] = 'foreign'
        elif change == 'not_returned': node['tool_returned'] = False
        elif change == 'foreign_owner': state['username'] = 'foreign'
        elif change == 'same_epoch': node['report_retry_epochs'] = [{'capability_revision': proof['capability_revision']}]
        elif change == 'budget_exhausted': node['report_retry_epochs'] = [{'capability_revision': 'old1'}, {'capability_revision': 'old2'}]
        elif change == 'historical_epoch': node['attempt_history'] = [{'report_retry_epochs': [{'capability_revision': proof['capability_revision']}]}]
        elif change == 'historical_budget': node['attempt_history'] = [{'report_retry_epochs': [{'capability_revision': 'old1'}, {'capability_revision': 'old2'}]}]
        elif change == 'prior_epoch_missing': state['failure_recovery']['incidents']['incident']['history'] = []
        elif change == 'already_published': node['result']['report_published'] = True
    with pytest.raises((ValueError, PermissionError, OSError)):
        commit(runtime, goal, proof, packet, control=(lambda: change != 'control'))
    assert runtime.snapshot()['nodes']['report']['status'] == 'failed'
    if change == 'target_exists': assert (runtime.root / 'report.md').read_text() == 'keep existing'


def test_two_concurrent_controllers_cannot_requeue_same_attempt_twice(candidate):
    runtime, goal, _, _ = candidate; proof, packet = capture(runtime)
    def run():
        set_context('owner', 'case', 'lead-orchestrator', 'case')
        try: return bool(commit(runtime, goal, proof, packet))
        except (ValueError, PermissionError, OSError): return False
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: run(), range(2)))
    assert sorted(results) == [False, True]
    node = runtime.snapshot()['nodes']['report']
    assert node['status'] == 'pending' and len(node['attempt_history']) == 1 and len(node['report_retry_epochs']) == 1


def test_user_control_revoked_during_packet_build_never_requeues(candidate, monkeypatch):
    runtime, _, _, _ = candidate
    import agents.report_authoring as report
    allowed = {'value': True}; original = report._read
    def revoke_after_read(*args, **kwargs):
        result = original(*args, **kwargs); allowed['value'] = False; return result
    monkeypatch.setattr(report, '_read', revoke_after_read)
    with pytest.raises(ValueError, match='control|superseded|changed'):
        retry.capture_report_retry(runtime, runtime.snapshot(), 'report', control_valid=lambda: allowed['value'])
    assert runtime.snapshot()['nodes']['report']['status'] == 'failed'
    assert not (runtime.root / 'report.md').exists()


def test_terminal_job_snapshot_rejects_owner_tamper_and_atomic_replacement(tmp_path, monkeypatch):
    watch = tmp_path / 'watch.json'
    row = {'job_id': '42', 'username': 'owner', 'conv_id': 'case', 'attempt_id': 'attempt',
           'recovery_key': 'key', 'tool': 'run_cdft', 'terminal': True, 'failed': False, 'state': 'COMPLETED'}
    watch.write_text(json.dumps({'42': row}))
    runtime = SimpleNamespace(username='owner', conv_id='case', job_watch=SimpleNamespace(_watch_file=str(watch)))
    state = {'nodes': {'report': {'contract': {'depends_on': ['source']}},
        'source': {'status': 'succeeded', 'contract': {'tool': 'run_cdft'}, 'job_ids': ['42'],
                   'attempt_id': 'attempt', 'recovery_key': 'key'}}}
    assert retry._source_state_and_jobs(runtime, state, 'report')['jobs']['42']['state'] == 'COMPLETED'
    changed = copy.deepcopy(row); changed['username'] = 'foreign'; watch.write_text(json.dumps({'42': changed}))
    with pytest.raises(ValueError, match='original owned successful attempt'):
        retry._source_state_and_jobs(runtime, state, 'report')
    watch.write_text(json.dumps({'42': row})); replacement = tmp_path / 'replacement.json'
    replacement.write_text(json.dumps({'42': row}))
    original_stat = os.stat; switched = {'value': False}
    def replace_during_read(path, *args, **kwargs):
        if isinstance(path, (str, os.PathLike)) and Path(path) == watch and not switched['value']:
            switched['value'] = True; os.replace(replacement, watch)
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(os, 'stat', replace_during_read)
    with pytest.raises(ValueError, match='job watch changed'):
        retry._source_state_and_jobs(runtime, state, 'report')
