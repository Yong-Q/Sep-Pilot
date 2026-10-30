"""Concurrent incident repair routing must be scoped to its native identity."""
import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import pytest
from agents.failure_recovery import FailureRecovery, _digest
from agents.parallel_workflow import WorkflowStore
from agents.workflow_patch import execution_contract


def setup(tmp_path):
    root = tmp_path / 'runs' / 'owner' / 'case'; (root / 'evidence').mkdir(parents=True)
    nodes = [{'step_id': key, 'agent': 'communicator', 'tool': 'generate_scientific_report',
              'arguments': {'output_path': key + '.md'}, 'depends_on': [], 'expected_outputs': [key + '.md']}
             for key in ('one', 'two')]
    goal = {'version': 1, 'execution_authorized': True, 'approved_plan_version': 1,
            'approved_nodes': copy.deepcopy(nodes), 'method': 'GCMC', 'gases': ['CO2'],
            'parameters': {}, 'execution_mode': 'workflow'}
    state = {'username': 'owner', 'conv_id': 'case', 'plan_version': 1, 'goal_contract': copy.deepcopy(goal),
             'status': 'needs_user', 'nodes': {}, 'failure_recovery': {'incidents': {}}, 'outbox': {}}
    class Goal:
        def to_dict(self): return copy.deepcopy(goal)
    path = tmp_path / 'workflows.json'; store = WorkflowStore(path)
    main = SimpleNamespace(goal_contract=Goal(), _current_line_id='case', config=SimpleNamespace(project_root=tmp_path))
    runtime = SimpleNamespace(username='owner', conv_id='case', workflow_id='workflow', root=root,
                              main=main, store=store, futures={}, flush_events=lambda: None)
    runtime.snapshot = lambda: store.snapshot('workflow')
    machine = FailureRecovery(runtime)
    for index, contract in enumerate(nodes, 1):
        key = contract['step_id']; token = str(index) * 32
        state['nodes'][key] = {'contract': copy.deepcopy(contract), 'status': 'failed', 'token': token,
                               'attempt_id': 'attempt-' + key, 'tool_returned': True, 'dispatch_phase': 'entered_tool'}
        incident = {'incident_id': 'incident-' + key, 'owner': {'username': 'owner', 'conv_id': 'case'},
                    'workflow_id': 'workflow', 'step_id': key, 'token': token, 'attempt_id': 'attempt-' + key,
                    'goal_version': 1, 'plan_version': 1, 'contract_sha256': _digest(execution_contract(contract)),
                    'phase': 'repair_required', 'revision': 4, 'epoch': 'epoch-' + key,
                    'last_result': {'status': 'diagnosed'}, 'rounds': {}}
        state['failure_recovery']['incidents'][incident['incident_id']] = incident
        machine._stage_event(state, incident, 'repair', 'repair_planner')
    path.write_text(json.dumps({'workflows': {'workflow': state}, 'leases': {}}))
    def actor():
        return SimpleNamespace(_evidence_root=root / 'evidence', config=SimpleNamespace(project_root=tmp_path),
            _on_checkpoint=lambda *args: None, _checkpoint=lambda *args: None,
            memory=SimpleNamespace(tool_call_log=[]))
    return machine, store, actor, root


def test_two_independent_repair_incidents_can_run_concurrently_without_retry(tmp_path):
    machine, store, actor, root = setup(tmp_path); barrier = threading.Barrier(2); seen = []
    def runner(event):
        seen.append(event['payload']['failure_recovery']['incident_id']); barrier.wait(timeout=5)
        return {'status': 'blocked', 'reason': 'no repair proposal'}
    def run(key):
        return machine.run_repair(actor(), 'incident-' + key, 4, checkpoint_path=root / (key + '.json'), stage_runner=runner)
    before = store.snapshot('workflow')['nodes']
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run, key) for key in ('one', 'two')]
        results = [f.result(timeout=10) for f in futures]
    assert set(seen) == {'incident-one', 'incident-two'}
    assert all(r['phase'] == 'blocked' and r['retry_granted'] is False for r in results)
    assert store.snapshot('workflow')['nodes'] == before


@pytest.mark.parametrize('change', ['token', 'epoch', 'revision', 'duplicate'])
def test_replayed_or_ambiguous_same_incident_is_rejected_before_claim(tmp_path, change):
    machine, store, actor, root = setup(tmp_path)
    from agents.state_io import json_transaction
    with json_transaction(store.path) as data:
        state = data['workflows']['workflow']; event = next(e for e in state['outbox'].values()
            if e['payload']['failure_recovery']['incident_id'] == 'incident-one')
        if change == 'token': event['payload']['correlation_token'] = 'f' * 32
        elif change == 'epoch': event['payload']['failure_recovery']['epoch'] = 'replayed-epoch'
        elif change == 'revision': event['payload']['failure_recovery']['revision'] = 3
        elif change == 'duplicate': state['outbox']['duplicate'] = copy.deepcopy(event)
    with pytest.raises(ValueError, match='unique native delegation event'):
        machine.run_repair(actor(), 'incident-one', 4, checkpoint_path=root / 'one.json', stage_runner=lambda e: pytest.fail('stale event dispatched'))
    assert store.snapshot('workflow')['failure_recovery']['incidents']['incident-one']['phase'] == 'repair_required'


def test_same_incident_concurrent_claim_has_exactly_one_owner(tmp_path):
    machine, store, actor, root = setup(tmp_path); entered = threading.Event(); release = threading.Event()
    def runner(event):
        entered.set(); assert release.wait(5); return {'status': 'blocked'}
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(machine.run_repair, actor(), 'incident-one', 4, checkpoint_path=root / 'one.json', stage_runner=runner)
        assert entered.wait(5)
        with pytest.raises(ValueError, match='stale or already claimed'):
            machine.run_repair(actor(), 'incident-one', 4, checkpoint_path=root / 'other.json', stage_runner=runner)
        release.set(); assert future.result(timeout=10)['retry_granted'] is False
    assert store.snapshot('workflow')['failure_recovery']['incidents']['incident-two']['phase'] == 'repair_required'
