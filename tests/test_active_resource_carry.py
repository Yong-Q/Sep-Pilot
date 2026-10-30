import copy
import json

import pytest

from agents.parallel_workflow import WorkflowStore
from tests.test_parallel_workflow import factory


@pytest.fixture
def active_case(tmp_path):
    root = str(tmp_path / 'output')
    old = {'step_id': 'n2', 'agent': 'worker', 'tool': 'run_cdft',
           'arguments': {'action': 'submit', 'gas': 'N2', 'input_dir': str(tmp_path / 'input')},
           'depends_on': ['input'], 'expected_outputs': [{'type': 'file', 'path': root + '/data/time/results.csv'}],
           'resources': [{'key': 'path:' + str(tmp_path / 'input'), 'mode': 'read'},
                         {'key': 'path:' + root, 'mode': 'write'},
                         {'key': 'path:' + root + '/results.csv', 'mode': 'write'}]}
    prior = {'contract': old, 'status': 'prefinish', 'phase': 'prefinish', 'token': 'original-token',
             'job_ids': ['110099'], 'tool_returned': True, 'result_ref': {'call_id': 'original-native'},
             'node_verification': {'owner': 'original-producer'}}
    lease = {'workflow_id': 'workflow', 'step_id': 'n2', 'owner_pid': 123,
             'resources': copy.deepcopy(old['resources'])}
    data = {'workflows': {'workflow': {'username': 'owner', 'conv_id': 'conv', 'plan_version': 1,
             'goal_contract': {'version': 1}, 'nodes': {'n2': prior}, 'user_paused': True,
             'user_pause_revision': 7}}, 'leases': {'original-token': lease}}
    incoming = copy.deepcopy(old)
    incoming['resources'][-1]['key'] = 'path:' + root + '/data/time/results.csv'
    store = WorkflowStore(tmp_path / 'state.json')
    return store, data, incoming


def activate(case):
    store, data, incoming = case
    store.path.write_text(json.dumps(data))
    return store.activate('workflow', [incoming], {'version': 2, 'approved_plan_version': 2},
                          'owner', 'conv', pause_control={'paused': True, 'revision': 7})


def test_derived_child_output_carries_exact_native_node_and_token_keyed_lease(active_case):
    store, data, incoming = active_case
    original_incoming = copy.deepcopy(incoming)
    result = activate(active_case)
    assert result['nodes']['n2'] == data['workflows']['workflow']['nodes']['n2']
    assert result['user_paused'] is True and result['user_pause_revision'] == 7
    assert result['plan_version'] == 2
    assert store.snapshot()['leases'] == data['leases']
    assert incoming == original_incoming


@pytest.mark.parametrize('change', ['arguments', 'outputs', 'outside', 'read_upgrade', 'missing_lease',
                                  'wrong_key', 'wrong_step', 'wrong_resources', 'duplicate_lease', 'foreign_lease'])
def test_active_carry_rejects_changed_science_or_unproven_lease(active_case, change):
    store, data, incoming = active_case
    lease = data['leases']['original-token']
    if change == 'arguments': incoming['arguments']['gas'] = 'CO2'
    elif change == 'outputs': incoming['expected_outputs'][0]['path'] += '.different'
    elif change == 'outside': incoming['resources'][-1]['key'] = 'path:' + str(store.path.parent / 'outside')
    elif change == 'read_upgrade': incoming['resources'][0]['mode'] = 'write'
    elif change == 'missing_lease': data['leases'].clear()
    elif change == 'wrong_key': data['leases']['other-token'] = data['leases'].pop('original-token')
    elif change == 'wrong_step': lease['step_id'] = 'other'
    elif change == 'wrong_resources': lease['resources'] = []
    elif change == 'duplicate_lease': data['leases']['duplicate'] = copy.deepcopy(lease)
    elif change == 'foreign_lease': data['leases']['foreign'] = {**copy.deepcopy(lease), 'workflow_id': 'other'}
    with pytest.raises(ValueError, match='resolve running/unknown'):
        activate(active_case)
    assert store.snapshot() == data


@pytest.fixture
def native_case(tmp_path):
    from agents.session import SessionMemory
    from agents.state_io import compact_checkpoint_evidence, write_checkpoint
    root = tmp_path / 'runs' / 'owner' / 'conv'
    root.mkdir(parents=True)
    (root / 'output').mkdir(); (root / 'output' / 'results.csv').write_text('material,density\nA,1\n')
    old = {'step_id': 'n2', 'agent': 'analyst', 'tool': 'run_cdft',
           'arguments': {'action': 'submit', 'gas': 'N2', 'memory_mb': 16384,
                         'resource_review_id': 'planning', 'num_processes': 49,
                         'input_dir': str(root / 'input'), 'job_work_dir': str(root / 'output')},
           'depends_on': [], 'expected_outputs': [{'kind': 'file', 'path': str(root / 'output' / 'results.csv')}],
           'resources': [{'key': 'path:' + str(root / 'input'), 'mode': 'read'},
                         {'key': 'path:' + str(root / 'output'), 'mode': 'write'}]}
    effective = {**old['arguments'], 'memory_mb': 50688, 'resource_review_id': 'actual'}
    result = {'submitted': True, 'job_id': '77', 'work_dir': str(root / 'output')}
    memory = SessionMemory()
    memory.record_tool_call('resource', 'resource_health', {}, json.dumps({
        'status': 'ready', 'resource_review_id': 'actual',
        'suggested_resources': {'memory_mb': 50688, 'num_processes': 49}}), failed=False)
    review_call_id = memory.tool_call_log[-1]['call_id']
    memory.record_tool_call('analyst', 'run_cdft', effective, json.dumps(result), failed=False)
    call = memory.tool_call_log[-1]
    checkpoint = {'owner': {'username': 'owner', 'conv_id': 'conv', 'step_id': 'n2'},
                  'checkpoint_at': call['time'] + 1, 'memory': {'tool_call_log': memory.tool_call_log}}
    compact_checkpoint_evidence(checkpoint, root / 'evidence')
    checkpoint_path = root / 'workers' / 'n2' / 'session_checkpoint.json'
    write_checkpoint(checkpoint_path, checkpoint)
    prior = {'contract': old, 'status': 'prefinish', 'phase': 'prefinish', 'token': 'original-token',
             'job_ids': ['77'], 'attempt_id': 'attempt', 'recovery_key': 'recovery', 'tool_returned': True, 'result': result,
             'effective_arguments': effective, 'resource_allocation': {'memory_mb': 50688, 'cpus': 49,
                 'resource_review_id': 'actual', 'budget_final': True, 'evidence_call_ids': [review_call_id]},
             'result_ref': {'call_id': call['call_id'], 'checkpoint_path': str(checkpoint_path),
                            'evidence_path': str(root / 'evidence' / (call['call_id'] + '.json'))}}
    goal = {'schema_version': 2, 'version': 2, 'approved_plan_version': 2, 'execution_authorized': True, 'method': 'CDFT',
            'gases': ['N2'], 'parameters': {}, 'execution_mode': 'workflow'}
    state = {'username': 'owner', 'conv_id': 'conv', 'plan_version': 1,
             'goal_contract': {**goal, 'approved_plan_version': 1}, 'nodes': {'n2': prior}, 'user_paused': True}
    incoming = copy.deepcopy(old); incoming['arguments'] = copy.deepcopy(effective)
    goal['approved_nodes'] = [copy.deepcopy(incoming)]
    jobs = {'77': {'job_id': '77', 'username': 'owner', 'conv_id': 'conv', 'step_id': 'n2',
                   'attempt_id': 'attempt', 'terminal': True, 'failed': False, 'state': 'COMPLETED'}}
    write_checkpoint(root / 'recovery_state.json', {'recovery': {'tool': 'run_cdft', 'params': effective,
        'attempt_id': 'attempt', 'job_ids': ['77'], 'status': 'completed_unverified'}})
    return state, incoming, root, goal, jobs


def test_native_executed_memory_metadata_gets_current_plan_proof(native_case):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches
    state, incoming, root, goal, jobs = native_case
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    assert proof['plan_version'] == 2 and proof['job_ids'] == ['77']
    assert proof_matches(state, 'n2', incoming, root, goal, proof)
    assert state['nodes']['n2']['contract']['arguments']['memory_mb'] == 16384


@pytest.mark.parametrize('change', ['running', 'cpu', 'science', 'memory', 'producer', 'checkpoint', 'job_owner'])
def test_native_budget_carry_rejects_unproven_or_new_budget(native_case, change):
    from agents.native_resource_carry import prove_native_resource_carry
    state, incoming, root, goal, jobs = native_case
    if change == 'running': jobs['77'].update(terminal=False, state='RUNNING')
    elif change == 'cpu': incoming['arguments']['num_processes'] = 48
    elif change == 'science': incoming['arguments']['gas'] = 'CO2'
    elif change == 'memory': incoming['arguments']['memory_mb'] += 1
    elif change == 'producer':
        path = __import__('pathlib').Path(state['nodes']['n2']['result_ref']['evidence_path'])
        call = json.loads(path.read_text()); call['params']['memory_mb'] += 1; path.write_text(json.dumps(call))
    elif change == 'checkpoint': __import__('pathlib').Path(state['nodes']['n2']['result_ref']['checkpoint_path']).write_text('{}')
    elif change == 'job_owner': jobs['77']['conv_id'] = 'foreign'
    with pytest.raises((ValueError, PermissionError)):
        prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)


@pytest.mark.parametrize('status,phase', [('prefinish', 'prefinish'), ('succeeded', 'finish')])
def test_store_native_carry_preserves_original_node_without_reexecution(native_case, status, phase):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches, approved_graph_matches
    state, incoming, root, goal, jobs = native_case
    state['nodes']['n2'].update(status=status, phase=phase)
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    leases = {} if status == 'succeeded' else {'original-token': {'workflow_id': 'workflow', 'step_id': 'n2',
               'resources': copy.deepcopy(state['nodes']['n2']['contract']['resources']), 'owner_pid': 123}}
    store = WorkflowStore(root / 'store.json'); store.path.write_text(json.dumps({'workflows': {'workflow': state}, 'leases': leases}))
    result = store.activate('workflow', [incoming], goal, 'owner', 'conv',
        native_resource_proofs={'n2': proof}, native_resource_validator=lambda s, k, n, p: proof_matches(s, k, n, root, goal, p))
    carried = copy.deepcopy(result['nodes']['n2']); carried.pop('approved_resource_carry')
    assert carried == state['nodes']['n2']
    assert store.snapshot()['leases'] == leases
    assert approved_graph_matches(result, [incoming], root, goal)
    altered = copy.deepcopy(incoming); altered['arguments']['memory_mb'] += 1
    assert not approved_graph_matches(result, [altered], root, goal)


def test_native_carry_ledger_identity_survives_other_jobs_and_acceptance_only(native_case):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches
    state, incoming, root, goal, jobs = native_case
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    path = root / 'recovery_state.json'; ledger = json.loads(path.read_text())
    ledger['other-job'] = {'status': 'running'}
    ledger['recovery']['status'] = 'completed'
    path.write_text(json.dumps(ledger))
    assert proof_matches(state, 'n2', incoming, root, goal, proof)
    assert not proof_matches(state, 'n2', incoming, root, goal, proof, check_ledger=False)
    ledger['recovery']['params']['memory_mb'] += 1
    path.write_text(json.dumps(ledger))
    assert not proof_matches(state, 'n2', incoming, root, goal, proof)


def test_native_carry_final_stat_rejects_same_size_producer_tamper(native_case):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches
    state, incoming, root, goal, jobs = native_case
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    path = __import__('pathlib').Path(state['nodes']['n2']['result_ref']['evidence_path'])
    original = path.stat(); path.write_bytes(path.read_bytes().replace(b'50688', b'50689'))
    __import__('os').utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    assert not proof_matches(state, 'n2', incoming, root, goal, proof)


@pytest.mark.parametrize('tamper', [False, True])
def test_real_start_carries_completed_budget_metadata_and_can_resume_same_plan(native_case, factory, tamper):
    from types import SimpleNamespace
    from agents.goal_contract import GoalContract
    from tests.test_parallel_workflow import node
    state, incoming, root, goal, jobs = native_case
    runtime = factory([node('placeholder')], cid='other', username='owner')
    runtime.root = root; runtime.conv_id = 'conv'; runtime.workflow_id = 'workflow'
    runtime.main.goal_contract = GoalContract.from_dict({**goal, 'approved_nodes': [incoming]})
    runtime.main._on_resource_review = None
    state['nodes']['n2'].update(status='succeeded', phase='finish', node_verification={'owner': 'analyst'})
    state['nodes']['n2']['artifacts'] = runtime._artifacts(state['nodes']['n2']['contract'], state['nodes']['n2']['result'])
    runtime.store.path.parent.mkdir(parents=True, exist_ok=True)
    runtime.store.path.write_text(json.dumps({'workflows': {'workflow': state}, 'leases': {}}))
    line = {'username': 'owner', 'conv_id': 'conv', 'plan_version': 2,
            'steps': [{**incoming, 'status': 'completed', 'done': True}]}
    runtime._line = lambda: copy.deepcopy(line)
    runtime.job_watch = SimpleNamespace(get=lambda job: jobs[str(job)], list=lambda: list(jobs.values()))
    if tamper:
        path = __import__('pathlib').Path(state['nodes']['n2']['result_ref']['evidence_path'])
        path.write_text('{}')
        with pytest.raises(ValueError): runtime.start(2, preserve_user_pause=True)
        assert runtime.snapshot()['nodes']['n2'] == state['nodes']['n2']
        return
    runtime.start(2, preserve_user_pause=True)
    first = runtime.snapshot()['nodes']['n2']
    runtime.start(2, preserve_user_pause=True)
    assert runtime.snapshot()['nodes']['n2'] == first
    assert first['status'] == 'succeeded' and first['contract'] == state['nodes']['n2']['contract']
    assert runtime.snapshot()['user_paused'] is True


def test_carry_view_keeps_native_status_and_unrelated_bound_output_does_not_invalidate(native_case):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches
    from agents.workflow_view import authoritative_line, workflow_completion
    state, incoming, root, goal, jobs = native_case
    state['nodes']['n2'].update(status='succeeded', phase='finish', node_verification={'owner': 'analyst'})
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    state['nodes']['n2']['approved_resource_carry'] = proof
    state['plan_version'] = 2; state['goal_contract'] = copy.deepcopy(goal)
    line = {'username': 'owner', 'conv_id': 'conv', 'plan_version': 2, 'steps': [incoming]}
    view = authoritative_line(line, state)
    assert view['steps'][0]['contract_matches_execution'] is True
    assert view['steps'][0]['done'] is True
    assert workflow_completion(line, state, [incoming], 2)['ok'] is True
    rebound = {**goal, 'approved_nodes': [incoming, {'step_id': 'unrelated', 'expected_outputs': ['new-timestamp.csv']}]}
    assert proof_matches(state, 'n2', incoming, root, rebound, proof)
    assert not proof_matches(state, 'n2', incoming, root, {**rebound, 'parameters': {'temperature': 310}}, proof)


def test_carry_view_requires_resource_review_evidence_even_with_valid_native_carry(native_case):
    from agents.native_resource_carry import prove_native_resource_carry, proof_matches
    from agents.workflow_view import authoritative_line, workflow_completion
    state, incoming, root, goal, jobs = native_case
    state['nodes']['n2'].update(status='succeeded', phase='finish', node_verification={'owner': 'analyst'})
    state['nodes']['n2']['resource_allocation'].pop('evidence_call_ids')
    proof = prove_native_resource_carry(state, 'n2', incoming, root, goal, jobs)
    state['nodes']['n2']['approved_resource_carry'] = proof
    state['plan_version'] = 2; state['goal_contract'] = copy.deepcopy(goal)
    line = {'username': 'owner', 'conv_id': 'conv', 'plan_version': 2, 'steps': [incoming]}
    assert proof_matches(state, 'n2', incoming, root, goal, proof)
    view = authoritative_line(line, state)
    assert view['steps'][0]['contract_matches_execution'] is True
    assert view['steps'][0]['effective_resource_receipt_status'] == 'unreviewed'
    assert view['steps'][0]['execution_acceptance_status'] == 'effective_resources_unverified'
    assert view['steps'][0]['done'] is False
    completion = workflow_completion(line, state, [incoming], 2)
    assert completion['ok'] is False
    assert completion['blockers'] == [{'step_id': 'n2', 'status': 'effective_arguments_unverified'}]
