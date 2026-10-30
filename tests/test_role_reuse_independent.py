"""Independent native charge identity and restored branch cleanup checks.

All files/receipts are synthetic pytest-owned fixtures. No scientific runner,
scheduler, model, or production workflow is called.
"""
import copy
import json
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

from agents.parallel_workflow import WorkflowStore, execution_fingerprint
from agents.report_recovery import verify_returned_readonly
from agents.state_io import json_transaction, write_checkpoint
from tests.test_parallel_workflow import factory, node
from tests.test_report_recovery import build_returned_readonly_case


def completed_charge(factory, tmp_path):
    root = tmp_path / 'runs/tester/charge-role'
    source, output = root / 'input', root / 'charged'
    spec = node('charge', tool='run_pacman_charge', args={
        'cif_dir': str(source), 'method': 'pacmof', 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 1}])
    runtime = factory([spec], cid='charge-role')
    source.mkdir(); output.mkdir()
    (source / 'fixture.cif').write_text('synthetic stable source\n')
    (output / 'fixture_pacmof.cif').write_text('synthetic validated charge result\n')
    runtime.start(1)
    contract = runtime.snapshot()['nodes']['charge']['contract']
    receipt = {'submitted': True, 'job_id': '771', 'output_dir': str(output),
               'charged_cifs': [str(output / 'fixture_pacmof.cif')]}
    call_id = 'independentNativeCharge'
    evidence_path = runtime.root / 'evidence' / (call_id + '.json')
    write_checkpoint(evidence_path, {'call_id': call_id, 'time': 1, 'agent': 'analyst',
        'tool': 'run_pacman_charge', 'params': contract['arguments'],
        'result': json.dumps(receipt), 'failed': False})
    with json_transaction(runtime.store.path) as data:
        workflow = data['workflows'][runtime.workflow_id]
        current = workflow['nodes']['charge']
        current.update(status='succeeded', phase='finish', token='originalChargeToken',
            attempt_id='originalChargeAttempt', job_ids=['771'], result=receipt,
            result_ref={'call_id': call_id, 'evidence_path': str(evidence_path)},
            node_verification={'owner': 'analyst', 'evidence_call_ids': ['priorNativeValidation']},
            verification_owner='analyst', execution_plan_version=1,
            origin_goal_version=runtime.main.goal_contract.version,
            execution_fingerprint=execution_fingerprint(contract, runtime.main))
        current['artifacts'] = runtime._artifacts(contract, receipt)
        workflow['status'] = 'completed'
    runtime.task_lines.upsert_step(runtime.main._current_line_id, 'charge',
        username=runtime.username, conv_id=runtime.conv_id, status='completed', done=True,
        job_ids=['771'], validation={'output_contract': 'passed'})
    runtime.job_watch = SimpleNamespace(list=lambda: [], get=lambda job_id: {
        'job_id': job_id, 'username': runtime.username, 'conv_id': runtime.conv_id,
        'terminal': True, 'failed': False, 'state': 'COMPLETED',
    })
    return runtime, spec, evidence_path


def approve_charge_role_change(runtime, spec):
    revised = copy.deepcopy(spec)
    revised['agent'] = 'an'  # Existing registered analyst alias, with charge capability.
    runtime.task_lines.apply_workflow_patch(runtime.main._current_line_id,
        [{'operation': 'upsert', 'step_id': 'charge', 'node': revised}],
        1, 2, 'approved assignment change, identical scientific computation')
    runtime.main.goal_contract.approved_plan_version = 2
    runtime.main.goal_contract.approved_nodes = [revised]


def test_charge_assignment_alias_never_dispatches_a_second_scientific_attempt(factory, tmp_path, monkeypatch):
    runtime, spec, evidence_path = completed_charge(factory, tmp_path)
    before = copy.deepcopy(runtime.snapshot()['nodes']['charge'])
    evidence_before = evidence_path.read_bytes()
    would_dispatch = []
    def fake_submit(function, step_id, ticket):
        would_dispatch.append(step_id)
        future = Future(); future.set_result(None)
        return future
    monkeypatch.setattr(runtime.pool, 'submit', fake_submit)
    approve_charge_role_change(runtime, spec)
    runtime.start(2); runtime.tick()
    after = runtime.snapshot()['nodes']['charge']
    assert would_dispatch == []
    assert after['status'] == 'succeeded'
    for key in ('token', 'attempt_id', 'job_ids', 'result', 'result_ref', 'artifacts',
                'execution_fingerprint', 'node_verification', 'verification_owner'):
        assert after[key] == before[key]
    assert after['contract']['agent'] == 'an'
    assert after['execution_producer']['agent'] == 'analyst'
    assert evidence_path.read_bytes() == evidence_before
    assert runtime.store.claim(runtime.workflow_id, 'charge') is None


def test_successful_but_changed_native_charge_receipt_cannot_reuse_or_requeue(factory, tmp_path):
    runtime, spec, evidence_path = completed_charge(factory, tmp_path)
    approve_charge_role_change(runtime, spec)
    record = json.loads(evidence_path.read_text())
    changed = json.loads(record['result'])
    changed['changed_scientific_summary'] = 'another successful receipt'
    record['result'] = json.dumps(changed)
    evidence_path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        runtime.start(2)
    assert runtime.snapshot()['nodes']['charge']['status'] == 'succeeded'
    assert runtime.snapshot()['nodes']['charge']['token'] == 'originalChargeToken'


def test_new_user_pause_between_role_proof_and_activation_is_not_cleared(factory, tmp_path, monkeypatch):
    runtime, spec, _evidence_path = completed_charge(factory, tmp_path)
    approve_charge_role_change(runtime, spec)
    activate = runtime.store.activate
    def pause_before_commit(*args, **kwargs):
        runtime.store.user_pause(runtime.workflow_id)
        return activate(*args, **kwargs)
    monkeypatch.setattr(runtime.store, 'activate', pause_before_commit)
    with pytest.raises(ValueError):
        runtime.start(2)
    assert runtime.snapshot()['user_paused'] is True
    assert runtime.snapshot()['nodes']['charge']['contract']['agent'] == 'analyst'


def test_restored_finish_removes_only_its_own_obsolete_branch_blocker(tmp_path):
    case = build_returned_readonly_case(tmp_path)
    workflow = case['state']
    node = workflow['nodes']['qc']
    node.update(owner_pid=2147483647, dispatch_phase='entered_tool')
    sibling = {'step_id': 'sibling', 'agent': 'analyst', 'tool': 'read_file',
        'arguments': {'path': str(case['root'] / 'other.txt')}, 'depends_on': [], 'expected_outputs': []}
    workflow['nodes']['sibling'] = {'status': 'failed', 'contract': sibling}
    workflow['goal_contract']['approved_nodes'].append(copy.deepcopy(sibling))
    workflow.update(status='needs_user', pause_kind='failure', branch_blockers={
        'qc': {'status': 'uncertain', 'reason': 'obsolete local dispatch uncertainty'},
        'sibling': {'status': 'failed', 'reason': 'independent unresolved failure'},
    })
    store = WorkflowStore(tmp_path / 'workflow.json')
    workflow_id = store.identity('owner', 'conv')
    write_checkpoint(store.path, {'workflows': {workflow_id: workflow}, 'leases': {}})
    goal = workflow['goal_contract']
    proof = verify_returned_readonly(workflow, 'qc', case['root'], approved_goal=goal)
    assert store.reconcile_proven_local_result(workflow_id, 'qc', node['token'],
        goal, 'owner', 'conv', case['root'], proof)
    current = store.snapshot(workflow_id)['nodes']['qc']
    store.update(workflow_id, 'qc', current['token'], {'node_verification': {
        'owner': 'analyst', 'evidence_call_ids': ['currentValidationDossier'],
    }})
    assert store.finish_proven_local_result(workflow_id, 'qc', current['token'], goal,
        'owner', 'conv', case['root'], {'artifacts': current['artifacts'],
            'path_manifest': current['path_manifest'], 'execution_fingerprint': 'fixtureReadOnlyProof'})
    after = store.snapshot(workflow_id)
    assert after['nodes']['qc']['status'] == 'succeeded'
    assert 'qc' not in after['branch_blockers']
    assert after['branch_blockers']['sibling'] == {
        'status': 'failed', 'reason': 'independent unresolved failure',
    }
    assert after['nodes']['sibling']['status'] == 'failed'
