"""A deployed repair gets one diagnostic opportunity without retrying science."""
import copy

import pytest

from agents.recovery import RecoveryGate, input_fingerprint
from agents.state_io import json_transaction
from tests.test_parallel_workflow import factory, node


def failed_attempt(factory, tmp_path):
    source = tmp_path / 'agents/registry.py'
    source.parent.mkdir(exist_ok=True)
    source.write_text('executor_epoch = 1\n')
    args = {'material_type': 'MOF', 'n_structures': 1, 'output_dir': str(tmp_path / 'structures')}
    contract = node('generate', tool='generate_structure', args=args,
        outputs=[{'kind': 'directory', 'path': args['output_dir'], 'pattern': '*.cif', 'min_count': 1}])
    contract['agent'] = 'harness'
    rt = factory([contract])
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'generate')
    gate = RecoveryGate(path=rt.root / 'recovery_state.json')
    attempt, error = gate.claim('failed-generation', 'generate_structure', args,
        input_fingerprint(args, tmp_path), 1, tmp_path)
    assert attempt and not error
    gate.outcome('failed-generation', attempt, {'error': 'required installation data missing'})
    rt.store.update(rt.workflow_id, 'generate', ticket['token'], {
        'status': 'failed', 'tool_returned': True, 'dispatch_phase': 'entered_tool',
        'attempt_id': attempt, 'recovery_key': 'failed-generation',
        'effective_arguments': args, 'job_ids': [], 'error': 'required installation data missing',
    }, release=True)
    return rt, gate, source


def capabilities(rt):
    return [e for e in rt.lifecycle.snapshot()['events'].values()
        if e['kind'] == 'worker_recovery_ready'
        and e['payload'].get('recovery', {}).get('action') == 'verify_changed_executor']


def test_failed_local_input_preparation_without_scheduler_ledger_can_inspect_a_changed_executor(factory, tmp_path):
    import os
    source = tmp_path / 'agents/registry.py'
    source.parent.mkdir(exist_ok=True)
    source.write_text('executor_epoch = 1\n')
    args = {'action': 'inputs', 'cif_dir': str(tmp_path / 'cifs'), 'gas': 'CO2',
        'temperature': 298, 'bulk_densities': [.04], 'input_dir': str(tmp_path / 'inputs')}
    contract = node('inputs', tool='run_cdft', args=args,
        outputs=[{'kind': 'directory', 'path': args['input_dir'], 'pattern': 'input.dat', 'min_count': 1}])
    contract['agent'] = 'analyst'
    rt = factory([contract])
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'inputs')
    rt.store.update(rt.workflow_id, 'inputs', ticket['token'], {'status': 'failed',
        'dispatch_phase': 'entered_tool', 'tool_returned': True,
        'result': {'failed': True, 'error': 'configured forcefield not found'},
        'result_ref': {'call_id': 'actual-input-failure'}, 'effective_arguments': args}, release=True)
    before = copy.deepcopy(rt.snapshot()['nodes']['inputs'])
    source.write_text('executor_epoch = 2\n')
    os.utime(source, (before['started_at'] + 1, before['started_at'] + 1))
    rt.recover_notifications()
    events = capabilities(rt)
    assert len(events) == 1
    assert events[0]['payload']['recovery']['arguments'] == {'tool_name': 'run_cdft', 'arguments': args}
    assert events[0]['payload']['recovery']['retry_granted'] is False
    assert events[0]['payload']['recovery']['retry_control'] == 'approved_workflow_revision'
    assert rt.snapshot()['nodes']['inputs'] == before
    assert not (rt.root / 'recovery_state.json').exists()


def test_deployed_executor_change_emits_one_diagnostic_capability_without_dispatch(factory, tmp_path):
    rt, gate, source = failed_attempt(factory, tmp_path)
    rt.recover_notifications()  # old failure notification already delivered
    assert not capabilities(rt)
    for event in rt.lifecycle.snapshot()['events'].values():
        if event['kind'] == 'worker_failed':
            for receiver in ['main_chat', 'supervisor']:
                rt.lifecycle.receipt(event['event_id'], receiver, 'delivered', {'status': 'handled'})
    before = copy.deepcopy(rt.snapshot()['nodes']['generate'])
    ledger = gate.path.read_bytes()
    source.write_text('executor_epoch = 2\n')

    rt.recover_notifications()
    events = capabilities(rt)
    assert len(events) == 1
    recovery = events[0]['payload']['recovery']
    assert recovery['tool'] == 'inspect_runtime'
    assert recovery['arguments'] == {'tool_name': 'generate_structure', 'arguments': before['effective_arguments']}
    assert recovery['retry_granted'] is False
    assert recovery['retry_control'] == 'prepare_retry'
    assert events[0]['payload']['requires_user'] is False
    assert rt.snapshot()['nodes']['generate'] == before
    assert gate.path.read_bytes() == ledger
    rt.recover_notifications()
    assert len(capabilities(rt)) == 1


@pytest.mark.parametrize('blocked', ['paused', 'unknown', 'live_job', 'exhausted', 'unapproved',
    'other_attempt', 'owner', 'changed_science', 'pending_patch', 'pending_approval', 'ledger_unknown'])
def test_executor_notice_does_not_offer_unsafe_recovery(factory, tmp_path, blocked):
    rt, gate, source = failed_attempt(factory, tmp_path)
    current = rt.snapshot()['nodes']['generate']
    if blocked == 'paused': rt.store.user_pause(rt.workflow_id)
    elif blocked == 'unknown':
        rt.store.update(rt.workflow_id, 'generate', current['token'], {'status': 'uncertain'})
    elif blocked == 'live_job':
        rt.store.update(rt.workflow_id, 'generate', current['token'], {'job_ids': ['42']})
    elif blocked == 'unapproved': rt.main.goal_contract.execution_authorized = False
    elif blocked == 'changed_science': rt.main.goal_contract.parameters['temperature_K'] = 310
    elif blocked == 'pending_patch': rt.main._pending_workflow_patch = {'new_version': 2}
    elif blocked == 'pending_approval': rt.main._awaiting_plan_approval = True
    elif blocked == 'owner':
        with json_transaction(rt.store.path) as data:
            data['workflows'][rt.workflow_id]['username'] = 'different_owner'
    else:
        with gate.transaction() as data:
            if blocked == 'exhausted': data['failed-generation']['attempts'] = 3
            elif blocked == 'ledger_unknown': data['failed-generation']['status'] = 'uncertain'
            else: data['failed-generation']['attempt_id'] = 'different-attempt'
    source.write_text('executor_epoch = 2\n')
    rt.recover_notifications()
    assert not capabilities(rt)
