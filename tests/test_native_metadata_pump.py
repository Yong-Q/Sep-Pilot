"""Natural tick must reconcile approved metadata before uncertain settlement."""
import copy

import pytest

from agents.recovery import RecoveryGate
from agents.state_io import json_transaction
from tests.test_native_receipt_recovery import native_attempt
from tests.test_parallel_workflow import factory


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
def test_tick_reconciles_same_approved_metadata_before_settling_native_uncertain(factory, tmp_path, monkeypatch, tool):
    rt, watch, _receipt = native_attempt(factory, tmp_path, tool)
    before = rt.snapshot()
    ledger = RecoveryGate(path=rt.root/'recovery_state.json').snapshot()
    leases = copy.deepcopy(rt.store.snapshot().get('leases'))
    rt.main.goal_contract.version = 4
    observed = []
    original = rt._native_attempt_receipt_proof
    def proof(*args, **kwargs):
        observed.append(rt.snapshot()['goal_contract']['version'])
        return original(*args, **kwargs)
    monkeypatch.setattr(rt, '_native_attempt_receipt_proof', proof)

    rt.tick(jobs=[watch])

    state = rt.snapshot()
    current = state['nodes']['native']
    assert current['status'] == 'prefinish'
    assert observed and all(version == 4 for version in observed)
    assert state['goal_contract']['version'] == 4
    assert state['goal_reconciliation_history'][-1]['from_version'] == 1
    assert current['token'] == before['nodes']['native']['token']
    assert current['attempt_id'] == before['nodes']['native']['attempt_id']
    assert current['origin_goal_version'] == 1 and current['job_ids'] == ['430']
    assert current['contract']['arguments'] == before['nodes']['native']['contract']['arguments']
    assert rt.store.snapshot().get('leases') == leases
    assert RecoveryGate(path=rt.root/'recovery_state.json').snapshot() == ledger
    rt.tick(jobs=[watch])
    assert len(rt.snapshot()['goal_reconciliation_history']) == 1


@pytest.mark.parametrize('blocker', ['parameters', 'gases', 'method', 'execution_mode', 'unauthorized',
    'approved_plan', 'contract', 'runtime_owner', 'line_owner', 'line_contract',
    'user_paused', 'paused', 'user_change', 'input_changed', 'awaiting_approval', 'pending_patch'])
def test_native_metadata_pump_never_advances_changed_scope_or_user_controls(factory, tmp_path, monkeypatch, blocker):
    rt, watch, _receipt = native_attempt(factory, tmp_path)
    rt.main.goal_contract.version = 4
    if blocker == 'parameters': rt.main.goal_contract.parameters['temperature_K'] = 310
    if blocker == 'gases': rt.main.goal_contract.gases = ['Kr']
    if blocker == 'method': rt.main.goal_contract.method = 'GCMC'
    if blocker == 'execution_mode': rt.main.goal_contract.execution_mode = 'prepare_only'
    if blocker == 'unauthorized': rt.main.goal_contract.execution_authorized = False
    if blocker == 'approved_plan': rt.main.goal_contract.approved_plan_version = 2
    if blocker == 'contract': rt.main.goal_contract.approved_nodes[0]['arguments']['running_steps'] = 999
    if blocker in {'runtime_owner', 'user_paused', 'user_change', 'input_changed'}:
        with json_transaction(rt.store.path) as data:
            state = data['workflows'][rt.workflow_id]
            if blocker == 'runtime_owner': state['username'] = 'foreign'
            if blocker == 'user_paused': state['user_paused'] = True
            if blocker in {'user_change', 'input_changed'}: state['pause_kind'] = blocker
    if blocker.startswith('line_'):
        line = rt._line()
        if blocker == 'line_owner': line['username'] = 'foreign'
        else: line['steps'][0]['arguments']['running_steps'] = 999
        monkeypatch.setattr(rt, '_line', lambda: copy.deepcopy(line))
    if blocker == 'awaiting_approval': rt.main._awaiting_plan_approval = True
    if blocker == 'pending_patch': rt.main._pending_workflow_patch = {'affected_nodes': ['native']}
    before = rt.snapshot()['nodes']['native']

    if blocker == 'line_owner':
        with pytest.raises(PermissionError):
            rt.tick(jobs=[watch])
    else:
        rt.tick(jobs=[watch], paused=blocker == 'paused')

    state = rt.snapshot()
    assert state['goal_contract']['version'] == 1
    assert not state.get('goal_reconciliation_history')
    assert state['nodes']['native']['status'] == 'uncertain'
    assert state['nodes']['native']['attempt_id'] == before['attempt_id']
    assert not state['nodes']['native'].get('native_receipt_reconciliation')


def test_unknown_nonterminal_job_stays_unknown_after_same_scope_metadata_pump(factory, tmp_path):
    rt, watch, _receipt = native_attempt(factory, tmp_path)
    rt.main.goal_contract.version = 4
    watch.update(terminal=False, state='RUNNING')
    with RecoveryGate(path=rt.root/'recovery_state.json').transaction() as ledger:
        ledger['receipt-key']['status'] = 'submitted'
    before = rt.snapshot()['nodes']['native']
    leases = copy.deepcopy(rt.store.snapshot().get('leases'))
    rt.tick(jobs=[watch])
    state = rt.snapshot()
    assert state['nodes']['native']['status'] == 'uncertain'
    assert state['nodes']['native']['job_ids'] == []
    assert state['nodes']['native']['token'] == before['token']
    assert rt.store.snapshot().get('leases') == leases


@pytest.mark.parametrize('change', ['saved_digest', 'pause', 'owner', 'plan'])
def test_native_metadata_reconciliation_rechecks_authority_at_commit(factory, tmp_path, monkeypatch, change):
    rt, _watch, _receipt = native_attempt(factory, tmp_path)
    rt.main.goal_contract.version = 4
    before = rt.snapshot()
    original = rt.store.reconcile_goal
    def changed_before_commit(*args, **kwargs):
        with json_transaction(rt.store.path) as data:
            state = data['workflows'][rt.workflow_id]
            if change == 'saved_digest': state['goal_contract']['version'] = 2
            if change == 'pause': state['user_paused'] = True
            if change == 'owner': state['username'] = 'foreign'
            if change == 'plan': state['plan_version'] = 2
        return original(*args, **kwargs)
    monkeypatch.setattr(rt.store, 'reconcile_goal', changed_before_commit)
    assert rt._reconcile_native_goal_metadata(before) is False
    state = rt.snapshot()
    assert state['goal_contract']['version'] != 4
    assert not state.get('goal_reconciliation_history')
    assert state['nodes']['native']['attempt_id'] == before['nodes']['native']['attempt_id']
