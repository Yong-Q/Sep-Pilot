"""Receipt reconciliation of an already submitted job never dispatches again."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.parallel_workflow import artifact_base
from agents.recovery import RecoveryGate, input_fingerprint
from agents.state_io import json_transaction, write_checkpoint
from tests.test_parallel_workflow import factory, node


def native_attempt(factory, tmp_path, tool='run_string_tst', *, uncertain=True):
    cid = 'native-receipt'
    approved = tmp_path / 'runs/tester' / cid / 'output'
    cif = tmp_path / 'framework.cif'
    cif.write_text('data_framework\n')
    arguments = {'cif_path': str(cif), 'output_dir': str(approved)}
    if tool == 'run_string_tst':
        arguments.update(gas='CO2', temperature=298., directions=[1], running_steps=20, n_string=11)
        outputs = [{'kind': 'directory', 'path': str(approved), 'pattern': pattern, 'min_count': 1}
            for pattern in ['string_path.dat', 'manifest.csv']]
    else:
        arguments.update(mode='single')
        outputs = [str(approved / 'framework_opt.cif'), str(approved / 'md_result.json')]
    contract = node('native', tool=tool, args=arguments, outputs=outputs)
    contract['agent'] = 'adsorption'
    rt = factory([contract], cid=cid)
    rt.start(1)
    claim = rt.store.claim(rt.workflow_id, 'native')
    actual = approved / 'attempt_native'
    actual.mkdir(parents=True)
    script = actual / 'submit.sh'
    script.write_text('#!/bin/sh\n# native fixture submission\n')
    receipt = {'submitted': True, 'job_id': '430', 'script': str(script),
        'output_dir': str(actual), 'work_dir': str(actual)}
    if tool == 'run_string_tst':
        data = actual / 'string_input_CO2'
        data.mkdir()
        (data / 'manifest.csv').write_text('structure,direction\nframework,1\n')
        (data / 'string_results.json').write_text('{"schema_version":1,"success":true,"results":[]}')
        receipt.update(manifest_path=str(data / 'manifest.csv'), result_json_path=str(data / 'string_results.json'))
    else:
        (actual / 'framework_opt.cif').write_text('data_optimized\n')
        (actual / 'md_result.json').write_text('{"success":true}')
        receipt.update(result_json_path=str(actual / 'md_result.json'), expected_structure_path=str(actual / 'framework_opt.cif'))
    key, attempt = 'receipt-key', 'original-attempt'
    gate = RecoveryGate(path=rt.root / 'recovery_state.json')
    with gate.transaction() as ledger:
        ledger[key] = {'tool': tool, 'params': copy.deepcopy(arguments), 'fingerprint': input_fingerprint(arguments, tmp_path),
            'attempt_id': attempt, 'attempts': 2, 'status': 'completed_unverified', 'job_ids': ['430'],
            'goal_version': 1, 'last_result': receipt}
    call_id = 'originalnativeevidence'
    call = {'call_id': call_id, 'time': claim['started_at'], 'agent': 'adsorption', 'tool': tool,
        'params': copy.deepcopy(arguments), 'result': json.dumps(receipt), 'failed': False}
    evidence = rt.root / 'evidence' / f'{call_id}.json'
    write_checkpoint(evidence, call)
    rt.store.update(rt.workflow_id, 'native', claim['token'], {
        'status': 'uncertain' if uncertain else 'running', 'dispatch_phase': 'entered_tool', 'tool_returned': True,
        'attempt_id': attempt, 'recovery_key': key, 'result': receipt, 'evidence_call': call,
        'result_ref': {'call_id': call_id, 'evidence_path': str(evidence)},
        'error': 'native output receipt differs from the approved session path' if uncertain else ''})
    with json_transaction(rt.store.path) as data:
        w = data['workflows'][rt.workflow_id]
        w.update(status='needs_user' if uncertain else 'active', pause_kind='reconciliation' if uncertain else None)
    watch = {'job_id': '430', 'username': rt.username, 'conv_id': rt.conv_id, 'tool': tool,
        'attempt_id': attempt, 'recovery_key': key, 'work_dir': str(actual),
        'state': 'COMPLETED', 'terminal': True, 'failed': False}
    rt.job_watch = SimpleNamespace(get=lambda jid: copy.deepcopy(watch) if str(jid) == '430' else None,
                                  list=lambda: [copy.deepcopy(watch)])
    rt.main.registry.get(tool).execute = lambda args: pytest.fail('existing job must not be submitted again')
    return rt, watch, receipt


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
def test_owned_fresh_attempt_receipt_rebases_outputs_and_preserves_contract_history(factory, tmp_path, tool):
    rt, watch, receipt = native_attempt(factory, tmp_path, tool, uncertain=False)
    original = copy.deepcopy(rt.snapshot()['nodes']['native']['contract'])
    assert rt._bind_native_result_path('native', rt.snapshot()['nodes']['native']['token'], receipt)
    current = rt.snapshot()['nodes']['native']
    actual = Path(receipt['work_dir'])
    assert artifact_base(current['contract'], tmp_path, rt.root, receipt) == actual
    assert current['contract']['arguments'] == original['arguments']
    assert current['output_contract_history'][-1]['before'] == original['expected_outputs']
    if tool == 'run_string_tst':
        assert current['contract']['expected_outputs'] == [
            {'kind': 'file', 'path': 'string_input_CO2/manifest.csv'},
            {'kind': 'file', 'path': 'string_input_CO2/string_results.json'}]
    else:
        assert all(str(actual) in value['path'] for value in current['contract']['expected_outputs'])
    assert rt._line()['steps'][0]['expected_outputs'] == current['contract']['expected_outputs']
    assert rt.main.goal_contract.approved_nodes[0]['expected_outputs'] == current['contract']['expected_outputs']
    assert rt.snapshot()['goal_contract']['approved_nodes'][0]['expected_outputs'] == current['contract']['expected_outputs']
    assert not rt._bind_native_result_path('native', current['token'], receipt)
    assert len(rt.snapshot()['nodes']['native']['output_contract_history']) == 1


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
def test_terminal_owned_native_receipt_recovers_uncertain_without_new_attempt(factory, tmp_path, tool):
    rt, watch, receipt = native_attempt(factory, tmp_path, tool)
    before = rt.snapshot()['nodes']['native']
    leases_before = rt.store.snapshot().get('leases')
    ledger_before = RecoveryGate(path=rt.root/'recovery_state.json').snapshot()
    rt.tick(jobs=[watch])
    current = rt.snapshot()['nodes']['native']
    assert current['status'] == 'prefinish'
    assert current['job_ids'] == ['430'] and current['jobs_confirmed_terminal'] is True
    assert current['token'] == before['token'] and current['attempt_id'] == before['attempt_id']
    assert current['native_receipt_reconciliation']['no_resubmission'] is True
    assert current.get('node_verification') is None
    assert current['validation_dossier']['requires_agent_review'] is True
    assert rt.store.snapshot().get('leases') == leases_before
    assert RecoveryGate(path=rt.root/'recovery_state.json').snapshot() == ledger_before
    rt.tick(jobs=[watch])
    assert rt.snapshot()['nodes']['native']['attempt_id'] == before['attempt_id']


@pytest.mark.parametrize('tool', ['run_md_optimize', 'run_string_tst'])
def test_native_fresh_attempt_is_bound_before_its_scheduler_job_finishes(factory, tmp_path, tool):
    rt, watch, receipt = native_attempt(factory, tmp_path, tool, uncertain=False)
    watch.update(terminal=False, state='RUNNING')
    with RecoveryGate(path=rt.root/'recovery_state.json').transaction() as ledger:
        ledger['receipt-key']['status'] = 'submitted'
    before = rt.snapshot()['nodes']['native']
    rt._settle('native', before['token'], receipt)
    current = rt.snapshot()['nodes']['native']
    assert current['status'] == 'waiting_jobs' and current['job_ids'] == ['430']
    assert not current.get('validation_dossier') and not current.get('node_verification')
    assert current['attempt_id'] == before['attempt_id']


@pytest.mark.parametrize('blocker', ['outside', 'traversal', 'symlink', 'conflicting_workdir',
    'foreign_job', 'different_attempt', 'different_recovery', 'different_workdir', 'missing_job',
    'multiple_jobs', 'changed_input', 'changed_params', 'evidence_tamper', 'missing_evidence',
    'foreign_evidence', 'script_outside', 'artifact_outside', 'not_terminal', 'failed_job',
    'paused', 'user_change', 'pending_patch', 'new_goal', 'unapproved', 'token'])
def test_native_receipt_recovery_rejects_unproven_or_changed_attempt(factory, tmp_path, blocker):
    rt, watch, receipt = native_attempt(factory, tmp_path)
    token = rt.snapshot()['nodes']['native']['token']
    if blocker in {'outside', 'traversal', 'symlink'}:
        outside = tmp_path/'outside'
        outside.mkdir()
        if blocker == 'outside': receipt['output_dir'] = str(outside)
        elif blocker == 'traversal': receipt['output_dir'] = str(Path(receipt['output_dir'])/'..'/'..'/'..'/'..'/'outside')
        else:
            link = Path(receipt['output_dir']).parent/'escape'
            link.symlink_to(outside, target_is_directory=True)
            receipt['output_dir'] = str(link)
    if blocker == 'conflicting_workdir': receipt['work_dir'] = str(tmp_path)
    if blocker == 'foreign_job': watch['conv_id'] = 'other'
    if blocker == 'different_attempt': watch['attempt_id'] = 'other'
    if blocker == 'different_recovery': watch['recovery_key'] = 'other'
    if blocker == 'different_workdir': watch['work_dir'] = str(tmp_path)
    if blocker == 'missing_job': rt.job_watch.get = lambda jid: None
    if blocker == 'not_terminal': watch.update(terminal=False, state='RUNNING')
    if blocker == 'failed_job': watch.update(failed=True, state='FAILED')
    if blocker == 'script_outside': receipt['script'] = str(tmp_path/'outside.sh')
    if blocker == 'artifact_outside': receipt['manifest_path'] = str(tmp_path/'manifest.csv')
    if blocker == 'changed_input': Path(rt.main.goal_contract.approved_nodes[0]['arguments']['cif_path']).write_text('changed source')
    if blocker in {'multiple_jobs', 'changed_params'}:
        with RecoveryGate(path=rt.root/'recovery_state.json').transaction() as ledger:
            if blocker == 'multiple_jobs': ledger['receipt-key']['job_ids'].append('431')
            else: ledger['receipt-key']['params']['temperature'] = 310
    if blocker == 'evidence_tamper':
        path = rt.root/'evidence/originalnativeevidence.json'
        call = json.loads(path.read_text())
        call['params']['running_steps'] = 999
        write_checkpoint(path, call)
    if blocker in {'missing_evidence', 'foreign_evidence', 'paused', 'user_change'}:
        with json_transaction(rt.store.path) as data:
            w = data['workflows'][rt.workflow_id]
            if blocker == 'missing_evidence': w['nodes']['native']['result_ref'] = {}
            if blocker == 'foreign_evidence': w['nodes']['native']['result_ref']['evidence_path'] = str(tmp_path/'foreign.json')
            if blocker == 'paused': w['user_paused'] = True
            if blocker == 'user_change': w['pause_kind'] = 'user_change'
    if blocker == 'new_goal': rt.main.goal_contract.parameters['temperature_K'] = 310
    if blocker == 'unapproved': rt.main.goal_contract.execution_authorized = False
    if blocker == 'pending_patch': rt.main._pending_workflow_patch = {'affected_nodes': ['native']}
    before = rt.store.snapshot()
    with pytest.raises((ValueError, OSError)):
        rt._bind_native_result_path('native', 'other-token' if blocker == 'token' else token, receipt)
    assert rt.store.snapshot() == before


def test_tick_preserves_unknown_native_receipt_when_proof_is_blocked(factory, tmp_path):
    rt, watch, receipt = native_attempt(factory, tmp_path)
    watch['conv_id'] = 'foreign'
    before = rt.snapshot()['nodes']['native']
    rt.tick(jobs=[watch])
    after = rt.snapshot()['nodes']['native']
    assert after['status'] == 'uncertain'
    assert after['token'] == before['token'] and after['attempt_id'] == before['attempt_id']
    assert after['native_receipt_reconciliation_blocked']['no_resubmission'] is True


def test_original_native_attempt_survives_same_scope_metadata_pump(factory, tmp_path):
    rt, watch, receipt = native_attempt(factory, tmp_path)
    original = rt.snapshot()['nodes']['native']['origin_goal_version']
    rt.main.goal_contract.version += 1
    rt.tick(jobs=[watch])
    current = rt.snapshot()['nodes']['native']
    assert current['status'] == 'prefinish'
    assert current['origin_goal_version'] == original
    assert current['native_receipt_reconciliation']['attempt_id'] == 'original-attempt'


@pytest.mark.parametrize('change', ['pause', 'token', 'approval'])
def test_native_projection_stays_unchanged_if_scope_changes_after_proof(factory, tmp_path, monkeypatch, change):
    rt, watch, receipt = native_attempt(factory, tmp_path)
    before = copy.deepcopy(rt._line()['steps'][0])
    original = rt._native_attempt_receipt_proof
    def proof_then_change(*args, **kwargs):
        proof = original(*args, **kwargs)
        if change == 'pause':
            rt.store.user_pause(rt.workflow_id)
        elif change == 'token':
            with json_transaction(rt.store.path) as data:
                data['workflows'][rt.workflow_id]['nodes']['native']['token'] = 'replacement-owner'
        else:
            rt.main.goal_contract.version += 1
        return proof
    monkeypatch.setattr(rt, '_native_attempt_receipt_proof', proof_then_change)
    with pytest.raises(ValueError, match='owner|pause|approval'):
        rt._bind_native_result_path('native', rt.snapshot()['nodes']['native']['token'], receipt)
    assert rt._line()['steps'][0]['expected_outputs'] == before['expected_outputs']
    assert rt._line()['steps'][0].get('output_contract_history', []) == before.get('output_contract_history', [])
    assert rt.snapshot()['nodes']['native']['contract']['expected_outputs'] == before['expected_outputs']


def test_native_projection_commit_serializes_concurrent_pause_without_clearing_leases(factory, tmp_path, monkeypatch):
    import threading
    rt, watch, receipt = native_attempt(factory, tmp_path)
    before = rt.snapshot()['nodes']['native']
    leases = copy.deepcopy(rt.store.snapshot().get('leases'))
    entering = threading.Event()
    attempted = threading.Event()
    paused = threading.Event()
    original = rt.task_lines.repair_output_contract
    def pause():
        assert entering.wait(2)
        attempted.set()
        rt.store.user_pause(rt.workflow_id)
        paused.set()
    def bounded_repair(*args, **kwargs):
        entering.set()
        assert attempted.wait(2)
        assert not paused.wait(.05), 'pause must serialize after the metadata transaction'
        return original(*args, **kwargs)
    monkeypatch.setattr(rt.task_lines, 'repair_output_contract', bounded_repair)
    thread = threading.Thread(target=pause)
    thread.start()
    try:
        assert rt._bind_native_result_path('native', before['token'], receipt)
    finally:
        thread.join(3)
    assert not thread.is_alive() and paused.is_set()
    after = rt.snapshot()
    current = after['nodes']['native']
    assert after['user_paused'] is True
    assert rt._line()['steps'][0]['expected_outputs'] == current['contract']['expected_outputs']
    assert rt.main.goal_contract.approved_nodes[0]['expected_outputs'] == current['contract']['expected_outputs']
    assert current['attempt_id'] == before['attempt_id'] and current['token'] == before['token']
    assert rt.store.snapshot().get('leases') == leases
