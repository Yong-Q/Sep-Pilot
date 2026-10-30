"""The native monitor can reserve only a current no-op approved rebase."""
import copy
import json
from types import SimpleNamespace
import pytest
from agents import parallel_workflow as pw
from tests.test_report_noop_rebase import rebased


def runtime_fixture(monkeypatch):
    state, goal, line, approval = rebased()
    state['status'] = 'needs_user'
    line['steps'] = copy.deepcopy(goal['approved_nodes'])
    class Goal:
        def to_dict(self): return copy.deepcopy(goal)
        def guard_tool_call(self, *args, **kwargs): return True, ''
    main = SimpleNamespace(goal_contract=Goal(), context={'approved_execution_proof': approval},
                           _pending_workflow_patch=None, _awaiting_plan_approval=False)
    rt = object.__new__(pw.ParallelWorkflow)
    rt.main = main; rt.username = 'owner'; rt.conv_id = 'conv'; rt._line = lambda: copy.deepcopy(line)
    monkeypatch.setattr(pw, 'activation_framework_fingerprint', lambda: 'reviewed-test-framework')
    return rt, state, goal, line, approval


def test_monitor_recognizes_proved_noop_rebase_and_bounds_reservation(monkeypatch, tmp_path):
    rt, state, _, _, _ = runtime_fixture(monkeypatch)
    proof = rt._approved_activation_candidate(state)
    assert proof and proof['noop_report_rebase']['retained_steps'] == ['figure']
    assert proof['noop_report_rebase']['no_execution_authority']
    path = tmp_path / 'workflows.json'
    path.write_text(json.dumps({'workflows': {'w': state}, 'leases': {}}))
    store = pw.WorkflowStore(path)
    claim = store.reserve_approved_activation('w', proof)
    assert claim and claim['status'] == 'attempting'
    assert store.reserve_approved_activation('w', proof) is None
    assert store.snapshot('w')['nodes'] == state['nodes']


@pytest.mark.parametrize('change', ['paused', 'pending_patch', 'unsigned', 'different_science',
                                   'different_figure', 'user_stop', 'history_gap', 'wrong_pause'])
def test_monitor_does_not_reserve_without_exact_current_authority(monkeypatch, change):
    rt, state, goal, line, approval = runtime_fixture(monkeypatch)
    if change == 'paused': state['user_paused'] = True
    elif change == 'pending_patch': rt.main._pending_workflow_patch = {'changes': []}
    elif change == 'unsigned': approval['approval_sha256'] = 'invalid'
    elif change == 'different_science': goal['approved_nodes'][0]['arguments']['gas'] = 'CO2'
    elif change == 'different_figure': goal['approved_nodes'][1]['arguments']['instructions'] = 'redraw'
    elif change == 'user_stop': goal['directives'][-1] = '不要继续原任务。'
    elif change == 'history_gap': line['plan_history'].pop()
    elif change == 'wrong_pause': state['pause_reason'] = 'unrelated block'
    if change in {'different_science', 'different_figure'}:
        line['steps'] = copy.deepcopy(goal['approved_nodes'])
        approval['approval_sha256'] = pw._approval_digest(goal)
    assert rt._approved_activation_candidate(state) is None


def test_reservation_alone_cannot_commit_without_native_presentation_proof(monkeypatch, tmp_path):
    rt, state, goal, _, _ = runtime_fixture(monkeypatch)
    proof = rt._approved_activation_candidate(state)
    path = tmp_path / 'workflows.json'
    path.write_text(json.dumps({'workflows': {'w': state}, 'leases': {}}))
    store = pw.WorkflowStore(path)
    claim = store.reserve_approved_activation('w', proof)
    with pytest.raises(ValueError, match='recovery ownership, pause or approval'):
        store.activate('w', goal['approved_nodes'], goal, 'owner', 'conv', recovery_proof=claim)
    assert store.snapshot('w')['plan_version'] == 6
