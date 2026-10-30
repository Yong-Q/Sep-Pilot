"""Independent native API gate: read-only orientation never answers the user."""
import asyncio
import copy
import json
from types import SimpleNamespace

import pytest

from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.lifecycle import LifecycleStore, obsolete_internal_question_for_recovery
from agents.registry import _build_default_registry
from agents.session import Session


@pytest.mark.parametrize('decision_tool', ['user_decision', 'failure_decision', 'workflow_patch_decision'])
def test_native_api_orientation_keeps_real_pending_and_never_starts_main(tmp_path, monkeypatch, decision_tool):
    import api
    config = AgentConfig(api_key='test-placeholder-not-a-credential', project_root=tmp_path, model_provider='anthropic')
    monkeypatch.setattr('agents.config._config', config)
    actor = Session(config=config, registry=_build_default_registry())
    actor.goal_contract = GoalContract.from_user_message('Run the approved calculation; ask me before changing scientific parameters.')
    contract = {'step_id': 'submit', 'tool': 'run_cdft', 'agent': 'analyst',
                'arguments': {'action': 'submit', 'input_dir': str(tmp_path / 'inputs')},
                'depends_on': [], 'expected_outputs': []}
    actor.goal_contract.execution_authorized = True
    actor.goal_contract.execution_mode = 'workflow'
    actor.goal_contract.approved_plan_version = 1
    actor.goal_contract.approved_nodes = [copy.deepcopy(contract)]
    actor._checkpoint = lambda *args, **kwargs: None
    actor._pending_jobs_for_conv = lambda: []
    actor._pending_user_interaction = {'tool': decision_tool, 'prompt': 'Choose the physical model before changing it.',
        'params': {'decision_id': 'real-user-decision', 'related_nodes': ['submit'],
                   'options': ['retain current model', 'change physical model']}}
    actor._waiting_for_user_input = True
    actor._awaiting_plan_approval = False
    actor._pending_workflow_patch = None
    actor._current_line_id = 'conversation'
    actor.task_complete = True
    state = {'username': 'owner', 'conv_id': 'conversation', 'status': 'needs_user',
             'plan_version': 1, 'user_paused': False,
             'goal_contract': actor.goal_contract.to_dict(),
             'nodes': {'submit': {'status': 'failed', 'token': 'a' * 32,
                 'attempt_id': 'failed-attempt', 'recovery_key': 'failed-recovery',
                 'tool_returned': True, 'job_ids': [], 'contract': contract}}}
    actor._runtime_snapshot = lambda: copy.deepcopy(state)
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    store.initialize()
    payload = {'step_id': 'submit', 'correlation_token': 'a' * 32,
               'plan_version': 1, 'origin_goal_version': actor.goal_contract.version,
               'requires_user': False,
               'recovery': {'action': 'verify_changed_executor', 'tool': 'inspect_runtime',
                   'arguments': {'tool_name': 'run_cdft', 'arguments': contract['arguments']},
                   'recovery_key': 'failed-recovery', 'failed_attempt_id': 'failed-attempt',
                   'runtime_fingerprint': 'b' * 64, 'retry_granted': False}}
    event_id = store.emit('worker_recovery_ready', payload)
    actor._lifecycle_store = store
    actor._on_lifecycle_event = store.emit
    conv = SimpleNamespace(session=actor, username='owner', conv_id='conversation',
        is_processing=False, current_uid='real-user-turn', current_agent='lead-orchestrator',
        last_progress_at=0, interrupt_requested=False, interrupt_message='',
        current_status='waiting user', conversation_active=False, execution_logs=[])
    monkeypatch.setattr(api, '_watch', SimpleNamespace(list=lambda: []))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', lambda value: SimpleNamespace(tick=lambda *a, **kw: None))
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', 0)
    entered = []
    monkeypatch.setattr(actor, '_execute_loop', lambda *args, **kwargs: entered.append('model-loop'))

    async def schedule_native_main(current, event, mailbox):
        entered.append('main-scheduled')
        actor.resume_lifecycle_event(event)
    monkeypatch.setattr(api, '_reserve_and_schedule_lifecycle_main', schedule_native_main)
    pending = copy.deepcopy(actor._pending_user_interaction)
    goal = actor.goal_contract.to_dict()
    before_calls = copy.deepcopy(actor.memory.tool_call_log)
    asyncio.run(api._dispatch_conversation_lifecycle('owner', conv, store))
    event = store.snapshot()['events'][event_id]
    assert entered == []
    assert event['supervisor'] == 'delivered'
    assert event['main_chat'] == 'waiting_user'
    assert actor._pending_user_interaction == pending
    assert actor._waiting_for_user_input is True
    assert actor.goal_contract.to_dict() == goal
    assert actor._runtime_snapshot() == state
    assert actor.memory.tool_call_log == before_calls
    assert 'obsolete_internal_question' not in actor.context


@pytest.mark.parametrize('tool', ['user_decision', 'failure_decision', 'workflow_patch_decision'])
def test_orientation_does_not_use_legacy_question_clear_shortcut(tool):
    pending = {'tool': tool, 'params': {'decision_id': 'actual-choice', 'related_nodes': ['step']}}
    actor = SimpleNamespace(_pending_user_interaction=copy.deepcopy(pending),
                            _waiting_for_user_input=True, context={}, _checkpoint=lambda *a: None)
    event = {'event_id': 'orientation', 'kind': 'worker_recovery_ready',
             'payload': {'step_id': 'step', 'requires_user': False,
                         'recovery': {'action': 'verify_changed_executor', 'tool': 'inspect_runtime',
                                      'retry_granted': False}}}
    assert obsolete_internal_question_for_recovery(actor, event, runtime={}) is False
    assert actor._pending_user_interaction == pending and actor._waiting_for_user_input
    assert actor.context == {}
