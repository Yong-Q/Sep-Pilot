import asyncio
import json
import time
from types import SimpleNamespace
from pathlib import Path

import pytest
from anthropic.types import ToolUseBlock, TextBlock

from agents.config import AgentConfig
from agents.defns import MONITOR, ORCHESTRATOR, SUPERVISOR, resolve_agent
from agents.goal_contract import GoalContract
from agents.lifecycle import LifecycleStore
from agents.session import Session
from agents.state_io import compact_checkpoint_evidence, apply_evidence_references, write_checkpoint
from agents.task_line import TaskLineStore
from agents.workflow_patch import WorkflowContractError, patched_graph


def node(agent='analyst', tool='run_cdft', depends_on=None):
    return {'agent': agent, 'tool': tool,
            'arguments': {'action': 'pipeline', 'cif_dir': 'charged', 'gas': 'Kr', 'temperature': 298},
            'depends_on': depends_on or [], 'expected_outputs': ['results.csv'], 'resource_locks': ['cdft-workdir']}


def test_two_lifetime_receivers_and_agent_serialization(tmp_path):
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    store.emit('tool_failed', {'workflow': {'steps': [{'step_id': 'calc', **node()}]}}, event_id='one')
    store.emit('job_completed', {'job_id': '42'}, event_id='two')
    assert store.claim('one', 'supervisor')
    assert not store.claim('two', 'supervisor')  # one observer session, no overlapping turns
    store.receipt('one', 'supervisor', 'delivered', {'next_action': 'diagnose_and_fix'})
    assert store.claim('two', 'supervisor')
    assert store.claim('one', 'main_chat')
    restart = LifecycleStore(tmp_path / 'lifecycle.json')
    assert restart.snapshot()['events']['one']['payload']['workflow']['steps'][0]['arguments']['gas'] == 'Kr'
    assert restart.snapshot()['events']['one']['supervisor'] == 'delivered'


def test_supervisor_is_distinct_from_resource_monitor_and_cannot_submit():
    assert resolve_agent('supervisor') is SUPERVISOR
    assert SUPERVISOR is not MONITOR
    names = {f.__name__ for f in SUPERVISOR.functions}
    assert 'supervisor_decision' in names
    assert not names & {'submit_job', 'run_bash', 'write_file'}
    assert not any(n.startswith('handoff_to_') for n in names)


def test_readonly_observer_rejects_unexposed_scheduler_call(tmp_path, monkeypatch):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    def forbidden(*args, **kwargs):
        pytest.fail('supervisor must never dispatch a job')
    monkeypatch.setattr(s.registry, 'execute_dict', forbidden)
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='bad', name='submit_job', input={'command': 'python compute.py'})]),
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='decision', name='supervisor_decision', input={
            'next_action': 'propose_plan_patch', 'reason': 'local repairs did not solve the dependency',
            'evidence_refs': ['event1'], 'suggested_changes': [],
        })]),
    ])
    s._call_api = lambda agent: next(responses)
    result = s.observe_lifecycle_event({'kind': 'tool_failed', 'event_id': 'event1', 'payload': {'workflow': {'steps': []}}})
    assert result['next_action'] == 'propose_plan_patch'
    assert 'read-only' in s.memory.tool_call_log[0]['result']
    assert s.memory.agent_memories['supervisor']['exit_reports']


def test_dag_patch_rejects_cycles_and_missing_dependencies():
    existing = [{'step_id': 'charge', **node(tool='run_pacman_charge')},
                {'step_id': 'calc', **node(depends_on=['charge'])}]
    changes = [{'operation': 'upsert', 'step_id': 'charge', 'node': node(depends_on=['calc'])}]
    with pytest.raises(ValueError, match='cycle'):
        patched_graph(existing, changes)
    with pytest.raises(ValueError, match='missing'):
        patched_graph(existing, [{'operation': 'remove', 'step_id': 'charge'}])


def test_identical_upsert_preserves_failed_node_and_completed_descendant(tmp_path):
    import copy
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    original = {'step_id': 'calc', **node()}
    child = {'step_id': 'child', **node(depends_on=['calc'])}
    store.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': n['step_id'], 'node': n}
        for n in [original, child]], 0, 1, 'fixture')
    store.upsert_step('c1', 'calc', status='failed', job_ids=['42'])
    store.upsert_step('c1', 'child', status='completed', done=True, output_files=['result.csv'])
    changed = copy.deepcopy(original)
    changed['description'] = 'A clearer user-facing label, not a new execution request'
    line = store.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': 'calc', 'node': changed}], 1, 2, 'same executable contract')
    assert line['steps'][0]['status'] == 'failed' and line['steps'][0]['job_ids'] == ['42']
    assert line['steps'][1]['done'] and line['steps'][1]['output_files'] == ['result.csv']


def test_patch_schema_does_not_accept_execution_state_as_a_contract():
    from agents.registry import _build_default_registry
    candidate = {**node(), 'status': 'completed', 'done': True}
    issues = _build_default_registry().validate_params('propose_workflow_patch', {
        'base_version': 1, 'reason': 'Restore an already completed result',
        'changes': [{'operation': 'upsert', 'step_id': 'calc', 'node': candidate}]})
    assert issues


def test_activation_receipt_does_not_turn_recovery_block_into_success(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._on_workflow_start = lambda version: {'status': 'needs_recovery', 'scheduled': False,
        'blocked_nodes': ['gen'], 'reason': 'Original generator result must be restored'}
    receipt = session._activate_approved_workflow()
    assert not receipt['scheduled'] and receipt['status'] == 'activation_blocked'
    assert receipt['runtime']['blocked_nodes'] == ['gen']


def test_only_unclaimed_runtime_pending_can_reconcile_a_ghost_running_label(tmp_path):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    original = {'step_id': 'calc', **node()}
    store.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': 'calc', 'node': original}], 0, 1, 'fixture')
    store.upsert_step('c1', 'calc', status='running')
    import copy
    revised = copy.deepcopy(original)
    revised['arguments']['temperature'] = 299  # A real execution-contract edit, not an identical upsert.
    edit = [{'operation': 'upsert', 'step_id': 'calc', 'node': revised}]
    for actual in ({'status': 'running', 'token': 'realowner'}, {'status': 'pending', 'token': 'oldowner'},
                   {'status': 'pending', 'job_ids': ['42']}, {'status': 'pending', 'dispatch_phase': 'entered_tool'}):
        with pytest.raises(ValueError, match='live node'):
            store.apply_workflow_patch('c1', edit, 1, 2, 'repair', runtime_nodes={'calc': actual})
    line = store.apply_workflow_patch('c1', edit, 1, 2, 'repair', runtime_nodes={'calc': {'status': 'pending'}})
    assert line['steps'][0]['status'] == 'pending'
    assert line['plan_history'][-1]['steps'][0]['status_reconciliation_history'][0]['reported'] == 'running'


def test_operational_repairs_compare_contracts_not_user_words():
    import copy
    from agents.workflow_patch import operational_patch
    before = {'step_id': 'calc', **node()}
    before['arguments'].update(memory_mb=51200, nodelist='node06')
    after = copy.deepcopy(before)
    after['arguments']['nodelist'] = 'node07'
    change = [{'operation': 'upsert', 'step_id': 'calc', 'node': after}]
    assert operational_patch([before], change)
    for key, value in [('memory_mb', 102400), ('gas', 'Xe'), ('temperature', 310), ('cif_dir', 'other')]:
        changed = copy.deepcopy(change)
        changed[0]['node']['arguments'][key] = value
        assert not operational_patch([before], changed)


def test_conversation_annotations_cannot_change_executor_owned_nodes(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._runtime_snapshot = lambda: {'nodes': {'calc': {'status': 'pending'}}}
    result = session._task_line_update(step_id='calc', status='running')
    assert result['blocked'] and not result['executed'] and result['actual_status'] == 'pending'


def test_lifecycle_tool_does_not_return_old_dags_as_current_state(tmp_path):
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._lifecycle_store = LifecycleStore(tmp_path / 'lifecycle.json')
    session._lifecycle_store.emit('plan_patch_proposed', {'patch': {'nodes': [{'step_id': 'obsolete'}]}}, event_id='old')
    session._runtime_snapshot = lambda: {'status': 'active', 'plan_version': 3,
        'nodes': {'actual': {'status': 'pending', 'contract': {'agent': 'analyst', 'tool': 'run_cdft'}}}}
    state = session._lifecycle_state()
    assert state['workflow_runtime']['plan_version'] == 3
    assert set(state['workflow_runtime']['nodes']) == {'actual'}
    assert 'patch' not in state['recent_events'][0] and 'events' not in state


def test_patch_archives_old_results_and_invalidates_descendants(tmp_path):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.upsert_step('c1', 'charge', tool='run_pacman_charge', status='completed', done=True, plan_version=1)
    store.upsert_step('c1', 'calc', tool='run_cdft', status='failed', depends_on=['charge'], plan_version=1)
    store.upsert_step('c1', 'analysis', tool='validate_method', status='completed', done=True,
                      depends_on=['calc'], output_files=['old.csv'], plan_version=1)
    store.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': 'calc', 'node': node(depends_on=['charge'])}],
                               1, 2, 'repair failed calculation node')
    line = store.get_line('c1')
    by_id = {n['step_id']: n for n in line['steps']}
    assert by_id['charge']['done']
    assert by_id['calc']['status'] == 'pending'
    assert not by_id['analysis']['done'] and by_id['analysis']['output_files'] == []
    assert line['plan_history'][0]['steps'][2]['output_files'] == ['old.csv']
    with pytest.raises(ValueError, match='changed'):
        store.apply_workflow_patch('c1', [], 1, 3, 'stale patch')


def test_user_negotiation_applies_structured_patch_only_after_approval(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.upsert_step('c1', 'calc', tool='run_cdft', status='failed', plan_version=1)
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.current_agent = ORCHESTRATOR
    s._current_line_id = 'c1'
    s.goal_contract = GoalContract.from_user_message('用cDFT计算Kr，298K，直接执行')
    proposal = s._propose_workflow_patch(1, 'Original input contract was wrong; repair node', [
        {'operation': 'upsert', 'step_id': 'calc', 'node': node()},
    ])
    assert not proposal['applied']
    assert store.get_line('c1')['steps'][0]['status'] == 'failed'
    state = s.export_state()
    assert state['pending_workflow_patch']
    s._execute_loop = lambda agent, **kwargs: 'main chat continues approved node'
    result = s.reply('确认编排补丁', verbose=False)
    assert result.startswith('main chat')
    assert s.goal_contract.approved_plan_version == 0  # raw words cannot apply the patch
    s._apply_workflow_patch(2)  # operation chosen by the main model; SDK dialogue covers this choice
    assert s.goal_contract.approved_plan_version == 2
    assert store.get_line('c1')['steps'][0]['arguments']['gas'] == 'Kr'
    assert s._pending_workflow_patch is None


def test_full_tool_evidence_is_a_file_not_unbounded_checkpoint_text(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.memory.record_tool_call('analyst', 'read_file', {'path': 'input.cif'}, '{"content":"full real evidence"}')
    state = s.export_state()
    refs = compact_checkpoint_evidence(state, tmp_path / 'evidence')
    write_checkpoint(tmp_path / 'session_checkpoint.json', state)
    apply_evidence_references(s.memory, refs)
    assert 'result' not in s.memory.tool_call_log[0]
    assert 'result' not in s.memory.agent_memories['analyst']['tool_call_log'][0]
    loaded = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    loaded.import_state(json.loads((tmp_path / 'session_checkpoint.json').read_text()))
    loaded._evidence_root = tmp_path / 'evidence'
    assert 'full real evidence' in loaded._load_evidence_call(loaded.memory.tool_call_log[0])['result']


def test_lifecycle_mailbox_auto_continues_but_preserves_user_freeze(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    import agents.job_watch
    monkeypatch.setattr(agents.job_watch.JobWatch, 'start', lambda self: self)
    import api
    main = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    main.goal_contract = GoalContract.from_user_message('用cDFT计算Kr，298K')
    main._lifecycle_store = LifecycleStore(tmp_path / 'lifecycle.json')
    main._lifecycle_store.emit('job_completed', {'workflow': {'steps': [{'step_id': 'calc', **node()}]}}, event_id='event1')
    conv = SimpleNamespace(session=main, conv_id='c1', username='u', is_processing=False,
                           current_uid='old', current_agent='analyst', last_progress_at=0,
                           interrupt_requested=True, interrupt_message='')
    user = SimpleNamespace(conversations={'c1': conv})
    monkeypatch.setattr(api.session_manager, 'list_users', lambda: ['u'])
    monkeypatch.setattr(api.session_manager, 'get_user', lambda name: user)
    scheduled, continued = [], []
    monkeypatch.setattr(api, '_schedule_lifecycle', lambda coroutine: scheduled.append(coroutine))
    async def supervisor(c, event):
        c.session._lifecycle_store.receipt(event['event_id'], 'supervisor', 'delivered', {
            'next_action': 'advance_dependencies', 'reason': 'confirmed', 'evidence_refs': ['42'],
        })
    async def continue_main(c, event):
        continued.append(event)
        c.is_processing = False
        c.session._lifecycle_store.receipt(event['event_id'], 'main_chat', 'delivered')
    monkeypatch.setattr(api, '_supervise_lifecycle_event', supervisor)
    monkeypatch.setattr(api, '_continue_lifecycle_event', continue_main)
    async def run():
        await api._dispatch_lifecycle_once()
        assert not scheduled  # a session freeze covers supervisor and main chat
        conv.interrupt_requested = False
        await api._dispatch_lifecycle_once()
        assert len(scheduled) == 1  # supervisor receives the durable event first
        await asyncio.gather(*scheduled)
        scheduled.clear()
        await api._dispatch_lifecycle_once()
        assert conv.is_processing  # main turn reserved before task scheduling
        await asyncio.gather(*scheduled)
    asyncio.run(run())
    assert len(continued) == 1
    assert continued[0]['payload']['workflow']['steps'][0]['arguments']['gas'] == 'Kr'
    events = main._lifecycle_store.snapshot()['events']
    assert events['event1']['main_chat'] == events['event1']['supervisor'] == 'delivered'


def test_pre_restart_event_without_formal_dag_is_archived_not_replayed(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    import agents.job_watch
    monkeypatch.setattr(agents.job_watch.JobWatch, 'start', lambda self: self)
    import api
    main = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    main._lifecycle_store = LifecycleStore(tmp_path / 'lifecycle.json')
    main._lifecycle_store.emit('supervisor_review', {'main_already_handling': False}, event_id='ghost')
    main._runtime_snapshot = lambda: {}
    conv = SimpleNamespace(session=main, conv_id='c1', username='u', is_processing=False,
                           current_uid='old', current_agent='lead-orchestrator', last_progress_at=0,
                           interrupt_requested=False, interrupt_message='')
    user = SimpleNamespace(conversations={'c1': conv})
    monkeypatch.setattr(api.session_manager, 'list_users', lambda: ['u'])
    monkeypatch.setattr(api.session_manager, 'get_user', lambda name: user)
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', time.time() + 1)
    scheduled = []
    monkeypatch.setattr(api, '_schedule_lifecycle', lambda coroutine: scheduled.append(coroutine))
    asyncio.run(api._dispatch_lifecycle_once())
    assert not scheduled
    event = main._lifecycle_store.snapshot()['events']['ghost']
    assert event['main_chat'] == event['supervisor'] == 'obsolete'
    assert event['main_chat_receipt']['not_resubmitted'] is True
    assert not conv.is_processing


def test_pre_restart_event_with_active_dag_remains_recoverable(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    import agents.job_watch
    monkeypatch.setattr(agents.job_watch.JobWatch, 'start', lambda self: self)
    import api
    main = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    main._lifecycle_store = LifecycleStore(tmp_path / 'lifecycle.json')
    main._lifecycle_store.emit('job_completed', {'job': {'job_id': '42'}}, event_id='real-work')
    main._runtime_snapshot = lambda: {'status': 'active', 'user_paused': False,
        'nodes': {'calc': {'status': 'waiting_jobs'}}}
    conv = SimpleNamespace(session=main, conv_id='c1', username='u', is_processing=False,
                           current_uid='old', current_agent='lead-orchestrator', last_progress_at=0,
                           interrupt_requested=False, interrupt_message='')
    user = SimpleNamespace(conversations={'c1': conv})
    monkeypatch.setattr(api.session_manager, 'list_users', lambda: ['u'])
    monkeypatch.setattr(api.session_manager, 'get_user', lambda name: user)
    monkeypatch.setattr(api, '_PROCESS_STARTED_AT', time.time() + 1)
    monkeypatch.setattr(api, '_watch', SimpleNamespace(list=lambda: []))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', lambda c: SimpleNamespace(tick=lambda jobs, frozen: None))
    scheduled = []
    monkeypatch.setattr(api, '_schedule_lifecycle', lambda coroutine: scheduled.append(coroutine))
    asyncio.run(api._dispatch_lifecycle_once())
    assert len(scheduled) == 1
    for coroutine in scheduled:
        coroutine.close()
    event = main._lifecycle_store.snapshot()['events']['real-work']
    assert event['supervisor'] == 'processing'
    assert event['main_chat'] == 'pending'


def test_idle_lifecycle_heartbeat_and_recovery_are_write_throttled(tmp_path):
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    store.initialize()
    store.heartbeat()
    first = store.snapshot()
    first_heartbeat = first['agents']['main_chat']['service_heartbeat']
    store.heartbeat()
    assert store.snapshot()['agents']['main_chat']['service_heartbeat'] == first_heartbeat

    # The first recovery audit may write its transaction; the immediate second
    # call must be a no-op rather than fsyncing the same idle mailbox again.
    store.recover_claims()
    first_mtime = store.path.stat().st_mtime_ns
    store.recover_claims()
    assert store.path.stat().st_mtime_ns == first_mtime


def test_job_group_requires_all_success_and_chain_prerequisite_stays_ready(tmp_path):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.upsert_step('c1', 'calc', tool='run_gcmc_batch', job_ids=['1', '2'], status='submitted')
    store.mark_job_done('1', True, 'COMPLETED')
    assert not store.get_line('c1')['steps'][0]['done']
    store.mark_job_done('2', True, 'COMPLETED')
    assert store.get_line('c1')['steps'][0]['done']
    store.upsert_step('c1', 'chain', tool='run_henry_chain', job_ids=['3'], status='submitted',
                      validation={'chain_status': 'waiting'})
    store.mark_job_done('3', True, 'COMPLETED')
    chain = next(n for n in store.get_line('c1')['steps'] if n['step_id'] == 'chain')
    assert not chain['done'] and chain['status'] == 'ready'


def test_fixed_chain_wait_does_not_repeat_submit_or_advance(tmp_path):
    from agents.chain import Chain, ChainStep, StepResult, StepStatus, HenryStep, GcmcStep
    calls = []
    class Wait(ChainStep):
        def execute(self, ctx):
            calls.append('wait')
            return StepResult('charge', StepStatus.RUNNING, output={'charge_job_id': '42'})
    class Next(ChainStep):
        def execute(self, ctx):
            pytest.fail('must not advance past a running prerequisite')
    context = Chain('fixed').add_step('charge', Wait()).add_step('henry', Next()).run({})
    assert calls == ['wait']
    assert context.get('charge_job_id') == '42'
    assert HenryStep.max_retries == GcmcStep.max_retries == 1


def test_main_can_receive_provisional_receipt_without_waiting_for_llm_audit(tmp_path):
    store = LifecycleStore(tmp_path / 'lifecycle.json')
    event = store.emit('tool_failed', {'error': 'bad input'})
    assert store.claim(event, 'supervisor')
    store.provisional(event, {'next_action': 'diagnose_and_fix', 'reason': 'do not repeat submit'})
    assert store.claim(event, 'main_chat')
    assert store.snapshot()['events'][event]['supervisor'] == 'processing'
    assert store.snapshot()['events'][event]['supervisor_receipt']['audit_pending']


def test_resource_monitor_cannot_handoff_to_main_or_compute(tmp_path, monkeypatch):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    responses = iter([
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='bad', name='handoff_to_adsorption', input={'task': 'compute'})]),
        SimpleNamespace(content=[TextBlock(type='text', text='UNKNOWN; resources unverified.')]),
    ])
    s._call_api = lambda agent: next(responses)
    monkeypatch.setattr(s.registry, 'execute_dict', lambda *a, **kw: pytest.fail('no compute/handoff should be executed'))
    verdict = s.run_readonly_observer('check resources', MONITOR)
    assert 'UNKNOWN' in verdict
    assert s.current_agent is MONITOR


def test_conversation_kernel_lease_blocks_other_worker_instance(tmp_path):
    from agents.state_io import ConversationLock
    a = ConversationLock(tmp_path / 'c1.lock')
    b = ConversationLock(tmp_path / 'c1.lock')
    assert a.acquire(blocking=False)
    assert not b.acquire(blocking=False)
    a.release()
    assert b.acquire(blocking=False)
    b.release()


def test_corrupt_taskline_is_not_replaced_with_empty_state(tmp_path):
    path = tmp_path / 'lines.json'
    store = TaskLineStore(str(path))
    store.upsert_step('c1', 'one', tool='run_cdft')
    path.write_text('broken json evidence')
    with pytest.raises(RuntimeError, match='preserving'):
        store.upsert_step('c1', 'two')
    assert path.read_text() == 'broken json evidence'


def test_api_checkpoint_writes_task_manifest_and_evidence_files(tmp_path, monkeypatch):
    pytest.importorskip('fastapi')
    monkeypatch.setenv('BIMEM_BACKGROUND_MONITOR', '0')
    from fastapi.testclient import TestClient
    import api
    import auth
    config = AgentConfig(api_key='test', project_root=tmp_path)
    monkeypatch.setattr('agents.config.get_config', lambda: config)
    monkeypatch.setattr('agents.workspace.get_config', lambda: config)
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    conv = auth.ConversationState(conv_id='c1', username='u')
    conv.session = Session(config=config)
    conv.session.current_agent = ORCHESTRATOR
    conv.session.goal_contract = GoalContract.from_user_message('Kr cDFT 298K')
    user = auth.UserData(username='u', conversations={'c1': conv}, current_conv_id='c1')
    monkeypatch.setattr(api.session_manager, 'get_user', lambda name: user)
    previous = dict(api.app.dependency_overrides)
    api.app.dependency_overrides[api._get_user] = lambda: 'u'
    try:
        with TestClient(api.app) as client:
            response = client.get('/api/conversations/c1/workflow')
            assert response.status_code == 200
            payload = response.json()
            assert payload['scope'] == {'username': 'u', 'conv_id': 'c1'}
            assert isinstance(payload['snapshot_at'], float)
            files = payload['memory_files']
            assert 'runs/u/c1/task_manifest.json' in files['task']
            conv.session.memory.record_tool_call('analyst', 'read_file', {'path': 'input.cif'}, '{"content":"real tool output"}')
            conv.session._checkpoint('after_tool_result')
            manifest = json.loads(Path(files['task']).read_text())
            assert manifest['negotiation_owner'] == manifest['delivery_owner'] == 'lead-orchestrator'
            assert manifest['goal_contract']['gases'] == ['Kr']
            call = conv.session.memory.tool_call_log[0]
            assert 'result' not in call and Path(call['evidence_path']).exists()
            observer = api._ensure_supervisor(conv)
            observer._checkpoint('supervisor_restore')
            supervisor = json.loads(Path(files['supervisor']).read_text())
            assert supervisor['current_agent'] == 'supervisor'
            assert supervisor['goal_contract']['gases'] == ['Kr']
    finally:
        api.app.dependency_overrides.clear()
        api.app.dependency_overrides.update(previous)


def test_initial_direct_execution_compiles_real_nodes_without_extra_user_pause(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._current_line_id = 'c1'
    s.goal_contract = GoalContract.from_user_message('生成42个HOF，然后用cDFT计算Kr/Xe，298K，直接执行')
    changes = [
        {'operation': 'upsert', 'step_id': 'generate', 'node': {
            'agent': 'harness', 'tool': 'generate_structure',
            'arguments': {'material_type': 'HOF', 'n_structures': 42, 'max_atoms': 1500, 'output_dir': str(tmp_path / 'structures')},
            'depends_on': [], 'expected_outputs': ['*.cif'],
        }},
        {'operation': 'upsert', 'step_id': 'calc', 'node': {
            'agent': 'analyst', 'tool': 'run_cdft',
            'arguments': {'action': 'pipeline', 'cif_dir': str(tmp_path / 'structures'), 'gases': ['Kr', 'Xe'], 'temperature': 298},
            'depends_on': ['generate'], 'expected_outputs': ['results.csv'],
        }},
    ]
    with pytest.raises(ValueError, match='complete workflow draft'):
        s._propose_workflow_patch(0, 'Partial initial compile must be rejected', changes)
    s.current_agent = ORCHESTRATOR
    s._record_workflow_draft([
        {'step_id': change['step_id'], 'description': change['step_id'],
         'agent': change['node']['agent'], 'tool': change['node']['tool'],
         'depends_on': change['node']['depends_on'], 'missing_parameters': []}
        for change in changes
    ], completion_criteria='A verified Kr/Xe cDFT result covering all 42 generated HOFs')
    proposal = s._propose_workflow_patch(0, 'Compile initial user-authorized generation and calculation', changes)
    assert proposal['applied'] and proposal['status'] == 'approved'
    assert not s._waiting_for_user_input
    assert s.goal_contract.approved_nodes
    calc = next(n for n in store.get_line('c1')['steps'] if n['step_id'] == 'calc')
    assert not s._check_taskline_dependencies('run_cdft', calc['arguments'])[0]


def test_initial_dag_compiles_defaults_and_serial_inputs_before_activation(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1', username='u', conv_id='c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    monkeypatch.setattr('agents.watch_context.get_context', lambda: {
        'username': 'u', 'conv_id': 'c1', 'line_id': 'c1',
    })
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._current_line_id = 'c1'
    session.current_agent = ORCHESTRATOR
    session.goal_contract = GoalContract.from_user_message(
        '生成一个MOF结构并建立CIF数据库，直接执行')
    root = tmp_path / 'runs/u/c1'
    changes = [
        {'operation': 'upsert', 'step_id': 'generate', 'node': {
            'agent': 'harness', 'tool': 'generate_structure',
            'arguments': {'material_type': 'MOF', 'n_structures': 1,
                          'output_dir': str(root / 'structures')},
            'depends_on': [],
            'expected_outputs': [{'kind': 'directory', 'path': str(root / 'structures'),
                                  'pattern': '*.cif', 'min_count': 1}],
        }},
        {'operation': 'upsert', 'step_id': 'inventory', 'node': {
            'agent': 'analyst', 'tool': 'build_mof_database', 'arguments': {},
            'depends_on': ['generate'], 'expected_outputs': [],
        }},
    ]
    session._record_workflow_draft([
        {'step_id': change['step_id'], 'description': change['step_id'],
         'agent': change['node']['agent'], 'tool': change['node']['tool'],
         'depends_on': change['node']['depends_on'], 'missing_parameters': []}
        for change in changes
    ], completion_criteria='A session-owned database of the generated CIF')

    def must_not_execute(*_args, **_kwargs):
        pytest.fail('DAG compilation must not execute a tool')

    monkeypatch.setattr(session.registry, 'execute', must_not_execute)
    monkeypatch.setattr(session.registry, 'execute_dict', must_not_execute)
    proposal = session._propose_workflow_patch(
        0, 'Compile complete serial generation and inventory workflow', changes)
    inventory = next(
        node for node in proposal['proposed_nodes'] if node['step_id'] == 'inventory')
    assert inventory['arguments'] == {
        'cif_dir': str(root / 'structures'), 'recursive': False, 'max_files': 1000,
    }
    assert {item['parameter'] for item in
            proposal['compiler_receipt']['defaults_applied']} == {
                'recursive', 'max_files',
            }


def test_initial_dag_reports_generic_missing_non_path_parameter(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1', username='u', conv_id='c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    monkeypatch.setattr('agents.watch_context.get_context', lambda: {
        'username': 'u', 'conv_id': 'c1', 'line_id': 'c1',
    })
    session = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    session._current_line_id = 'c1'
    session.current_agent = ORCHESTRATOR
    output = tmp_path / 'runs/u/c1/structures'
    changes = [{'operation': 'upsert', 'step_id': 'generate', 'node': {
        'agent': 'harness', 'tool': 'generate_structure',
        'arguments': {'material_type': 'MOF', 'output_dir': str(output)},
        'depends_on': [], 'expected_outputs': [str(output / '*.cif')],
    }}]
    with pytest.raises(WorkflowContractError) as error:
        session._propose_workflow_patch(
            0, 'Compile report and surface its exact missing parameters', changes)
    assert error.value.details['error_kind'] == 'schema_validation'
    assert error.value.details['missing_parameters'] == ['n_structures']
    assert error.value.details['tool'] == 'generate_structure'
    assert error.value.details['input_schema']


def test_same_tool_nodes_match_exact_params_not_first_tool_in_list(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.upsert_step('c1', 'upstream', tool='run_pacman_charge', status='failed')
    args_a = {'action': 'pipeline', 'cif_dir': 'a', 'gas': 'Kr', 'temperature': 298}
    args_b = {'action': 'pipeline', 'cif_dir': 'b', 'gas': 'Xe', 'temperature': 298}
    store.upsert_step('c1', 'calcA', tool='run_cdft', arguments=args_a, depends_on=['upstream'], status='pending')
    store.upsert_step('c1', 'calcB', tool='run_cdft', arguments=args_b, depends_on=[], status='pending')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._current_line_id = 'c1'
    assert not s._check_taskline_dependencies('run_cdft', args_a)[0]
    assert s._check_taskline_dependencies('run_cdft', args_b)[0]
    store.upsert_step('c1', 'calcB', tool='run_cdft', status='running')
    nodes = {n['step_id']: n for n in store.get_line('c1')['steps']}
    assert nodes['calcA']['status'] == 'pending'
    assert nodes['calcB']['status'] == 'running'


def test_unlisted_submission_is_not_part_of_approved_dag():
    goal = GoalContract.from_user_message('用GCMC计算Kr，298K')
    goal.approved_nodes = [{'tool': 'run_henry', 'arguments': {'cif': 'x.cif', 'gas': 'Kr', 'temperature': 298}}]
    assert not goal.guard_tool_call('run_henry_chain', {'material': 'x', 'gas': 'Kr', 'temperature': 298})[0]
    goal.method = 'CDFT'
    goal.approved_nodes = [{'tool': 'run_cdft', 'arguments': {'action': 'pipeline', 'cif_dir': 'x', 'gas': 'Kr', 'temperature': 298}}]
    assert goal.guard_tool_call('run_cdft', {'action': 'collect', 'job_work_dir': 'completed_job'})[0]


def test_four_agent_receipts_cannot_bypass_unfinished_workflow(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1', username='tester', conv_id='c1')
    monkeypatch.setattr('agents.watch_context.get_context', lambda: {'username': 'tester', 'conv_id': 'c1', 'line_id': 'c1'})
    store.upsert_step('c1', 'deliver', tool='task_line_update', status='pending')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    s.goal_contract = GoalContract.from_user_message('hello')
    s._current_line_id = 'c1'
    def text(value):
        return SimpleNamespace(content=[TextBlock(type='text', text=value)])
    responses = []
    for index, tool in enumerate(('handoff_to_adsorption', 'handoff_to_analyst', 'handoff_to_communicator', 'handoff_to_harness')):
        responses.append(SimpleNamespace(content=[ToolUseBlock(type='tool_use', id=f'h{index}', name=tool, input={'task': 'do your part'})]))
        responses.append(text('part receipt'))
    responses.extend([
        text('premature final must not be returned'),
        SimpleNamespace(content=[ToolUseBlock(type='tool_use', id='finish', name='task_line_update', input={
            'line_id': 'c1', 'step_id': 'deliver', 'status': 'completed', 'evidence': 'real delivery artifact checked',
        })]),
        text('verified final'),
    ])
    replies = iter(responses)
    s._call_api = lambda agent: next(replies)
    assert s._execute_loop(ORCHESTRATOR) == 'verified final'
    assert store.get_line('c1')['steps'][0]['done']


def test_only_main_chat_owns_decision_and_specialist_patch_delivery(tmp_path):
    from agents.defns import ADSORPTION
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.current_agent = ADSORPTION
    with pytest.raises(ValueError, match='main chat'):
        s._request_user_decision('Please confirm the original job identity', 'external status', ['calc'])
    handback = s._propose_workflow_patch(1, 'specialist recommends a real structured change', [])
    assert handback.agent is ORCHESTRATOR
    assert handback.context_variables['workflow_patch_proposal']['base_version'] == 1
    assert not s._waiting_for_user_input


def test_user_identity_decision_rebinds_existing_job_without_submitting(tmp_path, monkeypatch):
    from agents.job_watch import JobWatch
    from agents.watch_context import set_context, clear_context
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.current_agent = ORCHESTRATOR
    s.goal_contract = GoalContract.from_user_message('用cDFT计算Kr，298K')
    s._execute_loop = lambda agent, **kwargs: 'main continues after user answer'
    args = {'action': 'pipeline', 'cif_dir': 'charged', 'gas': 'Kr', 'temperature': 298}
    aid, _ = s.recovery_gate.claim('task', 'run_cdft', args, 'fp')
    watch = JobWatch(watch_file=str(tmp_path / 'jobs.json'))
    watch.register('42', username='u', conv_id='c1', tool='run_cdft', work_dir=str(tmp_path))
    monkeypatch.setattr('agents.job_watch.get_watch', lambda: watch)
    request = s._request_user_decision('请确认42是否为这个中断任务的原作业，确认后只绑定原作业而不重提。',
                                        'dispatch identity unclear', ['calc'], recovery_key='task', candidate_job_id='42')
    s.reply('确认，原作业就是42', verbose=False)
    s.memory.record_tool_call('supervisor', 'diagnose_job', {'job_id': '42', 'work_dir': str(tmp_path)},
                              '{"status":"FAILED","terminal":true,"failed":true,"error":"real stderr"}')
    call_id = s.memory.tool_call_log[-1]['call_id']
    set_context('u', 'c1', 'lead-orchestrator')
    try:
        receipt = s._reconcile_watched_job('task', '42', request['decision_id'], call_id)
    finally:
        clear_context()
    assert not receipt['new_submission']
    entry = s.recovery_gate.snapshot()['task']
    assert entry['job_ids'] == ['42'] and entry['status'] == 'failed'
    assert entry['attempts'] == 1


def test_fabricated_user_approval_cannot_reconcile_dispatch(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.current_agent = ORCHESTRATOR
    with pytest.raises(ValueError, match='genuine user'):
        s._reconcile_watched_job('task', '42', 'agent-says-user-agreed', 'invented-call')
