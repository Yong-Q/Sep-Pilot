"""Bounded barriers and fake scheduler receipts, never real submissions/LLMs."""
import copy
import json
import multiprocessing as mp
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore, conflicts, resources_for
from agents.registry import _build_default_registry
from agents.session import Session
from agents.state_io import json_transaction,write_checkpoint
from agents.task_line import TaskLineStore


def test_branch_failure_keeps_independent_ready_work_and_blocks_descendants(factory):
    rt = factory([node('bad'), node('dependent', deps=['bad']), node('independent')])
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'bad')
    rt.store.update(rt.workflow_id, 'bad', ticket['token'], {'status': 'failed'}, release=True)
    rt.store.pause(rt.workflow_id, 'failure', step_id='bad')
    assert rt.snapshot()['status'] == 'active'
    assert rt.store.claim(rt.workflow_id, 'dependent') is None
    independent = rt.store.claim(rt.workflow_id, 'independent')
    assert independent
    rt.store.update(rt.workflow_id, 'independent', independent['token'], {'status': 'succeeded'}, release=True)


def test_pending_patch_does_not_persist_user_pause(factory):
    rt = factory([node('a')])
    rt.start(1)
    rt.main._pending_workflow_patch = {'affected_nodes': ['a']}
    rt.tick(blocked_nodes=['a'])
    assert not rt.snapshot()['user_paused']
    assert rt.snapshot()['nodes']['a']['status'] == 'pending'


def test_start_and_tick_requires_a_real_root_claim(factory):
    rt = factory([node('a', path='/shared')])
    blocker = factory([node('blocker', tool='write_file',
                            args={'path': '/shared', 'content': 'held'})], cid='c2')
    blocker.start(1)
    assert blocker.store.claim(blocker.workflow_id, 'blocker')

    blocked = rt.start_and_tick(1)

    assert blocked['scheduled'] is False
    assert blocked['status'] == 'waiting_dispatch'
    assert 'resource lease conflict' in blocked['blocked_nodes']['a'][0]


def test_start_and_tick_reports_dispatch_only_after_root_claim(factory):
    rt = factory([node('a')])
    gate = threading.Event()
    rt.main.registry.get('read_file').execute = lambda args: gate.wait(5) or {'content': 'done'}

    receipt = rt.start_and_tick(1)

    assert receipt['scheduled'] is True
    assert receipt['dispatch_started'] is True
    assert receipt['root_nodes']['a']['token'] is True
    gate.set()


def test_goal_metadata_reconciles_but_science_does_not(factory):
    rt = factory([node('a')])
    rt.start(1)
    rt.main.goal_contract.version += 1
    assert rt.store.reconcile_goal(rt.workflow_id, rt.main.goal_contract.to_dict())
    rt.main.goal_contract.parameters['temperature_K'] = 999
    assert not rt.store.reconcile_goal(rt.workflow_id, rt.main.goal_contract.to_dict())


def test_resource_review_transient_failures_retry_without_submission_or_user_veto(factory, tmp_path):
    work = tmp_path / 'resource-work'
    work.mkdir()
    rt = factory([node('compute', tool='run_cdft', args={
        'action': 'pipeline', 'cif_dir': str(tmp_path / 'inputs'), 'gas': 'Kr',
        'temperature': 298, 'job_work_dir': str(work)}, outputs=[str(work / 'result.csv')])])
    rt.start(1)
    rt.main._on_resource_review = lambda request: {'status': 'review_retry', 'reason': 'temporary provider failure'}
    rt.main.registry.get('run_cdft').execute = lambda args: pytest.fail('must not submit without verified resources')
    for attempt in range(1, 4):
        rt.tick()
        eventually(lambda: rt.snapshot()['nodes']['compute']['status'] == 'needs_resources')
        for future in list(rt.futures.values()): future.result(timeout=5)
        current = rt.snapshot()['nodes']['compute']
        assert current['resource_review_attempts'] == attempt
        assert not rt.snapshot()['user_paused']
        assert not rt.store.snapshot().get('leases')
        if attempt < 3:
            assert current['resource_retry_at']
            rt.store.update(rt.workflow_id, 'compute', current['token'], {'resource_retry_at': time.time() - 1})
        else:
            assert current['resource_retry_at'] is None
    events = list(rt.snapshot()['outbox'].values())
    reviews = [event for event in events if event['kind'] == 'worker_resource_review_required']
    assert len(reviews) == 3
    assert all(not event['payload']['requires_user'] for event in reviews)
    assert reviews[-1]['payload']['automatic_retry'] is False


def test_reviewed_resources_reach_tool_and_failed_dispatch_releases_taskline_lease(factory, tmp_path):
    output = tmp_path / 'generated'
    contract = node('generate_cof', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 2, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 2}])
    contract['agent'] = 'harness'
    rt = factory([contract])
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-1',
        'resource_allocations': {'generate_cof': {
            'memory_mb': 8192, 'cpus': 4, 'nodelist': 'node03', 'partition': 'debug',
        }},
    }
    calls = []
    rt.main.registry.get('generate_structure').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled generator failure after dispatch entry',
    }
    rt.start(1)
    rt.tick()
    eventually(lambda: bool(calls))
    eventually(lambda: rt.snapshot()['nodes']['generate_cof']['status'] == 'failed')
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    assert calls[0]['memory_mb'] == 8192
    assert calls[0]['cpus_per_task'] == 4
    line_node = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert line_node['resource_allocation']['nodelist'] == 'node03'
    assert line_node['effective_arguments']['partition'] == 'debug'
    assert line_node['status'] == 'failed'
    assert line_node['validation']['runtime_token'] is None
    assert line_node['validation']['resource_leases'] == []
    assert not rt.store.snapshot().get('leases')


def test_resource_projection_failure_does_not_block_approved_tool(factory, tmp_path):
    output = tmp_path / 'generated-projection-fallback'
    contract = node('generate_cof', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 1, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 1}])
    contract['agent'] = 'harness'
    rt = factory([contract], cid='projection-fallback')
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-fallback',
        'resource_allocations': {'generate_cof': {
            'memory_mb': 4096, 'cpus': 2, 'nodelist': 'node03', 'partition': 'debug',
        }},
    }
    original = rt.task_lines.upsert_step
    def projection_write(*args, **kwargs):
        if 'resource_allocation' in kwargs:
            raise TypeError('simulated older TaskLine projection schema')
        return original(*args, **kwargs)
    rt.task_lines.upsert_step = projection_write
    calls = []
    rt.main.registry.get('generate_structure').execute = lambda args: calls.append(copy.deepcopy(args)) or {
        'error': 'controlled failure proves tool entry',
    }
    rt.start(1)
    rt.tick()
    eventually(lambda: bool(calls))
    for future in list(rt.futures.values()):
        future.result(timeout=5)
    state = rt.snapshot()['nodes']['generate_cof']
    assert state['resource_allocation']['memory_mb'] == 4096
    assert 'older TaskLine projection schema' in state['metadata_sync_error']
    assert calls[0]['memory_mb'] == 4096 and calls[0]['cpus_per_task'] == 2


def node(key, path=None, deps=(), locks=(), tool='read_file', args=None, outputs=()):
    return {'step_id': key, 'agent': 'adsorption' if tool == 'run_henry' else 'analyst', 'tool': tool,
            'arguments': args or {'path': path or key}, 'depends_on': list(deps),
            'expected_outputs': list(outputs), 'resource_locks': list(locks)}


def eventually(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate(): return
        time.sleep(.01)
    assert predicate(), 'condition did not become true before deadline'


def verify_prefinish(rt,step_id):
    """Unit proof fixture; real model decisions are covered by SDK handoff."""
    state=rt.snapshot()['nodes'][step_id]
    assert state['status']=='prefinish' and state.get('phase')=='prefinish'
    paths=list(state.get('artifacts',{}))
    assert paths and all(state['artifacts'].values())
    call_id='verification'+step_id
    write_checkpoint(rt.root/'evidence'/f'{call_id}.json',{'call_id':call_id,'time':time.time(),'tool':'read_file',
        'params':{'path':paths[0]},'result':'{"ok":true,"content":"actual verification fixture"}','failed':False})
    rt.finish_node(step_id,[call_id],'Test verifier checked fresh native output against the original contract.')


@pytest.fixture
def factory(tmp_path):
    runtimes = []
    def build(nodes, cid='c1', username='tester'):
        root = tmp_path / 'runs' / username / cid
        root.mkdir(parents=True)
        main = Session(config=AgentConfig(api_key='test', project_root=tmp_path), registry=_build_default_registry())
        main._current_line_id = cid
        main.goal_contract = GoalContract.from_user_message('hello')
        main.goal_contract.execution_authorized = True
        main.goal_contract.approved_plan_version = 1
        main.goal_contract.approved_nodes = copy.deepcopy(nodes)
        lines = TaskLineStore(str(tmp_path / (cid + '-lines.json')))
        line = lines.begin_line(cid, username=username, conv_id=cid)
        main._current_line_id = line['line_id']
        lines.apply_workflow_patch(line['line_id'], [{'operation': 'upsert', 'step_id': n['step_id'], 'node': n} for n in nodes], 0, 1, 'test approved DAG')
        mailbox = LifecycleStore(root / 'lifecycle.json')
        mailbox.initialize()
        runtime = ParallelWorkflow(main, root, WorkflowStore(tmp_path / 'parallel.json'), mailbox, lines, username, cid)
        runtime.main._runtime_snapshot = runtime.snapshot
        runtimes.append(runtime)
        return runtime
    yield build
    for runtime in runtimes:
        for future in runtime.futures.values():
            if not future.done(): future.result(timeout=6)
        runtime.shutdown()


def test_independent_branches_really_overlap_and_join_waits(factory):
    rt = factory([node('a'), node('b'), node('join', deps=['a', 'b'])])
    entered, release = threading.Barrier(3), threading.Event()
    executed = []
    def tool(args):
        executed.append(args['path'])
        if args['path'] != 'join':
            entered.wait(timeout=4)
            assert release.wait(4)
        return {'content': args['path']}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1)
    rt.tick()
    try:
        entered.wait(timeout=4)  # A/B and this thread must reach the barrier together.
        assert set(executed) == {'a', 'b'}
        assert rt.snapshot()['nodes']['join']['status'] == 'pending'
    finally: release.set()
    eventually(lambda: all(rt.snapshot()['nodes'][n]['status'] == 'succeeded' for n in ('a', 'b')))
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    join = rt.snapshot()['nodes']['join']
    saved = json.loads(Path(join['checkpoint_path']).read_text())
    assert set(saved['context']['dependency_results']) == {'a', 'b'}
    assert len({n['checkpoint_path'] for n in rt.snapshot()['nodes'].values()}) == 3
    assert rt.main.messages == [] and rt.main.memory.tool_call_log == []


def test_shared_resource_serializes_across_conversations_not_everything(factory):
    a = factory([node('a', locks=['global:same-output'])], 'a')
    b = factory([node('b', locks=['global:same-output']), node('unrelated')], 'b')
    entered, release = threading.Event(), threading.Event()
    def hold(args):
        entered.set()
        assert release.wait(4)
        return {'content': 'first'}
    a.main.registry.get('read_file').execute = hold
    b.main.registry.get('read_file').execute = lambda p: {'content': p['path']}
    a.start(1); b.start(1)
    a.tick()
    try:
        assert entered.wait(4)
        b.tick()
        eventually(lambda: b.snapshot()['nodes']['unrelated']['status'] == 'succeeded')
        assert b.snapshot()['nodes']['b']['status'] == 'pending'
    finally: release.set()
    eventually(lambda: a.snapshot()['status'] == 'completed')
    b.tick()
    eventually(lambda: b.snapshot()['status'] == 'completed')


@pytest.mark.parametrize('mode_a,mode_b,expected', [('read','read',False),('write','read',True),('read','write',True),('write','write',True)])
def test_parent_child_paths_and_shared_reads(mode_a, mode_b, expected):
    assert conflicts({'key':'path:/project/output','mode':mode_a}, {'key':'path:/project/output/result.csv','mode':mode_b}) == expected
    assert not conflicts({'key':'path:/project/output','mode':mode_a}, {'key':'path:/project/output-other','mode':mode_b})


def process_claim(path, wid, event, queue):
    event.wait(5)
    queue.put(bool(WorkflowStore(path).claim(wid, 'a')))


def test_multiple_processes_only_one_claims_node_and_all_resources(factory):
    rt = factory([node('a', locks=['r1', 'r2'])])
    rt.start(1)
    ctx = mp.get_context('fork')
    event, queue = ctx.Event(), ctx.Queue()
    processes = [ctx.Process(target=process_claim, args=(rt.store.path, rt.workflow_id, event, queue)) for _ in range(4)]
    for process in processes: process.start()
    event.set()
    assert sum(queue.get(timeout=6) for _ in processes) == 1
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0 and not process.is_alive()
    leases = rt.store.snapshot()['leases']
    assert len(leases) == 1 and len(next(iter(leases.values()))['resources']) == 3
    queue.close(); queue.join_thread()


def test_running_messages_notify_main_supervisor_and_worker_receipts(factory):
    rt = factory([node('a')])
    entered, release = threading.Event(), threading.Event()
    def tool(args):
        entered.set(); assert release.wait(4)
        return {'content': 'actual result'}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1); rt.tick()
    try:
        assert entered.wait(4)
        message = rt.send('a', 'Please retain input provenance', 'comment', 'same-id')
        assert rt.send('a', message['text'], 'comment', 'same-id') == message
        event = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_user_message')
        assert event['main_chat'] == event['supervisor'] == 'pending'
        assert event['payload']['message']['text'] == message['text']
    finally: release.set()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    state = rt.snapshot()['nodes']['a']
    assert state['messages']['same-id']['receipt'] == 'seen_at_tool_boundary'
    saved = json.loads(Path(state['checkpoint_path']).read_text())
    assert saved['context']['received_messages']['same-id']['text'] == message['text']


def test_change_request_does_not_mutate_running_parameters_or_start_descendant(factory):
    rt = factory([node('a'), node('next', deps=['a'])])
    entered, release = threading.Event(), threading.Event()
    seen = []
    def tool(args):
        seen.append(copy.deepcopy(args)); entered.set(); assert release.wait(4)
        return {'content':'old approved params'}
    rt.main.registry.get('read_file').execute = tool
    rt.start(1); rt.tick()
    try:
        assert entered.wait(4)
        assert rt.send('a', 'change input to other.cif', 'change')['receipt'] == 'awaiting_main_chat_approval'
    finally: release.set()
    eventually(lambda: rt.snapshot()['nodes']['a']['status'] == 'succeeded')
    rt.tick()
    assert rt.snapshot()['status'] == 'needs_user'
    assert rt.snapshot()['nodes']['next']['status'] == 'pending'
    assert seen == [{'path':'a'}]


def test_submitted_job_keeps_lease_unknown_cannot_release_then_fresh_output_unlocks(factory, tmp_path):
    output = tmp_path / 'fresh.csv'
    rt = factory([node('calc', tool='run_henry', args={'cif':'x.cif','gas':'Kr','temperature':298}, outputs=[str(output)])])
    rt.main.registry.get('run_henry').execute = lambda p: {'submitted':True,'job_id':'42','work_dir':str(tmp_path)}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs')
    assert len(rt.store.snapshot()['leases']) == 1
    rt.tick([{'job_id':'42','state':'UNKNOWN','terminal':True,'failed':False}])
    assert rt.snapshot()['nodes']['calc']['status'] == 'waiting_jobs'
    assert len(rt.store.snapshot()['leases']) == 1
    output.write_text('actual new output')
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
    assert rt.snapshot()['status']=='active'  # exit alone never finishes the task
    verify_prefinish(rt,'calc')
    assert rt.snapshot()['status'] == 'completed'
    assert not rt.store.snapshot()['leases']


def test_generator_can_revalidate_real_split_output_directories(factory, tmp_path):
    output = tmp_path / 'runs' / 'tester' / 'c1' / 'generated'
    contract = node('gen', tool='generate_structure', args={
        'material_type': 'COF', 'n_structures': 2, 'output_dir': str(output),
    }, outputs=[{'kind': 'directory', 'path': str(output), 'pattern': '*.cif', 'min_count': 2}])
    contract['agent'] = 'harness'
    rt = factory([contract])
    rt.job_watch = SimpleNamespace(
        list=lambda: [],
        get=lambda job_id: {
            'username': 'tester', 'conv_id': 'c1', 'terminal': True,
            'failed': False, 'state': 'COMPLETED', 'work_dir': str(output),
        },
    )
    rt.main._on_resource_review = lambda request: {
        'status': 'ready', 'resource_review_id': 'review-generator',
        'resource_allocations': {'gen': {'memory_mb': 4096, 'cpus': 2}},
    }
    rt.main.registry.get('generate_structure').execute = lambda args: {
        'submitted': True, 'job_id': '42', 'output_dir': str(output),
    }
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['gen']['status'] != 'running')
    current = rt.snapshot()['nodes']['gen']
    assert current['status'] == 'waiting_jobs', (current['status'], current.get('error'))
    (output / 'large').mkdir(parents=True)
    (output / 'small').mkdir()
    (output / 'large' / 'a.cif').write_text('data_a')
    (output / 'small' / 'b.cif').write_text('data_b')
    rt.tick([{'job_id': '42', 'state': 'COMPLETED', 'terminal': True, 'failed': False}])
    assert rt.snapshot()['nodes']['gen']['status'] == 'prefinish'

    result = rt.revalidate_outputs('gen', [
        {'kind': 'directory', 'path': str(output / 'large'), 'pattern': '*.cif', 'min_count': 1},
        {'kind': 'directory', 'path': str(output / 'small'), 'pattern': '*.cif', 'min_count': 1},
    ], completed_job_id='42')

    assert result['ok'] and result['verified_files'] == 2
    assert rt.snapshot()['nodes']['gen']['status'] == 'prefinish'
    verify_prefinish(rt, 'gen')
    assert rt.snapshot()['nodes']['gen']['status'] == 'succeeded'


@pytest.mark.parametrize('preexisting', [False, True])
def test_missing_or_old_output_blocks_dependency_but_allows_repair(factory, tmp_path, preexisting):
    output = tmp_path / 'old.csv'
    if preexisting:
        output.write_text('old task')
        os.utime(output, (1,1))
    rt = factory([node('a', outputs=[str(output)]), node('next', deps=['a'])])
    rt.main.registry.get('read_file').execute = lambda p: {'content':'no new output'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    assert rt.snapshot()['nodes']['a']['status'] == 'validation_failed'
    assert not rt.store.snapshot()['leases']
    rt.tick()
    assert rt.snapshot()['nodes']['next']['status'] == 'pending'


def test_real_error_preserved_and_no_automatic_retry(factory):
    rt = factory([node('a'), node('next', deps=['a'])])
    calls = []
    rt.main.registry.get('read_file').execute = lambda p: calls.append(p) or {'error':'real CIF parser failure'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    for _ in range(3): rt.tick()
    assert len(calls) == 1 and rt.snapshot()['nodes']['next']['status'] == 'pending'
    event = next(e for e in rt.lifecycle.pending() if e['kind'] == 'worker_failed')
    assert event['payload']['error'] == 'real CIF parser failure'
    assert event['payload']['evidence_call']['call_id']
    assert Path(event['payload']['evidence_call']['evidence_path']).is_file()


def test_dead_readonly_worker_restored_without_killing_main_role(factory):
    rt = factory([node('a')])
    rt.main.registry.get('read_file').execute = lambda p: {'content':'restored'}
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {'owner_pid':99999999})
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    rt.lifecycle.heartbeat()
    assert rt.lifecycle.snapshot()['agents']['main_chat']['lifetime']
    assert rt.lifecycle.snapshot()['agents']['supervisor']['lifetime']


def test_dead_mutating_worker_unknown_keeps_lease_and_never_redispatches(factory):
    rt = factory([node('a', tool='write_file', args={'path':'x','content':'new'})])
    rt.main.registry.get('write_file').execute = lambda p: pytest.fail('unknown mutation must not redispatch')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {'owner_pid':99999999,'dispatch_phase':'entered_tool'})
    rt.tick(); rt.tick()
    assert rt.snapshot()['status'] == 'needs_user'
    assert rt.snapshot()['nodes']['a']['status'] == 'uncertain'
    assert len(rt.store.snapshot()['leases']) == 1


def test_dead_preflight_writer_is_failed_not_a_permanently_unreconcilable_unknown(factory):
    rt=factory([node('a',tool='write_file',args={'path':'x','content':'new'})]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'owner_pid':99999999})
    rt.tick()
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert rt.snapshot()['status']=='needs_user' and not rt.store.snapshot()['leases']


def test_storage_failure_blocks_dispatch(factory, monkeypatch):
    rt = factory([node('a')])
    rt.start(1)
    rt.main.registry.get('read_file').execute = lambda p: pytest.fail('no dispatch without checkpoint')
    monkeypatch.setattr('agents.parallel_workflow.write_checkpoint', lambda *a: (_ for _ in ()).throw(OSError('disk full')))
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'needs_user')
    assert rt.snapshot()['nodes']['a']['status'] in {'failed','uncertain'}


def test_lifetime_heartbeat_busy_stall_warning_is_idempotent_and_mailbox_survives(factory):
    rt = factory([node('a')])
    rt.lifecycle.emit('task', {}, event_id='busy')
    assert rt.lifecycle.claim('busy','main_chat')
    with json_transaction(rt.lifecycle.path) as data:
        data['events']['busy']['main_chat_claimed_at'] = time.time()-400
    rt.lifecycle.heartbeat(); rt.lifecycle.heartbeat()
    state = rt.lifecycle.snapshot()
    assert state['agents']['main_chat']['busy_event_id'] == 'busy'
    assert len([e for e in state['events'].values() if e['kind']=='role_unresponsive']) == 1
    restored = LifecycleStore(rt.lifecycle.path)
    restored.recover_claims()
    assert restored.snapshot()['events']['busy']['main_chat'] == 'processing'  # live owner is never stolen


def test_wrong_version_unapproved_and_legacy_jobs_do_not_run(factory):
    rt = factory([node('a')])
    with pytest.raises(ValueError, match='exact approved'): rt.start(2)
    rt.main.goal_contract.execution_authorized = False
    with pytest.raises(ValueError, match='approval'): rt.start(1)
    rt.main.goal_contract.execution_authorized = True
    rt.task_lines.upsert_step('c1','a',job_ids=['old'],status='submitted')
    with pytest.raises(ValueError, match='legacy'): rt.start(1)


def test_busy_main_api_still_accepts_targeted_worker_messages(factory, monkeypatch):
    pytest.importorskip('fastapi')
    import api
    from fastapi.testclient import TestClient
    rt = factory([node('a')]); rt.start(1)
    conv = SimpleNamespace(is_processing=True, interrupt_requested=False)
    monkeypatch.setattr(api.session_manager, 'get_user', lambda u: SimpleNamespace(get_conversation=lambda cid: conv if cid=='c1' else None))
    monkeypatch.setattr(api, '_ensure_workflow_runtime', lambda c: rt)
    api.app.dependency_overrides[api._get_user] = lambda: 'tester'
    try:
        client = TestClient(api.app)
        response = client.post('/api/conversations/c1/workflow/nodes/a/messages',json={'text':'status please','kind':'status'})
        assert response.status_code == 200 and response.json()['receipt'] == 'status_returned'
        assert client.post('/api/conversations/c1/workflow/execute',json={'plan_version':1}).status_code == 409
        assert client.post('/api/conversations/other/workflow/nodes/a/messages',json={'text':'x'}).status_code == 404
        assert any(e['kind']=='worker_user_message' for e in rt.lifecycle.pending())
    finally: api.app.dependency_overrides.pop(api._get_user,None)


def test_message_outbox_recovers_crash_between_state_and_mailbox(factory, monkeypatch):
    rt = factory([node('a')]); rt.start(1)
    emit = rt.lifecycle.emit
    monkeypatch.setattr(rt.lifecycle, 'emit', lambda *a, **kw: (_ for _ in ()).throw(OSError('mailbox unavailable')))
    with pytest.raises(OSError): rt.send('a','retain this message','comment','durable-message')
    assert rt.snapshot()['nodes']['a']['messages']['durable-message']['text'] == 'retain this message'
    assert any(not e['sent'] for e in rt.snapshot()['outbox'].values())
    monkeypatch.setattr(rt.lifecycle, 'emit', emit)
    rt.flush_events(); rt.flush_events()
    events = [e for e in rt.lifecycle.pending() if e['kind']=='worker_user_message']
    assert len(events) == 1 and all(e['sent'] for e in rt.snapshot()['outbox'].values())


def test_helper_error_after_job_acceptance_never_releases_live_resource(factory):
    rt = factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    calls=[]
    rt.main.registry.get('run_henry').execute = lambda p: calls.append(p) or {'submitted':True,'job_id':'42','error':'helper failed after scheduler accepted'}
    rt.start(1); rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['a']['status']=='uncertain')
    assert rt.store.snapshot()['leases']
    rt.tick([{'job_id':'42','state':'RUNNING','terminal':False,'failed':False}])
    assert rt.store.snapshot()['leases'] and len(calls)==1
    rt.tick([{'job_id':'42','state':'FAILED','terminal':True,'failed':True}])
    assert rt.snapshot()['nodes']['a']['status']=='failed' and not rt.store.snapshot()['leases']
    assert len(calls)==1


def test_restart_waiting_job_does_not_call_tool_again(factory, tmp_path):
    output=tmp_path/'result.csv'
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=[str(output)])])
    calls=[]
    rt.main.registry.get('run_henry').execute=lambda p:calls.append(p) or {'submitted':True,'job_id':'42'}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    restored=ParallelWorkflow(rt.main,rt.root,rt.store,rt.lifecycle,rt.task_lines,'tester','c1')
    try:
        restored.tick([{'job_id':'42','state':'RUNNING','terminal':False,'failed':False}])
        assert len(calls)==1 and restored.snapshot()['nodes']['a']['status']=='waiting_jobs'
        output.write_text('new real artifact')
        restored.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
        verify_prefinish(restored,'a')
        assert restored.snapshot()['status']=='completed' and len(calls)==1
    finally:restored.shutdown()


def test_patch_preserves_verified_unaffected_branch_and_repairs_failed_node(factory):
    rt=factory([node('a'),node('b',deps=['a'])])
    calls=[]
    entered,release=threading.Event(),threading.Event()
    def execute(p):
        calls.append(p['path'])
        if p['path']=='a':
            entered.set();assert release.wait(4)
        return {'content':'ok'}
    rt.main.registry.get('read_file').execute=execute
    rt.start(1);rt.tick()
    try:
        assert entered.wait(4)
        rt.store.pause(rt.workflow_id,'repair remaining branch')
    finally:release.set()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='succeeded')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    replacement=node('b',path='fixed',deps=['a'])
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'upsert','step_id':'b','node':replacement}],1,2,'approved node repair')
    rt.main.goal_contract.approved_nodes=line['steps']
    rt.main.goal_contract.approved_plan_version=2
    rt.main.goal_contract.version+=1
    rt.start(2);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')
    assert calls==['a','fixed']


def test_new_plan_with_identical_failed_node_does_not_retry(factory):
    rt = factory([node('a')])
    calls = []
    def fail(params):
        calls.append(params)
        return {'error': 'Fixture failure requires a concrete repair'}
    rt.main.registry.get('read_file').execute = fail
    rt.start(1)
    rt.tick()
    eventually(lambda: rt.snapshot()['nodes']['a']['status'] == 'failed')
    original = rt.main.goal_contract.approved_nodes[0]
    line = rt.task_lines.apply_workflow_patch('c1', [{'operation': 'upsert', 'step_id': 'a', 'node': original}], 1, 2, 'identical patch must not act as retry')
    rt.main.goal_contract.approved_nodes = line['steps']
    rt.main.goal_contract.approved_plan_version = 2
    receipt = rt.start(2)
    rt.tick()
    assert receipt['status'] == 'needs_recovery' and not receipt['scheduled']
    assert rt.snapshot()['nodes']['a']['status'] == 'failed' and len(calls) == 1


def test_resource_contract_includes_absolute_output_even_when_not_in_tool_args(tmp_path):
    outputs=resources_for(node('a',outputs=[str(tmp_path/'shared.csv')]),tmp_path,tmp_path/'conv')
    assert {'key':'path:'+str(tmp_path/'shared.csv'),'mode':'write'} in outputs


def test_long_workflow_digest_is_bounded_without_losing_global_counts():
    from agents.parallel_workflow import runtime_summary
    state={'status':'active','plan_version':1,'nodes':{str(i):{'status':'succeeded' if i else 'running',
           'messages':{'long':{'text':'x'*100000}},'error':'e'*10000} for i in range(1000)}}
    summary=runtime_summary(state)
    assert summary['counts']=={'running':1,'succeeded':999}
    assert len(summary['nodes'])==24 and summary['omitted_nodes']==976
    assert len(json.dumps(summary))<18000


def test_api_dispatcher_runs_workers_and_delivers_evidence_when_busy_main_returns(factory, monkeypatch):
    pytest.importorskip('fastapi')
    import asyncio
    import api
    from anthropic.types import TextBlock
    rt=factory([node('a')])
    rt.main._lifecycle_store=rt.lifecycle
    rt.main._evidence_root=rt.root/'evidence'
    rt.main.registry.get('read_file').execute=lambda p:{'content':'actual worker evidence'}
    rt.main._call_api=lambda agent:SimpleNamespace(content=[TextBlock(type='text',text='Read validation results are available.')])
    conv=SimpleNamespace(session=rt.main,username='tester',conv_id='c1',is_processing=True,
        interrupt_requested=False,interrupt_message='',last_progress_at=0,current_uid='busy',current_status='busy',execution_logs=[])
    monkeypatch.setattr(api.session_manager,'list_users',lambda:['tester'])
    monkeypatch.setattr(api.session_manager,'get_user',lambda u:SimpleNamespace(conversations={'c1':conv}))
    monkeypatch.setattr(api,'_ensure_workflow_runtime',lambda c:rt)
    monkeypatch.setattr(api._watch,'list',lambda:[])
    monkeypatch.setattr(api,'_lifecycle_tasks',set())
    async def supervise(c,event):
        rt.lifecycle.receipt(event['event_id'],'supervisor','delivered',{'next_action':'advance_dependencies','reason':'test read-only receipt'})
    async def run_main(username,query,cid,lifecycle_event=None):
        rt.main.resume_lifecycle_event(lifecycle_event)
        c=conv;c.current_status='delivered';c.is_processing=False
    monkeypatch.setattr(api,'_supervise_lifecycle_event',supervise)
    monkeypatch.setattr(api,'run_agent_background',run_main)
    async def dispatch():
        await api._dispatch_lifecycle_once()
        if api._lifecycle_tasks:await asyncio.gather(*list(api._lifecycle_tasks))
    async def scenario():
        rt.start(1)
        await dispatch()
        for _ in range(200):
            if rt.snapshot()['status']=='completed':break
            await asyncio.sleep(.01)
        assert rt.snapshot()['status']=='completed'
        await dispatch();await dispatch()
        event=next(e for e in rt.lifecycle.pending() if e['kind']=='worker_completed')
        assert event['main_chat']=='pending'  # busy main does not lose or prematurely ACK it
        conv.is_processing=False
        await dispatch();await dispatch()
        assert rt.lifecycle.snapshot()['events'][event['event_id']]['main_chat']=='delivered'
        assert rt.main.memory.tool_call_log[0]['call_id']==event['payload']['evidence_call']['call_id']
        assert rt.main.memory.tool_call_log[0]['evidence_path']
    asyncio.run(scenario())


def test_shared_worker_evidence_deduplicates_and_keeps_original_agent_partition(factory):
    rt=factory([node('a')])
    rt.main._execute_loop=lambda agent:'received'
    call={'call_id':'shared-once','agent':'analyst','tool':'read_file','params':{'path':'x'},'result':'actual evidence'}
    event={'kind':'worker_completed','payload':{'evidence_call':call},'supervisor_receipt':{'evidence_calls':[call]}}
    rt.main.resume_lifecycle_event(event)
    rt.main.resume_lifecycle_event(event)
    assert len(rt.main.memory.tool_call_log)==1
    assert len(rt.main.memory.agent_memories['analyst']['tool_call_log'])==1
    assert not rt.main.memory.agent_memories.get('supervisor',{}).get('tool_call_log')


def test_user_freeze_survives_tick_and_explicit_start_resumes(factory):
    rt=factory([node('a')])
    rt.main.registry.get('read_file').execute=lambda p:{'content':'safe resumption'}
    rt.start(1)
    rt.store.user_pause(rt.workflow_id)
    assert rt.store.claim(rt.workflow_id,'a') is None
    rt.tick(paused=False)
    assert rt.snapshot()['nodes']['a']['status']=='pending'
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')


def test_deleting_main_cannot_orphan_live_owner_but_can_retire_unstarted_plan(factory):
    rt=factory([node('a')]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    with pytest.raises(ValueError,match='running/unknown'):rt.store.retire(rt.workflow_id)
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'status':'pending'},release=True)
    rt.store.retire(rt.workflow_id)
    assert rt.store.claim(rt.workflow_id,'a') is None
    assert rt.snapshot()['status']=='retired_by_user'


def test_global_admission_budget_is_not_multiplied_by_api_processes(factory):
    rt=factory([node('a'),node('b'),node('c')]);rt.start(1)
    store=WorkflowStore(rt.store.path,max_running=2)
    assert store.claim(rt.workflow_id,'a')
    assert store.claim(rt.workflow_id,'b')
    other_process_view=WorkflowStore(rt.store.path,max_running=2)
    assert other_process_view.claim(rt.workflow_id,'c') is None


def test_worker_initialization_failure_keeps_main_alive_and_does_not_hold_phantom_writer(factory, monkeypatch):
    rt=factory([node('a')]);rt.start(1)
    def broken_session(*a,**kw):raise RuntimeError('worker Session initialization failed')
    monkeypatch.setattr('agents.session.Session',broken_session)
    rt.tick()
    eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert not rt.store.snapshot()['leases']
    assert rt.main.goal_contract.original_goal=='hello'


def test_actual_main_loop_can_start_dag_after_old_handoff_threshold(factory, monkeypatch):
    from agents.agent import Agent
    from anthropic.types import ToolUseBlock,TextBlock
    rt=factory([node('a')])
    monkeypatch.setattr('agents.task_line.get_store',lambda:rt.task_lines)
    for _ in range(5):rt.main.memory.record_tool_call('lead-orchestrator','task_line_query',{},'{"ok":true}')
    def execute_workflow(**kwargs):raise AssertionError('Session must use its runtime callback')
    agent=Agent(name='lead-orchestrator',functions=[execute_workflow])
    rt.main.messages=[{'role':'user','content':'hello'}]
    rt.main._on_workflow_start=rt.start
    responses=iter([SimpleNamespace(content=[ToolUseBlock(type='tool_use',id='start',name='execute_workflow',input={'plan_version':1})]),
        SimpleNamespace(content=[TextBlock(type='text',text='Runtime has started.')])])
    rt.main._call_api=lambda agent:next(responses)
    rt.main._execute_loop(agent)
    assert rt.snapshot()['status']=='active' and not rt.main.task_complete
    assert rt.main.memory.tool_call_log[-1]['tool']=='execute_workflow'
    assert [n['tool'] for n in rt.task_lines.get_line('c1')['steps']]==['read_file']


def test_unknown_local_writer_has_user_negotiated_repair_exit_not_permanent_stall(factory,tmp_path):
    from agents.defns import ORCHESTRATOR
    target=tmp_path/'partial.txt';target.write_text('partial old value')
    rt=factory([node('a',tool='write_file',args={'path':str(target),'content':'correct value'})]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'owner_pid':99999999,'dispatch_phase':'entered_tool'})
    rt.tick()
    with pytest.raises(ValueError,match='genuine user'):rt.resolve_local_write('a','invented','invented')
    rt.main.current_agent=ORCHESTRATOR
    rt.main._request_user_decision('确认原写入已停止，是否允许检查后修补？','write_file stopped after crash; inspect original path',['a'])
    decision_id=rt.main._pending_user_interaction['params']['decision_id']
    rt.main._execute_loop=lambda agent,**kwargs:'main chat received genuine user approval'
    rt.main.reply('确认原写入已停止，同意修补',verbose=False)
    params={'path':str(target)}
    result=rt.main.registry.execute_dict('read_file',params)
    rt.main.memory.record_tool_call('lead-orchestrator','read_file',params,json.dumps(result))
    receipt=rt.resolve_local_write('a',decision_id,rt.main.memory.tool_call_log[-1]['call_id'])
    assert receipt['lease_released'] and not receipt['resubmitted']
    assert rt.snapshot()['nodes']['a']['status']=='failed' and rt.snapshot()['status']=='needs_user'
    assert not rt.store.snapshot()['leases'] and target.read_text()=='partial old value'


def test_two_scientific_submissions_with_disjoint_workdirs_really_overlap(factory,tmp_path):
    entered,release=threading.Barrier(3),threading.Event()
    nodes=[]
    for gas in ('Kr','Xe'):
        work=tmp_path/gas;work.mkdir()
        nodes.append(node(gas,tool='run_cdft',args={'action':'pipeline','cif_dir':str(tmp_path/'inputs'),
            'gas':gas,'temperature':298,'job_work_dir':str(work)},outputs=[str(work/'result.csv')]))
    rt=factory(nodes)
    seen=[]
    def compute(args):
        seen.append(args['gas']);entered.wait(timeout=4);assert release.wait(4)
        return {'submitted':True,'job_id':'42' if args['gas']=='Kr' else '43','work_dir':args['job_work_dir']}
    rt.main.registry.get('run_cdft').execute=compute
    rt.start(1);rt.tick()
    try:
        entered.wait(timeout=4)
        assert set(seen)=={'Kr','Xe'}
        assert len(rt.store.snapshot()['leases'])==2
    finally:release.set()
    eventually(lambda:all(n['status']=='waiting_jobs' for n in rt.snapshot()['nodes'].values()))
    for gas in ('Kr','Xe'):(tmp_path/gas/'result.csv').write_text('fresh simulated output')
    rt.tick([{'job_id':jid,'state':'COMPLETED','terminal':True,'failed':False} for jid in ('42','43')])
    for gas in ('Kr','Xe'):verify_prefinish(rt,gas)
    assert rt.snapshot()['status']=='completed' and not rt.store.snapshot()['leases']


def test_native_scheduler_receipt_binds_generic_output_csv_without_resubmission(factory, tmp_path):
    """A completed native job may repair only its approved session CSV location."""
    actual = tmp_path / 'runs' / 'tester' / 'c1' / 'pore_results.csv'
    approved = str(actual)
    # This reproduces a legacy contract whose project-relative output was later
    # interpreted relative to the tool work directory.
    legacy_output = 'runs/tester/c1/pore_results.csv'
    contract = node('pore', tool='run_pore_analysis', args={
        'cif_dir': str(tmp_path / 'runs' / 'tester' / 'c1' / 'subset_cifs'),
        'output_csv': approved,
    }, outputs=[legacy_output])
    rt = factory([contract])
    rt.start(1)
    claimed = rt.store.claim(rt.workflow_id, 'pore')
    assert claimed
    token = claimed['token']
    receipt = {'submitted': True, 'job_id': '77744', 'output_csv': approved,
               'work_dir': str(actual.parent / 'pore')}
    rt.store.update(rt.workflow_id, 'pore', token, {
        'status': 'prefinish', 'job_ids': ['77744'], 'jobs_confirmed_terminal': True, 'result': receipt,
        'result_ref': {'call_id': 'nativecall'},
        'evidence_call': {'call_id': 'nativecall', 'result': receipt},
    })

    assert rt._bind_native_result_path('pore', token, receipt)
    repaired = rt.snapshot()['nodes']['pore']
    assert repaired['contract']['expected_outputs'] == [{'kind': 'file', 'path': approved}]
    assert repaired['output_contract_history'][-1]['job_id'] == '77744'
    assert repaired['output_contract_history'][-1]['no_resubmission'] is True
    assert repaired['job_ids'] == ['77744']
    line_step = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert line_step['expected_outputs'] == [{'kind': 'file', 'path': approved}]

    # Including the exact execution receipt alongside real validation is a
    # harmless model habit. It must not poison the finish request, and it must
    # not itself be stored as scientific validation.
    actual.write_text('material,diameter\nMOF,12.3\n')
    evidence = rt.root / 'evidence'
    evidence.mkdir(exist_ok=True)
    write_checkpoint(evidence / 'nativecall.json', {'call_id': 'nativecall', 'time': time.time(),
        'tool': 'run_pore_analysis', 'params': contract['arguments'], 'result': receipt, 'failed': False})
    write_checkpoint(evidence / 'readcall.json', {'call_id': 'readcall', 'time': time.time(),
        'tool': 'read_file', 'params': {'path': approved}, 'result': {'content': actual.read_text()}, 'failed': False})
    settled = []
    rt._settle = lambda *args: settled.append(args)
    rt.finish_node('pore', ['nativecall', 'readcall'], 'CSV contains the expected real pore-analysis row.')
    verification = rt.snapshot()['nodes']['pore']['node_verification']
    assert verification['evidence_call_ids'] == ['readcall']
    assert settled and settled[0][0] == 'pore'


def test_native_scheduler_receipt_binds_output_directory_and_preserves_pattern(factory, tmp_path):
    actual = tmp_path / 'runs' / 'tester' / 'c1' / 'charged_cifs'
    contract = node('charge', tool='run_pacman_charge', args={
        'cif_dir': str(tmp_path / 'runs' / 'tester' / 'c1' / 'subset_cifs'),
        'method': 'pacmof', 'output_dir': str(actual),
    }, outputs=[{'kind': 'directory', 'path': 'runs/tester/c1/charged_cifs',
                 'pattern': '*_pacmof.cif', 'min_count': 6}])
    rt = factory([contract])
    actual.mkdir()
    rt.start(1)
    claimed = rt.store.claim(rt.workflow_id, 'charge')
    receipt = {'submitted': True, 'job_id': '77749', 'output_dir': str(actual),
               'work_dir': str(actual)}
    rt.store.update(rt.workflow_id, 'charge', claimed['token'], {
        'status': 'prefinish', 'job_ids': ['77749'], 'result': receipt,
        'result_ref': {'call_id': 'chargecall'},
        'evidence_call': {'call_id': 'chargecall', 'result': receipt},
    })
    assert rt._bind_native_result_path('charge', claimed['token'], receipt)
    expected = rt.snapshot()['nodes']['charge']['contract']['expected_outputs']
    assert expected == [{'kind': 'directory', 'path': str(actual),
                         'pattern': '*_pacmof.cif', 'min_count': 6}]
    assert rt.snapshot()['nodes']['charge']['job_ids'] == ['77749']


def test_repaired_parent_invalidates_identical_child_contract_not_just_changed_arguments(factory):
    rt=factory([node('a'),node('child',deps=['a'])])
    calls=[]
    rt.main.registry.get('read_file').execute=lambda p:calls.append(p['path']) or {'content':'validated'}
    rt.start(1)
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='completed':break
        time.sleep(.01)
    assert rt.snapshot()['status']=='completed'
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    replacement=node('a',path='repaired-parent')
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'upsert','step_id':'a','node':replacement}],1,2,'user approved parent repair')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['nodes']['child']['status']=='pending'  # same args/dependency ID, different parent generation
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='completed':break
        time.sleep(.01)
    assert calls.count('child')==2 and rt.snapshot()['status']=='completed'


def test_same_client_message_id_on_different_nodes_does_not_hide_second_notification(factory):
    rt=factory([node('a'),node('b')]);rt.start(1)
    rt.send('a','A status','status','client-id')
    rt.send('b','B status','status','client-id')
    events=[e for e in rt.lifecycle.pending() if e['kind']=='worker_user_message']
    assert len(events)==2
    assert {e['payload']['message']['step_id'] for e in events}=={'a','b'}


def test_api_does_not_label_idle_main_as_completed_while_branches_work(factory):
    pytest.importorskip('fastapi')
    import api
    rt=factory([node('a')]);rt.start(1)
    assert api._execution_status(rt.main)=='编排分支运行中（主chat在岗）'
    rt.store.user_pause(rt.workflow_id)
    assert api._execution_status(rt.main)=='编排已暂停（主chat在岗）'
    rt.store.pause(rt.workflow_id,'needs repair')
    assert api._execution_status(rt.main)=='等待用户协商编排'


def test_approved_removal_of_failed_branch_with_only_cached_results_finishes(factory):
    rt=factory([node('a'),node('b',deps=['a'])])
    rt.main.registry.get('read_file').execute=lambda p:{'content':'ok'} if p['path']=='a' else {'error':'known branch failure'}
    rt.start(1)
    for _ in range(200):
        rt.tick()
        if rt.snapshot()['status']=='needs_user':break
        time.sleep(.01)
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    line=rt.task_lines.apply_workflow_patch('c1',[{'operation':'remove','step_id':'b'}],1,2,'user approved removing failed branch')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['status']=='completed'


def test_pretty_helper_json_is_authoritative_not_only_last_brace():
    from agents.recovery import result_object,failure_reason
    result={'submitted':True,'stdout':'helper progress\n'+json.dumps({'submitted':False,'error':'scheduler refused'},indent=2)}
    assert failure_reason(result)=='scheduler refused'
    result['stdout']=json.dumps({'submitted':True,'job_id':'42'},indent=2)
    assert result_object(result)['job_id']=='42'


def test_missing_receipt_enters_user_reconciliation_not_empty_wait(factory):
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'stdout':'helper returned but no scheduler receipt'}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='uncertain' and rt.store.snapshot()['leases']
    eventually(lambda:any(e['kind']=='worker_uncertain' for e in rt.lifecycle.pending()))
    event=next(e for e in rt.lifecycle.pending() if e['kind']=='worker_uncertain')
    assert event['payload']['evidence_call']['call_id']


def test_real_cdft_executor_scopes_preparation_and_submission_to_explicit_directory(tmp_path,monkeypatch):
    from agents.registry import _exec_cdft
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.registry.get_config',lambda:config)
    monkeypatch.setattr('agents.workspace.session_dir',lambda kind:tmp_path/'unused-default')
    captured=[]
    def fake_run(command,**kwargs):
        captured.append((command[2],kwargs['cwd']))
        return SimpleNamespace(returncode=0,stdout='{"submitted":true,"job_id":"42"}',stderr='')
    monkeypatch.setattr('subprocess.run',fake_run)
    work=tmp_path/'isolated'
    _exec_cdft({'action':'pipeline','cif_dir':'cifs','gas':'Kr','temperature':298,'job_work_dir':str(work),'memory_mb':512})
    assert captured and all(cwd==str(work) for code,cwd in captured)
    generated='\n'.join(code for code,cwd in captured)
    assert str(work/'inputs') in generated and str(work/'data') in generated
    assert str(tmp_path/'cifs') in generated
    resource=resources_for(node('x',tool='run_cdft',args={'action':'pipeline','input_dir':'shared'}),tmp_path,tmp_path/'run')
    assert {'key':'path:'+str(tmp_path/'shared'),'mode':'write'} in resource


def test_runtime_registers_authoritative_receipt_when_helper_did_not(factory,tmp_path):
    from agents.job_watch import JobWatch
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])])
    rt.job_watch=JobWatch(watch_file=str(tmp_path/'watch.json'))
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'42','work_dir':str(tmp_path)}
    rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    record=rt.job_watch.get('42')
    assert record['username']=='tester' and record['conv_id']=='c1' and record['tool']=='run_henry'


def test_crash_after_transition_or_message_before_outbox_is_recovered_once(factory):
    rt=factory([node('a')]);rt.start(1)
    ticket=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',ticket['token'],{'status':'failed','error':'crashed before event enqueue'},release=True)
    rt.store.pause(rt.workflow_id,'failure')
    rt.store.send(rt.workflow_id,'a','message before enqueue','comment','lost-gap')
    rt.tick();rt.tick()
    events=rt.lifecycle.pending()
    assert len([e for e in events if e['kind']=='worker_failed'])==1
    assert len([e for e in events if e['kind']=='worker_user_message'])==1


def test_failed_preflight_runtime_repairs_ghost_running_taskline_without_user_cleanup(factory):
    rt = factory([node('a')], cid='preflight-ghost')
    rt.start(1)
    ticket = rt.store.claim(rt.workflow_id, 'a')
    rt.task_lines.upsert_step(rt.main._current_line_id, 'a', status='running', validation={
        'runtime_token': ticket['token'], 'resource_leases': ticket['contract']['resources'],
    })
    rt.store.update(rt.workflow_id, 'a', ticket['token'], {
        'status': 'failed', 'error': 'projection interface mismatch',
        'dispatch_phase': 'preflight', 'tool_returned': False, 'job_ids': [],
    }, release=True)
    rt.store.pause(rt.workflow_id, 'preflight framework failure', step_id='a')
    rt.recover_notifications()
    projected = rt.task_lines.get_line(rt.main._current_line_id)['steps'][0]
    assert projected['status'] == 'failed'
    assert projected['validation']['runtime_token'] is None
    assert projected['validation']['resource_leases'] == []
    assert projected['validation']['dispatch_not_entered'] is True


def test_new_attempt_resets_terminal_flags_and_keeps_old_facts_as_history(factory):
    rt=factory([node('a')]);rt.start(1)
    old=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',old['token'],{'status':'pending','tool_returned':True,
        'jobs_confirmed_terminal':True,'job_ids':['old-job'],'result_ref':{'call_id':'old-call'},'error':'old'},release=True)
    new=rt.store.claim(rt.workflow_id,'a')
    assert new['token']!=old['token'] and not new['tool_returned'] and not new['jobs_confirmed_terminal']
    assert new['job_ids']==[] and 'result_ref' not in new and 'error' not in new
    assert new['attempt_history'][0]['result_ref']=={'call_id':'old-call'}
    assert not rt.store.update(rt.workflow_id,'a',old['token'],{'status':'failed'},release=True)
    assert rt.store.snapshot()['leases']


def test_old_prerequisite_terminal_flag_cannot_release_new_live_job(factory):
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=['result.csv'])]);rt.start(1)
    old=rt.store.claim(rt.workflow_id,'a')
    rt.store.update(rt.workflow_id,'a',old['token'],{'status':'pending','jobs_confirmed_terminal':True},release=True)
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'43','error':'helper failed after new job accepted'}
    rt.tick();eventually(lambda:rt.snapshot()['nodes']['a']['status']=='uncertain')
    assert not rt.snapshot()['nodes']['a']['jobs_confirmed_terminal'] and rt.store.snapshot()['leases']


@pytest.mark.parametrize('user_change',[False,True])
def test_verified_recovery_finishes_unless_user_requested_contract_change(factory,tmp_path,user_change):
    output=tmp_path/'result.csv'
    rt=factory([node('a',tool='run_henry',args={'cif':'x.cif','gas':'Kr','temperature':298},outputs=[str(output)])])
    rt.main.registry.get('run_henry').execute=lambda p:{'submitted':True,'job_id':'42'}
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['a']['status']=='waiting_jobs')
    state=rt.snapshot()['nodes']['a']
    rt.store.update(rt.workflow_id,'a',state['token'],{'status':'uncertain'})
    rt.store.pause(rt.workflow_id,'dispatch reconciliation',kind='reconciliation')
    if user_change:rt.send('a','change temperature after this call','change')
    output.write_text('fresh verified result')
    rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
    verify_prefinish(rt,'a')
    assert rt.snapshot()['nodes']['a']['status']=='succeeded'
    assert rt.snapshot()['status']==('needs_user' if user_change else 'completed')


def test_relative_outputs_use_same_actual_anchor_and_do_not_serialize_disjoint_cdft(factory,tmp_path):
    entered,release=threading.Barrier(3),threading.Event()
    nodes=[]
    for name in ('a','b'):
        work=tmp_path/name;work.mkdir()
        nodes.append(node(name,tool='run_cdft',args={'action':'pipeline','cif_dir':str(tmp_path/'cifs'),
            'gas':'Kr','temperature':298,'job_work_dir':str(work)},outputs=['results.csv']))
    rt=factory(nodes)
    def compute(args):
        entered.wait(timeout=4);assert release.wait(4)
        return {'submitted':True,'job_id':'42' if args['job_work_dir'].endswith('a') else '43'}
    rt.main.registry.get('run_cdft').execute=compute
    rt.start(1);rt.tick()
    try:entered.wait(timeout=4)
    finally:release.set()
    eventually(lambda:all(n['status']=='waiting_jobs' for n in rt.snapshot()['nodes'].values()))
    for name in ('a','b'):(tmp_path/name/'results.csv').write_text('fresh')
    rt.tick([{'job_id':jid,'state':'COMPLETED','terminal':True,'failed':False} for jid in ('42','43')])
    for name in ('a','b'):verify_prefinish(rt,name)
    assert rt.snapshot()['status']=='completed'


@pytest.mark.parametrize('change',['input','executor','solver_version','output'])
def test_cache_rechecks_input_code_version_and_artifacts_before_reuse(factory,tmp_path,change):
    source=tmp_path/'input.cif';source.write_text('initial input')
    output=tmp_path/'result.csv'
    rt=factory([node('a',path=str(source),outputs=[str(output)])])
    calls=[]
    def tool(params):
        calls.append(params);output.write_text(source.read_text());return {'content':'inspected'}
    rt.main.registry.get('read_file').execute=tool
    rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['status']=='completed')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    if change=='input':source.write_text('changed input')
    elif change=='executor':
        def replacement(params):return tool(params)
        rt.main.registry.get('read_file').execute=replacement
    elif change=='solver_version':rt.main.registry.get('read_file').version='new-validated-solver'
    else:output.write_text('externally modified output')
    line=rt.task_lines.apply_workflow_patch('c1',[],1,2,'new approved version; unchanged argument paths')
    rt.main.goal_contract.approved_nodes=line['steps'];rt.main.goal_contract.approved_plan_version=2;rt.main.goal_contract.version+=1
    rt.start(2)
    assert rt.snapshot()['nodes']['a']['status']=='pending'
    rt.tick();eventually(lambda:rt.snapshot()['status']=='completed')
    assert len(calls)==2


def test_directory_artifact_cache_survives_json_checkpoint(factory, tmp_path):
    folder = tmp_path / 'dataset'
    folder.mkdir()
    result = folder / 'sample.dat'
    rt = factory([node('a', outputs=[{'kind': 'directory', 'path': str(folder), 'pattern': '*.dat', 'min_count': 1}])])
    calls = []
    def inspect(params):
        calls.append(params)
        result.write_text('neutral fixture, not a scientific calculation')
        return {'content': 'fixture inspected'}
    rt.main.registry.get('read_file').execute = inspect
    rt.start(1)
    rt.tick()
    eventually(lambda: rt.snapshot()['status'] == 'completed')
    line = rt.task_lines.apply_workflow_patch('c1', [], 1, 2, 'preserve the unchanged dataset')
    rt.main.goal_contract.approved_nodes = line['steps']
    rt.main.goal_contract.approved_plan_version = 2
    rt.main.goal_contract.version += 1
    rt.start(2)
    assert rt.snapshot()['nodes']['a']['status'] == 'succeeded'
    assert len(calls) == 1


def test_same_conversation_node_message_ids_in_different_users_do_not_share_state(factory):
    a=factory([node('same',locks=['workspace'])],'same-cid',username='alice')
    b=factory([node('same',locks=['workspace'])],'same-cid',username='bob')
    a.start(1);b.start(1)
    assert a.workflow_id!=b.workflow_id and a.main._current_line_id!=b.main._current_line_id
    assert a.store.claim(a.workflow_id,'same') and b.store.claim(b.workflow_id,'same')  # logical names are local
    a.send('same','Alice only','comment','same-message-id')
    b.send('same','Bob only','change','same-message-id')
    assert a.snapshot()['status']=='active' and b.snapshot()['status']=='needs_user'
    assert a.snapshot()['nodes']['same']['messages']['same-message-id']['text']=='Alice only'
    assert b.snapshot()['nodes']['same']['messages']['same-message-id']['text']=='Bob only'
    assert {e['payload']['message']['text'] for e in a.lifecycle.pending() if e['kind']=='worker_user_message'}=={'Alice only'}
    assert a.task_lines.get_by_conv('same-cid',username='alice')[0]['username']=='alice'
    assert b.task_lines.get_by_conv('same-cid',username='bob')[0]['username']=='bob'


def test_user_and_session_goals_memories_checkpoints_and_global_resources(factory):
    runtimes=[factory([node('same')],'one',username='alice'),factory([node('same')],'two',username='alice'),
              factory([node('same')],'one',username='bob')]
    for index,rt in enumerate(runtimes):
        rt.main.memory.record_user_preference('private',index)
        rt.main.registry.get('read_file').execute=lambda p:{'content':'own worker'}
        rt.start(1);rt.tick()
    for rt in runtimes:eventually(lambda:rt.snapshot()['status']=='completed')
    files=[Path(rt.snapshot()['nodes']['same']['checkpoint_path']) for rt in runtimes]
    assert len(set(files))==3
    for index,(rt,path) in enumerate(zip(runtimes,files)):
        assert rt.main.memory.user_preferences['private']==index
        assert rt.main.memory.tool_call_log==[]
        assert json.loads(path.read_text())['owner']=={'username':rt.username,'conv_id':rt.conv_id,'step_id':'same'}
    # Explicit global resources remain global; private scope must not hide a real conflict.
    a=factory([node('a',locks=['global:shared-device'])],'global-a',username='alice')
    b=factory([node('b',locks=['global:shared-device'])],'global-b',username='bob')
    a.start(1);b.start(1)
    assert a.store.claim(a.workflow_id,'a') and b.store.claim(b.workflow_id,'b') is None


def test_taskline_tools_never_enumerate_or_update_other_user_or_session(factory,monkeypatch):
    from agents.registry import _exec_task_line_query,_exec_task_line_update
    a=factory([node('a',path='Alice-private')],'same',username='alice')
    b=factory([node('b',path='Bob-private')],'same',username='bob')
    other=factory([node('c',path='Bob-other-session')],'other',username='bob')
    # One shared store contains both users' task lines, including a duplicate CID.
    monkeypatch.setattr('agents.task_line.get_store',lambda:b.task_lines)
    ctx={'username':'bob','conv_id':'same','line_id':b.main._current_line_id}
    monkeypatch.setattr('agents.watch_context.get_context',lambda:ctx)
    own=_exec_task_line_query({})
    assert own['ok'] and own['steps'][0]['arguments']['path']=='Bob-private'
    ambiguous=_exec_task_line_query({'line_id':'same'})
    assert 'Alice-private' not in json.dumps(ambiguous)
    assert not _exec_task_line_query({'conv_id':'other'})['ok']
    assert not _exec_task_line_update({'line_id':a.main._current_line_id,'step_id':'a','status':'failed'})['ok']
    ctx.clear()
    assert not _exec_task_line_query({})['ok']
    assert not _exec_task_line_update({'line_id':'same','step_id':'a','status':'failed'})['ok']


def test_api_jobs_admin_and_unknown_conversation_are_scoped(tmp_path,monkeypatch):
    pytest.importorskip('fastapi')
    import api,auth
    from fastapi.testclient import TestClient
    user=SimpleNamespace(conversations={'same':SimpleNamespace()},current_conv_id='same',get_conversation=lambda cid:None)
    monkeypatch.setattr(api.session_manager,'get_user',lambda username:user)
    monkeypatch.setattr(api._watch,'list',lambda:[{'job_id':'A','username':'alice','conv_id':'same'},
        {'job_id':'B','username':'bob','conv_id':'same'},{'job_id':'B2','username':'bob','conv_id':'other'}])
    monkeypatch.setattr(auth,'is_admin',lambda username:False)
    api.app.dependency_overrides[api._get_user]=lambda:'bob'
    try:
        client=TestClient(api.app)
        assert {j['job_id'] for j in client.get('/api/jobs').json()['jobs']}=={'B','B2'}
        assert [j['job_id'] for j in client.get('/api/jobs?conv_id=same').json()['jobs']]==['B']
        assert client.get('/api/admin/conversations').status_code==403
        assert client.get('/api/latest_answer?conv_id=foreign').status_code==404
    finally:api.app.dependency_overrides.pop(api._get_user,None)


@pytest.mark.parametrize('name',['..','.','../alice','/absolute','alice/c1','alice\\c1'])
def test_private_workspace_cannot_escape_its_user_session_scope(name,monkeypatch,tmp_path):
    from agents.workspace import session_root
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    with pytest.raises(ValueError):session_root('cid',name)
    with pytest.raises(ValueError):session_root(name,'alice')


def test_private_scope_cannot_symlink_into_another_user(monkeypatch,tmp_path):
    from agents.workspace import session_root
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    (tmp_path/'runs/bob/cid').mkdir(parents=True)
    (tmp_path/'runs/alice').symlink_to(tmp_path/'runs/bob',target_is_directory=True)
    with pytest.raises(ValueError,match='alias'):session_root('cid','alice')


def test_worker_checkpoint_foreign_owner_fails_before_tool(factory,tmp_path):
    import hashlib
    rt=factory([node('a')]);rt.start(1)
    path=rt.root/'workers'/hashlib.sha256(b'a').hexdigest()[:24]/'session_checkpoint.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'owner':{'username':'foreign','conv_id':'same','step_id':'a'}}))
    rt.main.registry.get('read_file').execute=lambda p:pytest.fail('foreign checkpoint must never dispatch')
    rt.tick();eventually(lambda:rt.snapshot()['status']=='needs_user')
    assert rt.snapshot()['nodes']['a']['status']=='failed'
    assert 'different user/session' in rt.snapshot()['nodes']['a']['error']


def test_signup_cannot_use_path_identity_or_grant_admin(tmp_path,monkeypatch):
    import auth
    monkeypatch.setattr(auth,'USER_DB',tmp_path/'users.json')
    monkeypatch.delenv('BIMEM_ADMIN_USERS',raising=False)
    assert not auth.register_user('../alice','password')['ok']
    assert auth.register_user('alice','password')['ok']
    assert not auth.is_admin('alice')


@pytest.mark.parametrize('target',['runs/alice/same/input.cif','runs/bob/other/input.cif','tokens.json','users.json','data/state/parallel_workflows.json','logs/api_v33.log'])
def test_native_tool_scope_rejects_other_private_sessions_and_global_stores(tmp_path,target,monkeypatch):
    from agents.workspace import tool_scope_issues
    from agents.registry import _exec_restart_backend
    assert tool_scope_issues({'path':str(tmp_path/target)},'bob','same',tmp_path)
    assert not tool_scope_issues({'path':str(tmp_path/'runs/bob/same/input.cif')},'bob','same',tmp_path)
    assert not tool_scope_issues({'cif_dir':str(tmp_path/'shared-cifs')},'bob','same',tmp_path)
    monkeypatch.setattr('agents.watch_context.get_context',lambda:{'username':'bob','conv_id':'same'})
    monkeypatch.setattr('auth.is_admin',lambda name:False)
    assert _exec_restart_backend({})['blocked']


def test_compiled_node_cannot_write_into_other_user_session(factory,tmp_path):
    target=tmp_path/'runs/alice/one/file.txt'
    rt=factory([node('a',tool='write_file',args={'path':str(target),'content':'bad cross-user write'})],'one',username='bob')
    with pytest.raises(ValueError,match='private user/session'):rt.start(1)
    assert not target.exists()


def test_default_grep_is_private_and_repo_search_excludes_global_memories(monkeypatch,tmp_path):
    from agents.registry import _exec_grep_search
    config=AgentConfig(api_key='test',project_root=tmp_path)
    monkeypatch.setattr('agents.workspace.get_config',lambda:config)
    monkeypatch.setattr('agents.registry.get_config',lambda:config)
    monkeypatch.setattr('agents.watch_context.get_context',lambda:{'username':'bob','conv_id':'same'})
    calls=[]
    monkeypatch.setattr('subprocess.run',lambda cmd,**kw:calls.append(cmd) or SimpleNamespace(stdout='',returncode=0))
    _exec_grep_search({'pattern':'secret'})
    assert calls[-1][-1]==str(tmp_path/'runs/bob/same')
    _exec_grep_search({'pattern':'source','path':str(tmp_path)})
    assert '--exclude-dir=runs' in calls[-1] and '--exclude=tokens.json' in calls[-1]


def test_admin_restart_defers_when_another_user_session_is_busy(monkeypatch):
    pytest.importorskip('fastapi')
    import asyncio,api
    original_sleep=asyncio.sleep
    monkeypatch.setattr(api.asyncio,'sleep',lambda duration:original_sleep(0))
    monkeypatch.setattr(api.session_manager,'list_users',lambda:['bob'])
    monkeypatch.setattr(api.session_manager,'get_user',lambda u:SimpleNamespace(conversations={'busy':SimpleNamespace(is_processing=True)}))
    requested=[]
    monkeypatch.setattr('agents._restart.request',lambda:requested.append(True))
    monkeypatch.setattr(api.subprocess,'Popen',lambda *a,**kw:pytest.fail('must not restart another user mid-turn'))
    asyncio.run(api._perform_backend_restart())
    assert requested==[True]


def test_same_approved_version_cannot_silently_return_stale_cache(factory,tmp_path):
    path=tmp_path/'input.cif';path.write_text('old')
    rt=factory([node('a',path=str(path))]);rt.start(1);rt.tick()
    eventually(lambda:rt.snapshot()['status']=='completed')
    eventually(lambda:all(f.done() for f in rt.futures.values()))
    path.write_text('new dataset under same path')
    with pytest.raises(ValueError,match='revised plan'):rt.start(1)


@pytest.mark.parametrize('store_kind',['users','task_lines'])
def test_kernel_lock_failure_never_falls_back_to_unlocked_shared_write(tmp_path,monkeypatch,store_kind):
    import fcntl,auth
    path=tmp_path/'shared.json';path.write_text('{}')
    def unavailable(*args):raise OSError('kernel lock unavailable')
    monkeypatch.setattr(fcntl,'flock',unavailable)
    if store_kind=='users':
        with pytest.raises(OSError):
            with auth._file_guard(path):
                pytest.fail('must not enter an unlocked user-state transaction')
    else:
        store=TaskLineStore(str(path))
        with pytest.raises(OSError):store.begin_line('c1',username='alice',conv_id='c1')
    assert path.read_text()=='{}'
