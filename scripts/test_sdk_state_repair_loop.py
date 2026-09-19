"""Real model repairs a ghost-running DAG without repeated approval/replay.

Scheduler and structure-generation outputs are explicitly MOCK fixtures. The
actual Session/model chooses every query, patch argument and apply operation.
"""
import copy
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.goal_contract import GoalContract
from agents.job_watch import JobWatch
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
from agents.session import Session, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context


def main():
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = ROOT / 'runs/sdk_state_repair' / cid
    root.mkdir(parents=True)
    set_context('sdk_state_repair', cid, 'lead-orchestrator')
    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client = session.client.with_options(timeout=60, max_retries=0)
    session._current_line_id = cid
    session._evidence_root = root / 'evidence'
    lines = TaskLineStore(str(root / 'lines.json'))
    lines.begin_line(cid, username='sdk_state_repair', conv_id=cid)
    def generator(key, deps=()):
        return {'step_id': key, 'agent': 'harness', 'tool': 'generate_structure',
            'arguments': {'material_type': 'MOF', 'n_structures': 1, 'output_dir': str(root / key),
                'partition': 'bigcpu', 'nodelist': 'node06', 'memory_mb': 1024},
            'depends_on': list(deps), 'expected_outputs': [
                {'kind': 'directory', 'path': str(root / key), 'pattern': '*.cif', 'min_count': 1}]}
    nodes = [generator('existing'), generator('next', ['existing'])]
    nodes[0]['expected_outputs'][0]['path'] = str(root / 'existing' / 'accepted')
    lines.apply_workflow_patch(cid, [{'operation': 'upsert', 'step_id': n['step_id'], 'node': n}
        for n in nodes], 0, 1, 'explicitly MOCK approved workflow', ROOT)
    mailbox = LifecycleStore(root / 'lifecycle.json')
    mailbox.initialize()
    watch = JobWatch(str(root / 'watch.json'))
    watch.register('99016', username='sdk_state_repair', conv_id=cid, work_dir=str(root / 'existing'))
    watch._jobs['99016'].update(state='COMPLETED', terminal=True, failed=False)
    watch._save()  # Real JobWatch.get refreshes from durable scheduler facts.
    runtime = ParallelWorkflow(session, root, WorkflowStore(root / 'runtime.json'), mailbox,
        lines, 'sdk_state_repair', cid, job_watch=watch)
    session.goal_contract = GoalContract.from_user_message('按已商定的MOF生成安排继续，别重做完成的部分。')
    session.goal_contract.execution_authorized = True
    session.goal_contract.approved_plan_version = 1
    session.goal_contract.approved_nodes = nodes
    runtime.start(1)
    ticket = runtime.store.claim(runtime.workflow_id, 'existing')
    dataset = root / 'existing'
    dataset.mkdir()
    (dataset / 'accepted').mkdir()
    original = 'data_MOCK_fixture\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n'
    (dataset / 'accepted' / 'fixture.cif').write_text(original)
    repaired_contract = copy.deepcopy(ticket['contract'])
    repaired_contract['resources'] = [r for r in repaired_contract['resources'] if r['key'] == 'path:' + str(dataset)]
    runtime.store.update(runtime.workflow_id, 'existing', ticket['token'], {'status': 'succeeded',
        'contract': repaired_contract,
        'phase': 'finish', 'job_ids': ['99016'], 'node_verification': {'evidence_call_ids': ['MOCKverified']},
        'artifacts': runtime._artifacts(ticket['contract'], {}), 'execution_fingerprint': 'old-sdk-release'}, release=True)
    lines.upsert_step(cid, 'existing', status='completed', done=True, job_ids=['99016'],
        validation={'output_contract': 'passed'})
    # Reproduce a conversational tool incorrectly reporting an unclaimed node.
    lines.upsert_step(cid, 'next', status='running')
    mailbox.emit('plan_patch_proposed', {'patch': {'new_version': 99,
        'proposed_nodes': [{'step_id': 'nonexistent_old_node'}]}, 'requires_user': True}, event_id='old-proposal')
    calls = []
    def mock_generator(params):
        calls.append(copy.deepcopy(params))
        return {'error': 'MOCK computation stopped after proving one dispatch; no scientific result claimed'}
    session.registry.get('generate_structure').execute = mock_generator
    session._runtime_snapshot = runtime.snapshot
    session._lifecycle_store = mailbox
    def activate(version):
        receipt = runtime.start(version)
        runtime.tick()
        return receipt
    session._on_workflow_start = activate
    def checkpoint(state, reason):
        refs = compact_checkpoint_evidence(state, root / 'evidence')
        write_checkpoint(root / 'session_checkpoint.json', state)
        apply_evidence_references(session.memory, refs)
    session._on_checkpoint = checkpoint
    agent = copy.copy(ORCHESTRATOR)
    agent.functions = [f for f in agent.functions if f.__name__ in
        {'lifecycle_state', 'task_line_query', 'get_tool_schema', 'propose_workflow_patch', 'apply_workflow_patch'}]
    real = session._call_api
    model_calls = []
    def bounded(current):
        if len(model_calls) >= 6:
            raise ExecutionBudgetExceeded('state-repair model budget')
        response = real(current)
        model_calls.append([{'name': b.name, 'arguments': b.input} for b in response.content
            if getattr(b, 'type', '') == 'tool_use'])
        return response
    session._call_api = bounded
    result = {'kind': 'REAL_SDK_GHOST_STATUS_SCOPED_REPAIR', 'scheduler': 'MOCK', 'model_calls': model_calls}
    try:
        with patch('agents.task_line.get_store', lambda: lines):
            result['answer'] = session.run_until_complete(
                '刚才好像又卡住了。完成的结构不要重做，后一步换到node07继续就行，分区、内存和科学设置都不变。'
                '看看现在真正执行的安排，不要按以前没执行的提案反复问我。这个测试的计算和调度明确是MOCK。',
                agent=agent, max_rounds=6, verbose=False)
            deadline = time.monotonic() + 5
            while not calls and time.monotonic() < deadline:
                time.sleep(.05)
            state = runtime.snapshot()
            result['checks'] = {
                'model_selected_patch': any(c['tool'] == 'propose_workflow_patch' for c in session.memory.tool_call_log),
                'model_selected_apply': any(c['tool'] == 'apply_workflow_patch' for c in session.memory.tool_call_log),
                'no_repeated_confirmation': session._pending_user_interaction is None,
                'one_next_dispatch': len(calls) == 1 and calls[0]['output_dir'] == str(root / 'next'),
                'no_existing_dataset_replay': state['nodes']['existing']['status'] == 'succeeded'
                    and state['nodes']['existing']['job_ids'] == ['99016'] and (dataset / 'accepted' / 'fixture.cif').read_text() == original,
                'node_actually_changed': state['nodes']['next']['contract']['arguments']['nodelist'] == 'node07',
                'history_not_current_dag': 'nonexistent_old_node' not in state['nodes']}
            result['passed'] = all(result['checks'].values())
    except Exception as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
    finally:
        runtime.shutdown()
        write_checkpoint(root / 'test_result.json', result)
        clear_context()
        print(json.dumps({'passed': result.get('passed'), 'report': str(root / 'test_result.json'),
            'error': result.get('error')}, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
