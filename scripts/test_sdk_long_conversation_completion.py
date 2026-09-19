"""Real SDK restart/multi-turn completion with historical errors.

Explicit MOCK completed job, neutral CSV and verification receipts. No real
calculation/scheduler writes; the real model chooses inspection and delivery.
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
from agents.workflow_view import bind_result_handoff


def main():
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    username = 'sdk_long_completion'
    root = ROOT / 'runs' / username / cid
    root.mkdir(parents=True, exist_ok=False)
    set_context(username, cid, 'lead-orchestrator')
    actual = root / 'calculation' / 'data' / 'MOCKnative'
    actual.mkdir(parents=True)
    csv = actual / 'results.csv'
    csv.write_text('MOF,status\nMOCK_fixture,neutral_fixture_not_physical_result\n')
    before_file = (csv.stat().st_size, csv.stat().st_mtime_ns)
    original = '用cDFT计算Xe/Kr，298K'
    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session._current_line_id = cid
    session.goal_contract = GoalContract.from_user_message(original)
    session.goal_contract.approved_plan_version = 1
    session.goal_contract.execution_authorized = True
    node = {'step_id': 'mix', 'agent': 'analyst', 'tool': 'run_cdft',
            'arguments': {'action': 'submit', 'gases': ['Xe', 'Kr'], 'temperature': 298,
                          'job_work_dir': str(root / 'calculation'), 'input_dir': str(root / 'inputs'),
                          'memory_mb': 1024, 'resource_review_id': 'MOCKapproved'},
            'depends_on': [], 'expected_outputs': [{'kind': 'file', 'path': str(csv)}]}
    session.goal_contract.approved_nodes = [node]
    session.messages = [{'role': 'user', 'content': original}]
    lines = TaskLineStore(str(root / 'lines.json'))
    lines.begin_line(cid, username=username, conv_id=cid)
    lines.apply_workflow_patch(cid, [{'operation': 'upsert', 'step_id': 'mix', 'node': node}],
                               0, 1, 'Explicit MOCK accepted computation fixture', ROOT)
    mailbox = LifecycleStore(root / 'lifecycle.json')
    mailbox.initialize()
    watch = JobWatch(str(root / 'watch.json'))
    watch.register('99044', username=username, conv_id=cid, work_dir=str(actual))
    watch._jobs['99044'].update(state='COMPLETED', terminal=True, failed=False)
    watch._save()
    runtime = ParallelWorkflow(session, root, WorkflowStore(root / 'runtime.json'), mailbox,
                               lines, username, cid, job_watch=watch)
    runtime.start(1)
    ticket = runtime.store.claim(runtime.workflow_id, 'mix')
    runtime.store.update(runtime.workflow_id, 'mix', ticket['token'], {
        'status': 'succeeded', 'phase': 'finish', 'job_ids': ['99044'],
        'result_ref': {'call_id': 'MOCKexecution'}, 'recovery_key': 'MOCKrecovery',
        'result': {'job_id': '99044', 'work_dir': str(actual)},
        'artifacts': runtime._artifacts(ticket['contract'], {}),
        'node_verification': {'evidence_call_ids': ['MOCKaccepted'], 'conclusion': 'Explicit MOCK file/process acceptance only'},
        'path_manifest': {'calculation_dir': str(actual), 'result_dir': str(actual), 'output_files': [str(csv)]}}, release=True)
    lines.upsert_step(cid, 'mix', status='completed', done=True, job_ids=['99044'])
    lines.upsert_step(cid, 'run_bash', status='failed', note='Historical blocked exploration, not a compiled delegation')
    with session.recovery_gate.transaction() as ledger:
        ledger['MOCKrecovery'] = {'status': 'completed', 'job_ids': ['99044'],
                                  'result_verification': {'call_id': 'MOCKaccepted', 'kind': 'MOCKfixture'}}
    with patch('agents.task_line.get_store', lambda: lines):
        for index in range(20):
            branch = session._create_error_branch('read_file', 'Explicit MOCK historical inspection failure')
            session._error_branches[branch]['failed_arguments'] = {'path': str(root / f'unrelated_{index}.txt')}
    state = session.export_state()
    # Reproduce the deployed legacy checkpoint: version remains, contracts gone.
    state['goal_contract']['approved_nodes'] = []
    state['messages'].append({'role': 'user', 'content':
        '[系统·结构化完成门禁] 批准方案仍有未完成节点：error_error_branch_11=failed(unassigned)。继续清理这些节点。'})
    state.pop('transcript_message_count', None)
    write_checkpoint(root / 'legacy_checkpoint.json', state)
    restored = Session(config=session.config, registry=session.registry)
    restored.client = restored.client.with_options(timeout=60, max_retries=0)
    restored.import_state(json.loads((root / 'legacy_checkpoint.json').read_text()))
    restored.reconcile_transcript([{'role': 'user', 'content': original},
                                   {'role': 'user', 'content': '继续推进整个项目'}])
    restored._restore_workflow_approval(runtime.snapshot(), {'username': username, 'conv_id': cid})
    restored._runtime_snapshot = runtime.snapshot
    restored._lifecycle_store = mailbox
    restored._evidence_root = root / 'evidence'
    restored._chain_jobs = watch.list
    runtime.main = restored
    bind_result_handoff(restored, lambda: runtime)
    def checkpoint(saved, reason):
        refs = compact_checkpoint_evidence(saved, root / 'evidence')
        saved['owner'] = {'username': username, 'conv_id': cid}
        write_checkpoint(root / 'session_checkpoint.json', saved)
        apply_evidence_references(restored.memory, refs)
    restored._on_checkpoint = checkpoint
    agent = copy.copy(ORCHESTRATOR)
    agent.functions = [f for f in agent.functions if f.__name__ in {
        'task_line_query', 'lifecycle_state', 'read_file', 'inspect_path', 'finish_workflow_node',
        'get_tool_schema', 'request_user_decision', 'propose_workflow_patch'}]
    restored.current_agent = agent
    real = restored._call_api
    model_calls = []
    phase_start = 0
    def bounded(current):
        if len(model_calls) - phase_start >= 8:
            raise ExecutionBudgetExceeded('long-conversation scenario budget')
        response = real(current)
        model_calls.append([{'tool': b.name, 'params': b.input} for b in response.content if getattr(b, 'type', '') == 'tool_use'])
        return response
    restored._call_api = bounded
    result = {'kind': 'REAL_SDK_LONG_CONVERSATION_COMPLETION', 'calculation': 'MOCK neutral CSV/accepted job only',
              'model_calls': model_calls}
    try:
        with patch('agents.task_line.get_store', lambda: lines), patch('agents.job_watch.get_watch', lambda: watch):
            result['answer'] = restored.reply('把这个项目已有结果检查后交付给我，不要重新计算，也不要把过去的查看报错当成新的计算任务。'
                '这个隔离测试的作业、验收回执和中性CSV明确是MOCK，只核对保存的执行与文件事实，不编造任何吸附数值。',
                max_rounds=8, verbose=False)
            first_complete = restored.task_complete
            restored.current_agent = agent
            phase_start = len(model_calls)
            result['followup'] = restored.reply('再告诉我保存的作业号和结果文件在哪里，不用重做。', max_rounds=8, verbose=False)
            chain = json.loads((root / 'orchestration_chain.json').read_text())
            calls = restored.memory.tool_call_log
            result['checks'] = {
                'restored_approval': [n['step_id'] for n in restored.goal_contract.approved_nodes] == ['mix'],
                'first_turn_delivered': first_complete,
                'second_turn_delivered': restored.task_complete,
                'real_model_inspected': any(c['tool'] in {'read_file', 'inspect_path', 'task_line_query'} for c in calls),
                'no_ghost_removal_or_recompile': not any(c['tool'] == 'propose_workflow_patch' for c in calls),
                'no_repeat_confirmation': restored._pending_user_interaction is None,
                'historical_errors_retained': len(restored._error_branches) == 20,
                'actual_node_still_verified': runtime.snapshot()['nodes']['mix']['status'] == 'succeeded',
                'original_result_unchanged': before_file == (csv.stat().st_size, csv.stat().st_mtime_ns),
                'jobs_paths_and_branches_persisted': '99044' in chain['current']['jobs'] and len(chain['current']['branches']) == 20
                    and str(csv) in json.dumps(chain['current']['nodes']),
                'two_turn_boundaries_saved': len(chain.get('turns', {})) >= 2,
                'no_ghost_completion_prompts': not any(isinstance(m.get('content'), str)
                    and '[系统·结构化完成门禁]' in m['content'] and 'error_error_branch_' in m['content'] for m in restored.messages)}
            result['passed'] = all(result['checks'].values())
    except Exception as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
    finally:
        runtime.shutdown()
        write_checkpoint(root / 'test_result.json', result)
        clear_context()
        print(json.dumps({'passed': result.get('passed'), 'report': str(root / 'test_result.json'), 'error': result.get('error')}, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
