"""Real SDK validates a native-receipt path and persists a multi-turn chain.

Explicit MOCK job and neutral fixture CSV; no scheduler calls/calculation.
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
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore, resources_for
from agents.session import Session, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context
from agents.workflow_view import bind_result_handoff


def main():
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = ROOT / 'runs/sdk_native_chain' / cid
    root.mkdir(parents=True)
    set_context('sdk_native_chain', cid, 'lead-orchestrator')
    parent = root / 'calculation'
    actual = parent / 'data' / 'native_timestamp'
    actual.mkdir(parents=True)
    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client = session.client.with_options(timeout=60, max_retries=0)
    session._current_line_id = cid
    session._evidence_root = root / 'evidence'
    lines = TaskLineStore(str(root/'lines.json'))
    lines.begin_line(cid, username='sdk_native_chain', conv_id=cid)
    compute = {'step_id': 'mix', 'agent': 'analyst', 'tool': 'run_cdft',
        'arguments': {'action': 'submit', 'gases': ['Xe', 'Kr'], 'temperature': 298,
            'input_dir': str(root/'inputs'), 'job_work_dir': str(parent), 'memory_mb': 1024,
            'resource_review_id': 'MOCKapproved'}, 'depends_on': [],
        'expected_outputs': [{'kind': 'file', 'path': str(parent/'results.csv')}]}
    after = {'step_id': 'after', 'agent': 'analyst', 'tool': 'read_file',
        'arguments': {'path': str(actual/'results.csv')}, 'depends_on': ['mix'],
        'expected_outputs': [str(actual/'results.csv')]}
    lines.apply_workflow_patch(cid, [{'operation':'upsert','step_id':n['step_id'],'node':n} for n in [compute,after]], 0, 1, 'MOCK native receipt fixture', ROOT)
    session.goal_contract = GoalContract.from_user_message('用cDFT计算Xe/Kr，298K，继续已确认任务。')
    session.goal_contract.execution_authorized = True
    session.goal_contract.approved_plan_version = 1
    session.goal_contract.approved_nodes = [compute, after]
    mailbox = LifecycleStore(root/'lifecycle.json')
    mailbox.initialize()
    watch = JobWatch(str(root/'watch.json'))
    watch.register('99034', username='sdk_native_chain', conv_id=cid, work_dir=str(actual))
    watch._jobs['99034'].update(state='COMPLETED', terminal=True, failed=False)
    watch._save()
    runtime = ParallelWorkflow(session, root, WorkflowStore(root/'runtime.json'), mailbox, lines, 'sdk_native_chain', cid, job_watch=watch)
    session._runtime_snapshot = runtime.snapshot
    bind_result_handoff(session, lambda: runtime)
    runtime.start(1)
    ticket = runtime.store.claim(runtime.workflow_id, 'mix')
    write_checkpoint(root/'evidence/nativeexecution.json', {'call_id':'nativeexecution','time':ticket['started_at'],
        'tool':'run_cdft','params':compute['arguments'],'result':json.dumps({'submitted':True,'job_id':'99034','work_dir':str(actual)})})
    csv = actual/'results.csv'
    csv.write_text('MOF,status\nMOCK_fixture,neutral_fixture_not_physical_result\n')
    runtime.store.update(runtime.workflow_id, 'mix', ticket['token'], {'status':'prefinish','phase':'prefinish',
        'job_ids':['99034'],'jobs_confirmed_terminal':True, 'result':{'submitted':True,'job_id':'99034','work_dir':str(actual)},
        'result_ref':{'call_id':'nativeexecution'},'output_baseline':{},
        'path_manifest':{'calculation_dir':str(actual),'result_dir':str(actual),'input_paths':[str(root/'inputs')], 'output_files':[]}})
    session._lifecycle_store = mailbox
    def checkpoint(state, reason):
        refs = compact_checkpoint_evidence(state, root/'evidence')
        write_checkpoint(root/'session_checkpoint.json', state)
        apply_evidence_references(session.memory, refs)
    session._on_checkpoint = checkpoint
    agent = copy.copy(ORCHESTRATOR)
    agent.functions = [f for f in agent.functions if f.__name__ in
        {'read_file','inspect_path','get_tool_schema','task_line_query','lifecycle_state','finish_workflow_node','revalidate_workflow_node_outputs'}]
    session.current_agent = agent
    real = session._call_api
    calls = []
    phase_start = 0
    def bounded(current):
        if len(calls) - phase_start >= 8: raise ExecutionBudgetExceeded('native chain scenario budget')
        response = real(current)
        calls.append([{'name':b.name,'arguments':b.input} for b in response.content if getattr(b,'type','')=='tool_use'])
        return response
    session._call_api = bounded
    result = {'kind':'REAL_SDK_NATIVE_RECEIPT_MULTI_TURN_CHAIN','calculation':'MOCK neutral fixture only','model_calls':calls}
    try:
        with patch('agents.task_line.get_store', lambda:lines):
            result['answer'] = session.reply('这一步已经运行结束，页面还在待验收。请根据实际执行安排查证已有结果，验收后让后续只读步骤继续，不重新计算。'
                '这个隔离任务的作业和CSV均明确是MOCK，只验收文件及流程事实，不解释为真实吸附科学结果。', max_rounds=8, verbose=False)
            deadline = time.monotonic()+5
            while runtime.snapshot()['nodes']['after']['status'] != 'succeeded' and time.monotonic()<deadline: time.sleep(.05)
            session.current_agent = agent
            phase_start = len(calls)
            result['followup'] = session.reply('再核对一下刚才的作业号和实际结果目录，从保存的执行安排里查，不要重做。', max_rounds=8, verbose=False)
            # A model can finish on the follow-up turn. Wait for the same
            # asynchronous readiness queue here too; do not sample a queued
            # read-only worker as failed before it has had a chance to run.
            deadline = time.monotonic()+5
            while runtime.snapshot()['nodes']['after']['status'] != 'succeeded' and time.monotonic()<deadline: time.sleep(.05)
            data = json.loads((root/'orchestration_chain.json').read_text())
            node = runtime.snapshot()['nodes']['mix']
            restored = Session(config=session.config, registry=session.registry)
            restored.import_state(json.loads((root/'session_checkpoint.json').read_text()))
            result['checks'] = {'model_verified_finish':node['status']=='succeeded' and bool(node.get('node_verification')),
                'actual_contract_bound':node['contract']['expected_outputs']==[{'kind':'file','path':str(csv)}],
                'no_recomputation':node['job_ids']==['99034'] and not (parent/'results.csv').exists(),
                'dependent_advanced':runtime.snapshot()['nodes']['after']['status']=='succeeded',
                'path_change_journaled':bool(node.get('output_contract_history')),
                'job_and_actual_path_in_chain':data['current']['nodes']['mix']['job_ids']==['99034'] and str(actual) in json.dumps(data['current']['nodes']['mix']),
                'two_turns_persisted':len(data.get('turns',{}))>=2 and all(t['last_boundary']=='turn_end' for t in data['turns'].values()),
                'linked_revisions':all(r['parent_hash']==data['revisions'][i-1]['hash'] for i,r in enumerate(data['revisions']) if i),
                'recovery_after_restart_reads_same_chain':restored.goal_contract.approved_nodes[0]['expected_outputs']==[{'kind':'file','path':str(csv)}]
                    and json.loads((root/'orchestration_chain.json').read_text())['current']==data['current']}
            result['passed'] = all(result['checks'].values())
    except Exception as error: result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        runtime.shutdown()
        write_checkpoint(root/'test_result.json', result)
        clear_context()
        print(json.dumps({'passed':result.get('passed'),'report':str(root/'test_result.json'),'error':result.get('error')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__': raise SystemExit(main())
