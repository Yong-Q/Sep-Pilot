"""Opt-in real LLM + real HPC SDK smoke. Never touches historical user tasks."""
import argparse
import copy
import json
from pathlib import Path
import shlex
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import anthropic
from agents.agent import Agent
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR, MONITOR
from agents.job_watch import JobWatch
from agents.lifecycle import LifecycleStore
from agents.node_inventory import node_inventory, candidates
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
from agents.registry import _build_default_registry
from agents.session import Session, LLMUnavailableError, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--real-hpc', action='store_true', required=True)
    args = parser.parse_args()
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    username = 'sdk_smoke'
    root = ROOT / 'runs' / username / cid
    root.mkdir(parents=True, exist_ok=False)
    config = AgentConfig.from_env(max_tokens=4096, temperature=.2)
    registry = _build_default_registry()
    watch = JobWatch(str(root / 'job_watch.json'))
    lines = TaskLineStore(str(root / 'task_lines.json'))
    line = lines.begin_line(cid, username=username, conv_id=cid)
    mailbox = LifecycleStore(root / 'lifecycle.json')
    mailbox.initialize()
    store = WorkflowStore(root / 'parallel_workflows.json')
    submitted = []
    model_calls = []
    results = {'scope': str(root), 'kind': 'REAL_LLM_REAL_HPC_SYNTHETIC_BUSINESS_SMOKE',
        'checks': {}, 'submitted_jobs': submitted, 'model_calls': model_calls}
    probe = ROOT / 'scripts/sdk_numeric_probe.py'
    native_submit = registry.get('submit_job').execute
    permitted = {'resource_health', 'get_tool_schema', 'task_line_query', 'task_line_update', 'read_file', 'write_file',
        'inspect_path', 'run_bash', 'check_job', 'diagnose_job', 'list_my_jobs', 'submit_job',
        'recovery_state', 'lifecycle_state', 'propose_workflow_patch', 'request_user_decision',
        'execute_workflow', 'message_workflow_node', 'accept_recovered_result', 'prepare_retry',
        'reconcile_watched_job', 'resolve_local_workflow_write', 'supervisor_decision'}
    for name in registry.all_names():
        if name not in permitted and not name.startswith('handoff_to_'):
            registry.get(name).execute = lambda p: {'blocked': True, 'error': 'this isolated smoke does not authorize other scientific tools'}
    def submit(parameters):
        if len(submitted) >= 2: return {'error': 'real smoke submission budget is two; no duplicate jobs permitted'}
        words = shlex.split(parameters.get('command', ''))
        if len(words) != 8 or words[:2] != ['/usr/bin/python3', str(probe)]:
            return {'error': 'only the declared numeric probe command is authorized in this smoke'}
        values = dict(zip(words[2::2], words[3::2]))
        if set(values) != {'--label', '--duration', '--output-dir'} or values['--label'] not in {'A', 'B'}:
            return {'error': 'invalid probe arguments'}
        work = Path(parameters.get('work_dir', '')).resolve()
        if not work.is_relative_to(root) or Path(values['--output-dir']).resolve() != work:
            return {'error': 'calculation/output paths must be the same own-session directory'}
        inventory = node_inventory(ROOT, refresh=True)
        allowed = {node['name']: node for node in candidates(inventory)}
        node = parameters.get('nodelist')
        if node not in allowed or parameters.get('partition') not in allowed[node]['partitions']:
            return {'error': 'choose a current available node and matching partition from resource_health.node_inventory'}
        if any(item['node'] == node or item['label'] == values['--label'] for item in submitted):
            return {'error': 'the two tasks require distinct labels/nodes; existing dispatch cannot be repeated'}
        if float(values['--duration']) != 45 or parameters.get('cpus_per_task') != 1 or parameters.get('walltime') != '00:02:00':
            return {'error': 'smoke resource bounds are duration45, cpus1, walltime00:02:00'}
        if parameters.get('mode', 'local') != 'local': return {'error': 'smoke must use the local scheduler'}
        item = {'label': values['--label'], 'node': node, 'arguments': copy.deepcopy(parameters), 'invoked_at': time.time()}
        submitted.append(item)  # consume even rejected/uncertain requests; no silent fallback
        receipt = native_submit(parameters)
        item.update(receipt=receipt, job_id=receipt.get('job_id'))
        write_checkpoint(root / 'dispatches.json', {'dispatches': submitted})
        print(json.dumps({'phase': 'actual_sdk_dispatch', 'label': item['label'], 'node': node, 'job_id': item['job_id']}, ensure_ascii=False), flush=True)
        return receipt
    registry.get('submit_job').execute = submit
    sessions = []
    runtimes = []
    interrupt = [False]
    def new_session(role='main'):
        s = Session(config=config, registry=registry)
        s.client = s.client.with_options(max_retries=0, timeout=90)
        s._current_line_id = line['line_id']
        s.recovery_gate.path = root / 'recovery_state.json'
        s._evidence_root = root / 'evidence'
        s._lifecycle_store = mailbox
        def save(state, reason):
            references = compact_checkpoint_evidence(state, root / 'evidence')
            write_checkpoint(root / (role + '_checkpoint.json'), state)
            apply_evidence_references(s.memory, references)
        s._on_checkpoint = save
        real_api = s._call_api
        def call(agent):
            if len(model_calls) >= 16: raise ExecutionBudgetExceeded('isolated live smoke reached its bounded LLM budget')
            record = {'agent': agent.name, 'started_at': time.time()}
            model_calls.append(record)
            response = real_api(agent)
            record.update(finished_at=time.time(), tools=[{'name': b.name, 'arguments': b.input} for b in response.content if getattr(b, 'type', '') == 'tool_use'],
                response_types=[getattr(b, 'type', '') for b in response.content],
                stop_reason=response.stop_reason, output_tokens=response.usage.output_tokens,
                text_summary='\n'.join(b.text for b in response.content if getattr(b, 'type', '') == 'text')[:1500])
            write_checkpoint(root / 'model_calls.json', {'calls': model_calls})
            return response
        s._call_api = call
        real_stream = s._stream_api
        def stream(**kw):
            try: return real_stream(**kw)
            except anthropic.APIError as error: raise LLMUnavailableError('live model request failed: ' + type(error).__name__) from error
        s._stream_api = stream
        sessions.append(s)
        return s
    def attach(s, stop_after_activation=False):
        rt = ParallelWorkflow(s, root, store, mailbox, lines, username, cid, job_watch=watch)
        runtimes.append(rt)
        s._runtime_snapshot = rt.snapshot
        def start(version):
            receipt = rt.start(version)
            rt.tick(watch.list())
            if stop_after_activation: interrupt[0] = True
            return receipt
        s._on_workflow_start = start
        s._on_interrupt_requested = lambda: interrupt[0]
        return rt
    def wait_until(predicate, timeout=30, tick=None):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() >= deadline: raise TimeoutError('scenario did not reach required state')
            if tick: tick()
            time.sleep(.25)
    try:
        with patch('agents.defns._registry', registry), patch('agents.task_line.get_store', lambda: lines), \
             patch('agents.job_watch.get_watch', lambda: watch), \
             patch.object(Session, '_pending_jobs_for_conv', lambda self: watch.list()):
            set_context(username, cid, 'monitor', line['line_id'])
            observer = new_session('monitor')
            assessment = observer.run_readonly_observer('为两个独立的45秒CPU数值验证任务检查可用节点和分区。必须调用resource_health，依据node_inventory给主agent选点建议；不要提交作业，不使用node01/node08。', MONITOR, max_rounds=4)
            results['resource_assessment'] = assessment
            results['checks']['monitor_called_real_resource_tool'] = any(c['tool'] == 'resource_health' for c in observer.memory.tool_call_log)
            s = new_session()
            rt = attach(s, stop_after_activation=True)
            set_context(username, cid, 'lead-orchestrator', line['line_id'])
            task = f'''直接执行一个SDK自主编排的真实计算冒烟测试，然后汇总，不是科学材料计算。使用两个独立任务A/B，各用1CPU连续数值积分45秒，walltime为00:02:00；选择监控确认空闲的两个不同节点及其匹配分区，mode=local。\n监控反馈：{assessment}\n节点完整事实由resource_health.node_inventory提供，也可读{ROOT / 'env/node_inventory.json'}。\n已有可运行脚本{probe}，使用/usr/bin/python3，其参数是--label A或B --duration 45 --output-dir 计算目录。自己根据工具schema组织完整submit_job参数、节点ID与依赖，不要照抄未提供的参数对象。计算目录和output-dir必须同一个绝对路径；分别使用{root / 'A'}和{root / 'B'}。\n编排需包含两个由harness执行的submit_job节点，expected_outputs分别为本目录compute_receipt.json和business_mock.csv；后面一个analyst/read_file节点读取A的compute_receipt.json，必须depends_on两个计算节点。实际积分已知答案π，业务CSV是假数据，必须注明MOCK不能当科学结果。\n先task_line_query，再propose_workflow_patch编译完整可执行DAG；此初始方案在本用户授权范围内可直接批准。不能用旧串行handoff提交、不使用run_bash/sbatch绕过、不调用其他科学工具。编排激活后本测试会主动中断主回合，再恢复同一批准链；不要将中断理解为取消已提交计算。'''
            results['initial_reply'] = s.run_until_complete(task, agent=copy.copy(ORCHESTRATOR), max_rounds=12, verbose=False)
            s._checkpoint('live_smoke_initial_turn_finished')
            if not rt.snapshot(): raise RuntimeError('live main agent did not compile and activate the structured DAG')
            wait_until(lambda: len(submitted) == 2 and all(item.get('job_id') for item in submitted), tick=lambda: rt.tick(watch.list()))
            wait_until(lambda: all(f.done() for f in rt.futures.values()))
            rt.tick(watch.list(), paused=True)
            before = rt.snapshot()
            pending = [key for key, node in before['nodes'].items() if node['status'] == 'pending']
            results['checks']['paused_with_pending_successor'] = before['user_paused'] and bool(pending)
            results['paused_snapshot'] = before
            s._checkpoint('live_smoke_interrupt_before_restore')
            rt.shutdown()
            restored = new_session('restored_main')
            restored.import_state(json.loads((root / 'main_checkpoint.json').read_text()))
            interrupt[0] = False
            resumed = attach(restored)
            results['checks']['goal_survived_restore'] = restored.goal_contract.to_dict() == s.goal_contract.to_dict()
            results['checks']['leases_survived_restore'] = len(store.snapshot().get('leases', {})) >= 2
            watch._poll_once()
            resumed.tick(watch.list(), paused=True)
            results['checks']['paused_successor_did_not_dispatch'] = all(resumed.snapshot()['nodes'][key]['status'] == 'pending' for key in pending)
            resumed.start(restored.goal_contract.approved_plan_version)
            deadline = time.monotonic() + 150
            while resumed.snapshot()['status'] != 'completed':
                if time.monotonic() >= deadline: raise TimeoutError('real workflow did not finish within the smoke deadline')
                watch._poll_once()
                resumed.tick(watch.list())
                if resumed.snapshot()['status'] == 'needs_user': raise RuntimeError('real workflow needs diagnosis; evidence preserved')
                time.sleep(2)
            receipts = []
            for item in submitted:
                receipt = json.loads((Path(item['arguments']['work_dir']) / 'compute_receipt.json').read_text())
                receipts.append(receipt)
                if receipt.get('validation_passed') is not True or receipt['absolute_error'] > 1e-10 or receipt['duration_seconds'] < 44:
                    raise RuntimeError('real calculation did not satisfy its known-answer/time check')
            results['real_calculation_receipts'] = receipts
            results['checks']['real_jobs_overlapped'] = max(r['started_at'] for r in receipts) < min(r['finished_at'] for r in receipts)
            results['checks']['distinct_compute_hosts'] = len({r['hostname'] for r in receipts}) == 2
            results['checks']['no_duplicate_dispatch_after_resume'] = len(submitted) == 2 and len(watch.list()) == 2
            final = resumed.snapshot()
            results['checks']['successor_advanced_after_success'] = all(final['nodes'][key]['status'] == 'succeeded' for key in pending)
            results['checks']['all_node_paths_are_monitorable'] = all(n.get('path_manifest', {}).get('checkpoint_path') and n.get('path_manifest', {}).get('evidence_path') for n in final['nodes'].values())
            results['checks']['lease_release_after_verified_outputs'] = not store.snapshot().get('leases')
            results['final_snapshot'] = final
            results['checks']['llm_autonomously_supplied_job_arguments'] = any(t['name'] == 'propose_workflow_patch' for call in model_calls for t in call.get('tools', []))
            results['checks']['discovery_did_not_become_worker_nodes'] = len(final['nodes']) == 3 and all(n['contract']['tool'] != 'get_tool_schema' for n in final['nodes'].values())
            set_context(username, cid, 'supervisor', line['line_id'])
            supervisor = new_session('supervisor')
            supervisor.goal_contract = copy.deepcopy(restored.goal_contract)
            supervisor._runtime_snapshot = resumed.snapshot
            decision = supervisor.observe_lifecycle_event({'event_id': 'real-smoke-completed', 'kind': 'job_completed',
                'payload': {'workflow': final, 'real_receipts': receipts, 'business_results_are_mock': True}})
            results['supervisor_receipt'] = decision
            set_context(username, cid, 'lead-orchestrator', line['line_id'])
            restored.resume_lifecycle_event({'event_id': 'smoke-resume-delivery', 'kind': 'job_completed', 'supervisor_receipt': decision,
                'payload': {'workflow': final, 'real_receipts': receipts, 'business_results_are_mock': True,
                            'instruction': '本次隔离测试恢复和计算已完成，请主chat交付真实积分结果并明确业务CSV是MOCK；不得新提交计算。'}})
            results['delivery'] = restored.last_text
            results['checks']['main_owns_delivery'] = restored.current_agent.name == 'lead-orchestrator'
            results['checks']['no_jobs_added_during_delivery'] = len(submitted) == 2
            results['passed'] = all(results['checks'].values())
    except Exception as error:
        results.update(passed=False, error=type(error).__name__ + ': ' + str(error))
    finally:
        for rt in runtimes:
            if not any(not future.done() for future in rt.futures.values()): rt.shutdown()
        clear_context()
        write_checkpoint(root / 'test_result.json', results)
        print(json.dumps({'passed': results.get('passed'), 'checks': results['checks'], 'error': results.get('error'),
                          'report': str(root / 'test_result.json')}, ensure_ascii=False), flush=True)
    return 0 if results.get('passed') else 1


if __name__ == '__main__': raise SystemExit(main())
