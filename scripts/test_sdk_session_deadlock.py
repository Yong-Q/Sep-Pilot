"""Real SDK autonomously repairs the recorded pre-dispatch loop; IO adapter MOCK."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
from agents.registry import _build_default_registry
from agents.session import Session, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint, compact_checkpoint_evidence, apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context, clear_context


def main():
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    username = 'sdk_deadlock'
    root = ROOT / 'runs' / username / cid
    charged = root / 'charged'; charged.mkdir(parents=True)
    for i in range(10):
        shutil.copyfile(ROOT / 'tests/fixtures/charges/neutral_p1.cif', charged / f'SYNTHETIC_{i}.cif')
    inputs = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in charged.glob('*.cif')}
    registry = _build_default_registry()
    lines = TaskLineStore(str(root / 'task_lines.json'))
    line = lines.begin_line(cid, username=username, conv_id=cid)
    # Replay only real historical audit labels, not a fabricated executable DAG.
    archive = ROOT / 'runs/test_hof/REDACTED/session_checkpoint.json'
    archived = json.loads(archive.read_text())
    for old in archived['goal_contract']['approved_nodes']:
        if 'expected_outputs' not in old and not old.get('job_ids'):
            item = copy.deepcopy(old); key = item.pop('step_id')
            for field in ('username', 'conv_id', 'plan_version'): item.pop(field, None)
            lines.upsert_step(line['line_id'], key, **{k:v for k,v in item.items() if k in {'tool','agent','arguments','status','done','validation'}})
    calls = []; dispatches = []
    result = {'kind':'REAL_SDK_RECORDED_DEADLOCK_RECOVERY_WITH_BUSINESS_MOCK', 'archive':str(archive),
              'model_calls':calls, 'dispatches':dispatches}
    def adapter(arguments):
        work = Path(arguments['job_work_dir']); work.mkdir(parents=True, exist_ok=True)
        output = Path(arguments.get('output') or work / 'results.csv')
        gases = arguments.get('gases') or [arguments['gas']]
        output.write_text('label,gas,mock_value\n' + ''.join(f'{p.stem},{gas},MOCK\n' for gas in gases for p in sorted(Path(arguments['cif_dir']).glob('*.cif'))))
        dispatches.append(copy.deepcopy(arguments))
        return {'status':'completed','submitted':False,'business_mock':True,'work_dir':str(work),'output_csv':str(output)}
    registry.get('run_cdft').execute = adapter
    session = Session(config=AgentConfig.from_env(max_tokens=4096), registry=registry)
    session.client = session.client.with_options(timeout=90, max_retries=0)
    session._current_line_id = line['line_id']; session._evidence_root = root / 'evidence'
    def checkpoint(state, reason):
        refs = compact_checkpoint_evidence(state, root / 'evidence')
        write_checkpoint(root / 'session_checkpoint.json', state)
        apply_evidence_references(session.memory, refs)
    session._on_checkpoint = checkpoint
    mailbox = LifecycleStore(root / 'lifecycle.json'); mailbox.initialize()
    runtime = ParallelWorkflow(session, root, WorkflowStore(root / 'parallel_workflows.json'), mailbox, lines, username, cid)
    session._runtime_snapshot = runtime.snapshot
    session._on_workflow_start = lambda version: runtime.start(version)
    session._on_interrupt_requested = lambda: bool(runtime.snapshot())
    real = session._call_api
    def bounded(agent):
        if len(calls) >= 10: raise ExecutionBudgetExceeded('deadlock planner model-call budget reached')
        answer = real(agent)
        calls.append({'agent':agent.name,'tools':[{'name':b.name,'arguments':b.input} for b in answer.content if getattr(b,'type','')=='tool_use']})
        return answer
    session._call_api = bounded
    try:
        with patch('agents.defns._registry', registry), patch('agents.task_line.get_store', lambda:lines):
            set_context(username, cid, 'lead-orchestrator', line['line_id'])
            directory_agent = copy.copy(ORCHESTRATOR)
            directory_agent.functions = [f for f in directory_agent.functions if f.__name__ in {'run_bash', 'inspect_path'}]
            result['directory_answer'] = session.run_until_complete(
                f'只读检查本会话目录：先run_bash执行ls runs/{username}/{cid}/charged/，再inspect_path检查绝对路径{charged}，两者都需要实际调用。报告真实目录状态；不提交计算、不修改文件。',
                agent=directory_agent, max_rounds=4, verbose=False)
            result['answer'] = session.run_until_complete(
                f'直接执行cDFT流程验收，Kr和Xe，298K，10个合成CIF在{charged}。计算适配器为BUSINESS_MOCK，只生成流程验收CSV，不能声称得到物理Henry系数。'
                'task_line_query中有归档的未派发历史标签，不是实际作业，也不是工作节点。'
                '自主查schema并以两个analyst run_cdft pipeline节点编译工作流，两个节点依赖为空、expected_outputs=[]，工作目录由编译器补齐。'
                '不要移除审计历史、不要清除DAG或串行handoff绕过执行器；让系统把未派发审计与具体执行节点分开。框架已有合成电荷，framework_charge=false。',
                agent=copy.copy(ORCHESTRATOR), max_rounds=10, verbose=False)
            if not runtime.snapshot(): raise RuntimeError('SDK did not compile and activate a valid workflow')
            deadline = time.monotonic()+30
            while runtime.snapshot().get('status')!='completed':
                if time.monotonic()>deadline: raise TimeoutError('SDK compiled IO workflow did not finish')
                runtime.tick([]); time.sleep(.05)
            state = runtime.snapshot(); nodes = state['nodes']
            shell = [c for c in session.memory.tool_call_log if c['tool']=='run_bash']
            result['checks'] = {
                'sdk_project_prefixed_shell_found_files':bool(shell) and any(json.loads(session._load_evidence_call(c)['result']).get('exit_code')==0 for c in shell),
                'sdk_used_absolute_directory_tool':any(c['tool']=='inspect_path' for c in session.memory.tool_call_log),
                'only_two_real_worker_contracts':len(nodes)==2 and all(n['contract']['tool']=='run_cdft' for n in nodes.values()),
                'both_requested_gases':{gas for a in dispatches for gas in (a.get('gases') or [a['gas']])}=={'Kr','Xe'},
                'no_duplicate_dispatch':len(dispatches)==2,
                'workspaces_disjoint':len({a['job_work_dir'] for a in dispatches})==2,
                'native_csv_contract_present':all(n['contract']['expected_outputs'] and n['status']=='succeeded' for n in nodes.values()),
                'leases_released':not runtime.store.snapshot().get('leases'),
                'inputs_unmodified':all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in inputs.items()),
            }
            result['passed']=all(result['checks'].values()); result['runtime']=state
    except Exception as error: result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        clear_context()
        if all(f.done() for f in runtime.futures.values()): runtime.shutdown()
        write_checkpoint(root/'test_result.json',result)
        print(json.dumps({'passed':result.get('passed'),'error':result.get('error'),'report':str(root/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__': raise SystemExit(main())
