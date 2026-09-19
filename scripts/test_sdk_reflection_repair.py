"""Real SDK model repairs a MOCK stale directory index, not a calculation.

No frontend conversation, scheduler command or resource monitoring is used.
"""
import copy
import json
from pathlib import Path
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.session import Session, ExecutionBudgetExceeded
from agents.state_io import write_checkpoint
from agents.watch_context import set_context, clear_context


def main():
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = ROOT / 'runs/sdk_reflection_repair' / cid
    root.mkdir(parents=True)
    for name, text in [('a', 'Alpha evidence'), ('b', 'Beta evidence'), ('c', 'Gamma evidence')]:
        write_checkpoint(root / f'{name}.json', {'name': name, 'value': text})
    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client = session.client.with_options(timeout=60, max_retries=0)
    session._evidence_root = root / 'evidence'
    session._on_checkpoint = lambda state, reason: write_checkpoint(root / 'checkpoint.json', state)
    agent = copy.copy(ORCHESTRATOR)
    agent.functions = [f for f in agent.functions if f.__name__ in {'read_file', 'inspect_path', 'get_tool_schema'}]
    read = session.registry.get('read_file').execute
    inspect = session.registry.get('inspect_path').execute
    refreshed = False
    rejected = 0
    def stale_index(params):
        nonlocal rejected, refreshed
        if rejected == 0 or not refreshed:
            refreshed = False
            rejected += 1
            return {'error': 'MOCK stale directory index. Refresh the directory index using a real directory inspection before another file fetch.',
                    'error_type': 'STALE_DIRECTORY_INDEX', 'executed': False}
        return read(params)
    def refresh(params):
        nonlocal refreshed
        if Path(params['path']).is_file() and not refreshed:
            return {'error': 'MOCK stale directory index applies to all file-content endpoints. Refresh the containing directory index first.',
                    'error_type': 'STALE_DIRECTORY_INDEX', 'executed': False}
        result = inspect(params)
        if str(Path(params['path']).resolve()) == str(root.resolve()) and not result.get('error'):
            refreshed = True
        return result
    session.registry.get('read_file').execute = stale_index
    session.registry.get('inspect_path').execute = refresh
    real = session._call_api
    model_calls = []
    reflection_seen = []
    def bounded(current):
        if len(model_calls) >= 7:
            raise ExecutionBudgetExceeded('offline reflection scenario budget')
        if session.context.get('reflection_required'):
            reflection_seen.append(copy.deepcopy(session.context['reflection_required']))
        response = real(current)
        model_calls.append([{'name': b.name, 'arguments': b.input} for b in response.content if getattr(b, 'type', '') == 'tool_use'])
        return response
    session._call_api = bounded
    result = {'kind': 'REAL_SDK_ACTUAL_ERROR_REPAIR', 'fault': 'MOCK stale directory index', 'model_calls': model_calls}
    try:
        set_context('sdk_reflection_repair', cid, agent.name)
        result['answer'] = session.run_until_complete(f'请直接读取这个任务目录{root}的a.json、b.json、c.json，告诉我三份记录的name和value。'
            '如果读取报错，先找出原因并修正读取流程，不要反复盲试或编造内容。只做文件查证，不运行或提交计算，不编排科研流程。',
            agent=agent, max_rounds=7, verbose=False)
        calls = session.memory.tool_call_log
        success = [c for c in calls if c['tool'] == 'read_file' and not c.get('failed')]
        result['checks'] = {'actual_fault_observed': rejected > 0,
            'model_selected_index_repair': refreshed,
            'all_actual_records_read': {Path(c['params']['path']).name for c in success} >= {'a.json', 'b.json', 'c.json'},
            'answer_has_real_values': all(value in result['answer'] for value in ['Alpha evidence', 'Beta evidence', 'Gamma evidence']),
            'bounded_failed_fetches': rejected <= 4,
            'no_fake_completion_or_budget_dead_end': '执行预算已达到' not in result['answer']}
        result['reflection_receipts'] = reflection_seen
        result['passed'] = all(result['checks'].values())
    except Exception as error:
        result.update(passed=False, error=f'{type(error).__name__}: {error}')
    finally:
        write_checkpoint(root / 'test_result.json', result)
        clear_context()
        print(json.dumps({'passed': result.get('passed'), 'report': str(root/'test_result.json'), 'error': result.get('error')}, ensure_ascii=False), flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__':
    raise SystemExit(main())
