"""Real SDK maintainer drives supplementary project checks; not E2E replacement."""
import copy
import json
from pathlib import Path
import sys
import time
import uuid
import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.config import AgentConfig
from agents.defns import HARNESS
from agents.session import Session
from agents.state_io import write_checkpoint
from agents.watch_context import set_context, clear_context


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--frontend',action='store_true');args=parser.parse_args()
    maintenance_tool='build_project_frontend' if args.frontend else 'run_project_regressions'
    cid = time.strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
    root = ROOT / 'runs/sdk_release' / cid
    root.mkdir(parents=True, exist_ok=False)
    agent = copy.copy(HARNESS)
    agent.functions = [fn for fn in agent.functions if fn.__name__ == maintenance_tool]
    session = Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client = session.client.with_options(timeout=90, max_retries=0)
    session._evidence_root = root / 'evidence'
    result = {'scope':str(root), 'kind':'REAL_SDK_MAINTAINER_SUPPLEMENTARY_REGRESSIONS'}
    try:
        set_context('sdk_release', cid, agent.name, cid)
        prompt='请自主调用build_project_frontend编译当前前端并报告真实结果，不要猜shell命令或改源码。' if args.frontend else '请自主使用run_project_regressions固定维护工具检查结构回归，报告真实结果。不提交计算、不改源码；这是补充诊断，不替代真实SDK场景。'
        result['answer'] = session.run_readonly_observer(prompt, agent, max_rounds=3)
        calls = [c for c in session.memory.tool_call_log if c['tool']==maintenance_tool]
        receipts = [session._load_evidence_call(c).get('result') for c in calls]
        receipts = [json.loads(r) if isinstance(r, str) else r for r in receipts]
        result['passed'] = bool(calls) and all(not c.get('failed') for c in calls) and all(
            isinstance(r, dict) and r.get('ok') is True for r in receipts)
        result['tool_calls'] = session.memory.tool_call_log
    except Exception as error:
        result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        write_checkpoint(root/'session_checkpoint.json',session.export_state())
        write_checkpoint(root/'test_result.json',result)
        clear_context()
        print(json.dumps({'passed':result.get('passed'),'report':str(root/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__ == '__main__': raise SystemExit(main())
