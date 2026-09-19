"""Real SDK autonomous, read-only assessment of an existing scoped queued job."""
import json
from pathlib import Path
import sys
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.config import AgentConfig
from agents.session import Session
from agents.resource_review import run_resource_review
from agents.job_watch import get_watch
from agents.state_io import write_checkpoint,compact_checkpoint_evidence,apply_evidence_references
from agents.watch_context import set_context,clear_context


def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--job-id',default='77272');args=parser.parse_args()
    cid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    root=ROOT/'runs/sdk_resource_review'/cid;root.mkdir(parents=True)
    session=Session(config=AgentConfig.from_env(max_tokens=4096))
    session.client=session.client.with_options(timeout=60,max_retries=0)
    session._evidence_root=root/'evidence'
    def checkpoint(state,reason):
        refs=compact_checkpoint_evidence(state,root/'evidence')
        write_checkpoint(root/'session_checkpoint.json',state);apply_evidence_references(session.memory,refs)
    session._on_checkpoint=checkpoint
    record=get_watch().get(args.job_id) or {}
    result={'kind':'REAL_SDK_READONLY_PENDING_RESOURCE_REVIEW','job_id':args.job_id}
    try:
        if not record.get('username') or not record.get('conv_id'):raise ValueError('Existing scoped watch record required')
        set_context(record['username'],record['conv_id'],'monitor')
        receipt=run_resource_review(session,{'kind':'pending_warning','job_id':args.job_id,
            'instruction':'自主assess_job_resources检查真实作业，resource_health查资源后交付。若PENDING且脚本隐式请求全节点内存，峰值未知应needs_user/propose_resource_patch，不猜小内存或取消重提；Priority正常排队可wait_existing。'})
        tools=session.memory.tool_call_log;result['receipt']=receipt
        result['checks']={'sdk_called_actual_job_assessment':any(c['tool']=='assess_job_resources' and not c.get('failed') for c in tools),
            'source_backed_structured_review':bool(receipt.get('resource_review_id')) and bool(receipt.get('evidence_call_ids')),
            'no_mutation_or_submission':all(c['tool'] in {'resource_health','assess_job_resources','resource_review_decision'} for c in tools),
            'valid_disposition':receipt.get('status') in {'wait_existing','needs_user','propose_resource_patch'}}
        result['passed']=all(result['checks'].values())
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        clear_context();write_checkpoint(root/'test_result.json',result)
        print(json.dumps({'passed':result.get('passed'),'error':result.get('error'),'report':str(root/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
