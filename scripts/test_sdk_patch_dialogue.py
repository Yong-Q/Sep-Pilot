"""Real main model applies a patch from natural dialogue, not approval words."""
import copy
import argparse
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import uuid

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.session import Session,ExecutionBudgetExceeded
from agents.state_io import write_checkpoint
from agents.task_line import TaskLineStore
from agents.watch_context import set_context,clear_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reject', action='store_true')
    args = parser.parse_args()
    cid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    scope=ROOT/'runs/sdk_patch_dialogue'/cid;scope.mkdir(parents=True)
    store=TaskLineStore(str(scope/'task_lines.json'))
    set_context('sdk_patch',cid,'lead-orchestrator')
    store.begin_line(cid,username='sdk_patch',conv_id=cid)
    node={'step_id':'gen','agent':'harness','tool':'generate_structure',
        'arguments':{'material_type':'MOF','n_structures':1,'output_dir':str(scope/'structures'),'memory_mb':1024},
        'depends_on':[],'expected_outputs':[{'kind':'directory','path':str(scope/'structures'),'pattern':'*.cif','min_count':1}]}
    store.apply_workflow_patch(cid,[{'operation':'upsert','step_id':'gen','node':node}],0,1,'initial approved fixture',ROOT)
    store.upsert_step(cid,'historical_read',tool='read_file',arguments={'path':str(scope/'old.log')},status='completed')
    revised=copy.deepcopy(node);revised['arguments']['memory_mb']=2048
    session=Session(config=AgentConfig.from_env(max_tokens=4096));session.client=session.client.with_options(timeout=60,max_retries=0)
    session._current_line_id=cid
    session._pending_workflow_patch={'base_version':1,'new_version':2,'reason':'same scientific inputs, resource-only repair',
        'changes':[{'operation':'upsert','step_id':'gen','node':revised}],'affected_nodes':['gen']}
    session._on_checkpoint=lambda state,reason:write_checkpoint(scope/'session_checkpoint.json',state)
    activations=[]
    session._on_workflow_start=lambda version:activations.append(version) or {'scheduled':True,'scheduler':'MOCK'}
    agent=copy.copy(ORCHESTRATOR);agent.functions=[f for f in agent.functions if f.__name__ in {'apply_workflow_patch','discard_workflow_patch','get_tool_schema'}]
    real=session._call_api;model_calls=[]
    def bounded(current):
        if len(model_calls)>=4:raise ExecutionBudgetExceeded('patch-dialogue budget')
        response=real(current);model_calls.append([{'name':b.name,'arguments':b.input} for b in response.content if getattr(b,'type','')=='tool_use']);return response
    session._call_api=bounded
    result={'kind':'REAL_SDK_MODEL_APPLIES_STRUCTURED_PATCH','model_calls':model_calls}
    try:
        with patch('agents.task_line.get_store',lambda:store):
            prompt = ('刚才那个新安排不要用了，留下当前执行链和已有结果，后面再讨论。不要让我为了拒绝重复确认；这里的调度是MOCK。' if args.reject else
                '你刚提出的待定编排补丁v2只把gen节点内存从1024改2048MiB，科学参数、目录和任务规模完全不变。就照这版办吧，别碰科学条件；把它落到执行链里继续。当前激活调度器为明确MOCK，不做实际计算。')
            result['answer']=session.run_until_complete(prompt,agent=agent,max_rounds=4,verbose=False)
            result['checks']={'model_selected_apply_operation':any(c['tool']=='apply_workflow_patch' for c in session.memory.tool_call_log),
                'exact_version_applied':session.goal_contract.approved_plan_version==2,
                'single_activation':activations==[2],
                'no_extra_confirmation':session._pending_user_interaction is None,
                'diagnostic_history_not_compiled':{n['step_id'] for n in session.goal_contract.approved_nodes}=={'gen'}}
            if args.reject:
                result['kind'] = 'REAL_SDK_MODEL_WITHDRAWS_CANDIDATE_NOT_EXECUTION'
                result['checks'] = {'model_selected_withdrawal': any(c['tool'] == 'discard_workflow_patch' for c in session.memory.tool_call_log),
                    'no_activation': not activations, 'approved_chain_unchanged': store.get_line(cid)['plan_version'] == 1,
                    'proposal_removed': session._pending_workflow_patch is None,
                    'no_extra_confirmation': session._pending_user_interaction is None,
                    'withdrawal_audited': bool(session.context.get('withdrawn_workflow_patches'))}
            result['passed']=all(result['checks'].values())
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        write_checkpoint(scope/'test_result.json',result);clear_context()
        print(json.dumps({'passed':result.get('passed'),'report':str(scope/'test_result.json'),'error':result.get('error')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
