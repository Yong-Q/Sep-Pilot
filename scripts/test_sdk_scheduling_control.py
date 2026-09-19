"""Real SDK chooses scoped retarget arguments; scheduler writes are simulated."""
import copy
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
import uuid
import argparse

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.goal_contract import GoalContract
from agents.job_watch import JobWatch
from agents.session import Session,ExecutionBudgetExceeded
from agents.state_io import write_checkpoint
from agents.watch_context import set_context,clear_context


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--cancel',choices=['confirmed','unknown'])
    parser.add_argument('--confirm-reply',action='store_true')
    args=parser.parse_args()
    cid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    root=ROOT/'runs/sdk_scheduling_control'/cid;root.mkdir(parents=True)
    username='sdk_scheduling';job_id='99077302'
    watch=JobWatch(str(root/'watch.json'));watch.register(job_id,username=username,conv_id=cid,tool='generate_structure')
    constraints={'JobState':'PENDING','ReqNodeList':'(null)','Partition':'compute','NumCPUs':'4','ReqTRES':'cpu=4,mem=1024M,node=1'}
    inventory={'verified':True,'stale':False,'policy':{'preferred_nodes':[],'excluded_nodes':['node01'],'allowed_states':['idle'],'tool_profiles':{}},
        'nodes':{'node07':{'name':'node07','states':['idle'],'partitions':['bigcpu'],'cpus':{'idle':8,'total':8},'memory':{'verified':True,'available_for_scheduling_mib':4096}}}}
    commands=[];calls=[]
    import subprocess
    native_run=subprocess.run
    def assess(job_id):return {'ok':True,'read_only':True,'effective_constraints':copy.deepcopy(constraints),'node_inventory':inventory}
    def scheduler(argv,**kwargs):
        if argv==['scancel',job_id]:
            commands.append(argv)
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        if argv[0]=='squeue' and job_id in argv:
            return SimpleNamespace(returncode=0,stdout='PENDING|0:00||(Priority)\n',stderr='')
        if argv[:2]!=['scontrol','update']:return native_run(argv,**kwargs)
        commands.append(argv)
        constraints.update(ReqNodeList='node07',Partition='bigcpu')
        return SimpleNamespace(returncode=0,stdout='',stderr='')
    session=Session(config=AgentConfig.from_env(max_tokens=4096));session.client=session.client.with_options(timeout=60,max_retries=0)
    session._on_checkpoint=lambda state,reason:write_checkpoint(root/'session_checkpoint.json',state)
    control_tool='cancel_watched_job' if args.cancel else 'retarget_queued_job'
    allowed={control_tool,'get_tool_schema'} | ({'request_user_decision'} if args.confirm_reply else set())
    agent=copy.copy(ORCHESTRATOR);agent.functions=[f for f in agent.functions if f.__name__ in allowed]
    from agents.control_policy import role_denial,bypasses_data_lease
    # A live data-path lease blocks ordinary writes, not scoped main controls.
    session._on_tool_guard=lambda name,params: role_denial(session.current_agent.name,name) or ('' if bypasses_data_lease(session.current_agent.name,name) or name=='request_user_decision' else 'test live data-path lease')
    real=session._call_api
    def bounded(current):
        if len(calls)>=(6 if args.confirm_reply else 4):raise ExecutionBudgetExceeded('Scheduling control model-call budget reached')
        response=real(current);calls.append([{'name':b.name,'arguments':b.input} for b in response.content if getattr(b,'type','')=='tool_use']);return response
    session._call_api=bounded
    result={'kind':'REAL_SDK_SCOPED_JOB_CONTROL_SCHEDULER_MOCK','case':args.cancel or 'retarget','model_calls':calls,'scheduler_commands':commands}
    try:
        status={'status':'CANCELLED' if args.cancel=='confirmed' else 'UNKNOWN','terminal':args.cancel=='confirmed','failed':args.cancel=='confirmed'}
        with patch('agents.job_watch.get_watch',lambda:watch),patch('agents.resource_review.assess_job_resources',assess),patch('agents.job_control.subprocess.run',scheduler),patch('agents.slurm.check_job_status',lambda *a,**k:status):
            set_context(username,cid,'lead-orchestrator')
            prompt=f'现有自有作业{job_id}仍PENDING。直接把这个job换到bigcpu/node07，不改变科学方法、CPU、内存或产物；不取消、不重新提交。使用正式retarget_queued_job工具，不要run_bash。调度器是明确的MOCK，仅验收控制流程。'
            if args.cancel:prompt=f'请取消作业{job_id}。这是本会话自有作业，明确授权取消。请自主使用cancel_watched_job，不用shell，不重提；如果终止尚未核验就如实报告，不能释放租约。调度器是明确MOCK，仅验收控制流程。'
            if args.confirm_reply:prompt=f'本会话作业{job_id}因workflow锁住，我现在只是询问：能取消吗？请先用request_user_decision正式询问我是否取消这个作业，candidate_job_id填真实ID，等待我回答后再行动；现在不执行scancel。调度器是明确MOCK，只验收多轮控制流程。'
            result['answer']=session.run_until_complete(prompt,agent=agent,max_rounds=4,verbose=False)
            if args.confirm_reply:
                result['first_answer']=result['answer']
                result['first_turn_no_scheduler_mutation']=not commands
                result['formal_question_created']=bool(session._pending_user_interaction and session._pending_user_interaction['tool']=='user_decision')
                result['answer']=session.reply('嗯',max_rounds=4,verbose=False)
            used=[c for c in session.memory.tool_call_log if c['tool']==control_tool]
            receipts=[json.loads(c['result']) for c in used]
            result['checks']={'sdk_called_formal_retarget':bool(used),'one_update_no_cancel_or_resubmit':len(commands)==1 and commands[0][:2]==['scontrol','update'],
                'same_job_id':bool(commands) and f'JobId={job_id}' in commands[0],
                'cpu_ram_preserved':constraints['NumCPUs']=='4' and constraints['ReqTRES']=='cpu=4,mem=1024M,node=1',
                'verified_control_receipt':bool(receipts) and all(r.get('ok') and r.get('not_resubmitted') for r in receipts)}
            if args.cancel:
                result['checks']={'sdk_selected_cancel_tool':bool(used),'single_local_cancel_no_submit':commands==[['scancel',job_id]],
                    'receipt_distinguishes_request_from_termination':bool(receipts) and all(r.get('request_sent') and r.get('cancelled')==(args.cancel=='confirmed') for r in receipts),
                    'no_fake_terminal_state':watch.get(job_id)['terminal']==(args.cancel=='confirmed'),
                    'unknown_retains_pending_watch':args.cancel=='confirmed' or watch.get(job_id)['state']=='PENDING',
                    'honest_control_delivery_not_budget_dead_end':'执行预算已达到' not in result['answer'] and '科研任务未宣称完成' in result['answer']}
                if args.confirm_reply:
                    result['checks'].update(first_turn_safe=result['first_turn_no_scheduler_mutation'],formal_question=result['formal_question_created'],
                        no_repeated_approval=session._pending_user_interaction is None)
            result['passed']=all(result['checks'].values())
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        clear_context();write_checkpoint(root/'test_result.json',result)
        print(json.dumps({'passed':result.get('passed'),'error':result.get('error'),'report':str(root/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
