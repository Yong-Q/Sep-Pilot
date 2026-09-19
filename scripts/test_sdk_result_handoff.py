"""Real SDK repairs grouped native outputs, verifies and advances a DAG."""
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
from agents.goal_contract import GoalContract
from agents.job_watch import JobWatch
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow,WorkflowStore
from agents.session import Session,ExecutionBudgetExceeded
from agents.state_io import write_checkpoint,compact_checkpoint_evidence,apply_evidence_references
from agents.task_line import TaskLineStore
from agents.watch_context import set_context,clear_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true')
    parser.add_argument('--conflict', action='store_true')
    args = parser.parse_args()
    model_budget = 8 if args.restore else 6
    cid=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    root=ROOT/'runs/sdk_handoff'/cid;root.mkdir(parents=True)
    data=root/'structures';(data/'small').mkdir(parents=True)
    main_session=Session(config=AgentConfig.from_env(max_tokens=4096));main_session.client=main_session.client.with_options(timeout=60,max_retries=0)
    main_session._current_line_id=cid;main_session._evidence_root=root/'evidence'
    lines=TaskLineStore(str(root/'lines.json'));lines.begin_line(cid,username='sdk_handoff',conv_id=cid)
    gen={'step_id':'gen','agent':'harness','tool':'generate_structure','arguments':{'material_type':'MOF','n_structures':1,'output_dir':str(data)},
        'depends_on':[],'expected_outputs':[{'kind':'directory','path':str(data),'pattern':'*.cif','min_count':1}]}
    join={'step_id':'join','agent':'analyst','tool':'read_file','arguments':{'path':str(data/'small/a.cif')},'depends_on':['gen'],'expected_outputs':[str(data/'small/a.cif')]}
    lines.apply_workflow_patch(cid,[{'operation':'upsert','step_id':n['step_id'],'node':n} for n in [gen,join]],0,1,'approved fixture',ROOT)
    mailbox=LifecycleStore(root/'lifecycle.json');mailbox.initialize()
    watch=JobWatch(str(root/'watch.json'));watch.register('99016',username='sdk_handoff',conv_id=cid,work_dir=str(data))
    runtime=ParallelWorkflow(main_session,root,WorkflowStore(root/'runtime.json'),mailbox,lines,'sdk_handoff',cid,job_watch=watch)
    main_session.goal_contract=GoalContract.from_user_message('既定任务，继续检查已有结果，不重新提交计算。')
    main_session.goal_contract.execution_authorized=True;main_session.goal_contract.execution_mode='workflow';main_session.goal_contract.approved_plan_version=1;main_session.goal_contract.approved_nodes=[gen,join]
    runtime.start(1);ticket=runtime.store.claim(runtime.workflow_id,'gen')
    # Scientifically neutral fixture; no fake adsorption/calculation result.
    (data/'small/a.cif').write_text('data_fixture\n_cell_length_a 10\n_cell_length_b 10\n_cell_length_c 10\n_cell_angle_alpha 90\n_cell_angle_beta 90\n_cell_angle_gamma 90\nloop_\n_atom_site_label\n_atom_site_type_symbol\n_atom_site_fract_x\n_atom_site_fract_y\n_atom_site_fract_z\nC1 C 0 0 0\n')
    runtime.store.update(runtime.workflow_id,'gen',ticket['token'],{'status':'validation_failed','job_ids':['99016'],
        'result':{'ok':True,'output_dir':str(data),'work_dir':str(data)},'result_ref':{'call_id':'fixtureexecution'},'output_baseline':{}},release=True)
    runtime.store.pause(runtime.workflow_id,'wrong flat pattern')
    if args.conflict:
        lines.upsert_step(cid, 'gen', status='completed', done=True, job_ids=['99016'])
    if args.restore:
        verified = copy.deepcopy(gen)
        verified['expected_outputs'] = [{'kind': 'directory', 'path': str(data / 'small'), 'pattern': '*.cif', 'min_count': 1}]
        proof = {'job_ids': ['99016'], 'execution_call_id': 'fixtureexecution',
            'artifacts': runtime._artifacts(verified, {})}
        lines.repair_output_contract(cid, 'gen', verified['expected_outputs'], proof, 1)
        lines.upsert_step(cid, 'gen', status='completed', done=True, job_ids=['99016'], validation={'output_contract': 'passed'})
        line = lines.apply_workflow_patch(cid, [{'operation': 'upsert', 'step_id': 'gen', 'node': verified}],
            1, 2, 'MOCK duplicate attempt that has subsequently stopped', ROOT)
        main_session.goal_contract.approved_plan_version = 2
        main_session.goal_contract.approved_nodes = [n for n in line['steps'] if 'expected_outputs' in n]
        prepared = [{k: copy.deepcopy(n[k]) for k in ('step_id', 'agent', 'tool', 'arguments', 'depends_on',
            'expected_outputs', 'resource_locks') if k in n} for n in main_session.goal_contract.approved_nodes]
        for n in prepared: n['resources'] = ticket['contract']['resources'] if n['step_id'] == 'gen' else []
        runtime.store.activate(runtime.workflow_id, prepared, main_session.goal_contract.to_dict(), 'sdk_handoff', cid)
        duplicate = runtime.store.claim(runtime.workflow_id, 'gen')
        runtime.store.update(runtime.workflow_id, 'gen', duplicate['token'], {'status': 'failed', 'job_ids': ['99017']}, release=True)
        watch._jobs['99016'].update(state='COMPLETED', terminal=True, failed=False)
        watch._save()
        watch.register('99017', username='sdk_handoff', conv_id=cid, work_dir=str(data))
        watch._jobs['99017'].update(state='CANCELLED', terminal=True, failed=True)
        watch._save()
        (data / 'small' / 'extra.cif').write_text('data_stopped_duplicate_fixture\n')
        write_checkpoint(root/'evidence/fixtureexecution.json', {'call_id': 'fixtureexecution', 'time': ticket['started_at'],
            'tool': 'generate_structure', 'params': gen['arguments'], 'failed': False,
            'result': json.dumps({'submitted': True, 'job_id': '99016', 'output_dir': str(data)})})
    main_session._runtime_snapshot=runtime.snapshot
    from agents.workflow_view import bind_result_handoff
    bind_result_handoff(main_session, lambda: runtime)
    def checkpoint(state,reason):
        refs=compact_checkpoint_evidence(state,root/'evidence')
        write_checkpoint(root/'session_checkpoint.json',state)
        apply_evidence_references(main_session.memory,refs)
    main_session._on_checkpoint=checkpoint
    agent=copy.copy(ORCHESTRATOR);agent.functions=[f for f in agent.functions if f.__name__ in {'inspect_path','read_file','get_tool_schema','revalidate_workflow_node_outputs','finish_workflow_node'}]
    main_session.current_agent=agent
    real=main_session._call_api;model_calls=[]
    def bounded(current):
        if len(model_calls)>=model_budget:raise ExecutionBudgetExceeded('result-handoff budget')
        response=real(current);model_calls.append([{'name':b.name,'arguments':b.input} for b in response.content if getattr(b,'type','')=='tool_use']);return response
    main_session._call_api=bounded
    result={'kind':'REAL_SDK_OUTPUT_REPAIR_PREFINISH_FINISH','model_calls':model_calls,'scheduler':'MOCK terminal only; no submission'}
    try:
        set_context('sdk_handoff',cid,'lead-orchestrator')
        with patch('agents.slurm.check_job_status',lambda *a,**k:{'status':'COMPLETED','terminal':True,'failed':False}),patch('agents.task_line.get_store',lambda:lines):
            prompt = (f'gen原来由job99016完成并已验收，历史记录和证据仍在，原目标只有1个结构a.cif，在{data}/small。'
                '后来误重提job99017，现已确认CANCELLED，但gen状态被这次重复尝试覆盖为failed，目录里多了extra.cif。'
                '两个作业都是本会话自有MOCK。请查证并恢复原作业的结果验收，不能把extra.cif算入原目标，也不要重新生成或改已有文件；'
                '验收后让只读join继续，用真实证据完成运行时接续。' if args.restore else
                f'生成任务已经结束，job99016是本会话自有已完成的MOCK作业。gen节点输出合同误写根目录*.cif，原目标只有1个结构，实际输出在{data}/small。请自主查证目录与CIF，修好产物校验后验收这个节点，接着推动join，不重新生成、不修改已有结果。用已有正式运行时工具处理，证据要真实引用。')
            result['answer']=main_session.reply(prompt,max_rounds=model_budget,verbose=False)
            runtime.tick([])
            deadline=time.monotonic()+5
            while time.monotonic()<deadline and runtime.snapshot()['nodes']['join']['status']!='succeeded':time.sleep(.05)
            state=runtime.snapshot()
            result['checks']={'model_repaired_actual_contract':any(c['tool']=='revalidate_workflow_node_outputs' for c in main_session.memory.tool_call_log),
                'model_chose_verified_finish':any(c['tool']=='finish_workflow_node' for c in main_session.memory.tool_call_log),
                'generation_finished_without_resubmit':state['nodes']['gen']['status']=='succeeded' and state['nodes']['gen']['job_ids']==['99016'],
                'dependent_advanced':state['nodes']['join']['status']=='succeeded',
                'prefinish_event_exists':any(e['kind']=='worker_prefinish' for e in mailbox.snapshot()['events'].values()),
                'finish_has_real_evidence':bool(state['nodes']['gen'].get('node_verification',{}).get('evidence_call_ids'))}
            result['checks']['no_budget_dead_end']='执行预算已达到' not in result['answer']
            if args.restore:
                result['kind'] = 'REAL_SDK_RESTORE_ARCHIVED_VERIFIED_DATASET'
                result['checks']['extra_file_preserved_but_excluded'] = (data/'small/extra.cif').exists() and all(
                    'extra.cif' not in str(x) for x in state['nodes']['gen']['contract']['expected_outputs'])
                result['checks']['duplicate_attempt_kept_as_evidence'] = bool(state['nodes']['gen'].get('attempt_history'))
            if args.conflict:
                result['kind'] = 'REAL_SDK_REPAIRS_CONTRADICTORY_REPORTED_STATE'
                result['checks']['reported_completion_did_not_skip_validation'] = any(
                    c['tool'] == 'revalidate_workflow_node_outputs' for c in main_session.memory.tool_call_log)
            result['passed']=all(result['checks'].values())
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        runtime.shutdown();write_checkpoint(root/'test_result.json',result);clear_context()
        print(json.dumps({'passed':result.get('passed'),'report':str(root/'test_result.json'),'error':result.get('error')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
