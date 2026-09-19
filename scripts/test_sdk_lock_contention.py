"""Real SDK planners, real worker Sessions, two users contending for one lease."""
import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
import threading
import time
from unittest.mock import patch
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from agents.config import AgentConfig
from agents.defns import ORCHESTRATOR
from agents.lifecycle import LifecycleStore
from agents.parallel_workflow import ParallelWorkflow, WorkflowStore
from agents.registry import _build_default_registry
from agents.session import Session,ExecutionBudgetExceeded
from agents.state_io import write_checkpoint
from agents.task_line import TaskLineStore
from agents.watch_context import set_context,get_context,clear_context


def main():
    token=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    evidence=ROOT/'reports'/('sdk_lock_contention_'+token);evidence.mkdir(exist_ok=False)
    registry=_build_default_registry();store=WorkflowStore(evidence/'parallel_workflows.json')
    lines=TaskLineStore(str(evidence/'task_lines.json'));timeline=[];guard=threading.Lock()
    native=registry.get('write_file').execute
    def tracked(arguments):
        ctx=get_context();start=time.time()
        time.sleep(2)  # Bounded IO coordination probe, not physics on login node.
        result=native(arguments);finish=time.time()
        with guard:
            timeline.append({'username':ctx['username'],'conv_id':ctx['conv_id'],'start':start,'finish':finish,'path':arguments['path']})
        return result
    registry.get('write_file').execute=tracked
    runtimes=[];sessions=[];result={'kind':'REAL_SDK_MULTIUSER_GLOBAL_LOCK_CONTENTION','timeline':timeline}
    def plan(suffix):
        username='sdk_lock_'+suffix;cid=token+'_'+suffix;root=ROOT/'runs'/username/cid;root.mkdir(parents=True)
        mailbox=LifecycleStore(root/'lifecycle.json');mailbox.initialize()
        line=lines.begin_line(cid,username=username,conv_id=cid)
        session=Session(config=AgentConfig.from_env(max_tokens=4096),registry=registry)
        session.client=session.client.with_options(timeout=90,max_retries=0)
        session._current_line_id=line['line_id'];session._evidence_root=root/'evidence'
        session._on_checkpoint=lambda state,reason:write_checkpoint(root/'session_checkpoint.json',state)
        runtime=ParallelWorkflow(session,root,store,mailbox,lines,username,cid)
        session._runtime_snapshot=runtime.snapshot;session._on_workflow_start=lambda version:runtime.start(version)
        session._on_interrupt_requested=lambda:bool(runtime.snapshot())  # Freeze planners after activation; tick both together to create real contention.
        real=session._call_api;calls=[]
        def bounded(agent):
            if len(calls)>=6:raise ExecutionBudgetExceeded('lock probe planner budget reached')
            response=real(agent);calls.append({'tools':[{'name':b.name,'arguments':b.input} for b in response.content if getattr(b,'type','')=='tool_use']})
            return response
        session._call_api=bounded
        with guard:runtimes.append(runtime);sessions.append(session)
        set_context(username,cid,'lead-orchestrator',cid)
        try:
            text=f'直接执行一个平台文件协调测试，不是科学计算。你是{username}会话{cid}，只能写自己目录{root}。自主编译两个analyst节点：用write_file写入probe.txt，内容为自己的username和conv_id；然后read_file读取同一文件，依赖写节点。两个节点都声明资源锁global:sdk-lock-{token}，写节点expected_outputs是实际probe.txt文件。自主查schema并propose_workflow_patch编译完整参数、依赖和产物；不要串行handoff、不要run_bash或计算提交。不同用户的同名全局锁必须互斥，但私有文件不能混同。'
            answer=session.run_until_complete(text,agent=copy.copy(ORCHESTRATOR),max_rounds=6,verbose=False)
            write_checkpoint(root/'planner_result.json',{'answer':answer,'model_calls':calls})
            if not runtime.snapshot():raise RuntimeError('SDK did not compile approved file DAG')
            return root
        finally:clear_context()
    try:
        with patch('agents.defns._registry',registry),patch('agents.task_line.get_store',lambda:lines):
            with ThreadPoolExecutor(max_workers=2) as pool:roots=list(pool.map(plan,['a','b']))
            deadline=time.monotonic()+30
            while not all(r.snapshot().get('status')=='completed' for r in runtimes):
                if time.monotonic()>deadline:raise TimeoutError('lease contention did not settle')
                for runtime in runtimes:runtime.tick([])
                time.sleep(.05)
            ordered=sorted(timeline,key=lambda x:x['start'])
            result['checks']={'two_user_writes':len(ordered)==2,
                'global_resource_serialized':len(ordered)==2 and ordered[1]['start']>=ordered[0]['finish'],
                'private_files_isolated':all((root/'probe.txt').read_text().find(root.parent.name)>=0 for root in roots),
                'all_workers_finished':all(all(n['status']=='succeeded' for n in r.snapshot()['nodes'].values()) for r in runtimes),
                'leases_released':not store.snapshot().get('leases')}
            result['passed']=all(result['checks'].values());result['snapshots']=[r.snapshot() for r in runtimes]
    except Exception as error:result.update(passed=False,error=f'{type(error).__name__}: {error}')
    finally:
        for runtime in runtimes:
            if all(f.done() for f in runtime.futures.values()):runtime.shutdown()
        write_checkpoint(evidence/'test_result.json',result)
        print(json.dumps({'passed':result.get('passed'),'error':result.get('error'),'report':str(evidence/'test_result.json')},ensure_ascii=False),flush=True)
    return 0 if result.get('passed') else 1


if __name__=='__main__':raise SystemExit(main())
