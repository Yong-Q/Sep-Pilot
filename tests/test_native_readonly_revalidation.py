"""A real native read survives an old freshness failure without reading again."""
import copy
import hashlib
import multiprocessing
import os
from pathlib import Path
import time
import pytest
from tests.test_parallel_workflow import factory,node


def previous_native_worker(spec):
    """Rebuild process-owned runtime lanes instead of inheriting live threads."""
    from agents.config import AgentConfig, set_config
    from agents.goal_contract import GoalContract
    from agents.lifecycle import LifecycleStore
    import agents.parallel_workflow as workflow
    from agents.registry import _build_default_registry
    from agents.session import Session
    from agents.task_line import TaskLineStore
    from agents.watch_context import set_context, clear_context
    config=AgentConfig(api_key='engineering-only',project_root=Path(spec['project_root']))
    set_config(config)
    main=Session(config=config,registry=_build_default_registry())
    main.goal_contract=GoalContract.from_dict(spec['goal'])
    main._current_line_id=spec['line_id']
    rt=workflow.ParallelWorkflow(main,Path(spec['root']),workflow.WorkflowStore(Path(spec['store'])),
        LifecycleStore(Path(spec['root'])/'lifecycle.json'),TaskLineStore(spec['lines']),
        'demo-user','observed-recovery')
    main._runtime_snapshot=rt.snapshot
    set_context('demo-user','observed-recovery','lead-orchestrator')
    # Only reproduce the retired freshness predicate; extraction/evidence stay native.
    workflow._readonly_observed_artifacts=lambda *args,**kwargs:set()
    try:
        ticket=rt.store.claim(rt.workflow_id,'read')
        rt._worker('read',ticket)
    finally:
        rt.shutdown()
        clear_context()


def retired_reader(factory,tmp_path,monkeypatch):
    from agents.config import AgentConfig
    from agents.watch_context import set_context
    import agents.registry as registry
    root=tmp_path/'runs/demo-user/observed-recovery'
    source=root/'article.html';root.mkdir(parents=True)
    source.write_text('<html><body><p>Original immutable observation.</p><p>Value is 12 units.</p></body></html>')
    old=time.time()-86400;os.utime(source,(old,old))
    config=AgentConfig(api_key='engineering-only',project_root=tmp_path)
    monkeypatch.setattr(registry,'get_config',lambda:config)
    contract=node('read',tool='read_scientific_document',args={'path':str(source),'format':'html','max_chars':1000,'max_pages':2},outputs=[str(source)])
    # Factory owns creation of the conversation root.
    source_data=source.read_bytes();source.unlink();root.rmdir()
    rt=factory([contract,node('follow',tool='read_file',args={'path':str(source)},deps=['read'],outputs=[])],
        cid='observed-recovery',username='demo-user')
    source.write_bytes(source_data);os.utime(source,(old,old))
    rt.main.goal_contract.execution_mode='workflow'
    set_context('demo-user','observed-recovery','lead-orchestrator')
    rt.start(1)
    spec={'project_root':str(tmp_path),'root':str(rt.root),'store':str(rt.store.path),
          'lines':rt.task_lines._file,'line_id':rt.main._current_line_id,
          'goal':rt.main.goal_contract.to_dict()}
    process=multiprocessing.get_context('spawn').Process(target=previous_native_worker,args=(spec,))
    try:
        process.start();process.join(15)
        assert not process.is_alive() and process.exitcode==0
    finally:
        # A failed bounded child must never hang pytest's process shutdown.
        if process.is_alive():
            process.terminate();process.join(5)
            if process.is_alive():process.kill();process.join(5)
    current=rt.snapshot()['nodes']['read']
    assert current['status']=='validation_failed' and current['tool_returned'] is True
    assert current['result']['status']=='text_extracted'
    assert current['result']['source_document']['sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
    assert current['result']['blocks'][0]['anchor'].startswith('html-block-')
    assert current['owner_pid']==process.pid
    process.close()
    return rt,source,current


def test_stopped_native_reader_revalidates_exact_contract_then_releases_dependents(factory,tmp_path,monkeypatch):
    rt,source,before=retired_reader(factory,tmp_path,monkeypatch)
    original=source.read_bytes();mtime=source.stat().st_mtime_ns
    monkeypatch.setattr(rt.main.registry.get('read_scientific_document'),'execute',lambda p:pytest.fail('original document was re-extracted'))
    receipt=rt.revalidate_outputs('read',before['contract']['expected_outputs'])
    assert receipt['ok'] and receipt['no_resubmission'] and receipt['no_output_write']
    assert receipt['document_reextracted'] is False
    after=rt.snapshot()['nodes']['read']
    assert after['status']=='succeeded' and after['token']==before['token']
    assert after['contract']==before['contract'] and after['result_ref']==before['result_ref']
    assert after['result']==before['result'] and after['job_ids']==[]
    assert after['node_verification']['scientific_acceptance'] is False
    assert source.read_bytes()==original and source.stat().st_mtime_ns==mtime
    rt.tick()
    for future in list(rt.futures.values()):future.result(timeout=8)
    assert rt.snapshot()['nodes']['follow']['status']=='succeeded'


def test_native_tick_uses_same_proof_after_backend_restart_without_new_model_or_parser(factory,tmp_path,monkeypatch):
    rt,source,before=retired_reader(factory,tmp_path,monkeypatch)
    monkeypatch.setattr(rt.main.registry.get('read_scientific_document'),'execute',lambda p:pytest.fail('original document was re-extracted'))
    rt.tick()
    assert rt.snapshot()['nodes']['read']['status']=='succeeded'
    assert rt.snapshot()['nodes']['read']['result_ref']==before['result_ref']
    for future in list(rt.futures.values()):future.result(timeout=8)
    assert rt.snapshot()['nodes']['follow']['status']=='succeeded'


@pytest.mark.parametrize('change',['source','call','checkpoint','args','goal','pause','live_owner','job','new_output','directory_output','anchor','parser'])
def test_revalidation_keeps_source_ownership_approval_and_native_proof_guards(factory,tmp_path,monkeypatch,change):
    import json
    from agents.state_io import json_transaction
    rt,source,before=retired_reader(factory,tmp_path,monkeypatch)
    outputs=copy.deepcopy(before['contract']['expected_outputs'])
    if change=='source':source.write_text('<html>different source</html>')
    elif change=='call':Path(before['result_ref']['evidence_path']).write_text('{}')
    elif change=='checkpoint':Path(before['result_ref']['checkpoint_path']).write_text('{}')
    elif change=='goal':rt.main.goal_contract.version+=1
    elif change=='pause':rt.store.user_pause(rt.workflow_id,True)
    elif change=='new_output':outputs=[str(rt.root/'unrelated.json')]
    elif change=='directory_output':outputs=[{'kind':'directory','path':str(rt.root/'evidence'),'pattern':'*.json'}]
    else:
        with json_transaction(rt.store.path)as data:
            saved=data['workflows'][rt.workflow_id]['nodes']['read']
            if change=='args':saved['contract']['arguments']['max_chars']=2000
            elif change=='live_owner':saved['owner_pid']=os.getpid()
            elif change=='job':saved['job_ids']=['70999']
            else:
                saved['result']=copy.deepcopy(saved['result'])
                if change=='anchor':saved['result']['blocks'][0]['anchor']='invented'
                if change=='parser':saved['result']['parser_implementation_sha256']='0'*64
    with pytest.raises((ValueError,PermissionError,OSError,KeyError,TypeError)):
        rt.revalidate_outputs('read',outputs)
    assert rt.snapshot()['nodes']['read']['status']!='succeeded'
    assert rt.snapshot()['nodes']['follow']['status']=='pending'
