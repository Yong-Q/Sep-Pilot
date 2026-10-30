import copy
import os
import pytest
from tests.test_data_sandbox import scalar_nodes
from tests.test_parallel_workflow import factory, eventually, node

@pytest.mark.parametrize('tamper',['none','source','artifact','effective','proof','token','consumer_token','goal','schema','consumer_receipt'])
def test_completed_dag_accepts_only_current_proven_scalar(factory,tmp_path,monkeypatch,tamper):
 from agents import registry,workspace,config
 from agents.native_resource_carry import view_contract_status
 from agents.workflow_view import workflow_completion,authoritative_line
 from agents.workflow_projection import thin_runtime,thin_line
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 rt=factory([producer,consumer],cid='data-scalar')
 monkeypatch.setattr(registry,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(config,'get_config',lambda:rt.main.config)
 (root/'source.csv').write_text('x,y\n1,2\n2,4\n')
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded',rt.snapshot()['nodes']['analyze']
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['convert']['status'] in {'succeeded','failed','uncertain'},timeout=30)
 eventually(lambda:any(e.get('kind')=='worker_completed' and (e.get('payload') or {}).get('step_id')=='convert' for e in rt.snapshot().get('outbox',{}).values()),timeout=30)
 state=rt.snapshot();n=state['nodes']['convert'];assert n['status']=='succeeded'
 goal=rt.main.goal_contract.to_dict();incoming=copy.deepcopy(n['contract'])
 if tamper=='source':(root/'source.csv').write_text('x,y\n9,9\n')
 elif tamper=='artifact':(root/'derived/artifacts/summary.json').write_text('{"mean":99}')
 elif tamper=='effective':n['effective_arguments']['value']=99
 elif tamper=='proof':n['analysis_input_proofs'][0]['artifact_sha256']='0'*64
 elif tamper=='token':state['nodes']['analyze']['token']='different'
 elif tamper=='consumer_token':n['token']='different'
 elif tamper=='goal':goal['version']+=1
 elif tamper=='schema':
  consumer_tool=registry.get_registry().get('convert_physical_units')
  monkeypatch.setitem(consumer_tool.input_schema['properties']['value'],'minimum',10)
 elif tamper=='consumer_receipt':n['result_ref']['call_id']='wrong'
 actual=view_contract_status(state,'convert',incoming,goal)['execution_acceptance_status']
 assert actual==('contract_accepted' if tamper=='none' else 'effective_science_changed')
 line={'username':'tester','conv_id':'data-scalar','steps':goal['approved_nodes']}
 result=workflow_completion(line,state,goal['approved_nodes'],1,goal)
 assert result['ok'] is (tamper=='none')
 view=authoritative_line(line,state,goal)
 projected=next(x for x in view['steps'] if x['step_id']=='convert')
 assert projected['done'] is (tamper=='none')
 thin=authoritative_line(line,thin_runtime(state),goal,state,advisory_cache=True)
 assert next(x for x in thin['steps'] if x['step_id']=='convert')['done'] is (tamper=='none')
 assert next(x for x in thin_line(thin)['steps'] if x['step_id']=='convert')['execution_acceptance_advisory'] is True
 if tamper=='none':
  assert not workflow_completion(line,state,goal['approved_nodes'],1)['ok']
  assert not next(x for x in authoritative_line(line,state)['steps'] if x['step_id']=='convert')['done']
  foreign=copy.deepcopy(state);foreign['username']='different'
  with pytest.raises(PermissionError):authoritative_line(line,thin_runtime(state),goal,foreign,advisory_cache=True)

def test_advisory_cache_large_source_invalidates_and_completion_rehashes(factory,tmp_path,monkeypatch):
 import threading
 from agents import registry,workspace,config,derived_completion_proof,task_line
 from agents.workflow_view import authoritative_line,workflow_completion
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 producer['arguments']['input_files'][0]['format']='text'
 downstream=node('later',tool='read_file',args={'path':str(root/'source.csv')},deps=['convert'])
 rt=factory([producer,consumer,downstream],cid='data-scalar')
 monkeypatch.setattr(registry,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(config,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(task_line,'get_store',lambda:rt.task_lines)
 source=root/'source.csv';source.write_bytes(b'x,y\n1,2\n'+b' '*12800000)
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded',rt.snapshot()['nodes']['analyze']
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['convert']['status'] in {'succeeded','failed','uncertain'},timeout=30)
 eventually(lambda:any(e.get('kind')=='worker_completed' and (e.get('payload') or {}).get('step_id')=='convert' for e in rt.snapshot().get('outbox',{}).values()),timeout=30)
 state=rt.snapshot();goal=rt.main.goal_contract.to_dict();line={'username':'tester','conv_id':'data-scalar','steps':goal['approved_nodes']}
 real=derived_completion_proof.proven_derived_arguments;calls=[];test_thread=threading.get_ident()
 def count(*args,**kwargs):
  if threading.get_ident()==test_thread:calls.append(1)
  return real(*args,**kwargs)
 monkeypatch.setattr(derived_completion_proof,'proven_derived_arguments',count)
 def status():return next(x for x in authoritative_line(line,state,goal,advisory_cache=True)['steps'] if x['step_id']=='convert')
 assert status()['done'] and status()['done'] and len(calls)==1
 assert rt.main._check_taskline_dependencies('read_file',{'path':str(root/'source.csv')})[0]
 assert len(calls)>=2
 before_completion=len(calls)
 assert not workflow_completion(line,state,goal['approved_nodes'],1,goal)['ok'] and len(calls)>before_completion
 old=source.stat()
 replacement=root/'replacement.csv';replacement.write_bytes(source.read_bytes());os.utime(replacement,ns=(old.st_atime_ns,old.st_mtime_ns));os.replace(replacement,source)
 assert source.stat().st_ino!=old.st_ino and source.stat().st_size==old.st_size
 previous=len(calls);assert status()['done'] and len(calls)==previous+1
 old=source.stat()
 with source.open('r+b') as f:f.seek(-1,2);f.write(b'!')
 os.utime(source,ns=(old.st_atime_ns,old.st_mtime_ns))
 assert source.stat().st_size==old.st_size and source.stat().st_mtime_ns==old.st_mtime_ns
 assert not status()['done']
 allowed,reason=rt.main._check_taskline_dependencies('read_file',{'path':str(root/'source.csv')})
 assert not allowed and 'DEPENDENCY_BLOCK' in reason
 assert not workflow_completion(line,state,goal['approved_nodes'],1,goal)['ok']

def test_outer_projection_cache_expires_even_with_same_signature_after_source_changes(tmp_path,monkeypatch):
 import asyncio
 from agents import workflow_projection as p
 source=tmp_path/'source';source.write_text('good');clock=[100.0]
 monkeypatch.setattr(p.time,'time',lambda:clock[0])
 cache=p.ProjectionCache(identity_max_age=30);sig=p.ProjectionSignature(('u','c'),('same-ledger',),('r',))
 def build():return {'data_valid':source.read_text()=='good'}
 async def run():
  first=await cache.get(('u','c'),lambda:sig,build);assert first['data_valid'] and first['_projection_view']['advisory']
  source.write_text('evil')
  still=await cache.get(('u','c'),lambda:sig,build);assert still['data_valid'] and still['_projection_view']['advisory']
  clock[0]+=31
  refreshed=await cache.get(('u','c'),lambda:sig,build);assert not refreshed['data_valid']
 asyncio.run(run());cache.close()
