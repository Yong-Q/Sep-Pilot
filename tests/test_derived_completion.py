import copy
import pytest
from tests.test_data_sandbox import scalar_nodes
from tests.test_parallel_workflow import factory, eventually

@pytest.mark.parametrize('tamper',['none','source','artifact','effective','proof','token','consumer_token','goal','schema','consumer_receipt'])
def test_completed_dag_accepts_only_current_proven_scalar(factory,tmp_path,monkeypatch,tamper):
 from agents import registry,workspace,config
 from agents.native_resource_carry import view_contract_status
 from agents.workflow_view import workflow_completion
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 rt=factory([producer,consumer],cid='data-scalar')
 monkeypatch.setattr(registry,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 monkeypatch.setattr(config,'get_config',lambda:rt.main.config)
 (root/'source.csv').write_text('x,y\n1,2\n2,4\n')
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded'
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
 if tamper=='none':
  assert not workflow_completion(line,state,goal['approved_nodes'],1)['ok']
