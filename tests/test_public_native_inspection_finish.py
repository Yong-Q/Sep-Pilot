"""The public serialized native inspection authenticates its own supported method."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from agents.state_io import compact_checkpoint_evidence,apply_evidence_references,write_checkpoint,json_transaction
from agents.workflow_result_inspection import native_inspection_kind
from tests.test_parallel_workflow import factory,node,eventually
from tests.test_gcmc_result_contract import ARGS,native_case


@pytest.mark.parametrize('contract,kind',[
 ({'tool':'run_gcmc_isotherm'},'gcmc_native'),({'tool':'run_gcmc_batch'},'gcmc_native'),
 ({'tool':'run_cdft','arguments':{'action':'submit'}},'cdft_native'),
 ({'tool':'run_cdft','arguments':{'action':'collect'}},None),
 ({'tool':'run_string_tst'},None),({'tool':'unknown'},None),(None,None)])
def test_only_implemented_native_capabilities_have_a_shared_fact_kind(contract,kind):
 assert native_inspection_kind(contract)==kind


@pytest.fixture(params=['run_gcmc_isotherm','run_gcmc_batch'])
def native_ready(factory,tmp_path,request):
 tool=request.param;cid='public-native-'+tool;scope=tmp_path/'runs/tester'/cid
 work=scope/'gcmc'
 cifs=scope/'inputs';cif=cifs/'fixture.cif'
 if tool=='run_gcmc_isotherm':
  args={**ARGS,'cif':str(cif)};output=work/'P_1.00e+05/Output/System_0/output.data'
 else:
  args={'gases':['CO2'],'cif_dir':str(cifs),'temperature':298,'pressure':1,'cycles':1000,'force_field':'UFF'}
  output=work/'CO2/298K/fixture/Output/System_0/output.data'
 compute=node('compute',tool=tool,args=args,outputs=[str(output)]);compute['agent']='adsorption'
 downstream=node('read_next',tool='read_file',args={'path':str(output)},deps=['compute'],outputs=[])
 rt=factory([compute,downstream],cid=cid);submissions=[]
 work.mkdir();cifs.mkdir();cif.write_text('data_fixture\n');script=work/'submit.sh';script.write_text('native controlled submission fixture\n')
 def submit(params):submissions.append(copy.deepcopy(params));return {'submitted':True,'job_id':'42','work_dir':str(work),'script':str(script)}
 rt.main.registry.get(tool).execute=submit
 def watched(jid):
  current=rt.snapshot().get('nodes',{}).get('compute',{})
  return {'job_id':'42','username':'tester','conv_id':cid,'tool':tool,'attempt_id':current.get('attempt_id'),'recovery_key':current.get('recovery_key'),'work_dir':str(work),'terminal':True,'failed':False,'state':'COMPLETED'}
 rt.job_watch=SimpleNamespace(get=watched,list=lambda:[])
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['compute']['status']=='waiting_jobs')
 if tool=='run_gcmc_isotherm':assert native_case(work)==output
 else:
  temp=work/'CO2/298K';created=native_case(temp);(temp/'P_1.00e+05').rename(temp/'fixture');assert output.is_file()
 rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}]);assert rt.snapshot()['nodes']['compute']['status']=='prefinish'
 rt.revalidate_outputs('compute',[str(output)])
 result=rt.inspect_workflow_result('compute');actor=rt.main;actor._evidence_root=rt.root/'evidence'
 actor.memory.record_tool_call('lead-orchestrator','inspect_workflow_result',{'step_id':'compute'},json.dumps(result),failed=False)
 saved=actor.export_state();refs=compact_checkpoint_evidence(saved,actor._evidence_root);write_checkpoint(rt.root/'session_checkpoint.json',saved);apply_evidence_references(actor.memory,refs)
 call=actor.memory.tool_call_log[-1];assert 'result' not in call
 return rt,output,submissions,call


def finish(rt,call):
 return rt.finish_node('compute',[call['call_id']],'Native finite data and original inputs are verified; statistical convergence remains unproven.')


def test_public_inspection_finishes_exact_owned_gcmc_and_allows_real_next_worker(native_ready):
 from agents.scientific_review import native_inspection_sources
 rt,output,submissions,call=native_ready;before=rt.snapshot()['nodes']['compute'];ledger=(rt.root/'recovery_state.json').read_bytes();data=output.read_bytes()
 proven=native_inspection_sources([call],rt.snapshot(),rt.root/'evidence');assert len(proven)==1
 assert proven[0]['method_facts']['gcmc_native']['statistical_convergence_established'] is False
 finish(rt,call);saved=rt.snapshot()['nodes']['compute']
 assert saved['status']=='succeeded' and saved['token']==before['token'] and saved['attempt_id']==before['attempt_id']
 assert saved['job_ids']==['42'] and saved['contract']==before['contract']
 assert saved['node_verification']['native_gcmc_contract_verified'] is True
 assert saved['node_verification']['statistical_convergence_established'] is False
 rt.tick([{'job_id':'42','state':'COMPLETED','terminal':True,'failed':False}])
 for future in list(rt.futures.values()):future.result(timeout=10)
 assert rt.snapshot()['nodes']['read_next']['status']=='succeeded'
 assert len(submissions)==1 and output.read_bytes()==data
 after_ledger=json.loads((rt.root/'recovery_state.json').read_text());before_ledger=json.loads(ledger)
 for key,entry in before_ledger.items():
  assert after_ledger[key]['attempt_id']==entry['attempt_id'] and after_ledger[key]['job_ids']==entry['job_ids']
  assert after_ledger[key]['attempts']==entry['attempts']


@pytest.mark.parametrize('change',['record_hash','producer','inspection','missing_memory','duplicate_memory','wrong_method_facts','wrong_fact_type','owner','token','goal','native_data','late_producer'])
def test_public_finish_keeps_identity_bytes_and_method_guards(native_ready,monkeypatch,change):
 rt,output,submissions,call=native_ready
 if change=='record_hash':call['result_sha256']='0'*64
 elif change=='missing_memory':rt.main.memory.tool_call_log.clear()
 elif change=='duplicate_memory':rt.main.memory.tool_call_log.append(copy.deepcopy(call))
 elif change in ['producer','late_producer']:
  path=Path(rt.snapshot()['nodes']['compute']['result_ref']['evidence_path'])
  if change=='producer':path.write_text(path.read_text()+' ')
  else:
   original=rt._artifacts
   def race(*a,**kw):
    result=original(*a,**kw);path.write_text(path.read_text()+' ');return result
   monkeypatch.setattr(rt,'_artifacts',race)
 elif change=='native_data':output.write_text(output.read_text().replace('2.75','8.75'))
 elif change in ['owner','token','goal']:
  with json_transaction(rt.store.path) as data:
   state=data['workflows'][rt.workflow_id]
   if change=='owner':state['username']='foreign'
   elif change=='token':state['nodes']['compute']['token']='foreign'
   else:state['goal_contract']['version']+=1
 else:
  path=Path(call['evidence_path']);record=json.loads(path.read_text());result=json.loads(record['result'])
  if change=='inspection':result['source_execution_sha256']='0'*64
  elif change=='wrong_method_facts':result['method_facts']={'cdft_native':result['method_facts']['gcmc_native']}
  else:result['method_facts']['gcmc_native']=['untyped']
  record['result']=json.dumps(result);path.write_text(json.dumps(record));call['result_sha256']=hashlib.sha256(record['result'].encode()).hexdigest()
 with pytest.raises((ValueError,PermissionError)):finish(rt,call)
 assert rt.snapshot()['nodes']['compute']['status']=='prefinish' and len(submissions)==1
