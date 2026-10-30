import hashlib,json
from pathlib import Path
import pytest
from agents.plot_sandbox import plot_result_code
from agents.plot_sandbox_worker import native_png_output_contract


def owned(tmp_path):
 root=tmp_path/'runs/u/c';root.mkdir(parents=True);source=root/'data.csv';source.write_text('x,y\n1,2\n2,4\n')
 return root,source

def run(root,source,script,outputs,name='attempt'):
 sha=hashlib.sha256(source.read_bytes()).hexdigest()
 return plot_result_code(script,input_files=[{'path':str(source),'input_id':'T1','format':'csv','expected_sha256':sha}],output_dir=str(root/name),owned_root=root,_mode='data',_declared_outputs=outputs,_owner={'username':'u','conv_id':'c'})

SCRIPT='''import csv,json,pathlib,socket\np=pathlib.Path('/inputs/T1.csv')\nrows=list(csv.DictReader(p.open()))\nassert not pathlib.Path('/home/other-user').exists()\ntry:\n p.write_text('bad')\n raise AssertionError('writable input')\nexcept OSError:pass\ntry:\n socket.socket()\n raise AssertionError('network available')\nexcept OSError:pass\nvalues=[float(row['y']) for row in rows]\npathlib.Path('/output/summary.json').write_text(json.dumps({'mean':sum(values)/len(values),'n':len(values)}))\npathlib.Path('/output/clean.csv').write_text('x,y\\n1,2\\n2,4\\n')\n'''
DECL=[{'name':'summary.json','format':'json','json_schema':{'type':'object','required':['mean','n'],'properties':{'mean':{'type':'number'},'n':{'type':'integer'}},'additionalProperties':False}}, {'name':'clean.csv','format':'csv','required_columns':['x','y'],'min_rows':2}]

def test_real_data_sandbox_has_owned_bound_artifacts(tmp_path):
 root,source=owned(tmp_path);r=run(root,source,SCRIPT,DECL)
 assert r['ok'] is True and r['sandbox']['os_isolated'] is True and r['scientific_acceptance'] is False
 assert r['owner']=={'username':'u','conv_id':'c'} and {x['name']for x in r['artifacts']}=={'summary.json','clean.csv'}
 assert json.loads((root/'attempt/artifacts/summary.json').read_text())=={'mean':3.0,'n':2}
 for a in r['artifacts']:
  path=Path(a['path']);assert path.is_relative_to(root/'attempt/artifacts') and hashlib.sha256(path.read_bytes()).hexdigest()==a['sha256']
 assert json.loads((root/'attempt/execution.json').read_text())['artifacts']==r['artifacts'] and source.read_text()=='x,y\n1,2\n2,4\n'

@pytest.mark.parametrize('script,decl',[
 ("open('/output/data.json','w').write('{\\\"x\\\":NaN}')",[{'name':'data.json','format':'json'}]),
 ("open('/output/data.json','w').write('{\\\"x\\\":1,\\\"x\\\":2}')",[{'name':'data.json','format':'json'}]),
 ("open('/output/data.csv','w').write('x,x\\n1,2\\n')",[{'name':'data.csv','format':'csv'}]),
 ("open('/output/data.txt','w').write('x')",[{'name':'data.json','format':'json'}]),
 ("import os;os.symlink('/inputs/T1.csv','/output/data.csv')",[{'name':'data.csv','format':'csv'}]),
 ("open('/output/data.json','w').write('{\\\"x\\\":1}')",[{'name':'data.json','format':'json','json_schema':{'type':'object','required':['missing']}}]),
])
def test_invalid_outputs_never_publish(tmp_path,script,decl):
 root,source=owned(tmp_path);r=run(root,source,script,decl)
 assert r['sandbox']['os_isolated'] is True and r['exit_code']==0
 assert r['ok'] is False and not (root/'attempt/artifacts').exists()

def test_schema_cannot_resolve_external_or_regex_references(tmp_path):
 root,source=owned(tmp_path)
 for schema in [{'$ref':'https://example.invalid/schema'}, {'type':'string','pattern':'(a+)+$'}]:
  with pytest.raises(ValueError):run(root,source,'pass',[{'name':'data.json','format':'json','json_schema':schema}])

def test_owner_and_path_cannot_escape_session(tmp_path):
 root,source=owned(tmp_path);foreign=tmp_path/'foreign.csv';foreign.write_text('x\n1\n')
 with pytest.raises((ValueError,PermissionError,OSError)):
  run(root,foreign,SCRIPT,DECL)
 with pytest.raises(PermissionError):
  plot_result_code(SCRIPT,input_files=[{'path':str(source),'input_id':'T1','format':'csv'}],output_dir=str(root/'x'),owned_root=root,_mode='data',_declared_outputs=DECL,_owner={'username':'other','conv_id':'c'})

def test_png_output_contract_unchanged():
 assert native_png_output_contract()['allowed_suffixes']==['.png'] and native_png_output_contract()['auxiliary_files_allowed'] is False


def test_declared_artifact_binds_to_downstream_file_parameter(tmp_path):
 from agents.registry import _build_default_registry
 from agents.workflow_compiler import compile_dataflow_bindings
 from agents.workflow_patch import WorkflowContractError
 root=tmp_path/'runs/u/c';out=root/'analysis';source=root/'data.csv'
 a={'step_id':'analyze','agent':'analyst','tool':'run_analysis_sandbox','arguments':{'script':'pass','input_files':[{'path':str(source),'input_id':'T1','format':'csv'}],'output_dir':str(out),'declared_outputs':[{'name':'summary.json','format':'json'}]},'depends_on':[],'expected_outputs':[str(out/'artifacts/summary.json')]}
 b={'step_id':'read','agent':'analyst','tool':'read_file','arguments':{},'depends_on':['analyze'],'expected_outputs':[],'input_bindings':{'path':{'step_id':'analyze','output_index':0}}}
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in [a,b]]
 compiled=compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 assert compiled[1]['node']['arguments']['path']==str(out/'artifacts/summary.json')
 changes[0]['node']['expected_outputs']=[str(root/'unrelated.json')]
 with pytest.raises(WorkflowContractError,match='exactly name declared artifacts'):
  compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)

def test_native_tool_is_registered_for_scientific_roles():
 from agents.registry import _build_default_registry
 from agents.defns import ANALYST,COMMUNICATOR
 registry=_build_default_registry();tool=registry.get('run_analysis_sandbox')
 assert tool is not None and not registry.validate_params('run_analysis_sandbox',{'script':'pass','input_files':[{'path':'/x','input_id':'T1','format':'csv'}],'output_dir':'/y','declared_outputs':[{'name':'data.json','format':'json'}]})
 assert 'run_analysis_sandbox' in ANALYST.function_map() and 'run_analysis_sandbox' in COMMUNICATOR.function_map()

from tests.test_parallel_workflow import factory,node,eventually

@pytest.mark.parametrize('tamper',['none','artifact','token','source'])
def test_actual_dag_binds_and_rechecks_original_data_artifact(factory,tmp_path,monkeypatch,tamper):
 import time
 from agents import registry,workspace
 root=tmp_path/'runs/tester/data-dag';source=root/'source.csv';out=root/'derived';artifact=out/'artifacts/summary.json'
 producer=node('analyze',tool='run_analysis_sandbox',args={'script':"import json,pathlib\npathlib.Path('/output/summary.json').write_text(json.dumps({'mean':3.0}))",'input_files':[{'path':str(source),'input_id':'T1','format':'csv'}],'output_dir':str(out),'declared_outputs':[{'name':'summary.json','format':'json','json_schema':{'type':'object','required':['mean']}}]},outputs=[str(artifact)])
 consumer=node('consume',tool='read_file',args={'path':str(artifact)},deps=['analyze']);consumer['input_bindings']={'path':{'step_id':'analyze','output_index':0}}
 rt=factory([producer,consumer],cid='data-dag');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 source.write_text('x,y\n1,2\n2,4\n')
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded',rt.snapshot()['nodes']['analyze']
 if tamper=='artifact':artifact.write_text('{"mean":999}')
 if tamper=='source':source.write_text('x,y\n1,99\n')
 if tamper=='token':
  from agents.state_io import json_transaction
  with json_transaction(rt.store.path) as data:data['workflows'][rt.workflow_id]['nodes']['analyze']['token']='other-token'
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['consume']['status'] in {'succeeded','failed','uncertain'},timeout=30)
 result=rt.snapshot()['nodes']['consume']
 if tamper!='none':
  assert result['status']=='failed' and not result.get('tool_returned')
  assert ('artifact bytes changed' in result['error'] if tamper=='artifact' else
          'original input changed' in result['error'] if tamper=='source' else 'producer receipt' in result['error'])
 else:
  assert result['status']=='succeeded' and result['analysis_input_proofs'][0]['scientific_acceptance'] is False
  assert '3.0' in str(result.get('result'))


def scalar_nodes(root,*,explicit=False,pointer='/mean',source_schema=None):
 out=root/'derived';artifact=out/'artifacts/summary.json';source=root/'source.csv'
 schema=source_schema or {'type':'object','required':['mean'],'properties':{'mean':{'type':'number'}}}
 producer=node('analyze',tool='run_analysis_sandbox',args={'script':"import json,pathlib\npathlib.Path('/output/summary.json').write_text(json.dumps({'mean':3.0}))",'input_files':[{'path':str(source),'input_id':'T1','format':'csv'}],'output_dir':str(out),'declared_outputs':[{'name':'summary.json','format':'json','json_schema':schema}]},outputs=[str(artifact)])
 args={'quantity':'length','from_unit':'angstrom','to_unit':'nm'}
 if explicit:args['value']=5.0
 consumer=node('convert',tool='convert_physical_units',args=args,deps=['analyze'])
 consumer['input_bindings']={'value':{'step_id':'analyze','output_index':0,'json_pointer':pointer,'value_type':'number'}}
 return producer,consumer


def test_scalar_binding_compilation_rejects_override_and_missing_schema(tmp_path):
 from agents.registry import _build_default_registry
 from agents.workflow_compiler import compile_dataflow_bindings
 from agents.workflow_patch import WorkflowContractError
 root=tmp_path/'runs/tester/data-scalar';registry=_build_default_registry()
 def changes(pair):return [{'operation':'upsert','step_id':x['step_id'],'node':x} for x in pair]
 result=compile_dataflow_bindings(changes(scalar_nodes(root)),registry,tmp_path,root)
 assert 'value' not in result[1]['node']['arguments'] and result[1]['node']['input_bindings']['value']['json_pointer']=='/mean'
 with pytest.raises(WorkflowContractError,match='unapproved'):
  compile_dataflow_bindings(changes(scalar_nodes(root,explicit=True)),registry,tmp_path,root)
 with pytest.raises(WorkflowContractError,match='not guaranteed'):
  compile_dataflow_bindings(changes(scalar_nodes(root,source_schema={'type':'object','properties':{'mean':{'type':'number'}}})),registry,tmp_path,root)
 with pytest.raises(WorkflowContractError,match='not guaranteed'):
  compile_dataflow_bindings(changes(scalar_nodes(root,pointer='/missing')),registry,tmp_path,root)


@pytest.mark.parametrize('tamper',[False,True])
def test_real_dag_resolves_bound_json_scalar_and_blocks_tampering(factory,tmp_path,monkeypatch,tamper):
 from agents import registry,workspace
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 rt=factory([producer,consumer],cid='data-scalar');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 (root/'source.csv').write_text('x,y\n1,2\n2,4\n')
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded',rt.snapshot()['nodes']['analyze']
 if tamper:(root/'derived/artifacts/summary.json').write_text('{"mean":99}')
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['convert']['status'] in {'succeeded','failed','uncertain'},timeout=30)
 result=rt.snapshot()['nodes']['convert']
 if tamper:
  assert result['status']=='failed' and 'artifact bytes changed' in result['error'] and not result.get('tool_returned')
 else:
  assert result['status']=='succeeded' and result['effective_arguments']['value']==3.0
  assert result['analysis_input_proofs'][0]['json_pointer']=='/mean' and result['analysis_input_proofs'][0]['scientific_acceptance'] is False

@pytest.mark.parametrize('approved_temperature,accepted',[(3,True),(298,False)])
def test_scalar_never_overrides_user_scientific_temperature(factory,tmp_path,monkeypatch,approved_temperature,accepted):
 from types import SimpleNamespace
 from agents import registry,workspace
 from agents.goal_contract import GoalContract
 from agents.analysis_artifact_binding import resolve_analysis_input_bindings
 root=tmp_path/'runs/tester/data-scope';producer,_=scalar_nodes(root)
 rt=factory([producer],cid='data-scope');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 (root/'source.csv').write_text('x,y\n1,2\n2,4\n');rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded'
 consumer=node('gcmc',tool='run_gcmc_isotherm',args={'gas':'CO2'},deps=['analyze']);consumer['input_bindings']={'temperature':{'step_id':'analyze','output_index':0,'json_pointer':'/mean','value_type':'number'}}
 goal=GoalContract.from_user_message(f'用 GCMC 计算 CO2 在 {approved_temperature} K 的吸附');goal.execution_authorized=True;goal.approved_plan_version=1;goal.approved_nodes=[producer,consumer]
 state=rt.snapshot();state['goal_contract']=goal.to_dict()
 class SchemaRegistry:
  def get(self,name):return SimpleNamespace(input_schema={'properties':{'temperature':{'type':'number'}}})
  def validate_params(self,name,args):return []
 if accepted:
  args,proof=resolve_analysis_input_bindings(root,state,consumer,consumer['arguments'],SchemaRegistry(),goal)
  assert args['temperature']==3.0 and proof[0]['scientific_acceptance'] is False
  with_resource={**consumer['arguments'],'memory_mb':1024}
  reviewed,proof2=resolve_analysis_input_bindings(root,state,consumer,with_resource,SchemaRegistry(),goal)
  assert reviewed['temperature']==3.0 and reviewed['memory_mb']==1024 and proof2==proof
  changed=GoalContract.from_dict(goal.to_dict());changed.version+=1
  with pytest.raises(ValueError,match='currently approved'):resolve_analysis_input_bindings(root,state,consumer,consumer['arguments'],SchemaRegistry(),changed)
 else:
  with pytest.raises(ValueError,match='scientific scope'):resolve_analysis_input_bindings(root,state,consumer,consumer['arguments'],SchemaRegistry(),goal)

def test_changed_scalar_after_ledger_claim_never_dispatches_scientific_job(factory,tmp_path,monkeypatch):
 from agents import registry,workspace,analysis_artifact_binding
 from agents.goal_contract import GoalContract
 root=tmp_path/'runs/tester/data-race';producer,_=scalar_nodes(root)
 consumer=node('gcmc',tool='run_gcmc_isotherm',args={'cif':str(root/'a.cif'),'gas':'CO2','pressure_start':1,'pressure_end':2},deps=['analyze'],outputs=[str(root/'result.csv')]);consumer['agent']='adsorption';consumer['input_bindings']={'temperature':{'step_id':'analyze','output_index':0,'json_pointer':'/mean','value_type':'number'}}
 rt=factory([producer,consumer],cid='data-race');monkeypatch.setattr(registry,'get_config',lambda:rt.main.config);monkeypatch.setattr(workspace,'get_config',lambda:rt.main.config)
 (root/'source.csv').write_text('x,y\n1,2\n2,4\n');(root/'a.cif').write_text('data_fixture\n')
 goal=GoalContract.from_user_message('用 GCMC 计算 CO2 在 3 K 下的吸附');goal.execution_authorized=True;goal.approved_plan_version=1;goal.approved_nodes=[producer,consumer];rt.main.goal_contract=goal
 calls=[];rt.main.registry.get('run_gcmc_isotherm').execute=lambda args:calls.append(args)
 rt.start(1);rt.tick();eventually(lambda:rt.snapshot()['nodes']['analyze']['status'] in {'succeeded','failed'},timeout=30)
 assert rt.snapshot()['nodes']['analyze']['status']=='succeeded'
 original=analysis_artifact_binding.resolve_analysis_input_bindings;counter=[]
 def change_before_dispatch(*args,**kwargs):
  counter.append(1)
  if len(counter)==2:(root/'derived/artifacts/summary.json').write_text('{"mean":4}')
  return original(*args,**kwargs)
 monkeypatch.setattr(analysis_artifact_binding,'resolve_analysis_input_bindings',change_before_dispatch)
 rt.tick();eventually(lambda:rt.snapshot()['nodes']['gcmc']['status'] in {'failed','uncertain','succeeded'},timeout=30)
 n=rt.snapshot()['nodes']['gcmc'];assert n['status']=='failed' and not calls and not n.get('tool_returned')
 assert n.get('dispatch_phase')!='entered_tool'

def test_unavailable_sandbox_never_runs_code_outside_isolation(tmp_path,monkeypatch):
 from agents import plot_sandbox
 root,source=owned(tmp_path);monkeypatch.setattr(plot_sandbox.shutil,'which',lambda name:None)
 r=run(root,source,SCRIPT,DECL)
 assert r['ok'] is False and r['status']=='sandbox_unavailable' and r['sandbox']['os_isolated'] is False
 assert not (root/'attempt/artifacts').exists() and not (root/'attempt/summary.json').exists()

def test_full_compiler_keeps_bound_required_scalar_deferred(tmp_path):
 from agents.registry import _build_default_registry
 from agents.workflow_compiler import compile_workflow_changes
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in (producer,consumer)]
 compiled=compile_workflow_changes(changes,_build_default_registry(),tmp_path,root)
 node=next(item['node']for item in compiled.changes if item['step_id']=='convert')
 assert 'value' not in node['arguments'] and node['input_bindings']['value']['json_pointer']=='/mean'
