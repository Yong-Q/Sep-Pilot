from pathlib import Path
import pytest
from agents.registry import _build_default_registry
from agents.workflow_compiler import compile_dataflow_bindings
from agents.workflow_patch import WorkflowContractError


def test_bad_sandbox_outputs_return_exact_actionable_paths_without_execution(tmp_path):
 root=tmp_path/'runs/u/c';out=root/'analysis';source=root/'source.csv'
 node={'step_id':'analysis','agent':'analyst','tool':'run_analysis_sandbox','arguments':{'script':'pass',
  'input_files':[{'path':str(source),'input_id':'T1','format':'csv'}], 'output_dir':str(out),
  'declared_outputs':[{'name':'measurements.csv','format':'csv'},{'name':'summary.json','format':'json'}]},
  'depends_on':[],'expected_outputs':[{'kind':'directory','path':str(out),'pattern':'*','min_count':1}]}
 change={'operation':'upsert','step_id':'analysis','node':node}
 with pytest.raises(WorkflowContractError,match='must exactly name')as error:
  compile_dataflow_bindings([change],_build_default_registry(),tmp_path,root)
 details=error.value.details
 assert details['error_kind']=='workflow_contract' and details['contract_code']=='sandbox_output_declaration'
 assert details['executed'] is False and details['required_outputs']==sorted([str(out/'artifacts/measurements.csv'),str(out/'artifacts/summary.json')])
 assert 'do not' in details['next_action'].lower() and not out.exists()
 # Correcting only the identified path shape succeeds without invented files.
 node['expected_outputs']=details['required_outputs']
 compiled=compile_dataflow_bindings([change],_build_default_registry(),tmp_path,root)
 assert compiled[0]['node']['expected_outputs']==details['required_outputs'] and not out.exists()


def test_bad_data_contract_reports_observed_count_and_never_publishes(tmp_path):
 import hashlib,os
 from agents.plot_sandbox import _publish_data_outputs
 root=tmp_path/'output';(root/'rendered').mkdir(parents=True);data=b'name,value\na,1\nb,2\n';(root/'rendered/table.csv').write_bytes(data)
 observed=[{'name':'table.csv','format':'csv','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'columns':['name','value'],'rows':2}]
 declared=[{'name':'table.csv','format':'csv','required_columns':['name','value'],'min_rows':3}]
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
 try:
  with pytest.raises(ValueError,match='has 2 rows; declared minimum is 3'):_publish_data_outputs(fd,root,observed,declared,[])
 finally:os.close(fd)
 assert not (root/'artifacts').exists()


def test_bad_json_schema_reports_location_without_echoing_values(tmp_path):
 import hashlib,os
 from agents.plot_sandbox import _publish_data_outputs
 root=tmp_path/'output';(root/'rendered').mkdir(parents=True);data=b'{"value":"private payload"}';(root/'rendered/value.json').write_bytes(data)
 observed=[{'name':'value.json','format':'json','sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}]
 declared=[{'name':'value.json','format':'json','json_schema':{'type':'object','properties':{'value':{'type':'number'}}}}]
 fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY)
 try:
  with pytest.raises(ValueError) as error:_publish_data_outputs(fd,root,observed,declared,[])
 finally:os.close(fd)
 assert 'schema rule type at /value' in str(error.value) and 'private payload' not in str(error.value)
 assert not (root/'artifacts').exists()


def test_duplicate_scalar_binding_gives_specific_non_overriding_feedback(tmp_path):
 from tests.test_data_sandbox import scalar_nodes
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root,explicit=True)
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in (producer,consumer)]
 with pytest.raises(WorkflowContractError,match='explicit parameter conflict')as error:
  compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 details=error.value.details
 assert details['error_kind']=='workflow_contract' and details['contract_code']=='sandbox_scalar_conflict'
 assert details['parameter']=='value' and details['executed'] is False
 assert 'Preserve user-specified scientific constraints' in details['next_action']
 assert consumer['arguments']['value']==5.0
