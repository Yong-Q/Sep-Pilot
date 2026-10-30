import json
from types import SimpleNamespace
from pathlib import Path
import pytest
from agents.config import AgentConfig
from agents.goal_contract import GoalContract
from agents.session import Session
from agents.sandbox_planning_contract import obvious_sandbox_path_issues
from agents.workflow_compiler import compile_dataflow_bindings
from agents.workflow_patch import WorkflowContractError
from agents.registry import _build_default_registry

@pytest.mark.parametrize('script',[
 "from pathlib import Path\nout=Path('/output/artifacts')\nout.mkdir()",
 "source='/home/example/private.csv'\nwith open(source) as f: f.read()",
 "from pathlib import Path\nPath('/private/file').read_text()",
])
def test_obviously_unavailable_file_operations_are_explained(script):
 assert obvious_sandbox_path_issues(script)

@pytest.mark.parametrize('script',[
 "from pathlib import Path\nPath('/inputs/I1.csv').read_text();Path('/output/result.json').write_text('{}')",
 "from pathlib import Path\nassert not Path('/home/other').exists()",
 "from pathlib import Path\ntry:\n Path('/home/other/x').write_text('probe')\nexcept OSError: pass",
 "message='The host path /home/example is not accessible'\nprint(message)",
 "source='/home/example/data'\nsource='/inputs/data'\nopen(source).read()",
 "from pathlib import Path\nPath('/usr/share/data.txt').read_text()",
])
def test_advisory_lint_does_not_block_safe_or_nonoperational_text(script):
 assert not obvious_sandbox_path_issues(script)


def test_compiler_rejects_only_known_invalid_paths_with_generic_feedback(tmp_path):
 root=tmp_path/'runs/u/c';out=root/'analysis';source=root/'input.csv'
 node={'step_id':'data','agent':'analyst','tool':'run_analysis_sandbox','arguments':{
  'script':"from pathlib import Path\nout=Path('/output/artifacts')\nout.mkdir()",'input_files':[{'path':str(source),'input_id':'I1','format':'csv'}],
  'output_dir':str(out),'declared_outputs':[{'name':'result.json','format':'json'}]},'depends_on':[],
  'expected_outputs':[str(out/'artifacts/result.json')]}
 changes=[{'operation':'upsert','step_id':'data','node':node}]
 with pytest.raises(WorkflowContractError,match='unavailable') as error:
  compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 assert error.value.details['error_kind']=='workflow_contract' and error.value.details['contract_code']=='sandbox_script_path'
 assert error.value.details['executed'] is False and '/inputs' in error.value.details['next_action']
 node['arguments']['script']="from pathlib import Path\nPath('/output/result.json').write_text('{}')"
 assert compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)


def test_planner_actual_provider_request_receives_current_registered_contract(tmp_path,monkeypatch):
 from agents import codex_provider
 from agents.defns import ORCHESTRATOR,ANALYST
 seen=[]
 def fake_provider(**kwargs):
  seen.append(kwargs['system']);return SimpleNamespace(content=[SimpleNamespace(type='text',text='ready')],stop_reason='end_turn')
 monkeypatch.setattr(codex_provider,'call_codex_model',fake_provider)
 session=Session(config=AgentConfig(model_provider='codex_cli',project_root=tmp_path))
 session.goal_contract=GoalContract.from_user_message('请在隔离环境自主分析已有数据')
 session.messages=[{'role':'user','content':'请在隔离环境自主分析已有数据'}]
 session._call_api(ORCHESTRATOR)
 payload=seen[-1].split('[CURRENT SANDBOX DATA CONTRACT]\n')[1].split('\n',1)[0];contract=json.loads(payload)
 assert contract['tool']==session.registry.get('run_analysis_sandbox').to_claude_schema()
 assert 'json_pointer' in contract['input_bindings_schema']['additionalProperties']['properties']
 assert 'not an execution grant' in contract['scope']
 session._call_api(ANALYST);assert '[CURRENT SANDBOX DATA CONTRACT]' not in seen[-1]
 session.goal_contract=GoalContract.from_user_message('查看现有数据');session._call_api(ORCHESTRATOR)
 assert '[CURRENT SANDBOX DATA CONTRACT]' not in seen[-1]


def test_dependent_conversion_requires_explicit_data_binding(tmp_path):
 from tests.test_data_sandbox import scalar_nodes
 root=tmp_path/'runs/tester/data-scalar';producer,consumer=scalar_nodes(root)
 consumer['input_bindings']={};consumer['arguments']['value']=12.0
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in (producer,consumer)]
 with pytest.raises(WorkflowContractError,match='unbound literal') as error:
  compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 assert error.value.details['contract_code']=='sandbox_literal_conversion' and error.value.details['executed'] is False
 # A truly independent conversion of an explicit user value remains valid.
 consumer['depends_on']=[]
 assert compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 # An approved JSON Pointer represents the data dependency without guessing.
 producer,consumer=scalar_nodes(root)
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in (producer,consumer)]
 result=compile_dataflow_bindings(changes,_build_default_registry(),tmp_path,root)
 assert result[1]['node']['input_bindings']['value']['json_pointer']=='/mean'
