from pathlib import Path
from agents.registry import _build_default_registry
from agents.result_guidance import read_result_guidance
from agents.workflow_compiler import compile_workflow_changes

def test_tool_schema_describes_real_isolated_and_published_paths():
 tool=_build_default_registry().tool_schema('run_analysis_sandbox');text=tool['description']
 assert '/output/<name>' in text and 'do not create /output/artifacts' in text
 assert '<output_dir>/artifacts/<name>' in text and '/inputs/<input_id>' in text
 assert 'omit any duplicate literal value' in text and 'scientific acceptance' in text

def test_scalar_binding_schema_and_skill_remain_hash_verified():
 registry=_build_default_registry();tool=registry.tool_schema('propose_workflow_patch')
 fields=tool['input_schema']['properties']['changes']['items']['properties']['node']['properties']['input_bindings']['additionalProperties']['properties']
 assert 'RFC 6901' in fields['json_pointer']['description'] and 'Omit the bound target argument' in fields['value_type']['description']
 result=read_result_guidance(['result-guidance/evidence-closure.md']);text=result['files'][0]['text']
 assert 'directly under `/output`' in text and 'do not claim a downstream tool supplied a value before' in text

def test_generic_schema_example_compiles_without_guessing_bound_value(tmp_path):
 registry=_build_default_registry();root=tmp_path/'runs/u/c';out=root/'analysis'
 producer={'step_id':'transform','agent':'analyst','tool':'run_analysis_sandbox','arguments':{'script':"open('/output/result.json','w').write('{\"result\":1}')",'input_files':[{'path':str(root/'input.csv'),'input_id':'I1','format':'csv'}],'output_dir':str(out),'declared_outputs':[{'name':'result.json','format':'json','json_schema':{'type':'object','properties':{'result':{'type':'number'}},'required':['result']}}]},'depends_on':[],'expected_outputs':[str(out/'artifacts/result.json')]}
 consumer={'step_id':'units','agent':'analyst','tool':'convert_physical_units','arguments':{'quantity':'length','from_unit':'angstrom','to_unit':'nm'},'input_bindings':{'value':{'step_id':'transform','output_index':0,'json_pointer':'/result','value_type':'number'}},'depends_on':['transform'],'expected_outputs':[]}
 changes=[{'operation':'upsert','step_id':x['step_id'],'node':x}for x in (producer,consumer)]
 result=compile_workflow_changes(changes,registry,tmp_path,root)
 compiled=next(x['node']for x in result.changes if x['step_id']=='units')
 assert 'value' not in compiled['arguments'] and compiled['input_bindings']['value']['json_pointer']=='/result'
