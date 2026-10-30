from agents.registry import _build_default_registry
from agents.defns import ORCHESTRATOR,ANALYST,COMMUNICATOR

def test_real_registry_exposes_concrete_sandbox_from_general_shell_discovery():
 registry=_build_default_registry();related=registry.tool_schema('run_bash')['related_capabilities']
 assert len(related)==1 and related[0]['name']=='run_analysis_sandbox'
 discovered=registry.tool_schema(related[0]['name']);assert discovered['ok'] is True
 assert {'script','input_files','declared_outputs','output_dir'}<=set(discovered['input_schema']['required'])
 assert 'related_capabilities' not in registry.tool_schema('check_job')

def test_planner_discovers_but_worker_owns_sandbox_execution():
 assert 'run_analysis_sandbox' in ORCHESTRATOR.instructions
 assert 'run_analysis_sandbox' not in ORCHESTRATOR.function_map()
 assert 'run_analysis_sandbox' in ANALYST.function_map() and 'run_analysis_sandbox' in COMMUNICATOR.function_map()
