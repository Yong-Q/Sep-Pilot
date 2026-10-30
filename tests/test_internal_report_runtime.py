from pathlib import Path
from types import SimpleNamespace
from agents import installations

ROOT=Path(__file__).resolve().parents[1]

def test_native_report_and_figure_runtime_facts_are_readonly():
 for tool in ('generate_scientific_report','generate_result_figures','analyze_gcmc_screening',
              'validate_framework_charges','read_scientific_document','download_scientific_file'):
  fact=installations.inspect_installation(tool,{},SimpleNamespace(project_root=ROOT))
  assert fact['ok'] is True and fact['read_only'] is True
  assert fact['kind']=='internal_python_source_identity'
  assert all(row['loaded_matches_source']for row in fact['checks'])
  assert fact['execution_authorized'] is False and fact['scientific_acceptance'] is False
  assert fact['provider_connectivity_verified'] is False
  assert fact['external_dependencies_verified'] is False

def test_stale_loaded_function_or_foreign_sdk_root_fails_closed(tmp_path,monkeypatch):
 from agents import report_source_metadata
 def altered(*args,**kwargs):return 'different'
 monkeypatch.setattr(report_source_metadata,'metadata_tables',altered)
 fact=installations.inspect_installation('generate_scientific_report',{},SimpleNamespace(project_root=ROOT))
 assert fact['ok'] is False and fact['execution_authorized'] is False
 foreign=installations._inspect_internal_report_capability('generate_result_figures',{},tmp_path)
 assert foreign['ok'] is False and foreign['scientific_acceptance'] is False

def test_missing_function_does_not_become_success(monkeypatch):
 from agents import report_source_metadata
 monkeypatch.delattr(report_source_metadata,'metadata_tables')
 fact=installations.inspect_installation('generate_scientific_report',{},SimpleNamespace(project_root=ROOT))
 assert fact['ok'] is False and fact['error_type']=='ValueError'
