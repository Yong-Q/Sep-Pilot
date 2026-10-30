"""Real Python processes verify package isolation and the preserved public API."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

SDK = Path(__file__).resolve().parents[1]


def child(code, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(SDK), PYTHONDONTWRITEBYTECODE='1')
    return subprocess.run([sys.executable, '-c', code], cwd=str(tmp_path),
                          env=env, capture_output=True, text=True, timeout=40)


@pytest.mark.parametrize('module', ['agents.placement_budget', 'agents.node_inventory',
                                    'agents.cdft_grid_guard', 'agents.workload_budget'])
def test_scientific_helpers_do_not_load_model_runtime(module, tmp_path):
    result = child("""import importlib,sys
class NoModelRuntime:
 def find_spec(self, fullname, *args):
  if fullname.split('.')[0] in {'anthropic','openai'} or fullname in {'agents.router','agents.session','agents.registry'}:
   raise AssertionError('scientific import loaded model runtime: '+fullname)
sys.meta_path.insert(0,NoModelRuntime())
importlib.import_module(%r)
print('OK')
""" % module, tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'OK'


def test_public_exports_keep_the_existing_api_and_identity(tmp_path):
    result = child("""import agents,importlib,json
expected={'Agent':('agents.agent','Agent'),'AgentResult':('agents.agent','AgentResult'),
'AgentConfig':('agents.config','AgentConfig'),'get_config':('agents.config','get_config'),
'set_config':('agents.config','set_config'),'ToolRegistry':('agents.registry','ToolRegistry'),
'get_registry':('agents.registry','get_registry'),'Router':('agents.router','Router'),
'Session':('agents.session','Session'),'AGENT_REGISTRY':('agents.defns','AGENT_REGISTRY'),
'resolve_agent':('agents.defns','resolve_agent'),'ORCHESTRATOR':('agents.defns','ORCHESTRATOR')}
assert set(agents.__all__)==set(expected)
for name,(module,attr) in expected.items():
 assert getattr(agents,name) is getattr(importlib.import_module(module),attr)
 assert name in dir(agents)
namespace={};exec('from agents import *',namespace)
assert all(namespace[name] is getattr(agents,name) for name in expected)
try: agents.nonexistent_export
except AttributeError: pass
else: raise AssertionError('unknown attribute silently accepted')
print('OK')
""", tmp_path)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'OK'


def test_native_child_in_minimal_python_retains_gpu_memory_guard(tmp_path):
    # -S intentionally removes every installed model SDK; these helpers use stdlib.
    code = """import sys,json
sys.path.insert(0,%r)
from agents.placement_budget import validate_memory_budget
policy={'placement_memory_limits':[{'nodes':['gpu'],'partitions':['gpu'],'max_memory_mb_per_cpu':2048,'basis':'operator test policy'}]}
assert validate_memory_budget(policy,cpus=1,memory_mb=1536,nodelist='gpu',partition='gpu')['limited']
try:validate_memory_budget(policy,cpus=1,memory_mb=3072,nodelist='gpu',partition='gpu')
except ValueError:pass
else:raise AssertionError('memory guard lost')
assert 'anthropic' not in sys.modules and 'agents.registry' not in sys.modules
print('OK')
""" % str(SDK)
    result = subprocess.run([sys.executable, '-S', '-c', code], cwd=str(tmp_path),
                            capture_output=True,text=True,timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == 'OK'


@pytest.mark.parametrize('dependency', ['agents/__init__.py', 'agents/placement_budget.py',
                                       'agents/node_inventory.py'])
def test_changed_lightweight_boundary_is_executor_provenance(tmp_path, monkeypatch, dependency):
    import importlib.util
    overlay = SDK / 'agents/installations.py'
    spec = importlib.util.spec_from_file_location('agents.lazy_candidate_installations', overlay)
    installations = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installations)
    assert dependency in installations.CDFT_RECOVERY_DEPENDENCIES
    repo = tmp_path / 'sdk'
    target = repo / dependency
    target.parent.mkdir(parents=True)
    target.write_text('original = True\n')
    monkeypatch.setattr(installations, 'cdft_installation_files',
                        lambda root, args: {'coarse_forcefield': (repo / 'coarse', 'data')})
    first = installations.executor_fingerprint('run_cdft', repo)
    target.write_text('updated = True\n')
    assert installations.executor_fingerprint('run_cdft', repo) != first
