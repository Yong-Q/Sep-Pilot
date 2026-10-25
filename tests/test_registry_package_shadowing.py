import subprocess
import sys
from pathlib import Path


def test_registry_does_not_shadow_project_science_knowledge_package():
    root = Path(__file__).resolve().parents[1]
    code = ("import agents.registry; import importlib.util; "
            "assert importlib.util.find_spec('bimem_agent.knowledge') is not None")
    result = subprocess.run([sys.executable, "-c", code], cwd=root,
                            capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr[-350:]
