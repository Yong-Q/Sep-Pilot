"""BiMemAgent package - legacy orchestration + new agents system."""

from .catalog import load_catalog
from .task_runner import run_task_spec

__all__ = ["load_catalog", "run_task_spec"]
