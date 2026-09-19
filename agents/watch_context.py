"""Thread-local execution context — lets tool executors (which run inside a
session.reply/start thread) discover WHICH conversation / user invoked them,
so long-running SLURM jobs can be auto-registered with the JobWatch service.

api.py sets the context right before session.reply/start and clears it after;
registry tool executors read it via get_context().
"""
from __future__ import annotations

import threading
from typing import Dict, Any

_local = threading.local()


def set_context(username: str = "", conv_id: str = "", agent_name: str = "",
                line_id: str = "", tool_name: str = '', recovery_key: str = '',
                attempt_id: str = '', recovery_path: str = '', step_id: str = '', resource_allocation=None) -> None:
    """Set the current thread's execution context (called by api.py).

    line_id identifies the pipeline this conversation is running (defaults to
    conv_id). Tool executors use it to write TaskLine records so downstream
    agents know where the previous step's artifacts live.
    """
    _local.ctx = {
        "username": username or "",
        "conv_id": conv_id or "",
        "agent_name": agent_name or "",
        "line_id": line_id or conv_id or "",
        'tool_name': tool_name, 'recovery_key': recovery_key,
        'attempt_id': attempt_id, 'recovery_path': recovery_path,
        'step_id': step_id,
    }
    if resource_allocation:
        _local.ctx['resource_allocation'] = dict(resource_allocation)


def get_context() -> Dict[str, Any]:
    """Read the current thread's context (safe to call anywhere)."""
    return dict(getattr(_local, "ctx", {}) or {})


def clear_context() -> None:
    _local.ctx = {}
