"""Backend-restart coordination (patcher → api).

A framework-patching turn must NOT kill the uvicorn process while it is still
answering. So the `restart_backend` tool only *requests* a restart via this
leaf module; api.py consumes the flag in the turn's `finally` and performs the
actual detached restart strictly after the response has been written/persisted.
Leaf module (no imports) to avoid a circular dependency api ⇄ registry.
"""
import threading

_requested = threading.Event()


def request() -> None:
    """Patcher tool calls this to schedule a backend restart after the turn."""
    _requested.set()


def consume() -> bool:
    """api.py calls this once per turn-finish; returns True if a restart was requested."""
    if _requested.is_set():
        _requested.clear()
        return True
    return False
