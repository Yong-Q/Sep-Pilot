"""BiMemAgent public agent API, loaded only when its exports are requested.

Scientific tools also reuse lightweight modules in this package from their own
Python environments. Importing a scheduler policy must not load the model SDK,
router, registry or session, which belong to the application environment.
"""
from importlib import import_module

_EXPORTS = {
    'Agent': ('.agent', 'Agent'),
    'AgentResult': ('.agent', 'AgentResult'),
    'AgentConfig': ('.config', 'AgentConfig'),
    'get_config': ('.config', 'get_config'),
    'set_config': ('.config', 'set_config'),
    'ToolRegistry': ('.registry', 'ToolRegistry'),
    'get_registry': ('.registry', 'get_registry'),
    'Router': ('.router', 'Router'),
    'Session': ('.session', 'Session'),
    'AGENT_REGISTRY': ('.defns', 'AGENT_REGISTRY'),
    'resolve_agent': ('.defns', 'resolve_agent'),
    'ORCHESTRATOR': ('.defns', 'ORCHESTRATOR'),
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
    module, attribute = target
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
