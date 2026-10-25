"""BiMemAgent package - load operational services only when needed."""

__all__ = ["load_catalog", "run_task_spec"]


def __getattr__(name):
    # Curated knowledge can be retrieved without private runtime/HPC config.
    if name == "load_catalog":
        from .catalog import load_catalog
        return load_catalog
    if name == "run_task_spec":
        from .task_runner import run_task_spec
        return run_task_spec
    raise AttributeError(name)
