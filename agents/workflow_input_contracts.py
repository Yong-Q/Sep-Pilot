"""Declared path semantics for automatic dependency receipt bindings.

A collection root defines a hierarchy, not a single producer's child. More
interfaces may declare this policy without changing any workflow/node identity.
"""
CONTAINER_INPUTS = {
    'analyze_gcmc_screening': frozenset({'work_dir'}),
}


def may_refine_to_child(tool, parameter):
    return parameter not in CONTAINER_INPUTS.get(tool, frozenset())
