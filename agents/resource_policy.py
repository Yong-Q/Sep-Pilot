"""Native hardware demand shared by review, inventory selection and submission.

Operator profiles may add constraints, but cannot turn a GPU executable into a
CPU workload. Node names and partition names are never GPU-capacity evidence.
"""
import copy
import re

_NATIVE = {
    # Adapter launches one LAMMPS MPI task with four ranks by default.
    'run_md_optimize': {'default_cpus': 4, 'gpus': 0},
    # Preserve the adapter's existing excluded-device constraint in review too.
    'run_string_tst': {'default_cpus': 8, 'gpus': 1, 'excluded_nodes': ['gpu3']},
}


def apply_resource_allocation(arguments, allocation, properties):
    """Map reviewed resources to supported tool arguments in place.

    A final budget replaces planning hints. Legacy/non-final hints only fill
    missing arguments; neither path invents fields outside the tool schema.
    Preserve the adapter's CPU convention (task threads before MPI ranks).
    """
    values = {key: allocation.get(key) for key in ('memory_mb', 'nodelist', 'partition')}
    if allocation.get('cpus') is not None:
        cpu_key = next((key for key in ('cpus_per_task', 'num_processes') if key in properties), None)
        if cpu_key:
            values[cpu_key] = allocation['cpus']
    for key, value in values.items():
        if value is not None and key in properties:
            if allocation.get('budget_final') is True:
                arguments[key] = value
            else:
                arguments.setdefault(key, value)
    return arguments


def native_requirements(tool, arguments=None):
    if tool == 'run_pacman_charge' and (arguments or {}).get('method', 'pacmof') == 'pacman':
        return {'default_cpus': 4, 'gpus': 1, 'excluded_nodes': ['gpu3']}
    return copy.deepcopy(_NATIVE.get(tool, {'default_cpus': 1, 'gpus': 0}))


def tool_profile(snapshot, tool, arguments=None):
    native = native_requirements(tool, arguments)
    operator = snapshot.get('policy', {}).get('tool_profiles', {}).get(tool, {})
    profile = {**native, **operator}
    profile['gpus'] = max(native['gpus'], int(operator.get('gpus', operator.get('gpu', 0)) or 0))
    profile['excluded_nodes'] = sorted(set(native.get('excluded_nodes', [])) | set(operator.get('excluded_nodes', [])))
    return profile


def gpu_facts(fields):
    """Read scheduler TRES counters, including an explicitly empty AllocTRES."""
    if 'CfgTRES' not in fields:
        return {'verified': False, 'total': 0, 'allocated': 0, 'available': 0}
    def count(value):
        match = re.search(r'(?:^|,)gres/gpu=(\d+)(?:,|$)', value)
        return int(match.group(1)) if match else 0
    total = count(fields['CfgTRES'])
    allocated = count(fields.get('AllocTRES', ''))
    verified = 'AllocTRES' in fields and 0 <= allocated <= total
    return {'verified': verified, 'total': total, 'allocated': allocated,
            'available': max(0, total-allocated) if verified else 0}


def allocation_hardware(snapshot, selected, profile):
    gpus = profile.get('gpus', 0)
    if not gpus:
        return selected
    node = snapshot['nodes'][selected['nodelist']]
    return {**selected, 'gpus': gpus, 'gres': f'gpu:{gpus}',
            'hardware_requirements': {'gpus': gpus},
            'hardware_evidence': {'node': node['name'], 'partitions': node['partitions'],
                                  'gpus': dict(node['gpus']), 'observed_at': snapshot.get('observed_at')}}


def validate_gpu_allocation(allocation, tool, gres=''):
    """Reject old/inconsistent receipts; never silently substitute a GPU node."""
    match = re.search(r'(?:^|,)gpu(?::[^:,]+)?:(\d+)(?:,|$)', gres)
    required = max(native_requirements(tool)['gpus'], int(match.group(1)) if match else 0)
    if not required:
        return
    facts = allocation.get('hardware_evidence') or {}
    gpus = facts.get('gpus') or {}
    requested = allocation.get('gpus', 0)
    if (not isinstance(requested, int) or isinstance(requested, bool) or requested < required
            or allocation.get('gres') != f'gpu:{requested}'
            or allocation.get('hardware_requirements', {}).get('gpus') != requested
            or not gpus.get('verified') or gpus.get('total', 0) < requested
            or facts.get('node') != allocation.get('nodelist')
            or allocation.get('partition') not in facts.get('partitions', [])):
        raise ValueError('GPU allocation requires a fresh resource review of native hardware demand and eligible node/partition')
