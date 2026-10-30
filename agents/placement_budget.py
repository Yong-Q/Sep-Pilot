"""Operator-scoped memory ceilings for scheduler placement, not tool defaults.

Rules select literal nodes or partitions. Thus a CPU workload on an operator's
GPU nodes is covered, while ordinary compute nodes keep their own RAM budgets.
This module never increases CPU counts or silently truncates required memory.
"""
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path
import re
import shlex

_NAME = re.compile(r'[A-Za-z0-9_.-]+')
_RESOURCE_ENV = {'SBATCH_MEM_PER_NODE', 'SBATCH_MEM_PER_CPU', 'SBATCH_MEM_PER_GPU',
    'SBATCH_CPUS_PER_TASK', 'SBATCH_NTASKS', 'SBATCH_NPROCS', 'SBATCH_NODES',
    'SBATCH_NODELIST', 'SBATCH_PARTITION', 'SBATCH_NTASKS_PER_NODE'}


def project_policy(project_root=None):
    if project_root is None:
        from .config import get_config
        project_root = get_config().project_root
    # A clean source export need not contain private operator settings. Missing
    # settings mean no site-specific ceiling; malformed existing settings fail.
    explicit = Path(project_root) / 'env/node_policy.json'
    fallback = Path(__file__).resolve().parents[1] / 'env/node_policy.json'
    if not explicit.exists() and not fallback.exists():
        return {'placement_memory_limits': []}
    from .node_inventory import load_policy
    return load_policy(Path(project_root))


def _names(value):
    if not value:
        return []
    if isinstance(value, str):
        # Scheduler-reviewed placements are one literal node. Comma lists and
        # simple host ranges are supported for callers of the public helper.
        parts = re.findall(r'[^,\[\]]+(?:\[[^\]]+\])?', value)
        result = []
        for part in parts:
            if '[' not in part:
                if not _NAME.fullmatch(part): raise ValueError('invalid placement name')
                result.append(part); continue
            match = re.fullmatch(r'([A-Za-z0-9_.-]+)\[([0-9,-]+)\]', part)
            if not match: raise ValueError('unsupported placement host expression')
            for span in match[2].split(','):
                bounds = span.split('-')
                if len(bounds) > 2: raise ValueError('invalid placement host range')
                first, last = int(bounds[0]), int(bounds[-1])
                if first > last or last - first > 255: raise ValueError('placement host range exceeds bound')
                width = len(bounds[0]) if bounds[0].startswith('0') else 0
                result.extend(match[1] + str(number).zfill(width) for number in range(first, last + 1))
        if len(result) > 256: raise ValueError('placement host list exceeds bound')
        return result
    raise ValueError('placement names must use scheduler strings')


def memory_ceiling(policy, *, nodelist='', partition=''):
    """Return the tightest matching MiB/CPU rule; no rule means no limit."""
    rules = policy.get('placement_memory_limits', [])
    if not isinstance(rules, list): raise ValueError('operator placement_memory_limits must be a list')
    if not rules: return None
    nodes, partitions = set(_names(nodelist)), set(_names(partition))
    ceilings = []
    for rule in rules:
        if (not isinstance(rule, dict) or set(rule) - {'nodes','partitions','max_memory_mb_per_cpu','basis'}
                or type(rule.get('max_memory_mb_per_cpu')) is not int
                or rule['max_memory_mb_per_cpu'] < 64 or not isinstance(rule.get('basis'), str)
                or not rule['basis'].strip()):
            raise ValueError('invalid operator placement memory rule')
        selectors = []
        for field in ('nodes', 'partitions'):
            names = rule.get(field, [])
            if not isinstance(names, list) or any(not isinstance(name, str) or not _NAME.fullmatch(name) for name in names):
                raise ValueError('placement rule selectors must be literal names')
            selectors.append(set(names))
        if not any(selectors): raise ValueError('placement memory rule needs an explicit scope')
        if nodes & selectors[0] or partitions & selectors[1]:
            ceilings.append(rule['max_memory_mb_per_cpu'])
    return min(ceilings) if ceilings else None


def validate_memory_budget(policy, *, cpus, memory_mb, nodelist='', partition=''):
    ceiling = memory_ceiling(policy, nodelist=nodelist, partition=partition)
    if ceiling is None: return {'limited': False}
    if type(cpus) is not int or cpus < 1 or type(memory_mb) is not int or memory_mb < 64:
        raise ValueError('scoped placement requires explicit positive CPU and finite MiB memory')
    if memory_mb > cpus * ceiling:
        raise ValueError('placement memory exceeds operator MiB-per-CPU ceiling; do not increase CPUs to justify RAM')
    return {'limited': True, 'max_memory_mb_per_cpu': ceiling, 'cpus': cpus, 'memory_mb': memory_mb}


def _memory(value):
    match = re.fullmatch(r'(\d+(?:\.\d+)?)([KMGT]?)', str(value), re.I)
    if not match: raise ValueError('finite scheduler memory value required')
    try: number = Decimal(match[1]) * {'K': Decimal(1)/1024, '': 1, 'M': 1, 'G': 1024, 'T': 1024**2}[match[2].upper()]
    except InvalidOperation: raise ValueError('invalid scheduler memory')
    if number <= 0 or number != int(number): raise ValueError('scheduler memory must be positive integral MiB')
    return int(number)


def script_budget(script, policy, *, reviewed_allocation=None):
    """Validate effective header options and return explicit submission flags.

    Explicit flags prevent SBATCH_* environment settings overriding an already
    checked GPU budget. Directives after the first shell command are ignored,
    matching Slurm's script-header boundary.
    """
    if not isinstance(script, str): raise ValueError('scheduler script must be text')
    if not policy.get('placement_memory_limits'): return []
    short = {'-p':'partition','-w':'nodelist','-c':'cpus-per-task','-n':'ntasks','-N':'nodes'}
    keys = {'partition','nodelist','cpus-per-task','ntasks','nodes','ntasks-per-node',
            'mem','mem-per-cpu','mem-per-gpu'}
    values = {}; occurrences = {}
    for line in script.splitlines():
        stripped = line.lstrip()
        if not stripped or stripped.startswith('#!'): continue
        if not stripped.startswith('#'):
            break
        if not stripped.startswith('#SBATCH'): continue
        words = shlex.split(stripped[len('#SBATCH'):]); index = 0
        while index < len(words):
            word = words[index]; key = value = None
            if word.startswith('--'):
                field = word[2:].split('=',1); key = field[0]
                if key not in keys: index += 1; continue
                if len(field) == 2: value = field[1]
            elif word[:2] in short:
                key = short[word[:2]]; value = word[2:] or None
            if key in keys:
                if value is None:
                    index += 1
                    if index >= len(words): raise ValueError('missing scheduler option value')
                    value = words[index]
                values[key] = value; occurrences[key] = occurrences.get(key,0) + 1
            index += 1
    scope = dict(nodelist=values.get('nodelist',''), partition=values.get('partition',''))
    ceiling = memory_ceiling(policy, **scope)
    if ceiling is None:
        # Preserve declared CPU placement even if a SBATCH_PARTITION/NODELIST
        # environment variable tries to divert it to a scoped GPU node.
        if not any(scope.values()):
            raise ValueError('explicit scheduler placement is required when placement budgets are configured')
        return ['--'+key+'='+value for key,value in scope.items() if value]
    if any(count > 1 for key,count in occurrences.items()):
        raise ValueError('duplicate scoped scheduler resource directives require a single reviewed budget')
    if 'mem-per-gpu' in values or ('mem' in values and 'mem-per-cpu' in values):
        raise ValueError('ambiguous scoped scheduler memory directives')
    def positive(key, default):
        value = values.get(key, str(default))
        if not re.fullmatch(r'[1-9][0-9]*', value): raise ValueError('positive integral scheduler resource required')
        return int(value)
    cpus_per_task = positive('cpus-per-task', 1); tasks = positive('ntasks', 1)
    nodes = positive('nodes', 1)
    if nodes != 1 and 'mem-per-cpu' not in values:
        raise ValueError('scoped multi-node memory requires an explicit per-CPU budget')
    cpus = cpus_per_task * tasks
    reviewed = reviewed_allocation or {}
    if reviewed.get('resource_review_id'):
        if type(reviewed.get('cpus')) is not int or reviewed['cpus'] < 1:
            raise ValueError('reviewed CPU budget is invalid')
        if cpus != reviewed['cpus']:
            raise ValueError('script CPU allocation differs from the reviewed CPU ceiling')
    if 'mem-per-cpu' in values:
        per_cpu = _memory(values['mem-per-cpu'])
        validate_memory_budget(policy, cpus=1, memory_mb=per_cpu, **scope)
        total_memory = per_cpu * cpus
        memory_flag = '--mem-per-cpu=' + str(per_cpu) + 'M'
    else:
        if 'mem' not in values:
            raise ValueError('scoped GPU placement requires explicit memory; scheduler default is not a budget')
        memory = _memory(values['mem'])
        validate_memory_budget(policy, cpus=cpus, memory_mb=memory, **scope)
        total_memory = memory
        memory_flag = '--mem=' + str(memory) + 'M'
    if (reviewed.get('resource_review_id') and (type(reviewed.get('memory_mb')) is not int
            or total_memory != reviewed['memory_mb'])):
        raise ValueError('script memory allocation differs from its reviewed finite budget')
    flags = [memory_flag, '--cpus-per-task=' + str(cpus_per_task), '--ntasks=' + str(tasks),
             '--nodes=' + str(nodes)]
    flags += ['--'+key+'='+value for key,value in scope.items() if value]
    return flags


def submission_flags(script, *, project_root=None, reviewed_allocation=None):
    return script_budget(script, project_policy(project_root), reviewed_allocation=reviewed_allocation)


def clean_submission_environment():
    """Remove resource environment overrides; preserve credentials in memory."""
    return {key:value for key,value in os.environ.items() if key not in _RESOURCE_ENV}
