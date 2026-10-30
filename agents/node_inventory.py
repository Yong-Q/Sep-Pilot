"""Source-attributed cluster facts. Availability is a snapshot, not a lease."""
import json
from pathlib import Path
import subprocess
import time


def parse_sinfo(output):
    nodes = {}
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) != 5:
            raise ValueError('unexpected sinfo node row')
        name, partition, state, cpus, memory = fields
        allocated, idle, other, total = map(int, cpus.split('/'))
        if any(n < 0 for n in (allocated, idle, other, total)) or allocated + idle + other != total:
            raise ValueError('inconsistent scheduler CPU counters')
        node = nodes.setdefault(name, {'name': name, 'partitions': [], 'states': [],
            'cpus': {'allocated': allocated, 'idle': idle, 'other': other, 'total': total},
            'memory_mib': int(memory), 'glibc': None, 'reported_os': None})
        if node['cpus'] != {'allocated': allocated, 'idle': idle, 'other': other, 'total': total}:
            raise ValueError('conflicting duplicate node resources')
        node['partitions'].append(partition.rstrip('*'))
        node['states'].append(state)
    for node in nodes.values():
        node['partitions'] = sorted(set(node['partitions']))
        node['states'] = sorted(set(node['states']))
    return nodes


def parse_pbsnodes(output):
    nodes = {}
    current = None
    for line in output.splitlines():
        if line and not line[0].isspace():
            current = line.strip()
            nodes[current] = {}
        elif current and '=' in line:
            key, value = (s.strip() for s in line.split('=', 1))
            if key == 'state': nodes[current]['pbs_state'] = value
            if key == 'status':
                import re
                match = re.search(r'(?:^|,)opsys=(.*?)(?=,arch=|$)', value)
                if match: nodes[current]['reported_os'] = match[1]
    return nodes


def load_policy(root):
    path = Path(root) / 'env/node_policy.json'
    if not path.exists(): path = Path(__file__).resolve().parents[1] / 'env/node_policy.json'
    policy = json.loads(path.read_text())
    from bimem_agent.paths import get_runtime_section
    site = get_runtime_section('resource_policy')
    if site:
        scope = site.get('scope_root')
        if not isinstance(scope, str) or not scope:
            raise ValueError('site resource_policy requires an explicit scope_root')
        if Path(scope).expanduser().resolve() == Path(root).resolve():
            import re
            extra = site.get('excluded_nodes', [])
            current = policy.get('excluded_nodes', [])
            if any(not isinstance(value, list) or any(
                    not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', name)
                    for name in value) for value in (current, extra)):
                raise ValueError('operator excluded_nodes must be a list of literal node names')
            policy['excluded_nodes'] = sorted(set(current) | set(extra))
    return policy


def read_partition_limits():
    """Native read-only partition facts; unknown results never authorize jobs."""
    import re
    result = {'verified': False, 'source': 'scontrol_show_partition',
              'observed_at': time.time(), 'limits': {}}
    try:
        probe = subprocess.run(['scontrol', 'show', 'partition', '-o'],
                               capture_output=True, text=True, timeout=8)
        if probe.returncode != 0 or len(probe.stdout) > 1024*1024:
            return result
        lines = probe.stdout.splitlines()
        if not lines or len(lines) > 512: return result
        limits = {}
        for line in lines:
            fields = dict(re.findall(r'(\w+)=(\S+)', line))
            name = fields.get('PartitionName')
            if not name or not re.fullmatch(r'[A-Za-z0-9_.-]+', name) or name in limits:
                return result
            limits[name] = {'verified': True, 'state': fields.get('State'),
                            'max_time': fields.get('MaxTime')}
        return {**result, 'verified': True, 'observed_at': time.time(), 'limits': limits}
    except (OSError, subprocess.TimeoutExpired, ValueError):
        return result


def node_inventory(root, refresh=False, persist=False):
    root = Path(root)
    path = root / 'env/node_inventory.json'
    previous = json.loads(path.read_text()) if path.exists() else {}
    if not refresh:
        snapshot = previous
    else:
        sources = []
        nodes = {}
        for name, command in [('sinfo', ['sinfo', '-N', '-h', '-o', '%N|%P|%t|%C|%m']),
                              ('pbsnodes', ['pbsnodes', '-a'])]:
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=8)
                source = {'program': name, 'verified': result.returncode == 0,
                          'exit_code': result.returncode, 'error': result.stderr[:500]}
                if result.returncode == 0:
                    if name == 'sinfo': nodes = parse_sinfo(result.stdout)
                    else:
                        for node, facts in parse_pbsnodes(result.stdout).items():
                            if node in nodes: nodes[node].update(facts)
                sources.append(source)
            except (OSError, subprocess.TimeoutExpired, ValueError) as error:
                sources.append({'program': name, 'verified': False, 'error': str(error)})
        # Never assign a libc version merely from an old node-name table.
        try:
            import re
            memory_probe = subprocess.run(['scontrol', 'show', 'nodes', '-o'], capture_output=True, text=True, timeout=8)
            if memory_probe.returncode == 0:
                for row in memory_probe.stdout.splitlines():
                    fields = dict(re.findall(r'(\w+)=(\S*)', row))
                    node = nodes.get(fields.get('NodeName'))
                    if node:
                        from .resource_policy import gpu_facts
                        node['gpus'] = gpu_facts(fields)
                    if node and 'RealMemory' in fields and 'AllocMem' in fields:
                        total, allocated = int(fields['RealMemory']), int(fields['AllocMem'])
                        node['memory'] = {'total_mib': total, 'allocated_mib': allocated,
                            'available_for_scheduling_mib': max(0, total-allocated),
                            'reported_free_mib': int(fields.get('FreeMem', 0)), 'verified': True}
            sources.append({'program':'scontrol_nodes', 'verified':memory_probe.returncode==0})
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            sources.append({'program':'scontrol_nodes', 'verified':False, 'error':str(error)})
        for name, node in nodes.items():
            old = previous.get('nodes', {}).get(name, {})
            if old.get('glibc_probe') and old.get('reported_os') == node.get('reported_os'):
                node.update(glibc=old.get('glibc'), glibc_probe=old['glibc_probe'])
        partition_facts = read_partition_limits()
        snapshot = {'schema_version': 1, 'observed_at': time.time(), 'nodes': nodes, 'sources': sources,
                    'partition_facts': partition_facts,
                    'verified': bool(nodes) and any(s['program'] == 'sinfo' and s['verified'] for s in sources)}
        if persist:
            from .state_io import write_checkpoint
            write_checkpoint(path, snapshot)
    snapshot = dict(snapshot)
    snapshot['age_seconds'] = max(0, time.time() - snapshot.get('observed_at', 0))
    snapshot['stale'] = snapshot['age_seconds'] > load_policy(root)['max_age_seconds']
    snapshot['policy'] = load_policy(root)
    return snapshot


def candidates(snapshot, cpus=1, min_glibc=None, memory_mb=None, for_queue=False, gpus=0):
    if not snapshot.get('verified') or snapshot.get('stale'): return []
    policy = snapshot['policy']
    preferred = policy.get('preferred_nodes', [])
    result = []
    for name, node in snapshot['nodes'].items():
        if name in policy['excluded_nodes'] or node['cpus']['total' if for_queue else 'idle'] < cpus: continue
        gpu = node.get('gpus', {})
        if gpus and (not gpu.get('verified') or gpu.get('total' if for_queue else 'available', 0) < gpus): continue
        allowed = set(policy['allowed_states']) | ({'alloc'} if for_queue else set())
        if not node['states'] or any(state not in allowed for state in node['states']): continue
        capacity = node.get('memory_mib', 0) if for_queue else node.get('memory', {}).get('available_for_scheduling_mib', 0)
        if memory_mb is not None and (not node.get('memory', {}).get('verified') or capacity < memory_mb): continue
        if min_glibc and (not node.get('glibc') or tuple(map(int, node['glibc'].split('.'))) < tuple(map(int, min_glibc.split('.')))):
            continue
        result.append(node)
    return sorted(result, key=lambda node: (preferred.index(node['name']) if node['name'] in preferred else len(preferred), node['cpus']['total'], node['name']))


def profiled_resources(snapshot, tool, cpus=None, nodelist='', partition='', memory_mb=None,
                       arguments=None, for_queue=False, preferred_nodelist='', preferred_partition=''):
    """Select verified capacity; hard constraints precede soft preferences.

    Immediate candidates use idle CPU/GPU and available RAM. Queue candidates
    use compatible total capacity only when the caller explicitly requests it.
    Operator pins/allow lists and explicit caller pins remain mandatory in both.
    """
    from .resource_policy import tool_profile, allocation_hardware
    profile = tool_profile(snapshot, tool, arguments)
    cpus = int(cpus or profile.get('default_cpus') or 1)
    minimum = int(profile.get('minimum_memory_mb') or 64)
    requested = int(memory_mb) if memory_mb is not None else minimum
    choices = candidates(snapshot, cpus=cpus, memory_mb=requested,
                         min_glibc=profile.get('min_glibc'), gpus=profile['gpus'], for_queue=for_queue)
    def names(value):
        return value.split(',') if isinstance(value, str) else list(value or [])
    for pin in (nodelist, profile.get('nodelist'), profile.get('allowed_nodes')):
        if pin:
            allowed = names(pin)
            choices = [node for node in choices if node['name'] in allowed]
    choices = [node for node in choices if node['name'] not in profile.get('excluded_nodes', [])]
    preferences = list(dict.fromkeys(names(preferred_partition) + names(profile.get('partition_preference'))
                                    + sorted({p for node in choices for p in node.get('partitions', [])})))
    targets = []
    for part in preferences:
        if any(pin and part not in names(pin) for pin in
               (partition, profile.get('partition'), profile.get('allowed_partitions'))):
            continue
        matches = [node for node in choices if part in node.get('partitions', [])]
        matches.sort(key=lambda node: node['name'] not in names(preferred_nodelist))
        targets.extend((part, node) for node in matches)
    if not targets:
        return None
    from .placement_budget import memory_ceiling, validate_memory_budget
    for selected_partition, node in targets:
        requested = int(memory_mb) if memory_mb is not None else minimum
        if memory_mb is None:
            fraction = float(profile.get('memory_fraction_by_partition', {}).get(
                selected_partition, profile.get('memory_fraction_of_available')) or 0)
            available = int(node.get('memory', {}).get('available_for_scheduling_mib') or 0)
            if not 0 < fraction <= 1 or available < minimum:
                continue
            requested = max(minimum, int(available*fraction))
            ceiling = memory_ceiling(snapshot['policy'], nodelist=node['name'], partition=selected_partition)
            if ceiling is not None:
                # This fraction is a default sizing hint, not measured demand.
                # Bound that default; never truncate an explicit memory budget.
                requested = min(requested, cpus * ceiling)
            if requested < minimum:
                continue
        try:
            budget = validate_memory_budget(snapshot['policy'], cpus=cpus, memory_mb=requested,
                nodelist=node['name'], partition=selected_partition)
        except ValueError:
            continue
        selected = {'nodelist': node['name'], 'partition': selected_partition, 'memory_mb': requested,
            'cpus': cpus, 'available_memory_mb': int(node['memory']['available_for_scheduling_mib']),
            'scheduling_mode': 'queue' if for_queue else 'immediate', 'observed_at': snapshot.get('observed_at')}
        if budget['limited']:
            selected['placement_memory_budget'] = budget
        return allocation_hardware(snapshot, selected, profile)
    return None


def record_probe_receipt(root, receipt_path):
    """Operator-only publication of non-private, scheduler-backed libc facts."""
    import re
    from .state_io import json_transaction
    root, path = Path(root).resolve(), Path(receipt_path).resolve()
    if not path.is_relative_to(root / 'runs') or path.name != 'compute_receipt.json':
        raise ValueError('probe receipt must be an explicit task-owned computation receipt')
    receipt = json.loads(path.read_text())
    watch = json.loads((path.parent.parent / 'job_watch.json').read_text())
    job_id = str(receipt.get('job_id'))
    job = watch.get(job_id, {})
    # JobWatch persists either a direct mapping or versioned envelope.
    if not job: job = watch.get('jobs', {}).get(job_id, {})
    if (job.get('terminal') is not True or job.get('failed') or job.get('state') != 'COMPLETED'
            or Path(job.get('work_dir', '')).resolve() != path.parent
            or receipt.get('kind') != 'real_numeric_calculation' or receipt.get('validation_passed') is not True):
        raise ValueError('libc publication requires the same actual successfully completed probe job')
    name = str(receipt['hostname']).split('.')[0]
    family, version = receipt.get('glibc', [None, None])
    if family != 'glibc' or not re.fullmatch(r'\d+\.\d+(?:\.\d+)?', str(version)):
        raise ValueError('invalid or unprobed libc version')
    with json_transaction(root / 'env/node_inventory.json') as inventory:
        node = inventory['nodes'][name]
        node.update(glibc=version, glibc_probe={'observed_at': receipt['finished_at'], 'host': receipt['hostname'],
                    'source': 'scheduler-backed real numeric probe; private job receipts stay in their task scope'})
    return {'node': name, 'glibc': version}
