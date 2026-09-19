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
    return json.loads(path.read_text())


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
                    fields = dict(re.findall(r'(\w+)=(\S+)', row))
                    node = nodes.get(fields.get('NodeName'))
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
        snapshot = {'schema_version': 1, 'observed_at': time.time(), 'nodes': nodes, 'sources': sources,
                    'verified': bool(nodes) and any(s['program'] == 'sinfo' and s['verified'] for s in sources)}
        if persist:
            from .state_io import write_checkpoint
            write_checkpoint(path, snapshot)
    snapshot = dict(snapshot)
    snapshot['age_seconds'] = max(0, time.time() - snapshot.get('observed_at', 0))
    snapshot['stale'] = snapshot['age_seconds'] > load_policy(root)['max_age_seconds']
    snapshot['policy'] = load_policy(root)
    return snapshot


def candidates(snapshot, cpus=1, min_glibc=None, memory_mb=None, for_queue=False):
    if not snapshot.get('verified') or snapshot.get('stale'): return []
    policy = snapshot['policy']
    preferred = policy.get('preferred_nodes', [])
    result = []
    for name, node in snapshot['nodes'].items():
        if name in policy['excluded_nodes'] or node['cpus']['total' if for_queue else 'idle'] < cpus: continue
        allowed = set(policy['allowed_states']) | ({'alloc'} if for_queue else set())
        if not node['states'] or any(state not in allowed for state in node['states']): continue
        capacity = node.get('memory_mib', 0) if for_queue else node.get('memory', {}).get('available_for_scheduling_mib', 0)
        if memory_mb is not None and (not node.get('memory', {}).get('verified') or capacity < memory_mb): continue
        if min_glibc and (not node.get('glibc') or tuple(map(int, node['glibc'].split('.'))) < tuple(map(int, min_glibc.split('.')))):
            continue
        result.append(node)
    return sorted(result, key=lambda node: (preferred.index(node['name']) if node['name'] in preferred else len(preferred), node['cpus']['total'], node['name']))


def profiled_resources(snapshot, tool, cpus=None, nodelist='', partition='', memory_mb=None):
    """Choose one currently schedulable target from operator policy.

    Dynamic RAM is based on scheduler-available memory, never Linux FreeMem.
    The returned snapshot is advisory; callers must revalidate at mutation or
    submission time and let Slurm remain the final allocator.
    """
    profile=snapshot.get('policy',{}).get('tool_profiles',{}).get(tool,{})
    cpus=int(cpus or profile.get('default_cpus') or 1)
    minimum=int(profile.get('minimum_memory_mb') or 64)
    requested=int(memory_mb) if memory_mb is not None else minimum
    choices=candidates(snapshot,cpus=cpus,memory_mb=requested,min_glibc=profile.get('min_glibc'))
    if nodelist: choices=[node for node in choices if node['name']==nodelist]
    preferences=[partition] if partition else profile.get('partition_preference',[])
    targets=[(part,node) for part in preferences for node in choices if part in node.get('partitions',[])]
    if not targets:return None
    selected_partition,node=targets[0]
    if memory_mb is None:
        fraction=float(profile.get('memory_fraction_by_partition',{}).get(
            selected_partition,profile.get('memory_fraction_of_available')) or 0)
        available=int(node.get('memory',{}).get('available_for_scheduling_mib') or 0)
        if not 0 < fraction <= 1 or available < minimum:return None
        requested=max(minimum,int(available*fraction))
    return {'nodelist':node['name'],'partition':selected_partition,'memory_mb':requested,
            'cpus':cpus,'available_memory_mb':int(node['memory']['available_for_scheduling_mib']),
            'observed_at':snapshot.get('observed_at')}


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
