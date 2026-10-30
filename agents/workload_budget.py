"""Bounded workload observations and pure, evidence-based CPU/RAM sizing.

No scheduler, scientific program, or file writer is invoked. Caller must supply
resolved, approved inputs and trusted evidence provenance. A source label alone
is not authentication; integration must verify its owner/source before calling.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import stat

MAX_ITEMS = 4096
MAX_ENTRIES = 8192
MAX_FILE_BYTES = 16 * 1024 * 1024
MAX_TOTAL_BYTES = 64 * 1024 * 1024


def _positive(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _owned_fd(root, path, directory=False):
    root, path = Path(root), Path(path)
    if not root.is_absolute() or not path.is_absolute() or '..' in path.parts or root.is_symlink():
        raise ValueError('owned absolute paths required')
    relative = path.relative_to(root)
    fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for index, part in enumerate(relative.parts):
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if index < len(relative.parts)-1 or directory:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=fd)
            os.close(fd); fd = child
        return fd
    except BaseException:
        os.close(fd)
        raise


def _fact(fd, name):
    child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
    try:
        before = os.fstat(child)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_FILE_BYTES:
            raise ValueError('invalid or oversized regular input')
        digest = hashlib.sha256(); count = 0
        while True:
            data = os.read(child, min(1024*1024, MAX_FILE_BYTES+1-count))
            if not data: break
            count += len(data)
            if count > MAX_FILE_BYTES: raise ValueError('input exceeds byte bound')
            digest.update(data)
        after = os.fstat(child)
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) or count != before.st_size:
            raise ValueError('input changed while observing')
        return {'path': name, 'sha256': digest.hexdigest(), 'bytes': count}
    finally:
        os.close(child)


def collect_workload_facts(tool, effective_arguments, *, owned_root, manifest=None):
    """Hash exact owned prepared inputs, never infer arbitrary tool parallelism.

    Optional cDFT manifest: {schema_version:1, items:[{path:basename,sha256:hex}]}.
    It must cover every actual .dat input exactly; no omissions or duplicates.
    Only prepared submit is supported; pipeline preparation must complete first.
    """
    base = {'tool': tool, 'status': 'invalid_workload'}
    if tool not in {'run_cdft', 'run_md_optimize'}:
        return {**base, 'status': 'unsupported_workload'}
    if tool == 'run_cdft' and effective_arguments.get('action') != 'submit':
        return {**base, 'status': 'needs_prepared_inputs'}
    directory_fd = None
    try:
        if manifest is not None and (tool != 'run_cdft' or not isinstance(manifest, dict)
                or manifest.get('schema_version') != 1 or not isinstance(manifest.get('items'), list)
                or len(manifest['items']) > MAX_ITEMS):
            raise ValueError('unsupported manifest')
        if tool == 'run_cdft':
            directory = Path(effective_arguments['input_dir'])
            directory_fd = _owned_fd(owned_root, directory, directory=True)
            before = os.fstat(directory_fd)
            names = []
            with os.scandir(directory_fd) as entries:
                for index, entry in enumerate(entries):
                    if index >= MAX_ENTRIES: raise ValueError('directory entry bound exceeded')
                    if entry.name.endswith('.dat'):
                        names.append(entry.name)
                        if len(names) > MAX_ITEMS: raise ValueError('work item bound exceeded')
            items, total = [], 0
            for name in sorted(names):
                item = _fact(directory_fd, name)
                total += item['bytes']
                if total > MAX_TOTAL_BYTES:
                    raise ValueError('aggregate byte bound exceeded')
                items.append(item)
            after = os.fstat(directory_fd)
            if before.st_mtime_ns != after.st_mtime_ns: raise ValueError('input directory changed')
            model = 'independent_processes'
        else:
            path = Path(effective_arguments['cif_path'])
            directory_fd = _owned_fd(owned_root, path.parent, directory=True)
            items = [_fact(directory_fd, path.name)]
            model = 'single_task_mpi'
        if not items or sum(i['bytes'] for i in items) > MAX_TOTAL_BYTES:
            raise ValueError('empty workload or aggregate byte bound exceeded')
        if manifest is not None:
            supplied = manifest['items']
            expected = [{'path': i['path'], 'sha256': i['sha256']} for i in items]
            if len(supplied) != len(expected) or sorted(supplied, key=lambda i:i['path']) != expected:
                raise ValueError('manifest is not an exact input match')
        identity = {'tool': tool, 'execution_model': model, 'inputs': items}
        return {**base, 'status': 'ready', 'execution_model': model,
                'work_items': len(items), 'threads_per_worker': 1,
                'thread_basis': 'declared_by_adapter', 'input_fingerprint': _digest(identity),
                'inputs': items}
    except (OSError, ValueError, TypeError, KeyError):
        return {**base, 'reason': 'owned_input_or_manifest_verification_failed'}
    finally:
        if directory_fd is not None: os.close(directory_fd)


def size_workload_budget(facts, snapshot, *, suggested_resources=None,
                         constraints=(), memory_evidence=None, overhead_mb=0,
                         for_queue=False, allow_estimates=False):
    """Return feasible candidates; queue total capacity only when explicitly asked.

    Hard constraints: source user/operator, field cpus/workers/memory_mb,
    kind exact/max, positive integer value, nonempty basis. Hint/default never
    cap independent workers. Memory requires source calibrated/operator, basis,
    workload_fingerprint and per_worker_mb. For MPI this bound is per rank and
    must explicitly have unit='per_mpi_rank'; ordinary batch unit='per_worker'.
    Suggested total RAM is a cap/estimate, never evidence of per-worker RSS.
    """
    result = {'status': 'invalid_workload', 'candidates': []}
    fingerprint = facts.get('input_fingerprint')
    if (facts.get('status') != 'ready' or not _positive(facts.get('work_items'))
            or not _positive(facts.get('threads_per_worker'))
            or not isinstance(fingerprint, str) or not re.fullmatch('[a-f0-9]{64}', fingerprint)):
        return result
    model = facts.get('execution_model')
    if model not in {'independent_processes', 'single_task_mpi'}: return result
    suggested = suggested_resources or {}
    memory = memory_evidence or {}
    expected_unit = 'per_mpi_rank' if model == 'single_task_mpi' else 'per_worker'
    sources = {'calibrated', 'operator'}
    if allow_estimates is True:
        sources.update({'resource_agent_estimate', 'single_worker_budget_extrapolation'})
    if (memory.get('source') not in sources or not memory.get('basis')
            or not _positive(memory.get('per_worker_mb'))
            or memory.get('workload_fingerprint') != facts.get('input_fingerprint')
            or memory.get('unit', 'per_worker') != expected_unit):
        status = 'needs_estimate' if _positive(suggested.get('memory_mb')) else 'needs_memory_evidence'
        return {**result, 'status': status, 'needs_estimate': ['per_worker_peak_memory_mb', 'basis', 'workload_fingerprint', 'unit']}
    if (not isinstance(overhead_mb, int) or isinstance(overhead_mb, bool) or overhead_mb < 0
            or not isinstance(for_queue, bool) or not isinstance(allow_estimates, bool)):
        return {**result, 'status': 'invalid_constraints'}
    hard = {key: {'max': None, 'exact': None} for key in ('cpus', 'workers', 'memory_mb')}
    try:
        for constraint in constraints:
            source, field, kind = (constraint.get(k) for k in ('source', 'field', 'kind'))
            if source not in {'user', 'operator', 'hint', 'default'} or field not in hard or kind not in {'max', 'exact', 'preferred'} or not _positive(constraint.get('value')) or not constraint.get('basis'):
                raise ValueError('invalid constraint provenance')
            if source in {'user', 'operator'} and kind != 'preferred':
                value = constraint['value']; previous = hard[field][kind]
                if kind == 'exact' and previous not in (None, value): raise ValueError('conflicting exact constraints')
                hard[field][kind] = min(previous, value) if kind == 'max' and previous else value
        if any(v['exact'] and v['max'] and v['exact'] > v['max'] for v in hard.values()):
            raise ValueError('inconsistent constraints')
    except (ValueError, TypeError, AttributeError):
        return {**result, 'status': 'invalid_constraints'}
    if not snapshot.get('verified') or snapshot.get('stale'):
        return {**result, 'status': 'needs_resource_evidence'}
    from .node_inventory import profiled_resources
    per_worker, threads = memory['per_worker_mb'], facts['threads_per_worker']
    for name, node in sorted(snapshot.get('nodes', {}).items()):
        available = (node.get('memory', {}).get('total_mib', node.get('memory_mib')) if for_queue
                     else node.get('memory', {}).get('available_for_scheduling_mib'))
        idle = node.get('cpus', {}).get('total' if for_queue else 'idle')
        if not node.get('memory', {}).get('verified') or not _positive(available) or not _positive(idle): continue
        cpu_limit, ram_limit = idle, available
        for value in hard['cpus'].values():
            if value: cpu_limit = min(cpu_limit, value)
        for value in hard['memory_mb'].values():
            if value: ram_limit = min(ram_limit, value)
        if _positive(suggested.get('memory_mb')): ram_limit = min(ram_limit, suggested['memory_mb'])
        units = min(cpu_limit // threads, max(0, (ram_limit-overhead_mb)//per_worker))
        if model == 'independent_processes':
            units = min(units, facts['work_items'])
            for value in hard['workers'].values():
                if value: units = min(units, value)
        else:
            # One MPI task: hint chooses ranks, not independent file workers.
            hint = suggested.get('cpus_per_task', suggested.get('num_processes'))
            if _positive(hint): units = min(units, hint // threads)
            if hard['workers']['exact'] not in (None, 1): continue
        exact_cpus = hard['cpus']['exact']
        if exact_cpus:
            # Native adapters derive process/rank count from allocated CPUs.
            # Never reserve more exact CPUs than the independently RAM-bounded
            # concurrency can actually launch, or override only the CPU field.
            if exact_cpus % threads or exact_cpus // threads > units:
                continue
            units = exact_cpus // threads
        if units < 1: continue
        workers = units if model == 'independent_processes' else 1
        cpus, ram = units*threads, overhead_mb + units*per_worker
        ram = hard['memory_mb']['exact'] or ram
        if cpus > idle or ram > available or (hard['workers']['exact'] and workers != hard['workers']['exact']): continue
        for partition in sorted(node.get('partitions', [])):
            try:
                eligible = profiled_resources(snapshot, facts['tool'], cpus=cpus,
                    memory_mb=ram, nodelist=name, partition=partition, for_queue=for_queue)
            except (ValueError, TypeError, KeyError): eligible = None
            if not eligible: continue
            result['candidates'].append({**eligible, 'cpus': cpus, 'memory_mb': ram,
                'workers': workers, 'mpi_ranks': units if model == 'single_task_mpi' else 0,
                'threads_per_worker': threads, 'work_items': facts['work_items'],
                'memory_basis': memory['basis'], 'input_fingerprint': facts['input_fingerprint']})
    result['candidates'].sort(key=lambda c: (-c['cpus'], c['nodelist'], c['partition']))
    result['status'] = 'ready' if result['candidates'] else 'waiting_resources'
    return result
