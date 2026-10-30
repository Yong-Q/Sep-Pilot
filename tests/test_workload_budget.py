import copy
import hashlib

import pytest

from agents.workload_budget import collect_workload_facts, size_workload_budget


def facts(tmp_path, count=11):
    directory = tmp_path / 'inputs'; directory.mkdir(exist_ok=True)
    for i in range(count):
        (directory / f'{i}.dat').write_text(f'fixture {i}')
    return collect_workload_facts('run_cdft', {'action': 'submit', 'input_dir': str(directory)}, owned_root=tmp_path)


def snapshot(cpu=32, ram=32000):
    return {'verified': True, 'stale': False, 'nodes': {'free': {
        'name': 'free', 'partitions': ['compute'], 'states': ['idle'],
        'cpus': {'idle': cpu, 'total': cpu}, 'memory_mib': ram,
        'memory': {'verified': True, 'available_for_scheduling_mib': ram},
        'gpus': {'verified': True, 'total': 0, 'available': 0}}},
        'policy': {'preferred_nodes': [], 'excluded_nodes': [], 'allowed_states': ['idle', 'mix'], 'tool_profiles': {}}}


def estimate(f, mb=1000, source='operator'):
    return {'source': source, 'basis': 'verified bound fixture', 'per_worker_mb': mb,
            'unit': 'per_mpi_rank' if f['execution_model'] == 'single_task_mpi' else 'per_worker',
            'workload_fingerprint': f['input_fingerprint']}


@pytest.mark.parametrize('count', [3, 11])
def test_batch_worker_budget_tracks_real_work_items_not_hint_one(tmp_path, count):
    f = facts(tmp_path, count)
    r = size_workload_budget(f, snapshot(), suggested_resources={'num_processes': 1, 'memory_mb': 32000}, memory_evidence=estimate(f))
    assert f['work_items'] == count and f['threads_per_worker'] == 1
    assert r['status'] == 'ready' and r['candidates'][0]['workers'] == count
    assert r['candidates'][0]['cpus'] == count


def test_each_candidate_scaled_by_available_cpu_ram_and_overhead(tmp_path):
    f = facts(tmp_path)
    s = snapshot(cpu=8, ram=8000)
    second = copy.deepcopy(s['nodes']['free']); second['name'] = 'small'
    second['memory']['available_for_scheduling_mib'] = 4000
    s['nodes']['small'] = second
    r = size_workload_budget(f, s, memory_evidence=estimate(f), overhead_mb=2000)
    assert {c['nodelist']: c['workers'] for c in r['candidates']} == {'free': 6, 'small': 2}
    assert all(c['memory_mb'] == c['workers'] * 1000 + 2000 for c in r['candidates'])


@pytest.mark.parametrize('source,expected', [('user', 1), ('operator', 1), ('hint', 11), ('default', 11)])
def test_only_proven_hard_constraints_cap_cpu(tmp_path, source, expected):
    f = facts(tmp_path)
    r = size_workload_budget(f, snapshot(), memory_evidence=estimate(f), constraints=[
        {'source': source, 'field': 'cpus', 'kind': 'max', 'value': 1, 'basis': 'explicit origin'}])
    assert r['candidates'][0]['cpus'] == expected


def test_unknown_worker_ram_never_fabricates_single_cpu_allocation(tmp_path):
    f = facts(tmp_path)
    assert size_workload_budget(f, snapshot())['status'] == 'needs_memory_evidence'
    r = size_workload_budget(f, snapshot(), suggested_resources={'memory_mb': 32000})
    assert r['status'] == 'needs_estimate' and not r['candidates']
    bad = estimate(f); bad['source'] = 'hint'
    assert size_workload_budget(f, snapshot(), memory_evidence=bad)['status'] == 'needs_memory_evidence'


def test_manifest_exact_match_and_input_change_invalidate_memory_calibration(tmp_path):
    f = facts(tmp_path, 3)
    items = [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted((tmp_path/'inputs').iterdir())]
    args = {'action': 'submit', 'input_dir': str(tmp_path/'inputs')}
    good = collect_workload_facts('run_cdft', args, owned_root=tmp_path, manifest={'schema_version': 1, 'items': items})
    assert good['status'] == 'ready'
    assert collect_workload_facts('run_cdft', args, owned_root=tmp_path, manifest={'schema_version': 1, 'items': items[:-1]})['status'] == 'invalid_workload'
    (tmp_path/'inputs'/'0.dat').write_text('changed')
    changed = collect_workload_facts('run_cdft', args, owned_root=tmp_path)
    assert size_workload_budget(changed, snapshot(), memory_evidence=estimate(f))['status'] == 'needs_memory_evidence'


def test_unsafe_inputs_and_bounded_manifest_are_not_silently_counted(tmp_path):
    outside = tmp_path / 'outside.dat'; outside.write_text('fixture')
    d = tmp_path/'inputs';d.mkdir();(d/'link.dat').symlink_to(outside)
    args = {'action': 'submit', 'input_dir': str(d)}
    assert collect_workload_facts('run_cdft', args, owned_root=tmp_path)['status'] == 'invalid_workload'
    assert collect_workload_facts('run_cdft', {**args, 'input_dir': str(tmp_path/'..')}, owned_root=tmp_path)['status'] == 'invalid_workload'
    assert collect_workload_facts('run_cdft', args, owned_root=tmp_path, manifest={'schema_version': 1, 'items': [{}]*4097})['status'] == 'invalid_workload'


def test_md_single_task_mpi_not_capped_to_one_rank_by_file_count(tmp_path):
    p = tmp_path/'input.cif';p.write_text('fixture')
    f = collect_workload_facts('run_md_optimize', {'cif_path': str(p)}, owned_root=tmp_path)
    assert f['work_items'] == 1 and f['execution_model'] == 'single_task_mpi'
    r = size_workload_budget(f, snapshot(cpu=8), suggested_resources={'num_processes': 4}, memory_evidence=estimate(f))
    assert r['candidates'][0]['workers'] == 1 and r['candidates'][0]['mpi_ranks'] == 4
    assert r['candidates'][0]['cpus'] == 4
    bad = estimate(f); bad['unit'] = 'per_worker'
    assert size_workload_budget(f, snapshot(), memory_evidence=bad)['status'] == 'needs_memory_evidence'


def test_unprepared_or_unsupported_tool_never_invents_workload(tmp_path):
    assert collect_workload_facts('run_cdft', {'action': 'pipeline'}, owned_root=tmp_path)['status'] == 'needs_prepared_inputs'
    assert collect_workload_facts('run_string_tst', {}, owned_root=tmp_path)['status'] == 'unsupported_workload'


def test_aggregate_byte_bound_stops_reading_later_inputs(tmp_path, monkeypatch):
    import agents.workload_budget as budget
    directory = tmp_path/'inputs';directory.mkdir()
    for i in range(3): (directory/f'{i}.dat').write_text('1234567')
    monkeypatch.setattr(budget, 'MAX_TOTAL_BYTES', 9)
    read = []; original = budget._fact
    def measured(fd, name):
        read.append(name)
        return original(fd, name)
    monkeypatch.setattr(budget, '_fact', measured)
    result = collect_workload_facts('run_cdft', {'action':'submit','input_dir':str(directory)}, owned_root=tmp_path)
    assert result['status'] == 'invalid_workload'
    assert read == ['0.dat', '1.dat']
