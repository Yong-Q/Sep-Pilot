"""Prefer usable verified capacity over soft hints pinning a fully busy node."""
import copy
import json
from types import SimpleNamespace
import pytest
from agents.resource_review import allocate_reviewed_resources


def snapshot():
    def node(name, partition, idle, total=40):
        return {'name': name, 'partitions': [partition], 'states': ['mix' if idle else 'alloc'],
            'cpus': {'idle': idle, 'total': total}, 'memory_mib': 64000,
            'memory': {'verified': True, 'available_for_scheduling_mib': 16000 if idle else 0},
            'glibc': '2.39', 'gpus': {'verified': True, 'total': 0, 'available': 0, 'allocated': 0}}
    return {'verified': True, 'stale': False, 'observed_at': 123,
        'policy': {'preferred_nodes': [], 'excluded_nodes': [], 'allowed_states': ['idle', 'mix'],
                   'tool_profiles': {}},
        'nodes': {'busy': node('busy', 'compute', 0), 'free': node('free', 'debug', 29, 96)}}


def allocate(snap, tool='run_md_optimize', cpus=4, arguments=None, hint=None):
    session = SimpleNamespace(memory=SimpleNamespace(tool_call_log=[{'tool': 'resource_health', 'call_id': 'proof'}]),
        _load_evidence_call=lambda c: {'result': json.dumps({'node_inventory': snap})})
    return allocate_reviewed_resources(session, {'nodes': [{'step_id': 'n', 'tool': tool,
        'arguments': arguments or {}}]}, {'status': 'ready', 'resource_review_id': 'r',
        'evidence_call_ids': ['proof'], 'suggested_resources': {'num_processes': cpus, 'memory_mb': 2000,
            'partition': 'compute', 'nodelist': 'busy', **(hint or {})}})


@pytest.mark.parametrize('tool,cpus', [('run_pore_analysis', 1), ('run_md_optimize', 4)])
def test_idle_compatible_node_wins_over_soft_busy_node_partition_hint(tool, cpus):
    result = allocate(snapshot(), tool, cpus)
    assert result['status'] == 'ready'
    a = result['resource_allocations']['n']
    assert (a['nodelist'], a['partition'], a['cpus']) == ('free', 'debug', cpus)
    assert a['scheduling_mode'] == 'immediate'


@pytest.mark.parametrize('constraint', [{'partition': 'compute'}, {'nodelist': 'busy'}])
def test_explicit_user_constraint_keeps_compatible_busy_queue(constraint):
    a = allocate(snapshot(), arguments=constraint)['resource_allocations']['n']
    assert a['nodelist'] == 'busy' and a['scheduling_mode'] == 'queue'


@pytest.mark.parametrize('constraint', [{'partition': 'compute'}, {'nodelist': 'busy'},
                                      {'allowed_partitions': ['compute']}, {'allowed_nodes': ['busy']}])
def test_operator_hard_constraint_is_not_bypassed_for_idle_cpu(constraint):
    s = snapshot();s['policy']['tool_profiles']['run_md_optimize'] = constraint
    a = allocate(s, hint={'partition': 'debug', 'nodelist': 'free'})['resource_allocations']['n']
    assert a['nodelist'] == 'busy' and a['scheduling_mode'] == 'queue'


def test_operator_partition_preference_is_soft_when_only_other_partition_is_idle():
    s = snapshot();s['policy']['tool_profiles']['run_md_optimize'] = {'partition_preference': ['compute']}
    a = allocate(s)['resource_allocations']['n']
    assert a['nodelist'] == 'free'


@pytest.mark.parametrize('fault', ['ram', 'unknown_ram', 'glibc', 'excluded', 'cpus'])
def test_idle_node_must_still_meet_verified_hard_requirements(fault):
    s = snapshot();n = s['nodes']['free']
    if fault == 'ram': n['memory']['available_for_scheduling_mib'] = 100
    if fault == 'unknown_ram': n['memory']['verified'] = False
    if fault == 'glibc':
        n['glibc'] = None;s['policy']['tool_profiles']['run_md_optimize'] = {'min_glibc': '2.28'}
    if fault == 'excluded': s['policy']['excluded_nodes'] = ['free']
    if fault == 'cpus': n['cpus']['idle'] = n['cpus']['total'] = 1
    a = allocate(s)['resource_allocations']['n']
    assert a['nodelist'] == 'busy' and a['scheduling_mode'] == 'queue'


def test_no_immediate_capacity_uses_total_capacity_without_claiming_no_resources():
    s = snapshot();s['nodes'].pop('free')
    a = allocate(s)['resource_allocations']['n']
    assert a['nodelist'] == 'busy' and a['scheduling_mode'] == 'queue'


@pytest.mark.parametrize('tool,cpus,expected', [('run_pore_analysis', 1, 'node04'), ('run_md_optimize', 4, 'node05')])
def test_two_and_twenty_nine_idle_cores_choose_actual_matching_capacity(tool, cpus, expected):
    s = snapshot();template = s['nodes'].pop('free')
    for name, idle in [('node04', 2), ('node05', 29)]:
        n = copy.deepcopy(template);n.update(name=name, cpus={'idle': idle, 'total': 40})
        s['nodes'][name] = n
    a = allocate(s, tool, cpus)['resource_allocations']['n']
    assert a['nodelist'] == expected and a['scheduling_mode'] == 'immediate'


def test_idle_cpu_does_not_override_busy_gpu_native_requirement():
    s = snapshot();busy = s['nodes']['busy'];busy['partitions'] = ['accelerator']
    busy['gpus'] = {'verified': True, 'total': 1, 'allocated': 1, 'available': 0}
    a = allocate(s, 'run_string_tst', 1)['resource_allocations']['n']
    assert a['nodelist'] == 'busy' and a['partition'] == 'accelerator'
    assert a['gpus'] == 1 and a['scheduling_mode'] == 'queue'


def test_missing_finite_ram_budget_does_not_use_idle_node_as_implicit_permission():
    assert allocate(snapshot(), hint={'memory_mb': None})['status'] == 'waiting_resources'
