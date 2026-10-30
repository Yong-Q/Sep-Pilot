import copy

import pytest

from agents.resource_review import allocate_reviewed_resources
from agents.workload_budget import size_workload_budget
from tests.test_resource_workload_integration import setup
from tests.test_workload_budget import facts, snapshot, estimate


def busy_node(node):
    node['states'] = ['alloc']
    node['cpus']['idle'] = 0
    node['memory']['available_for_scheduling_mib'] = 0


def test_pure_queue_uses_verified_total_capacity_not_idle_and_no_serial_fallback(tmp_path):
    f = facts(tmp_path)
    inventory = snapshot(cpu=16, ram=32000)
    busy_node(inventory['nodes']['free'])
    assert size_workload_budget(f, inventory, memory_evidence=estimate(f))['status'] == 'waiting_resources'
    queued = size_workload_budget(f, inventory, memory_evidence=estimate(f), for_queue=True)
    a = queued['candidates'][0]
    assert queued['status'] == 'ready' and a['cpus'] == a['workers'] == 11
    assert a['scheduling_mode'] == 'queue' and a['memory_mb'] == 11000


@pytest.mark.parametrize('state', ['down', 'planned', 'mix-'])
def test_queue_never_normalizes_ineligible_native_state(tmp_path, state):
    f = facts(tmp_path); inventory = snapshot()
    inventory['nodes']['free']['states'] = [state]
    assert not size_workload_budget(f, inventory, memory_evidence=estimate(f), for_queue=True)['candidates']


def test_prepared_model_node_partition_hints_cannot_pin_busy_capacity(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    busy = copy.deepcopy(inv['nodes']['free']); busy['name'] = 'busy'; busy['partitions'] = ['busy_partition']
    busy_node(busy); inv['nodes']['busy'] = busy
    inv['partition_facts']['limits']['busy_partition'] = {'verified':True,'state':'UP','max_time':'3-00:00:00'}
    req['nodes'][0]['arguments'].update(nodelist='busy', partition='busy_partition')
    allocation = allocate_reviewed_resources(s, req, receipt)['resource_allocations']['batch']
    assert allocation['nodelist'] == 'free' and allocation['scheduling_mode'] == 'immediate'
    req['resource_pins'] = {'batch': {'nodelist':['busy'], 'partition':['busy_partition']}}
    allocation = allocate_reviewed_resources(s, req, receipt)['resource_allocations']['batch']
    assert allocation['nodelist'] == 'busy' and allocation['scheduling_mode'] == 'queue'
    assert allocation['cpus'] == 11


def test_prepared_all_busy_queues_useful_batch_budget_with_time_guard(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    busy_node(inv['nodes']['free'])
    result = allocate_reviewed_resources(s, req, receipt)
    allocation = result['resource_allocations']['batch']
    assert allocation['cpus'] == 11 and allocation['scheduling_mode'] == 'queue'
    assert allocation['memory_mb'] == 11500
    inv['partition_facts']['limits']['compute']['state'] = 'DOWN'
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'waiting_resources'


def test_trusted_pin_cannot_bypass_exclusion_or_unknown_memory(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    req['resource_pins'] = {'batch': {'nodelist':['free']}}
    inv['policy']['excluded_nodes'] = ['free']
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'waiting_resources'
    inv['policy']['excluded_nodes'] = []
    inv['nodes']['free']['memory']['verified'] = False
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'waiting_resources'


def test_invalid_trusted_pin_shape_fails_closed(tmp_path, monkeypatch):
    s, req, receipt, _, _ = setup(tmp_path, monkeypatch)
    req['resource_pins'] = {'batch': {'nodelist':'free'}}
    result = allocate_reviewed_resources(s, req, receipt)
    assert result['status'] == 'waiting_resources' and not result.get('resource_allocations')
