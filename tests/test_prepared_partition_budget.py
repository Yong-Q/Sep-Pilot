import copy
import subprocess
import time
from types import SimpleNamespace

import pytest

from agents.resource_review import allocate_reviewed_resources
from tests.test_resource_workload_integration import setup


def native_limits(state='UP', maximum='3-00:00:00'):
    return {'verified': True, 'source': 'scontrol_show_partition', 'observed_at': time.time(),
            'limits': {'compute': {'verified': True, 'state': state, 'max_time': maximum}}}


@pytest.mark.parametrize('kind', ['down', 'short', 'unknown', 'stale', 'future', 'bad_time'])
def test_prepared_budget_requires_fresh_up_partition_with_enough_walltime(tmp_path, monkeypatch, kind):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    value = native_limits()
    if kind == 'down': value['limits']['compute']['state'] = 'DOWN'
    if kind == 'short': value['limits']['compute']['max_time'] = '01:00:00'
    if kind == 'unknown': value['limits'] = {}
    if kind == 'stale': value['observed_at'] -= 1000
    if kind == 'future': value['observed_at'] += 1000
    if kind == 'bad_time': value['limits']['compute']['max_time'] = 'unknown'
    inv['partition_facts'] = value
    r = allocate_reviewed_resources(s, req, receipt)
    assert r['status'] == 'waiting_resources' and not r.get('resource_allocations')


def test_native_partition_facts_allow_actual_default_72h_and_explicit_shorter(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    inv['partition_facts'] = native_limits()
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'ready'
    inv['partition_facts'] = native_limits(maximum='02:00:00')
    req['nodes'][0]['arguments']['walltime'] = '01:30:00'
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'ready'


def test_model_request_cannot_forge_missing_partition_native_evidence(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    inv.pop('partition_facts', None)
    req['partition_facts'] = native_limits()
    req['partition_limits'] = native_limits()['limits']
    assert allocate_reviewed_resources(s, req, receipt)['status'] == 'waiting_resources'


def test_native_partition_reader_bounded_command_and_timeout_fail_closed(monkeypatch):
    from agents.node_inventory import read_partition_limits
    calls = []
    def execute(argv, **kwargs):
        calls.append((argv, kwargs['timeout']))
        return SimpleNamespace(returncode=0, stdout='PartitionName=compute State=UP MaxTime=3-00:00:00\n')
    monkeypatch.setattr('agents.node_inventory.subprocess.run', execute)
    actual = read_partition_limits()
    assert actual['verified'] and actual['limits']['compute']['max_time'] == '3-00:00:00'
    assert calls == [(['scontrol','show','partition','-o'], 8)]
    def timeout(*args, **kwargs): raise subprocess.TimeoutExpired('scontrol', 8)
    monkeypatch.setattr('agents.node_inventory.subprocess.run', timeout)
    assert read_partition_limits()['verified'] is False


def test_plan_memory_hint_not_hard_cap_but_trusted_user_memory_cap_is(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path, monkeypatch)
    inv['partition_facts'] = native_limits()
    req['nodes'][0]['arguments']['memory_mb'] = 1000
    receipt['suggested_resources']['memory_mb'] = 1000
    assert allocate_reviewed_resources(s, req, receipt)['resource_allocations']['batch']['workers'] == 11
    req['resource_constraints'] = [{'source':'user','field':'memory_mb','kind':'max','value':2500,'basis':'actual user bound'}]
    assert allocate_reviewed_resources(s, req, receipt)['resource_allocations']['batch']['workers'] == 2
