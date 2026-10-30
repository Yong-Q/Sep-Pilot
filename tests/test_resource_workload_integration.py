import copy
import json
import time
from types import SimpleNamespace

import pytest

from agents.resource_review import allocate_reviewed_resources
from tests.test_workload_budget import facts, snapshot


def setup(tmp_path, monkeypatch):
    observed = facts(tmp_path)
    observed['native_bounds'] = {'max_grid_points': 216000, 'max_gases': 1, 'max_sites_per_gas': 2}
    inventory = snapshot(cpu=32, ram=32000)
    inventory['partition_facts'] = {'verified':True,'source':'scontrol_show_partition','observed_at':time.time(),
        'limits':{'compute':{'verified':True,'state':'UP','max_time':'3-00:00:00'}}}
    private = {'scope_root': str(tmp_path), 'workload_profiles': {'run_cdft': {
        'executable_sha256': 'a'*64, 'basis': 'operator audited source bound; not RSS measurement',
        'per_worker_mb': 1000, 'overhead_mb': 500,
        'max_grid_points': 216000, 'max_gases': 1, 'max_sites_per_gas': 2}}}
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section', lambda name: private if name == 'resource_policy' else {})
    session = SimpleNamespace(config=SimpleNamespace(project_root=tmp_path),
        memory=SimpleNamespace(tool_call_log=[{'tool':'resource_health','call_id':'proof'}]),
        _load_evidence_call=lambda c: {'result': json.dumps({'node_inventory':inventory})})
    node = {'step_id':'batch','tool':'run_cdft','arguments':{'action':'submit','num_processes':1},
            'workload_facts':observed,'executable_sha256':'a'*64}
    request = {'kind':'before_submission','review_phase':'dispatch','nodes':[node]}
    receipt = {'status':'ready','resource_review_id':'review','evidence_call_ids':['proof'],
               'suggested_resources':{'num_processes':1,'memory_mb':32000}}
    return session, request, receipt, private, inventory


def test_prepared_batch_sizes_real_workload_ignoring_plan_cpu_one(tmp_path, monkeypatch):
    s, req, receipt, _, _ = setup(tmp_path,monkeypatch)
    result = allocate_reviewed_resources(s,req,receipt)
    allocation = result['resource_allocations']['batch']
    assert allocation['cpus'] == 11 and allocation['workers'] == 11
    assert allocation['memory_mb'] == 11500 and allocation['budget_final'] is True
    assert allocation['estimate_kind'] == 'operator_conservative_estimate'
    assert 'RSS' not in allocation.get('measured_quantity','')


def test_user_cpu_one_is_preserved_with_native_bound(tmp_path, monkeypatch):
    s, req, receipt, _, _ = setup(tmp_path,monkeypatch)
    req['resource_constraints'] = [{'source':'user','field':'cpus','kind':'exact','value':1,'basis':'actual user turn'}]
    assert allocate_reviewed_resources(s,req,receipt)['resource_allocations']['batch']['cpus'] == 1


@pytest.mark.parametrize('bad', ['exe', 'scope', 'bounds', 'missing_basis', 'unprepared', 'atoms', 'rotations'])
def test_unbound_or_missing_memory_profile_never_finalizes_budget(tmp_path, monkeypatch, bad):
    s, req, receipt, private, _ = setup(tmp_path,monkeypatch)
    p = private['workload_profiles']['run_cdft']
    if bad=='exe':req['nodes'][0]['executable_sha256']='b'*64
    if bad=='scope':private['scope_root']=str(tmp_path/'other')
    if bad=='bounds':req['nodes'][0]['workload_facts']['native_bounds']['max_grid_points']+=1
    if bad=='missing_basis':p.pop('basis')
    if bad=='unprepared':req['nodes'][0].pop('workload_facts')
    if bad in {'atoms','rotations'}:
        key='max_framework_atoms' if bad=='atoms' else 'max_rotations'
        p[key]=12;req['nodes'][0]['workload_facts']['native_bounds'][key]=13
    result = allocate_reviewed_resources(s,req,receipt)
    assert result['status'] == 'waiting_resources' and not result.get('resource_allocations')
    assert result['workload_budget']['status'] in {'needs_memory_evidence','needs_estimate','needs_prepared_inputs'}


def test_planning_pipeline_does_not_persist_a_default_cpu_one(tmp_path, monkeypatch):
    s, req, receipt, _, _ = setup(tmp_path,monkeypatch)
    req['review_phase']='planning';req['nodes'][0]['arguments']['action']='pipeline'
    req['nodes'][0].pop('workload_facts')
    result=allocate_reviewed_resources(s,req,receipt)
    a=result['resource_allocations']['batch']
    assert result['status']=='ready' and not a['budget_final'] and 'cpus' not in a
    req['review_phase']='dispatch'
    assert allocate_reviewed_resources(s,req,receipt)['status']=='waiting_resources'


def test_node_pins_compatibility_exclusions_and_ram_bound_survive_sizing(tmp_path, monkeypatch):
    s, req, receipt, _, inv = setup(tmp_path,monkeypatch)
    req['nodes'][0]['arguments'].update(nodelist='free',partition='compute')
    inv['nodes']['free']['memory']['available_for_scheduling_mib']=4500
    other=copy.deepcopy(inv['nodes']['free']);other['name']='excluded';other['memory']['available_for_scheduling_mib']=32000
    inv['nodes']['excluded']=other;inv['policy']['excluded_nodes']=['excluded']
    a=allocate_reviewed_resources(s,req,receipt)['resource_allocations']['batch']
    assert a['cpus']==4 and a['nodelist']=='free'


def test_operator_cap_and_step_scoped_user_constraint(tmp_path, monkeypatch):
    s, req, receipt, private, _ = setup(tmp_path,monkeypatch)
    private['workload_profiles']['run_cdft']['worker_cap']=5
    req['resource_constraints']=[{'step_id':'unrelated','source':'user','field':'cpus','kind':'max','value':1,'basis':'different branch'}]
    assert allocate_reviewed_resources(s,req,receipt)['resource_allocations']['batch']['cpus']==5
