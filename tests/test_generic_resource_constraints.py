import json
from types import SimpleNamespace

from agents.resource_review import allocate_reviewed_resources
from tests.test_workload_budget import snapshot


def review(cpu=None, constraints=(), pins=None, inventory=None):
    observed = inventory or snapshot(cpu=16, ram=32000)
    session = SimpleNamespace(memory=SimpleNamespace(tool_call_log=[{'tool':'resource_health','call_id':'proof'}]),
        _load_evidence_call=lambda c:{'result':json.dumps({'node_inventory':observed})})
    request={'kind':'before_submission','review_phase':'dispatch','resource_constraints':list(constraints),
        'resource_pins':pins or {},'nodes':[{'step_id':'md','tool':'run_md_optimize','arguments':{}}]}
    suggestion={'memory_mb':16000}
    if cpu is not None:suggestion['num_processes']=cpu
    return allocate_reviewed_resources(session,request,{'status':'ready','resource_review_id':'r',
        'evidence_call_ids':['proof'],'suggested_resources':suggestion})


def hard(field, value, kind='exact', step='md'):
    return {'step_id':step,'source':'user','field':field,'kind':kind,'value':value,'basis':'authenticated user control'}


def test_md_native_default_is_four_mpi_ranks_without_model_cpu_guess():
    assert review()['resource_allocations']['md']['cpus']==4


def test_generic_user_exact_cpu_and_ram_override_model_hints():
    a=review(cpu=1,constraints=[hard('cpus',8),hard('memory_mb',8192)])['resource_allocations']['md']
    assert a['cpus']==8 and a['memory_mb']==8192
    assert review(cpu=8,constraints=[hard('cpus',1)])['resource_allocations']['md']['cpus']==1


def test_generic_max_caps_are_hard_and_step_scoped():
    a=review(cpu=16,constraints=[hard('cpus',2,'max'),hard('memory_mb',4096,'max'),hard('cpus',1,step='other')])['resource_allocations']['md']
    assert a['cpus']==2 and a['memory_mb']==4096


def test_generic_exact_budget_queues_without_shrinking_or_waits_if_capacity_insufficient():
    inv=snapshot(cpu=8,ram=32000);inv['nodes']['free']['cpus']['idle']=0;inv['nodes']['free']['states']=['alloc']
    a=review(constraints=[hard('cpus',8)],inventory=inv)['resource_allocations']['md']
    assert a['cpus']==8 and a['scheduling_mode']=='queue'
    assert review(constraints=[hard('cpus',9)],inventory=inv)['status']=='waiting_resources'
    assert review(constraints=[hard('memory_mb',64000)],inventory=inv)['status']=='waiting_resources'
    assert review(constraints=[hard('memory_mb',1)],inventory=inv)['status']=='waiting_resources'


def test_generic_verified_pins_are_applied_and_conflicting_controls_rejected():
    assert review(pins={'md':{'nodelist':['missing']}})['status']=='waiting_resources'
    assert review(constraints=[hard('cpus',8),hard('cpus',2,'max')])['status']=='waiting_resources'
