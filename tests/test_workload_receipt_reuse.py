import time
from types import SimpleNamespace

import pytest

from agents.parallel_workflow import persisted_resource_review
from agents.workload_budget import _digest
from agents.workload_context import enrich_resource_request
from tests.test_workload_context import case


@pytest.mark.parametrize('change', [None, 'input', 'binary', 'profile', 'policy', 'goal', 'intent', 'age', 'future', 'legacy'])
def test_cached_budget_revalidates_native_proof_without_model_call(tmp_path, monkeypatch, change):
    root, directory, executable, request = case(tmp_path)
    observed = enrich_resource_request(request, owned_root=root, cdft_executable=executable)['nodes'][0]
    profile = {'basis':'bounded estimate','per_worker_mb':1024,'overhead_mb':512,
        'executable_sha256':observed['executable_sha256'], **observed['workload_facts']['native_bounds']}
    policy = {'excluded_nodes':['node06']}
    goal = SimpleNamespace(version=1, original_goal='计算CO2吸附', directives=['计算CO2吸附'])
    monkeypatch.setattr('agents.node_inventory.load_policy', lambda root:policy)
    monkeypatch.setattr('agents.installations.cdft_installation_files',
        lambda *a, **k: {'cdft_executable':(executable,'executable')})
    monkeypatch.setattr('bimem_agent.paths.get_runtime_section',
        lambda name: {'scope_root':str(tmp_path),'workload_profiles':{'run_cdft':profile}})
    args = observed['arguments']
    from agents.resource_intent import resource_intent
    state = {'contract':{'tool':'run_cdft','arguments':args},
        'effective_arguments':{**args,'resource_review_id':'cached'},
        'resource_allocation':{'resource_review_id':'cached','evidence_call_ids':['native-proof'],
            'budget_final':True,'observed_at':time.time(),
            'input_fingerprint':observed['workload_facts']['input_fingerprint'],
            'executable_sha256':observed['executable_sha256'],
            'memory_profile_sha256':_digest(profile), 'goal_version':1,
            'operator_policy_sha256':_digest(policy),
            'resource_intent_sha256':_digest(resource_intent(goal,[{'tool':'run_cdft','step_id':'one'}]))}}
    if change=='input':
        path=directory/'one.dat';path.write_text(path.read_text().replace('298 0.95','300 0.95'))
    elif change=='binary':executable.write_bytes(b'changed configured binary')
    elif change=='profile':profile['per_worker_mb']=2048
    elif change=='policy':policy['excluded_nodes'].append('node25')
    elif change=='goal':goal.version=2
    elif change=='intent':goal.directives.append('cDFT只用1核')
    elif change=='age':state['resource_allocation']['observed_at']-=121
    elif change=='future':state['resource_allocation']['observed_at']+=5
    elif change=='legacy':state['resource_allocation'].pop('budget_final')
    receipt=persisted_resource_review(state,'one',project_root=tmp_path,owned_root=root,goal_contract=goal)
    assert (receipt is not None) is (change is None)
    if receipt:
        assert receipt['reused'] and receipt['resource_review_id']=='cached'


@pytest.mark.parametrize('changed', [False,True])
def test_mpi_budget_reuse_preserves_current_user_controls(tmp_path,monkeypatch,changed):
    from agents.resource_intent import resource_intent
    goal=SimpleNamespace(version=1,original_goal='MD使用4核',directives=['MD使用4核'])
    policy={'excluded_nodes':['node06']}
    monkeypatch.setattr('agents.node_inventory.load_policy',lambda root:policy)
    contract={'tool':'run_md_optimize','arguments':{'cif_path':'owned.cif'}}
    state={'contract':contract,'effective_arguments':{**contract['arguments'],'resource_review_id':'md-review'},
        'resource_allocation':{'budget_final':True,'cpus':4,'resource_review_id':'md-review',
            'evidence_call_ids':['native'],'goal_version':1,'operator_policy_sha256':_digest(policy),
            'resource_intent_sha256':_digest(resource_intent(goal,[{**contract,'step_id':'md'}]))}}
    if changed:goal.directives.append('MD改为1核')
    assert bool(persisted_resource_review(state,'md',project_root=tmp_path,goal_contract=goal)) is not changed


def test_legacy_generic_budget_cannot_skip_current_user_cpu_requirement(tmp_path):
    goal=SimpleNamespace(version=2,original_goal='MD使用4个MPI进程',directives=['MD使用4个MPI进程'])
    state={'contract':{'tool':'run_md_optimize','arguments':{'cif_path':'owned.cif'}},
        'effective_arguments':{'cif_path':'owned.cif','resource_review_id':'legacy-review'},
        'resource_allocation':{'resource_review_id':'legacy-review','cpus':1,'memory_mb':4096,
            'evidence_call_ids':['old-proof']}}
    assert persisted_resource_review(state,'md',project_root=tmp_path,goal_contract=goal) is None
