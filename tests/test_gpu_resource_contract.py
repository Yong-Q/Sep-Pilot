"""Native GPU demand must survive review, allocation and Slurm rendering."""
import copy
import json
from types import SimpleNamespace
import pytest

from agents.resource_review import allocate_reviewed_resources
from agents.node_inventory import node_inventory
from agents.slurm import render_sbatch_script
from agents.watch_context import set_context, clear_context


def inventory(busy=False):
    def node(name, part, gpu):
        return {'name': name, 'partitions': [part], 'states': ['alloc' if busy else 'idle'],
                'cpus': {'idle': 0 if busy else 16, 'total': 16}, 'memory_mib': 64000,
                'memory': {'verified': True, 'available_for_scheduling_mib': 0 if busy else 64000},
                'gpus': {'verified': True, 'total': gpu, 'allocated': gpu if busy else 0,
                         'available': 0 if busy else gpu}}
    return {'verified': True, 'stale': False, 'observed_at': 1,
            'policy': {'preferred_nodes': [], 'excluded_nodes': [], 'allowed_states': ['idle', 'mix'],
                       'tool_profiles': {}},
            'nodes': {'cpu': node('cpu', 'bigcpu', 0), 'accelerator': node('accelerator', 'cuda', 2)}}


def review(snapshot, **suggestions):
    session = SimpleNamespace(memory=SimpleNamespace(tool_call_log=[{'tool': 'resource_health', 'call_id': 'proof'}]),
        _load_evidence_call=lambda c: {'result': json.dumps({'node_inventory': snapshot})})
    return allocate_reviewed_resources(session, {'nodes': [{'step_id': 's', 'tool': 'run_string_tst',
        'arguments': {'gas': 'CO2', 'directions': [1], 'running_steps': 20, 'n_string': 11}}]},
        {'status': 'ready', 'resource_review_id': 'review', 'evidence_call_ids': ['proof'],
         'suggested_resources': {'memory_mb': 4096, 'num_processes': 1, **suggestions}})


@pytest.mark.parametrize('busy', [False, True])
def test_cpu_reviewer_hint_cannot_override_native_gpu_demand(busy):
    result = review(inventory(busy), partition='bigcpu', nodelist='cpu')
    assert result['status'] == 'ready'
    allocation = result['resource_allocations']['s']
    assert (allocation['partition'], allocation['nodelist']) == ('cuda', 'accelerator')
    assert allocation['gpus'] == 1
    set_context('u', 'c', tool_name='run_string_tst', resource_allocation=allocation)
    try:
        script = render_sbatch_script('test', 'true', '/unused', partition='gpu', gres='gpu:1')
    finally:
        clear_context()
    assert '#SBATCH -p cuda' in script and '#SBATCH --nodelist=accelerator' in script
    assert '#SBATCH --gres=gpu:1' in script and '#SBATCH --cpus-per-task=1' in script
    assert '#SBATCH --mem=4096M' in script


@pytest.mark.parametrize('fault', ['missing', 'unverified', 'zero', 'operator_downgrade'])
def test_gpu_capacity_cannot_be_invented_from_partition_name(fault):
    snapshot = inventory()
    gpu = snapshot['nodes']['accelerator']
    gpu['partitions'] = ['gpu']
    if fault == 'missing': gpu.pop('gpus')
    if fault == 'unverified': gpu['gpus']['verified'] = False
    if fault == 'zero': gpu['gpus']['total'] = gpu['gpus']['available'] = 0
    if fault == 'operator_downgrade':
        snapshot['policy']['tool_profiles']['run_string_tst'] = {'gpus': 0, 'partition_preference': ['bigcpu']}
        gpu['gpus']['total'] = gpu['gpus']['available'] = 0
    result = review(snapshot)
    assert result['status'] == 'waiting_resources'
    assert not result.get('resource_allocations')


def test_old_cpu_only_review_is_rejected_before_rendering_gpu_script():
    set_context('u', 'c', tool_name='run_string_tst', resource_allocation={
        'resource_review_id': 'old', 'memory_mb': 4096, 'cpus': 1, 'partition': 'bigcpu', 'nodelist': 'cpu'})
    try:
        with pytest.raises(ValueError, match='GPU|gpu'):
            render_sbatch_script('test', 'true', '/unused', partition='gpu', gres='gpu:1')
    finally:
        clear_context()


def test_scheduler_gpu_facts_come_from_scontrol_not_node_name(tmp_path, monkeypatch):
    def run(argv, **kwargs):
        stdout = ''
        if argv[0] == 'sinfo': stdout = 'accelerator|cuda|mix|4/12/0/16|64000\ncpu|gpu|idle|0/16/0/16|64000'
        if argv[0] == 'scontrol':
            stdout = ('NodeName=accelerator RealMemory=64000 AllocMem=1024 FreeMem=60000 '
                      'CfgTRES=cpu=16,mem=64000M,gres/gpu=2 AllocTRES=cpu=4,gres/gpu=1\n'
                      'NodeName=cpu RealMemory=64000 AllocMem=0 FreeMem=60000 CfgTRES=cpu=16 AllocTRES=')
        return SimpleNamespace(returncode=0, stdout=stdout, stderr='')
    monkeypatch.setattr('agents.node_inventory.subprocess.run', run)
    snapshot = node_inventory(tmp_path, refresh=True)
    assert snapshot['nodes']['accelerator']['gpus'] == {'verified': True, 'total': 2, 'allocated': 1, 'available': 1}
    assert snapshot['nodes']['cpu']['gpus']['total'] == 0


def test_resource_executor_fingerprint_changes_with_hardware_policy(tmp_path):
    from agents.installations import executor_fingerprint
    p = tmp_path/'agents/resource_policy.py';p.parent.mkdir();p.write_text('gpu = 1')
    before = executor_fingerprint('run_string_tst', tmp_path)
    p.write_text('gpu = 2')
    assert executor_fingerprint('run_string_tst', tmp_path) != before


def test_tool_exclusion_also_applies_before_review_selection():
    snapshot = inventory()
    only_gpu = snapshot['nodes'].pop('accelerator')
    only_gpu['name'] = 'gpu3'
    snapshot['nodes']['gpu3'] = only_gpu
    assert review(snapshot)['status'] == 'waiting_resources'


def test_operator_cannot_remove_native_gpu_with_valid_cpu_preference():
    snapshot = inventory()
    snapshot['policy']['tool_profiles']['run_string_tst'] = {'gpus': 0, 'partition_preference': ['bigcpu']}
    assert review(snapshot)['resource_allocations']['s']['nodelist'] == 'accelerator'


@pytest.mark.parametrize('field,value', [('partition', 'bigcpu'), ('nodelist', 'cpu'), ('gres', 'gpu:2'), ('gpus', 0)])
def test_gpu_receipt_evidence_must_describe_exact_submission(field, value):
    allocation = review(inventory())['resource_allocations']['s']
    allocation[field] = value
    set_context('u', 'c', tool_name='run_string_tst', resource_allocation=allocation)
    try:
        with pytest.raises(ValueError, match='GPU|gpu'):
            render_sbatch_script('test', 'true', '/unused', partition='gpu', gres='gpu:1')
    finally:
        clear_context()


@pytest.mark.parametrize('method,expected_node', [('pacman', 'accelerator'), ('pacmof', 'cpu')])
def test_existing_charge_methods_keep_their_native_hardware_boundary(method, expected_node):
    snapshot = inventory()
    session = SimpleNamespace(memory=SimpleNamespace(tool_call_log=[{'tool': 'resource_health', 'call_id': 'proof'}]),
        _load_evidence_call=lambda c: {'result': json.dumps({'node_inventory': snapshot})})
    result = allocate_reviewed_resources(session, {'nodes': [{'step_id': 'charge', 'tool': 'run_pacman_charge',
        'arguments': {'method': method}}]}, {'status': 'ready', 'resource_review_id': 'r', 'evidence_call_ids': ['proof'],
        'suggested_resources': {'memory_mb': 4096, 'partition': 'bigcpu', 'nodelist': 'cpu'}})
    assert result['status'] == 'ready'
    allocation = result['resource_allocations']['charge']
    assert allocation['nodelist'] == expected_node
    if method == 'pacman':
        set_context('u', 'c', tool_name='run_pacman_charge', resource_allocation=allocation)
        try:
            script = render_sbatch_script('test', 'true', '/unused', partition='gpu', gres='gpu:1')
        finally:
            clear_context()
        assert '#SBATCH -p cuda' in script
