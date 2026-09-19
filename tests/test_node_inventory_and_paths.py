import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.node_inventory import parse_sinfo, parse_pbsnodes, node_inventory, candidates
from agents.parallel_workflow import node_path_manifest, resources_for, conflicts
from agents.registry import ToolDef, ToolRegistry, _build_default_registry, _exec_resource_health
from agents.defns import ORCHESTRATOR
from agents.task_line import TaskLineStore
from agents.agent import Agent
from anthropic.types import TextBlock
from agents.registry import _exec_cdft
from agents.workflow_patch import canonical_arguments, patched_graph, WorkflowContractError
from agents.session import Session
from agents.config import AgentConfig
from agents.recovery import RecoveryGate
from agents.workspace import tool_scope_issues


def test_merge_multiple_partitions_without_double_counting_cpus():
    nodes = parse_sinfo('node03|debug*|idle|0/40/0/40|125814\nnode03|long|idle|0/40/0/40|125814')
    assert nodes['node03']['partitions'] == ['debug', 'long']
    assert nodes['node03']['cpus']['idle'] == 40
    assert nodes['node03']['glibc'] is None


@pytest.mark.parametrize('line', ['node03|debug|idle|10/40/0/40|1024', 'broken row', 'node03|debug|idle|-1/41/0/40|1024'])
def test_malformed_inventory_does_not_claim_available_nodes(line):
    with pytest.raises(ValueError): parse_sinfo(line)


def test_reported_os_does_not_invent_glibc_version():
    nodes = parse_pbsnodes('node03\n    state = free\n    status = state=free,opsys=linux el10,arch=x86_64\n')
    assert nodes['node03']['reported_os'] == 'linux el10'
    assert 'glibc' not in nodes['node03']


@pytest.mark.parametrize('fault', ['stale', 'unknown', 'busy', 'excluded', 'unprobed_libc'])
def test_selection_requires_current_verified_eligible_resources(fault):
    snapshot = {'verified': True, 'stale': False, 'nodes': parse_sinfo('node03|debug|idle|0/40/0/40|125814'),
                'policy': {'preferred_nodes': ['node03'], 'excluded_nodes': [], 'allowed_states': ['idle', 'mix']}}
    minimum = None
    if fault == 'stale': snapshot['stale'] = True
    if fault == 'unknown': snapshot['verified'] = False
    if fault == 'busy': snapshot['nodes']['node03']['cpus']['idle'] = 0
    if fault == 'excluded': snapshot['policy']['excluded_nodes'] = ['node03']
    if fault == 'unprobed_libc': minimum = '2.28'
    assert not candidates(snapshot, min_glibc=minimum)


def test_resource_inventory_readonly_refresh_does_not_write_env(tmp_path, monkeypatch):
    def process(command, **kw):
        stdout = 'node03|debug|idle|0/40/0/40|125814' if command[0] == 'sinfo' else 'node03\n    state = free\n'
        return SimpleNamespace(returncode=0, stdout=stdout, stderr='')
    monkeypatch.setattr('agents.node_inventory.subprocess.run', process)
    snapshot = node_inventory(tmp_path, refresh=True, persist=False)
    assert snapshot['verified'] and not (tmp_path / 'env/node_inventory.json').exists()


def test_paths_are_canonical_and_dynamic_result_dir_follows_receipt(tmp_path):
    args = canonical_arguments({'cif_dir': 'cifs/../cifs', 'job_work_dir': 'runs/u/c/Kr', 'gas': 'Kr'}, tmp_path)
    assert args['cif_dir'] == str(tmp_path / 'cifs')
    node = {'tool': 'run_cdft', 'arguments': {**args, 'action': 'pipeline'}, 'expected_outputs': ['results.csv']}
    job = tmp_path / 'runs/u/c/Kr/data/new-job'
    manifest = node_path_manifest(node, tmp_path, tmp_path / 'runs/u/c', {'work_dir': str(job)}, {'checkpoint_path': '/checkpoint', 'evidence_path': '/evidence'})
    assert manifest['calculation_dir'] == manifest['result_dir'] == str(job)
    assert manifest['expected_outputs'] == [str(job / 'results.csv')]


def test_independent_generic_jobs_do_not_share_accidental_fallback_lock(tmp_path):
    def node(work): return {'tool': 'submit_job', 'arguments': {'command': 'probe', 'work_dir': str(work)}, 'expected_outputs': []}
    root = tmp_path / 'runs/u/c'
    a = resources_for(node(root / 'A'), tmp_path, root)
    b = resources_for(node(root / 'B'), tmp_path, root)
    assert not any(conflicts(x, y) for x in a for y in b)
    same = resources_for(node(root / 'A'), tmp_path, root)
    assert any(conflicts(x, y) for x in a for y in same)


def test_completed_discovery_is_not_accidentally_redelegated():
    preflight = {'step_id': 'read_file', 'tool': 'read_file', 'arguments': {'path': 'env/node_inventory.json'},
                 'agent': 'lead-orchestrator', 'status': 'completed', 'done': True, 'plan_version': 0}
    compute = {'step_id': 'calculate', 'tool': 'submit_job', 'arguments': {'command': 'probe'},
               'agent': 'harness', 'depends_on': [], 'expected_outputs': ['receipt.json']}
    graph, _ = patched_graph([preflight], [{'operation': 'upsert', 'step_id': 'calculate', 'node': compute}])
    assert [n['step_id'] for n in graph] == ['calculate']


def test_schema_discovery_uses_current_registry_without_dispatch_or_auth_grant():
    registry = _build_default_registry()
    registry.register(ToolDef('custom_probe', 'probe', {'type': 'object', 'properties': {'x': {'type': 'integer'}}, 'required': ['x']},
                              lambda p: pytest.fail('discovery dispatched custom computation')))
    result = registry.execute_dict('get_tool_schema', {'tool_name': 'custom_probe'})
    assert result['read_only'] and result['input_schema']['required'] == ['x']
    assert 'permission' in result['note']
    result['input_schema']['required'].append('bad')
    assert registry.get('custom_probe').input_schema['required'] == ['x']


def test_scheduler_node_option_is_explicit_and_cannot_inject_directives():
    registry = _build_default_registry()
    assert not registry.validate_params('submit_job', {'command': 'probe', 'nodelist': 'node03'})
    assert registry.validate_params('submit_job', {'command': 'probe', 'nodelist': 'node03\n#SBATCH --gres=gpu:4'})


def test_blocked_dispatch_is_not_successful_computation():
    gate = RecoveryGate()
    attempt, _ = gate.claim('blocked', 'submit_job', {'command': 'probe'}, 'source')
    gate.outcome('blocked', attempt, {'blocked': True, 'executed': False, 'error': 'invalid node/command'})
    assert gate.snapshot()['blocked']['status'] == 'failed'


@pytest.mark.parametrize('target', ['env/settings.json', 'config.json', '~/.ssh/id_rsa'])
def test_canonical_native_paths_do_not_expose_server_credentials(tmp_path, target):
    assert tool_scope_issues({'path': target}, 'user', 'session', tmp_path)


@pytest.mark.parametrize('missing', ['none', 'result_ref', 'validation', 'terminal'])
def test_main_delivery_uses_worker_certificates_not_redundant_submission(tmp_path, missing):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    node = {'status': 'succeeded', 'contract': {'tool': 'submit_job', 'arguments': {'command': 'probe'}},
            'result_ref': {'call_id': 'actual-worker-call'}, 'recovery_key': 'k', 'artifacts': {'/result.json': {'size': 64}}}
    s._runtime_snapshot = lambda: {'status': 'completed', 'nodes': {'compute': node}}
    s.recovery_gate.state = {'k': {'status': 'completed', 'job_ids': ['42'], 'result_verification': {'artifacts': '/result.json'}}}
    job = {'job_id': '42', 'terminal': True, 'failed': False}
    s._pending_jobs_for_conv = lambda: [job]
    if missing == 'result_ref': node['result_ref'] = {}
    if missing == 'validation': s.recovery_gate.state['k']['result_verification'] = {}
    if missing == 'terminal': job['terminal'] = False
    assert s._has_verified_runtime_computation() == (missing == 'none')


@pytest.mark.parametrize('action', ['inputs', 'collect'])
def test_successful_non_submission_cdft_receipts_are_decoded(tmp_path, monkeypatch, action):
    config = AgentConfig(api_key='test', project_root=tmp_path)
    monkeypatch.setattr('agents.registry.get_config', lambda: config)
    monkeypatch.setattr('agents.workspace.session_dir', lambda kind: tmp_path / kind)
    inner = {'n_inputs': 10, 'input_dir': str(tmp_path / 'inputs')} if action == 'inputs' else {
        'rows': [{'MOF': 'material', 'Kr_henry_mol_L_atm': .25}], 'output_csv': str(tmp_path / 'results.csv')}
    monkeypatch.setattr('subprocess.run', lambda *a, **k: SimpleNamespace(returncode=0, stdout='helper progress\n'+json.dumps(inner), stderr=''))
    result = _exec_cdft({'action': action, 'cif_dir': 'cifs', 'job_work_dir': str(tmp_path / 'job'), 'gas': 'Kr', 'temperature': 298})
    assert not result.get('failed') and not result['submitted']
    if action == 'inputs': assert result['n_inputs'] == 10
    else: assert result['validation_passed'] and result['rows'] == inner['rows'] and result['output_csv'] == inner['output_csv']


def test_ambiguous_tool_mutex_returns_resource_diagnosis_before_any_dispatch(tmp_path, monkeypatch):
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    store.begin_line('c1')
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._current_line_id = 'c1'
    s.current_agent = ORCHESTRATOR
    node = {'agent': 'harness', 'tool': 'submit_job', 'arguments': {'command': 'probe', 'work_dir': 'A'},
            'depends_on': [], 'expected_outputs': ['receipt.json'], 'resource_locks': ['submit_job']}
    with pytest.raises(WorkflowContractError) as error:
        s._propose_workflow_patch(0, 'Compile independently isolated CPU jobs safely', [{'operation': 'upsert', 'step_id': 'A', 'node': node}])
    assert error.value.details['executed'] is False and error.value.details['ambiguous_locks'] == ['submit_job']
    assert store.get_line('c1')['steps'] == []


def test_consumed_read_receipt_uses_shared_read_lease_not_write(tmp_path):
    source = str(tmp_path / 'receipt.json')
    node = {'tool': 'read_file', 'arguments': {'path': source}, 'expected_outputs': [source]}
    resources = resources_for(node, tmp_path, tmp_path / 'runs/u/c')
    assert resources == [{'key': 'path:' + source, 'mode': 'read'}]


def test_cdft_scheduler_parameters_are_forwarded_to_the_real_helper(tmp_path, monkeypatch):
    config = AgentConfig(api_key='test', project_root=tmp_path)
    monkeypatch.setattr('agents.registry.get_config', lambda: config)
    monkeypatch.setattr('agents.workspace.session_dir', lambda kind: tmp_path / kind)
    captured = []
    def process(argv, **kw):
        captured.append(argv[-1])
        return SimpleNamespace(returncode=0, stdout='{"submitted":true,"job_id":"9","work_dir":"'+str(tmp_path/'actual')+'"}', stderr='')
    monkeypatch.setattr('subprocess.run', process)
    _exec_cdft({'action': 'submit', 'input_dir': 'inputs', 'gas': 'Kr', 'nodelist': 'node07',
                'partition': 'bigcpu', 'num_processes': 2, 'walltime': '00:02:00','memory_mb':512})
    assert "'partition': 'bigcpu'" in captured[0] and "'nodelist': 'node07'" in captured[0]
    assert "'num_processes': 2" in captured[0]


@pytest.mark.parametrize('occupied', [False, True])
def test_cdft_default_target_reads_env_profile_instead_of_legacy_node_table(tmp_path, monkeypatch, occupied):
    config = AgentConfig(api_key='test', project_root=tmp_path)
    monkeypatch.setattr('agents.registry.get_config', lambda: config)
    monkeypatch.setattr('agents.workspace.session_dir', lambda kind: tmp_path / kind)
    (tmp_path / 'env').mkdir()
    (tmp_path / 'env/node_inventory.json').write_text('{}')
    snapshot = {'verified': True, 'stale': False, 'nodes': parse_sinfo('candidate|bigcpu|idle|0/40/0/40|1024'),
        'policy': {'preferred_nodes': [], 'excluded_nodes': [], 'allowed_states': ['idle'],
                   'tool_profiles': {'run_cdft': {'default_cpus': 2, 'min_glibc': '2.28', 'partition_preference': ['bigcpu']}}}}
    snapshot['nodes']['candidate']['glibc'] = '2.39'
    snapshot['nodes']['candidate']['memory']={'verified':True,'available_for_scheduling_mib':1024}
    if occupied:
        snapshot['nodes']['candidate'].update(states=['alloc'], cpus={'idle':0,'total':40})
        snapshot['nodes']['candidate']['memory']['available_for_scheduling_mib'] = 0
    monkeypatch.setattr('agents.node_inventory.node_inventory', lambda *a, **kw: snapshot)
    captured = []
    monkeypatch.setattr('subprocess.run', lambda argv, **kw: captured.append(argv[-1]) or SimpleNamespace(returncode=0,
        stdout='{"submitted":true,"job_id":"9","work_dir":"'+str(tmp_path/'actual')+'"}', stderr=''))
    _exec_cdft({'action': 'submit', 'input_dir': 'inputs', 'gas': 'Kr','memory_mb':512})
    assert "'nodelist': 'candidate'" in captured[0] and "'partition': 'bigcpu'" in captured[0]


def test_empty_reasoning_only_model_reply_cannot_mark_task_complete(tmp_path):
    from agents.session import LLMUnavailableError
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    class Stream:
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def __iter__(self): return iter([])
        def get_final_message(self): return SimpleNamespace(content=[SimpleNamespace(type='thinking')], stop_reason='max_tokens')
    s.client = SimpleNamespace(messages=SimpleNamespace(stream=lambda **kw: Stream()))
    with pytest.raises(LLMUnavailableError, match='stop_reason=max_tokens'):
        s._stream_api()


def test_keyword_routing_does_not_seed_unrequested_scientific_goals(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._call_api = lambda agent: SimpleNamespace(content=[TextBlock(type='text', text='请确认输入和目标指标。')])
    s.run_until_complete('优化Xe吸附材料，先协商方法，不提交计算', agent=Agent(name='lead-orchestrator', functions=[]), verbose=False)
    hints = [m['content'] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], str) and '[关键词弱提示' in m['content']]
    assert not hints  # The model interprets the original request; no lexical routing injection.
    user_context = '\n'.join(m['content'] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], str))
    assert 'CO2/N2' not in user_context and '>50' not in user_context and 'Ni-MOF-74' not in user_context
    assert s.goal_contract.gases == ['Xe']
