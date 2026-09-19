"""No real scheduler/model calls: tool identity and late-receipt regressions."""
import json
from types import SimpleNamespace

import pytest
from anthropic.types import TextBlock, ToolUseBlock

from agents.agent import Agent
from agents.config import AgentConfig
from agents.recovery import RecoveryGate
from agents.registry import ToolDef, ToolRegistry, get_registry
from agents.session import Session
from agents.workflow_patch import patched_graph


def test_agent_registry_has_unique_canonical_identities():
    from agents.defns import AGENT_ALIASES, AGENT_REGISTRY, _validate_agent_registry
    assert len({agent.name for agent in AGENT_REGISTRY.values()}) == len(AGENT_REGISTRY)
    assert set(AGENT_ALIASES.values()) <= set(AGENT_REGISTRY)
    _validate_agent_registry()


def test_same_tool_different_arguments_are_not_batch_duplicates(tmp_path):
    calls = []
    def read_file(**params):
        calls.append(params['path'])
        return {'content': params['path']}
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    agent = Agent(name='lead-orchestrator', functions=[read_file])
    responses = iter([
        SimpleNamespace(content=[
            ToolUseBlock(type='tool_use', id='a', name='read_file', input={'path': 'first.cif'}),
            ToolUseBlock(type='tool_use', id='b', name='read_file', input={'path': 'second.cif'}),
            ToolUseBlock(type='tool_use', id='c', name='read_file', input={'path': 'first.cif'}),
        ]),
        SimpleNamespace(content=[TextBlock(type='text', text='Done.')]),
    ])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(agent)
    assert calls == ['first.cif', 'second.cif']
    results = next(m['content'] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], list))
    assert sorted(r['tool_use_id'] for r in results) == ['a', 'b', 'c']
    duplicate = json.loads(next(r['content'] for r in results if r['tool_use_id'] == 'c'))
    assert duplicate['deduplicated'] and duplicate['original_tool_use_id'] == 'a'


def test_conflicting_tool_names_cannot_silently_replace_executor():
    registry = ToolRegistry()
    tool = ToolDef('probe', 'probe', {'type': 'object', 'properties': {}}, lambda p: p)
    registry.register(tool)
    registry.register(tool)  # idempotent same registration
    with pytest.raises(ValueError, match='already registered'):
        registry.register(ToolDef('probe', 'different', tool.input_schema, lambda p: {'wrong': True}))
    assert registry.get('probe') is tool


def test_agent_function_names_cannot_silently_replace_executor():
    def probe():
        pass
    def another():
        pass
    another.__name__ = 'probe'
    agent = Agent(functions=[probe, another])
    with pytest.raises(ValueError, match='ambiguous'):
        agent.function_map()


def test_handoff_schema_names_are_unique():
    schemas = get_registry().claude_tools(['handoff_to_analyst', 'handoff_to_analyst'])
    assert [s['name'] for s in schemas] == ['handoff_to_analyst']


@pytest.mark.parametrize('params', [{}, {'task': ' '}, {'task': 'calculate', 'gas': 'Kr'}, {'task': 'calculate', 'context': {}}])
def test_handoff_arguments_are_not_silently_discarded(params):
    assert get_registry().validate_handoff_params(params)


def test_extra_handoff_is_explicitly_deferred_with_full_arguments(tmp_path):
    dispatched = []
    def handoff_to_analyst(**params):
        dispatched.append(('analyst', params))
        return {'ok': True}
    def handoff_to_adsorption(**params):
        pytest.fail('second active delegation in the same batch must not dispatch')
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    agent = Agent(name='lead-orchestrator', functions=[handoff_to_analyst, handoff_to_adsorption])
    deferred_args = {'task': 'Kr isotherm after validation', 'context': '{"depends_on":["validate"]}'}
    responses = iter([
        SimpleNamespace(content=[
            ToolUseBlock(type='tool_use', id='first', name='handoff_to_analyst', input={'task': 'validate'}),
            ToolUseBlock(type='tool_use', id='second', name='handoff_to_adsorption', input=deferred_args),
        ]),
        SimpleNamespace(content=[TextBlock(type='text', text='Done.')]),
    ])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(agent)
    assert dispatched == [('analyst', {'task': 'validate'})]
    result = next(r for m in s.messages if m['role'] == 'user' and isinstance(m['content'], list)
                  for r in m['content'] if r.get('tool_use_id') == 'second')
    receipt = json.loads(result['content'])
    assert receipt['status'] == 'deferred' and not receipt['executed']
    assert receipt['arguments'] == deferred_args
    assert receipt['blocking_tool_use_id'] == 'first' and receipt['requires_reissue']


@pytest.mark.parametrize('changes,match', [
    ([{'operation': 'remove', 'step_id': 'missing'}], 'missing'),
    ([{'operation': 'remove', 'step_id': 'a'}, {'operation': 'upsert', 'step_id': 'a', 'node': {}}], 'duplicate'),
    ([{'operation': 'upsert', 'step_id': 'a', 'node': {'step_id': 'b'}}], 'disagrees'),
    ([{'operation': 'upsert', 'step_id': 1, 'node': {}}], 'non-empty string'),
    ([{'operation': 'upsert', 'step_id': 'a', 'node': {'depends_on': 'b'}}], 'list'),
])
def test_ambiguous_workflow_patch_is_rejected(changes, match):
    with pytest.raises(ValueError, match=match):
        patched_graph([{'step_id': 'a', 'depends_on': []}, {'step_id': 'b', 'depends_on': []}], changes)


def test_existing_duplicate_workflow_ids_are_not_last_writer_wins():
    with pytest.raises(ValueError, match='duplicate'):
        patched_graph([{'step_id': 'a'}, {'step_id': 'a'}], [])


def test_submission_params_are_a_snapshot_not_mutable_reference():
    gate = RecoveryGate()
    params = {'gases': ['Kr'], 'nested': {'temperature': 298}}
    gate.claim('task', 'run_gcmc_batch', params, 'fp')
    params['gases'].append('Xe')
    params['nested']['temperature'] = 300
    assert gate.snapshot()['task']['params'] == {'gases': ['Kr'], 'nested': {'temperature': 298}}


def test_early_job_receipt_survives_parent_result_without_job_id():
    gate = RecoveryGate()
    aid, _ = gate.claim('task', 'run_henry', {}, 'fp')
    gate.register_job('task', aid, '42')
    gate.outcome('task', aid, {'ok': True})
    assert gate.snapshot()['task']['status'] == 'submitted'
    assert gate.snapshot()['task']['job_ids'] == ['42']


@pytest.mark.parametrize('failed', [True, False])
def test_terminal_watcher_state_is_not_erased_by_late_submit_result(failed):
    gate = RecoveryGate()
    aid, _ = gate.claim('task', 'run_henry', {}, 'fp')
    gate.register_job('task', aid, '42')
    gate.sync_jobs([{'job_id': '42', 'terminal': True, 'failed': failed,
                     'state': 'FAILED' if failed else 'COMPLETED'}])
    expected = 'failed' if failed else 'completed_unverified'
    gate.outcome('task', aid, {'submitted': True, 'job_id': '42'})
    gate.register_job('task', aid, '42')
    assert gate.snapshot()['task']['status'] == expected
    if failed:
        assert gate.snapshot()['task']['error']


@pytest.mark.parametrize('tool,params', [
    ('read_file', {'path': 'x', 'made_up_offset': 2}),
    ('run_xtb_optimize', {'input_path': 'x.xyz', 'output_dir': 'out', 'gfn': 7}),
    ('run_md_optimize', {'cif_path': 'x.cif', 'mode': 'batch'}),
    ('run_md_optimize', {'cif_path': 'x.cif', 'mode': 'md'}),
    ('run_md_optimize', {'cif_path': 'x.cif', 'mode': 'single', 'temperature': 298}),
    ('run_md_optimize', {'cif_path': 'x.cif', 'submit_pbs': True}),
    ('run_string_tst', {'cif_dir': 'cifs', 'output_dir': 'out', 'gas': 'Kr', 'gases': ['Xe']}),
    ('run_cdft', {'cif_path': 'x.cif', 'cif_dir': 'cifs', 'gas': 'Kr', 'temperature': 298}),
    ('run_cdft', {'cif_dir': 'cifs', 'gas': 'Kr', 'temperature': 298, 'bulk_densities': [float('nan')]}),
    ('generate_structure', {'material_type': 'HOF', 'n_structures': 1, 'output_dir': 'out', 'topologies': ['srs', 'cds']}),
    ('generate_structure', {'material_type': 'HOF', 'n_structures': 3, 'output_dir': 'out', 'topologies': ['srs', 'cds'], 'structures_per_topology': 2}),
])
def test_unapplied_ambiguous_or_nonfinite_arguments_are_rejected(tool, params):
    assert get_registry().validate_params(tool, params)


@pytest.mark.parametrize('action', ['parse', 'setup', 'setup-batch'])
def test_vasp_non_submission_actions_never_submit(action, monkeypatch):
    monkeypatch.setattr('agents.registry.slurm.submit_and_return', lambda **kw: pytest.fail('must not submit'))
    result = get_registry().execute_dict('run_vasp', {'action': action, 'work_dir': 'vasp'})
    assert result['error'] and not result['submitted']


@pytest.mark.parametrize('tool,params,expected', [
    ('run_xtb_optimize', {'input_path': 'sample with space.xyz', 'output_dir': 'out dir', 'gfn': 1, 'charge': -1, 'opt_level': 'tight'},
     ["'sample with space.xyz'", '--gfn 1', '--chrg -1', '--opt tight']),
    ('build_guest_forcefield', {'name': 'guest', 'input_path': 'input molecule.sdf', 'output_dir': 'out dir'},
     ["--input 'input molecule.sdf'", "--outdir 'out dir'"]),
    ('run_md_optimize', {'cif_path': 'sample.cif', 'mode': 'md', 'temperature': 320, 'force_field': 'Dreiding', 'cutoff': 7},
     ['from tools.lammps_optimize import run_md', "'temperature': 320", "'force_field': 'Dreiding'", "'cutoff': 7", 'if not r.success: sys.exit(1)']),
])
def test_advertised_parameters_reach_actual_compute_command(tool, params, expected, tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr('agents.registry._ws', lambda *a: str(tmp_path))
    monkeypatch.setattr('agents.registry.slurm.submit_and_return', lambda **kw: captured.append(kw) or {'submitted': True, 'job_id': '42'})
    result = get_registry().execute_dict(tool, params)
    assert result['submitted']
    command = captured[0]['command']
    assert all(fragment in command for fragment in expected)
    import subprocess
    assert subprocess.run(['bash', '-n'], input=command, text=True, capture_output=True).returncode == 0


def test_structure_distribution_never_exceeds_requested_count(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr('agents.registry.slurm.submit_and_return', lambda **kw: captured.append(kw) or {'submitted': True, 'job_id': '42'})
    monkeypatch.setattr('agents.registry._register_jobwatch', lambda *a, **kw: None)
    params = {'material_type': 'HOF', 'n_structures': 5, 'output_dir': str(tmp_path), 'topologies': ['srs', 'cds']}
    assert get_registry().execute_dict('generate_structure', params)['submitted']
    import re
    assert list(map(int, re.findall(r'--n (\d+)', captured[0]['command']))) == [3, 2]


def test_job_listing_without_owner_is_fail_closed(monkeypatch):
    monkeypatch.setattr('agents.watch_context.get_context', lambda: {})
    result = get_registry().execute_dict('list_my_jobs', {})
    assert result['error'] and not result['jobs']


def test_soft_warning_never_creates_two_results_for_one_tool_use(tmp_path):
    def read_file(**params):
        return {'content': params['path']}
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    agent = Agent(name='lead-orchestrator', functions=[read_file])
    responses = iter([
        *[SimpleNamespace(content=[ToolUseBlock(type='tool_use', id=f'r{i}', name='read_file', input={'path': f'{i}.cif'})]) for i in range(13)],
        SimpleNamespace(content=[TextBlock(type='text', text='Done.')]),
    ])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(agent)
    ids = [r['tool_use_id'] for m in s.messages if m['role'] == 'user' and isinstance(m['content'], list)
           for r in m['content'] if r.get('type') == 'tool_result']
    assert len(ids) == 13 and len(set(ids)) == 13


def test_read_mutation_read_same_arguments_must_read_new_state(tmp_path):
    observed, state = [], {'value': 'before'}
    def read_file(**params):
        observed.append(state['value'])
        return {'content': state['value']}
    def write_file(**params):
        state['value'] = params['content']
        return {'ok': True}
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s.messages = [{'role': 'user', 'content': 'hello'}]
    agent = Agent(name='lead-orchestrator', functions=[read_file, write_file])
    responses = iter([
        SimpleNamespace(content=[
            ToolUseBlock(type='tool_use', id='before', name='read_file', input={'path': 'x.cif'}),
            ToolUseBlock(type='tool_use', id='write', name='write_file', input={'path': 'x.cif', 'content': 'after'}),
            ToolUseBlock(type='tool_use', id='after', name='read_file', input={'path': 'x.cif'}),
        ]),
        SimpleNamespace(content=[TextBlock(type='text', text='Done.')]),
    ])
    s._call_api = lambda agent: next(responses)
    s._execute_loop(agent)
    assert observed == ['before', 'after']


def test_delegation_identity_uses_full_envelope_alias_and_evidence(tmp_path):
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    args = {'task': 'CO2 isotherm 298K', 'context': 'input:first.cif'}
    original = s._delegation_fingerprint('handoff_to_harness', args)
    assert original == s._delegation_fingerprint('handoff_to_harness-maintainer', args)
    assert original != s._delegation_fingerprint('handoff_to_harness', {**args, 'task': 'CO2 isotherm 320K'})
    assert original != s._delegation_fingerprint('handoff_to_harness', {**args, 'context': 'input:second.cif'})
    s.memory.record_tool_call('analyst', 'read_file', {'path': 'first.cif'}, '{"content":"new diagnosis"}')
    assert original != s._delegation_fingerprint('handoff_to_harness', args)


def test_serial_same_arguments_match_next_node_not_completed_node(tmp_path, monkeypatch):
    from agents.task_line import TaskLineStore
    store = TaskLineStore(str(tmp_path / 'lines.json'))
    args = {'path': 'mutable.cif'}
    store.upsert_step('c1', 'before', tool='read_file', arguments=args)
    store.upsert_step('c1', 'after', tool='read_file', arguments=args, depends_on=['before'])
    monkeypatch.setattr('agents.task_line.get_store', lambda: store)
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._current_line_id = 'c1'
    assert s._workflow_node_for_call('read_file', args)['step_id'] == 'before'
    store.upsert_step('c1', 'before', status='completed', done=True)
    assert s._workflow_node_for_call('read_file', args)['step_id'] == 'after'
    assert s._check_taskline_dependencies('read_file', args)[0]


def test_parallel_indistinguishable_nodes_are_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr('agents.task_line.get_store', lambda: SimpleNamespace(get_line=lambda *a: {
        'steps': [{'step_id': 'a', 'tool': 'read_file', 'arguments': {'path': 'x'}},
                  {'step_id': 'b', 'tool': 'read_file', 'arguments': {'path': 'x'}}]}))
    s = Session(config=AgentConfig(api_key='test', project_root=tmp_path))
    s._current_line_id = 'c1'
    with pytest.raises(ValueError, match='AMBIGUOUS_NODE'):
        s._workflow_node_for_call('read_file', {'path': 'x'})
    assert not s._check_taskline_dependencies('read_file', {'path': 'x'})[0]
