import copy
import json

import pytest

from agents.orchestration_chain import sync_chain
from agents.orchestration_chain import read_last_graph


def test_complete_graph_survives_empty_restore_and_is_owner_scoped(tmp_path):
    line, runtime = fixture()
    sync_chain(tmp_path, line, runtime, 'saved')
    sync_chain(tmp_path, {'username': 'u', 'conv_id': 'c', 'steps': []}, {}, 'empty_restore')
    graph = read_last_graph(tmp_path, {'username': 'u', 'conv_id': 'c'})
    assert graph['version'] == 2
    assert graph['nodes'][0]['step_id'] == 'compute'
    assert graph['retained'] is True
    assert json.loads((tmp_path / 'orchestration_chain.json').read_text())['current']['nodes']['compute']
    assert read_last_graph(tmp_path, {'username': 'other', 'conv_id': 'c'}) == {}


def test_graph_replaces_atomically_and_rejects_stale_plan(tmp_path):
    line, runtime = fixture()
    sync_chain(tmp_path, line, runtime, 'version2')
    revised = copy.deepcopy(line)
    revised['plan_version'] = 3
    revised['steps'].append({**copy.deepcopy(revised['steps'][0]),
                            'step_id': 'report', 'depends_on': ['compute']})
    sync_chain(tmp_path, revised, runtime, 'version3')
    sync_chain(tmp_path, line, runtime, 'late_version2')
    graph = read_last_graph(tmp_path, {'username': 'u', 'conv_id': 'c'})
    assert graph['version'] == 3
    assert [n['step_id'] for n in graph['nodes']] == ['compute', 'report']
    assert graph['nodes'][1]['depends_on'] == ['compute']


def fixture():
    node = {'step_id': 'compute', 'agent': 'analyst', 'tool': 'read_file', 'arguments': {'path': 'a.csv'},
            'depends_on': [], 'expected_outputs': ['a.csv'], 'status': 'pending'}
    old = {**copy.deepcopy(node), 'step_id': 'generated', 'status': 'completed', 'done': True, 'job_ids': ['10']}
    line = {'username': 'u', 'conv_id': 'c', 'plan_version': 2, 'steps': [node],
            'plan_history': [{'version': 1, 'steps': [old]}]}
    runtime = {'username': 'u', 'conv_id': 'c', 'plan_version': 2, 'status': 'active',
        'nodes': {'compute': {'contract': {k: v for k, v in node.items() if k not in {'status'}}, 'status': 'prefinish',
            'job_ids': ['11'], 'path_manifest': {'calculation_dir': '/owned/job11', 'output_files': ['/owned/job11/a.csv']}}}}
    return line, runtime


def test_chain_retains_retired_nodes_paths_branches_and_turn_boundaries(tmp_path):
    line, runtime = fixture()
    state = {'goal_contract': {'method': 'CDFT'}, 'error_branches': {'branch1': {'status': 'open'}}}
    first = sync_chain(tmp_path, line, runtime, 'turn_begin', state=state, turn_id='turn1')
    runtime['nodes']['compute']['status'] = 'succeeded'
    second = sync_chain(tmp_path, line, runtime, 'turn_end', state=state, turn_id='turn1')
    data = json.loads((tmp_path/'orchestration_chain.json').read_text())
    assert second['revision'] == first['revision'] + 1
    assert data['current']['retired_nodes']['generated'][0]['node']['job_ids'] == ['10']
    assert data['current']['nodes']['compute']['path_manifest']['output_files'] == ['/owned/job11/a.csv']
    assert data['current']['branches']['branch1']['status'] == 'open'
    assert data['turns']['turn1']['last_boundary'] == 'turn_end'
    assert data['revisions'][1]['parent_hash'] == data['revisions'][0]['hash']


def test_background_update_does_not_erase_main_state_and_noop_does_not_add_revision(tmp_path):
    line, runtime = fixture()
    sync_chain(tmp_path, line, runtime, 'main', state={'error_branches': {'b': {'status': 'open'}}})
    sync_chain(tmp_path, line, runtime, 'worker')
    data = json.loads((tmp_path/'orchestration_chain.json').read_text())
    assert 'b' in data['current']['branches']
    assert len(data['revisions']) == 1


def test_chain_refuses_foreign_job_or_owner(tmp_path):
    line, runtime = fixture()
    with pytest.raises(PermissionError):
        sync_chain(tmp_path, line, runtime, 'bad', jobs=[{'job_id': '1', 'username': 'other', 'conv_id': 'c'}])
    sync_chain(tmp_path, line, runtime, 'good')
    line['username'] = runtime['username'] = 'other'
    with pytest.raises(PermissionError): sync_chain(tmp_path, line, runtime, 'bad owner')


def test_chain_supplier_reads_latest_execution_under_lock(tmp_path):
    line, runtime = fixture()
    sync_chain(tmp_path, lambda: line, lambda: runtime, 'old')
    runtime['nodes']['compute']['status'] = 'succeeded'
    sync_chain(tmp_path, lambda: line, lambda: runtime, 'new')
    assert json.loads((tmp_path/'orchestration_chain.json').read_text())['current']['nodes']['compute']['status'] == 'succeeded'


def test_supervisor_session_checkpoint_preserves_main_branches_and_collects_owned_jobs(tmp_path, monkeypatch):
    from agents.session import Session
    from agents import task_line
    from types import SimpleNamespace
    line, runtime = fixture()
    sync_chain(tmp_path, line, runtime, 'main_restore', state={
        'goal_contract': {'method': 'CDFT'}, 'error_branches': {'b': {'status': 'open'}},
        'pending_workflow_patch': {'id': 'proposal'}})
    monkeypatch.setattr(task_line, 'get_store', lambda: SimpleNamespace(get_line=lambda _: line))
    observer = Session.__new__(Session)
    observer._evidence_root = tmp_path / 'evidence'
    observer._current_line_id = 'c'
    observer._runtime_snapshot = lambda: runtime
    observer._chain_state_is_primary = False
    observer._chain_jobs = lambda: [{'username': 'u', 'conv_id': 'c', 'job_id': '11', 'work_dir': '/owned/job11'}]
    observer.context = {}
    observer._persist_chain('supervisor_restore', {'error_branches': {}, 'goal_contract': {}, 'pending_workflow_patch': None})
    data = json.loads((tmp_path/'orchestration_chain.json').read_text())['current']
    assert data['branches']['b']['status'] == 'open'
    assert data['goal']['method'] == 'CDFT'
    assert data['pending_patch']['id'] == 'proposal'
    assert data['jobs']['11']['work_dir'] == '/owned/job11'
