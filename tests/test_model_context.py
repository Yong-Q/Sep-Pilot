import copy
import json

from agents.model_context import compact_state_snapshots


def observation(call_id, tool, arguments, value):
    return [
        {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': call_id, 'name': tool, 'input': arguments}]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': call_id, 'content': json.dumps(value)}]},
    ]


def test_old_identical_state_query_is_compacted_without_changing_history_or_pairing():
    history = observation('old', 'task_line_query', {}, {'ok': True, 'state': 'old' * 4000})
    history += [{'role': 'user', 'content': 'Please continue with the approved task.'}]
    history += observation('new', 'task_line_query', {}, {'ok': True, 'state': 'latest' * 2000})
    original = copy.deepcopy(history)
    view, metrics = compact_state_snapshots(history)
    assert history == original
    assert metrics['replaced_snapshots'] == 1
    assert metrics['characters_after'] < metrics['characters_before']
    assert view[1]['content'][0]['tool_use_id'] == 'old'
    assert json.loads(view[1]['content'][0]['content'])['superseded_by'] == 'new'
    assert view[2:] == history[2:]


def test_distinct_queries_and_errors_and_scientific_results_are_preserved():
    history = observation('a', 'task_line_query', {'step_id': 'a'}, {'ok': True, 'state': 'a' * 5000})
    history += observation('b', 'task_line_query', {'step_id': 'b'}, {'ok': True, 'state': 'b' * 5000})
    history += observation('error', 'task_line_query', {}, {'error': 'failed' * 1000})
    history += observation('ok', 'task_line_query', {}, {'ok': True, 'state': 'new'})
    history += observation('s1', 'run_cdft', {}, {'density': 'measurement' * 800})
    history += observation('s2', 'run_cdft', {}, {'density': 'measurement' * 800})
    view, metrics = compact_state_snapshots(history)
    assert view == history
    assert metrics['replaced_snapshots'] == 0


def test_large_latest_snapshot_uses_verified_evidence_file(tmp_path):
    payload = {'ok': True, 'state': 'large-state' * 4000}
    history = observation('snapshot', 'lifecycle_state', {}, {
        **payload, 'evidence_ref': {'call_id': 'evidence123'}})
    (tmp_path / 'evidence123.json').write_text(json.dumps({
        'call_id': 'evidence123', 'result': json.dumps(payload)}))
    view, metrics = compact_state_snapshots(history, evidence_root=tmp_path)
    reference = json.loads(view[1]['content'][0]['content'])
    assert reference['full_result_path'] == str(tmp_path / 'evidence123.json')
    assert metrics['referenced_snapshots'] == 1
    assert metrics['characters_after'] < metrics['characters_before'] / 4
    assert len(history[1]['content'][0]['content']) > 30000


def test_large_result_without_matching_evidence_is_retained(tmp_path):
    history = observation('snapshot', 'lifecycle_state', {}, {
        'state': 'large-state' * 4000, 'evidence_ref': {'call_id': 'evidence123'}})
    (tmp_path / 'evidence123.json').write_text(json.dumps({
        'call_id': 'evidence123', 'result': '{"state":"different"}'}))
    view, metrics = compact_state_snapshots(history, evidence_root=tmp_path)
    assert view == history
    assert metrics['referenced_snapshots'] == 0
