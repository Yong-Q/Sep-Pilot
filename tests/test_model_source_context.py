"""Only historical, fully evidenced source reads may leave the model view."""
import copy
import json

import pytest

from agents.model_context import compact_state_snapshots


def source_observation(project, evidence, relative='agents/adapter.py', **payload_changes):
    path = project / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('# source\n' * 1200)
    arguments = {'path': str(path), 'offset': 0, 'limit': 1200}
    payload = {'path': str(path), 'content': '# source\n' * 1200,
               'offset': 0, 'limit': 1200, 'total_lines': 1200, **payload_changes}
    record = {'call_id': 'proof123', 'tool': 'read_file', 'params': arguments,
              'result': json.dumps(payload)}
    evidence.mkdir(exist_ok=True)
    (evidence / 'proof123.json').write_text(json.dumps(record))
    messages = [
        {'role': 'user', 'content': 'Inspect the adapter.'},
        {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'source_call', 'name': 'read_file', 'input': arguments}]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'source_call', 'content': json.dumps({
                **payload, 'evidence_ref': {'call_id': 'proof123', 'tool': 'read_file'}})}]},
        {'role': 'user', 'content': 'The review is finished; check the next step.'},
    ]
    return messages, record


def test_historical_source_preview_preserves_full_audit_and_provenance(tmp_path):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, _ = source_observation(project, evidence)
    original = copy.deepcopy(messages)
    view, metrics = compact_state_snapshots(messages, evidence_root=evidence, project_root=project)
    preview = json.loads(view[2]['content'][0]['content'])
    assert messages == original
    assert view[:2] == original[:2] and view[3:] == original[3:]
    assert view[2]['content'][0]['tool_use_id'] == 'source_call'
    assert preview['historical_source_read'] is True
    assert preview['read_arguments']['offset'] == 0
    assert preview['full_result_path'] == str(evidence / 'proof123.json')
    assert len(preview['source_content_sha256']) == 64
    assert metrics['referenced_source_reads'] == 1
    assert metrics['characters_after'] < metrics['characters_before'] / 2
    assert len(json.loads((evidence / 'proof123.json').read_text())['result']) > 10000


def test_current_turn_source_and_unconfigured_source_compaction_stay_complete(tmp_path):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, _ = source_observation(project, evidence)
    view, _ = compact_state_snapshots(messages[:-1], evidence_root=evidence, project_root=project)
    assert view == messages[:-1]
    view, _ = compact_state_snapshots(messages, evidence_root=evidence)
    assert view == messages


def test_machine_notification_does_not_make_current_source_a_previous_user_turn(tmp_path):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, _ = source_observation(project, evidence)
    messages[-1] = {'role': 'user', 'content': '[生命周期事件]\n{"event_id":"next"}'}
    view, _ = compact_state_snapshots(messages, evidence_root=evidence, project_root=project)
    assert view == messages


@pytest.mark.parametrize('bad_field', ['call_id', 'tool', 'params', 'result', 'missing', 'symlink'])
def test_source_preview_requires_exact_owned_evidence(tmp_path, bad_field):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, record = source_observation(project, evidence)
    if bad_field == 'missing':
        (evidence / 'proof123.json').unlink()
    elif bad_field == 'symlink':
        outside = tmp_path / 'outside.json'
        outside.write_text(json.dumps(record))
        (evidence / 'proof123.json').unlink()
        (evidence / 'proof123.json').symlink_to(outside)
    else:
        record[bad_field] = {'path': 'different'} if bad_field == 'params' else 'different'
        (evidence / 'proof123.json').write_text(json.dumps(record))
    view, metrics = compact_state_snapshots(messages, evidence_root=evidence, project_root=project)
    assert view == messages
    assert metrics['referenced_source_reads'] == 0


@pytest.mark.parametrize('relative', ['runs/result.py', 'data/table.py', 'config/site.py',
                                     'tools/results.csv', 'notes/analysis.py'])
def test_scientific_outputs_and_unapproved_source_locations_are_preserved(tmp_path, relative):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, _ = source_observation(project, evidence, relative)
    view, metrics = compact_state_snapshots(messages, evidence_root=evidence, project_root=project)
    assert view == messages
    assert metrics['referenced_source_reads'] == 0


def test_failed_source_read_remains_complete_even_with_exact_evidence(tmp_path):
    project, evidence = tmp_path / 'project', tmp_path / 'evidence'
    messages, _ = source_observation(project, evidence, error='source unreadable')
    view, metrics = compact_state_snapshots(messages, evidence_root=evidence, project_root=project)
    assert view == messages
    assert metrics['referenced_source_reads'] == 0


def test_truncated_lifecycle_wire_cannot_be_previewed_from_a_shorter_saved_record(tmp_path):
    raw = '{"state":"' + 'state' * 12000
    evidence = tmp_path / 'evidence'
    evidence.mkdir()
    (evidence / 'proof123.json').write_text(json.dumps({'call_id': 'proof123', 'result': raw}))
    messages = [
        {'role': 'assistant', 'content': [
            {'type': 'tool_use', 'id': 'state_call', 'name': 'lifecycle_state', 'input': {}}]},
        {'role': 'user', 'content': [
            {'type': 'tool_result', 'tool_use_id': 'state_call', 'content': json.dumps({
                'result': raw + '\n... (truncated)', 'evidence_ref': {'call_id': 'proof123'}})}]},
    ]
    view, metrics = compact_state_snapshots(messages, evidence_root=evidence)
    assert view == messages
    assert metrics['referenced_snapshots'] == 0
