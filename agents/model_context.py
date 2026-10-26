"""Small model-input views of replaceable state, with the audit history intact."""
import copy
import json
from pathlib import Path
import re

from .recovery import failure_reason, result_object


STATE_QUERIES = frozenset({
    'task_line_query', 'lifecycle_state', 'recovery_state',
    'resource_health', 'get_tool_schema',
})


def compact_state_snapshots(messages, minimum_characters=2000, evidence_root=None,
                            maximum_snapshot_characters=6000):
    """Replace older successful identical state reads in a temporary view.

    Authentic user messages, failures and scientific outputs stay unchanged.
    The latest state stays complete unless verified evidence backs a preview.
    Tool IDs and call/result
    pairing remain intact. Complete results stay in the original transcript
    and its evidence files; this function never mutates or writes history.
    """
    view = copy.deepcopy(messages)
    calls = {}
    for message in view:
        if message.get('role') != 'assistant' or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if not isinstance(block, dict) or block.get('type') != 'tool_use':
                continue
            if block.get('name') in STATE_QUERIES:
                calls[block.get('id')] = (block['name'], json.dumps(
                    block.get('input', {}), sort_keys=True, ensure_ascii=False))
    before = len(json.dumps(view, ensure_ascii=False, default=str))
    newest = {}
    replaced = 0
    for message in reversed(view):
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        for block in reversed(message['content']):
            if not isinstance(block, dict) or block.get('type') != 'tool_result':
                continue
            call_id = block.get('tool_use_id')
            query = calls.get(call_id)
            if not query or block.get('is_error'):
                continue
            content = block.get('content')
            if not isinstance(content, str):
                continue
            result = result_object(content)
            if (not result or failure_reason(result) or result.get('blocked')
                    or result.get('executed') is False):
                continue
            if query in newest and len(content) >= minimum_characters:
                reference = {'tool': query[0], 'superseded_by': newest[query],
                    'note': 'A newer successful result for this identical state query is included later. Full historical evidence remains in the session transcript.'}
                if result.get('evidence_ref'):
                    reference['evidence_ref'] = result['evidence_ref']
                block['content'] = json.dumps(reference, ensure_ascii=False)
                replaced += 1
            else:
                newest.setdefault(query, call_id)
    referenced = 0
    if evidence_root:
        root = Path(evidence_root).resolve()
        for message in view:
            if message.get('role') != 'user' or not isinstance(message.get('content'), list):
                continue
            for block in message['content']:
                if (not isinstance(block, dict) or block.get('type') != 'tool_result'
                        or block.get('tool_use_id') not in calls or block.get('is_error')):
                    continue
                content = block.get('content', '')
                if not isinstance(content, str) or len(content) <= maximum_snapshot_characters:
                    continue
                obj = result_object(content)
                reference = obj.get('evidence_ref')
                if not isinstance(reference, dict):
                    continue
                call_id = reference.get('call_id')
                if (not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', call_id)
                        or failure_reason(obj) or obj.get('blocked') or obj.get('executed') is False):
                    continue
                path = root / (call_id + '.json')
                if path.is_symlink() or not path.is_file():
                    continue
                try:
                    saved = json.loads(path.read_text())
                    if not isinstance(saved, dict):
                        continue
                    expected = {k: v for k, v in obj.items() if k != 'evidence_ref'}
                    actual = result_object(saved.get('result'))
                    actual.pop('evidence_ref', None)
                    if saved.get('call_id') != call_id or actual != expected:
                        continue
                except (ValueError, OSError):
                    continue
                block['content'] = json.dumps({
                    'tool': calls[block['tool_use_id']][0], 'evidence_ref': obj['evidence_ref'],
                    'full_result_path': str(path), 'original_characters': len(content),
                    'preview': content[:minimum_characters],
                    'note': 'Preview of a state snapshot. The complete verified result is in this evidence file; use read_file when details are needed.',
                }, ensure_ascii=False)
                referenced += 1
    return view, {'replaced_snapshots': replaced, 'referenced_snapshots': referenced, 'characters_before': before,
                  'characters_after': len(json.dumps(view, ensure_ascii=False, default=str))}
