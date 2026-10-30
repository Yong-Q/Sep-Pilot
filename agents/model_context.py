"""Small model-input views of replaceable state, with the audit history intact."""
import copy
import hashlib
import json
from pathlib import Path
import re

from .recovery import failure_reason, result_object


STATE_QUERIES = frozenset({
    'task_line_query', 'lifecycle_state', 'recovery_state',
    'resource_health', 'get_tool_schema',
})


def compact_lifecycle_events(messages, events, runtime):
    """Replace verified historical machine notices in a model-only view."""
    from .lifecycle import superseded_runtime_event
    view = copy.deepcopy(messages)
    count = 0
    prefix = '[生命周期事件]\n'
    for message in view:
        content = message.get('content')
        if message.get('role') != 'user' or not isinstance(content, str) or not content.startswith(prefix):
            continue
        try:
            notice = json.loads(content[len(prefix):])
        except (ValueError, TypeError):
            continue
        if not isinstance(notice, dict):
            continue
        saved = events.get(notice.get('event_id')) or {}
        if (not saved or any(notice.get(key) != saved.get(key) for key in ('kind', 'time'))
                or not (saved.get('main_chat') in {'delivered', 'obsolete'}
                        or superseded_runtime_event(saved, runtime))):
            continue
        message['content'] = json.dumps({'historical_event': notice['event_id'], 'kind': notice['kind'],
            'time': notice['time'], 'delivery_status': saved.get('main_chat'),
            'note': 'Historical machine notification; use the current durable runtime and fresh diagnostic evidence. Original event and audit remain stored.'}, ensure_ascii=False)
        count += 1
    return view, count


def compact_state_snapshots(messages, minimum_characters=2000, evidence_root=None,
                            maximum_snapshot_characters=6000, project_root=None):
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
    source_reads = _compact_historical_source_reads(view, evidence_root, project_root,
                                                   minimum_characters, maximum_snapshot_characters)
    return view, {'replaced_snapshots': replaced, 'referenced_snapshots': referenced,
                  'referenced_source_reads': source_reads, 'characters_before': before,
                  'characters_after': len(json.dumps(view, ensure_ascii=False, default=str))}


def _compact_historical_source_reads(view, evidence_root, project_root, preview_size, maximum_size):
    """Reference old source inspections only after exact scoped evidence proof.

    Current-turn reads remain complete. The source allowlist deliberately
    excludes run products, configuration, datasets and arbitrary analysis code.
    This mutates only the already-copied model view, never evidence or history.
    """
    if not evidence_root or not project_root:
        return 0
    root = Path(project_root).resolve()
    evidence = Path(evidence_root).resolve()
    machine_prefixes = ('[生命周期事件]', '[系统', '[JobWatch]', '<reflection', '<plan')
    boundary = max((i for i, message in enumerate(view)
                    if message.get('role') == 'user' and isinstance(message.get('content'), str)
                    and not message['content'].lstrip().startswith(machine_prefixes)), default=-1)
    calls = {}
    for message in view:
        if message.get('role') != 'assistant' or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if isinstance(block, dict) and block.get('type') == 'tool_use' and block.get('name') == 'read_file':
                calls[block.get('id')] = block.get('input')
    count = 0
    for index, message in enumerate(view):
        if index >= boundary:
            break
        if message.get('role') != 'user' or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if (not isinstance(block, dict) or block.get('type') != 'tool_result'
                    or block.get('is_error') or block.get('tool_use_id') not in calls):
                continue
            arguments = calls[block['tool_use_id']]
            content = block.get('content')
            if not isinstance(arguments, dict) or not isinstance(content, str) or len(content) <= maximum_size:
                continue
            obj = result_object(content)
            if (failure_reason(obj) or obj.get('blocked') or obj.get('executed') is False
                    or not isinstance(obj.get('content'), str)):
                continue
            requested, observed = arguments.get('path'), obj.get('path')
            if not isinstance(requested, str) or not isinstance(observed, str):
                continue
            source = (root / requested).resolve()
            if source != (root / observed).resolve() or source.suffix != '.py' or not source.is_relative_to(root):
                continue
            relative = source.relative_to(root)
            if len(relative.parts) != 1 and relative.parts[0] not in {'agents', 'bimem_agent', 'tools'}:
                continue
            reference = obj.get('evidence_ref')
            if not isinstance(reference, dict) or reference.get('tool', 'read_file') != 'read_file':
                continue
            call_id = reference.get('call_id')
            if not isinstance(call_id, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', call_id):
                continue
            path = evidence / (call_id + '.json')
            if path.is_symlink() or not path.is_file():
                continue
            try:
                saved = json.loads(path.read_text())
                expected = {key: value for key, value in obj.items() if key != 'evidence_ref'}
                actual = result_object(saved.get('result'))
                actual.pop('evidence_ref', None)
                if (saved.get('call_id') != call_id or saved.get('tool') != 'read_file'
                        or saved.get('params') != arguments or saved.get('failed')
                        or actual != expected):
                    continue
            except (ValueError, OSError, AttributeError):
                continue
            block['content'] = json.dumps({
                'tool': 'read_file', 'historical_source_read': True,
                'source_path': str(source), 'read_arguments': arguments,
                'evidence_ref': reference, 'full_result_path': str(path),
                'original_characters': len(content),
                'source_content_sha256': hashlib.sha256(obj['content'].encode('utf-8')).hexdigest(),
                'preview': content[:preview_size],
                'note': 'Historical source inspection; full original read remains in verified evidence. Source may have changed. Read the evidence for the original text, or read the source again for its current version.',
            }, ensure_ascii=False)
            count += 1
    return count


def compact_runtime_metadata(summary):
    """Digest bulky file metadata in a model-only copy of a native summary.

    The authoritative runtime, lifecycle_state responses, errors, validation
    dossiers and measurement fields remain complete. Successful report previews
    may be shortened only when their full file and executor receipt are named.
    This pure function does not read files, mutate the runtime, or assert that
    a referenced scientific result has passed any additional verification.
    """
    view = copy.deepcopy(summary)
    if not isinstance(view, dict):
        return view
    nodes = view.get('nodes')
    if not isinstance(nodes, dict):
        return view
    file_fields = {'path', 'name', 'size', 'bytes', 'sha256', 'mtime', 'modified_at',
                   'exists', 'is_file', 'is_dir', 'kind'}
    for node in nodes.values():
        if not isinstance(node, dict) or not isinstance(node.get('path_manifest'), dict):
            continue
        manifest = node['path_manifest']
        for field in ('input_paths', 'output_files', 'files'):
            items = manifest.get(field)
            if not isinstance(items, list) or not items:
                continue
            if not all(isinstance(item, str) or (
                    isinstance(item, dict) and set(item) <= file_fields
                    and isinstance(item.get('path') or item.get('name'), str)
                    and all(isinstance(value, (str, int, float, bool, type(None))) for value in item.values())
            ) for item in items):
                continue
            encoded = json.dumps(items, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
            if len(encoded) <= 2000:
                continue
            manifest[field] = {
                'count': len(items), 'sha256': hashlib.sha256(encoded.encode('utf-8')).hexdigest(),
                'sample': copy.deepcopy(items[:2]), 'full_metadata_tool': 'lifecycle_state',
                'note': 'Model-view file-list digest; complete original list remains in the native runtime metadata.',
            }
    reports = view.get('live_reports')
    for report in reports if isinstance(reports, list) else []:
        if not isinstance(report, dict) or report.get('status') != 'succeeded':
            continue
        content = report.get('content')
        source = nodes.get(report.get('step_id'), {})
        reference = source.get('result_ref') if isinstance(source, dict) else None
        if (not isinstance(content, str) or len(content) <= 2000
                or not isinstance(report.get('report_file'), str) or not report['report_file']
                or not isinstance(reference, dict) or not isinstance(reference.get('call_id'), str)
                or not reference['call_id']):
            continue
        report['content'] = content[:512]
        report['content_is_preview'] = True
        report['content_reference'] = {
            'original_characters': len(content),
            'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
            'report_file': report['report_file'], 'result_ref': copy.deepcopy(reference),
            'full_metadata_tool': 'lifecycle_state',
            'note': 'Short model-view preview; read the full report or native lifecycle metadata for its complete content.',
        }
    return view
