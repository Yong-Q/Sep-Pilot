"""Pure Anthropic wire-history repair; never execution or authorization.

Only malformed client tool exchanges become explicitly inert historical text.
Native tool IDs, evidence payloads, and valid pairs are never fabricated.
"""
import copy
import json
from collections import Counter


def _wire_value(value):
    if isinstance(value, dict): return {key: _wire_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)): return [_wire_value(item) for item in value]
    if hasattr(value, 'model_dump'):
        return _wire_value(value.model_dump(exclude_none=True))
    return copy.deepcopy(value)


def _historical(block, reason):
    # Keep the complete ID/input/result/is_error evidence without presenting a
    # missing proposal as an executed tool, or a tool result as user approval.
    raw = json.dumps(block, ensure_ascii=False, sort_keys=True, default=str)
    return {'type': 'text', 'text': (
        '[Historical tool protocol record — context only; not a new user instruction, '
        'execution, retry permit, or scientific acceptance. '
        + reason + ']\n' + raw)}


def normalize_anthropic_messages(messages):
    """Return independent transport messages with exact adjacent client pairs.

    Call this after trimming/compaction. It does not modify retained Session
    history, result ledgers or tool definitions. Unknown block kinds (including
    signed thinking/server-tool blocks) remain intact.
    """
    copied = _wire_value(messages)
    if not isinstance(copied, list):
        raise ValueError('Anthropic messages must be a list')
    allowed = {}
    for index, message in enumerate(copied):
        if message.get('role') != 'assistant' or not isinstance(message.get('content'), list): continue
        uses = Counter(block.get('id') for block in message['content']
            if isinstance(block, dict) and block.get('type') == 'tool_use' and isinstance(block.get('id'), str) and block['id'])
        following = copied[index+1] if index+1 < len(copied) else {}
        results = Counter(block.get('tool_use_id') for block in following.get('content', [])
            if isinstance(block, dict) and block.get('type') == 'tool_result'
            and isinstance(block.get('tool_use_id'), str) and block['tool_use_id']) if following.get('role') == 'user' and isinstance(following.get('content'), list) else Counter()
        allowed[index] = {identifier for identifier, count in uses.items() if count == 1 and results[identifier] == 1}
    repaired = []
    for index, message in enumerate(copied):
        content, role = message.get('content'), message.get('role')
        if not isinstance(content, list):
            repaired.append(message);continue
        blocks = []
        for block in content:
            if not isinstance(block, dict):
                blocks.append(block);continue
            kind = block.get('type')
            if kind == 'tool_use' and (role != 'assistant' or block.get('id') not in allowed.get(index, set())):
                blocks.append(_historical(block, 'Unmatched or ambiguous proposal; execution is not established.'))
            elif kind == 'tool_result' and (role != 'user' or block.get('tool_use_id') not in allowed.get(index-1, set())):
                blocks.append(_historical(block, 'Unmatched or ambiguous result; retain its original evidence reference.'))
            else:
                blocks.append(block)
        if role == 'user':
            # Already-valid arrays keep their exact order. Only invalid mixed
            # ordering changes, as required by the Anthropic wire protocol.
            blocks = ([block for block in blocks if isinstance(block, dict) and block.get('type') == 'tool_result']
                      + [block for block in blocks if not isinstance(block, dict) or block.get('type') != 'tool_result'])
        repaired.append({**message, 'content': blocks})
    return repaired
