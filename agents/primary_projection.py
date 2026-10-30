"""Display-only primary state loaded from the durable main checkpoint.

No live Session is accessed by this renderer. A supervisor's local checkpoint
cannot replace the main goal or pending user interaction. These files grant no
execution or scientific acceptance authority.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat

MAX_CHECKPOINT_BYTES = 64 * 1024 * 1024


def read_primary_checkpoint(root, owner):
    root = Path(os.path.abspath(root))
    if (not isinstance(owner, dict) or set(owner) != {'username', 'conv_id'}
            or (root.parent.name, root.name) != (owner.get('username'), owner.get('conv_id'))
            or root.resolve(strict=True) != root):
        raise PermissionError('primary projection owner/root mismatch')
    parent = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        root_identity = os.fstat(parent)
        try:
            fd = os.open('session_checkpoint.json', os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        except FileNotFoundError:
            return None, None
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= MAX_CHECKPOINT_BYTES:
                raise ValueError('primary checkpoint must be a bounded regular file')
            pieces, size = [], 0
            while True:
                piece = os.read(fd, 256 * 1024)
                if not piece: break
                size += len(piece)
                if size > MAX_CHECKPOINT_BYTES: raise ValueError('primary checkpoint exceeds bounded format')
                pieces.append(piece)
            after = os.fstat(fd)
            linked = os.stat('session_checkpoint.json', dir_fd=parent, follow_symlinks=False)
            identity = lambda value: (value.st_dev, value.st_ino, value.st_mode, value.st_size,
                                      value.st_mtime_ns, value.st_ctime_ns)
            current_root = os.stat(root, follow_symlinks=False)
            if (size != before.st_size or identity(before) != identity(after) or identity(after) != identity(linked)
                    or (root_identity.st_dev, root_identity.st_ino) != (current_root.st_dev, current_root.st_ino)):
                raise ValueError('primary checkpoint changed during display read')
            raw = b''.join(pieces)
        finally:
            os.close(fd)
    finally:
        os.close(parent)
    state = json.loads(raw)
    if not isinstance(state, dict) or state.get('owner') != owner:
        raise PermissionError('primary checkpoint belongs to another conversation')
    stamp = state.get('checkpoint_at')
    if type(stamp) not in (int, float) or not math.isfinite(stamp) or not isinstance(state.get('goal_contract'), dict):
        raise ValueError('primary checkpoint has invalid display metadata')
    return state, hashlib.sha256(raw).hexdigest()


def _evidence_references(root, state):
    references = {}
    memory = state.get('memory') or {}
    logs = [memory.get('tool_call_log', [])]
    logs.extend(item.get('tool_call_log', []) for item in (memory.get('agent_memories') or {}).values())
    for log in logs:
        for call in log:
            cid, path = call.get('call_id'), call.get('evidence_path')
            if not path: continue
            if not isinstance(cid, str) or not cid.isalnum() or path != str(Path(root)/'evidence'/(cid+'.json')):
                raise PermissionError('primary evidence reference escaped its conversation')
            references[cid] = path
    return references


def render_primary_projection(root, owner, line, runtime, reason, *, jobs=None):
    from .orchestration_chain import sync_chain
    from .state_io import write_checkpoint
    from .workflow_view import authoritative_line
    root = Path(root)
    for value in (line, runtime):
        if value and (value.get('username'), value.get('conv_id')) != (owner['username'], owner['conv_id']):
            raise PermissionError('display graph belongs to another conversation')
    state, checkpoint_sha = read_primary_checkpoint(root, owner)
    result = sync_chain(root, line, runtime, reason, state=state, jobs=jobs,
                        turn_id=((state or {}).get('context') or {}).get('active_turn_id'))
    if state is not None:
        references = _evidence_references(root, state)
        write_checkpoint(root/'task_manifest.json', {
            'updated_at': state['checkpoint_at'], 'goal_contract': state['goal_contract'],
            'workflow': authoritative_line(line, runtime,state['goal_contract']), 'current_agent': state.get('current_agent'),
            'chain_file': str(root/'orchestration_chain.json'),
            'pending_workflow_patch': state.get('pending_workflow_patch'),
            'waiting_for_user_input': state.get('waiting_for_user_input'),
            'pending_user_interaction': state.get('pending_user_interaction'),
            'recovery_file': str(root/'recovery_state.json'), 'lifecycle_file': str(root/'lifecycle.json'),
            'session_checkpoint_file': str(root/'session_checkpoint.json'),
            'supervisor_checkpoint_file': str(root/'supervisor_checkpoint.json'),
            'tool_evidence_refs': references, 'source_checkpoint_sha256': checkpoint_sha,
            'display_only': True, 'delivery_owner': 'lead-orchestrator', 'negotiation_owner': 'lead-orchestrator',
        })
    return result
