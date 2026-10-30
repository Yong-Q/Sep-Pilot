"""Current, scoped diagnostic facts for guard failures; never permissions."""
import hashlib
import json
from pathlib import Path


def current_workflow_feedback(session, step_id=None):
    """Read once on failure; a failed diagnostic must not mask the guard."""
    try:
        snapshot = getattr(session, '_runtime_snapshot', None)
        evidence_root = getattr(session, '_evidence_root', None)
        if not snapshot or not evidence_root:
            return None
        runtime = snapshot()
        root = Path(evidence_root)
        owner = {'username': root.parent.parent.name, 'conv_id': root.parent.name}
        if not isinstance(runtime, dict) or any(runtime.get(k) != v for k, v in owner.items()):
            return None
        goal = session.goal_contract.to_dict()
        rows, epoch_rows, waiting, counts = [], [], [], {}
        for key, node in sorted((runtime.get('nodes') or {}).items()):
            job_ids = list(node.get('job_ids') or [])
            row = {'step_id': key, 'status': node.get('status'), 'token': node.get('token'),
                   'tool': node.get('contract', {}).get('tool'), 'job_ids': job_ids[:8],
                   'job_count': len(job_ids), 'job_ids_truncated': len(job_ids) > 8}
            dossier = node.get('validation_dossier') or {}
            epoch_rows.append((key, row['status'], row['token'], job_ids, dossier.get('evidence_call_id')))
            counts[row['status']] = counts.get(row['status'], 0) + 1
            if row['status'] in {'running', 'waiting_jobs', 'waiting_prerequisite'}:
                waiting.append(key)
            if len(rows) < 32:
                # No raw outputs, arguments, histories or error transcripts.
                rows.append(row)
        result = {'diagnostic_only': True, 'scope': owner, 'runtime_plan_version': runtime.get('plan_version'),
            'approved_plan_version': goal.get('approved_plan_version'), 'goal_version': goal.get('version'),
            'runtime_goal_version': (runtime.get('goal_contract') or {}).get('version'),
            'status': runtime.get('status'), 'user_paused': bool(runtime.get('user_paused')),
            'pause_kind': runtime.get('pause_kind'), 'nodes': rows, 'node_counts': counts,
            'nodes_truncated': len(epoch_rows) > len(rows), 'waiting_on_current_attempts': waiting[:32],
            'waiting_attempt_count': len(waiting),
            'plan_activation_pending': runtime.get('plan_version') != goal.get('approved_plan_version')}
        node = (runtime.get('nodes') or {}).get(step_id, {})
        dossier = node.get('validation_dossier') or {}
        call_id = dossier.get('evidence_call_id')
        workflow_id = hashlib.sha256(json.dumps(list(owner.values())).encode()).hexdigest()[:24]
        if (node.get('status') == 'prefinish' and node.get('token')
                and dossier.get('workflow_id') == workflow_id and dossier.get('step_id') == step_id
                and dossier.get('plan_version') == runtime.get('plan_version')
                and dossier.get('attempt_id') == (node.get('attempt_id') or node['token'])
                and dossier.get('contract') == node.get('contract')
                and sorted(map(str, dossier.get('job_ids') or [])) == sorted(map(str, node.get('job_ids') or []))
                and isinstance(call_id, str) and call_id.isalnum() and len(call_id) <= 128
                and dossier.get('evidence_path') == str(root / (call_id + '.json'))):
            result['current_dossier'] = {'call_id': call_id, 'evidence_path': dossier['evidence_path']}
        epoch = {'owner': owner, 'goal': result['goal_version'], 'approved': result['approved_plan_version'],
                 'runtime': result['runtime_plan_version'], 'paused': result['user_paused'],
                 'status': result['status'], 'pause_kind': result['pause_kind'], 'nodes': epoch_rows,
                 'science': {key: goal.get(key) for key in ('method', 'gases', 'parameters', 'execution_authorized')}}
        result['state_epoch'] = hashlib.sha256(json.dumps(epoch, sort_keys=True).encode()).hexdigest()
        return result
    except Exception:
        return None
