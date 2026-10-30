"""Bounded executor identity verification for notification routing, not acceptance."""
import hashlib
import json
import math
from pathlib import Path


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def verify_executor_dossier_event(event, runtime, *, owned_root, goal_contract, jobs, session=None):
    """Return native provenance only; uncertain evidence retains the model reviewer.

    A successful result saves only a redundant supervisor turn. The scientific
    owner still reviews the dossier and must invoke the normal finish guard.
    No execution, scheduler lookup, output write or status transition occurs.
    """
    owned = None
    try:
        from .report_recovery import _OwnedFiles
        from .result_validation import VALIDATION_SCHEMA_VERSION, _identity
        from .workflow_patch import execution_contract
        from .recovery import result_object
        payload = event.get('payload') or {}
        if event.get('kind') != 'worker_prefinish' or payload.get('requires_user'):
            return None
        root = Path(owned_root)
        owner = {'username': root.parent.name, 'conv_id': root.name}
        goal = goal_contract.to_dict() if hasattr(goal_contract, 'to_dict') else goal_contract
        retained = None
        if payload.get('origin_goal_version') != goal.get('version') and session is not None:
            from .retained_prerequisite_scope import retained_node_event_proof
            retained = retained_node_event_proof(session,event,runtime,jobs=jobs)
        if ((runtime.get('username'), runtime.get('conv_id')) != (owner['username'], owner['conv_id'])
                or runtime.get('user_paused') or runtime.get('status') not in {'active', 'needs_user'}
                or goal.get('execution_authorized') is not True or goal.get('execution_mode') != 'workflow'
                or goal.get('version') != (runtime.get('goal_contract') or {}).get('version')
                or payload.get('origin_goal_version') != goal.get('version') and retained is None
                or goal.get('approved_plan_version') != runtime.get('plan_version')
                or payload.get('plan_version') != runtime.get('plan_version')):
            return None
        wid = hashlib.sha256(json.dumps([owner['username'], owner['conv_id']]).encode()).hexdigest()[:24]
        step = payload.get('step_id'); node = (runtime.get('nodes') or {}).get(step) or {}
        contract = node.get('contract') or {}; dossier = node.get('validation_dossier') or {}
        token = node.get('token'); attempt = node.get('attempt_id') or token
        job_ids = node.get('job_ids') or []
        approved = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step]
        if (not isinstance(token, str) or not token or not attempt or not job_ids
                or len(set(job_ids)) != len(job_ids) or any(not isinstance(jid, str) or not jid.isdigit() for jid in job_ids)
                or node.get('status') != 'prefinish' or node.get('phase') != 'prefinish'
                or node.get('origin_goal_version') != goal.get('version') and retained is None
                or node.get('execution_plan_version') != runtime.get('plan_version') and retained is None
                or node.get('jobs_confirmed_terminal') is not True or node.get('machine_output_check') is not True
                or payload.get('machine_output_check') is not True
                or payload.get('runtime_id') != wid or payload.get('workflow_id') != owner['conv_id']
                or payload.get('correlation_token') != token or payload.get('attempt_id') != attempt
                or event.get('event_id') != f'runtime:{wid}:{token}:prefinish'
                or payload.get('job_ids') != job_ids or payload.get('contract') != contract
                or _canonical(payload.get('validation_dossier')) != _canonical(dossier)
                or payload.get('result_ref') != node.get('result_ref')
                or len(approved) != 1 or execution_contract(approved[0]) != execution_contract(contract)):
            return None
        if (dossier.get('schema_version') != VALIDATION_SCHEMA_VERSION or dossier.get('verdict') != 'pass'
                or dossier.get('requires_agent_review') is not True):
            return None
        owned = _OwnedFiles(root, owner)
        call_id = dossier.get('evidence_call_id')
        expected_path = root / 'evidence' / (str(call_id) + '.json')
        if not isinstance(call_id, str) or not call_id.isalnum() or dossier.get('evidence_path') != str(expected_path):
            return None
        record, record_fact = owned.read(expected_path, as_json=True)
        native = record.get('result') or {}; inventory = native.get('inventory') or {}
        identity = {'workflow_id': wid, 'plan_version': runtime['plan_version'], 'step_id': step,
                    'job_ids': job_ids, 'contract': contract}
        if (record.get('call_id') != call_id or record.get('tool') != 'inspect_workflow_result'
                or record.get('agent') != contract.get('agent') or record.get('failed') is not False
                or record.get('params') != {'step_id': step, 'path': inventory.get('root')}
                or native.get('evidence_call_id') != call_id or native.get('evidence_path') != str(expected_path)
                or native.get('attempt_id') != attempt or native.get('contract') != contract
                or native.get('workflow_id') != wid or native.get('plan_version') != runtime['plan_version']
                or native.get('step_id') != step or native.get('job_ids') != job_ids
                or native.get('verification_owner') != contract.get('agent')
                or inventory.get('escaped_paths') or inventory.get('truncated') is not False
                or _identity(identity, attempt, inventory) != call_id):
            return None
        for key, value in dossier.items():
            if key == 'inventory':
                if not isinstance(value, dict) or any(_canonical(inventory.get(k)) != _canonical(v) for k, v in value.items()):
                    return None
            elif _canonical(native.get(key)) != _canonical(value):
                return None
        reference = node.get('result_ref') or {}; producer_id = reference.get('call_id')
        producer_path = root / 'evidence' / (str(producer_id) + '.json')
        if (not isinstance(producer_id, str) or not producer_id.isalnum()
                or reference.get('evidence_path') != str(producer_path)):
            return None
        producer, producer_fact = owned.read(producer_path, as_json=True)
        checkpoint, checkpoint_fact = owned.read(reference['checkpoint_path'], as_json=True)
        logged = [item for item in (checkpoint.get('memory') or {}).get('tool_call_log', []) if item.get('call_id') == producer_id]
        if (checkpoint.get('owner') != {**owner, 'step_id': step} or len(logged) != 1
                or execution_contract((checkpoint.get('context') or {}).get('assigned_node') or {}) != execution_contract(contract)
                or producer.get('call_id') != producer_id or producer.get('agent') != contract.get('agent')
                or producer.get('tool') != contract.get('tool') or producer.get('failed') is not False
                or _canonical(producer.get('params')) != _canonical(node.get('effective_arguments') or contract.get('arguments'))
                or logged[0].get('evidence_path') != str(producer_path)
                or any(_canonical(logged[0].get(key)) != _canonical(producer.get(key)) for key in ('call_id', 'time', 'agent', 'tool', 'params', 'failed'))):
            return None
        started, produced, inspected = node.get('started_at'), producer.get('time'), native.get('generated_at')
        if (any(type(value) not in (int, float) or not math.isfinite(value) for value in (started, produced, inspected))
                or not started <= produced <= inspected or record.get('time') != inspected):
            return None
        result = result_object(producer.get('result'))
        produced_jobs = result.get('job_ids') or ([str(result['job_id'])] if result.get('job_id') else [])
        if sorted(map(str, produced_jobs)) != sorted(job_ids):
            return None
        matching = [job for job in jobs if str(job.get('job_id')) in job_ids]
        if len(matching) != len(job_ids):
            return None
        for job in matching:
            if ((job.get('username'), job.get('conv_id')) != (owner['username'], owner['conv_id'])
                    or job.get('tool') != contract.get('tool') or job.get('state') != 'COMPLETED'
                    or job.get('terminal') is not True or job.get('failed') is not False
                    or job.get('attempt_id') != node.get('attempt_id') or job.get('recovery_key') != node.get('recovery_key')
                    or job.get('work_dir') != result.get('work_dir')):
                return None
            owned.owned(job['work_dir'])
        # Re-read only this attempt's bounded samples and declared artifacts; no scan.
        observed, total = {}, 0
        def artifact(path):
            nonlocal total
            key = str(path)
            if key not in observed:
                _, fact = owned.read(path)
                total += fact['size']
                if len(observed) >= 64 or total > 64 * 1024 * 1024:
                    raise ValueError('fast dossier artifact bound exceeded')
                observed[key] = fact
            return observed[key]
        samples = native.get('samples') or []
        if not samples or len(samples) > 64:
            return None
        for sample in samples:
            path = Path(inventory['root']) / sample['path']
            fact = artifact(path)
            if fact['sha256'] != sample.get('sha256') or fact['size'] != sample.get('size'):
                return None
        artifacts = node.get('artifacts') or {}
        if not artifacts:
            return None
        for path, expected in artifacts.items():
            if not expected:
                return None
            entries = expected.get('files') or [[path, expected['size'], expected['mtime_ns']]]
            if len(entries) > 64:
                return None
            for name, size, modified in entries:
                fact = artifact(name)
                if fact['size'] != size or fact['mtime_ns'] != modified or size <= 0:
                    return None
        return {'kind': 'executor_dossier_identity', 'runtime_id': wid, 'step_id': step, 'token': token,
                'goal_version': goal['version'], 'plan_version': runtime['plan_version'],
                'attempt_id': attempt, 'job_ids': job_ids, 'dossier_call_id': call_id,
                'contract_sha256': hashlib.sha256(_canonical(contract).encode()).hexdigest(),
                'dossier_sha256': record_fact['sha256'], 'producer_call_id': producer_id,
                'producer_evidence_sha256': producer_fact['sha256'], 'producer_checkpoint_sha256': checkpoint_fact['sha256'],
                'needs_scientific_owner_review': True, 'not_resubmitted': True,
                **({'retained_prerequisite_scope_sha256':retained['proof_sha256']} if retained else {})}
    except (OSError, ValueError, TypeError, KeyError, AttributeError, OverflowError):
        return None
    finally:
        if owned is not None:
            owned.close()
