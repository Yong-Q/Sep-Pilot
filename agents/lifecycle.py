"""Durable lifecycle mailbox for main chat and the read-only supervisor.

Events carry the executable workflow/node detail, not a reconstructed prose
plan. One event is acknowledged independently by each long-lived agent.
"""
import json
import copy
import math
import os
from pathlib import Path
import stat
import tempfile
import time
import uuid
import threading

from .state_io import ConversationLock, json_transaction


def _publish_liveness(path, mailbox):
    """Replace only expendable liveness metadata, without reading old history."""
    lease = ConversationLock(str(path) + '.lock')
    lease.acquire()
    temporary = None
    try:
        data = {'schema_version': 1, 'mailbox': str(mailbox.resolve()), 'agents': {
            role: {'service_heartbeat': time.time(), 'service_owner_pid': os.getpid(), 'lifetime': True}
            for role in ('main_chat', 'supervisor')}}
        fd, temporary = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            json.dump(data, stream)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try: os.fsync(directory)
        finally: os.close(directory)
    finally:
        try:
            if temporary:
                try: os.unlink(temporary)
                except FileNotFoundError: pass
        finally:
            lease.release()


def _read_liveness(path, mailbox):
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode) or info.st_size > 16384: return {}
            value = json.loads(stream.read(16385))
        if (not isinstance(value, dict) or value.get('schema_version') != 1
                or value.get('mailbox') != str(mailbox.resolve())):
            return {}
        result = {}
        for role in ('main_chat', 'supervisor'):
            item = value.get('agents', {}).get(role, {})
            stamp, pid = item.get('service_heartbeat'), item.get('service_owner_pid')
            if (type(stamp) in (int, float) and math.isfinite(stamp) and stamp > 0
                    and type(pid) is int and pid > 0):
                result[role] = {'service_heartbeat': stamp, 'service_owner_pid': pid, 'lifetime': True}
        return result
    except (OSError, ValueError, TypeError, AttributeError):
        # Optional display liveness can never replace mailbox authority.
        return {}



def _native_role_warning(event_id, event):
    """Only the heartbeat's exact maintenance identity, never user/science work."""
    if not isinstance(event,dict)or event.get('kind')!='role_unresponsive' or event.get('event_id')!=event_id:
        return False
    payload=event.get('payload')or{}
    if not isinstance(payload,dict):return False
    receiver,parent=payload.get('receiver'),payload.get('busy_event_id')
    return (isinstance(receiver,str)and receiver in {'main_chat','supervisor'} and isinstance(parent,str)and bool(parent)
        and payload.get('requires_user')is True
        and event_id=='role-unresponsive:'+receiver+':'+parent)


def _recursive_role_warning(event_id, event, events):
    if not _native_role_warning(event_id,event):return False
    parent=event['payload']['busy_event_id']
    return _native_role_warning(parent,events.get(parent))


def superseded_runtime_event(event, runtime):
    """Ignore old attempt/phase notices while preserving their durable audit."""
    payload = event.get('payload') or {}
    nodes = (runtime or {}).get('nodes') or {}
    node = nodes.get(payload.get('step_id'))
    if node:
        token = payload.get('correlation_token')
        if not token and payload.get('runtime_id'):
            prefix = 'runtime:' + str(payload['runtime_id']) + ':'
            event_id = str(event.get('event_id', ''))
            candidate = event_id[len(prefix):].split(':', 1)[0] if event_id.startswith(prefix) else ''
            if len(candidate) == 32 and all(c in '0123456789abcdef' for c in candidate):
                token = candidate
        if token and node.get('token') and token != node['token']:
            return True
        obsolete_phases = {
            'succeeded': {'worker_failed', 'worker_uncertain', 'worker_prefinish', 'worker_recovery_ready', 'worker_failure_recovery', 'worker_execution_deferred'},
            'failed': {'worker_uncertain', 'worker_started'},
        }
        if token == node.get('token') and event.get('kind') in obsolete_phases.get(node.get('status'), set()):
            return True
    job = payload.get('job') or {}
    if job.get('recovery_key') and job.get('attempt_id'):
        for current in nodes.values():
            if (current.get('recovery_key') == job['recovery_key'] and current.get('attempt_id')
                    and current['attempt_id'] != job['attempt_id']):
                return True
    return False


def _verified_queued_resource_proposal(session, event, runtime, related):
    """A fresh monitor proposal permits re-evaluation, never scheduling itself."""
    if not isinstance(runtime, dict) or not related:
        return None
    payload = event.get('payload') or {}
    review = (event.get('supervisor_receipt') or {}).get('resource_review') or {}
    if (event.get('kind') != 'job_pending_warning' or event.get('supervisor') != 'delivered'
            or payload.get('requires_user', False) is not False
            or review.get('status') != 'propose_resource_patch' or review.get('readonly') is not True
            or review.get('not_resubmitted') is not True or not review.get('resource_review_id')
            or runtime.get('user_paused') or runtime.get('pause_kind') in {'user_change', 'input_changed'}
            or getattr(session, '_awaiting_plan_approval', False)):
        return None
    now = time.time()
    fresh = lambda value: (not isinstance(value, bool) and isinstance(value, (int, float))
                           and math.isfinite(value) and -5 <= now - value <= 120)
    if not fresh(review.get('reviewed_at')):
        return None
    goal, saved = session.goal_contract.to_dict(), runtime.get('goal_contract') or {}
    scope = ('gases', 'parameters', 'execution_mode', 'execution_authorized', 'approved_plan_version')
    if (any(not isinstance(value, int) or isinstance(value, bool) for value in (goal.get('version'), saved.get('version')))
            or goal['version'] < saved['version']
            or any(goal.get(key) != saved.get(key) for key in scope) or goal.get('execution_authorized') is not True
            or goal.get('execution_mode') != 'workflow' or runtime.get('plan_version') != goal.get('approved_plan_version')
            or (payload.get('goal_contract') or {}).get('version') != goal.get('version')):
        return None
    from .workflow_patch import execution_contract
    approved = goal.get('approved_nodes') or []
    current_nodes = runtime.get('nodes') or {}
    # Current approved contracts, including outputs, are authoritative after native binding.
    if (not approved or len({item.get('step_id') for item in approved}) != len(approved)
            or {item.get('step_id'): execution_contract(item) for item in approved}
            != {key: execution_contract(node.get('contract') or {}) for key, node in current_nodes.items()}):
        return None
    pending_patch = getattr(session, '_pending_workflow_patch', None)
    if pending_patch:
        protected = set(related)
        frontier = list(related)
        while frontier:
            node = current_nodes.get(frontier.pop())
            dependencies = (node.get('contract') or {}).get('depends_on', []) if node else None
            if not isinstance(dependencies, list) or any(not isinstance(dep, str) or dep not in current_nodes for dep in dependencies):
                return None
            for dep in dependencies:
                if dep not in protected:
                    protected.add(dep)
                    frontier.append(dep)
        changes = pending_patch.get('changes')
        if pending_patch.get('base_version') != runtime.get('plan_version') or not isinstance(changes, list) or not changes:
            return None
        for change in changes:
            if (not isinstance(change, dict) or change.get('operation') not in {'upsert', 'remove'}
                    or not isinstance(change.get('step_id'), str)
                    or change['step_id'] not in current_nodes or change['step_id'] in protected):
                return None
            if change['operation'] == 'upsert':
                node = change.get('node')
                if (not isinstance(node, dict) or node.get('step_id') != change['step_id']
                        or not isinstance(node.get('tool'), str) or not isinstance(node.get('arguments'), dict)
                        or not isinstance(node.get('depends_on', []), list)
                        or any(not isinstance(dep, str) or dep not in current_nodes for dep in node.get('depends_on', []))):
                    return None
    evidence_root = getattr(session, '_evidence_root', None)
    if not evidence_root:
        return None
    evidence_root = Path(evidence_root)
    owner = (runtime.get('username'), runtime.get('conv_id'))
    if owner != (evidence_root.parent.parent.name, evidence_root.parent.name):
        return None
    job = payload.get('job') or {}; job_id = str(job.get('job_id') or '')
    if not job_id.isdigit() or (job.get('username'), job.get('conv_id')) != owner:
        return None
    provider = getattr(session, '_chain_jobs', None)
    if not provider:
        return None
    watched = {str(item.get('job_id')): item for item in provider()}
    bound = False
    for step_id in related:
        node = (runtime.get('nodes') or {}).get(step_id) or {}
        approved_node = [item for item in approved if item.get('step_id') == step_id]
        if (node.get('status') != 'waiting_jobs' or not node.get('token') or not node.get('job_ids')
                or len(approved_node) != 1 or execution_contract(node.get('contract') or {}) != execution_contract(approved_node[0])):
            return None
        for value in node['job_ids']:
            original = watched.get(str(value)) or {}
            try: updated = time.mktime(time.strptime(original.get('updated_at', ''), '%Y-%m-%d %H:%M:%S'))
            except (ValueError, TypeError): return None
            if ((original.get('username'), original.get('conv_id')) != owner or original.get('state') != 'PENDING'
                    or original.get('terminal') or original.get('failed') or not -5 <= now - updated <= 30
                    or original.get('tool') != (node.get('contract') or {}).get('tool')
                    or any(node.get(key) and original.get(key) != node.get(key) for key in ('attempt_id', 'recovery_key'))
                    or (original.get('cancel_request') or {}).get('status') in {'intent', 'sent', 'uncertain'}):
                return None
            if str(value) == job_id:
                if any(node.get(key) and job.get(key) != node.get(key) for key in ('attempt_id', 'recovery_key')):
                    return None
                bound = True
    if not bound:
        return None
    proposed = review.get('suggested_resources') or {}
    if (set(proposed) - {'nodelist', 'partition', 'memory_mb', 'num_processes', 'cpus', 'walltime'}
            or not proposed.get('nodelist') or not proposed.get('partition')):
        return None
    from .report_recovery import _OwnedFiles
    from .recovery import result_object
    import re
    calls = review.get('evidence_calls') or []; cited = set(review.get('evidence_call_ids') or [])
    owned = _OwnedFiles(evidence_root.parent, {'username': owner[0], 'conv_id': owner[1]})
    try:
        # Bind the exact outstanding question to native, unexecuted same-job failures.
        pending = getattr(session, '_pending_user_interaction', None) or {}
        pending_params = pending.get('params') or {}
        decision_id = pending_params.get('decision_id')
        if pending.get('tool') != 'user_decision' or not decision_id:
            return None
        references = (getattr(getattr(session, 'memory', None), 'tool_call_log', None) or [])[-128:]
        natives = []
        for reference in references:
            cid = reference.get('call_id')
            if (not isinstance(cid, str) or not cid.isalnum()
                    or reference.get('tool') not in {'request_user_decision', 'retarget_queued_job'}
                    or Path(reference.get('evidence_path') or '') != evidence_root / (cid + '.json')):
                continue
            try: native, _ = owned.read(reference['evidence_path'], as_json=True)
            except (ValueError, PermissionError, OSError): continue
            if (native.get('call_id') == cid and native.get('tool') == reference.get('tool')
                    and native.get('agent') == 'lead-orchestrator' and native.get('failed') is False
                    and isinstance(native.get('time'), (int, float)) and not isinstance(native.get('time'), bool)
                    and math.isfinite(native['time'])):
                natives.append(native)
        decision_params = {key: value for key, value in pending_params.items() if key != 'decision_id'}
        decisions = [native for native in natives if native['tool'] == 'request_user_decision'
            and native.get('params') == decision_params
            and result_object(native.get('result')).get('decision_id') == decision_id
            and result_object(native.get('result')).get('ok') is True
            and result_object(native.get('result')).get('status') == 'needs_user']
        if len(decisions) != 1 or decisions[0]['time'] > event.get('time', 0) + 5:
            return None
        decision_time = decisions[0]['time']
        related_jobs = {str(jid) for key in related for jid in current_nodes[key]['job_ids']}
        if str(pending_params.get('candidate_job_id') or '') not in related_jobs:
            return None
        for related_job in related_jobs:
            attempts = [native for native in natives if native['tool'] == 'retarget_queued_job'
                and str((native.get('params') or {}).get('job_id')) == related_job
                and 0 <= decision_time - native['time'] <= 120]
            if not attempts:
                return None
            result = result_object(max(attempts, key=lambda native: native['time']).get('result'))
            if (result.get('blocked') is not True or result.get('executed') is not False
                    or result.get('requires_user') is not True
                    or result.get('reason') != 'Fresh CPU/RAM/partition facts do not support the requested node; preserve original job'):
                return None
        for reference in calls:
            call_id = reference.get('call_id')
            if (not isinstance(call_id, str) or not call_id.isalnum() or call_id not in cited
                    or reference.get('tool') != 'assess_job_resources'
                    or Path(reference.get('evidence_path') or '') != evidence_root / (call_id + '.json')):
                continue
            try: call, fact = owned.read(reference['evidence_path'], as_json=True)
            except (ValueError, PermissionError, OSError): continue
            assessment = result_object(call.get('result')); constraints = assessment.get('effective_constraints') or {}
            if (call.get('call_id') != call_id or call.get('tool') != 'assess_job_resources' or call.get('agent') != 'monitor'
                    or call.get('failed') is not False or not fresh(call.get('time'))
                    or call.get('time', 0) < event.get('time', 0) - 5
                    or str((call.get('params') or {}).get('job_id')) != job_id
                    or assessment.get('ok') is not True or assessment.get('read_only') is not True
                    or str(assessment.get('job_id')) != job_id or not fresh(assessment.get('observed_at'))
                    or constraints.get('JobState') != 'PENDING' or assessment.get('execution_started') is not False):
                continue
            match = re.search(r'(?:^|,)mem=(\d+)([KMGT]?)', constraints.get('ReqTRES') or '')
            if not match or not str(constraints.get('NumCPUs', '')).isdigit():
                continue
            memory = int(match[1]) * {'K': 1/1024, 'M': 1, 'G': 1024, 'T': 1024**2, '': 1}[match[2]]
            cpus = int(constraints['NumCPUs'])
            if (proposed.get('memory_mb', memory) != memory or proposed.get('num_processes', cpus) != cpus
                    or proposed.get('cpus', cpus) != cpus
                    or proposed.get('walltime', constraints.get('TimeLimit')) != constraints.get('TimeLimit')):
                continue
            for target in assessment.get('compatible_targets') or []:
                if (target.get('nodelist') == proposed['nodelist'] and target.get('partition') == proposed['partition']
                        and target.get('memory_mb') == memory and target.get('cpus', target.get('num_processes')) == cpus
                        and target.get('walltime') == constraints.get('TimeLimit') and fresh(target.get('observed_at'))):
                    return {'job_id': job_id, 'resource_review_id': review['resource_review_id'],
                            'assessment_call_id': call_id, 'native_evidence_sha256': fact['sha256']}
    finally:
        owned.close()
    return None


def current_goal_runtime_event(session, event, runtime=None, *, jobs=None):
    """Original events retain scope only through one native per-node proof."""
    from .retained_prerequisite_scope import event_goal_is_current
    return event_goal_is_current(session, event, runtime or {}, jobs=jobs)


def obsolete_internal_question_for_recovery(session, event, runtime=None):
    """Release an internal technical question only for current bounded recovery."""
    payload = event.get('payload') or {}
    # An executor upgrade authorizes observation only. It cannot answer or
    # withdraw a genuine user decision, even when the related node matches.
    if (event.get('kind') == 'worker_recovery_ready'
            and (payload.get('recovery') or {}).get('action') == 'verify_changed_executor'):
        return False
    pending = getattr(session, '_pending_user_interaction', None) or {}
    if pending.get('tool') not in {'failure_decision', 'workflow_patch_decision', 'user_decision'}:
        return False
    step_id = payload.get('step_id')
    related = set((pending.get('params') or {}).get('related_nodes') or [])
    if not step_id and event.get('kind') == 'job_pending_warning' and isinstance(runtime, dict):
        job_id = str((payload.get('job') or {}).get('job_id') or '')
        matches = [key for key in related if job_id in set(map(str, ((runtime.get('nodes') or {}).get(key) or {}).get('job_ids') or []))]
        if len(matches) == 1:
            step_id = matches[0]
    if not step_id or step_id not in related:
        return False
    legacy = (event.get('kind') == 'worker_recovery_ready'
              and not payload.get('requires_user') and payload.get('recovery', {}).get('tool'))
    resource = _verified_queued_resource_proposal(session, event, runtime, related) if event.get('kind') == 'job_pending_warning' else None
    proof = None
    if not legacy and not resource:
        if (event.get('kind') != 'worker_validation_dossier_ready'
                or payload.get('requires_user', False) is not False or not isinstance(runtime, dict)
                or getattr(session, '_awaiting_plan_approval', False)
                or getattr(session, '_pending_workflow_patch', None)
                or runtime.get('user_paused') or runtime.get('pause_kind') not in {None, 'failure', 'reconciliation'}):
            return False
        from .workflow_patch import execution_contract
        import hashlib
        evidence_root = getattr(session, '_evidence_root', None)
        if not evidence_root:
            return False
        evidence_root = Path(evidence_root)
        owner = (runtime.get('username'), runtime.get('conv_id'))
        if owner != (evidence_root.parent.parent.name, evidence_root.parent.name):
            return False
        runtime_id = hashlib.sha256(json.dumps(list(owner)).encode()).hexdigest()[:24]
        goal = session.goal_contract.to_dict()
        saved_goal = runtime.get('goal_contract') or {}
        scope = ('version', 'approved_plan_version', 'execution_authorized', 'execution_mode',
                 'method', 'gases', 'parameters', 'approved_nodes')
        if (any(goal.get(key) != saved_goal.get(key) for key in scope)
                or goal.get('execution_authorized') is not True or goal.get('execution_mode') != 'workflow'
                or runtime.get('plan_version') != goal.get('approved_plan_version')
                or payload.get('runtime_id') != runtime_id
                or payload.get('plan_version') != runtime.get('plan_version')
                or not current_goal_runtime_event(session,event,runtime)):
            return False
        node = (runtime.get('nodes') or {}).get(step_id) or {}
        dossier = payload.get('validation_dossier') or {}
        current_dossier = node.get('validation_dossier') or {}
        restoration = node.get('restoration') or {}
        proof = (dossier.get('method_facts') or {}).get('native_result_restoration') or {}
        call_id = dossier.get('evidence_call_id')
        approved = [item for item in goal.get('approved_nodes', []) if item.get('step_id') == step_id]
        sha = restoration.get('proof_sha256')
        if (node.get('status') != 'prefinish' or node.get('phase') != 'prefinish'
                or not node.get('token') or payload.get('correlation_token') != node['token']
                or not current_goal_runtime_event(session,event,runtime)
                or node.get('execution_plan_version') != runtime.get('plan_version')
                or node.get('job_ids') or len(approved) != 1
                or execution_contract(approved[0]) != execution_contract(node.get('contract') or {})
                or dossier != current_dossier
                or dossier.get('contract') != node.get('contract') or dossier.get('step_id') != step_id
                or dossier.get('workflow_id') != runtime_id or dossier.get('plan_version') != runtime.get('plan_version')
                or dossier.get('attempt_id') != (node.get('attempt_id') or node['token'])
                or dossier.get('job_ids') != [] or dossier.get('verdict') != 'pass'
                or dossier.get('requires_agent_review') is not True
                or not isinstance(call_id, str) or not call_id.isalnum()
                or dossier.get('evidence_path') != str(evidence_root / (call_id + '.json'))
                or restoration.get('kind') not in {'archived_verified_report_fragment', 'current_returned_evidence'}
                or proof.get('kind') != restoration.get('kind')
                or not isinstance(sha, str) or len(sha) != 64 or any(ch not in '0123456789abcdef' for ch in sha)
                or proof.get('proof_sha256') != sha
                or any(value.get('no_resubmission') is not True or value.get('no_output_write') is not True
                       for value in (restoration, proof))):
            return False
    session.context['obsolete_internal_question'] = {
        'event_id': event.get('event_id'), 'step_id': step_id,
        'reason': ('bounded recovery capability superseded the internal question' if legacy else
                   'fresh same-job resource proposal superseded the infrastructure question' if resource else
                   'verified current restored-result dossier superseded the technical question'),
        'previous_question': pending,
    }
    if resource:
        session.context['obsolete_internal_question'].update(resource)
    if proof:
        session.context['obsolete_internal_question'].update(
            proof_sha256=proof['proof_sha256'], dossier_call_id=payload['validation_dossier']['evidence_call_id'])
    session._pending_user_interaction = None
    session._waiting_for_user_input = False
    session._checkpoint('obsolete_internal_question_for_recovery')
    return True

_event_wakeup=threading.Event()


def wait_for_lifecycle_change(timeout=5):
    changed=_event_wakeup.wait(timeout)
    _event_wakeup.clear()
    return changed


class LifecycleStore:
    def __init__(self, path):
        self.path = Path(path)
        self._last_recovery_check = 0.0
        self._last_heartbeat = 0.0

    def initialize(self):
        with json_transaction(self.path) as data:
            data.setdefault('events', {})
            data.setdefault('agents', {'main_chat': {'agent': 'lead-orchestrator', 'last_action': 'restored'},
                                       'supervisor': {'agent': 'supervisor', 'read_only': True, 'last_action': 'restored'}})

    def emit(self, kind, payload, event_id=None):
        event_id = event_id or uuid.uuid4().hex
        with json_transaction(self.path) as data:
            events = data.setdefault('events', {})
            if event_id not in events:
                events[event_id] = {
                    'event_id': event_id, 'kind': kind, 'time': time.time(),
                    'payload': payload, 'main_chat': 'pending', 'supervisor': 'pending',
                }
            data.setdefault('agents', {
                'main_chat': {'agent': 'lead-orchestrator'},
                'supervisor': {'agent': 'supervisor', 'read_only': True},
            })
        _event_wakeup.set()
        return event_id

    def claim(self, event_id, receiver):
        with json_transaction(self.path) as data:
            event = data.get('events', {}).get(event_id)
            if receiver=='scientific_owner':
                if not event or event.get('kind')!='worker_scientific_owner_review':return False
                event.setdefault(receiver,'pending')
            if not event or event.get(receiver) not in {'pending', 'retry'}:
                return False
            agent = data.setdefault('agents', {}).setdefault(receiver, {})
            if agent.get('busy_event_id'):
                return False
            event[receiver] = 'processing'
            event[receiver + '_claimed_at'] = time.time()
            event[receiver + '_owner_pid'] = os.getpid()
            if receiver=='scientific_owner':event['scientific_owner_process']=_scientific_owner_process()
            agent['busy_event_id'] = event_id
            agent['last_action'] = 'processing'
            return True

    def receipt(self, event_id, receiver, outcome, details=None):
        with json_transaction(self.path) as data:
            event = data.get('events', {}).get(event_id)
            if not event:
                return
            agent = data.get('agents', {}).get(receiver, {})
            if (outcome in {'waiting_user', 'delivered', 'obsolete', 'retry'}
                    and event.get(receiver) == outcome
                    and receiver + '_updated_at' in event
                    and (details is None or event.get(receiver + '_receipt') == details)
                    and agent.get('busy_event_id') != event_id):
                # Polling the same durable decision is not a new delivery.
                # Preserve its original timestamp, a newer event's agent
                # state, and the wakeup state. A real transition, changed
                # details or a claim to clear still commits synchronously.
                return
            event[receiver] = outcome
            if details is not None:
                event[receiver + '_receipt'] = details
            event[receiver + '_updated_at'] = time.time()
            data.setdefault('agents', {}).setdefault(receiver, {}).update({
                'last_event_id': event_id, 'last_action': outcome, 'updated_at': time.time(),
                'details': details,
            })
            if data['agents'][receiver].get('busy_event_id') == event_id:
                data['agents'][receiver].pop('busy_event_id', None)
        if outcome in {'delivered','retry'}:_event_wakeup.set()

    def provisional(self, event_id, details):
        """Immediate rule-based guard receipt while the LLM audit is running."""
        with json_transaction(self.path) as data:
            event = data.get('events', {}).get(event_id)
            if not event:
                return
            event['supervisor_receipt'] = {**details, 'audit_pending': True}
            event['supervisor_provisional_at'] = time.time()
        # The dispatcher already consumed the original emit notification to
        # start supervision. Wake it again after this receipt is durable so
        # main can diagnose promptly while the independent model audit runs.
        _event_wakeup.set()

    def _file_identity(self):
        try:
            value = self.path.stat(follow_symlinks=False)
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(value.st_mode): return None
        return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)

    @staticmethod
    def _active_claim_index(data):
        claims = []
        for key, event in data.get('events', {}).items():
            if not isinstance(event, dict): return None
            for receiver in ('main_chat', 'supervisor', 'scientific_owner'):
                if event.get(receiver) != 'processing': continue
                if not isinstance(key, str) or len(key) > 512 or len(claims) >= 4096: return None
                pid = event.get(receiver + '_owner_pid')
                if pid is not None and type(pid) is not int and not (isinstance(pid, str) and len(pid) <= 32):
                    return None
                pin = event.get('scientific_owner_process')
                if isinstance(pin, dict):
                    pin = {field: pin.get(field) for field in ('hostname', 'pid', 'boot_id', 'starttime')}
                    if any(not isinstance(value, (str, int, type(None))) or
                           isinstance(value, str) and len(value) > 256 for value in pin.values()): return None
                else: pin = None
                claims.append((key, receiver, pid, event.get('kind') == 'worker_scientific_owner_review', pin))
        return tuple(claims)

    @staticmethod
    def _claims_need_recovery(claims):
        if claims is None: return True
        for _key, receiver, pid, scientific_kind, pin in claims:
            if receiver == 'scientific_owner':
                if scientific_kind and _scientific_owner_process_gone(pin): return True
                continue
            try:
                if pid:
                    os.kill(int(pid), 0)
                    continue
            except ProcessLookupError:
                return True
            except PermissionError:
                continue
            except (ValueError, TypeError, OSError):
                return True  # Unknown/malformed inputs go through original fresh checks.
            return True
        return False

    def recover_claims(self):
        """Sparse no-dead-owner filter; any recovery retains the native CAS."""
        identity = self._file_identity()
        cached = getattr(self, '_active_claim_cache', None)
        candidates = None
        if cached is not None and identity == cached[0]:
            candidates = cached[1]
            if not self._claims_need_recovery(candidates): return
        elif identity is not None:
            observed = self.snapshot()
            claims = self._active_claim_index(observed)
            candidates = claims
            linked = self._file_identity()
            if linked == identity and claims is not None:
                self._active_claim_cache = (linked, claims)
                if not self._claims_need_recovery(claims): return
        # Positive/dead/unknown decisions must re-read under the original lock.
        # Nothing in the sparse index can authorize a retry or release a lease.
        now = time.monotonic()
        malformed = candidates is None or any(pid and not (type(pid) is int and 0 < pid < 2**31
            or isinstance(pid,str) and pid.isdigit() and 0 < int(pid) < 2**31)
            for _key,receiver,pid,_kind,_pin in (candidates or ()) if receiver != 'scientific_owner')
        # Known death/new native data goes directly to the original CAS. Keep
        # only malformed legacy inputs rate-limited, never treat them as dead.
        if malformed and now - self._last_recovery_check < 30: return
        self._last_recovery_check = now
        with json_transaction(self.path) as data:
            for event in data.get('events', {}).values():
                for receiver in ('main_chat', 'supervisor', 'scientific_owner'):
                    if event.get(receiver) == 'processing':
                        if receiver=='scientific_owner':
                            if (event.get('kind')!='worker_scientific_owner_review'
                                    or not _scientific_owner_process_gone(event.get('scientific_owner_process'))):continue
                            event[receiver]='retry'
                            agent=data.get('agents',{}).get(receiver,{})
                            if agent.get('busy_event_id')==event['event_id']:agent.pop('busy_event_id',None)
                            continue
                        pid = event.get(receiver + '_owner_pid')
                        try:
                            if pid:
                                os.kill(int(pid), 0)
                                continue
                        except ProcessLookupError:
                            pass
                        except PermissionError:
                            continue
                        event[receiver] = 'retry'
                        data.get('agents', {}).get(receiver, {}).pop('busy_event_id', None)

        # Transaction may publish a changed inode. Derive only sparse metadata
        # from its fresh data and bind it to the linked post-commit file.
        claims = self._active_claim_index(data)
        self._active_claim_cache = (self._file_identity(), claims) if claims is not None else None

    def view_summary(self):
        """Bounded advisory UI headers; never used by claim/control methods."""
        identity = self._file_identity(); cached = getattr(self, '_view_header_cache', None)
        if cached is not None and cached[0] == identity:
            result = copy.deepcopy(cached[1])
        else:
            data = self.snapshot()
            from collections import deque
            fields = ('agent', 'status', 'read_only', 'last_seen', 'last_active_at',
                      'service_heartbeat', 'last_action', 'busy_event_id')
            def header(value, names):
                return {key: (item[:512] if isinstance(item, str) else item)
                        for key in names if isinstance(value, dict) and key in value
                        for item in [value[key]] if item is None or isinstance(item, (str, int, float, bool))}
            result = {'agents': {key: header(value, fields) for key, value in list(data.get('agents', {}).items())[:32]},
                'recent_events': [header(event, ('event_id', 'kind', 'created_at', 'main_chat', 'supervisor'))
                                  for event in deque(data.get('events', {}).values(), maxlen=8)]}
            linked = self._file_identity()
            if linked == identity and len(json.dumps(result, ensure_ascii=False).encode()) <= 65536:
                self._view_header_cache = (linked, copy.deepcopy(result))
        # Liveness is a tiny independent sidecar and must remain current even
        # while the immutable historical-event file stays unchanged.
        for role, metadata in _read_liveness(Path(str(self.path) + '.heartbeat.json'), self.path).items():
            result['agents'].setdefault(role, {}).update(metadata)
        return result

    def snapshot(self):
        if not self.path.exists():
            data = {'agents': {'main_chat': {'agent': 'lead-orchestrator'},
                               'supervisor': {'agent': 'supervisor', 'read_only': True}}, 'events': {}}
        else:
            data = json.loads(self.path.read_text())
        if not isinstance(data, dict):
            raise ValueError('lifecycle mailbox is corrupt')
        for role, metadata in _read_liveness(Path(str(self.path) + '.heartbeat.json'), self.path).items():
            data.setdefault('agents', {}).setdefault(role, {}).update(metadata)
        return data

    def _poll_snapshot(self):
        """Read a mailbox through a pinned regular-file descriptor for caching."""
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        except FileNotFoundError:
            return self.snapshot(), None
        with os.fdopen(fd, 'r', encoding='utf-8') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError('lifecycle mailbox is not a regular file')
            data = json.load(stream)
            after = os.fstat(stream.fileno())
        def identity(value):
            return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns, value.st_mode)
        bound = identity(before) if identity(before) == identity(after) == self._file_identity() else None
        if not isinstance(data, dict):
            raise ValueError('lifecycle mailbox is corrupt')
        for role, metadata in _read_liveness(Path(str(self.path) + '.heartbeat.json'), self.path).items():
            data.setdefault('agents', {}).setdefault(role, {}).update(metadata)
        return data, bound

    @staticmethod
    def _heartbeat_view(observed):
        known = observed.get('events', {})
        agents = observed.get('agents', {})
        heartbeat_agents, heartbeat_events = {}, {}
        for receiver in ('main_chat', 'supervisor'):
            agent = agents.get(receiver, {})
            event_id = agent.get('busy_event_id') if isinstance(agent, dict) else None
            heartbeat_agents[receiver] = {'busy_event_id': event_id}
            if event_id:
                if event_id in known: heartbeat_events[event_id] = known[event_id]
                warning_id = 'role-unresponsive:' + receiver + ':' + str(event_id)
                if warning_id in known: heartbeat_events[warning_id] = known[warning_id]
        return {'agents': heartbeat_agents, 'events': heartbeat_events}

    def _cache_heartbeat(self, identity, observed):
        if identity is None or self._file_identity() != identity: return
        view = self._heartbeat_view(observed)
        try:
            if len(json.dumps(view, ensure_ascii=False).encode('utf-8')) <= 65536:
                self._heartbeat_cache = (identity, copy.deepcopy(view))
        except (TypeError, ValueError, OverflowError):
            pass

    def heartbeat(self):
        """Role service stays present even when its model is idle or waiting."""
        now_mono = time.monotonic()
        # Heartbeats express role liveness, not a scheduler polling cadence.
        # Throttle durable writes while retaining a 30 s heartbeat and the
        # existing 360 s unresponsive threshold.
        if now_mono - self._last_heartbeat < 30:
            return
        identity = self._file_identity()
        cached = getattr(self, '_heartbeat_cache', None)
        if identity is not None and cached and cached[0] == identity:
            observed = copy.deepcopy(cached[1])
        else:
            observed, identity = self._poll_snapshot()
            self._cache_heartbeat(identity, observed)
        overdue = {}
        for receiver in ('main_chat', 'supervisor'):
            event_id = observed.get('agents', {}).get(receiver, {}).get('busy_event_id')
            event = observed.get('events', {}).get(event_id, {})
            # A diagnostic warning is not another business operation to watch.
            # Its normal turn keeps ownership until it returns; do not create
            # a warning about handling a previous warning.
            if _native_role_warning(event_id,event):
                continue
            warning_id = 'role-unresponsive:' + receiver + ':' + str(event_id)
            if (event_id and time.time() - event.get(receiver + '_claimed_at', time.time()) > 360
                    and warning_id not in observed.get('events', {})):
                overdue[receiver] = event_id
        if overdue:
            # Receipt/claim may have changed after the read. Recheck under the
            # authoritative mailbox lock before publishing an actual event.
            with json_transaction(self.path) as data:
                for receiver, expected_id in overdue.items():
                    agent = data.get('agents', {}).get(receiver, {})
                    if agent.get('busy_event_id') != expected_id: continue
                    event = data.get('events', {}).get(expected_id, {})
                    if (_native_role_warning(expected_id,event) or event.get(receiver) != 'processing'
                            or time.time() - event.get(receiver + '_claimed_at', time.time()) <= 360):
                        continue
                    event_id = 'role-unresponsive:' + receiver + ':' + expected_id
                    data.setdefault('events', {}).setdefault(event_id, {
                        'event_id': event_id, 'kind': 'role_unresponsive', 'time': time.time(),
                        'payload': {'receiver': receiver, 'busy_event_id': expected_id, 'requires_user': True},
                        'main_chat': 'pending', 'supervisor': 'pending'})
            _event_wakeup.set()
        _publish_liveness(Path(str(self.path) + '.heartbeat.json'), self.path)
        self._last_heartbeat = now_mono

    def pending(self):
        identity = self._file_identity()
        cached = getattr(self, '_poll_cache', None)
        if identity is not None and cached and cached[0] == identity:
            return copy.deepcopy(cached[1])
        observed, identity = self._poll_snapshot()
        known=observed.get('events',{})
        recursive={key for key,event in known.items()if _recursive_role_warning(key,event,known)}
        if any(known[key].get(receiver)in {'pending','retry','waiting_user'}
                for key in recursive for receiver in ('main_chat','supervisor')):
            # One-time restart migration of already queued recursive warnings.
            # Recheck native child AND parent identities under the mailbox lock;
            # live processing owners must finish normally and are never cleared.
            with json_transaction(self.path)as data:
                current=data.get('events',{})
                for key in recursive:
                    event=current.get(key,{})
                    if not _recursive_role_warning(key,event,current):continue
                    for receiver in ('main_chat','supervisor'):
                        if event.get(receiver)not in {'pending','retry','waiting_user'}:continue
                        event[receiver]='obsolete'
                        event[receiver+'_updated_at']=time.time()
                        event[receiver+'_receipt']={'status':'obsolete_recursive_maintenance',
                            'maintenance_only':True,'parent_event_id':event['payload']['busy_event_id'],
                            'original_event_preserved':True,'no_scientific_action':True}
                observed=data
        known=observed.get('events',{})
        events = [e for key,e in known.items()
                  if not _recursive_role_warning(key,e,known) and (
                      e.get('main_chat') not in {'delivered', 'obsolete'}
                      or e.get('supervisor') not in {'delivered', 'obsolete'})]
        # Current-attempt evidence can release a completed calculation and its
        # dependents, so historical warnings must not starve this event.
        priority = {
            'worker_recovery_ready': 0,
            'worker_failure_recovery': 0,
            'worker_validation_dossier_ready': 1,
        }
        events = sorted(events, key=lambda event: (
            priority.get(event.get('kind'), 2),
            -float(event.get('time', 0) or 0),
        ))
        # Keep only pending events and the two tiny busy-event observations.
        # The cache is advisory: every changed inode, timestamp or size forces
        # a fresh native read, and all claims/receipts still use transactions.
        # A bounded cache prevents historical mailboxes consuming memory.
        heartbeat_view = self._heartbeat_view(observed)
        linked = self._file_identity()
        if identity is not None and linked == identity:
            try:
                size = len(json.dumps([events, heartbeat_view], ensure_ascii=False).encode('utf-8'))
                if size <= 1024 * 1024:
                    self._poll_cache = (identity, copy.deepcopy(events), copy.deepcopy(heartbeat_view))
                    self._cache_heartbeat(identity, observed)
            except (TypeError, ValueError, OverflowError):
                pass
        return events


def _scientific_owner_process():
    import socket
    return {'hostname':socket.gethostname(),'pid':os.getpid(),
        'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        'starttime':Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19]}


def _scientific_owner_process_gone(pin):
    """Unknown/cross-host ownership is never a dead-process proof."""
    import socket
    if (not isinstance(pin,dict) or pin.get('hostname')!=socket.gethostname()
            or type(pin.get('pid')) is not int or not 0<pin['pid']<2**31
            or not isinstance(pin.get('starttime'),str) or not pin['starttime'].isdigit()
            or not isinstance(pin.get('boot_id'),str)):
        return False
    try:uuid.UUID(pin['boot_id'])
    except ValueError:return False
    if pin['boot_id']!=Path('/proc/sys/kernel/random/boot_id').read_text().strip():return True
    try:actual=Path('/proc/'+str(pin['pid'])+'/stat').read_text().rsplit(')',1)[1].split()[19]
    except FileNotFoundError:return True
    except (PermissionError,OSError,IndexError):return False
    return actual!=pin['starttime']
