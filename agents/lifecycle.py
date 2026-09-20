"""Durable lifecycle mailbox for main chat and the read-only supervisor.

Events carry the executable workflow/node detail, not a reconstructed prose
plan. One event is acknowledged independently by each long-lived agent.
"""
import json
import os
from pathlib import Path
import time
import uuid
import threading

from .state_io import json_transaction


def obsolete_internal_question_for_recovery(session, event):
    """Release an obsolete internal question when a bounded repair now exists."""
    if (event.get('kind') != 'worker_recovery_ready'
            or event.get('payload', {}).get('requires_user')
            or not event.get('payload', {}).get('recovery', {}).get('tool')):
        return False
    pending = getattr(session, '_pending_user_interaction', None) or {}
    if pending.get('tool') not in {'failure_decision', 'workflow_patch_decision', 'user_decision'}:
        return False
    step_id = event.get('payload', {}).get('step_id')
    related = set((pending.get('params') or {}).get('related_nodes') or [])
    if not step_id or step_id not in related:
        return False
    session.context['obsolete_internal_question'] = {
        'event_id': event.get('event_id'),
        'step_id': step_id,
        'reason': 'bounded recovery capability superseded the internal question',
    }
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
            if not event or event.get(receiver) not in {'pending', 'retry'}:
                return False
            agent = data.setdefault('agents', {}).setdefault(receiver, {})
            if agent.get('busy_event_id'):
                return False
            event[receiver] = 'processing'
            event[receiver + '_claimed_at'] = time.time()
            event[receiver + '_owner_pid'] = os.getpid()
            agent['busy_event_id'] = event_id
            agent['last_action'] = 'processing'
            return True

    def receipt(self, event_id, receiver, outcome, details=None):
        with json_transaction(self.path) as data:
            event = data.get('events', {}).get(event_id)
            if not event:
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
            if event:
                event['supervisor_receipt'] = {**details, 'audit_pending': True}

    def recover_claims(self):
        """Only called when restoring after process restart, never mid-turn."""
        now = time.monotonic()
        # The API dispatcher may inspect every restored conversation.  A claim
        # owner cannot become a dead *different process* several times per
        # second, so repeated fsync transactions add no correctness.  A new
        # LifecycleStore after restart starts at zero and always checks once.
        if now - self._last_recovery_check < 30:
            return
        self._last_recovery_check = now
        with json_transaction(self.path) as data:
            for event in data.get('events', {}).values():
                for receiver in ('main_chat', 'supervisor'):
                    if event.get(receiver) == 'processing':
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

    def snapshot(self):
        if not self.path.exists():
            return {'agents': {'main_chat': {'agent': 'lead-orchestrator'},
                               'supervisor': {'agent': 'supervisor', 'read_only': True}}, 'events': {}}
        data = json.loads(self.path.read_text())
        if not isinstance(data, dict):
            raise ValueError('lifecycle mailbox is corrupt')
        return data

    def heartbeat(self):
        """Role service stays present even when its model is idle or waiting."""
        now_mono = time.monotonic()
        # Heartbeats express role liveness, not a scheduler polling cadence.
        # Throttle durable writes while retaining a 30 s heartbeat and the
        # existing 360 s unresponsive threshold.
        if now_mono - self._last_heartbeat < 30:
            return
        self._last_heartbeat = now_mono
        with json_transaction(self.path) as data:
            for receiver, name in (('main_chat', 'lead-orchestrator'), ('supervisor', 'supervisor')):
                agent = data.setdefault('agents', {}).setdefault(receiver, {})
                agent.update(agent=name, service_heartbeat=time.time(), service_owner_pid=os.getpid(), lifetime=True)
                event = data.get('events', {}).get(agent.get('busy_event_id'), {})
                started = event.get(receiver + '_claimed_at', time.time())
                if time.time() - started > 360:
                    event_id = 'role-unresponsive:' + receiver + ':' + event['event_id']
                    data.setdefault('events', {}).setdefault(event_id, {
                        'event_id': event_id, 'kind': 'role_unresponsive', 'time': time.time(),
                        'payload': {'receiver': receiver, 'busy_event_id': event['event_id'], 'requires_user': True},
                        'main_chat': 'pending', 'supervisor': 'pending'})

    def pending(self):
        events = [e for e in self.snapshot().get('events', {}).values()
                  if e.get('main_chat') not in {'delivered', 'obsolete'}
                  or e.get('supervisor') not in {'delivered', 'obsolete'}]
        # Current-attempt evidence can release a completed calculation and its
        # dependents, so historical warnings must not starve this event.
        priority = {
            'worker_recovery_ready': 0,
            'worker_validation_dossier_ready': 1,
        }
        return sorted(events, key=lambda event: (
            priority.get(event.get('kind'), 2),
            -float(event.get('time', 0) or 0),
        ))
