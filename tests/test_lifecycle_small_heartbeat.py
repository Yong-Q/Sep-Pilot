"""Service liveness must not rewrite historical scientific event payloads."""
import json
import time

import pytest

from agents.lifecycle import LifecycleStore


def test_heartbeat_persists_small_liveness_without_rewriting_mailbox(tmp_path):
    path = tmp_path / 'lifecycle.json'
    store = LifecycleStore(path)
    store.initialize()
    store.emit('old', {'full_evidence': 'x' * 250000}, event_id='old')
    raw, stamp = path.read_bytes(), path.stat().st_mtime_ns
    store.heartbeat()
    assert path.read_bytes() == raw
    assert path.stat().st_mtime_ns == stamp
    liveness = tmp_path / 'lifecycle.json.heartbeat.json'
    assert liveness.stat().st_size < 2000
    state = LifecycleStore(path).snapshot()
    assert state['agents']['main_chat']['service_heartbeat'] > 0
    assert state['agents']['supervisor']['read_only'] is True
    assert state['events']['old']['payload']['full_evidence'] == 'x' * 250000


def test_overdue_busy_role_still_emits_one_durable_warning(tmp_path):
    path = tmp_path / 'lifecycle.json'
    store = LifecycleStore(path); store.initialize()
    store.emit('work', {}, event_id='work'); assert store.claim('work', 'main_chat')
    raw = json.loads(path.read_text())
    raw['events']['work']['main_chat_claimed_at'] = time.time() - 400
    path.write_text(json.dumps(raw))
    store.heartbeat()
    warning = 'role-unresponsive:main_chat:work'
    assert store.snapshot()['events'][warning]['kind'] == 'role_unresponsive'
    first = path.read_bytes()
    store._last_heartbeat = 0
    store.heartbeat()
    assert path.read_bytes() == first


def test_receipt_during_heartbeat_observation_cannot_generate_stale_warning(tmp_path, monkeypatch):
    path = tmp_path / 'lifecycle.json'
    store = LifecycleStore(path); store.initialize()
    store.emit('work', {}, event_id='work'); store.claim('work', 'main_chat')
    raw = json.loads(path.read_text())
    raw['events']['work']['main_chat_claimed_at'] = time.time() - 400
    path.write_text(json.dumps(raw))
    snapshot = store.snapshot
    poll_snapshot = store._poll_snapshot
    def complete_after_observation():
        observed = poll_snapshot()
        store.receipt('work', 'main_chat', 'delivered')
        return observed
    monkeypatch.setattr(store, '_poll_snapshot', complete_after_observation)
    store.heartbeat()
    assert not any(e['kind'] == 'role_unresponsive' for e in snapshot()['events'].values())


@pytest.mark.parametrize('problem', ['foreign', 'malformed', 'symlink'])
def test_untrusted_liveness_does_not_change_authoritative_events(tmp_path, problem):
    path = tmp_path / 'lifecycle.json'
    store = LifecycleStore(path); store.initialize(); store.emit('work', {}, event_id='work')
    sidecar = tmp_path / 'lifecycle.json.heartbeat.json'
    value = {'mailbox': str(tmp_path / 'other.json'), 'agents': {'main_chat': {
        'service_heartbeat': 99999999999999, 'busy_event_id': 'forged', 'status': 'forged'}},
        'events': {'work': {'main_chat': 'delivered'}}}
    if problem == 'symlink':
        external = tmp_path / 'external.json'
        value['mailbox'] = str(path.resolve())
        external.write_text(json.dumps(value)); sidecar.symlink_to(external)
    else:
        sidecar.write_text('invalid JSON' if problem == 'malformed' else json.dumps(value))
    state = store.snapshot()
    assert state['events']['work']['main_chat'] == 'pending'
    assert state['agents']['main_chat'].get('busy_event_id') is None
    assert state['agents']['main_chat'].get('service_heartbeat') != 99999999999999
