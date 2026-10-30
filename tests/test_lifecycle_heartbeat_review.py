"""Independent sidecar overlay and failed-write heartbeat checks."""
import json

import pytest

from agents import lifecycle


def test_valid_liveness_overlay_cannot_modify_claim_receipt_or_event_payload(tmp_path):
    path = tmp_path / 'lifecycle.json'
    store = lifecycle.LifecycleStore(path)
    store.initialize(); store.emit('work', {'measured_value': 42}, event_id='work')
    assert store.claim('work', 'main_chat')
    before = store.snapshot()
    liveness = {'schema_version': 1, 'mailbox': str(path.resolve()), 'agents': {
        'main_chat': {'service_heartbeat': 1234, 'service_owner_pid': 4321,
            'busy_event_id': 'forged', 'last_action': 'delivered', 'read_only': True},
    }, 'events': {'work': {'main_chat': 'delivered', 'payload': {'measured_value': 0}}}}
    (tmp_path / 'lifecycle.json.heartbeat.json').write_text(json.dumps(liveness))
    after = store.snapshot()
    assert after['agents']['main_chat']['service_heartbeat'] == 1234
    assert after['agents']['main_chat']['service_owner_pid'] == 4321
    assert after['agents']['main_chat']['busy_event_id'] == 'work'
    assert after['agents']['main_chat']['last_action'] == 'processing'
    assert after['events'] == before['events']


def test_failed_liveness_write_does_not_advance_throttle_or_rewrite_scientific_mailbox(tmp_path, monkeypatch):
    path = tmp_path / 'lifecycle.json'
    store = lifecycle.LifecycleStore(path)
    store.initialize(); store.emit('fact', {'scientific_result': 'preserve'}, event_id='fact')
    original, calls = lifecycle._publish_liveness, []
    before, timestamp = path.read_bytes(), path.stat().st_mtime_ns
    def failing_once(*args):
        calls.append(1)
        if len(calls) == 1:
            raise OSError('injected liveness write failure')
        return original(*args)
    monkeypatch.setattr(lifecycle, '_publish_liveness', failing_once)
    with pytest.raises(OSError, match='injected liveness'):
        store.heartbeat()
    assert store._last_heartbeat == 0
    store.heartbeat()
    assert len(calls) == 2
    assert store._last_heartbeat > 0
    assert path.read_bytes() == before and path.stat().st_mtime_ns == timestamp
    assert store.snapshot()['events']['fact']['payload']['scientific_result'] == 'preserve'


def test_valid_liveness_does_not_hide_corrupt_authoritative_mailbox(tmp_path):
    path = tmp_path / 'lifecycle.json'
    store = lifecycle.LifecycleStore(path)
    store.initialize(); store.heartbeat()
    path.write_text('corrupt authoritative mailbox')
    with pytest.raises(ValueError):
        store.snapshot()
