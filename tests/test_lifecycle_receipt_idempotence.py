"""Repeated stable delivery acknowledgements preserve durable bytes and leases."""
import pytest
from agents.lifecycle import LifecycleStore
from agents.state_io import json_transaction


def watch_writes(monkeypatch):
    import agents.state_io as io
    import agents.lifecycle as lifecycle
    calls = []
    original = io.os.replace
    fsync = io.os.fsync
    monkeypatch.setattr(io.os, 'fsync', lambda *a, **kw: calls.append('fsync') or fsync(*a, **kw))
    monkeypatch.setattr(io.os, 'replace', lambda *a, **kw: calls.append('replace') or original(*a, **kw))
    monkeypatch.setattr(lifecycle._event_wakeup, 'set', lambda: calls.append('signal'))
    return calls


@pytest.mark.parametrize('outcome', ['waiting_user', 'delivered', 'obsolete', 'retry'])
def test_repeated_stable_receipt_neither_writes_nor_signals(tmp_path, monkeypatch, outcome):
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event = store.emit('plan_patch_proposed', {'source': 'synthetic'})
    details = {'negotiation_owner': 'lead-orchestrator'}
    store.receipt(event, 'main_chat', outcome, details)
    before = store.path.read_bytes()
    calls = watch_writes(monkeypatch)
    store.receipt(event, 'main_chat', outcome, details)
    assert store.path.read_bytes() == before
    assert calls == []


def test_stable_receipt_without_new_details_keeps_original_receipt(tmp_path, monkeypatch):
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event = store.emit('case', {})
    store.receipt(event, 'main_chat', 'waiting_user', {'reason': 'original'})
    before = store.path.read_bytes()
    calls = watch_writes(monkeypatch)
    store.receipt(event, 'main_chat', 'waiting_user')
    assert store.path.read_bytes() == before
    assert calls == []


@pytest.mark.parametrize('change', ['details', 'outcome', 'owned_lease', 'first_receipt'])
def test_real_receipt_transition_still_persists_and_clears_only_own_lease(tmp_path, monkeypatch, change):
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event = store.emit('case', {})
    store.receipt(event, 'main_chat', 'waiting_user', {'reason': 'original'})
    with json_transaction(store.path) as data:
        if change == 'owned_lease':
            data['agents']['main_chat']['busy_event_id'] = event
        else:
            data['agents']['main_chat']['busy_event_id'] = 'other-event'
        if change == 'first_receipt':
            data['events'][event].pop('main_chat_updated_at')
    calls = watch_writes(monkeypatch)
    outcome = 'delivered' if change == 'outcome' else 'waiting_user'
    details = {'reason': 'different' if change == 'details' else 'original'}
    store.receipt(event, 'main_chat', outcome, details)
    assert calls.count('replace') == 1
    state = store.snapshot()
    assert state['events'][event]['main_chat_receipt'] == details
    assert 'main_chat_updated_at' in state['events'][event]
    assert state['agents']['main_chat'].get('busy_event_id') == (None if change == 'owned_lease' else 'other-event')
    assert calls.count('signal') == (1 if outcome == 'delivered' else 0)


def test_repeated_stable_receipt_does_not_replace_newer_other_event_agent_state(tmp_path, monkeypatch):
    store = LifecycleStore(tmp_path / 'mailbox.json')
    first = store.emit('first', {})
    second = store.emit('second', {})
    store.receipt(first, 'main_chat', 'waiting_user', {'reason': 'first'})
    store.receipt(second, 'main_chat', 'delivered', {'reason': 'newer'})
    before = store.path.read_bytes()
    calls = watch_writes(monkeypatch)
    store.receipt(first, 'main_chat', 'waiting_user', {'reason': 'first'})
    assert store.path.read_bytes() == before
    assert calls == []


def test_processing_receipts_keep_their_existing_refresh_semantics(tmp_path, monkeypatch):
    store = LifecycleStore(tmp_path / 'mailbox.json')
    event = store.emit('case', {})
    store.receipt(event, 'main_chat', 'processing', {'reason': 'working'})
    calls = watch_writes(monkeypatch)
    store.receipt(event, 'main_chat', 'processing', {'reason': 'working'})
    assert calls.count('replace') == 1
