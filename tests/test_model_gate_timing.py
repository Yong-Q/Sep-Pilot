"""Deterministic admission/provider phase timings; no network/model calls."""
from types import SimpleNamespace
import threading

import pytest

from agents import model_gate


@pytest.fixture
def timed_gate(monkeypatch):
    clock = SimpleNamespace(now=10.0)
    monkeypatch.setattr(model_gate.time, 'monotonic', lambda: clock.now)
    monkeypatch.setattr(model_gate, '_cooldown_until', 0.0)
    monkeypatch.setattr(model_gate, '_consecutive_limits', 0)
    def wait(*a): clock.now += 2
    monkeypatch.setattr(model_gate, '_wait_for_circuit', wait)
    slots = SimpleNamespace(releases=0)
    def acquire(*a): clock.now += 3
    def release(): slots.releases += 1
    slots.acquire, slots.release = acquire, release
    monkeypatch.setattr(model_gate, '_slots', slots)
    return clock, slots


def test_gate_separates_queue_cooldown_and_provider_walltime(timed_gate):
    clock, slots = timed_gate
    def provider(): clock.now += 5; return 'reply'
    assert model_gate.call_model(provider) == 'reply'
    timing = model_gate.last_call_timing()
    assert timing == {'queue_wait_seconds': 3.0, 'cooldown_wait_seconds': 2.0,
        'provider_seconds': 5.0, 'gate_total_seconds': 10.0, 'gate_overhead_seconds': 0.0,
        'provider_calls': 1}
    assert slots.releases == 1
    timing['provider_seconds'] = 999
    assert model_gate.last_call_timing()['provider_seconds'] == 5


def test_provider_failure_records_timings_without_changing_the_exception(timed_gate):
    clock, slots = timed_gate
    failure = ValueError('synthetic failure')
    def provider(): clock.now += 4; raise failure
    with pytest.raises(ValueError) as caught:
        model_gate.call_model(provider)
    assert caught.value is failure
    assert model_gate.last_call_timing()['provider_seconds'] == 4
    assert model_gate.last_call_timing()['gate_total_seconds'] == 9
    assert slots.releases == 1


def test_interrupted_queue_has_no_provider_time_and_does_not_release_unowned_slot(timed_gate):
    clock, slots = timed_gate
    def acquire(*a): clock.now += 3; raise model_gate.ModelGateInterrupted()
    slots.acquire = acquire
    with pytest.raises(model_gate.ModelGateInterrupted):
        model_gate.call_model(lambda: pytest.fail('must not call provider'))
    timing = model_gate.last_call_timing()
    assert timing['queue_wait_seconds'] == 3 and timing['cooldown_wait_seconds'] == 2
    assert timing['provider_seconds'] == 0 and timing['provider_calls'] == 0
    assert slots.releases == 0


def test_timing_is_thread_local(timed_gate):
    model_gate.call_model(lambda: 'main')
    other = []
    def worker(): other.append(model_gate.last_call_timing())
    thread = threading.Thread(target=worker); thread.start(); thread.join(2)
    assert other == [{}]
    assert model_gate.last_call_timing()['provider_calls'] == 1
