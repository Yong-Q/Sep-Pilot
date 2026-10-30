"""Fair process-wide gate and circuit breaker for model-provider calls."""
from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from typing import Callable, TypeVar

import anthropic


T = TypeVar("T")


def _configured_concurrency() -> int:
    try:
        return max(1, min(8, int(os.environ.get("BIMEM_MODEL_CALL_CONCURRENCY", "4"))))
    except (TypeError, ValueError):
        return 4


MODEL_CALL_CONCURRENCY = _configured_concurrency()
class FairSlots:
    """FIFO admission with cancellable waiters and bounded parallel calls."""
    def __init__(self, capacity):
        self.capacity = capacity
        self.active = 0
        self.waiters = deque()
        self.condition = threading.Condition()

    def acquire(self, interrupted=None, on_queue=None):
        ticket = object()
        with self.condition:
            self.waiters.append(ticket)
            try:
                if on_queue and (self.active >= self.capacity or len(self.waiters) > 1):
                    on_queue(len(self.waiters))
                while True:
                    if interrupted and interrupted():
                        raise ModelGateInterrupted()
                    if self.waiters[0] is ticket and self.active < self.capacity:
                        self.waiters.popleft()
                        self.active += 1
                        self.condition.notify_all()
                        return
                    self.condition.wait(.1)
            finally:
                if ticket in self.waiters:
                    self.waiters.remove(ticket)
                    self.condition.notify_all()

    def release(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()


_slots = FairSlots(MODEL_CALL_CONCURRENCY)
_state_lock = threading.Lock()
_cooldown_until = 0.0
_consecutive_limits = 0
_timing_local = threading.local()


def last_call_timing() -> dict:
    """Content-free phases for this thread's most recent gate invocation.

    Provider time is callback wall time (including transport/stream parsing),
    not remote compute time or time to first token. Other threads see no data.
    """
    return dict(getattr(_timing_local, 'last', {}))


class ProviderRateLimited(Exception):
    def __init__(self, wait_seconds: int, attempt: int):
        super().__init__(f"429 Too many requests; shared cooldown {wait_seconds}s")
        self.wait_seconds = wait_seconds
        self.attempt = attempt


class ModelGateInterrupted(Exception):
    pass


def _wait_for_circuit(on_wait: Callable[[int, int], None] | None,
                      interrupted: Callable[[], bool] | None) -> None:
    last_reported = None
    while True:
        if interrupted and interrupted():
            raise ModelGateInterrupted()
        with _state_lock:
            remaining = max(0, math.ceil(_cooldown_until - time.monotonic()))
            attempt = _consecutive_limits
        if remaining <= 0:
            return
        if on_wait and remaining != last_reported:
            on_wait(remaining, attempt)
            last_reported = remaining
        time.sleep(min(1, remaining))


def call_model(call: Callable[[], T], *,
               on_wait: Callable[[int, int], None] | None = None,
               on_queue=None,
               interrupted: Callable[[], bool] | None = None) -> T:
    """Run one provider call behind a rechecked slot and shared circuit."""
    global _cooldown_until, _consecutive_limits
    started = time.monotonic()
    queue_seconds = cooldown_seconds = provider_seconds = 0.0
    provider_calls = 0
    _timing_local.last = {}
    try:
        while True:
            phase = time.monotonic()
            try:
                _wait_for_circuit(on_wait, interrupted)
            finally:
                cooldown_seconds += time.monotonic() - phase
            phase = time.monotonic()
            try:
                _slots.acquire(interrupted, on_queue)
            finally:
                queue_seconds += time.monotonic() - phase
            try:
                with _state_lock:
                    if _cooldown_until > time.monotonic():
                        continue
                phase = time.monotonic()
                provider_calls += 1
                try:
                    result = call()
                except anthropic.RateLimitError as error:
                    with _state_lock:
                        _consecutive_limits += 1
                        wait_seconds = min(60, 15 * _consecutive_limits)
                        _cooldown_until = max(_cooldown_until, time.monotonic() + wait_seconds)
                        attempt = _consecutive_limits
                    raise ProviderRateLimited(wait_seconds, attempt) from error
                finally:
                    provider_seconds += time.monotonic() - phase
                with _state_lock:
                    _consecutive_limits = 0
                    _cooldown_until = 0.0
                return result
            finally:
                _slots.release()
    finally:
        total = time.monotonic() - started
        _timing_local.last = {
            'queue_wait_seconds': round(queue_seconds, 6),
            'cooldown_wait_seconds': round(cooldown_seconds, 6),
            'provider_seconds': round(provider_seconds, 6),
            'gate_total_seconds': round(total, 6),
            'gate_overhead_seconds': round(max(0.0, total - queue_seconds - cooldown_seconds - provider_seconds), 6),
            'provider_calls': provider_calls,
        }


def provider_state() -> dict:
    with _state_lock:
        return {
            "concurrency": MODEL_CALL_CONCURRENCY,
            "cooldown_seconds": max(0, math.ceil(_cooldown_until - time.monotonic())),
            "consecutive_limits": _consecutive_limits,
        }
