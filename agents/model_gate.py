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
        return max(1, min(8, int(os.environ.get("BIMEM_MODEL_CALL_CONCURRENCY", "1"))))
    except (TypeError, ValueError):
        return 1


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
    while True:
        _wait_for_circuit(on_wait, interrupted)
        _slots.acquire(interrupted, on_queue)
        try:
            with _state_lock:
                if _cooldown_until > time.monotonic():
                    continue
            try:
                result = call()
            except anthropic.RateLimitError as error:
                with _state_lock:
                    _consecutive_limits += 1
                    wait_seconds = min(60, 15 * _consecutive_limits)
                    _cooldown_until = max(_cooldown_until, time.monotonic() + wait_seconds)
                    attempt = _consecutive_limits
                raise ProviderRateLimited(wait_seconds, attempt) from error
            with _state_lock:
                _consecutive_limits = 0
                _cooldown_until = 0.0
            return result
        finally:
            _slots.release()


def provider_state() -> dict:
    with _state_lock:
        return {
            "concurrency": MODEL_CALL_CONCURRENCY,
            "cooldown_seconds": max(0, math.ceil(_cooldown_until - time.monotonic())),
            "consecutive_limits": _consecutive_limits,
        }
