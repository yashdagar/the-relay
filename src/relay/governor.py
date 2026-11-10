import asyncio
import json
import math
import re
import time
from collections.abc import Callable

from relay.config import ModelLimits

LIMIT_CODES = {
    "requests per minute": "rpm",
    "tokens per minute": "tpm",
    "requests per day": "rpd",
    "tokens per day": "tpd",
}
DAILY = {"rpd", "tpd"}
DURATION_PART = re.compile(r"([\d.]+)(ms|h|m|s)")
UNIT_S = {"ms": 0.001, "s": 1, "m": 60, "h": 3600}


def parse_duration_s(text: str) -> float | None:
    parts = DURATION_PART.findall(text)
    if not parts:
        return None
    return sum(float(value) * UNIT_S[unit] for value, unit in parts)


def parse_rate_limit(body: str) -> tuple[str | None, float | None]:
    try:
        message = json.loads(body)["error"]["message"]
    except (ValueError, KeyError, TypeError):
        return None, None
    limit = next((code for text, code in LIMIT_CODES.items() if text in message), None)
    wait = re.search(r"try again in ([0-9hms.]+?)\.?(?:\s|$)", message)
    if wait is None:
        return limit, math.inf if "a day" in message else None
    return limit, parse_duration_s(wait.group(1))


class _Bucket:
    def __init__(self, capacity: float, period_s: float, now: float):
        self.capacity = capacity
        self.rate = capacity / period_s
        self.level = capacity
        self.last = now

    def refill(self, now: float) -> None:
        self.level = min(self.capacity, self.level + (now - self.last) * self.rate)
        self.last = now

    def wait_s(self, amount: float, now: float) -> float:
        self.refill(now)
        return 0.0 if self.level >= amount else (amount - self.level) / self.rate


class Governor:
    def __init__(
        self,
        limits: dict[str, ModelLimits],
        *,
        minute_s: float = 60.0,
        day_s: float = 86_400.0,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.clock = clock
        now = clock()
        self.buckets = {
            model: {
                "rpm": _Bucket(lim.requests_per_minute, minute_s, now),
                "tpm": _Bucket(lim.tokens_per_minute, minute_s, now),
                "rpd": _Bucket(lim.requests_per_day, day_s, now),
                "tpd": _Bucket(lim.tokens_per_day, day_s, now),
            }
            for model, lim in limits.items()
        }
        self.blocked_until: dict[str, float] = {}
        self.exhausted_until: dict[str, float] = {}
        self.waiting = 0

    def _amounts(self, tokens: int) -> dict[str, float]:
        return {"rpm": 1, "tpm": tokens, "rpd": 1, "tpd": tokens}

    def wait_s(self, model: str, tokens: int) -> float:
        now = self.clock()
        blocked = max(self.blocked_until.get(model, now), self.exhausted_until.get(model, now))
        if model not in self.buckets:
            return max(0.0, blocked - now)
        amounts = self._amounts(tokens)
        waits = [b.wait_s(amounts[n], now) for n, b in self.buckets[model].items()]
        return max(blocked - now, *waits)

    def reserve(self, model: str, tokens: int) -> None:
        if model not in self.buckets:
            return
        for name, amount in self._amounts(tokens).items():
            self.buckets[model][name].level -= amount

    def daily_exhausted(self, model: str) -> bool:
        return self.exhausted_until.get(model, -math.inf) > self.clock()

    def exhausted_for_s(self, model: str) -> float:
        return max(0.0, self.exhausted_until.get(model, 0.0) - self.clock())

    def observe(self, model: str, headers: dict[str, str]) -> None:
        if model not in self.buckets:
            return
        now = self.clock()
        for header, name in (
            ("x-ratelimit-remaining-tokens", "tpm"),
            ("x-ratelimit-remaining-requests", "rpd"),
        ):
            if header in headers:
                bucket = self.buckets[model][name]
                bucket.refill(now)
                bucket.level = min(bucket.level, float(headers[header]))

    def penalize(self, model: str, limit: str | None, wait_s: float) -> None:
        until = self.clock() + wait_s
        if limit in DAILY:
            self.exhausted_until[model] = until
        else:
            self.blocked_until[model] = max(self.blocked_until.get(model, 0.0), until)

    async def acquire(self, model: str, tokens: int, deadline: float) -> bool:
        self.waiting += 1
        try:
            while True:
                wait = self.wait_s(model, tokens)
                if wait == 0:
                    self.reserve(model, tokens)
                    return True
                if self.clock() + wait > deadline:
                    return False
                await asyncio.sleep(wait)
        finally:
            self.waiting -= 1
