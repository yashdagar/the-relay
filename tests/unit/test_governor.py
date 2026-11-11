import json
import math

import pytest

from relay.config import ModelLimits
from relay.governor import Governor, parse_duration_s, parse_rate_limit

MODEL = "m"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


@pytest.fixture
def clock():
    return Clock()


def governor(clock, **limits):
    base = {"requests_per_minute": 30, "tokens_per_minute": 8000, "requests_per_day": 1000}
    return Governor({MODEL: ModelLimits(**{**base, **limits})}, clock=clock)


def test_request_within_limits_is_admitted_immediately(clock):
    assert governor(clock).wait_s(MODEL, tokens=2000) == 0


def test_tokens_per_minute_refill_continuously(clock):
    gov = governor(clock)
    gov.reserve(MODEL, tokens=8000)
    assert gov.wait_s(MODEL, tokens=1000) == pytest.approx(1000 / (8000 / 60))


def test_reserving_spends_one_request_and_the_tokens(clock):
    gov = governor(clock, requests_per_minute=1)
    gov.reserve(MODEL, tokens=10)
    assert gov.wait_s(MODEL, tokens=10) == pytest.approx(60)


def test_response_headers_lower_the_local_estimate(clock):
    gov = governor(clock)
    gov.observe(MODEL, {"x-ratelimit-remaining-tokens": "500"})
    assert gov.wait_s(MODEL, tokens=600) > 0


def test_response_headers_never_raise_the_local_estimate(clock):
    gov = governor(clock)
    gov.reserve(MODEL, tokens=8000)
    gov.observe(MODEL, {"x-ratelimit-remaining-tokens": "8000"})
    assert gov.wait_s(MODEL, tokens=1000) > 0


def test_rate_limit_blocks_the_model_for_the_stated_wait(clock):
    gov = governor(clock)
    gov.penalize(MODEL, limit="tpm", wait_s=0.5)
    assert gov.wait_s(MODEL, tokens=1) == pytest.approx(0.5)


def test_per_day_rate_limit_marks_the_model_exhausted(clock):
    gov = governor(clock)
    gov.penalize(MODEL, limit="rpd", wait_s=math.inf)
    assert gov.daily_exhausted(MODEL)


def test_exhausted_daily_budget_comes_back_one_request_at_a_time(clock):
    gov = governor(clock, requests_per_day=1000)
    gov.penalize(MODEL, limit="rpd", wait_s=86.4)
    clock.now += 86.4
    assert not gov.daily_exhausted(MODEL)


def test_unknown_model_is_never_throttled_locally(clock):
    assert governor(clock).wait_s("other", tokens=10**6) == 0


def groq_429(limit: str, wait: str) -> str:
    message = (
        f"Rate limit reached on {limit}: Limit 1, Used 1. Please try again in {wait}. Need more"
    )
    return json.dumps({"error": {"message": message, "type": "tokens"}})


def test_groq_millisecond_wait_is_parsed():
    body = groq_429("tokens per minute (TPM)", "82.499999ms")
    assert parse_rate_limit(body) == ("tpm", pytest.approx(0.0825))


def test_groq_requests_per_minute_wait_is_parsed():
    assert parse_rate_limit(groq_429("requests per minute (RPM)", "2s")) == ("rpm", 2.0)


def test_per_day_limit_with_minutes_is_parsed():
    body = groq_429("tokens per day (TPD)", "7m12.5s")
    assert parse_rate_limit(body) == ("tpd", pytest.approx(432.5))


def test_unparseable_body_falls_back_to_unknown():
    assert parse_rate_limit("not json") == (None, None)


def test_reset_header_durations_are_parsed():
    assert parse_duration_s("1m26.4s") == pytest.approx(86.4)
    assert parse_duration_s("622ms") == pytest.approx(0.622)
    assert parse_duration_s("28m48s") == pytest.approx(1728)


def test_spending_the_local_daily_budget_marks_the_model_exhausted(clock):
    gov = governor(clock, requests_per_day=2)
    gov.reserve(MODEL, tokens=10)
    gov.reserve(MODEL, tokens=10)
    assert gov.daily_exhausted(MODEL)


def test_exhaustion_from_local_budget_reports_when_one_request_returns(clock):
    gov = governor(clock, requests_per_day=1000)
    for _ in range(1000):
        gov.reserve(MODEL, tokens=1)
    assert gov.exhausted_for_s(MODEL) == pytest.approx(86.4)
