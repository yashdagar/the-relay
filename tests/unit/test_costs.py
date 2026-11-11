import pytest

from relay.records import request_usd


def step(model, tokens_in, tokens_out):
    return {"model": model, "tokens_in": tokens_in, "tokens_out": tokens_out}


def test_request_cost_prices_each_step_at_its_own_model():
    steps = [step("openai/gpt-oss-120b", 1_000_000, 0), step("qwen/qwen3.8-27b", 0, 1_000_000)]
    assert request_usd(steps) == pytest.approx(0.15 + 4.00)


def test_failed_attempts_are_still_paid_for():
    steps = [step("openai/gpt-oss-20b", 1_000_000, 0), step("openai/gpt-oss-20b", 1_000_000, 0)]
    assert request_usd(steps) == pytest.approx(0.15)


def test_unknown_model_costs_nothing():
    assert request_usd([step("stand-in-model", 10**6, 10**6)]) == 0.0
