import asyncio

import pytest

import thresholds
from gold import QUESTIONS
from relay.config import Config
from relay.eval import percentile
from relay.load import SCENARIOS, run
from relay.standin import running
from relay.tables import RESULT_COLUMNS

pytestmark = pytest.mark.slow

REQUIRED = {"steady", "burst", "overload", "rpm_halved", "daily_exhausted"}
STATUSES = {"ok", "rate_limited", "timeout", "error"}
GRADES = {"correct", "wrong", "refused_correctly", "refused_wrongly"}
SCALED_DURATION_S = 8


@pytest.fixture(scope="module", params=sorted(REQUIRED))
def scenario_run(request):
    scenario = SCENARIOS[request.param].scaled(duration_s=SCALED_DURATION_S)
    with running(scenario.standin_config()) as base_url:
        config = Config.standin(base_url)
        rows = asyncio.run(run(scenario, config, QUESTIONS))
    return scenario, config, rows


def share(rows, predicate):
    return sum(map(predicate, rows)) / len(rows)


def test_all_required_scenarios_are_defined():
    assert REQUIRED <= set(SCENARIOS)


def test_every_request_sent_produces_exactly_one_row(scenario_run):
    scenario, _, rows = scenario_run
    assert len(rows) == len(scenario.arrivals())


def test_rows_follow_the_results_schema(scenario_run):
    _, _, rows = scenario_run
    assert all(list(row) == RESULT_COLUMNS for row in rows)


def test_every_request_ends_in_a_known_status(scenario_run):
    _, _, rows = scenario_run
    assert {row["status"] for row in rows} <= STATUSES


def test_every_request_has_a_known_grade(scenario_run):
    _, _, rows = scenario_run
    assert {row["answer_grade"] for row in rows} <= GRADES


def test_rows_are_marked_as_stand_in(scenario_run):
    _, _, rows = scenario_run
    assert {row["source"] for row in rows} == {"stand-in"}


def test_no_wrong_answers_under_load(scenario_run):
    _, _, rows = scenario_run
    assert share(rows, lambda r: r["answer_grade"] == "wrong") <= thresholds.SYSTEM_MAX_WRONG_RATE


def test_no_request_outlives_its_deadline(scenario_run):
    _, config, rows = scenario_run
    assert max(row["total_ms"] for row in rows) <= (
        config.deadline_ms + thresholds.SYSTEM_DEADLINE_GRACE_MS
    )


def test_first_token_never_precedes_a_request_start(scenario_run):
    _, _, rows = scenario_run
    assert all(row["first_token_ms"] is None or row["first_token_ms"] >= 0 for row in rows)


def test_steady_load_succeeds(scenario_run):
    scenario, _, rows = scenario_run
    if scenario.name != "steady":
        pytest.skip("steady only")
    assert share(rows, lambda r: r["status"] == "ok") >= thresholds.SYSTEM_STEADY_MIN_SUCCESS_RATE


def test_burst_gives_up_on_few_requests(scenario_run):
    scenario, _, rows = scenario_run
    if scenario.name != "burst":
        pytest.skip("burst only")
    assert (
        share(rows, lambda r: r["status"] == "timeout") <= thresholds.SYSTEM_BURST_MAX_GIVEN_UP_RATE
    )


def test_daily_budget_runs_out_during_the_run(scenario_run):
    scenario, _, rows = scenario_run
    if scenario.name != "daily_exhausted":
        pytest.skip("daily_exhausted only")
    assert any(row["status"] == "ok" for row in rows)
    assert any(row["reason"] == "budget_exhausted" for row in rows)


def test_requests_after_the_daily_budget_is_gone_are_answered_quickly(scenario_run):
    scenario, _, rows = scenario_run
    if scenario.name != "daily_exhausted":
        pytest.skip("daily_exhausted only")
    exhausted = [row for row in rows if row["reason"] == "budget_exhausted"]
    gone_at = min(row["start_time"] + row["total_ms"] / 1000 for row in exhausted)
    later = [row["total_ms"] for row in exhausted if row["start_time"] >= gone_at]
    assert later and percentile(later, 99) <= thresholds.SYSTEM_BUDGET_EXHAUSTED_MAX_REPLY_MS
