import pandas as pd
import pytest

from relay.tables import RESULT_COLUMNS, latency_table, outcomes_table, retrieval_table


def row(i, scenario="burst", source="stand-in", **overrides):
    base = {
        "request_id": f"r{i}",
        "scenario": scenario,
        "source": source,
        "start_time": 1_700_000_000.0 + i,
        "first_token_ms": 1000.0 * (i + 1),
        "total_ms": 2000.0 * (i + 1),
        "status": "ok",
        "retries": 0,
        "tokens_in": 1000,
        "tokens_out": 200,
        "retrieval_setting": "300w-k5-hybrid",
        "answer_grade": "correct",
        "queue_depth": i,
        "reason": "",
    }
    return {**base, **overrides}


@pytest.fixture
def results():
    rows = [row(i) for i in range(4)]
    rows[2].update(
        status="rate_limited", first_token_ms=None, retries=2, answer_grade="refused_wrongly"
    )
    rows[3].update(status="timeout", first_token_ms=None, retries=1, answer_grade="refused_wrongly")
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def test_result_columns_start_with_the_brief_schema_in_order():
    brief = [
        "request_id", "scenario", "source", "start_time", "first_token_ms", "total_ms",
        "status", "retries", "tokens_in", "tokens_out", "retrieval_setting", "answer_grade",
    ]  # fmt: skip
    assert RESULT_COLUMNS[: len(brief)] == brief


def test_latency_row_counts_every_request_sent(results):
    assert latency_table(results).loc[("burst", "stand-in"), "requests_sent"] == 4


def test_first_token_percentiles_ignore_requests_that_never_got_a_token(results):
    assert latency_table(results).loc[("burst", "stand-in"), "first_token_p50_ms"] == 1500.0


def test_total_time_percentiles_include_failed_requests(results):
    assert latency_table(results).loc[("burst", "stand-in"), "total_p50_ms"] == 5000.0


def test_duration_runs_from_first_start_to_last_finish(results):
    assert latency_table(results).loc[("burst", "stand-in"), "duration_s"] == pytest.approx(3 + 8.0)


def test_offered_rate_is_requests_per_minute_over_the_arrival_window(results):
    assert latency_table(results).loc[
        ("burst", "stand-in"), "offered_rate_per_min"
    ] == pytest.approx(4 / 3 * 60)


def test_outcome_percentages_have_one_decimal(results):
    outcomes = outcomes_table(results, usd_per_mtok_in=0.0, usd_per_mtok_out=0.0)
    assert outcomes.loc[
        ("burst", "stand-in"), ["succeeded_pct", "rate_limited_pct", "timed_out_pct"]
    ].tolist() == [
        50.0,
        25.0,
        25.0,
    ]


def test_retries_per_request_averages_over_all_requests(results):
    outcomes = outcomes_table(results, usd_per_mtok_in=0.0, usd_per_mtok_out=0.0)
    assert outcomes.loc[("burst", "stand-in"), "retries_per_request"] == pytest.approx(0.75)


def test_peak_queue_length_is_the_largest_queue_depth_seen(results):
    outcomes = outcomes_table(results, usd_per_mtok_in=0.0, usd_per_mtok_out=0.0)
    assert outcomes.loc[("burst", "stand-in"), "peak_queue_length"] == 3


def test_tokens_per_answer_counts_only_successful_requests(results):
    outcomes = outcomes_table(results, usd_per_mtok_in=0.0, usd_per_mtok_out=0.0)
    assert outcomes.loc[("burst", "stand-in"), "tokens_per_answer"] == 1200


def test_cost_per_thousand_answers_uses_token_prices(results):
    outcomes = outcomes_table(results, usd_per_mtok_in=0.5, usd_per_mtok_out=1.0)
    assert outcomes.loc[("burst", "stand-in"), "usd_per_1000_answers"] == pytest.approx(0.7)


def test_each_scenario_and_source_pair_gets_its_own_row():
    rows = [row(0, scenario="steady", source="real"), row(1, scenario="steady", source="stand-in")]
    table = latency_table(pd.DataFrame(rows, columns=RESULT_COLUMNS))
    assert len(table) == 2


def test_retrieval_table_reports_grade_shares_per_setting():
    grades = ["correct", "correct", "wrong", "refused_correctly"]
    rows = [row(i, scenario="retrieval", answer_grade=g) for i, g in enumerate(grades)]
    table = retrieval_table(pd.DataFrame(rows, columns=RESULT_COLUMNS))
    assert table.loc[
        "300w-k5-hybrid", ["correct_pct", "wrong_pct", "refused_correctly_pct"]
    ].tolist() == [
        50.0,
        25.0,
        25.0,
    ]
