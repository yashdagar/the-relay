import pandas as pd
import pytest

from gold import ROOT
from relay.tables import RESULT_COLUMNS, latency_table, outcomes_table, retrieval_table

RESULTS = ROOT / "results" / "results.csv"
REQUIRED = {"steady", "burst", "overload", "rpm_halved", "daily_exhausted"}


@pytest.fixture(scope="module")
def results():
    if not RESULTS.exists():
        pytest.skip("no results/results.csv yet")
    return pd.read_csv(RESULTS, keep_default_na=False, na_values=[""])


def test_columns_match_the_schema(results):
    assert list(results.columns) == RESULT_COLUMNS


def test_request_ids_are_unique(results):
    assert results["request_id"].is_unique


def test_statuses_are_from_the_brief(results):
    assert set(results["status"]) <= {"ok", "rate_limited", "timeout", "error"}


def test_every_request_is_graded_with_a_grade_from_the_brief(results):
    grades = set(results["answer_grade"])
    assert grades <= {"correct", "wrong", "refused_correctly", "refused_wrongly"}


def test_every_required_scenario_was_run(results):
    assert REQUIRED <= set(results["scenario"])


def test_both_real_and_stand_in_runs_are_present(results):
    assert set(results["source"]) == {"real", "stand-in"}


def test_at_least_three_retrieval_settings_were_evaluated(results):
    assert results.loc[results["scenario"] == "retrieval", "retrieval_setting"].nunique() >= 3


def test_at_least_fifteen_questions_were_graded(results):
    graded = results[results["scenario"] == "retrieval"]
    assert graded.groupby("retrieval_setting")["request_id"].count().min() >= 15


def test_every_table_can_be_computed_from_the_raw_file(results):
    assert not latency_table(results).empty
    assert not outcomes_table(results, usd_per_mtok_in=0.0, usd_per_mtok_out=0.0).empty
    assert not retrieval_table(results).empty
