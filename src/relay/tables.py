import numpy as np
import pandas as pd

from relay.eval import percentile

RESULT_COLUMNS = [
    "request_id",
    "scenario",
    "source",
    "start_time",
    "first_token_ms",
    "total_ms",
    "status",
    "retries",
    "tokens_in",
    "tokens_out",
    "retrieval_setting",
    "answer_grade",
    "queue_depth",
    "reason",
]

SCENARIO_KEY = ["scenario", "source"]


def _percentiles(values: pd.Series, ps: tuple[int, ...], prefix: str) -> dict[str, float]:
    values = values.dropna()
    return {f"{prefix}_p{p}_ms": percentile(values, p) if len(values) else np.nan for p in ps}


def _pct(mask: pd.Series) -> float:
    return round(100 * mask.mean(), 1)


def _tokens(group: pd.DataFrame) -> pd.Series:
    return group["tokens_in"] + group["tokens_out"]


def _latency_row(group: pd.DataFrame) -> pd.Series:
    starts = group["start_time"]
    finishes = starts + group["total_ms"] / 1000
    duration_s = finishes.max() - starts.min()
    arrival_window_s = (starts.max() - starts.min()) or duration_s
    return pd.Series(
        {
            "requests_sent": len(group),
            "duration_s": duration_s,
            "offered_rate_per_min": len(group) / arrival_window_s * 60,
            **_percentiles(group["first_token_ms"], (50, 95, 99), "first_token"),
            **_percentiles(group["total_ms"], (50, 95, 99), "total"),
        }
    )


def _outcome_row(group: pd.DataFrame, usd_per_mtok_in: float, usd_per_mtok_out: float):
    ok = group[group["status"] == "ok"]
    usd = (ok["tokens_in"] * usd_per_mtok_in + ok["tokens_out"] * usd_per_mtok_out) / 1e6
    return pd.Series(
        {
            "succeeded_pct": _pct(group["status"] == "ok"),
            "rate_limited_pct": _pct(group["status"] == "rate_limited"),
            "timed_out_pct": _pct(group["status"] == "timeout"),
            "error_pct": _pct(group["status"] == "error"),
            "retries_per_request": group["retries"].mean(),
            "peak_queue_length": group["queue_depth"].max(),
            "tokens_per_answer": _tokens(ok).mean(),
            "usd_per_1000_answers": usd.mean() * 1000,
        }
    )


def _retrieval_row(group: pd.DataFrame) -> pd.Series:
    grades = group["answer_grade"]
    return pd.Series(
        {
            "questions_asked": len(group),
            "correct_pct": _pct(grades == "correct"),
            "wrong_pct": _pct(grades == "wrong"),
            "refused_correctly_pct": _pct(grades == "refused_correctly"),
            "refused_wrongly_pct": _pct(grades == "refused_wrongly"),
            "tokens_per_answer": _tokens(group[group["status"] == "ok"]).mean(),
            **_percentiles(group["first_token_ms"], (50, 95), "first_token"),
            **_percentiles(group["total_ms"], (50, 95), "total"),
        }
    )


def latency_table(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(SCENARIO_KEY).apply(_latency_row, include_groups=False)


def outcomes_table(
    results: pd.DataFrame, *, usd_per_mtok_in: float, usd_per_mtok_out: float
) -> pd.DataFrame:
    return results.groupby(SCENARIO_KEY).apply(
        _outcome_row, usd_per_mtok_in, usd_per_mtok_out, include_groups=False
    )


def retrieval_table(results: pd.DataFrame) -> pd.DataFrame:
    graded = results[results["scenario"] == "retrieval"]
    return graded.groupby("retrieval_setting").apply(_retrieval_row, include_groups=False)
