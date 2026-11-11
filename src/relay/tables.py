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
    "usd",
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


def _outcome_row(group: pd.DataFrame) -> pd.Series:
    ok = group[group["status"] == "ok"]
    return pd.Series(
        {
            "succeeded_pct": _pct(group["status"] == "ok"),
            "rate_limited_pct": _pct(group["status"] == "rate_limited"),
            "timed_out_pct": _pct(group["status"] == "timeout"),
            "error_pct": _pct(group["status"] == "error"),
            "retries_per_request": group["retries"].mean(),
            "peak_queue_length": group["queue_depth"].max(),
            "tokens_per_answer": _tokens(ok).mean(),
            "usd_per_1000_answers": ok["usd"].mean() * 1000,
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


def outcomes_table(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby(SCENARIO_KEY).apply(_outcome_row, include_groups=False)


def retrieval_table(results: pd.DataFrame) -> pd.DataFrame:
    graded = results[results["scenario"] == "retrieval"]
    return graded.groupby("retrieval_setting").apply(_retrieval_row, include_groups=False)


SCENARIO_LABELS = {
    "steady": "Steady, 2/min for 30 min",
    "burst": "Burst, 20 requests in 30 s on 1/min",
    "overload": "Sustained overload, 6/min for 20 min",
    "rpm_halved": "Per-minute limits halved, 2/min",
    "daily_exhausted": "Daily budget exhausted mid-run",
    "retrieval": "Paced 3/min, default setting (Table 3 run)",
}


def _ms(value: float) -> str:
    return "not measured" if pd.isna(value) else f"{value:.0f}"


def _triple(row: pd.Series, prefix: str, ps=(50, 95, 99)) -> str:
    return " / ".join(_ms(row[f"{prefix}_p{p}_ms"]) for p in ps)


def _md(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    return "\n".join(lines)


def _setting_label(name: str) -> str:
    size, k, method, *rerank = name.split("-")
    return (
        f"{size.removesuffix('w')} words, {k.removeprefix('k')} pieces, {method.upper()}, "
        f"re-rank {'yes' if rerank else 'no'}"
    )


def render_markdown(results: pd.DataFrame) -> str:
    from relay.retrieval import DEFAULT_SETTING

    paced = results["scenario"] == "retrieval"
    default = results["retrieval_setting"] == DEFAULT_SETTING.name
    shown = results[results["scenario"].isin(SCENARIO_LABELS) & (~paced | default)]
    order = {name: i for i, name in enumerate(SCENARIO_LABELS)}
    latency = latency_table(shown).sort_index(key=lambda idx: idx.map(lambda v: order.get(v, v)))
    outcomes = outcomes_table(shown).loc[latency.index]
    rows_2a, rows_2b = [], []
    for (scenario, source), row in latency.iterrows():
        label = SCENARIO_LABELS[scenario]
        rows_2a.append(
            [
                label,
                source,
                int(row["requests_sent"]),
                f"{row['duration_s']:.0f}",
                f"{row['offered_rate_per_min']:.1f}",
                _triple(row, "first_token"),
                _triple(row, "total"),
            ]
        )
        out = outcomes.loc[(scenario, source)]
        given_up = f"{out['timed_out_pct']:.1f}"
        if out["error_pct"]:
            given_up += f" (+{out['error_pct']:.1f} error)"
        rows_2b.append(
            [
                label,
                f"{out['succeeded_pct']:.1f}",
                f"{out['rate_limited_pct']:.1f}",
                given_up,
                f"{out['retries_per_request']:.2f}",
                int(out["peak_queue_length"]),
                _ms(out["tokens_per_answer"]),
                f"${out['usd_per_1000_answers']:.2f}",
            ]
        )
    rows_3 = [
        [
            _setting_label(name),
            int(row["questions_asked"]),
            f"{row['correct_pct']:.1f}",
            f"{row['wrong_pct']:.1f}",
            f"{row['refused_correctly_pct']:.1f}",
            _ms(row["tokens_per_answer"]),
            _triple(row, "first_token", (50, 95)),
            _triple(row, "total", (50, 95)),
        ]
        for name, row in retrieval_table(results).iterrows()
    ]
    return "\n\n".join(
        [
            "# Results tables",
            "Generated by `uv run python -m relay.tables` from `results/results.csv`. "
            "Times in milliseconds; percentiles use linear interpolation.",
            "## Table 2a: Latency",
            _md(
                [
                    "Scenario",
                    "Real or stand-in",
                    "Requests sent",
                    "Duration (s)",
                    "Offered rate (per min)",
                    "First token p50 / p95 / p99 (ms)",
                    "Total time p50 / p95 / p99 (ms)",
                ],
                rows_2a,
            ),
            "## Table 2b: Outcomes",
            _md(
                [
                    "Scenario",
                    "Succeeded %",
                    "Rate-limited %",
                    "Timed out or given up %",
                    "Retries per request",
                    "Peak queue length",
                    "Tokens per answer",
                    "Cost per 1,000 answers",
                ],
                rows_2b,
            ),
            "## Table 3: Retrieval settings",
            _md(
                [
                    "Setting",
                    "Questions asked",
                    "Correct and cited %",
                    "Wrong %",
                    "Correctly refused %",
                    "Tokens per answer",
                    "First token p50 / p95 (ms)",
                    "Total time p50 / p95 (ms)",
                ],
                rows_3,
            ),
        ]
    )


def main() -> None:
    from relay.library import ROOT

    results = pd.read_csv(ROOT / "results" / "results.csv", keep_default_na=False, na_values=[""])
    out = ROOT / "docs" / "tables.md"
    out.write_text(render_markdown(results) + "\n")
    print(out)


if __name__ == "__main__":
    main()
