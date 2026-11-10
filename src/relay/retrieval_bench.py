import argparse
import json
import platform
import subprocess
import time
from datetime import datetime

import pandas as pd

from relay.eval import contains_evidence, percentile, precision_at_k, recall_at_k, reciprocal_rank
from relay.library import ROOT
from relay.questions import QUESTIONS, expects_answer
from relay.retrieval import DEFAULT_SETTING, SETTINGS, Retriever


def _commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _first_relevant_rank(texts: list[str], evidence) -> int | None:
    return next((i for i, t in enumerate(texts, 1) if contains_evidence(t, evidence)), None)


def run(reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    quality, latency = [], []
    for name, setting in SETTINGS.items():
        retriever = Retriever(setting)
        for q in filter(expects_answer, QUESTIONS):
            hits = retriever.search(q["question"])
            texts = [h.chunk.text for h in hits]
            quality.append(
                {
                    "setting": name,
                    "question_id": q["id"],
                    "recall_at_k": recall_at_k(texts, q["evidence"], setting.top_k),
                    "precision_at_k": precision_at_k(texts, q["evidence"], setting.top_k),
                    "reciprocal_rank": reciprocal_rank(texts, q["evidence"]),
                    "first_relevant_rank": _first_relevant_rank(texts, q["evidence"]),
                    "retrieved": ";".join(h.chunk.cite() for h in hits),
                }
            )
        for rep in range(reps):
            for q in QUESTIONS:
                start = time.perf_counter()
                retriever.search(q["question"])
                latency.append(
                    {
                        "setting": name,
                        "question_id": q["id"],
                        "rep": rep,
                        "ms": (time.perf_counter() - start) * 1000,
                    }
                )
    return pd.DataFrame(quality), pd.DataFrame(latency)


def summarize(quality: pd.DataFrame, latency: pd.DataFrame) -> pd.DataFrame:
    q = quality.groupby("setting")[["recall_at_k", "reciprocal_rank", "precision_at_k"]].mean()
    q.columns = ["recall_at_k", "mrr", "precision_at_k"]
    lat = latency.groupby("setting")["ms"].agg(
        searches="count",
        p50_ms=lambda v: percentile(v, 50),
        p95_ms=lambda v: percentile(v, 95),
        p99_ms=lambda v: percentile(v, 99),
        max_ms="max",
    )
    words = pd.Series({n: s.piece_size * s.top_k for n, s in SETTINGS.items()}, name="words_k")
    return q.join(lat).join(words).loc[list(SETTINGS)].round(3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval quality and latency")
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    started = datetime.now()
    out = ROOT / "results" / f"{started:%Y%m%d-%H%M}-retrieval-offline"
    out.mkdir(parents=True)
    quality, latency = run(args.reps)
    quality.to_csv(out / "quality.csv", index=False)
    latency.to_csv(out / "latency.csv", index=False)
    summarize(quality, latency).to_csv(out / "summary.csv")
    run_info = {
        "commit": _commit(),
        "started": started.isoformat(timespec="seconds"),
        "reps": args.reps,
        "default_setting": DEFAULT_SETTING.name,
        "settings": {n: vars(s) for n, s in SETTINGS.items()},
        "machine": f"{platform.platform()} {platform.processor()}",
        "python": platform.python_version(),
    }
    (out / "run.json").write_text(json.dumps(run_info, indent=2))
    print(out)


if __name__ == "__main__":
    main()
