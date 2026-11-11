import json
import subprocess
from pathlib import Path

import pandas as pd

from relay.config import PRICES_USD_PER_MTOK
from relay.reply import Reply
from relay.tables import RESULT_COLUMNS


def request_usd(steps: list[dict]) -> float:
    total = 0.0
    for step in steps:
        usd_in, usd_out = PRICES_USD_PER_MTOK.get(step["model"], (0.0, 0.0))
        total += (step["tokens_in"] * usd_in + step["tokens_out"] * usd_out) / 1e6
    return total


def result_row(
    *,
    request_id: str,
    scenario: str,
    source: str,
    started_at: float,
    reply: Reply,
    retrieval_setting: str,
    answer_grade: str,
) -> dict:
    row = {
        "request_id": request_id,
        "scenario": scenario,
        "source": source,
        "start_time": round(started_at, 3),
        "first_token_ms": None if reply.first_token_ms is None else round(reply.first_token_ms, 1),
        "total_ms": None if reply.total_ms is None else round(reply.total_ms, 1),
        "status": reply.status,
        "retries": reply.retries,
        "tokens_in": reply.tokens_in,
        "tokens_out": reply.tokens_out,
        "retrieval_setting": retrieval_setting,
        "answer_grade": answer_grade,
        "queue_depth": reply.queue_depth,
        "reason": reply.reason or "",
        "usd": round(request_usd(reply.steps), 8),
    }
    return {column: row[column] for column in RESULT_COLUMNS}


def commit() -> str:
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "src"], capture_output=True, text=True)
    return head.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def write_run(out: Path, rows: list[dict], replies: list[dict], info: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=RESULT_COLUMNS).to_csv(out / "results.csv", index=False)
    with (out / "replies.jsonl").open("w") as f:
        for entry in replies:
            f.write(json.dumps(entry) + "\n")
    (out / "run.json").write_text(json.dumps({**info, "commit": commit()}, indent=2))
