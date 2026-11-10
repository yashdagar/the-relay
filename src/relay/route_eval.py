import argparse
import asyncio
import json
import time
from datetime import datetime

import pandas as pd

from relay.client import ModelClient
from relay.config import Config
from relay.library import ROOT
from relay.pipeline import _parse_json
from relay.prompts import understand_messages
from relay.questions import QUESTIONS, expects_answer
from relay.records import commit


def expected_in_scope(question: dict) -> bool:
    return question["kind"] == "unanswerable" or expects_answer(question)


async def route_all(config: Config, spacing_s: float) -> list[dict]:
    client = ModelClient(config.base_url, config.api_key)
    rows = []
    try:
        for q in QUESTIONS:
            tick = time.monotonic()
            result = await client.chat(
                config.understand_model,
                understand_messages(q["question"]),
                max_tokens=config.understand_max_tokens,
                deadline_s=config.understand_timeout_ms / 1000,
                step="understand",
                extra=config.extra(config.understand_model),
            )
            parsed = _parse_json(result.text)
            rows.append(
                {
                    "question_id": q["id"],
                    "kind": q["kind"],
                    "expected_in_scope": expected_in_scope(q),
                    "in_scope": parsed.get("in_scope"),
                    "sub_questions": json.dumps(parsed.get("sub_questions") or []),
                    "outcome": result.outcome,
                    "total_ms": round(result.total_ms, 1),
                    "tokens_in": result.tokens_in,
                    "tokens_out": result.tokens_out,
                    "text": result.text,
                }
            )
            print(q["id"], parsed.get("in_scope"), expected_in_scope(q), result.outcome)
            await asyncio.sleep(max(0.0, spacing_s - (time.monotonic() - tick)))
    finally:
        await client.aclose()
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the understand step's routing alone")
    parser.add_argument("--spacing", type=float, default=8.0)
    args = parser.parse_args()
    started = datetime.now()
    rows = asyncio.run(route_all(Config.groq(), args.spacing))
    out = ROOT / "results" / f"{started:%Y%m%d-%H%M}-router"
    out.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(out / "router.csv", index=False)
    info = {"started": started.isoformat(timespec="seconds"), "commit": commit()}
    (out / "run.json").write_text(json.dumps(info, indent=2))
    print(out)


if __name__ == "__main__":
    main()
