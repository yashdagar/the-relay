import asyncio
import json
import os
from datetime import datetime
from pathlib import Path

import pytest
from dotenv import load_dotenv

from gold import QUESTIONS, ROOT

load_dotenv(ROOT / ".env")


@pytest.fixture
def standin():
    from relay.standin import StandInConfig, running

    stack = []

    def start(**overrides) -> str:
        context = running(StandInConfig(**overrides))
        stack.append(context)
        return context.__enter__()

    yield start
    for context in reversed(stack):
        context.__exit__(None, None, None)


@pytest.fixture
def standin_pipeline(standin):
    from relay.config import Config
    from relay.pipeline import Pipeline

    def build(config_overrides: dict | None = None, **standin_overrides) -> Pipeline:
        base_url = standin(**standin_overrides)
        return Pipeline(Config.standin(base_url, **(config_overrides or {})))

    return build


@pytest.fixture(scope="session")
def real_replies() -> dict[str, "object"]:
    from relay.config import Config
    from relay.pipeline import Pipeline, Reply

    cached = os.environ.get("RELAY_EVAL_REPLIES")
    if cached:
        rows = [json.loads(line) for line in Path(cached).read_text().splitlines()]
        return {row["question_id"]: Reply.from_dict(row["reply"]) for row in rows}

    if not os.environ.get("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY is not set")

    pipeline = Pipeline(Config.groq())

    async def run_all():
        return [await pipeline.answer(q["question"]) for q in QUESTIONS]

    replies = asyncio.run(run_all())
    out_dir = ROOT / "results" / f"{datetime.now():%Y%m%d-%H%M}-eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "replies.jsonl").open("w") as f:
        for q, reply in zip(QUESTIONS, replies, strict=True):
            f.write(json.dumps({"question_id": q["id"], "reply": reply.to_dict()}) + "\n")
    return {q["id"]: reply for q, reply in zip(QUESTIONS, replies, strict=True)}
