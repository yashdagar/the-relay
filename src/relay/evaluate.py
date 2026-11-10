import argparse
import asyncio
import time
from dataclasses import replace
from datetime import datetime

from relay.config import Config
from relay.eval import grade
from relay.library import ROOT
from relay.pipeline import Pipeline
from relay.questions import QUESTIONS
from relay.records import result_row, write_run
from relay.retrieval import SETTINGS


async def evaluate(config: Config, questions: list[dict], spacing_s: float, scenario: str):
    pipeline = Pipeline(config)
    rows, replies = [], []
    try:
        for i, q in enumerate(questions):
            started_at, tick = time.time(), time.monotonic()
            reply = await pipeline.answer(q["question"])
            answer_grade = grade(q, reply)
            rows.append(
                result_row(
                    request_id=f"{scenario}-{config.retrieval.name}-{q['id']}",
                    scenario=scenario,
                    source=config.source,
                    started_at=started_at,
                    reply=reply,
                    retrieval_setting=config.retrieval.name,
                    answer_grade=answer_grade,
                )
            )
            replies.append(
                {"question_id": q["id"], "grade": answer_grade, "reply": reply.to_dict()}
            )
            progress = f"{i + 1:>2}/{len(questions)} {q['id']:7} {answer_grade:17}"
            print(f"{progress} {reply.kind:8} {reply.total_ms:7.0f} ms")
            if i + 1 < len(questions):
                await asyncio.sleep(max(0.0, spacing_s - (time.monotonic() - tick)))
    finally:
        await pipeline.aclose()
    return rows, replies


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the gold questions through the pipeline")
    parser.add_argument("--setting", default="300w-k5-bm25", choices=sorted(SETTINGS))
    parser.add_argument("--spacing", type=float, default=20.0, help="seconds between questions")
    parser.add_argument("--ids", nargs="*", help="only these question ids")
    parser.add_argument("--scenario", default="retrieval")
    args = parser.parse_args()

    config = replace(Config.groq(), retrieval=SETTINGS[args.setting])
    questions = [q for q in QUESTIONS if not args.ids or q["id"] in args.ids]
    started = datetime.now()
    rows, replies = asyncio.run(evaluate(config, questions, args.spacing, args.scenario))
    out = ROOT / "results" / f"{started:%Y%m%d-%H%M}-eval-{args.setting}"
    info = {
        "started": started.isoformat(timespec="seconds"),
        "setting": args.setting,
        "spacing_s": args.spacing,
        "questions": [q["id"] for q in questions],
        "models": {
            "understand": config.understand_model,
            "draft": config.draft_model,
            "lean_draft": config.lean_draft_model,
            "check": config.check_model,
        },
    }
    write_run(out, rows, replies, info)
    print(out)


if __name__ == "__main__":
    main()
