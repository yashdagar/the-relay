import json

import pandas as pd

from relay.eval import grade
from relay.library import ROOT
from relay.questions import QUESTIONS
from relay.records import result_row
from relay.reply import Reply
from relay.tables import RESULT_COLUMNS

MANIFEST = ROOT / "results" / "manifest.json"
RESULTS = ROOT / "results" / "results.csv"
BY_ID = {q["id"]: q for q in QUESTIONS}


def _replies(run: str) -> list[dict]:
    lines = (ROOT / "results" / run / "replies.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def assemble_run(run: str, scenario: str | None = None) -> pd.DataFrame:
    rows = pd.read_csv(ROOT / "results" / run / "results.csv", keep_default_na=False)
    replies = _replies(run)
    if len(rows) != len(replies):
        raise ValueError(f"{run}: {len(rows)} rows but {len(replies)} replies")
    out = []
    for row, entry in zip(rows.to_dict("records"), replies, strict=True):
        if not row["request_id"].endswith(entry.get("request_id", entry["question_id"])):
            raise ValueError(f"{run}: {row['request_id']} does not match its reply")
        reply = Reply.from_dict(entry["reply"])
        out.append(
            result_row(
                request_id=f"{run}/{row['request_id']}",
                scenario=scenario or row["scenario"],
                source=row["source"],
                started_at=float(row["start_time"]),
                reply=reply,
                retrieval_setting=row["retrieval_setting"],
                answer_grade=grade(BY_ID[entry["question_id"]], reply),
            )
        )
    return pd.DataFrame(out, columns=RESULT_COLUMNS)


def assemble(manifest: dict) -> pd.DataFrame:
    frames = [assemble_run(run["dir"], run.get("scenario")) for run in manifest["runs"]]
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    results = assemble(json.loads(MANIFEST.read_text()))
    results.to_csv(RESULTS, index=False)
    print(f"{RESULTS}: {len(results)} rows")


if __name__ == "__main__":
    main()
