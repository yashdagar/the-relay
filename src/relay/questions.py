import json

from relay.library import ROOT

QUESTIONS_PATH = ROOT / "data" / "questions" / "eval.jsonl"
QUESTIONS = [json.loads(line) for line in QUESTIONS_PATH.read_text().splitlines() if line.strip()]


def questions(*kinds: str) -> list[dict]:
    return [q for q in QUESTIONS if q["kind"] in kinds]


def expects_answer(question: dict) -> bool:
    if question["kind"] == "injection":
        return question["expect"] == "answer"
    return question["kind"] == "answerable"
