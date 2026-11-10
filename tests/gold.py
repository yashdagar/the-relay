import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = [
    json.loads(line)
    for line in (ROOT / "data" / "questions" / "eval.jsonl").read_text().splitlines()
    if line.strip()
]


def questions(*kinds: str) -> list[dict]:
    return [q for q in QUESTIONS if q["kind"] in kinds]


def expects_answer(question: dict) -> bool:
    if question["kind"] == "injection":
        return question["expect"] == "answer"
    return question["kind"] == "answerable"
