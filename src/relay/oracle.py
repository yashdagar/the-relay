import json
import re

from relay.eval import normalize
from relay.questions import QUESTIONS, expects_answer

GOLD = {q["question"]: q for q in QUESTIONS}
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")
MAX_SENTENCE_WORDS = 60


def _phrase_pattern(phrase: str) -> re.Pattern:
    words = normalize(phrase).split()
    return re.compile(r"[\W_]*".join(re.escape(w) for w in words), re.I)


def _sentence_around(text: str, phrase: str) -> str | None:
    flat = re.sub(r"-\s+", "", text)
    match = _phrase_pattern(phrase).search(flat)
    if match is None:
        return None
    starts = [0] + [m.end() for m in SENTENCE_END.finditer(flat) if m.end() <= match.start()]
    ends = [m.start() for m in SENTENCE_END.finditer(flat) if m.start() >= match.end()]
    sentence = flat[starts[-1] : ends[0] if ends else len(flat)]
    words = sentence.split()
    if len(words) > MAX_SENTENCE_WORDS:
        sentence = flat[match.start() : match.end()]
    return sentence.strip().rstrip(".")


def oracle(step: str, question: str, passages: list[tuple[str, str]]) -> str | None:
    gold = GOLD.get(question)
    if gold is None:
        return None
    if step == "understand":
        in_scope = gold["kind"] != "out_of_scope" and (
            gold["kind"] != "injection" or expects_answer(gold)
        )
        return json.dumps({"in_scope": in_scope, "sub_questions": []})
    if step != "draft":
        return None
    if not expects_answer(gold):
        return "NOT_IN_PAPERS"
    sentences = []
    for item in gold["evidence"]:
        alternatives = [item] if isinstance(item, str) else item
        found = next(
            (
                (chunk_id, sentence)
                for phrase in alternatives
                for chunk_id, text in passages
                if (sentence := _sentence_around(text, phrase))
            ),
            None,
        )
        if found is None:
            return "NOT_IN_PAPERS"
        chunk_id, sentence = found
        sentences.append(f"{sentence} [{chunk_id}].")
    return " ".join(dict.fromkeys(sentences))
