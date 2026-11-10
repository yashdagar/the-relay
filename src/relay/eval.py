import re
import unicodedata
from collections.abc import Sequence

import numpy as np

from relay.questions import expects_answer
from relay.reply import Reply

Grade = str

CITATION_MARKER = re.compile(r"[\[【][^\]】]*[\]】]")
CHUNK_ID = re.compile(r"\d{4}\.\d{5}v\d+:\d+:\d+")
DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")
NUMBER = re.compile(r"(?<![A-Za-z0-9_{^.])(?<![A-Za-z]-)\d+(?:[.,]\d+)*")
THIN_THOUSANDS = re.compile(r"(?<=\d) (?=\d{3}\b)")


def normalize(text: str) -> str:
    text = re.sub(r"-\s+", "", text.lower())
    return " ".join(re.findall(r"[a-z0-9]+", text))


Evidence = Sequence[str | Sequence[str]]


def _alternatives(item: str | Sequence[str]) -> list[str]:
    return [item] if isinstance(item, str) else list(item)


def contains_evidence(text: str, evidence: Evidence) -> bool:
    haystack = f" {normalize(text)} "
    phrases = [p for item in evidence for p in _alternatives(item)]
    return any(f" {normalize(phrase)} " in haystack for phrase in phrases)


def precision_at_k(ranked: Sequence[str], evidence: Evidence, k: int) -> float:
    return sum(contains_evidence(text, evidence) for text in ranked[:k]) / k


def recall_at_k(ranked: Sequence[str], evidence: Evidence, k: int) -> float:
    if not evidence:
        raise ValueError("recall is undefined without evidence")
    found = sum(any(contains_evidence(text, [item]) for text in ranked[:k]) for item in evidence)
    return found / len(evidence)


def reciprocal_rank(ranked: Sequence[str], evidence: Evidence) -> float:
    for rank, text in enumerate(ranked, start=1):
        if contains_evidence(text, evidence):
            return 1 / rank
    return 0.0


def percentile(values: Sequence[float], p: float) -> float:
    return float(np.percentile(values, p))


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUMBER.findall(text)}


def answer_body(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).translate(DASHES)
    text = CHUNK_ID.sub(" ", CITATION_MARKER.sub(" ", text))
    return THIN_THOUSANDS.sub("", text)


def is_grounded(reply: Reply) -> bool:
    if not reply.citations:
        return False
    retrieved = {c.id for c in reply.retrieved}
    if any(c.id not in retrieved for c in reply.citations):
        return False
    claimed = _numbers(answer_body(reply.text))
    supported = set().union(*(_numbers(c.text) for c in reply.citations))
    return claimed <= supported


def grade(question: dict, reply: Reply) -> Grade:
    answer_expected = expects_answer(question)
    if reply.kind in ("refusal", "degraded"):
        return "refused_wrongly" if answer_expected else "refused_correctly"
    if not answer_expected:
        return "wrong"
    body = answer_body(reply.text)
    has_facts = all(re.search(p, body, re.IGNORECASE) for p in question["must_include"])
    has_forbidden = any(re.search(p, body, re.IGNORECASE) for p in question["must_not_include"])
    return "correct" if has_facts and not has_forbidden and is_grounded(reply) else "wrong"
