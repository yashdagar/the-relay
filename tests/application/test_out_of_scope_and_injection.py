import pytest

import thresholds
from gold import questions
from relay.eval import grade

pytestmark = pytest.mark.real

OUT_OF_SCOPE = questions("out_of_scope")
INJECTION = questions("injection")


def rate(flags):
    flags = list(flags)
    return sum(flags) / len(flags)


def test_out_of_scope_questions_are_refused(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in OUT_OF_SCOPE]
    assert (
        rate(g == "refused_correctly" for g in grades)
        >= thresholds.APPLICATION_MIN_OUT_OF_SCOPE_REFUSED_RATE
    )


def test_out_of_scope_refusals_are_flagged_as_out_of_scope(real_replies):
    for q in OUT_OF_SCOPE:
        assert real_replies[q["id"]].reason == "out_of_scope", q["id"]


def test_out_of_scope_questions_skip_drafting(real_replies):
    for q in OUT_OF_SCOPE:
        assert real_replies[q["id"]].model_calls <= 1, q["id"]


STRICT_NUMBERS_COST = pytest.mark.xfail(
    strict=True,
    reason="the correct reply repeats the user's false number, which the strict number check "
    "rejects; kept strict by choice (results/20261111-0320-eval-300w-k5-bm25/NOTES.md)",
)
ANSWERABLE_INJECTIONS = [
    pytest.param(q, id=q["id"], marks=STRICT_NUMBERS_COST if q["id"] == "inj-03" else ())
    for q in INJECTION
    if q["expect"] == "answer"
]


@pytest.mark.parametrize("q", INJECTION, ids=lambda q: q["id"])
def test_injected_instructions_are_not_followed(real_replies, q):
    assert grade(q, real_replies[q["id"]]) != "wrong"


def test_injection_safety_rate_meets_threshold(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in INJECTION]
    assert rate(g != "wrong" for g in grades) >= thresholds.APPLICATION_MIN_INJECTION_SAFE_RATE


@pytest.mark.parametrize("q", ANSWERABLE_INJECTIONS)
def test_answerable_injections_still_get_the_right_answer(real_replies, q):
    assert grade(q, real_replies[q["id"]]) == "correct"
