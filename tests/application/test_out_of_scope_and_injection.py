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


@pytest.mark.parametrize("q", INJECTION, ids=lambda q: q["id"])
def test_injected_instructions_are_not_followed(real_replies, q):
    assert grade(q, real_replies[q["id"]]) in {"correct", "refused_correctly"}


def test_injection_pass_rate_meets_threshold(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in INJECTION]
    assert (
        rate(g in {"correct", "refused_correctly"} for g in grades)
        >= thresholds.APPLICATION_MIN_INJECTION_PASS_RATE
    )
