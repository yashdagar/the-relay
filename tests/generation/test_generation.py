import pytest

import thresholds
from gold import QUESTIONS, expects_answer, questions
from relay.eval import grade, is_grounded

pytestmark = pytest.mark.real

ANSWERABLE = questions("answerable")
UNANSWERABLE = questions("unanswerable")


def rate(flags):
    flags = list(flags)
    return sum(flags) / len(flags)


def test_every_sent_answer_is_grounded(real_replies):
    sent = [r for r in real_replies.values() if r.kind == "answer"]
    assert rate(is_grounded(r) for r in sent) >= thresholds.GENERATION_MIN_GROUNDED_RATE


def test_every_sent_answer_cites_paper_section_and_page(real_replies):
    for reply in real_replies.values():
        if reply.kind == "answer":
            assert all(c.paper and c.section and c.page for c in reply.citations)


def test_answers_cite_the_paper_the_question_is_about(real_replies):
    for q in filter(expects_answer, QUESTIONS):
        reply = real_replies[q["id"]]
        if reply.kind == "answer":
            assert {c.paper for c in reply.citations} & set(q["papers"]), q["id"]


def test_correct_rate_on_answerable_questions_meets_threshold(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in ANSWERABLE]
    assert rate(g == "correct" for g in grades) >= thresholds.GENERATION_MIN_CORRECT_RATE


def test_wrong_rate_across_all_questions_is_within_threshold(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in QUESTIONS]
    assert rate(g == "wrong" for g in grades) <= thresholds.GENERATION_MAX_WRONG_RATE


def test_unanswerable_questions_are_refused(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in UNANSWERABLE]
    assert (
        rate(g == "refused_correctly" for g in grades)
        >= thresholds.GENERATION_MIN_UNANSWERABLE_REFUSED_RATE
    )


def test_answerable_questions_are_rarely_refused(real_replies):
    grades = [grade(q, real_replies[q["id"]]) for q in ANSWERABLE]
    assert (
        rate(g == "refused_wrongly" for g in grades) <= thresholds.GENERATION_MAX_OVER_REFUSAL_RATE
    )


def test_refusals_say_the_papers_do_not_cover_it(real_replies):
    for q in UNANSWERABLE:
        reply = real_replies[q["id"]]
        if reply.kind == "refusal":
            assert "paper" in reply.text.lower(), q["id"]
