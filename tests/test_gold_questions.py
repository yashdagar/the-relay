from collections import Counter

import pytest

from gold import QUESTIONS, expects_answer
from relay.eval import contains_evidence
from relay.library import PAPERS, load_chunks

KINDS = {"answerable", "unanswerable", "out_of_scope", "injection"}
PIECE_SIZES = (120, 300, 600)


def test_question_ids_are_unique():
    assert len({q["id"] for q in QUESTIONS}) == len(QUESTIONS)


def test_there_are_at_least_fifteen_self_written_questions():
    assert len(QUESTIONS) >= 15


def test_every_question_has_a_known_kind():
    assert {q["kind"] for q in QUESTIONS} <= KINDS


def test_every_kind_is_covered():
    counts = Counter(q["kind"] for q in QUESTIONS)
    assert all(counts[kind] >= 3 for kind in KINDS)


def test_every_question_names_only_library_papers():
    assert all(set(q["papers"]) <= set(PAPERS) for q in QUESTIONS)


def test_questions_expecting_an_answer_have_evidence_and_facts():
    for q in filter(expects_answer, QUESTIONS):
        assert q["evidence"] and q["must_include"], q["id"]


@pytest.mark.parametrize("size", PIECE_SIZES)
def test_every_evidence_phrase_fits_inside_one_piece_of_its_paper(size):
    chunks = load_chunks(size)
    for q in QUESTIONS:
        for phrase in q["evidence"]:
            assert any(
                contains_evidence(c.text, [phrase]) for c in chunks if c.paper in q["papers"]
            ), (q["id"], phrase)
