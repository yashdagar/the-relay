import pytest

from relay.eval import (
    contains_evidence,
    grade,
    is_grounded,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from relay.library import Chunk
from relay.pipeline import Reply

EVIDENCE = ["warmup_steps = 4000"]
RELEVANT = "We varied the learning rate ... warmup_steps = 4000. This corresponds to"
IRRELEVANT = "The encoder is composed of a stack of N = 6 identical layers."

QUESTION = {
    "id": "q",
    "kind": "answerable",
    "papers": ["1706.03762v7"],
    "evidence": EVIDENCE,
    "must_include": ["4,?000"],
    "must_not_include": ["8000"],
}
UNANSWERABLE = {**QUESTION, "kind": "unanswerable", "evidence": [], "must_include": []}
PASSAGE = Chunk(
    id="1706.03762v7:300:9", paper="1706.03762v7", section="Optimizer", page=7, text=RELEVANT
)


def answer(text: str, citations=(PASSAGE,), retrieved=(PASSAGE,)) -> Reply:
    return Reply(kind="answer", text=text, citations=list(citations), retrieved=list(retrieved))


def refusal() -> Reply:
    return Reply(kind="refusal", reason="not_in_papers", text="The papers do not say.")


def test_evidence_matches_across_line_break_hyphenation_and_case():
    assert contains_evidence("we use WARMUP_STEPS =\n4000 steps", EVIDENCE)


def test_evidence_does_not_match_a_different_number():
    assert not contains_evidence("warmup_steps = 40000", EVIDENCE)


def test_precision_counts_relevant_pieces_in_top_k():
    assert precision_at_k([RELEVANT, IRRELEVANT, IRRELEVANT, RELEVANT], EVIDENCE, k=4) == 0.5


def test_precision_only_looks_at_the_first_k_pieces():
    assert precision_at_k([IRRELEVANT, IRRELEVANT, RELEVANT], EVIDENCE, k=2) == 0.0


def test_recall_is_the_share_of_evidence_phrases_found_in_top_k():
    evidence = ["warmup_steps = 4000", "label smoothing of value"]
    assert recall_at_k([RELEVANT, IRRELEVANT], evidence, k=2) == 0.5


def test_any_alternative_phrase_satisfies_an_evidence_item():
    evidence = [["warmup_steps = 4000", "a 4000 step warmup"]]
    assert recall_at_k(["followed by a 4000 step warmup"], evidence, k=1) == 1.0


def test_alternatives_count_as_one_evidence_item():
    evidence = [["warmup_steps = 4000", "a 4000 step warmup"], "label smoothing of value"]
    assert recall_at_k([RELEVANT], evidence, k=1) == 0.5


def test_reciprocal_rank_uses_the_first_relevant_piece():
    assert reciprocal_rank([IRRELEVANT, IRRELEVANT, RELEVANT], EVIDENCE) == pytest.approx(1 / 3)


def test_reciprocal_rank_is_zero_when_nothing_relevant_is_retrieved():
    assert reciprocal_rank([IRRELEVANT], EVIDENCE) == 0.0


def test_percentile_interpolates_linearly_between_ranks():
    assert percentile([100, 200, 300, 400], 50) == 250


def test_percentile_of_one_value_is_that_value():
    assert percentile([42], 99) == 42


def test_answer_whose_numbers_appear_in_its_citations_is_grounded():
    assert is_grounded(answer("warmup_steps was 4000 [1706.03762v7:300:9]."))


def test_answer_with_a_number_missing_from_its_citations_is_not_grounded():
    assert not is_grounded(answer("warmup_steps was 4000 and the batch size was 25000."))


def test_answer_without_citations_is_not_grounded():
    assert not is_grounded(answer("warmup_steps was 4000.", citations=()))


def test_answer_citing_a_passage_that_was_not_retrieved_is_not_grounded():
    assert not is_grounded(answer("warmup_steps was 4000.", retrieved=()))


def test_grounded_answer_with_all_facts_is_graded_correct():
    assert grade(QUESTION, answer("warmup_steps was 4000.")) == "correct"


def test_answer_missing_a_required_fact_is_graded_wrong():
    assert grade(QUESTION, answer("It used a warmup phase.")) == "wrong"


def test_answer_containing_a_forbidden_fact_is_graded_wrong():
    assert grade(QUESTION, answer("4000, or 8000 in some runs.")) == "wrong"


def test_ungrounded_answer_with_all_facts_is_graded_wrong():
    assert grade(QUESTION, answer("warmup_steps was 4000.", citations=())) == "wrong"


def test_refusing_an_answerable_question_is_graded_refused_wrongly():
    assert grade(QUESTION, refusal()) == "refused_wrongly"


def test_refusing_an_unanswerable_question_is_graded_refused_correctly():
    assert grade(UNANSWERABLE, refusal()) == "refused_correctly"


def test_answering_an_unanswerable_question_is_graded_wrong():
    assert grade(UNANSWERABLE, answer("It was 4000.")) == "wrong"


def test_citation_ids_in_lenticular_brackets_are_not_counted_as_claims():
    assert is_grounded(answer("warmup_steps was 4000【1706.03762v7:300:9】."))


def test_bare_citation_ids_are_not_counted_as_claims():
    assert is_grounded(answer("warmup_steps was 4000 (see 1706.03762v7:300:9)."))


def test_forbidden_fact_inside_a_citation_id_does_not_count():
    question = {**QUESTION, "must_not_include": ["\\b9\\b"]}
    assert grade(question, answer("warmup_steps was 4000 [1706.03762v7:300:9].")) == "correct"


def test_narrow_no_break_space_matches_a_plain_space_in_facts():
    question = {**QUESTION, "must_include": ["4,?000 steps"]}
    assert grade(question, answer("warmup_steps was 4000 steps.")) == "correct"


def test_non_breaking_hyphen_matches_a_plain_hyphen_in_facts():
    passage = Chunk(id="p:1", paper="2507.07955v2", section="S", page=1, text="FineWeb-Edu 100B")
    question = {**QUESTION, "must_include": ["FineWeb-Edu"]}
    reply = answer("Trained on FineWeb‑Edu.", citations=(passage,), retrieved=(passage,))
    assert grade(question, reply) == "correct"


def test_thin_space_thousands_separator_reads_as_one_number():
    passage = Chunk(id="p:1", paper="2410.00037v1", section="S", page=1, text="a dimension of 1024")
    question = {**QUESTION, "must_include": ["1,?024"]}
    reply = answer("A dimension of 1 024.", citations=(passage,), retrieved=(passage,))
    assert grade(question, reply) == "correct"


def test_digits_inside_names_are_not_counted_as_claims():
    assert is_grounded(answer("warmup_steps was 4000 with Mamba-2 and GPT-3 style k_{t-1}."))


def test_a_year_not_in_the_cited_passage_is_not_grounded():
    assert not is_grounded(answer("warmup_steps was 4000, as proposed by Chen et al., 2022."))


def test_digits_trailing_a_name_are_not_counted_as_claims():
    assert is_grounded(answer("warmup_steps was 4000 in layer L12."))


def test_number_words_match_digit_facts():
    passage = Chunk(id="p:1", paper="1706.03762v7", section="S", page=1, text="h = 8 heads")
    question = {**QUESTION, "must_include": ["\\b8\\b"]}
    reply = answer("It uses eight heads.", citations=(passage,), retrieved=(passage,))
    assert grade(question, reply) == "correct"
