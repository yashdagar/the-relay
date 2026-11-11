import time

import pytest

import thresholds
from gold import QUESTIONS, expects_answer
from relay.eval import percentile, precision_at_k, recall_at_k, reciprocal_rank
from relay.retrieval import DEFAULT_SETTING, SETTINGS, Retriever, interleave

ANSWERABLE = [q for q in QUESTIONS if expects_answer(q)]


@pytest.fixture(scope="module")
def retriever():
    return Retriever(DEFAULT_SETTING)


@pytest.fixture(scope="module")
def ranked(retriever):
    return {
        q["id"]: [hit.chunk.text for hit in retriever.search(q["question"])] for q in ANSWERABLE
    }


def mean(values):
    values = list(values)
    return sum(values) / len(values)


def test_there_are_at_least_three_retrieval_settings_to_compare():
    assert len(SETTINGS) >= 3


def test_default_setting_is_one_of_the_compared_settings():
    assert DEFAULT_SETTING in SETTINGS.values()


def test_search_returns_top_k_hits(retriever):
    assert len(retriever.search("multi-head attention")) == DEFAULT_SETTING.top_k


def test_hits_are_ordered_by_descending_score(retriever):
    scores = [hit.score for hit in retriever.search("multi-head attention")]
    assert scores == sorted(scores, reverse=True)


def test_every_hit_carries_paper_section_and_page(retriever):
    for hit in retriever.search("residual vector quantization"):
        assert hit.chunk.paper and hit.chunk.section and hit.chunk.page >= 1


def test_search_is_deterministic(retriever):
    query = "how are chunk boundaries predicted"
    first = [hit.chunk.id for hit in retriever.search(query)]
    assert first == [hit.chunk.id for hit in retriever.search(query)]


def test_reference_lists_are_never_retrieved(retriever):
    hits = retriever.search("Vaswani Shazeer Parmar Uszkoreit Jones Gomez Kaiser Polosukhin 2017")
    assert all(hit.chunk.section != "References" for hit in hits)


def test_search_can_be_restricted_to_named_papers(retriever):
    hits = retriever.search("attention heads", papers=["2410.00037v1"])
    assert {hit.chunk.paper for hit in hits} == {"2410.00037v1"}


def test_recall_at_k_meets_threshold(ranked):
    recall = mean(
        recall_at_k(ranked[q["id"]], q["evidence"], DEFAULT_SETTING.top_k) for q in ANSWERABLE
    )
    assert recall >= thresholds.RETRIEVAL_MIN_RECALL_AT_K


def test_mean_reciprocal_rank_meets_threshold(ranked):
    mrr = mean(reciprocal_rank(ranked[q["id"]], q["evidence"]) for q in ANSWERABLE)
    assert mrr >= thresholds.RETRIEVAL_MIN_MRR


def test_precision_at_k_meets_threshold(ranked):
    precision = mean(
        precision_at_k(ranked[q["id"]], q["evidence"], DEFAULT_SETTING.top_k) for q in ANSWERABLE
    )
    assert precision >= thresholds.RETRIEVAL_MIN_PRECISION_AT_K


def test_search_latency_p95_is_within_budget(retriever):
    timings = []
    for q in QUESTIONS:
        start = time.perf_counter()
        retriever.search(q["question"])
        timings.append((time.perf_counter() - start) * 1000)
    assert percentile(timings, 95) <= thresholds.RETRIEVAL_MAX_SEARCH_P95_MS


def test_interleave_takes_the_best_of_each_list_first():
    assert interleave(["a1", "a2", "a3"], ["b1", "b2"], ["c1"]) == [
        "a1",
        "b1",
        "c1",
        "a2",
        "b2",
        "a3",
    ]


def test_interleave_drops_repeats():
    assert interleave(["a", "b"], ["a", "c"]) == ["a", "b", "c"]


def test_lexical_top_score_does_not_depend_on_the_ranking_method():
    query = "how are chunk boundaries predicted"
    bm25 = Retriever(SETTINGS["300w-k5-bm25"])
    hybrid = Retriever(SETTINGS["300w-k5-hybrid"])
    assert hybrid.lexical_top_score(query) == pytest.approx(bm25.search(query)[0].score)
