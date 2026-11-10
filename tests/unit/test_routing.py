from relay.pipeline import sub_queries


def test_sub_question_naming_a_library_paper_is_searched_in_that_paper():
    parsed = {"sub_questions": [{"query": "encoder layers", "paper": "1706.03762v7"}]}
    assert sub_queries(parsed, limit=2) == [("encoder layers", ["1706.03762v7"])]


def test_sub_question_naming_an_unknown_paper_searches_everything():
    parsed = {"sub_questions": [{"query": "encoder layers", "paper": "1234.56789v1"}]}
    assert sub_queries(parsed, limit=2) == [("encoder layers", None)]


def test_plain_string_sub_questions_search_everything():
    assert sub_queries({"sub_questions": ["encoder layers"]}, limit=2) == [("encoder layers", None)]


def test_sub_questions_are_capped():
    parsed = {"sub_questions": ["a", "b", "c"]}
    assert [q for q, _ in sub_queries(parsed, limit=2)] == ["a", "b"]


def test_malformed_sub_questions_are_ignored():
    assert sub_queries({"sub_questions": [3, {"paper": "x"}, None]}, limit=2) == []
