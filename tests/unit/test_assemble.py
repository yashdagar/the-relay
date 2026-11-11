import json

import pandas as pd
import pytest

from relay import assemble as assembly
from relay.reply import Reply
from relay.tables import RESULT_COLUMNS


@pytest.fixture
def results_root(tmp_path, monkeypatch):
    monkeypatch.setattr(assembly, "ROOT", tmp_path)
    return tmp_path / "results"


def write_run(root, name, rows, replies):
    run = root / name
    run.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(run / "results.csv", index=False)
    (run / "replies.jsonl").write_text("".join(json.dumps(r) + "\n" for r in replies))


def run_files(grade_at_run_time="wrong"):
    reply = Reply(kind="refusal", reason="out_of_scope", text="no", total_ms=12.0)
    row = {
        "request_id": "retrieval-300w-k5-bm25-ood-01",
        "scenario": "retrieval",
        "source": "real",
        "start_time": 1.0,
        "retrieval_setting": "300w-k5-bm25",
        "answer_grade": grade_at_run_time,
    }
    entry = {"question_id": "ood-01", "grade": grade_at_run_time, "reply": reply.to_dict()}
    return [row], [entry]


def test_rows_are_regraded_with_the_current_grader(results_root):
    write_run(results_root, "run-a", *run_files(grade_at_run_time="wrong"))
    results = assembly.assemble({"runs": [{"dir": "run-a"}]})
    assert results.loc[0, "answer_grade"] == "refused_correctly"


def test_request_ids_are_prefixed_with_their_run(results_root):
    write_run(results_root, "run-a", *run_files())
    write_run(results_root, "run-b", *run_files())
    results = assembly.assemble({"runs": [{"dir": "run-a"}, {"dir": "run-b"}]})
    assert results["request_id"].is_unique


def test_manifest_can_relabel_a_run_scenario(results_root):
    write_run(results_root, "run-a", *run_files())
    results = assembly.assemble({"runs": [{"dir": "run-a", "scenario": "development"}]})
    assert results.loc[0, "scenario"] == "development"


def test_assembled_columns_follow_the_schema(results_root):
    write_run(results_root, "run-a", *run_files())
    assert list(assembly.assemble({"runs": [{"dir": "run-a"}]}).columns) == RESULT_COLUMNS


def test_a_reply_that_does_not_match_its_row_is_rejected(results_root):
    rows, replies = run_files()
    replies[0]["question_id"] = "att-01"
    write_run(results_root, "run-a", rows, replies)
    with pytest.raises(ValueError):
        assembly.assemble({"runs": [{"dir": "run-a"}]})
