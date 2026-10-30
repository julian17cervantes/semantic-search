import pytest

import answer_settings as cfg
from answering import combine, passages_for_reading


def read(answer, score, null=0.0):
    return {"answer": answer, "score": score, "null_score": null}


def test_scores_of_identical_answers_are_added():
    reads = [read("Paris", 0.4), read("paris.", 0.4), read("Lyon", 0.7)]
    result = combine(reads, ["A", "B", "C"])
    assert result["answer"] == "Paris"
    assert result["score"] == pytest.approx(0.8)


def test_answer_is_ignored_when_no_answer_scores_higher():
    assert combine([read("Paris", 0.3, null=0.6)], ["A"])["answer"] is None


def test_below_the_cutoff_gives_no_answer(monkeypatch):
    monkeypatch.setattr(cfg, "NO_ANSWER_BELOW", 0.5)
    result = combine([read("Paris", 0.45)], ["A"])
    assert result["answer"] is None
    assert result["confidence"] == "none"


def test_confidence_levels(monkeypatch):
    monkeypatch.setattr(cfg, "NO_ANSWER_BELOW", 0.5)
    monkeypatch.setattr(cfg, "CONFIDENT_FROM", 1.0)
    assert combine([read("Paris", 0.7)], ["A"])["confidence"] == "low"
    assert combine([read("Paris", 1.2)], ["A"])["confidence"] == "high"


def test_source_is_the_best_chunk_for_the_winning_answer():
    reads = [read("Paris", 0.3), read("Paris", 0.6), read("Lyon", 0.5)]
    assert combine(reads, ["A", "B", "C"])["source"] == "B"


def test_no_reads_gives_no_answer():
    assert combine([], [])["answer"] is None


def test_titles_are_added_only_when_enabled(monkeypatch):
    monkeypatch.setattr(cfg, "USE_TITLES", True)
    assert passages_for_reading(["Paris"], ["Capital of France."]) == ["Paris. Capital of France."]
    monkeypatch.setattr(cfg, "USE_TITLES", False)
    assert passages_for_reading(["Paris"], ["Capital of France."]) == ["Capital of France."]