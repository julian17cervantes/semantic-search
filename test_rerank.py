import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import rerank_settings as rs
from main import app
from searcher import Searcher


class FakeModel:
    def encode(self, texts, normalize_embeddings=True):
        return np.zeros((len(texts), 4), dtype=np.float32)


class FakeIndex:
    def __init__(self, ids, scores):
        self.ids = ids
        self.scores = scores

    def search(self, vector, k):
        return (np.array([self.scores[:k]], dtype=np.float32), np.array([self.ids[:k]]))


class FakeReranker:
    def __init__(self, scores):
        self.scores = scores
        self.calls = []

    def predict(self, pairs):
        self.calls.append(list(pairs))
        return np.array(self.scores[: len(pairs)], dtype=np.float32)


CHUNKS = pd.DataFrame(
    {
        "title": ["A", "B", "C"],
        "url": ["http://x/A", "http://x/B", "http://x/C"],
        "text": ["text about A", "text about B", "text about C"],
    }
)


def make(reranker_scores):
    """The index returns chunks in the order C, A, B, so reranker scores follow that order."""
    reranker = FakeReranker(reranker_scores) if reranker_scores is not None else None
    app.state.searcher = Searcher(
        FakeModel(), FakeIndex([2, 0, 1], [0.9, 0.5, 0.1]), CHUNKS, reranker=reranker
    )
    return TestClient(app), reranker


def titles(body):
    return [r["title"] for r in body["results"]]


def test_rerank_reorders_the_results():
    client, _ = make([0.1, 0.9, 0.5])
    body = client.get("/search", params={"q": "x", "k": 3, "rerank": "true"}).json()
    assert titles(body) == ["A", "B", "C"]
    assert body["reranked"] is True
    assert body["rerank_ms"] >= 0


def test_rerank_keeps_the_semantic_score_and_adds_its_own():
    client, _ = make([0.1, 0.9, 0.5])
    first = client.get("/search", params={"q": "x", "k": 3, "rerank": "true"}).json()["results"][0]
    assert first["title"] == "A"
    assert first["score"] == pytest.approx(0.5)
    assert first["rerank_score"] == pytest.approx(0.9, abs=1e-3)


def test_rerank_is_off_by_default():
    client, reranker = make([0.1, 0.9, 0.5])
    body = client.get("/search", params={"q": "x", "k": 3}).json()
    assert titles(body) == ["C", "A", "B"]
    assert body["reranked"] is False
    assert "rerank_ms" not in body
    assert reranker.calls == []


def test_asking_for_rerank_without_a_reranker_is_a_clear_error():
    client, _ = make(None)
    assert client.get("/search", params={"q": "x", "rerank": "true"}).status_code == 400


def test_only_the_top_candidates_are_reranked(monkeypatch):
    monkeypatch.setattr(rs, "RERANK_DEPTH", 2)
    client, reranker = make([0.1, 0.9, 0.5])
    body = client.get("/search", params={"q": "x", "k": 3, "rerank": "true"}).json()
    assert len(reranker.calls[0]) == 2
    assert titles(body) == ["A", "C", "B"]


def test_titles_are_given_to_the_reranker_when_enabled(monkeypatch):
    monkeypatch.setattr(rs, "RERANK_USE_TITLES", True)
    client, reranker = make([0.5, 0.5, 0.5])
    client.get("/search", params={"q": "x", "k": 1, "rerank": "true"})
    assert reranker.calls[0][0] == ("x", "C. text about C")


def test_plain_text_is_given_to_the_reranker_when_titles_are_disabled(monkeypatch):
    monkeypatch.setattr(rs, "RERANK_USE_TITLES", False)
    client, reranker = make([0.5, 0.5, 0.5])
    client.get("/search", params={"q": "x", "k": 1, "rerank": "true"})
    assert reranker.calls[0][0] == ("x", "text about C")