import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

import answer_settings as cfg
from answerer import Answerer
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
        return (
            np.array([self.scores[:k]], dtype=np.float32),
            np.array([self.ids[:k]]),
        )


class FakeReader:
    def __init__(self, reads):
        self.reads = reads
        self.calls = []

    def read(self, question, passages):
        self.calls.append((question, passages))
        return self.reads[: len(passages)]


def read(answer, score, null=0.0):
    return {"answer": answer, "score": score, "null_score": null}


CHUNKS = pd.DataFrame(
    {
        "title": ["A", "B", "C"],
        "url": ["http://x/A", "http://x/B", "http://x/C"],
        "text": ["text about A", "text about B", "text about C"],
    }
)


def make_client(reads):
    reader = FakeReader(reads)
    # the index returns chunks in the order C, A, B
    searcher = Searcher(FakeModel(), FakeIndex([2, 0, 1], [0.9, 0.5, 0.1]), CHUNKS)
    app.state.answerer = Answerer(searcher, reader)
    return TestClient(app), reader


def test_retrieve_returns_the_full_chunk_text():
    searcher = Searcher(FakeModel(), FakeIndex([2, 0, 1], [0.9, 0.5, 0.1]), CHUNKS)
    hits, _, _ = searcher.retrieve("anything", 2)
    assert [h["text"] for h in hits] == ["text about C", "text about A"]


def test_answer_combines_chunks_and_reports_the_source():
    client, _ = make_client([read("Paris", 0.9), read("paris", 0.4), read("Lyon", 0.3)])
    body = client.get("/answer", params={"q": "capital?"}).json()
    assert body["answer"] == "Paris"
    assert body["confidence"] == "high"
    assert body["source"] == {"title": "C", "url": "http://x/C"}
    assert [r["title"] for r in body["retrieved"]] == ["C", "A", "B"]


def test_low_confidence_answer_is_labeled():
    client, _ = make_client([read("Paris", 0.6)])
    body = client.get("/answer", params={"q": "capital?"}).json()
    assert body["answer"] == "Paris"
    assert body["confidence"] == "low"


def test_answer_below_the_cutoff_is_withheld():
    client, _ = make_client([read("Paris", 0.2)])
    body = client.get("/answer", params={"q": "capital?"}).json()
    assert body["answer"] is None
    assert body["confidence"] == "none"
    assert body["source"] is None


def test_titles_are_added_to_the_passages_when_enabled(monkeypatch):
    monkeypatch.setattr(cfg, "USE_TITLES", True)
    client, reader = make_client([read("Paris", 0.9)])
    client.get("/answer", params={"q": "capital?"})
    assert reader.calls[0][1] == ["C. text about C", "A. text about A", "B. text about B"]


def test_passages_are_plain_when_titles_are_disabled(monkeypatch):
    monkeypatch.setattr(cfg, "USE_TITLES", False)
    client, reader = make_client([read("Paris", 0.9)])
    client.get("/answer", params={"q": "capital?"})
    assert reader.calls[0][1] == ["text about C", "text about A", "text about B"]


def test_response_includes_timings():
    client, _ = make_client([read("Paris", 0.9)])
    body = client.get("/answer", params={"q": "capital?"}).json()
    for key in ("encode_ms", "search_ms", "read_ms"):
        assert body[key] >= 0


def test_blank_question_is_rejected():
    client, _ = make_client([])
    assert client.get("/answer", params={"q": "   "}).status_code == 400


def test_missing_question_is_rejected():
    client, _ = make_client([])
    assert client.get("/answer").status_code == 422