import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

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
        return (np.array([self.scores[:k]], dtype=np.float32), np.array([self.ids[:k]]),)


@pytest.fixture()
def chunks():
    return pd.DataFrame({"title": ["A", "B", "C"], "url": ["http://x/A", "http://x/B", "http://x/C"], "text": ["text about A", "text about B", "text about C"],})


@pytest.fixture()
def searcher(chunks):
    return Searcher(FakeModel(), FakeIndex([2, 0, 1], [0.9, 0.5, 0.1]), chunks)


@pytest.fixture()
def client(searcher):
    app.state.searcher = searcher
    return TestClient(app)