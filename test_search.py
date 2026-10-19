from conftest import FakeIndex, FakeModel
from searcher import Searcher


def test_searcher_returns_results_in_index_order(searcher):
    result = searcher.search("anything", k=2)
    assert [r["title"] for r in result["results"]] == ["C", "A"]
    assert result["results"][0]["score"] == 0.9


def test_searcher_skips_missing_results(chunks):
    searcher = Searcher(FakeModel(), FakeIndex([1, -1], [0.8, 0.0]), chunks)
    result = searcher.search("anything", k=2)
    assert [r["title"] for r in result["results"]] == ["B"]


def test_searcher_reports_timings(searcher):
    result = searcher.search("anything")
    assert result["encode_ms"] >= 0
    assert result["search_ms"] >= 0


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_search_endpoint(client):
    response = client.get("/search", params={"q": "hello", "k": 2})
    assert response.status_code == 200
    assert [r["title"] for r in response.json()["results"]] == ["C", "A"]


def test_empty_query_is_rejected(client):
    assert client.get("/search", params={"q": "   "}).status_code == 400


def test_missing_query_is_rejected(client):
    assert client.get("/search").status_code == 422


def test_k_above_limit_is_rejected(client):
    assert client.get("/search", params={"q": "hello", "k": 21}).status_code == 422


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Semantic Search" in response.text