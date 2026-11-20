import sys
import types

import pytest
from fastapi.testclient import TestClient

import rag_settings
from main import app
from rag import ABSTAIN, LLMUnavailable, OllamaClient, Rag

HITS = [
    {"title": "Canberra", "url": "http://x/1", "text": "Canberra is the capital of Australia."},
    {"title": "Sydney", "url": "http://x/2", "text": "Sydney is the largest city in Australia."},
]


class FakeLLM:
    def __init__(self, reply=None, error=None):
        self.reply = reply
        self.error = error
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        if self.error:
            raise self.error
        return self.reply, {"model": "fake"}


def client_with(hits, llm):
    app.state.rag = Rag(lambda q: hits, llm)
    return TestClient(app)


def test_rag_answers_and_resolves_citations_to_sources():
    client = client_with(HITS, FakeLLM("Canberra is the capital [1][2]."))
    body = client.get("/rag", params={"q": "Capital?"}).json()
    assert body["answer"] == "Canberra is the capital [1][2]."
    assert body["abstained"] is False
    assert body["citations"] == [
        {"number": 1, "title": "Canberra", "url": "http://x/1"},
        {"number": 2, "title": "Sydney", "url": "http://x/2"},
    ]
    assert [s["title"] for s in body["sources"]] == ["Canberra", "Sydney"]
    assert body["model"] == rag_settings.RAG_MODEL


def test_rag_reports_an_abstention():
    body = client_with(HITS, FakeLLM(ABSTAIN)).get("/rag", params={"q": "Who?"}).json()
    assert body["abstained"] is True
    assert body["answer"] is None
    assert body["citations"] == []


def test_citations_to_sources_that_do_not_exist_are_dropped():
    body = client_with(HITS, FakeLLM("It is Rome [9].")).get("/rag", params={"q": "Capital?"}).json()
    assert body["answer"] == "It is Rome [9]."
    assert body["citations"] == []


def test_rag_does_not_call_the_model_when_nothing_was_retrieved():
    llm = FakeLLM("anything")
    body = client_with([], llm).get("/rag", params={"q": "Capital?"}).json()
    assert body["abstained"] is True
    assert llm.calls == []


def test_a_missing_model_server_gives_503_with_a_message():
    client = client_with(HITS, FakeLLM(error=LLMUnavailable("Start Ollama.")))
    response = client.get("/rag", params={"q": "Capital?"})
    assert response.status_code == 503
    assert response.json()["detail"] == "Start Ollama."


def test_rag_reports_timings():
    body = client_with(HITS, FakeLLM("x [1]")).get("/rag", params={"q": "Q"}).json()
    assert body["retrieve_ms"] >= 0
    assert body["llm_ms"] >= 0


def test_blank_and_missing_and_overlong_questions_are_rejected():
    client = client_with(HITS, FakeLLM("x [1]"))
    assert client.get("/rag", params={"q": "   "}).status_code == 400
    assert client.get("/rag").status_code == 422
    assert client.get("/rag", params={"q": "x" * 501}).status_code == 422


def fake_ollama(chat):
    module = types.ModuleType("ollama")

    class ResponseError(Exception):
        pass

    module.ResponseError = ResponseError
    module.chat = chat
    return module


def test_a_connection_failure_becomes_llm_unavailable(monkeypatch):
    def chat(**kwargs):
        raise ConnectionError("no server")

    monkeypatch.setitem(sys.modules, "ollama", fake_ollama(chat))
    with pytest.raises(LLMUnavailable):
        OllamaClient("m").generate([])


def test_a_missing_model_becomes_llm_unavailable(monkeypatch):
    module = fake_ollama(None)

    def chat(**kwargs):
        raise module.ResponseError("model not found")

    module.chat = chat
    monkeypatch.setitem(sys.modules, "ollama", module)
    with pytest.raises(LLMUnavailable):
        OllamaClient("m").generate([])


def test_a_missing_package_becomes_llm_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "ollama", None)
    with pytest.raises(LLMUnavailable):
        OllamaClient("m").generate([])


def test_a_dict_style_reply_is_read(monkeypatch):
    def chat(**kwargs):
        return {
            "message": {"content": "hi [1]"},
            "total_duration": 2_000_000_000,
            "eval_count": 5,
            "prompt_eval_count": 50,
        }

    monkeypatch.setitem(sys.modules, "ollama", fake_ollama(chat))
    text, info = OllamaClient("m").generate([])
    assert text == "hi [1]"
    assert info["seconds"] == 2.0
    assert info["output_tokens"] == 5
    assert info["prompt_tokens"] == 50