from rag import ABSTAIN, Rag, build_messages, parse_answer

HITS = [
    {"title": "Canberra", "url": "http://x/1", "text": "Canberra is the capital of Australia."},
    {"title": "Sydney", "url": "http://x/2", "text": "Sydney is the largest city in Australia."},
]


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        return self.reply, {"model": "fake"}


def test_sources_are_numbered_from_one_in_hit_order():
    user = build_messages("Q?", HITS)[1]["content"]
    assert "[1] Canberra: Canberra is the capital of Australia." in user
    assert "[2] Sydney: Sydney is the largest city in Australia." in user
    assert user.index("[1]") < user.index("[2]")


def test_messages_carry_the_system_prompt_and_the_question():
    messages = build_messages("Who is it?", HITS)
    assert messages[0]["role"] == "system"
    assert ABSTAIN in messages[0]["content"]
    assert messages[1]["content"].endswith("Question: Who is it?")


def test_exact_abstention():
    parsed = parse_answer("I can't tell from the sources.", 2)
    assert parsed["abstained"]
    assert parsed["answer"] is None
    assert parsed["citations"] == []


def test_abstention_with_a_curly_apostrophe_and_no_period():
    assert parse_answer("I can\u2019t tell from the sources", 2)["abstained"]


def test_quoted_abstention():
    assert parse_answer('"I can\'t tell from the sources."', 2)["abstained"]


def test_abstention_followed_by_extra_text_is_still_an_abstention():
    parsed = parse_answer("I can't tell from the sources. [1] and [2] do not mention it.", 2)
    assert parsed["abstained"]
    assert parsed["citations"] == []


def test_an_answer_with_a_citation():
    parsed = parse_answer("Canberra is the capital [1].", 2)
    assert not parsed["abstained"]
    assert parsed["answer"] == "Canberra is the capital [1]."
    assert parsed["citations"] == [1]
    assert parsed["has_citation"]


def test_several_citation_styles():
    assert parse_answer("Yes [1][2].", 2)["citations"] == [1, 2]
    assert parse_answer("Yes [1, 2].", 2)["citations"] == [1, 2]
    assert parse_answer("Yes [2] and again [2].", 2)["citations"] == [2]


def test_citations_outside_the_sources_are_invalid():
    parsed = parse_answer("It is Rome [3].", 2)
    assert parsed["citations"] == []
    assert parsed["invalid_citations"] == [3]
    assert not parsed["has_citation"]
    assert parse_answer("x [0]", 2)["invalid_citations"] == [0]


def test_an_answer_without_a_citation():
    parsed = parse_answer("Canberra.", 2)
    assert not parsed["abstained"]
    assert not parsed["has_citation"]


def test_rag_sends_the_sources_to_the_model_and_parses_the_reply():
    llm = FakeLLM("Canberra is the capital [1].")
    result = Rag(lambda q: HITS, llm).answer("Capital?")
    assert result["answer"] == "Canberra is the capital [1]."
    assert result["citations"] == [1]
    assert [s["title"] for s in result["sources"]] == ["Canberra", "Sydney"]
    assert "[2] Sydney" in llm.calls[0][1]["content"]
    assert result["llm"] == {"model": "fake"}


def test_rag_reports_an_abstention():
    assert Rag(lambda q: HITS, FakeLLM(ABSTAIN)).answer("Who?")["abstained"]


def test_rag_does_not_call_the_model_when_nothing_was_retrieved():
    llm = FakeLLM("anything")
    result = Rag(lambda q: [], llm).answer("Q")
    assert result["abstained"]
    assert llm.calls == []


def test_rag_reports_timings():
    result = Rag(lambda q: HITS, FakeLLM("x [1]")).answer("Q")
    assert result["retrieve_ms"] >= 0
    assert result["llm_ms"] >= 0