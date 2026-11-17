from rag import parse_answer
from rag_metrics import answer_correct, citation_valid, contains_gold, is_grounded

HITS = [
    {"title": "Canberra", "text": "Canberra is the capital city of Australia."},
    {"title": "Sydney", "text": "Sydney is the largest city in Australia."},
]


def parsed(text):
    return parse_answer(text, 2)


def test_gold_matches_whole_words_only():
    assert contains_gold("The Sun is a star.", ["Sun"])
    assert not contains_gold("Sunday is a day.", ["Sun"])


def test_gold_ignores_case_punctuation_and_articles():
    assert contains_gold("It is the Pacific Ocean!", ["pacific ocean"])


def test_numbers_match_with_units():
    assert contains_gold("Water freezes at 32 degrees.", ["32"])
    assert not contains_gold("It freezes at 132 degrees.", ["32"])


def test_any_accepted_answer_counts():
    assert contains_gold("Shakespeare wrote it.", ["William Shakespeare", "Shakespeare"])


def test_abstaining_is_never_correct():
    assert not answer_correct(parsed("I can't tell from the sources."), ["Canberra"])


def test_a_correct_answer():
    assert answer_correct(parsed("Canberra is the capital [1]."), ["Canberra"])


def test_a_wrong_answer():
    assert not answer_correct(parsed("Sydney is the capital [2]."), ["Canberra"])


def test_grounded_needs_a_cited_source_that_contains_the_answer():
    assert is_grounded(parsed("Canberra is the capital [1]."), HITS, ["Canberra"])
    assert not is_grounded(parsed("Canberra is the capital [2]."), HITS, ["Canberra"])


def test_an_answer_without_a_citation_is_not_grounded():
    assert not is_grounded(parsed("Canberra is the capital."), HITS, ["Canberra"])


def test_an_invalid_citation_does_not_ground_an_answer():
    assert not is_grounded(parsed("Canberra is the capital [3]."), HITS, ["Canberra"])


def test_citation_validity():
    assert citation_valid(parsed("Canberra [1]."))
    assert not citation_valid(parsed("Canberra."))
    assert not citation_valid(parsed("Canberra [1][3]."))
    assert not citation_valid(parsed("I can't tell from the sources."))