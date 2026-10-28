from answer_metrics import is_correct, normalize


def test_normalize_lowercases_and_drops_punctuation_and_articles():
    assert normalize("The Eiffel Tower!") == "eiffel tower"


def test_exact_match():
    assert is_correct("Leonardo da Vinci", ["Leonardo da Vinci"])


def test_match_ignores_case_and_punctuation():
    assert is_correct("george washington.", ["George Washington"])


def test_any_accepted_answer_counts():
    assert is_correct("Shakespeare", ["William Shakespeare", "Shakespeare"])


def test_gold_inside_a_longer_answer_counts():
    assert is_correct("the Pacific Ocean", ["Pacific Ocean"])


def test_number_with_unit_counts():
    assert is_correct("100 °C", ["100"])


def test_partial_word_does_not_count():
    assert not is_correct("Washingtonian", ["Washington"])


def test_wrong_answer():
    assert not is_correct("Lhotse", ["Mount Everest"])


def test_empty_prediction_is_wrong():
    assert not is_correct("", ["Paris"])


def test_long_rambling_answer_is_wrong():
    assert not is_correct("Paris " + "and so on " * 10, ["Paris"])