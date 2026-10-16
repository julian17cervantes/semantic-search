from chunking import chunk_text


def test_blank_text_gives_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("  \n \n ") == []


def test_short_text_is_one_chunk():
    assert chunk_text("A short paragraph.") == ["A short paragraph."]


def test_short_paragraphs_are_packed_together():
    text = "First paragraph.\nSecond paragraph."
    assert chunk_text(text, max_chars=100) == ["First paragraph.\nSecond paragraph."]


def test_no_chunk_exceeds_the_limit():
    text = "word " * 500
    chunks = chunk_text(text, max_chars=100)
    assert all(len(c) <= 100 for c in chunks)


def test_splitting_never_loses_or_cuts_words():
    text = "alpha beta gamma delta " * 100
    chunks = chunk_text(text, max_chars=60)
    assert " ".join(chunks).split() == text.split()