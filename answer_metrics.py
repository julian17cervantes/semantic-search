import re

ARTICLES = {"a", "an", "the"}


def normalize(text):
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return " ".join(word for word in text.split() if word not in ARTICLES)


def is_correct(prediction, gold_answers, max_words=12):
    """True if the prediction contains an accepted answer and is not a long ramble."""
    predicted = normalize(prediction)
    if not predicted or len(predicted.split()) > max_words:
        return False
    padded = f" {predicted} "
    for gold in gold_answers:
        wanted = normalize(gold)
        if wanted and f" {wanted} " in padded:
            return True
    return False