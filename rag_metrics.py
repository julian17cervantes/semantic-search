from answer_metrics import normalize


def contains_gold(text, golds):
    padded = f" {normalize(text)} "
    return any(f" {normalize(g)} " in padded for g in golds if normalize(g))


def answer_correct(result, golds):
    return (not result["abstained"]) and contains_gold(result["answer"], golds)


def is_grounded(result, hits, golds):
    if result["abstained"]:
        return False
    for number in result["citations"]:
        hit = hits[number - 1]
        if contains_gold(f"{hit['title']}. {hit['text']}", golds):
            return True
    return False


def citation_valid(result):
    return (not result["abstained"]) and result["has_citation"] and not result["invalid_citations"]