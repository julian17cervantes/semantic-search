import answer_settings as cfg
from answer_metrics import normalize


def passages_for_reading(titles, texts):
    if cfg.USE_TITLES:
        return [f"{title}. {text}" for title, text in zip(titles, texts)]
    return list(texts)


def combine(reads, titles):
    totals, best_chunk = {}, {}
    for read, title in zip(reads, titles):
        if not read["answer"] or read["score"] <= read["null_score"]:
            continue
        key = normalize(read["answer"])
        if not key:
            continue
        totals[key] = totals.get(key, 0.0) + read["score"]
        if key not in best_chunk or read["score"] > best_chunk[key][0]:
            best_chunk[key] = (read["score"], read["answer"], title)

    if not totals:
        return {"answer": None, "score": 0.0, "confidence": "none", "source": None}

    key = max(totals, key=totals.get)
    score = totals[key]
    if score < cfg.NO_ANSWER_BELOW:
        return {"answer": None, "score": round(score, 3), "confidence": "none", "source": None}

    _, text, title = best_chunk[key]
    confidence = "high" if score >= cfg.CONFIDENT_FROM else "low"
    return {"answer": text, "score": round(score, 3), "confidence": confidence, "source": title}