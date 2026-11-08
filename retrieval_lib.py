import numpy as np

TOP = 10


def first_rank(chunk_ids, expected, chunk_titles, top=TOP):
    seen = []
    for i in chunk_ids:
        title = chunk_titles[i]
        if title not in seen:
            seen.append(title)
        if len(seen) == top:
            break
    for rank, title in enumerate(seen, 1):
        if title in expected:
            return rank
    return None


def rrf(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, 1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hit_at(ranks, k):
    return sum(r is not None and r <= k for r in ranks) / len(ranks)


def mrr(ranks):
    return sum(1 / r for r in ranks if r) / len(ranks)


def bootstrap_ci(ranks, metric, n=2000, seed=0, level=0.95):
    rng = np.random.default_rng(seed)
    ranks = list(ranks)
    size = len(ranks)
    values = []
    for _ in range(n):
        sample = [ranks[i] for i in rng.integers(0, size, size)]
        values.append(metric(sample))
    low, high = np.percentile(values, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(low), float(high)

def paired_bootstrap_diff(ranks_a, ranks_b, metric, n=2000, seed=0, level=0.95):
    if len(ranks_a) != len(ranks_b):
        raise ValueError("both runs must score the same queries")
    rng = np.random.default_rng(seed)
    a, b = list(ranks_a), list(ranks_b)
    size = len(a)
    diffs = []
    for _ in range(n):
        idx = rng.integers(0, size, size)
        diffs.append(metric([b[i] for i in idx]) - metric([a[i] for i in idx]))
    low, high = np.percentile(diffs, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return metric(b) - metric(a), float(low), float(high)