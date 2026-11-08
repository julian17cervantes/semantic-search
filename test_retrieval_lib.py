import numpy as np
import pytest

from retrieval_lib import bootstrap_ci, first_rank, hit_at, mrr, rrf

TITLES = np.array(["A", "A", "B", "C"])


def test_first_rank_counts_each_article_once():
    assert first_rank([0, 1, 2], {"B"}, TITLES) == 2


def test_first_rank_is_none_when_the_article_is_missing():
    assert first_rank([0, 1, 2, 3], {"Z"}, TITLES) is None


def test_first_rank_only_looks_at_the_top_articles():
    assert first_rank([0, 2, 3], {"C"}, TITLES, top=2) is None
    assert first_rank([0, 2, 3], {"C"}, TITLES, top=3) == 3


def test_first_rank_accepts_any_of_several_titles():
    assert first_rank([2, 3], {"X", "C"}, TITLES) == 2


def test_rrf_ranks_chunks_found_by_both_lists_first():
    assert rrf([[1, 2], [2, 3]]) == [2, 1, 3]


def test_hit_at_and_mrr():
    ranks = [1, None, 3, 10]
    assert hit_at(ranks, 1) == 0.25
    assert hit_at(ranks, 5) == 0.5
    assert hit_at(ranks, 10) == 0.75
    assert mrr(ranks) == pytest.approx((1 + 1 / 3 + 0.1) / 4)


def test_bootstrap_interval_of_identical_results_is_a_point():
    low, high = bootstrap_ci([1, 1, 1, 1], lambda r: hit_at(r, 1))
    assert (low, high) == (1.0, 1.0)


def test_bootstrap_interval_contains_the_estimate_and_is_repeatable():
    ranks = [1, 2, None, 1, 5, None, 1, 3]

    def metric(r):
        return hit_at(r, 5)

    low, high = bootstrap_ci(ranks, metric)
    assert low <= metric(ranks) <= high
    assert bootstrap_ci(ranks, metric) == (low, high)