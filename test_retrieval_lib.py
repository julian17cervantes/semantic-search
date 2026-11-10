import numpy as np
import pytest

from retrieval_lib import bootstrap_ci, first_rank, hit_at, meets_adoption_rule, mrr, paired_bootstrap_diff, rerank_head, rrf

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

def test_paired_difference_of_identical_runs_is_zero():
    ranks = [1, 2, None, 4]
    diff, low, high = paired_bootstrap_diff(ranks, ranks, lambda r: hit_at(r, 5))
    assert (diff, low, high) == (0.0, 0.0, 0.0)


def test_paired_difference_shows_a_consistent_improvement():
    worse = [5, 5, 5, 5, 5, 5]
    better = [1, 1, 1, 1, 1, 1]
    diff, low, high = paired_bootstrap_diff(worse, better, mrr)
    assert diff == pytest.approx(0.8)
    assert low > 0


def test_paired_difference_needs_the_same_queries():
    with pytest.raises(ValueError):
        paired_bootstrap_diff([1, 2], [1], mrr)

def test_rerank_head_reorders_only_the_head():
    assert rerank_head([10, 11, 12, 13], [0.1, 0.9, 0.5], depth=3) == [11, 12, 10, 13]


def test_rerank_head_keeps_the_original_order_on_ties():
    assert rerank_head([10, 11, 12], [1.0, 1.0, 1.0], depth=3) == [10, 11, 12]


def test_rerank_head_handles_a_depth_larger_than_the_list():
    assert rerank_head([10, 11], [0.2, 0.8], depth=20) == [11, 10]


def test_rerank_head_needs_one_score_per_id():
    with pytest.raises(ValueError):
        rerank_head([10, 11, 12], [0.1, 0.2], depth=3)


def test_adoption_rule():
    assert meets_adoption_rule(0.05, 0.0, 40.0)
    assert not meets_adoption_rule(0.02, 0.0, 40.0)  # gain too small
    assert not meets_adoption_rule(0.05, -0.02, 40.0)  # hit@10 dropped
    assert not meets_adoption_rule(0.05, 0.0, 150.0)  # too slow
    assert not meets_adoption_rule(0.03, 0.0, 100.0)  # latency must be strictly under 100