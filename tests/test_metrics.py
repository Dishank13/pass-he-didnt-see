import numpy as np

from phds.eval.metrics import (
    ece,
    mean_ci,
    paired_diff_ci,
    receiver_nll,
    topk_accuracy,
)


def test_topk_accuracy():
    probs = np.array([[0.1, 0.6, 0.3, 0.0], [0.5, 0.2, 0.3, 0.0]])
    target = np.array([2, 1])
    np.testing.assert_array_equal(topk_accuracy(probs, target, 1), [0, 0])
    np.testing.assert_array_equal(topk_accuracy(probs, target, 2), [1, 0])
    np.testing.assert_array_equal(topk_accuracy(probs, target, 3), [1, 1])


def test_topk_accuracy_ties_scored_by_expectation():
    uniform = np.array([[0.25, 0.25, 0.25, 0.25]])
    for target in range(4):  # array position must not matter
        assert np.isclose(topk_accuracy(uniform, np.array([target]), 1)[0], 0.25)
        assert np.isclose(topk_accuracy(uniform, np.array([target]), 3)[0], 0.75)
    partial = np.array([[0.5, 0.2, 0.2, 0.1]])  # target tied with one other, one above
    assert np.isclose(topk_accuracy(partial, np.array([2]), 2)[0], 0.5)


def test_receiver_nll_perfect_prediction_is_zero():
    probs = np.array([[0.0, 1.0]])
    assert receiver_nll(probs, np.array([1]))[0] < 1e-6


def test_ece_zero_for_perfectly_calibrated_bins():
    p = np.repeat([0.2, 0.8], 500)
    y = np.concatenate([np.r_[np.ones(100), np.zeros(400)], np.r_[np.ones(400), np.zeros(100)]])
    assert ece(p, y, n_bins=2) < 1e-9


def test_cluster_bootstrap_is_wider_for_correlated_groups():
    rng = np.random.default_rng(0)
    # 40 matches x 10 corners, with a strong match effect -> corners within a match correlate
    match_effect = np.repeat(rng.normal(0, 1, 40), 10)
    values = match_effect + rng.normal(0, 0.1, 400)
    by_match = np.repeat(np.arange(40), 10)
    independent = np.arange(400)
    _, lo_m, hi_m = mean_ci(values, by_match, n_boot=500)
    _, lo_i, hi_i = mean_ci(values, independent, n_boot=500)
    assert (hi_m - lo_m) > 2 * (hi_i - lo_i)


def test_paired_difference_detects_constant_offset():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, 300)
    groups = np.repeat(np.arange(30), 10)
    point, lo, _ = paired_diff_ci(a + 0.1, a, groups, n_boot=300)
    assert abs(point - 0.1) < 1e-9 and lo > 0.09  # shared noise cancels exactly
