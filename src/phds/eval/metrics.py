"""Metrics with honest uncertainty.

Receiver prediction (a distribution over candidate nodes per corner):
    top-k accuracy, negative log-likelihood.
Shot prediction (one probability per corner):
    log loss, Brier score, ROC AUC, expected calibration error (ECE).

Uncertainty: **cluster bootstrap by match**. Corners from one match share teams,
routines and camera, so they're not independent. Resampling individual corners
would understate the variance and make CIs too narrow. We resample *matches*
with replacement and take every corner of each drawn match.

Model comparisons use the **paired** bootstrap: both models are scored on the
same resampled matches, and we report the CI of the *difference*. That is much
tighter than comparing two separate CIs, because shared difficulty cancels out.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import roc_auc_score


# --- receiver prediction -------------------------------------------------------------
def topk_accuracy(probs: np.ndarray, target: np.ndarray, k: int = 1) -> np.ndarray:
    """Per-corner probability that the target is among the k highest-scoring nodes.

    probs: [N, M], non-candidates must already be 0. target: [N] node index.

    Ties are scored by their *expected* value under a random tiebreak. With g nodes
    strictly above the target and t nodes tied with it (target included), the target
    lands in the top k with probability clip((k - g) / t, 0, 1). A plain argsort would
    break ties by array order instead. That made a uniform baseline look good on
    whichever team StatsBomb happens to list first in the frame.
    """
    p_t = probs[np.arange(len(target)), target][:, None]
    greater = (probs > p_t).sum(axis=1)
    ties = np.isclose(probs, p_t, rtol=0, atol=1e-12).sum(axis=1)
    return np.clip((k - greater) / ties, 0.0, 1.0)


def receiver_nll(probs: np.ndarray, target: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """Per-corner negative log-likelihood of the true first toucher."""
    return -np.log(probs[np.arange(len(target)), target] + eps)


# --- shot prediction -----------------------------------------------------------------
def log_loss_each(p: np.ndarray, y: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def brier_each(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    return (p - y) ** 2


def auc(p: np.ndarray, y: np.ndarray, w: np.ndarray | None = None) -> float:
    return float(roc_auc_score(y, p, sample_weight=w)) if 0 < y.sum() < len(y) else np.nan


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 10, w: np.ndarray | None = None):
    """Equal-mass bins: (mean predicted, observed rate, total weight) per bin.

    Equal-mass (quantile) bins, not equal-width: predictions usually concentrate in a
    narrow range, and equal-width bins would leave the extremes nearly empty and noisy.
    Optional sample weights `w` (e.g. selection-correction weights).
    """
    w = np.ones_like(p, dtype=float) if w is None else np.asarray(w, float)
    order = np.argsort(p)
    bins = np.array_split(order, n_bins)
    return [(np.average(p[b], weights=w[b]), np.average(y[b], weights=w[b]), w[b].sum())
            for b in bins if len(b) and w[b].sum() > 0]  # fmt: skip


def ece(p: np.ndarray, y: np.ndarray, n_bins: int = 10, w: np.ndarray | None = None) -> float:
    """Expected calibration error: weight-averaged |predicted - observed| over bins."""
    rows = reliability_bins(p, y, n_bins, w)
    n = sum(c for _, _, c in rows)
    return float(sum(c * abs(pm - om) for pm, om, c in rows) / n)


# --- bootstrap -----------------------------------------------------------------------
def _match_resamples(groups: np.ndarray, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == g) for g in range(len(uniq))]
    for _ in range(n_boot):
        drawn = rng.integers(0, len(uniq), len(uniq))
        yield np.concatenate([members[g] for g in drawn])


def cluster_bootstrap(
    stat: Callable[[np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """(point estimate, lower, upper) for `stat(indices)`, resampling whole groups (matches)."""
    all_idx = np.arange(len(groups))
    point = stat(all_idx)
    samples = np.array([stat(idx) for idx in _match_resamples(groups, n_boot, seed)])
    samples = samples[~np.isnan(samples)]
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


def mean_ci(values: np.ndarray, groups: np.ndarray, **kw) -> tuple[float, float, float]:
    """Cluster-bootstrap CI for the mean of a per-corner metric."""
    return cluster_bootstrap(lambda idx: float(values[idx].mean()), groups, **kw)


def weighted_mean_ci(values: np.ndarray, weights: np.ndarray, groups: np.ndarray, **kw):
    """Cluster-bootstrap CI for a weighted mean of a per-example metric."""
    return cluster_bootstrap(
        lambda idx: float(np.average(values[idx], weights=weights[idx])), groups, **kw
    )


def paired_diff_ci(a: np.ndarray, b: np.ndarray, groups: np.ndarray, **kw):
    """CI for mean(a) - mean(b), with a and b per-corner metrics on the same corners."""
    return cluster_bootstrap(lambda idx: float(a[idx].mean() - b[idx].mean()), groups, **kw)


def fmt_ci(ci: tuple[float, float, float], pct: bool = False, digits: int = 3) -> str:
    m = 100 if pct else 1
    d = 1 if pct else digits
    return f"{ci[0] * m:.{d}f} [{ci[1] * m:.{d}f}, {ci[2] * m:.{d}f}]"
