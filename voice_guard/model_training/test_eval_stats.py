"""eval_stats: the vectorized compute_eer must equal the pre-2026-09-11 loop
exactly; the other statistics are checked against brute force / scipy."""
from __future__ import annotations

import numpy as np
import pytest

from eval_stats import (
    auc_separability,
    balanced_accuracy,
    bootstrap_eer_ci,
    compute_eer,
    compute_eer_threshold,
    confusion_matrix,
    rates_at_threshold,
    spearman_rho,
)


def _old_compute_eer(scores, labels):
    """The original O(n * thresholds) loop from the retired train.py."""
    order = np.argsort(scores)
    scores, labels = scores[order], labels[order]
    n_pos = labels.sum()
    if n_pos == 0 or n_pos == len(labels):
        return float("nan")
    best_gap, best_eer = float("inf"), float("nan")
    for t in np.unique(scores):
        far = (scores[labels == 0] >= t).mean()
        frr = (scores[labels == 1] < t).mean()
        if abs(far - frr) < best_gap:
            best_gap, best_eer = abs(far - frr), (far + frr) / 2
    return float(best_eer)


@pytest.mark.parametrize("seed", range(6))
def test_vectorized_eer_equals_old_loop(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(20, 400))
    labels = rng.integers(0, 2, n)
    scores = np.round(rng.normal(labels * rng.uniform(0, 2), 1.0), int(rng.integers(1, 4)))  # rounding -> ties
    assert compute_eer(scores, labels) == pytest.approx(_old_compute_eer(scores, labels), abs=1e-12)


def test_eer_extremes_and_degenerate():
    assert compute_eer(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 0.0
    assert np.isnan(compute_eer(np.array([0.1, 0.2]), np.array([0, 0])))


def test_eer_threshold_reproduces_its_rates():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    s = rng.normal(y, 1.0)
    eer, t = compute_eer_threshold(s, y)
    r = rates_at_threshold(s, y, t)
    assert (r["fpr"] + r["fnr"]) / 2 == pytest.approx(eer)


def test_bootstrap_groups_by_source_and_is_deterministic():
    rng = np.random.default_rng(2)
    files = np.repeat([f"f{i}" for i in range(80)], 3)
    y = np.repeat(rng.integers(0, 2, 80), 3)
    p = rng.normal(y, 1.0)
    a = bootstrap_eer_ci(p, y, files, n_bootstrap=200, seed=5)
    b = bootstrap_eer_ci(p, y, files, n_bootstrap=200, seed=5)
    assert a["n_sources"] == 80
    assert a["ci_lo"] <= a["point"] <= a["ci_hi"]
    assert np.array_equal(a["samples"], b["samples"])


def test_spearman_matches_scipy_with_ties():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x = rng.integers(0, 10, 300).astype(float)
    y = x + rng.normal(0, 3, 300)
    assert spearman_rho(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-10)
    assert np.isnan(spearman_rho(np.ones(10), np.arange(10.0)))


def test_auc_matches_brute_force():
    rng = np.random.default_rng(4)
    v = rng.integers(0, 5, 120).astype(float)
    y = rng.integers(0, 2, 120)
    pos, neg = v[y == 1], v[y == 0]
    brute = ((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg))
    assert auc_separability(v, y) == pytest.approx(brute)


def test_balanced_accuracy_and_confusion():
    yt = np.array([0, 0, 0, 0, 1, 1])
    yp = np.array([0, 0, 0, 1, 1, 0])
    assert balanced_accuracy(yt, yp) == pytest.approx((0.75 + 0.5) / 2)
    assert confusion_matrix(yt, yp) == [[3, 1], [1, 1]]
