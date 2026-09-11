"""
Metrics and statistical rigor for VoiceGuard evals. This is the one home
for `compute_eer` (moved here from the retired MLP `train.py`, 2026-09-11).

History: built 2026-09-10 after a fine-grained checkpoint sweep
(v8_finegrain/v8b_seed1) produced two apparent sub-baseline "wins" (0.1557,
0.1538) that did not replicate at the same training step across two
independent runs. With ~2,800 independent held-out *files* (not the ~4,700
correlated windows EER is computed over), the standard error of one EER
point estimate is roughly sqrt(EER*(1-EER)/n_files), about 0.007. Two
checkpoints less than ~1.5 points apart can be indistinguishable from
noise, and sweeping many checkpoints without accounting for that finds
*something* below any fixed threshold by chance.

`bootstrap_eer_ci` resamples at the SOURCE-FILE level. Windows from one
file (and, since v12, every channel rendition of it) are correlated, so
resampling windows independently would understate the uncertainty.

`compute_eer` was rewritten 2026-09-11 from an O(n x unique-thresholds)
loop to an O(n log n) sort/searchsorted form with identical semantics
(pinned by test_eval_stats.py against the old loop). The multi-channel
eval sets are ~10-30x larger than before, so the old form was unusable.
"""
from __future__ import annotations

import numpy as np


def _rates_at_all_thresholds(scores: np.ndarray, labels: np.ndarray):
    """(thresholds, far, frr) with far = P(real score >= t),
    frr = P(fake score < t), evaluated at every unique score."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    real = np.sort(scores[labels == 0])
    fake = np.sort(scores[labels == 1])
    thresholds = np.unique(scores)
    far = (len(real) - np.searchsorted(real, thresholds, side="left")) / max(len(real), 1)
    frr = np.searchsorted(fake, thresholds, side="left") / max(len(fake), 1)
    return thresholds, far, frr


def compute_eer(scores: np.ndarray, labels: np.ndarray) -> float:
    """Equal error rate: the threshold where false-accept rate (real flagged
    as fake) equals false-reject rate (fake cleared). Returns the mean of the
    two at the unique-score threshold minimizing |far - frr|, ties going to
    the lowest threshold (identical to the pre-2026-09-11 loop)."""
    eer, _t = compute_eer_threshold(scores, labels)
    return eer


def compute_eer_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """(eer, threshold). threshold is the score t at which the decision rule
    `score >= t -> fake` achieves the EER."""
    labels = np.asarray(labels)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), float("nan")
    thresholds, far, frr = _rates_at_all_thresholds(scores, labels)
    i = int(np.argmin(np.abs(far - frr)))  # argmin returns the first (lowest-threshold) minimum
    return float((far[i] + frr[i]) / 2), float(thresholds[i])


def rates_at_threshold(scores: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    """FPR (reals flagged, score >= threshold) and FNR (fakes missed) at a
    fixed threshold, with counts. NaN when a class is absent."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels)
    real, fake = scores[labels == 0], scores[labels == 1]
    return {
        "threshold": float(threshold),
        "fpr": float((real >= threshold).mean()) if len(real) else float("nan"),
        "fnr": float((fake < threshold).mean()) if len(fake) else float("nan"),
        "n_real": int(len(real)),
        "n_fake": int(len(fake)),
    }


def _group_indices(source_files) -> tuple[np.ndarray, list[np.ndarray]]:
    source_files = np.asarray(source_files)
    uniq, inverse = np.unique(source_files, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    groups = np.split(order, bounds)
    return uniq, groups


def bootstrap_eer_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    source_files,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """95% bootstrap CI on EER, resampled at the source-file level.
    Returns {"point", "ci_lo", "ci_hi", "std", "n_sources", "samples"}."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    point = compute_eer(probs, labels)
    uniq, groups = _group_indices(source_files)
    n_sources = len(uniq)
    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        picks = rng.integers(0, n_sources, size=n_sources)
        idx = np.concatenate([groups[p] for p in picks])
        samples[i] = compute_eer(probs[idx], labels[idx])
    return {
        "point": float(point),
        "ci_lo": float(np.nanpercentile(samples, 2.5)),
        "ci_hi": float(np.nanpercentile(samples, 97.5)),
        "std": float(np.nanstd(samples)),
        "n_sources": int(n_sources),
        "samples": samples,
    }


def cis_overlap(ci_a: dict, ci_b: dict) -> bool:
    """Whether two bootstrap CIs overlap. Non-overlap is reasonably strong
    evidence of a real difference; overlap means don't trust the ordering of
    the point estimates."""
    return not (ci_a["ci_hi"] < ci_b["ci_lo"] or ci_b["ci_hi"] < ci_a["ci_lo"])


def spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (average ranks for ties). NaN if either
    input is constant or has fewer than 3 points."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return float("nan")
    rx, ry = _average_ranks(x), _average_ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else float("nan")


def _average_ranks(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[order] = np.arange(len(a), dtype=np.float64)
    sorted_a = a[order]
    # average ranks over runs of equal values
    boundaries = np.flatnonzero(np.diff(sorted_a)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(a)]])
    for s, e in zip(starts, ends):
        if e - s > 1:
            ranks[order[s:e]] = (s + e - 1) / 2.0
    return ranks


def auc_separability(values: np.ndarray, labels: np.ndarray) -> float:
    """P(value_fake > value_real) + 0.5 P(tie): the Mann-Whitney AUC of one
    scalar for separating label 1 from label 0. 0.5 = no information."""
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    if n0 == 0 or n1 == 0:
        return float("nan")
    ranks = _average_ranks(values) + 1.0
    r1 = ranks[labels == 1].sum()
    return float((r1 - n1 * (n1 + 1) / 2) / (n0 * n1))


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> float:
    """Mean per-class recall over classes present in y_true."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    recalls = [float((y_pred[y_true == c] == c).mean()) for c in range(n_classes) if (y_true == c).any()]
    return float(np.mean(recalls)) if recalls else float("nan")


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> list[list[int]]:
    """rows = true class, cols = predicted class."""
    m = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        m[int(t), int(p)] += 1
    return m.tolist()
