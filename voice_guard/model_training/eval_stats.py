"""
Statistical rigor for held-out EER comparisons — built 2026-09-10 after a
fine-grained checkpoint sweep (v8_finegrain/v8b_seed1) produced two
apparent sub-baseline "wins" (0.1557, 0.1538) that failed to replicate at
the same training step between two independent runs. Quantified why: with
~2,802 independent held-out *files* (not the ~4,662 correlated per-file
chunks `compute_eer` is actually computed over), the standard error on a
single EER point estimate is roughly sqrt(EER*(1-EER)/n_files) ≈ 0.007 —
meaning two checkpoints differing by less than ~1.5 points can be
indistinguishable from noise. Sweeping ~100 checkpoints without correcting
for that is a multiple-comparisons trap: it will usually find *something*
that dips below any fixed threshold by chance alone.

`bootstrap_eer_ci` resamples at the FILE level (source_file), not the
window level — windows from the same clip are correlated (same recording
conditions, same generator), so resampling windows independently would
understate the true uncertainty. Each bootstrap replicate resamples files
with replacement and recomputes EER over all windows belonging to the
resampled files (with repeats), giving a distribution whose spread reflects
genuine file-level sampling variance.
"""
from __future__ import annotations

import numpy as np

from train import compute_eer


def bootstrap_eer_ci(
    probs: np.ndarray,
    labels: np.ndarray,
    source_files: list[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict:
    """Returns {"point": float, "ci_lo": float, "ci_hi": float, "std": float,
    "samples": np.ndarray} for a 95% bootstrap CI, resampled at the file
    (source_files) level, not the window level."""
    probs = np.asarray(probs)
    labels = np.asarray(labels)
    source_files = np.asarray(source_files)

    point = compute_eer(probs, labels)

    unique_sources = np.array(sorted(set(source_files)))
    indices_by_source = {s: np.where(source_files == s)[0] for s in unique_sources}
    n_sources = len(unique_sources)

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        resampled_sources = rng.choice(unique_sources, size=n_sources, replace=True)
        idx = np.concatenate([indices_by_source[s] for s in resampled_sources])
        samples[i] = compute_eer(probs[idx], labels[idx])

    return {
        "point": float(point),
        "ci_lo": float(np.percentile(samples, 2.5)),
        "ci_hi": float(np.percentile(samples, 97.5)),
        "std": float(np.std(samples)),
        "n_sources": n_sources,
        "samples": samples,
    }


def cis_overlap(ci_a: dict, ci_b: dict) -> bool:
    """Whether two bootstrap CIs overlap — a necessary (not sufficient) check
    for "these two results might not actually differ." Non-overlapping CIs
    is reasonably strong evidence of a real difference; overlapping CIs
    means don't trust the point estimates' ordering."""
    return not (ci_a["ci_hi"] < ci_b["ci_lo"] or ci_b["ci_hi"] < ci_a["ci_lo"])
