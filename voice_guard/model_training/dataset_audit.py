"""
Corpus-integrity checks that `test_pipeline_smoke.py` structurally cannot
catch (it only ever sees synthetic 4s tones through the real code path).

Found 2026-09-10, the hard way, after four retraining attempts in a row
regressed cross-generator held-out EER despite ever-larger training sets:
a real/fake WAV directory pair can look like a clean, balanced accent cell
by file count alone while being perfectly separable by a *technical*
recording artifact (sample rate, bit depth, channel count) that has nothing
to do with genuine spoof detection — e.g. every `fake` clip in a cell being
a single TTS engine's native 22050Hz output, resampled to 16kHz, while every
`real` clip in the same cell was recorded natively at 16kHz. A model trained
on that will happily learn "was this resampled from 22050Hz" as a proxy for
"is this fake" — scoring great on validation split from the *same*
contaminated cell, and falling apart on any real held-out benchmark that
doesn't share the same generator/sample-rate confound.

This module scans for exactly that: does any cheap technical metadata field
split real vs. fake almost perfectly within a directory pair? If so, that
pair is not safe to train on as-is, no matter how much data it adds.

**Added after the code-orange literature review (2026-09-10, later same
session):** fixing the sample-rate confound above did not fix cross-generator
generalization (it got worse). Wide-net research turned up a second,
well-documented confound class in this exact literature — leading/trailing
silence duration correlating with real/fake label (Kwak et al. 2021, "Speech
is Silver, Silence is Golden," arXiv:2106.12914: ASVspoof2019 bonafide clips
have systematically longer lead/trail silence than spoofed ones; correcting
for it moved a model's EER from 3.6% to 15.5%). Measuring our own corpus
found exactly this, and — worse — **the direction is inconsistent across
cells**: the ASVspoof-derived base corpus and the held-out ITW benchmark
both show "real has somewhat more silence," but `en_foreign`/`hi_native`
(this project's own TTS-generated accent cells) show the *opposite*
direction (fake has ~3x more trailing silence). A model can't use silence as
one consistent global shortcut across a corpus like that, but it also can't
cleanly ignore it — it's exactly the kind of contradictory, per-domain
signal literature on multi-source/negative-transfer training (e.g. the
"harmful systems"/system-fingerprint framing in recent deepfake-audio
negative-transfer work) describes as actively degrading, not just neutral.
`find_acoustic_shortcuts` below generalizes the categorical-field check
above to continuous descriptors (lead/trail silence, duration) via a
rank-based separability (AUC) measure, since these are distributions, not a
single dominant category.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

from dataset import trim_edge_silence

# A field that separates real/fake this cleanly is almost certainly a
# recording/generation-pipeline artifact, not a genuine spoof signal.
# 1.0 would be perfect separation; real spoof-detection features are never
# this clean on legitimate signal, so anything above this is suspicious.
SEPARABILITY_THRESHOLD = 0.85


def scan_technical_metadata(directory: Path, sample_n: int | None = 300, seed: int = 0) -> list[dict]:
    """Reads only WAV headers (sf.info, no decode) for up to sample_n files
    in `directory`. Returns a list of {samplerate, channels, subtype, duration}."""
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    if sample_n is not None and len(files) > sample_n:
        rng = random.Random(seed)
        files = rng.sample(files, sample_n)
    out = []
    for f in files:
        try:
            info = sf.info(str(f))
            out.append({
                "samplerate": info.samplerate,
                "channels": info.channels,
                "subtype": info.subtype,
                "duration": info.duration,
            })
        except Exception:
            continue
    return out


def _max_one_sided_fraction(values: list) -> tuple[object, float]:
    """For a categorical field, returns (most common value, fraction of
    records holding it) — used to detect near-total dominance by one value."""
    if not values:
        return None, 0.0
    counts = Counter(values)
    value, n = counts.most_common(1)[0]
    return value, n / len(values)


def find_technical_shortcuts(
    real_meta: list[dict], fake_meta: list[dict], threshold: float = SEPARABILITY_THRESHOLD
) -> list[str]:
    """Checks samplerate/channels/subtype for near-perfect real-vs-fake
    separability. Returns a list of human-readable warnings (empty if clean).

    Method: for each field, find the fake class's dominant value. If that
    value covers >=threshold of fake examples AND covers <=(1-threshold) of
    real examples (i.e. it's essentially fake-exclusive), or vice versa,
    that field is flagged as a usable shortcut."""
    if not real_meta or not fake_meta:
        return ["one or both classes have zero scannable files (see chunk-yield report)"]

    warnings = []
    for field in ("samplerate", "channels", "subtype"):
        real_vals = [m[field] for m in real_meta]
        fake_vals = [m[field] for m in fake_meta]
        for label_a, vals_a, label_b, vals_b in (
            ("fake", fake_vals, "real", real_vals),
            ("real", real_vals, "fake", fake_vals),
        ):
            dominant, frac_a = _max_one_sided_fraction(vals_a)
            frac_b = vals_b.count(dominant) / len(vals_b) if vals_b else 0.0
            if frac_a >= threshold and frac_b <= (1 - threshold):
                warnings.append(
                    f"{field}={dominant!r} covers {frac_a:.0%} of {label_a} examples but "
                    f"only {frac_b:.0%} of {label_b} examples - a model can trivially use "
                    f"{field} as a {label_a}/{label_b} shortcut instead of learning genuine "
                    f"spoof artifacts. Root-cause the source pipeline (likely a single TTS "
                    f"engine's native output rate leaking through) before training on this "
                    f"directory pair."
                )
                break  # one direction is enough per field
    return warnings


def _leading_trailing_silence(
    path: Path, thresh: float = 0.01, post_trim: bool = True
) -> tuple[float, float, float]:
    """(leading_silence_s, trailing_silence_s, duration_s). Decodes the whole
    file (unlike scan_technical_metadata's header-only reads), so this is
    meaningfully slower — keep sample_n modest when calling this.

    `post_trim=True` (default) applies dataset.trim_edge_silence first, i.e.
    measures what train.py actually trains on since 2026-09-10's fix, not
    the raw on-disk file — that's the number that should determine whether a
    corpus is safe to train on now. Pass False to see the raw, pre-fix
    numbers (e.g. for before/after comparisons)."""
    data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if post_trim:
        data = trim_edge_silence(data, sr=sr, thresh=thresh, rng=np.random.default_rng(0))
    above = np.abs(data) > thresh
    if not above.any():
        return len(data) / sr, len(data) / sr, len(data) / sr
    first = np.argmax(above) / sr
    last = (len(data) - 1 - np.argmax(above[::-1])) / sr
    return float(first), float((len(data) / sr) - last), float(len(data) / sr)


def scan_acoustic_metadata(
    directory: Path, sample_n: int | None = 150, seed: int = 0, post_trim: bool = True
) -> list[dict]:
    """Decodes up to sample_n files in `directory` and returns
    [{lead_silence_s, trail_silence_s, duration_s}, ...]. Slower than
    scan_technical_metadata (full decode, not just headers) — used for the
    continuous-descriptor shortcut check, not the cheap preflight pass.
    See `_leading_trailing_silence` for what `post_trim` means."""
    directory = Path(directory)
    files = sorted(directory.glob("*.wav"))
    if sample_n is not None and len(files) > sample_n:
        rng = random.Random(seed)
        files = rng.sample(files, sample_n)
    out = []
    for f in files:
        try:
            lead, trail, duration = _leading_trailing_silence(f, post_trim=post_trim)
            out.append({
                "lead_silence_s": lead,
                "trail_silence_s": trail,
                "duration_s": duration,
            })
        except Exception:
            continue
    return out


def _rank_auc(vals_a: list[float], vals_b: list[float]) -> float:
    """P(a random value from vals_a > a random value from vals_b), via the
    Mann-Whitney U / rank-sum identity — no scipy dependency. 0.5 = no
    separability; near 0 or 1 = near-perfect separability (usable shortcut,
    in whichever direction)."""
    n_a, n_b = len(vals_a), len(vals_b)
    if n_a == 0 or n_b == 0:
        return 0.5
    combined = sorted((v, 0) for v in vals_a) + sorted((v, 1) for v in vals_b)
    combined.sort(key=lambda x: x[0])
    # average ranks for ties
    ranks = [0.0] * len(combined)
    i = 0
    while i < len(combined):
        j = i
        while j < len(combined) and combined[j][0] == combined[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-indexed rank average over the tie block
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j
    rank_sum_a = sum(r for r, (_, grp) in zip(ranks, combined) if grp == 0)
    u_a = rank_sum_a - n_a * (n_a + 1) / 2.0
    return u_a / (n_a * n_b)


# A continuous descriptor this separable between classes is very unlikely to
# be genuine spoof signal at this scale — real spoofing artifacts are noisy
# and don't cleanly separate two ~150-file samples via rank position alone.
ACOUSTIC_AUC_THRESHOLD = 0.85


def find_acoustic_shortcuts(
    real_meta: list[dict], fake_meta: list[dict], auc_threshold: float = ACOUSTIC_AUC_THRESHOLD
) -> list[str]:
    """Continuous-descriptor counterpart to find_technical_shortcuts: flags
    lead/trail silence or duration if it separates real vs. fake far better
    than chance (AUC far from 0.5), in either direction. See this module's
    docstring for why this matters as much as the categorical check —
    ASVspoof-lineage corpora are documented (Kwak et al. 2021) to have a
    real/bonafide-has-more-silence bias, and this project's own TTS-generated
    accent cells were measured to have the *opposite* bias — a corpus mixing
    both isn't simply "no shortcut," it's an inconsistent one."""
    if not real_meta or not fake_meta:
        return ["one or both classes have zero scannable files (see chunk-yield report)"]

    warnings = []
    for field in ("lead_silence_s", "trail_silence_s", "duration_s"):
        real_vals = [m[field] for m in real_meta]
        fake_vals = [m[field] for m in fake_meta]
        auc = _rank_auc(real_vals, fake_vals)  # P(real > fake)
        if auc >= auc_threshold or auc <= (1 - auc_threshold):
            direction = "real > fake" if auc > 0.5 else "fake > real"
            warnings.append(
                f"{field}: separability AUC={auc:.2f} ({direction}) - a model can use "
                f"{field} as a real/fake shortcut instead of learning genuine spoof "
                f"artifacts. real median={np.median(real_vals):.3f} vs "
                f"fake median={np.median(fake_vals):.3f}. See Kwak et al. 2021 "
                f"(arXiv:2106.12914) for this exact confound class in ASVspoof-lineage "
                f"corpora."
            )
    return warnings


def audit_directory_pair(
    real_dir: Path, fake_dir: Path, sample_n: int = 300, seed: int = 0,
    check_acoustic: bool = True, acoustic_sample_n: int = 150,
) -> list[str]:
    real_meta = scan_technical_metadata(real_dir, sample_n=sample_n, seed=seed)
    fake_meta = scan_technical_metadata(fake_dir, sample_n=sample_n, seed=seed)
    issues = find_technical_shortcuts(real_meta, fake_meta)
    if check_acoustic:
        real_ac = scan_acoustic_metadata(real_dir, sample_n=acoustic_sample_n, seed=seed)
        fake_ac = scan_acoustic_metadata(fake_dir, sample_n=acoustic_sample_n, seed=seed)
        issues += find_acoustic_shortcuts(real_ac, fake_ac)
    return issues


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--fake", type=Path, required=True)
    args = ap.parse_args()

    issues = audit_directory_pair(args.real, args.fake)
    if issues:
        print(f"SHORTCUT RISK in {args.real} vs {args.fake}:")
        for w in issues:
            print(f"  - {w}")
        raise SystemExit(1)
    print(f"OK: no obvious technical shortcut found in {args.real} vs {args.fake}")
