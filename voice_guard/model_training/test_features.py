"""Tests for features.py, including the jitter/shimmer/HNR addition
(remediation track 2, see docs/CRITICAL-entity-vs-style-confound.md §4)."""
from __future__ import annotations

import numpy as np

from features import (
    N_LFCC,
    SAMPLE_RATE,
    extract_features,
    extract_physio,
    extract_prosody,
)


def _synthetic_voiced_tone(f0: float = 150.0, duration_s: float = 3.0,
                            jitter_frac: float = 0.0, amp_wobble_frac: float = 0.0,
                            noise_amp: float = 0.0, seed: int = 0) -> np.ndarray:
    """A deterministic near-periodic tone standing in for voiced speech:
    a fundamental + 2 harmonics, optional per-cycle period jitter and
    amplitude wobble (shimmer), plus optional additive white noise (for
    HNR). Used because it gives known-shape expectations (near-zero
    jitter/shimmer, high HNR when noise_amp=0) without needing a real
    speech corpus checked into the repo."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SAMPLE_RATE)
    out = np.zeros(n)
    t = 0.0
    i = 0
    base_period = 1.0 / f0
    while i < n:
        period = base_period * (1.0 + jitter_frac * (rng.random() - 0.5) * 2)
        amp = 1.0 + amp_wobble_frac * (rng.random() - 0.5) * 2
        n_samples_cycle = max(1, int(round(period * SAMPLE_RATE)))
        cycle_t = np.arange(n_samples_cycle) / SAMPLE_RATE
        cycle = amp * (0.6 * np.sin(2 * np.pi * f0 * cycle_t)
                        + 0.3 * np.sin(2 * np.pi * 2 * f0 * cycle_t)
                        + 0.1 * np.sin(2 * np.pi * 3 * f0 * cycle_t))
        end = min(n, i + n_samples_cycle)
        out[i:end] = cycle[: end - i]
        i = end
    if noise_amp > 0:
        out = out + rng.normal(0, noise_amp, size=n)
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float64)


def test_extract_physio_shape_and_order():
    pcm = _synthetic_voiced_tone()
    physio = extract_physio(pcm)
    assert physio.shape == (3,)


def test_clean_periodic_tone_has_low_jitter_and_shimmer_high_hnr():
    pcm = _synthetic_voiced_tone(jitter_frac=0.0, amp_wobble_frac=0.0, noise_amp=0.0)
    jitter, shimmer, hnr = extract_physio(pcm)
    assert jitter < 0.02
    # Shimmer here is frame-RMS-based (not per-cycle peak amplitude, see
    # extract_physio's docstring), so even a perfectly amplitude-stable
    # tone shows nonzero apparent shimmer from frame/cycle-boundary
    # misalignment. 0.05 is loose enough to absorb that windowing noise
    # while still catching a real amplitude-stability regression.
    assert shimmer < 0.05
    assert hnr > 15.0


def test_jittery_tone_has_higher_jitter_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(jitter_frac=0.0))
    jittery = extract_physio(_synthetic_voiced_tone(jitter_frac=0.08, seed=1))
    assert jittery[0] > clean[0]


def test_wobbly_amplitude_has_higher_shimmer_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(amp_wobble_frac=0.0))
    wobbly = extract_physio(_synthetic_voiced_tone(amp_wobble_frac=0.10, seed=2))
    assert wobbly[1] > clean[1]


def test_noisy_tone_has_lower_hnr_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(noise_amp=0.0))
    noisy = extract_physio(_synthetic_voiced_tone(noise_amp=0.15, seed=3))
    assert noisy[2] < clean[2]


def test_silence_returns_zeros_not_nan_or_exception():
    pcm = np.zeros(SAMPLE_RATE * 3, dtype=np.float64)
    physio = extract_physio(pcm)
    assert np.all(np.isfinite(physio))
    assert np.array_equal(physio, np.zeros(3))


def test_short_buffer_returns_zeros():
    pcm = np.zeros(100, dtype=np.float64)
    physio = extract_physio(pcm)
    assert np.array_equal(physio, np.zeros(3))


def test_extract_features_is_now_66_dimensional_in_documented_order():
    pcm = _synthetic_voiced_tone()
    feats = extract_features(pcm)
    assert feats.shape == (66,)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    np.testing.assert_allclose(feats[N_LFCC:N_LFCC + 3], prosody, rtol=1e-5)
    np.testing.assert_allclose(feats[N_LFCC + 3:N_LFCC + 6], physio, rtol=1e-5)
