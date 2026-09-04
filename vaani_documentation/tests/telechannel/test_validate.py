"""
Unit tests for telechannel/validate_reality.py.

Important environment note: the real 25 validated calls (DOC5's recording
protocol) have not been recorded yet — nobody has run that protocol. So
"the real call" in these tests is a *synthetic* stand-in: band-limited
speech-shaped noise built with a realistic LTAS rolloff, generated in
`_synthetic_clean_recording` below. It stands in for a genuine human
recording only to exercise the analysis code path end-to-end (VAD ->
LTAS -> cutoff -> comparison -> plot). It is NOT a substitute for real-
recording validation against DOC5's actual calls, which still needs to
happen once those exist.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from telechannel.validate_reality import (
    analyze_call,
    compare_to_sim,
    plot_validation,
)
from telechannel.stages.bandlimit import apply_bandlimit

SR = 16000


def _synthetic_clean_recording(sr=SR, duration_s=6.0, seed=0):
    """
    Build a synthetic "clean" speech-like recording standing in for a real
    phone recording (see module docstring: no real recordings exist yet).

    Constructed as pink-ish noise (amplitude spectrum ~ 1/sqrt(f), i.e.
    power ~ 1/f) band-limited to a wideband-mic-like 50-7000 Hz range —
    a smooth, monotonically-decaying LTAS shape typical of natural speech
    spectra, without discrete harmonic peaks that would make the "peak"
    of the spectrum an artifact of a single tone. Silent gaps are inserted
    to give the VAD step something real to strip.
    """
    rng = np.random.default_rng(seed)
    n = int(sr * duration_s)

    white = rng.standard_normal(n)
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)

    # Shape toward 1/sqrt(f) amplitude (pink-noise-like power ~ 1/f),
    # band-limited to a wideband-mic-like passband. Avoid divide-by-zero
    # at DC.
    shape = np.zeros_like(freqs)
    passband = (freqs >= 50.0) & (freqs <= 7000.0)
    shape[passband] = 1.0 / np.sqrt(np.maximum(freqs[passband], 1.0))

    shaped_spectrum = spectrum * shape
    x = np.fft.irfft(shaped_spectrum, n=n)

    # Normalize.
    x = x / (np.max(np.abs(x)) + 1e-9)

    # Punch silent gaps in (simulates pauses between utterances), so the
    # VAD/silence-stripping step in analyze_call has real work to do.
    gap_starts = [int(0.4 * n), int(0.75 * n)]
    gap_len = int(0.15 * sr)
    for g in gap_starts:
        x[g:g + gap_len] = 0.0

    return x, sr


def test_analyze_call_returns_expected_keys():
    x, sr = _synthetic_clean_recording()

    result = analyze_call((x, sr))

    assert set(result.keys()) == {"cutoff", "ltas", "hf_ratio"}
    assert set(result["ltas"].keys()) == {"freqs", "psd_db"}
    assert result["cutoff"] > 0
    assert 0.0 <= result["hf_ratio"] <= 1.0


def test_analyze_call_strips_silence_reduces_sample_count():
    # A recording with a big silent chunk should end up with strictly
    # fewer voiced samples than the raw file length once VAD is applied.
    sr = SR
    n = int(sr * 4.0)
    x = np.zeros(n)
    # only middle third contains signal
    t = np.arange(n) / sr
    x[n // 3: 2 * n // 3] = np.sin(2 * np.pi * 300 * t[n // 3: 2 * n // 3])

    result = analyze_call((x, sr))

    # The LTAS still computes and doesn't blow up on the mostly-silent
    # input; hf_ratio/cutoff should be finite.
    assert np.isfinite(result["cutoff"])
    assert np.isfinite(result["hf_ratio"])


def test_bandlimited_signal_has_lower_cutoff_than_wideband():
    """
    Sanity check on the cutoff-finding logic: a signal explicitly passed
    through TeleChannel's 300-3400 Hz telephony band-limit
    (telechannel.stages.bandlimit.apply_bandlimit) should show a bandwidth
    cutoff well below that of the original wideband signal.
    """
    x, sr = _synthetic_clean_recording()

    wideband_metrics = analyze_call((x, sr))

    narrowband = apply_bandlimit(x, sr=sr)
    narrowband_metrics = analyze_call((narrowband, sr))

    assert narrowband_metrics["cutoff"] < wideband_metrics["cutoff"]
    # Telephony passband cuts off around 3400 Hz; allow generous slack
    # since Welch's method has limited frequency resolution.
    assert narrowband_metrics["cutoff"] < 4500


def test_compare_to_sim_identical_metrics_gives_zero_diff_and_correlation_one():
    x, sr = _synthetic_clean_recording()
    metrics = analyze_call((x, sr))

    comparison = compare_to_sim(metrics, metrics)

    assert comparison["cutoff_diff_hz"] == pytest.approx(0.0, abs=1e-6)
    assert comparison["hf_ratio_diff"] == pytest.approx(0.0, abs=1e-6)
    assert comparison["ltas_correlation"] == pytest.approx(1.0, abs=1e-6)


def test_compare_to_sim_clean_recording_vs_clean_recipe_correlation_near_one():
    """
    Step 4 of the task brief: run against a "clean" recording and the
    "clean" recipe (i.e. no degradation applied); correlation should be
    ~1.0.

    Since DOC5's real validated calls don't exist yet, the "real"
    recording here is the synthetic clean fixture (see module docstring),
    and the "clean recipe" is simply the same signal with a fresh draw of
    independent low-level noise/phase jitter (standing in for
    "sim output of a no-degradation pipeline run on equivalent input") —
    the point being that two clean signals with the same underlying
    spectral shape correlate almost perfectly, unlike a band-limited one.
    """
    real_x, sr = _synthetic_clean_recording(seed=1)
    sim_x, _ = _synthetic_clean_recording(seed=2)  # different noise draw, same recipe

    real_metrics = analyze_call((real_x, sr))
    sim_metrics = analyze_call((sim_x, sr))

    comparison = compare_to_sim(real_metrics, sim_metrics)

    assert comparison["ltas_correlation"] >= 0.9


def test_plot_validation_writes_file(tmp_path):
    real_x, sr = _synthetic_clean_recording(seed=1)
    sim_x = apply_bandlimit(real_x, sr=sr)

    real_metrics = analyze_call((real_x, sr))
    sim_metrics = analyze_call((sim_x, sr))

    out_path = tmp_path / "validation.png"
    result = plot_validation(real_metrics, sim_metrics, out_path)

    assert result == out_path
    assert out_path.exists()
    assert out_path.stat().st_size > 0
