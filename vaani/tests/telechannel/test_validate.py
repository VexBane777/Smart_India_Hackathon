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
    _find_bandwidth_cutoff,
    _strip_silence,
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


def test_strip_silence_reduces_sample_count():
    # A recording with a big silent chunk should end up with strictly
    # fewer voiced samples than the raw signal length once VAD is applied
    # (this directly exercises _strip_silence, rather than just checking
    # that downstream metrics are finite, which would also pass if
    # stripping were a no-op).
    sr = SR
    n = int(sr * 4.0)
    x = np.zeros(n)
    # only middle third contains signal
    t = np.arange(n) / sr
    x[n // 3: 2 * n // 3] = np.sin(2 * np.pi * 300 * t[n // 3: 2 * n // 3])

    stripped = _strip_silence(x, sr)

    assert len(stripped) < len(x)
    # The voiced region is ~n/3 samples; allow generous slack for the
    # librosa frame/hop granularity of the split boundaries.
    signal_len = n // 3
    assert stripped.size <= signal_len + sr  # well under the full 4 s
    assert stripped.size >= signal_len - sr


def test_analyze_call_on_mostly_silent_input_still_finite():
    # The full analyze_call pipeline should not blow up on mostly-silent
    # input; hf_ratio/cutoff should be finite even after VAD strips most
    # of the clip.
    sr = SR
    n = int(sr * 4.0)
    x = np.zeros(n)
    t = np.arange(n) / sr
    x[n // 3: 2 * n // 3] = np.sin(2 * np.pi * 300 * t[n // 3: 2 * n // 3])

    result = analyze_call((x, sr))

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


def test_find_bandwidth_cutoff_robust_to_dominant_off_band_peak():
    """
    Regression test for the original harmonic-fixture bug: an earlier
    version of _find_bandwidth_cutoff anchored its "0 dB" reference to
    the single global-argmax PSD bin. That broke when a dominant peak
    sat *outside* the frequency range that actually determines the
    signal's bandwidth (e.g. a strong low-frequency fundamental that
    band-limiting removes) — removing that dominant bin shifted which
    bin was "the peak", which shifted the entire reference level, which
    made the computed cutoff jump around for reasons having nothing to
    do with the signal's actual bandwidth.

    This test builds that exact failure shape directly against
    `_find_bandwidth_cutoff` (bypassing VAD/Welch estimation noise): a
    spectrum with real signal content up to ~2000 Hz, plus a huge
    dominant peak at ~100 Hz (well below the 300-800 Hz reference band)
    that a telephony band-limit filter would strip out entirely. The
    cutoff should reflect the ~2000 Hz signal extent in both cases, and
    should barely move when the dominant off-band peak is removed —
    proving robustness to peak-shifting, not just avoidance of it via
    fixture choice (see task-11-report.md for the harmonic-fixture
    postmortem this codifies).
    """
    freqs = np.linspace(0, 8000, 801)  # 10 Hz per bin

    def _build(include_dominant_peak):
        psd_db = np.full_like(freqs, -40.0)  # noise floor
        # Reference band + "true" signal band, both at a similar level:
        # this is the actual bandwidth we want _find_bandwidth_cutoff to
        # recover, from 300 Hz up to ~2000 Hz.
        signal_mask = (freqs >= 300) & (freqs <= 2000)
        psd_db[signal_mask] = -10.0
        if include_dominant_peak:
            # A huge peak far below the reference band — this used to be
            # the argmax bin and would have wrecked the old algorithm's
            # "0 dB" reference.
            dominant_mask = (freqs >= 90) & (freqs <= 110)
            psd_db[dominant_mask] = 20.0
        return psd_db

    cutoff_with_dominant_peak = _find_bandwidth_cutoff(freqs, _build(True))
    cutoff_without_dominant_peak = _find_bandwidth_cutoff(freqs, _build(False))

    # Both should reflect the true ~2000 Hz signal extent.
    assert 1800 <= cutoff_with_dominant_peak <= 2200
    assert 1800 <= cutoff_without_dominant_peak <= 2200

    # And, crucially, removing the dominant off-band peak (simulating
    # what band-limiting does to a fixture with a strong sub-300-Hz
    # fundamental) should barely move the cutoff.
    assert abs(cutoff_with_dominant_peak - cutoff_without_dominant_peak) < 150


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


def test_compare_to_sim_different_sample_rates_restricts_to_overlap():
    """
    compare_to_sim's docstring claims comparison happens "on their
    overlapping frequency range." All other tests use matched sample
    rates (so real_freqs == sim_freqs and this code path is never
    exercised) — this test uses genuinely different sample rates, which
    is also a realistic scenario (a real narrowband/PSTN call vs. a
    wideband simulated recipe, or vice versa), to prove overlap
    restriction actually happens instead of numpy.interp silently
    flat-extrapolating past the simulated grid's real range.
    """
    # Two independent draws of the same shaped-noise "recipe" at
    # different sample rates (16 kHz "real", 8 kHz "sim" — e.g. standing
    # in for a real wideband call vs. a narrowband-simulated one). Using
    # independent generation rather than resampling one into the other
    # avoids conflating this test with anti-aliasing-filter artifacts
    # near the lower sample rate's Nyquist edge; the point here is purely
    # to prove the mismatched-grid comparison path restricts to the true
    # overlap instead of extrapolating.
    real_x, sr_hi = _synthetic_clean_recording(sr=16000, seed=3)
    sim_x, sr_lo = _synthetic_clean_recording(sr=8000, seed=4)

    real_metrics = analyze_call((real_x, sr_hi))
    sim_metrics = analyze_call((sim_x, sr_lo))

    # Sanity: the two frequency grids really do differ in range (sim's
    # Nyquist is half of real's), so this genuinely exercises the
    # mismatched-grid branch.
    assert real_metrics["ltas"]["freqs"].max() > sim_metrics["ltas"]["freqs"].max() * 1.5

    comparison = compare_to_sim(real_metrics, sim_metrics)

    # No NaN/inf from extrapolation artifacts, and since both sides share
    # the same underlying 1/sqrt(f) shaping recipe, restricting to the
    # true overlap (0 Hz - sim's Nyquist) should give a strong positive
    # correlation despite the independent noise draws.
    assert np.isfinite(comparison["ltas_correlation"])
    assert comparison["ltas_correlation"] > 0.7


def test_compare_to_sim_raises_on_non_overlapping_frequency_grids():
    x, sr = _synthetic_clean_recording(seed=4)
    metrics = analyze_call((x, sr))

    fake_disjoint_metrics = {
        "cutoff": metrics["cutoff"],
        "hf_ratio": metrics["hf_ratio"],
        "ltas": {
            "freqs": metrics["ltas"]["freqs"] + 1_000_000.0,
            "psd_db": metrics["ltas"]["psd_db"],
        },
    }

    with pytest.raises(ValueError):
        compare_to_sim(metrics, fake_disjoint_metrics)


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
