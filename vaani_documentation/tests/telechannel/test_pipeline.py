"""
Unit / integration tests for telechannel/pipeline.py.

Two kinds of tests here, mirroring test_codec.py's split:

1. Tests that need no real ffmpeg binary: they exercise the full recipe
   chain (RIR -> Noise -> Mic -> Codec -> Loss -> Bandlimit) either via the
   `clean` recipe (whose codec list is empty by design) or by monkeypatching
   `telechannel.pipeline.codec_roundtrip` to a length-preserving passthrough
   so the orchestration logic — including tandem transcoding and the
   bandlimit stage — is exercised for real without needing ffmpeg.

2. Tests that genuinely invoke ffmpeg (the `whatsapp` recipe's Step 4
   end-to-end check). These are gated with
   `pytest.mark.skipif(shutil.which("ffmpeg") is None, ...)`, exactly like
   test_codec.py, and are SKIPPED (not passed, not failed) in an
   environment without a real ffmpeg binary on PATH — which is the case in
   the environment this file was authored in. See task-8-report.md for
   what was verified for real vs. skipped.
"""

import shutil

import numpy as np
import pytest

from telechannel import pipeline
from telechannel.pipeline import load_config, process_clip

HAS_FFMPEG = shutil.which("ffmpeg") is not None

ALL_RECIPES = ["clean", "pstn", "gsm_2g", "cellular_3g", "volte", "whatsapp", "tandem_xnet"]


def _sine(freq_hz, sr, duration_s, amplitude=0.2):
    t = np.arange(int(sr * duration_s)) / sr
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def _band_energy(x, sr, lo_hz, hi_hz):
    """Energy of `x` in the [lo_hz, hi_hz) band, via an rFFT power spectrum."""
    spec = np.fft.rfft(x)
    power = np.abs(spec) ** 2
    freqs = np.fft.rfftfreq(len(x), d=1.0 / sr)
    band = (freqs >= lo_hz) & (freqs < hi_hz)
    return np.sum(power[band])


def _identity_codec_roundtrip(x, codec, bitrate):
    """Stand-in for codec_roundtrip that needs no ffmpeg: passes audio
    through unchanged (same length), so callers can verify pipeline
    orchestration (stage order, tandem transcoding call count, bandlimit
    behavior) without a real codec round trip."""
    return np.asarray(x, dtype=np.float64).copy()


# ---------------------------------------------------------------------------
# Config / recipe definitions
# ---------------------------------------------------------------------------


def test_load_config_defines_all_required_recipes():
    config = load_config()
    recipes = config["recipes"]

    for name in ["pstn", "gsm_2g", "cellular_3g", "volte", "whatsapp", "tandem_xnet"]:
        assert name in recipes, f"missing required recipe: {name}"

    # The ffmpeg-free control recipe used to exercise orchestration without
    # a real ffmpeg binary.
    assert "clean" in recipes
    assert recipes["clean"]["codec"] == []


def test_tandem_xnet_recipe_has_multiple_codec_entries():
    config = load_config()
    codec_list = config["recipes"]["tandem_xnet"]["codec"]
    assert isinstance(codec_list, list)
    assert len(codec_list) >= 2


# ---------------------------------------------------------------------------
# Orchestration tests that need no real ffmpeg
# ---------------------------------------------------------------------------


def test_process_clip_unknown_recipe_raises_keyerror():
    x = np.zeros(100)
    with pytest.raises(KeyError):
        process_clip(x, "not_a_real_recipe")


def test_process_clip_clean_recipe_end_to_end_no_ffmpeg_needed():
    sr = 16000
    x = _sine(440, sr, duration_s=2.0)

    y = process_clip(x, "clean", sr=sr, rng=np.random.default_rng(0))

    assert y.ndim == 1
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_process_clip_is_reproducible_given_same_rng_seed():
    sr = 16000
    x = _sine(440, sr, duration_s=2.0)

    y1 = process_clip(x, "clean", sr=sr, rng=np.random.default_rng(42))
    y2 = process_clip(x, "clean", sr=sr, rng=np.random.default_rng(42))

    assert np.array_equal(y1, y2)


def test_process_clip_different_seeds_diverge():
    sr = 16000
    x = _sine(440, sr, duration_s=2.0)

    y1 = process_clip(x, "clean", sr=sr, rng=np.random.default_rng(1))
    y2 = process_clip(x, "clean", sr=sr, rng=np.random.default_rng(2))

    assert not np.array_equal(y1, y2)


@pytest.mark.parametrize("recipe_name", ALL_RECIPES)
def test_all_recipes_run_end_to_end_with_mocked_codec(monkeypatch, recipe_name):
    """
    Exercises every recipe's full stage chain (RIR -> Noise -> Mic -> Codec
    -> Loss -> Bandlimit) with codec_roundtrip mocked to a length-preserving
    passthrough, so this proves the orchestration/config-wiring logic for
    every recipe (including tandem_xnet's 2-codec list and whatsapp's
    bandlimit-enabled config) without requiring ffmpeg.
    """
    monkeypatch.setattr(pipeline, "codec_roundtrip", _identity_codec_roundtrip)

    sr = 16000
    x = _sine(440, sr, duration_s=2.0)

    y = process_clip(x, recipe_name, sr=sr, rng=np.random.default_rng(0))

    assert y.ndim == 1
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))


def test_tandem_transcoding_calls_codec_roundtrip_once_per_list_entry(monkeypatch):
    calls = []

    def recording_codec_roundtrip(x, codec, bitrate):
        calls.append((codec, bitrate))
        return np.asarray(x, dtype=np.float64).copy()

    monkeypatch.setattr(pipeline, "codec_roundtrip", recording_codec_roundtrip)

    sr = 16000
    x = _sine(440, sr, duration_s=2.0)
    config = load_config()
    expected = config["recipes"]["tandem_xnet"]["codec"]

    process_clip(x, "tandem_xnet", sr=sr, rng=np.random.default_rng(0))

    assert len(calls) == len(expected)
    assert [c[0] for c in calls] == [entry["codec"] for entry in expected]


def test_single_codec_recipe_calls_codec_roundtrip_once(monkeypatch):
    calls = []

    def recording_codec_roundtrip(x, codec, bitrate):
        calls.append((codec, bitrate))
        return np.asarray(x, dtype=np.float64).copy()

    monkeypatch.setattr(pipeline, "codec_roundtrip", recording_codec_roundtrip)

    sr = 16000
    x = _sine(440, sr, duration_s=2.0)
    process_clip(x, "pstn", sr=sr, rng=np.random.default_rng(0))

    assert len(calls) == 1
    assert calls[0][0] == "pcm_mulaw"


def test_clean_recipe_calls_codec_roundtrip_zero_times(monkeypatch):
    calls = []
    monkeypatch.setattr(
        pipeline, "codec_roundtrip",
        lambda x, codec, bitrate: calls.append(codec) or x,
    )

    sr = 16000
    x = _sine(440, sr, duration_s=2.0)
    process_clip(x, "clean", sr=sr, rng=np.random.default_rng(0))

    assert calls == []


# ---------------------------------------------------------------------------
# Step 4: whatsapp end-to-end, including the 2s bandlimit boundary check
# ---------------------------------------------------------------------------


def test_whatsapp_2s_clip_bandlimit_no_edge_artifact_mocked_codec(monkeypatch):
    """
    Carried-forward review finding: bandlimit.py's unit test only validated
    attenuation with a 10s synthetic tone; this verifies its behavior on a
    real 2s window (the system's actual window size) as part of the
    whatsapp recipe's end-to-end run (codec mocked so this needs no
    ffmpeg — the orchestration and bandlimit stage are exercised for real;
    only the codec round-trip is stubbed).
    """
    monkeypatch.setattr(pipeline, "codec_roundtrip", _identity_codec_roundtrip)

    sr = 16000
    duration_s = 2.0
    # Mix of an in-band tone (1kHz) and an out-of-band tone (5kHz), quiet
    # enough that mic's up-to-+12dB gain doesn't push it into clipping.
    x = _sine(1000, sr, duration_s, amplitude=0.15) + _sine(5000, sr, duration_s, amplitude=0.15)

    y = process_clip(x, "whatsapp", sr=sr, rng=np.random.default_rng(7))

    # No edge/transient artifact: finite, no wraparound clipping/blow-up.
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
    assert np.max(np.abs(y)) < 5.0  # generous bound; a windowing blow-up would be orders larger

    # Compare against the same recipe/seed with bandlimit disabled, to
    # isolate the bandlimit stage's real effect at this short duration
    # (everything upstream of it — RIR/noise/mic/loss — is identical).
    config = load_config()
    no_band_config = {
        "version": config["version"],
        "sample_rate": config["sample_rate"],
        "rooms": config["rooms"],
        "recipes": {
            **config["recipes"],
            "whatsapp_no_band": {
                **config["recipes"]["whatsapp"],
                "bandlimit": {"enabled": False},
            },
        },
    }
    y_no_band = process_clip(
        x, "whatsapp_no_band", sr=sr, config=no_band_config, rng=np.random.default_rng(7)
    )

    out_of_band_lo, out_of_band_hi = 4000, 8000
    in_band_lo, in_band_hi = 300, 3400

    banded_oob = _band_energy(y, sr, out_of_band_lo, out_of_band_hi)
    unbanded_oob = _band_energy(y_no_band, sr, out_of_band_lo, out_of_band_hi)
    banded_inband = _band_energy(y, sr, in_band_lo, in_band_hi)

    # The bandlimit stage should still measurably suppress the out-of-band
    # content relative to leaving it in, even at 2s.
    relative_db = 10 * np.log10(banded_oob / unbanded_oob)
    assert relative_db < 0, (
        f"expected bandlimit to reduce out-of-band energy at 2s, "
        f"measured {relative_db:.1f} dB"
    )
    # And in-band energy should still dominate the band-limited output.
    assert banded_inband > banded_oob


@pytest.mark.skipif(not HAS_FFMPEG, reason="ffmpeg not installed in this environment")
def test_whatsapp_2s_clip_real_ffmpeg_output_16khz_mono():
    """
    Brief Step 4, run for real: process a 2s clip through the `whatsapp`
    recipe (real codec_roundtrip, real ffmpeg/libopus) and verify it passes
    through every stage and the output is 16kHz mono.
    """
    sr = 16000
    x = _sine(440, sr, duration_s=2.0, amplitude=0.15)

    y = process_clip(x, "whatsapp", sr=sr, rng=np.random.default_rng(3))

    assert y.ndim == 1  # mono
    # codec_roundtrip pads/trims back to the input length at OUTPUT_SR
    # (16000Hz), so length staying at 2s worth of samples confirms 16kHz.
    assert len(y) == len(x)
    assert np.all(np.isfinite(y))
