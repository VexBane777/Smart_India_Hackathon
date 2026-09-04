"""
Unit tests for telechannel/stages/bandlimit.py.
"""

import numpy as np

from telechannel.stages.bandlimit import apply_bandlimit


def _sine(freq_hz, sr, duration_s=1.0):
    t = np.arange(int(sr * duration_s)) / sr
    return np.sin(2 * np.pi * freq_hz * t)


def test_apply_bandlimit_output_same_length_as_input():
    sr = 16000
    x = _sine(1000, sr)

    y = apply_bandlimit(x, sr=sr)

    assert len(y) == len(x)


def test_apply_bandlimit_attenuates_out_of_band_5khz_tone_by_40db():
    sr = 16000
    # A long duration is used so that sosfiltfilt's edge-padding transient
    # (unavoidable for a sine wave with a discontinuous-derivative onset)
    # is negligible relative to the ~9.6s of accurately-filtered steady
    # state; over a short window the transient dominates total energy and
    # masks the filter's true (much stronger) steady-state attenuation.
    x = _sine(5000, sr, duration_s=10.0)

    y = apply_bandlimit(x, sr=sr)

    input_energy = np.sum(x ** 2)
    output_energy = np.sum(y ** 2)
    relative_db = 10 * np.log10(output_energy / input_energy)

    assert relative_db < -40


def test_apply_bandlimit_passes_in_band_1khz_tone():
    sr = 16000
    x = _sine(1000, sr)

    y = apply_bandlimit(x, sr=sr)

    input_energy = np.sum(x ** 2)
    output_energy = np.sum(y ** 2)
    relative_db = 10 * np.log10(output_energy / input_energy)

    # In-band tone should pass through with only modest attenuation.
    assert relative_db > -3
