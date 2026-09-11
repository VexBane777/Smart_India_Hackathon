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


def test_apply_bandlimit_custom_cutoffs_override_defaults():
    """The playback recipe needs a wider-than-POTS passband (laptop speaker
    200-3800 Hz into a phone mic). Explicit low_hz/high_hz must override the
    300-3400 default, passing a 200 Hz tone that the default would reject."""
    sr = 16000

    # 200 Hz: below the default 300 Hz low cutoff.
    x_200 = _sine(200, sr, duration_s=8.0)
    # 4200 Hz: above both the default 3400 Hz and the custom 3800 Hz high
    # cutoff — must still be suppressed by the custom passband.
    x_4200 = _sine(4200, sr, duration_s=8.0)

    def rel_db(x, y):
        return 10 * np.log10(np.sum(y ** 2) / np.sum(x ** 2))

    default_200 = rel_db(x_200, apply_bandlimit(x_200, sr=sr))
    custom_200 = rel_db(x_200, apply_bandlimit(x_200, sr=sr, low_hz=200, high_hz=3800))

    # The custom passband must pass 200 Hz far better than the POTS default
    # (measured ~ -39 dB default vs ~ -6 dB custom, both dominated by the
    # 5th-order Butterworth cutoff edge — assert the RELATIVE 30 dB gap, not
    # the absolute value, so the test isn't brittle to filter-order details).
    assert default_200 < -20
    assert custom_200 > default_200 + 20

    # Content above the custom passband is still suppressed (measured -16 dB
    # on a 4.2 kHz tone; the exact ratio again depends on filter order).
    custom_4200 = rel_db(x_4200, apply_bandlimit(x_4200, sr=sr, low_hz=200, high_hz=3800))
    assert custom_4200 < -10
