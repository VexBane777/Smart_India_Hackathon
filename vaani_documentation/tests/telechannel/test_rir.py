"""
Unit tests for telechannel/stages/rir.py.
"""

import numpy as np

from telechannel.stages.rir import apply_rir


def test_apply_rir_output_same_length_as_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(4000)
    rir = rng.standard_normal(500) * np.exp(-np.linspace(0, 5, 500))

    y = apply_rir(x, rir, wet_gain=0.7)

    assert len(y) == len(x)


def test_apply_rir_dirac_delta_recovers_rir():
    # A room impulse response whose direct path (peak) is the very first
    # sample, as is typical for a real RIR (direct sound arrives first,
    # reflections follow).
    n = 256
    rir = np.exp(-np.linspace(0, 8, n))
    rir[0] = 1.0  # ensure the peak is unambiguously at index 0

    delta = np.zeros(n)
    delta[0] = 1.0

    y = apply_rir(delta, rir, wet_gain=1.0)

    expected = rir / np.max(np.abs(rir)) * 0.9
    assert np.allclose(y, expected)


def test_apply_rir_wet_gain_zero_returns_dry_signal():
    rng = np.random.default_rng(1)
    x = rng.standard_normal(1000)
    rir = rng.standard_normal(200) * np.exp(-np.linspace(0, 5, 200))

    y = apply_rir(x, rir, wet_gain=0.0)

    assert np.allclose(y, x)
