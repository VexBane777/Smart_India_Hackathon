"""
Unit tests for telechannel/stages/noise.py.
"""

import numpy as np
import pytest

from telechannel.stages.noise import apply_noise


def _rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


def test_apply_noise_output_same_length_as_input():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(2000)

    y = apply_noise(x, "white", snr_db=10)

    assert len(y) == len(x)


def test_apply_noise_achieves_requested_snr_white():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(20000)
    snr_db = 10

    y = apply_noise(x, "white", snr_db=snr_db)
    added_noise = y - x

    signal_rms = _rms(x)
    noise_rms = _rms(added_noise)
    achieved_snr_db = 20 * np.log10(signal_rms / noise_rms)

    assert abs(achieved_snr_db - snr_db) <= 1.0


def test_apply_noise_10db_snr_means_signal_power_10x_noise_power():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(20000)

    y = apply_noise(x, "white", snr_db=10)
    added_noise = y - x

    signal_power = np.mean(np.square(x))
    noise_power = np.mean(np.square(added_noise))
    power_ratio_db = 10 * np.log10(signal_power / noise_power)

    # 10 dB power ratio == 10x power, within the ±1 dB tolerance used
    # elsewhere for SNR checks.
    assert abs(power_ratio_db - 10.0) <= 1.0


def test_apply_noise_pink_achieves_requested_snr():
    rng = np.random.default_rng(3)
    x = rng.standard_normal(20000)
    snr_db = 5

    y = apply_noise(x, "pink", snr_db=snr_db)
    added_noise = y - x

    signal_rms = _rms(x)
    noise_rms = _rms(added_noise)
    achieved_snr_db = 20 * np.log10(signal_rms / noise_rms)

    assert abs(achieved_snr_db - snr_db) <= 1.0


def test_apply_noise_unknown_type_raises():
    x = np.zeros(100)
    with pytest.raises(ValueError):
        apply_noise(x, "purple", snr_db=10)


def test_apply_noise_injected_rng_is_reproducible():
    x = np.random.default_rng(0).standard_normal(1000)

    y1 = apply_noise(x, "white", snr_db=10, rng=np.random.default_rng(123))
    y2 = apply_noise(x, "white", snr_db=10, rng=np.random.default_rng(123))

    assert np.array_equal(y1, y2)


def test_apply_noise_pink_injected_rng_is_reproducible():
    x = np.random.default_rng(0).standard_normal(1000)

    y1 = apply_noise(x, "pink", snr_db=10, rng=np.random.default_rng(123))
    y2 = apply_noise(x, "pink", snr_db=10, rng=np.random.default_rng(123))

    assert np.array_equal(y1, y2)
