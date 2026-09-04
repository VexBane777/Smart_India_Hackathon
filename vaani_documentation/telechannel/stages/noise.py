"""
noise.py — Additive noise stage.

Mixes a noise signal into the input at a requested signal-to-noise ratio
(SNR), by computing the RMS of both signals and scaling the noise so the
resulting mix hits the target SNR in dB.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import numpy as np


def _rms(signal):
    return np.sqrt(np.mean(np.square(signal)))


def apply_noise(x, noise_type, snr_db, rng=None):
    """
    Add noise to `x` at the requested SNR (in dB).

    Args:
        x: 1-D clean input signal (numpy array).
        noise_type: One of "white" or "pink". Selects the spectral shape
            of the generated noise.
        snr_db: Target signal-to-noise ratio in dB. E.g. 10 dB means the
            signal power is 10x the noise power.
        rng: Optional numpy.random.Generator for reproducibility. If not
            given, a fresh `np.random.default_rng()` is used (the previous,
            non-reproducible behavior).

    Returns:
        numpy array the same length as `x`: signal + scaled noise.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if rng is None:
        rng = np.random.default_rng()

    if noise_type == "white":
        noise = rng.standard_normal(n)
    elif noise_type == "pink":
        noise = _pink_noise(n, rng)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type!r}")

    signal_rms = _rms(x)
    noise_rms = _rms(noise)

    # snr_db = 20*log10(signal_rms / target_noise_rms)
    target_noise_rms = signal_rms / (10 ** (snr_db / 20))
    if noise_rms > 0:
        noise = noise * (target_noise_rms / noise_rms)

    return x + noise


def _pink_noise(n, rng=None):
    """
    Generate approximate pink (1/f) noise of length `n` via spectral
    shaping of white noise in the frequency domain.
    """
    if rng is None:
        rng = np.random.default_rng()
    white = np.fft.rfft(rng.standard_normal(n))
    freqs = np.arange(1, len(white) + 1)
    pink_spectrum = white / np.sqrt(freqs)
    pink = np.fft.irfft(pink_spectrum, n)
    return pink
