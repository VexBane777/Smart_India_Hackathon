"""
tests for app/components/spectrogram.py — mel computation correctness.
"""

import numpy as np

from app.components.spectrogram import N_MELS, SR, compute_mel_db


class TestComputeMel:
    def test_shape(self):
        audio = np.random.default_rng(0).standard_normal(SR * 2) * 0.05
        mel = compute_mel_db(audio, SR)
        assert mel.shape[0] == N_MELS
        assert mel.shape[1] > 0

    def test_tone_louder_than_silence(self):
        t = np.arange(SR) / SR
        tone = 0.5 * np.sin(2 * np.pi * 440.0 * t)
        silence = np.zeros(SR)
        assert compute_mel_db(tone).max() > compute_mel_db(silence).max()

    def test_finite(self):
        audio = np.random.default_rng(1).standard_normal(SR) * 0.1
        assert np.isfinite(compute_mel_db(audio)).all()

    def test_short_input_padded(self):
        mel = compute_mel_db(np.array([0.1, -0.1, 0.05]))
        assert np.isfinite(mel).all()
