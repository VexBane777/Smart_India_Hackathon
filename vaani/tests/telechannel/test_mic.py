"""
Unit tests for telechannel/stages/mic.py.
"""

import numpy as np

from telechannel.stages.mic import apply_mic


def test_apply_mic_output_same_length_as_input():
    x = np.ones(500) * 0.1

    y = apply_mic(x, clip_prob=0.0)

    assert len(y) == len(x)


def test_apply_mic_gain_within_minus6_to_plus12_db():
    x = np.ones(1) * 0.1
    min_gain_linear = 10 ** (-6 / 20)
    max_gain_linear = 10 ** (12 / 20)

    for _ in range(200):
        y = apply_mic(x, clip_prob=0.0)
        ratio = y[0] / x[0]
        assert min_gain_linear - 1e-9 <= ratio <= max_gain_linear + 1e-9


def test_apply_mic_clipping_caps_samples_at_exactly_plus_minus_half():
    # High-amplitude signal: even the minimum gain (-6 dB, ~0.5x) keeps
    # values above the 0.5 clip limit, so with clip_prob=1.0 clipping is
    # guaranteed to engage on out-of-range samples.
    x = np.array([10.0, -10.0, 5.0, -5.0, 0.0])

    y = apply_mic(x, clip_prob=1.0)

    assert np.max(y) <= 0.5
    assert np.min(y) >= -0.5
    assert np.isclose(np.max(y), 0.5)
    assert np.isclose(np.min(y), -0.5)


def test_apply_mic_clip_prob_zero_never_clips():
    x = np.array([10.0, -10.0, 5.0, -5.0])

    for _ in range(100):
        y = apply_mic(x, clip_prob=0.0)
        assert np.max(np.abs(y)) > 0.5


def test_apply_mic_never_exceeds_full_scale_even_without_clip_prob():
    """Regression test: apply_mic's up-to-+12dB gain, when clip_prob doesn't
    engage (80% of the time at the default clip_prob=0.2), previously left
    the signal unbounded -- a 0.69-peak real recording could come out above
    1.18 after gain, well outside valid PCM range. A safety clamp to
    [-1, 1] must always apply, regardless of clip_prob / whether the
    probabilistic ±0.5 clip engaged.
    """
    # Near-full-scale input: even the minimum gain (-6dB, ~0.5x) keeps this
    # in range, but the max gain (+12dB, ~4x) would push it to ~3.6 without
    # a safety clamp.
    x = np.array([0.9, -0.9, 0.5, -0.5, 0.0])

    for seed in range(50):
        y = apply_mic(x, clip_prob=0.0, rng=np.random.default_rng(seed))
        assert np.max(np.abs(y)) <= 1.0 + 1e-9


def test_apply_mic_injected_rng_is_reproducible():
    x = np.array([10.0, -10.0, 5.0, -5.0, 0.0])

    y1 = apply_mic(x, clip_prob=0.5, rng=np.random.default_rng(99))
    y2 = apply_mic(x, clip_prob=0.5, rng=np.random.default_rng(99))

    assert np.array_equal(y1, y2)
