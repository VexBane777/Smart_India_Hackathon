"""
mic.py — Microphone simulation stage.

Applies a random gain (simulating varying mic sensitivity / recording
levels) and, with some probability, hard-clips the signal (simulating a
cheap/overloaded mic preamp).

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import numpy as np

GAIN_DB_MIN = -6
GAIN_DB_MAX = 12
CLIP_LIMIT = 0.5


def apply_mic(x, clip_prob=0.2):
    """
    Apply a random gain in [-6, +12] dB to `x`, then, with probability
    `clip_prob`, hard-clip the result to [-0.5, 0.5].

    Args:
        x: 1-D input signal (numpy array).
        clip_prob: Probability in [0, 1] that clipping is applied after
            the gain stage. Defaults to 0.2.

    Returns:
        numpy array the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng()

    gain_db = rng.uniform(GAIN_DB_MIN, GAIN_DB_MAX)
    gain_linear = 10 ** (gain_db / 20)
    y = x * gain_linear

    if rng.random() < clip_prob:
        y = np.clip(y, -CLIP_LIMIT, CLIP_LIMIT)

    return y
