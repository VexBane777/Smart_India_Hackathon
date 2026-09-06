"""
mic.py — Microphone simulation stage.

Applies a random gain (simulating varying mic sensitivity / recording
levels) and, with some probability, hard-clips the signal (simulating a
cheap/overloaded mic preamp). A final [-1, 1] safety clamp always applies
regardless of `clip_prob` -- see the note below.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).

Full-scale safety clamp (2026-09-06):

    The +12dB gain ceiling (~4x linear) is only followed by the cheap-mic
    ±0.5 clip `clip_prob` of the time (default 20%) -- the other 80%, gain
    output was previously returned unbounded. A real recorded clip (not
    this repo's low-amplitude synthetic test fixtures) with a ~0.7 peak
    input and +12dB gain overshoots 1.0 (measured up to ~1.18 after the
    rest of the `gsm_2g`/`cellular_3g` recipe chain) -- out of the [-1, 1]
    range every downstream consumer (FLAC write, mel-spectrogram
    extraction) assumes. `apply_mic` now always clamps its output to
    [-1, 1] as a final safety net, on top of (not instead of) the existing
    probabilistic ±0.5 "cheap preamp" clip -- see also
    `telechannel/pipeline.py::process_clip`'s matching end-of-chain clamp,
    which catches overshoot from any other stage (e.g. bandlimit filter
    ringing), not just this one.
"""

import numpy as np

GAIN_DB_MIN = -6
GAIN_DB_MAX = 12
CLIP_LIMIT = 0.5


def apply_mic(x, clip_prob=0.2, rng=None):
    """
    Apply a random gain in [-6, +12] dB to `x`, then, with probability
    `clip_prob`, hard-clip the result to [-0.5, 0.5].

    Args:
        x: 1-D input signal (numpy array).
        clip_prob: Probability in [0, 1] that clipping is applied after
            the gain stage. Defaults to 0.2.
        rng: Optional numpy.random.Generator for reproducibility. If not
            given, a fresh `np.random.default_rng()` is used (the previous,
            non-reproducible behavior).

    Returns:
        numpy array the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    if rng is None:
        rng = np.random.default_rng()

    gain_db = rng.uniform(GAIN_DB_MIN, GAIN_DB_MAX)
    gain_linear = 10 ** (gain_db / 20)
    y = x * gain_linear

    if rng.random() < clip_prob:
        y = np.clip(y, -CLIP_LIMIT, CLIP_LIMIT)

    # Always-on safety clamp: independent of clip_prob, gain alone can push
    # a near-full-scale input past [-1, 1] (see module docstring).
    y = np.clip(y, -1.0, 1.0)

    return y
