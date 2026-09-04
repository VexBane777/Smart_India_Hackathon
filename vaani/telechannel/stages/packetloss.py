"""
packetloss.py — Packet loss stage (Gilbert-Elliott channel model).

Simulates bursty packet loss over VoIP-style 20ms frames using a two-state
Markov chain (Gilbert-Elliott model): a "Good" state (frame delivered) and
a "Bad" state (frame dropped). `p_gb` is the per-frame probability of
transitioning Good -> Bad; `p_bg` is the per-frame probability of
transitioning Bad -> Good. Because loss is driven by state persistence
rather than independent per-frame coin flips, drops cluster into bursts
(consecutive-drop run lengths follow a geometric distribution), which is
much closer to real network behaviour than i.i.d. (binomial) loss.

Dropped frames are concealed by repeating the last successfully received
frame, with a 5ms linear crossfade applied at loss-state transitions
(entering and leaving a burst) to avoid audible clicks.

See DOC1_TELECHANNEL_SPEC.md for the TeleChannel stage pipeline this module
plugs into (a later task chains these stages together; this module is
independent and does not do that chaining).
"""

import numpy as np

FRAME_MS = 20
FADE_MS = 5


def generate_ge_mask(n_frames, p_gb, p_bg, rng=None):
    """
    Generate a boolean loss mask of length `n_frames` using a two-state
    Gilbert-Elliott Markov chain.

    Args:
        n_frames: Number of 20ms frames to simulate.
        p_gb: Probability of transitioning Good -> Bad each frame.
        p_bg: Probability of transitioning Bad -> Good each frame.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        Boolean numpy array of length `n_frames`; True means the frame is
        dropped (Bad state), False means delivered (Good state).
    """
    if rng is None:
        rng = np.random.default_rng()

    mask = np.zeros(n_frames, dtype=bool)
    state_bad = False
    for i in range(n_frames):
        mask[i] = state_bad
        if state_bad:
            if rng.random() < p_bg:
                state_bad = False
        else:
            if rng.random() < p_gb:
                state_bad = True
    return mask


def apply_packet_loss(x, p_gb, p_bg, sr=16000, rng=None):
    """
    Simulate bursty packet loss on `x` and conceal dropped frames by
    repeating the last good frame with a 5ms linear crossfade at burst
    boundaries.

    Args:
        x: 1-D input signal (numpy array).
        p_gb: Good -> Bad transition probability per 20ms frame.
        p_bg: Bad -> Good transition probability per 20ms frame.
        sr: Sample rate in Hz, used to size the 20ms frames / 5ms
            crossfade. Defaults to 16000.
        rng: Optional numpy Generator for reproducibility.

    Returns:
        numpy array the same length as `x`.
    """
    x = np.asarray(x, dtype=np.float64)
    n = len(x)

    frame_len = max(1, sr * FRAME_MS // 1000)
    fade_len = max(1, sr * FADE_MS // 1000)
    fade_len = min(fade_len, frame_len)

    n_frames = int(np.ceil(n / frame_len)) if n > 0 else 0
    if n_frames == 0:
        return x.copy()

    pad = n_frames * frame_len - n
    xp = np.concatenate([x, np.zeros(pad)]) if pad else x.copy()
    frames = xp.reshape(n_frames, frame_len)

    mask = generate_ge_mask(n_frames, p_gb, p_bg, rng=rng)

    out = np.empty_like(frames)
    last_good = frames[0].copy()
    for i in range(n_frames):
        if mask[i]:
            content = last_good.copy()
        else:
            content = frames[i].copy()
            last_good = content.copy()

        if i > 0 and mask[i] != mask[i - 1]:
            prev_tail = out[i - 1][-fade_len:]
            w = np.linspace(0.0, 1.0, fade_len, endpoint=False)
            content[:fade_len] = (1 - w) * prev_tail + w * content[:fade_len]

        out[i] = content

    y = out.reshape(-1)[:n]
    return y
