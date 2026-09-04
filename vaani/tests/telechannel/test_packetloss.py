"""
Unit tests for telechannel/stages/packetloss.py.
"""

import numpy as np

from telechannel.stages.packetloss import apply_packet_loss, generate_ge_mask


def _run_lengths(mask):
    """Lengths of consecutive True ("dropped") runs in a boolean mask."""
    runs = []
    count = 0
    for v in mask:
        if v:
            count += 1
        else:
            if count > 0:
                runs.append(count)
            count = 0
    if count > 0:
        runs.append(count)
    return runs


def test_apply_packet_loss_output_same_length_as_input():
    x = np.random.default_rng(0).standard_normal(1600)

    y = apply_packet_loss(x, p_gb=0.1, p_bg=0.3)

    assert len(y) == len(x)


def test_apply_packet_loss_no_loss_when_p_gb_zero():
    # p_gb=0 means the channel can never leave the Good state, so the
    # output must be identical to the input.
    x = np.random.default_rng(1).standard_normal(1600)

    y = apply_packet_loss(x, p_gb=0.0, p_bg=0.5)

    assert np.array_equal(y, x)


def test_apply_packet_loss_conceals_dropped_frames_by_repeating_last_good_frame():
    sr = 16000
    frame_len = sr * 20 // 1000  # 320 samples per 20ms frame
    # 5 distinct constant-valued frames.
    x = np.concatenate([np.full(frame_len, v, dtype=np.float64) for v in (1, 2, 3, 4, 5)])

    # p_gb=1.0 -> guaranteed transition to Bad right after the first
    # (always-good) frame; p_bg=0.0 -> never recovers. So frame 0 is
    # delivered and every subsequent frame is a repeat of frame 0.
    y = apply_packet_loss(x, p_gb=1.0, p_bg=0.0, sr=sr)

    assert np.allclose(y, 1.0)


def test_generate_ge_mask_is_bursty_not_binomial():
    # Geometric run lengths: mean consecutive-drop length should be close
    # to 1/p_bg, and bursts materially longer than 1 frame should occur
    # (unlike i.i.d./binomial loss at the same overall drop rate, which
    # would produce almost exclusively single-frame drops).
    rng = np.random.default_rng(123)
    p_gb, p_bg = 0.05, 0.3
    mask = generate_ge_mask(3000, p_gb, p_bg, rng=rng)

    runs = _run_lengths(mask)
    assert len(runs) > 5  # enough bursts to make the statistics meaningful

    mean_run_length = np.mean(runs)
    expected_mean = 1.0 / p_bg  # geometric distribution mean
    assert 0.5 * expected_mean <= mean_run_length <= 2.0 * expected_mean

    # Burstiness: at least some runs of length > 1 must occur.
    assert max(runs) > 1


def test_generate_ge_mask_returns_boolean_array_of_requested_length():
    mask = generate_ge_mask(500, 0.1, 0.2, rng=np.random.default_rng(0))

    assert mask.dtype == bool
    assert len(mask) == 500
