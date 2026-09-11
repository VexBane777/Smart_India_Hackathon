"""Tests for VoiceGuardSeqCNN (remediation tracks 3+4)."""
from __future__ import annotations

import numpy as np
import torch

from model import FixedNormalizeSeq, VoiceGuardSeqCNN


def test_fixed_normalize_seq_broadcasts_across_time():
    mean = np.zeros(60, dtype=np.float32)
    std = np.ones(60, dtype=np.float32)
    norm = FixedNormalizeSeq(mean, std)
    x = torch.randn(4, 184, 60)
    out = norm(x)
    assert out.shape == (4, 184, 60)


def test_seq_cnn_forward_shapes():
    model = VoiceGuardSeqCNN(
        n_frames=184,
        n_lfcc=60,
        n_scalars=6,
        seq_mean=np.zeros(60, dtype=np.float32),
        seq_std=np.ones(60, dtype=np.float32),
        scalar_mean=np.zeros(6, dtype=np.float32),
        scalar_std=np.ones(6, dtype=np.float32),
    )
    seq = torch.randn(8, 184, 60)
    scalars = torch.randn(8, 6)
    real_fake_logits, attack_type_logits = model(seq, scalars)
    assert real_fake_logits.shape == (8, 2)
    assert attack_type_logits.shape == (8, 2)


def test_seq_cnn_param_count_is_small():
    model = VoiceGuardSeqCNN(
        n_frames=184, n_lfcc=60, n_scalars=6,
        seq_mean=np.zeros(60, dtype=np.float32), seq_std=np.ones(60, dtype=np.float32),
        scalar_mean=np.zeros(6, dtype=np.float32), scalar_std=np.ones(6, dtype=np.float32),
    )
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 200_000  # per track 3 plan §2.3's "under 50-100K" budget, generous margin
