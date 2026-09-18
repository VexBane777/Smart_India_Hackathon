from __future__ import annotations

import numpy as np
import torch

from finetune_oc_softmax_v13 import embed_v13
from model import VoiceGuardSeqTCN


def _tiny_v13_model():
    n_lfcc, n_scalars = 60, 6
    return VoiceGuardSeqTCN(n_lfcc=n_lfcc, n_scalars=n_scalars,
                             seq_mean=np.zeros(n_lfcc, dtype=np.float32), seq_std=np.ones(n_lfcc, dtype=np.float32),
                             scalar_mean=np.zeros(n_scalars, dtype=np.float32), scalar_std=np.ones(n_scalars, dtype=np.float32))


def test_embed_v13_matches_forwards_own_trunk_out_shape():
    model = _tiny_v13_model().eval()
    seq = torch.randn(3, 184, 60)
    scalars = torch.randn(3, 6)
    emb = embed_v13(model, seq, scalars)
    assert emb.shape == (3, model.trunk[0].out_features)


def test_embed_v13_is_deterministic_and_does_not_mutate_model_weights():
    model = _tiny_v13_model().eval()
    seq, scalars = torch.randn(2, 184, 60), torch.randn(2, 6)
    before = [p.clone() for p in model.parameters()]
    e1 = embed_v13(model, seq, scalars)
    e2 = embed_v13(model, seq, scalars)
    assert torch.allclose(e1, e2)
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))


def test_embed_v13_matches_forwards_trunk_out_value_directly():
    """embed_v13 must be the SAME computation VoiceGuardSeqTCN.forward uses
    internally, not just shape-compatible -- reimplemented here from
    forward()'s own source (model.py) as an independent check."""
    model = _tiny_v13_model().eval()
    seq, scalars = torch.randn(4, 184, 60), torch.randn(4, 6)
    with torch.no_grad():
        h = model.blocks(model.stem(model.normalize_sequence(seq).transpose(1, 2)))
        mean = h.mean(dim=2)
        std = torch.sqrt(((h - mean.unsqueeze(2)) ** 2).mean(dim=2) + 1e-5)
        pooled = torch.cat([mean, std, h.amax(dim=2)], dim=1)
        expected = model.trunk(torch.cat([pooled, model.scalar_normalize(scalars)], dim=1))
        actual = embed_v13(model, seq, scalars)
    assert torch.allclose(actual, expected)
