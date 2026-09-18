from __future__ import annotations

import torch

from oc_softmax import OCSoftmaxLoss, oc_softmax_score


def test_loss_is_lower_when_genuine_embeddings_align_with_w0():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    aligned = loss_fn.w0.detach().unsqueeze(0).repeat(4, 1)
    y_real = torch.zeros(4, dtype=torch.long)
    aligned_loss = loss_fn(aligned, y_real)
    random_loss = loss_fn(torch.randn(4, 8), y_real)
    assert aligned_loss.item() < random_loss.item()


def test_loss_is_finite_and_scalar_for_mixed_batch():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    emb = torch.randn(5, 8, requires_grad=True)
    y = torch.tensor([0, 1, 0, 1, 1])
    loss = loss_fn(emb, y)
    assert loss.dim() == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert emb.grad is not None


def test_score_is_bounded_and_gradient_direction_makes_sense():
    loss_fn = OCSoftmaxLoss(embedding_dim=8)
    aligned = loss_fn.w0.detach().unsqueeze(0)
    opposite = -loss_fn.w0.detach().unsqueeze(0)
    s_aligned = oc_softmax_score(aligned, loss_fn.w0)
    s_opposite = oc_softmax_score(opposite, loss_fn.w0)
    assert -1.0 <= s_opposite.item() <= s_aligned.item() <= 1.0
