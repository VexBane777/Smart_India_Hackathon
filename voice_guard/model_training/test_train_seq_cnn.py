"""Tests for train_seq_cnn.py's masked multi-task loss."""
from __future__ import annotations

from pathlib import Path

import torch

from train_seq_cnn import compute_masked_attack_type_loss


def test_masked_loss_ignores_ignore_index_examples():
    # 2 labeled (tts=0, vc=1) + 2 unlabeled (-100) examples.
    logits = torch.tensor([
        [5.0, -5.0],  # confidently "tts" -> should contribute ~0 loss for target 0
        [-5.0, 5.0],  # confidently "vc" -> should contribute ~0 loss for target 1
        [0.0, 0.0],   # ignored regardless of logits
        [0.0, 0.0],   # ignored regardless of logits
    ])
    targets = torch.tensor([0, 1, -100, -100])
    loss = compute_masked_attack_type_loss(logits, targets)
    assert loss.item() < 0.01


def test_masked_loss_all_ignored_does_not_crash():
    logits = torch.zeros(3, 2)
    targets = torch.tensor([-100, -100, -100])
    loss = compute_masked_attack_type_loss(logits, targets)
    assert torch.isfinite(loss) or torch.isnan(loss)  # CrossEntropyLoss returns NaN when the whole batch is ignored — documented PyTorch behavior, caller must guard (see Step 3's train loop)


def test_train_seq_cnn_end_to_end_smoke(tmp_path, monkeypatch):
    import subprocess
    import sys

    import numpy as np
    import soundfile as sf

    real_dir, fake_dir = tmp_path / "real", tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    rng = np.random.default_rng(0)
    sr = 16000
    for i in range(6):
        t = np.arange(int(3.5 * sr)) / sr
        sig = (0.3 * np.sin(2 * np.pi * 150 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
        sf.write(str(real_dir / f"r{i}.wav"), sig, sr)
        sig2 = (0.3 * np.sin(2 * np.pi * 220 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32)
        sf.write(str(fake_dir / f"f{i}.wav"), sig2, sr)

    out_dir = tmp_path / "run"
    result = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--out", str(out_dir), "--epochs", "1", "--workers", "1"],
        cwd=Path(__file__).parent, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (out_dir / "model.pt").exists()
    assert (out_dir / "model.onnx").exists()
    assert (out_dir / "norm_stats.npz").exists()
