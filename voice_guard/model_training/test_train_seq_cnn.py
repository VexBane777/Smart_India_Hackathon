"""train_seq_cnn.py: masked attack loss, LR schedule, EMA, the channel
policy at the CLI, and an end-to-end cache-backed smoke run with ONNX
export checked through onnxruntime."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from train_seq_cnn import EMA, compute_masked_attack_type_loss, lr_at

HERE = Path(__file__).parent


def test_masked_loss_ignores_ignore_index_examples():
    logits = torch.tensor([[5.0, -5.0], [-5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    targets = torch.tensor([0, 1, -100, -100])
    assert compute_masked_attack_type_loss(logits, targets).item() < 0.01


def test_masked_loss_all_ignored_is_nan_and_caller_must_guard():
    loss = compute_masked_attack_type_loss(torch.zeros(3, 2), torch.tensor([-100, -100, -100]))
    assert torch.isnan(loss)


def test_lr_schedule_warmup_then_cosine():
    total, warm, base = 1000, 100, 1e-3
    assert lr_at(0, total, warm, base) == base / 100
    assert lr_at(99, total, warm, base) == base
    assert lr_at(999, total, warm, base) < 2e-5
    lrs = [lr_at(s, total, warm, base) for s in range(100, 1000)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))


def test_ema_tracks_parameters():
    m = torch.nn.Linear(2, 1)
    ema = EMA(m, decay=0.9)
    with torch.no_grad():
        m.weight.fill_(1.0)
    for _ in range(200):
        ema.update(m)
    assert torch.allclose(ema.model.weight, m.weight, atol=1e-3)


def _corpus(tmp_path):
    real_dir, fake_dir = tmp_path / "real", tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    rng = np.random.default_rng(0)
    sr = 16000
    for i in range(6):
        for d, f0 in ((real_dir, 150), (fake_dir, 220)):
            secs = 2.0 + 0.5 * i
            t = np.arange(int(secs * sr)) / sr
            sf.write(str(d / f"x{i}.wav"), (0.3 * np.sin(2 * np.pi * f0 * t) + rng.normal(0, 0.02, len(t))).astype(np.float32), sr)
    return real_dir, fake_dir


def test_clean_only_training_is_refused(tmp_path):
    real_dir, fake_dir = _corpus(tmp_path)
    r = subprocess.run([sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
                        "--channels", "none", "--out", str(tmp_path / "run"), "--cache-root", str(tmp_path / "c")],
                       cwd=HERE, capture_output=True, text=True)
    assert r.returncode != 0 and "CleanOnlyEvalError" in r.stderr
    assert not (tmp_path / "c").exists() or not any((tmp_path / "c").iterdir())


def test_train_seq_cnn_end_to_end_smoke(tmp_path):
    import onnxruntime as ort

    real_dir, fake_dir = _corpus(tmp_path)
    out_dir = tmp_path / "run"
    r = subprocess.run(
        [sys.executable, "train_seq_cnn.py", "--real", str(real_dir), "--fake", str(fake_dir),
         "--channels", "none", "whatsapp", "--out", str(out_dir), "--cache-root", str(tmp_path / "cache"),
         "--epochs", "2", "--workers", "1", "--batch-size", "8"],
        cwd=HERE, capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr[-3000:]
    for f in ("model.pt", "model.onnx", "norm_stats.npz", "history.json", "train_config.json",
              "checkpoints/epoch_01.pt", "checkpoints/epoch_02.pt"):
        assert (out_dir / f).exists(), f
    assert str(np.load(out_dir / "norm_stats.npz")["arch"]) == "seqtcn_v2"
    sess = ort.InferenceSession(str(out_dir / "model.onnx"))
    outs = sess.run(None, {"lfcc_sequence": np.zeros((1, 184, 60), np.float32), "scalars": np.zeros((1, 6), np.float32)})
    assert [o.shape for o in outs] == [(1, 2), (1, 2)]
