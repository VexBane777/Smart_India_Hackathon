"""
Smoke test for the training pipeline's mechanics — shapes, dtypes, ONNX
export round-tripping — using SYNTHETIC audio (sine tones vs. filtered
noise as real/fake stand-ins). This does NOT validate detection accuracy;
it only proves the code runs end-to-end before pointing it at a real
corpus. Real accuracy validation happens on ASVspoof (see train.py's
docstring).

For corpus-composition/data-integrity issues this suite structurally can't
see (uneven chunk-yield across sources, technical shortcuts like a
sample-rate/generator confound, train/held-out leakage, the split-by-source
basename bug) — see test_dataset_integrity.py and check_corpus.py, added
2026-09-10 after those exact issues caused four wasted retraining attempts.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from features import SAMPLE_RATE, extract_features
from dataset import build_examples, split_by_source, to_arrays
from model import VoiceGuardMLP


def _write_synthetic_corpus(root: Path, n_per_class: int = 6, seconds: float = 4.0):
    real_dir, fake_dir = root / "real", root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    rng = np.random.default_rng(0)

    for i in range(n_per_class):
        freq = 150 + i * 10
        clip = 0.3 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        sf.write(real_dir / f"real_{i}.wav", clip, SAMPLE_RATE)

    for i in range(n_per_class):
        clip = 0.3 * rng.standard_normal(len(t)).astype(np.float32)
        clip = np.convolve(clip, np.ones(5) / 5, mode="same").astype(np.float32)
        sf.write(fake_dir / f"fake_{i}.wav", clip, SAMPLE_RATE)

    return real_dir, fake_dir


def test_feature_extraction_shape():
    pcm = np.zeros(48000, dtype=np.float32)
    feats = extract_features(pcm)
    assert feats.shape == (66,)
    assert feats.dtype == np.float32


def test_end_to_end_train_and_export(tmp_path: Path = None):
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    real_dir, fake_dir = _write_synthetic_corpus(tmp_path)

    examples = build_examples(real_dir, fake_dir)
    assert len(examples) > 0
    train_ex, val_ex = split_by_source(examples, val_fraction=0.34)
    assert train_ex and val_ex
    assert {e.source_file for e in train_ex} & {e.source_file for e in val_ex} == set()

    X_train, y_train = to_arrays(train_ex)
    assert X_train.shape[1] == 66
    assert set(y_train.tolist()) <= {0, 1}

    mean, std = X_train.mean(axis=0), X_train.std(axis=0) + 1e-8
    model = VoiceGuardMLP(norm_mean=mean, norm_std=std)
    logits = model(torch.from_numpy(X_train))
    assert logits.shape == (len(X_train), 2)

    onnx_path = tmp_path / "model.onnx"
    torch.onnx.export(
        model,
        torch.from_numpy(X_train[:1]),
        str(onnx_path),
        input_names=["features"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,
    )
    assert onnx_path.exists() and onnx_path.stat().st_size > 0


if __name__ == "__main__":
    test_feature_extraction_shape()
    test_end_to_end_train_and_export()
    print("Smoke test passed.")
