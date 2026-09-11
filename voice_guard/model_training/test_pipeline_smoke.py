"""
Smoke test for the training pipeline's mechanics — shapes, dtypes, ONNX
export round-tripping — using SYNTHETIC audio (sine tones vs. filtered
noise as real/fake stand-ins). This does NOT validate detection accuracy;
it only proves the code runs end-to-end before pointing it at a real
corpus. Real accuracy validation happens under the channel protocol — see
../docs/EVAL-PROTOCOL.md and evaluate.py (train.py was retired 2026-09-11).

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

from features import N_LFCC, SAMPLE_RATE, extract_features
from dataset import build_examples, split_by_source, to_arrays


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


def test_build_examples_and_to_arrays_produce_sequence_shaped_dataset(tmp_path: Path = None):
    """Dataset-pipeline mechanics only (build_examples -> split_by_source ->
    to_arrays), using the same synthetic corpus this file has always used.
    The full model+ONNX-export exercise for the sequence-CNN pipeline lives
    in test_train_seq_cnn.py's own end-to-end smoke test (added alongside
    VoiceGuardSeqCNN) — VoiceGuardMLP no longer consumes this shape, so it
    doesn't belong in this test anymore."""
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    real_dir, fake_dir = _write_synthetic_corpus(tmp_path)

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None], workers=1)
    assert len(examples) > 0
    train_ex, val_ex = [], []
    for seed in range(10):
        # A 12-file toy corpus doesn't always straddle val_fraction=0.34
        # (P(all 12 on the train side) ~ 0.8% by hash luck); sweep seeds
        # until both sides are populated. The contract under test is
        # source-disjointness and shapes, not this particular hash.
        train_ex, val_ex = split_by_source(examples, val_fraction=0.34, seed=seed)
        if train_ex and val_ex:
            break
    assert train_ex and val_ex
    assert {e.file_id for e in train_ex} & {e.file_id for e in val_ex} == set()

    X_seq, X_scalars, y_train, attack_y = to_arrays(train_ex)
    assert X_seq.shape[0] == len(train_ex)
    assert X_seq.shape[2] == N_LFCC
    assert X_scalars.shape == (len(train_ex), 6)
    assert set(y_train.tolist()) <= {0, 1}
    assert attack_y.shape == (len(train_ex),)


if __name__ == "__main__":
    test_feature_extraction_shape()
    test_build_examples_and_to_arrays_produce_sequence_shaped_dataset()
    print("Smoke test passed.")
