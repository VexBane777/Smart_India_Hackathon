"""Tests for dataset.py's sequence/scalar/attack-type Example fields
(remediation tracks 3+4)."""
from __future__ import annotations

import numpy as np
import soundfile as sf

from attack_labels import register_tts_only_directory
from dataset import build_examples, to_arrays
from features import CHUNK_SAMPLES, N_LFCC, SAMPLE_RATE


def _write_tone(path, seconds=3.5, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    sig = 0.3 * np.sin(2 * np.pi * 150 * t) + rng.normal(0, 0.01, n)
    sf.write(str(path), sig.astype(np.float32), SAMPLE_RATE)


def test_build_examples_populates_sequence_scalars_and_attack_type(tmp_path):
    real_dir = tmp_path / "real"
    fake_dir = tmp_path / "fake"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=1)
    _write_tone(fake_dir / "f1.wav", seed=2)
    register_tts_only_directory(fake_dir)

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    assert len(examples) > 0

    real_ex = [e for e in examples if e.label == 0][0]
    fake_ex = [e for e in examples if e.label == 1][0]

    assert real_ex.lfcc_seq.shape[1] == N_LFCC
    assert real_ex.scalars.shape == (6,)
    assert real_ex.attack_type == -100  # real examples are never attack-typed

    assert fake_ex.attack_type == 0  # "tts", via the registered directory default


def test_to_arrays_returns_four_arrays_with_matching_lengths(tmp_path):
    real_dir = tmp_path / "real2"
    fake_dir = tmp_path / "fake2"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=3)
    _write_tone(fake_dir / "f1.wav", seed=4)

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    X_seq, X_scalars, y, attack_y = to_arrays(examples)
    n = len(examples)
    assert X_seq.shape[0] == n
    assert X_scalars.shape == (n, 6)
    assert y.shape == (n,)
    assert attack_y.shape == (n,)


def test_unlabeled_fake_directory_gets_unknown_attack_type(tmp_path):
    real_dir = tmp_path / "real3"
    fake_dir = tmp_path / "fake3_unlabeled"
    real_dir.mkdir()
    fake_dir.mkdir()
    _write_tone(real_dir / "r1.wav", seed=5)
    _write_tone(fake_dir / "f1.wav", seed=6)
    # deliberately NOT registered as tts-only, and no attack_type_maps entry

    examples = build_examples(real_dir, fake_dir, channel_recipes=[None])
    fake_ex = [e for e in examples if e.label == 1][0]
    assert fake_ex.attack_type == -100  # unknown -> masked out, same sentinel as real
