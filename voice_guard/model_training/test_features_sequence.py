"""Tests for the per-frame LFCC sequence extraction added for the frame-
level CNN (remediation track 3, see
docs/superpowers/plans/2026-09-11-frame-level-seq-model-and-attack-type-plan.md)."""
from __future__ import annotations

import numpy as np

from features import (
    CHUNK_SAMPLES,
    FFT_SIZE,
    HOP_LENGTH,
    N_LFCC,
    extract_lfcc,
    extract_lfcc_sequence,
    extract_physio,
    extract_prosody,
    extract_scalars,
)


def test_extract_lfcc_sequence_shape():
    pcm = np.random.default_rng(0).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    seq = extract_lfcc_sequence(pcm)
    expected_frames = 1 + (CHUNK_SAMPLES - FFT_SIZE) // HOP_LENGTH
    assert seq.shape == (expected_frames, N_LFCC)


def test_extract_lfcc_sequence_short_input_returns_zeros():
    pcm = np.zeros(100, dtype=np.float64)
    seq = extract_lfcc_sequence(pcm)
    assert seq.shape == (0, N_LFCC)


def test_extract_lfcc_sequence_mean_matches_pooled_extract_lfcc():
    # extract_lfcc's own mean/std normalization happens AFTER pooling, so
    # this checks the pre-normalization DCT coefficients agree, not the
    # final normalized vectors directly comparable value-for-value.
    pcm = np.random.default_rng(1).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    seq = extract_lfcc_sequence(pcm)
    pooled = extract_lfcc(pcm)
    raw_mean = seq.mean(axis=0)
    m = raw_mean.mean()
    std = np.sqrt(((raw_mean - m) ** 2).mean() + 1e-8)
    normalized = (raw_mean - m) / std
    np.testing.assert_allclose(normalized, pooled, rtol=1e-5)


def test_extract_scalars_is_prosody_then_physio_in_order():
    pcm = np.random.default_rng(2).normal(0, 0.1, CHUNK_SAMPLES).astype(np.float64)
    scalars = extract_scalars(pcm)
    assert scalars.shape == (6,)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    np.testing.assert_allclose(scalars[:3], prosody, rtol=1e-5)
    np.testing.assert_allclose(scalars[3:], physio, rtol=1e-5)
