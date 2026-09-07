"""
Python port of lib/utils/audio_processor.dart — MUST stay numerically
equivalent to the Dart extractor, since the trained model is only valid
if train-time features match what the phone computes at inference time.

Dart's `extractLfcc`/`extractProsody` operate on a 3s, 16kHz PCM buffer
(48000 samples) and produce a 60-d LFCC vector (mean-pooled over frames)
plus a 3-d prosody vector: [pauseRatio, energyVariance, zcrVariance].
This module reproduces the same frame size (1024), hop (256), filterbank
count (513, effectively identity since the FFT already yields 513 bins
for a 1024-pt real FFT), and DCT-II (orthonormal, scipy convention) —
using numpy's FFT/DCT instead of Dart's hand-rolled radix-2 FFT, which
computes the same DFT up to floating-point rounding.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
CHUNK_SAMPLES = 48000  # 3s
FFT_SIZE = 1024
HOP_LENGTH = 256
N_LFCC = 60
N_FILTERBANKS = 513  # == FFT_SIZE // 2 + 1, i.e. identity mapping


def _frame_signal(pcm: np.ndarray) -> np.ndarray:
    n_frames = 1 + (len(pcm) - FFT_SIZE) // HOP_LENGTH if len(pcm) >= FFT_SIZE else 0
    if n_frames <= 0:
        return np.empty((0, FFT_SIZE), dtype=np.float64)
    frames = np.stack(
        [pcm[i * HOP_LENGTH : i * HOP_LENGTH + FFT_SIZE] for i in range(n_frames)]
    )
    return frames


def _hamming(frames: np.ndarray) -> np.ndarray:
    n = frames.shape[-1]
    window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(n) / (n - 1))
    return frames * window


def _magnitude_spectrum(frames: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfft(frames, n=FFT_SIZE, axis=-1)
    return np.abs(spectrum) / FFT_SIZE


def _linear_filterbank(mag: np.ndarray) -> np.ndarray:
    # mag is already (n_frames, 513) for a 1024-pt real FFT — identity mapping,
    # matching Dart's `_linearFilterbank` fast path.
    assert mag.shape[-1] == N_FILTERBANKS
    return mag**2 + 1e-10


def _dct2_orthonormal(x: np.ndarray) -> np.ndarray:
    """DCT-II, orthonormal scaling — matches Dart's `_dct`."""
    n = x.shape[-1]
    k = np.arange(n).reshape(-1, 1)
    ni = np.arange(n).reshape(1, -1)
    basis = np.cos(np.pi * k * (2 * ni + 1) / (2 * n))  # (n, n)
    out = x @ basis.T * np.sqrt(2.0 / n)
    out[..., 0] *= 1.0 / np.sqrt(2.0)
    return out


def extract_lfcc(pcm: np.ndarray) -> np.ndarray:
    """3s (or shorter) PCM float buffer in [-1, 1] -> 60 LFCC coefficients."""
    if len(pcm) < FFT_SIZE:
        return np.zeros(N_LFCC, dtype=np.float64)
    frames = _hamming(_frame_signal(pcm))
    mag = _magnitude_spectrum(frames)
    energies = _linear_filterbank(mag)
    log_e = np.log(energies + 1e-10)
    lfcc_frames = _dct2_orthonormal(log_e)[:, :N_LFCC]
    mean = lfcc_frames.mean(axis=0)
    m = mean.mean()
    std = np.sqrt(((mean - m) ** 2).mean() + 1e-8)
    return (mean - m) / std


def extract_prosody(pcm: np.ndarray) -> np.ndarray:
    """3s PCM float buffer -> [pauseRatio, energyVariance, zcrVariance]."""
    frame_len = 512
    n_frames = len(pcm) // frame_len
    if n_frames == 0:
        return np.zeros(3, dtype=np.float64)
    frames = pcm[: n_frames * frame_len].reshape(n_frames, frame_len)
    energies = (frames**2).mean(axis=1)
    signs = frames >= 0
    zcrs = (signs[:, 1:] != signs[:, :-1]).sum(axis=1) / frame_len

    max_e = energies.max()
    thresh = max_e * 0.02
    pause_ratio = (energies < thresh).mean()
    return np.array([pause_ratio, energies.var(), zcrs.var()], dtype=np.float64)


def extract_features(pcm: np.ndarray) -> np.ndarray:
    """Full 63-d feature vector: 60 LFCC + 3 prosody, matching TFLiteService.infer's
    `[...lfcc, ...prosody]` concatenation order exactly."""
    lfcc = extract_lfcc(pcm)
    prosody = extract_prosody(pcm)
    return np.concatenate([lfcc, prosody]).astype(np.float32)


def chunk_audio(pcm: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> list[np.ndarray]:
    """Split a clip into non-overlapping 3s chunks, dropping a shorter final chunk."""
    n_chunks = len(pcm) // chunk_samples
    return [pcm[i * chunk_samples : (i + 1) * chunk_samples] for i in range(n_chunks)]
