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


# --- Physiological voice-quality features (remediation track 2, see
# docs/CRITICAL-entity-vs-style-confound.md §4 item 2) ---
#
# Standard autocorrelation-based pitch tracking (the same family of
# algorithm Praat uses for jitter/shimmer/HNR), computed directly on the
# 3s/16kHz window — no new dependency, pure numpy. This is a deliberate
# simplification vs. full pitch-synchronous waveform matching: amplitude
# for shimmer is taken as frame RMS rather than per-cycle peak amplitude,
# which is standard practice when a lighter-weight implementation is
# preferred over a full Praat-style pitch-marking pipeline, and is exactly
# mirrored on the Dart side (see audio_processor.dart) so both sides stay
# numerically equivalent to each other, which is the actual invariant this
# project depends on (not exact agreement with Praat itself).
_PITCH_FRAME_LEN = 480  # 30ms @ 16kHz
_PITCH_HOP = 160  # 10ms @ 16kHz
_MIN_F0 = 75.0  # Hz, typical human voice floor
_MAX_F0 = 500.0  # Hz, typical human voice ceiling
_MIN_LAG = int(round(SAMPLE_RATE / _MAX_F0))  # 32 samples
_MAX_LAG = int(round(SAMPLE_RATE / _MIN_F0))  # 213 samples
_VOICING_THRESHOLD = 0.30  # normalized autocorrelation peak


def _pitch_track(pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (periods_s, amplitudes, autocorr_peaks, voiced_mask) — one
    entry per frame, aligned; unvoiced frames carry period=0/amp=0/r=0 and
    voiced_mask=False, filtered out by callers."""
    n = len(pcm)
    if n < _PITCH_FRAME_LEN:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool))
    n_frames = 1 + (n - _PITCH_FRAME_LEN) // _PITCH_HOP
    periods = np.zeros(n_frames)
    amps = np.zeros(n_frames)
    peaks = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * _PITCH_HOP
        frame = pcm[start:start + _PITCH_FRAME_LEN]
        amps[i] = np.sqrt(np.mean(frame ** 2))
        energy0 = np.dot(frame, frame)
        if energy0 <= 1e-12:
            continue
        best_lag, best_r = 0, 0.0
        for lag in range(_MIN_LAG, min(_MAX_LAG, _PITCH_FRAME_LEN - 1) + 1):
            a, b = frame[:-lag], frame[lag:]
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom <= 1e-12:
                continue
            r = np.dot(a, b) / denom
            if r > best_r:
                best_r, best_lag = r, lag
        peaks[i] = best_r
        if best_r >= _VOICING_THRESHOLD and best_lag > 0:
            voiced[i] = True
            periods[i] = best_lag / SAMPLE_RATE
    return periods, amps, peaks, voiced


def extract_physio(pcm: np.ndarray) -> np.ndarray:
    """3s PCM float buffer -> [jitter_local, shimmer_local, hnr_db].
    Zeros if fewer than 2 consecutive voiced frames are found (silence,
    noise, or too-short input) — mirrors extract_prosody's degenerate-input
    convention of returning zeros rather than raising or returning NaN."""
    periods, amps, peaks, voiced = _pitch_track(pcm)
    if voiced.sum() < 2:
        return np.zeros(3, dtype=np.float64)

    # Jitter/shimmer: only over PAIRS of consecutive (hop-adjacent) voiced
    # frames — a voiced frame next to an unvoiced one contributes no pair,
    # same convention Praat uses (skip across unvoiced gaps).
    pair_mask = voiced[:-1] & voiced[1:]
    if pair_mask.sum() < 1:
        return np.zeros(3, dtype=np.float64)

    p0, p1 = periods[:-1][pair_mask], periods[1:][pair_mask]
    a0, a1 = amps[:-1][pair_mask], amps[1:][pair_mask]

    mean_period = (p0 + p1).mean() / 2
    jitter_local = np.mean(np.abs(p1 - p0)) / mean_period if mean_period > 1e-12 else 0.0

    mean_amp = (a0 + a1).mean() / 2
    shimmer_local = np.mean(np.abs(a1 - a0)) / mean_amp if mean_amp > 1e-12 else 0.0

    voiced_peaks = np.clip(peaks[voiced], 0.0, 0.999999)
    hnr_db = float(np.mean(10 * np.log10(voiced_peaks / (1 - voiced_peaks) + 1e-12)))

    return np.array([jitter_local, shimmer_local, hnr_db], dtype=np.float64)


def extract_features(pcm: np.ndarray) -> np.ndarray:
    """Full 66-d feature vector: 60 LFCC + 3 prosody + 3 physio, matching
    TFLiteService.infer's `[...lfcc, ...prosody, ...physio]` concatenation
    order exactly."""
    lfcc = extract_lfcc(pcm)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    return np.concatenate([lfcc, prosody, physio]).astype(np.float32)


def chunk_audio(pcm: np.ndarray, chunk_samples: int = CHUNK_SAMPLES) -> list[np.ndarray]:
    """Split a clip into non-overlapping 3s chunks, dropping a shorter final chunk."""
    n_chunks = len(pcm) // chunk_samples
    return [pcm[i * chunk_samples : (i + 1) * chunk_samples] for i in range(n_chunks)]
