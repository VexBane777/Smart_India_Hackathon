"""
spectrogram.py — Live spectrogram component (Module C Task 4 Step 3).

Renders the most recent 2 s window's Mel-ish spectrogram with matplotlib only
(no librosa — keeps the dependency surface for airplane mode minimal). The
"mel-ish" filterbank uses a simple triangular-mel approximation; it is a
*visual aid*, not the model's feature extractor, and is labeled as such.
"""

from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

matplotlib.use("Agg")  # Streamlit-safe: no interactive backend

SR = 16000
N_FFT = 512
HOP = 160            # 10 ms
N_MELS = 48

_MEL_BANK: Optional[np.ndarray] = None


def _hz_to_mel(f: np.ndarray) -> np.ndarray:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _mel_filterbank() -> np.ndarray:
    """Triangular mel filterbank [N_MELS, N_FFT//2+1] on SR's Nyquist band."""
    global _MEL_BANK
    if _MEL_BANK is not None:
        return _MEL_BANK
    n_bins = N_FFT // 2 + 1
    mels = _hz_to_mel(np.array([0.0, SR / 2.0]))
    m_pts = np.linspace(mels[0], mels[1], N_MELS + 2)
    hz_pts = _mel_to_hz(m_pts)
    bins = hz_pts / SR * N_FFT
    bank = np.zeros((N_MELS, n_bins))
    for i in range(N_MELS):
        left, center, right = bins[i], bins[i + 1], bins[i + 2]
        for k in range(n_bins):
            if left < k < center:
                bank[i, k] = (k - left) / (center - left)
            elif center <= k < right:
                bank[i, k] = (right - k) / (right - center)
    _MEL_BANK = bank
    return bank


def compute_mel_db(audio: np.ndarray, sr: int = SR) -> np.ndarray:
    """Mel spectrogram in dB, shape [N_MELS, frames]. Pure numpy/scipy."""
    audio = np.asarray(audio, dtype=np.float64)
    if len(audio) < N_FFT:
        audio = np.pad(audio, (0, N_FFT - len(audio)))
    n_frames = 1 + (len(audio) - N_FFT) // HOP
    frames = np.lib.stride_tricks.sliding_window_view(audio, N_FFT)[::HOP][:n_frames]
    window = np.hanning(N_FFT)
    spec = np.abs(np.fft.rfft(frames * window, axis=1)).T  # [bins, frames]
    mel = _mel_filterbank() @ spec
    mel_db = 20.0 * np.log10(np.maximum(mel, 1e-10))
    return mel_db


def render_spectrogram(audio_chunk: np.ndarray, sr: int = SR,
                       caption: str = "current 2 s window (mel, visual aid)") -> None:
    """Show the mel spectrogram of the current window via st.pyplot."""
    if len(audio_chunk) == 0:
        st.info("Waiting for audio...")
        return
    mel_db = compute_mel_db(audio_chunk, sr)
    fig, ax = plt.subplots(figsize=(6, 2.4), dpi=100)
    ax.imshow(mel_db, origin="lower", aspect="auto", cmap="magma",
              extent=[0, len(audio_chunk) / sr, 0, N_MELS])
    ax.set_xlabel("time in window (s)")
    ax.set_ylabel("mel bin")
    ax.set_title(caption, fontsize=9)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)
