"""
validate_reality.py — Real-call validation tooling.

Compares TeleChannel's simulated telephony degradation against real phone
recordings, per DOC5_REAL_CALLS.md §5.6 ("Analysis"):

    Per call, over VAD-detected speech: LTAS (long-term average spectrum),
    bandwidth cutoff (-20 dB point), HF ratio (energy > 3.5 kHz). Stratify
    by network x capture; overlay vs matching TeleChannel recipes.
    Targets: median cutoff difference <= 300 Hz per stratum; LTAS
    correlation >= 0.9.

This module is standalone (numpy/scipy/librosa/matplotlib only) — it does
not depend on the rest of `telechannel/`, though `analyze_call` can be
pointed at output from `telechannel.stages.bandlimit.apply_bandlimit` (or
the full pipeline, once chained) to build simulated-side fixtures.

VAD note: there is no production VAD model (e.g. Silero VAD) installed in
this environment. Silence-stripping here uses a plain energy-based VAD via
`librosa.effects.split`, which thresholds each frame's RMS relative to the
clip's peak. This is adequate for LTAS purposes (its only job is to keep
PSD estimation from being diluted by silence) but should not be presented
as a real voice-activity-detection model.

Real-call validation status: DOC5's 25 validated real calls have not been
recorded yet (nobody has run the recording protocol). Until they exist,
this module can only be exercised against synthetic "real" stand-ins.
See tests/telechannel/test_validate.py for how the synthetic fixture is
built and the caveat this implies.
"""

from __future__ import annotations

import numpy as np
import librosa
from scipy.signal import welch

# --- Constants -----------------------------------------------------------

# librosa.effects.split threshold, in dB below the clip's peak amplitude.
# Frames quieter than this are treated as silence and dropped before PSD
# estimation.
VAD_TOP_DB = 30

# Welch PSD window length, in samples, used for LTAS estimation.
NPERSEG = 1024

# The "bandwidth cutoff" is the highest frequency at which the LTAS is
# still within CUTOFF_DB of its peak value (DOC5 §5.6: "-20 dB point").
CUTOFF_DB = -20.0

# HF ratio: fraction of total spectral energy above this frequency (Hz).
# DOC5 §5.6 specifies 3.5 kHz.
HF_SPLIT_HZ = 3500.0


def _strip_silence(x: np.ndarray, sr: int, top_db: float = VAD_TOP_DB) -> np.ndarray:
    """
    Energy-based VAD: keep only the non-silent intervals of `x`.

    Uses `librosa.effects.split`, which finds intervals whose RMS energy
    (computed per short frame) exceeds `top_db` decibels below the clip's
    peak amplitude. This is a simple energy-gate VAD, not a learned voice-
    activity model — sufficient to strip leading/trailing/inter-utterance
    silence before spectral averaging, but it will not reject non-speech
    sounds (e.g. steady background noise) that happen to be loud.

    Returns the concatenation of all detected non-silent segments. If no
    non-silent interval is found (e.g. a near-silent clip), the original
    signal is returned unchanged so downstream analysis still has samples
    to work with.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x

    intervals = librosa.effects.split(x, top_db=top_db)
    if len(intervals) == 0:
        return x

    return np.concatenate([x[start:end] for start, end in intervals])


def _compute_ltas(x: np.ndarray, sr: int, nperseg: int = NPERSEG):
    """
    Compute the long-term average spectrum (LTAS) of `x` via Welch's method.

    Welch's method averages the periodogram over overlapping windows,
    which is exactly what an LTAS is: a time-averaged power spectral
    density. Returns (freqs, psd_db), where psd_db is the PSD in dB
    (10*log10), floored to avoid -inf on exact zeros.

    Args:
        x: 1-D signal (already silence-stripped).
        sr: sample rate in Hz.
        nperseg: Welch window length in samples.

    Returns:
        freqs: 1-D array of frequency bins (Hz).
        psd_db: 1-D array of PSD values in dB, same length as freqs.
    """
    x = np.asarray(x, dtype=np.float64)
    nperseg_eff = min(nperseg, max(len(x), 8))
    freqs, psd = welch(x, fs=sr, nperseg=nperseg_eff)
    psd_db = 10 * np.log10(np.maximum(psd, 1e-20))
    return freqs, psd_db


def _find_bandwidth_cutoff(freqs: np.ndarray, psd_db: np.ndarray, cutoff_db: float = CUTOFF_DB) -> float:
    """
    Find the bandwidth cutoff: the highest frequency at which the LTAS is
    still within `cutoff_db` dB of its peak (DOC5's "-20 dB point").

    Method: normalize psd_db so its peak is 0 dB, then walk frequency bins
    from the peak upward and return the last bin at or above `cutoff_db`
    before the spectrum drops below it and never recovers above the
    threshold from that point on (i.e. the true rolloff, not a transient
    dip-then-recovery in a noisy spectrum). Concretely: find the last index
    whose relative level is still >= cutoff_db, restricted to the
    contiguous run starting at the peak.

    Returns the cutoff frequency in Hz. If every bin is above the
    threshold (e.g. a very short/flat clip), returns the Nyquist
    frequency (last freq bin).
    """
    if len(freqs) == 0:
        return 0.0

    peak_db = np.max(psd_db)
    rel_db = psd_db - peak_db
    peak_idx = int(np.argmax(psd_db))

    # Walk upward from the peak; stop at the first bin that falls below
    # cutoff_db and does not come back above it later (avoids stopping on
    # a single noisy dip).
    above = rel_db >= cutoff_db
    cutoff_idx = len(freqs) - 1
    for i in range(peak_idx, len(freqs)):
        if not above[i] and not np.any(above[i:]):
            cutoff_idx = max(i - 1, peak_idx)
            break

    return float(freqs[cutoff_idx])


def _hf_ratio(freqs: np.ndarray, psd_db: np.ndarray, split_hz: float = HF_SPLIT_HZ) -> float:
    """
    Fraction of total spectral energy at/above `split_hz`.

    PSD is converted back from dB to linear power before integrating, so
    the ratio is a true energy ratio (not a dB-domain average).
    """
    if len(freqs) == 0:
        return 0.0

    power = 10 ** (psd_db / 10.0)
    total = np.sum(power)
    if total <= 0:
        return 0.0

    hf_mask = freqs >= split_hz
    return float(np.sum(power[hf_mask]) / total)


def analyze_call(wav_path, sr: int | None = None) -> dict:
    """
    Analyze a phone-call recording per DOC5 §5.6.

    Pipeline:
      1. Load the audio (mono).
      2. VAD-strip silence (energy-based, see `_strip_silence`).
      3. Compute the LTAS (PSD via Welch's method).
      4. Find the -20 dB bandwidth cutoff.
      5. Compute the HF ratio (energy above 3.5 kHz).

    Args:
        wav_path: path to a WAV (or other librosa-readable) file, OR a
            (signal, sr) tuple / (signal, None) pair for in-memory audio
            (used by tests to avoid round-tripping through disk).
        sr: if given, resample to this rate on load; if None, the file's
            native rate is used (or the rate given alongside an in-memory
            signal).

    Returns:
        dict with keys:
            "cutoff": bandwidth cutoff frequency in Hz (float).
            "ltas": {"freqs": np.ndarray, "psd_db": np.ndarray}.
            "hf_ratio": fraction of energy above 3.5 kHz (float, 0..1).
    """
    if isinstance(wav_path, tuple):
        x, loaded_sr = wav_path
        x = np.asarray(x, dtype=np.float64)
        if sr is not None and loaded_sr != sr:
            x = librosa.resample(x, orig_sr=loaded_sr, target_sr=sr)
            loaded_sr = sr
    else:
        x, loaded_sr = librosa.load(str(wav_path), sr=sr, mono=True)
        x = np.asarray(x, dtype=np.float64)

    voiced = _strip_silence(x, loaded_sr)
    freqs, psd_db = _compute_ltas(voiced, loaded_sr)
    cutoff = _find_bandwidth_cutoff(freqs, psd_db)
    hf_ratio = _hf_ratio(freqs, psd_db)

    return {
        "cutoff": cutoff,
        "ltas": {"freqs": freqs, "psd_db": psd_db},
        "hf_ratio": hf_ratio,
    }


def compare_to_sim(real_metrics: dict, sim_metrics: dict) -> dict:
    """
    Compare a real-call `analyze_call` result against a simulated one.

    LTAS spectra are compared on their overlapping frequency range,
    resampling the simulated LTAS onto the real LTAS's frequency grid via
    linear interpolation if the two grids differ (they will, in general,
    if the two clips have different lengths / Welch window counts).

    Returns:
        dict with keys:
            "cutoff_diff_hz": |real cutoff - sim cutoff| (Hz).
            "hf_ratio_diff": |real hf_ratio - sim hf_ratio|.
            "ltas_correlation": Pearson correlation coefficient between
                the two LTAS curves (dB domain), in [-1, 1]. 1.0 means the
                two spectral shapes are identical up to a scale/offset.
    """
    real_freqs = real_metrics["ltas"]["freqs"]
    real_psd = real_metrics["ltas"]["psd_db"]
    sim_freqs = sim_metrics["ltas"]["freqs"]
    sim_psd = sim_metrics["ltas"]["psd_db"]

    if len(real_freqs) == len(sim_freqs) and np.allclose(real_freqs, sim_freqs):
        sim_psd_aligned = sim_psd
    else:
        sim_psd_aligned = np.interp(real_freqs, sim_freqs, sim_psd)

    if np.std(real_psd) == 0 or np.std(sim_psd_aligned) == 0:
        # A perfectly flat spectrum has undefined correlation; treat
        # identical flat spectra as perfectly correlated, else 0.
        correlation = 1.0 if np.allclose(real_psd, sim_psd_aligned) else 0.0
    else:
        correlation = float(np.corrcoef(real_psd, sim_psd_aligned)[0, 1])

    return {
        "cutoff_diff_hz": abs(real_metrics["cutoff"] - sim_metrics["cutoff"]),
        "hf_ratio_diff": abs(real_metrics["hf_ratio"] - sim_metrics["hf_ratio"]),
        "ltas_correlation": correlation,
    }


def plot_validation(real_metrics: dict, sim_metrics: dict, out_path, title: str = "Real vs Simulated LTAS"):
    """
    Save a matplotlib figure overlaying the real-call LTAS against the
    simulated (TeleChannel recipe) LTAS, per DOC5 §5.6 ("overlay vs
    matching TeleChannel recipes").

    The figure is saved directly to `out_path` (never shown interactively)
    so this works headlessly in CI/test environments; a non-interactive
    Agg backend is selected explicitly for the same reason.

    Args:
        real_metrics: output of `analyze_call` for the real recording.
        sim_metrics: output of `analyze_call` for the simulated recording.
        out_path: file path to save the figure to (e.g. "validation.png").
        title: plot title.

    Returns:
        The `out_path` that was written (as given).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    real_freqs = real_metrics["ltas"]["freqs"]
    real_psd = real_metrics["ltas"]["psd_db"]
    sim_freqs = sim_metrics["ltas"]["freqs"]
    sim_psd = sim_metrics["ltas"]["psd_db"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(real_freqs, real_psd, label="Real call", color="tab:blue", linewidth=1.5)
    ax.plot(sim_freqs, sim_psd, label="Simulated (TeleChannel)", color="tab:orange",
            linewidth=1.5, linestyle="--")
    ax.axvline(real_metrics["cutoff"], color="tab:blue", linestyle=":", alpha=0.6,
               label=f"Real cutoff ({real_metrics['cutoff']:.0f} Hz)")
    ax.axvline(sim_metrics["cutoff"], color="tab:orange", linestyle=":", alpha=0.6,
               label=f"Sim cutoff ({sim_metrics['cutoff']:.0f} Hz)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power spectral density (dB)")
    ax.set_title(title)
    ax.legend(loc="lower left", fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    return out_path
