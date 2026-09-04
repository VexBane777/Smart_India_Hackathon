"""
qa.py — QA gates for TeleChannel pipeline audio validation.

This module implements a set of checks on processed audio arrays to catch
pipeline defects: silence, clipping/wraparound, wrong bandwidth for narrowband
recipes, and wrong duration.

The verify_clip() function aggregates these checks and optionally logs failures
to qa_failures.jsonl.

Reference: DOC1_TELECHANNEL_SPEC.md for the TeleChannel pipeline spec.
"""

import json
from pathlib import Path
from typing import Tuple

import numpy as np
from scipy.signal import butter, sosfiltfilt


# QA threshold constants
MIN_RMS_DB = -50.0  # RMS must be > -50 dBFS (not silent)
NARROWBAND_ENERGY_THRESHOLD_DB = -25.0  # Energy > 3.5 kHz must be < -25 dB
NARROWBAND_CUTOFF_HZ = 3500.0  # Cutoff frequency for narrowband check
MIN_DURATION_S = 1.9  # Minimum clip duration in seconds
MAX_DURATION_S = 2.1  # Maximum clip duration in seconds
QA_FAILURES_LOG = "qa_failures.jsonl"  # Log file for failures


def check_silence(audio: np.ndarray, sr: int = 16000) -> Tuple[bool, str]:
    """
    Check that audio is not silent (RMS > -50 dBFS).

    Args:
        audio: 1-D audio signal (float64), values in [-1, 1].
        sr: Sample rate in Hz (used for context, not calculations).

    Returns:
        (bool, str): (passed, reason)
            - passed: True if audio RMS > -50 dBFS
            - reason: Description of the check result
    """
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:  # Avoid log(0)
        rms_db = -np.inf
    else:
        rms_db = 20 * np.log10(rms)

    if rms_db > MIN_RMS_DB:
        return True, f"RMS check passed: {rms_db:.2f} dB > {MIN_RMS_DB} dB"
    else:
        return False, f"Silence detected: RMS {rms_db:.2f} dB <= {MIN_RMS_DB} dB"


def check_clipping(audio: np.ndarray) -> Tuple[bool, str]:
    """
    Check that audio has no samples at exactly ±1.0 (wraparound clipping).

    Args:
        audio: 1-D audio signal (float64), values in [-1, 1].

    Returns:
        (bool, str): (passed, reason)
            - passed: True if no clipping detected
            - reason: Description of the check result
    """
    # Check for samples at exactly ±1.0 (perfect clipping artifacts)
    has_clipping = np.any(np.isclose(np.abs(audio), 1.0, atol=1e-10))

    if not has_clipping:
        return True, "Clipping check passed: no samples at ±1.0"
    else:
        clip_count = np.sum(np.isclose(np.abs(audio), 1.0, atol=1e-10))
        return False, f"Clipping detected: {clip_count} samples at ±1.0"


def check_narrowband(audio: np.ndarray, sr: int = 16000) -> Tuple[bool, str]:
    """
    Check that narrowband audio has energy > 3.5 kHz attenuated below -25 dB.

    This check verifies that a narrowband-limited signal (e.g., telephony
    passband 300-3400 Hz) has been properly band-limited, with energy above
    3.5 kHz reduced by at least 25 dB relative to the in-band energy.

    Args:
        audio: 1-D audio signal (float64), values in [-1, 1].
        sr: Sample rate in Hz.

    Returns:
        (bool, str): (passed, reason)
            - passed: True if energy > 3.5 kHz is < -25 dB
            - reason: Description of the check result
    """
    # Compute frequency-domain power using FFT
    fft = np.fft.rfft(audio)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(audio), 1 / sr)

    # Find the total in-band power (0-3.5 kHz)
    in_band_idx = freqs <= NARROWBAND_CUTOFF_HZ
    in_band_power = np.sum(power[in_band_idx])

    # Find the out-of-band power (> 3.5 kHz)
    out_band_idx = freqs > NARROWBAND_CUTOFF_HZ
    out_band_power = np.sum(power[out_band_idx])

    if in_band_power < 1e-20:  # Avoid log(0)
        relative_db = -np.inf
    else:
        relative_db = 10 * np.log10(out_band_power / in_band_power)

    if relative_db < NARROWBAND_ENERGY_THRESHOLD_DB:
        return True, f"Narrowband check passed: {relative_db:.2f} dB < {NARROWBAND_ENERGY_THRESHOLD_DB} dB"
    else:
        return False, f"Narrowband check failed: {relative_db:.2f} dB >= {NARROWBAND_ENERGY_THRESHOLD_DB} dB"


def check_duration(audio: np.ndarray, sr: int = 16000) -> Tuple[bool, str]:
    """
    Check that audio duration is within 1.9s - 2.1s.

    Args:
        audio: 1-D audio signal (float64), values in [-1, 1].
        sr: Sample rate in Hz.

    Returns:
        (bool, str): (passed, reason)
            - passed: True if duration is within 1.9s - 2.1s
            - reason: Description of the check result
    """
    duration_s = len(audio) / sr

    if MIN_DURATION_S <= duration_s <= MAX_DURATION_S:
        return True, f"Duration check passed: {duration_s:.3f}s within [{MIN_DURATION_S}, {MAX_DURATION_S}]s"
    else:
        return False, f"Duration check failed: {duration_s:.3f}s outside [{MIN_DURATION_S}, {MAX_DURATION_S}]s"


def verify_clip(
    audio: np.ndarray,
    sr: int = 16000,
    clip_id: str = None,
    is_narrowband: bool = False,
    log_failures: bool = False,
) -> Tuple[bool, str]:
    """
    Verify that an audio clip passes all QA gates.

    Runs silence, clipping, (optionally) narrowband, and duration checks
    on the provided audio array. Returns True only if all enabled checks pass.

    Args:
        audio: 1-D audio signal (float64), values in [-1, 1].
        sr: Sample rate in Hz. Defaults to 16000.
        clip_id: Optional clip identifier for logging purposes.
        is_narrowband: If True, check narrowband energy constraints.
        log_failures: If True, write failures to qa_failures.jsonl.

    Returns:
        (bool, str): (passed, reason)
            - passed: True if all checks pass
            - reason: Concatenated reasons from all check functions
    """
    audio = np.asarray(audio, dtype=np.float64)
    reasons = []
    all_passed = True

    # Run silence check
    passed, reason = check_silence(audio, sr)
    reasons.append(reason)
    all_passed = all_passed and passed

    # Run clipping check
    passed, reason = check_clipping(audio)
    reasons.append(reason)
    all_passed = all_passed and passed

    # Run narrowband check if enabled
    if is_narrowband:
        passed, reason = check_narrowband(audio, sr)
        reasons.append(reason)
        all_passed = all_passed and passed

    # Run duration check
    passed, reason = check_duration(audio, sr)
    reasons.append(reason)
    all_passed = all_passed and passed

    combined_reason = " | ".join(reasons)

    # Log failures if requested
    if log_failures and not all_passed:
        _log_failure(clip_id or "unknown", combined_reason)

    return all_passed, combined_reason


def _log_failure(clip_id: str, reason: str) -> None:
    """
    Append a QA failure record to qa_failures.jsonl.

    Args:
        clip_id: Identifier for the failed clip.
        reason: Description of the failure(s).
    """
    log_path = Path(QA_FAILURES_LOG)
    record = {"clip_id": clip_id, "reason": reason}

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
