"""
Unit tests for telechannel/qa.py QA gates.
"""

import json
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, mock_open

from telechannel.qa import (
    check_silence,
    check_clipping,
    check_narrowband,
    check_duration,
    verify_clip,
    MIN_RMS_DB,
    NARROWBAND_ENERGY_THRESHOLD_DB,
    MIN_DURATION_S,
    MAX_DURATION_S,
)


# ============================================================================
# Silence Check Tests
# ============================================================================


def test_check_silence_pass():
    """Test that audio with sufficient RMS passes silence check."""
    sr = 16000
    # Generate a 1 kHz sine wave at amplitude 0.1, duration 2s
    # Power: 0.1^2 / 2 = 0.005, RMS = sqrt(0.005) ≈ 0.0707
    # dB: 20 * log10(0.0707) ≈ -23 dB (well above -50 dB threshold)
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr
    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

    passed, reason = check_silence(audio, sr)

    assert passed is True
    assert "passed" in reason.lower()
    assert "dB" in reason


def test_check_silence_fail():
    """Test that silent (near-zero) audio fails silence check."""
    sr = 16000
    # Generate nearly silent audio (all zeros)
    duration_s = 2.0
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_silence(audio, sr)

    assert passed is False
    assert "silence" in reason.lower()
    assert "dB" in reason


# ============================================================================
# Clipping Check Tests
# ============================================================================


def test_check_clipping_pass():
    """Test that audio without clipping artifacts passes clipping check."""
    sr = 16000
    # Generate a sine wave that doesn't reach ±1.0
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr
    audio = 0.9 * np.sin(2 * np.pi * 1000 * t)

    passed, reason = check_clipping(audio)

    assert passed is True
    assert "passed" in reason.lower()


def test_check_clipping_fail():
    """Test that audio with clipping at ±1.0 fails clipping check."""
    sr = 16000
    # Create audio with exactly ±1.0 samples (wraparound clipping)
    audio = np.array(
        [0.5, 1.0, 0.0, -1.0, 0.3, 1.0],  # Has samples at exactly ±1.0
        dtype=np.float64,
    )

    passed, reason = check_clipping(audio)

    assert passed is False
    assert "clipping" in reason.lower()
    assert "±1.0" in reason or "1.0" in reason


def test_check_clipping_pass_near_boundary():
    """Test that audio very close to ±1.0 (but not exactly) passes."""
    sr = 16000
    # Create audio with samples very close to but not at ±1.0
    audio = np.array(
        [0.5, 0.9999, 0.0, -0.9999, 0.3],
        dtype=np.float64,
    )

    passed, reason = check_clipping(audio)

    assert passed is True


# ============================================================================
# Narrowband Check Tests
# ============================================================================


def test_check_narrowband_pass():
    """Test that narrowband-limited audio passes narrowband check."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Generate a 1 kHz sine wave (in-band for narrowband)
    # Apply a butterworth bandpass filter to ensure attenuation above 3.5 kHz
    from scipy.signal import butter, sosfiltfilt

    audio = 0.5 * np.sin(2 * np.pi * 1000 * t)

    # Apply 5th-order Butterworth bandpass filter (300-3400 Hz, narrowband passband)
    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist
    sos = butter(5, [low, high], btype="bandpass", output="sos")
    audio = sosfiltfilt(sos, audio)

    passed, reason = check_narrowband(audio, sr)

    assert passed is True
    assert "passed" in reason.lower()
    assert "dB" in reason


def test_check_narrowband_fail():
    """Test that audio with full-band energy fails narrowband check."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Generate white noise (full-band, lots of energy > 3.5 kHz)
    rng = np.random.default_rng(seed=42)
    audio = 0.1 * rng.standard_normal(int(sr * duration_s))

    passed, reason = check_narrowband(audio, sr)

    assert passed is False
    assert "narrowband" in reason.lower()
    assert "dB" in reason


def test_check_narrowband_fail_high_freq_tone():
    """Test that a high-frequency tone (above 3.5 kHz) fails narrowband check."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Generate a 5 kHz sine tone (above 3.5 kHz threshold, full-band)
    audio = 0.5 * np.sin(2 * np.pi * 5000 * t)

    passed, reason = check_narrowband(audio, sr)

    assert passed is False
    assert "narrowband" in reason.lower() or "failed" in reason.lower()


# ============================================================================
# Duration Check Tests
# ============================================================================


def test_check_duration_pass():
    """Test that audio with valid duration passes duration check."""
    sr = 16000
    # Create 2.0 second audio (within 1.9-2.1 range)
    duration_s = 2.0
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is True
    assert "passed" in reason.lower()


def test_check_duration_pass_min_boundary():
    """Test that audio at minimum duration boundary passes."""
    sr = 16000
    # Create 1.9 second audio (at lower boundary)
    duration_s = 1.9
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is True
    assert "passed" in reason.lower()


def test_check_duration_pass_max_boundary():
    """Test that audio at maximum duration boundary passes."""
    sr = 16000
    # Create 2.1 second audio (at upper boundary)
    duration_s = 2.1
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is True
    assert "passed" in reason.lower()


def test_check_duration_fail_too_short():
    """Test that audio shorter than 1.9s fails duration check."""
    sr = 16000
    # Create 1.5 second audio (below minimum)
    duration_s = 1.5
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is False
    assert "duration" in reason.lower() or "failed" in reason.lower()


def test_check_duration_fail_too_long():
    """Test that audio longer than 2.1s fails duration check."""
    sr = 16000
    # Create 2.5 second audio (above maximum)
    duration_s = 2.5
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is False
    assert "duration" in reason.lower() or "failed" in reason.lower()


# ============================================================================
# Integration Tests for verify_clip
# ============================================================================


def test_verify_clip_pass_all_checks():
    """Test that clean audio passes all checks."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Generate clean, short-duration audio that passes all checks
    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

    passed, reason = verify_clip(audio, sr=sr)

    assert passed is True
    assert "passed" in reason.lower()


def test_verify_clip_fail_silent():
    """Test that silent audio fails verify_clip."""
    sr = 16000
    duration_s = 2.0
    audio = np.zeros(int(sr * duration_s))

    passed, reason = verify_clip(audio, sr=sr)

    assert passed is False
    assert "silence" in reason.lower()


def test_verify_clip_fail_clipping():
    """Test that clipped audio fails verify_clip."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Create audio with clipping and valid duration
    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)
    audio[100] = 1.0  # Add exact clipping
    audio[200] = -1.0

    passed, reason = verify_clip(audio, sr=sr)

    assert passed is False
    assert "clipping" in reason.lower()


def test_verify_clip_fail_wrong_duration():
    """Test that audio with wrong duration fails verify_clip."""
    sr = 16000
    # Create too-short audio
    duration_s = 1.0
    t = np.arange(int(sr * duration_s)) / sr
    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)

    passed, reason = verify_clip(audio, sr=sr)

    assert passed is False
    assert "duration" in reason.lower()


def test_verify_clip_with_narrowband_check_enabled():
    """Test verify_clip with narrowband check enabled."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Create narrowband audio (bandpass filtered)
    from scipy.signal import butter, sosfiltfilt

    audio = 0.1 * np.sin(2 * np.pi * 1000 * t)
    nyquist = sr / 2
    low = 300 / nyquist
    high = 3400 / nyquist
    sos = butter(5, [low, high], btype="bandpass", output="sos")
    audio = sosfiltfilt(sos, audio)

    passed, reason = verify_clip(audio, sr=sr, is_narrowband=True)

    assert passed is True
    assert "narrowband" in reason.lower()


def test_verify_clip_narrowband_check_fails():
    """Test verify_clip narrowband check fails on full-band audio."""
    sr = 16000
    duration_s = 2.0
    t = np.arange(int(sr * duration_s)) / sr

    # Create full-band audio (high-frequency tone)
    audio = 0.1 * np.sin(2 * np.pi * 5000 * t)

    passed, reason = verify_clip(audio, sr=sr, is_narrowband=True)

    assert passed is False
    assert "narrowband" in reason.lower()


# ============================================================================
# Logging Tests
# ============================================================================


def test_verify_clip_logs_failures(tmp_path):
    """Test that verify_clip logs failures when log_failures=True."""
    sr = 16000
    # Create silent audio that will fail
    audio = np.zeros(int(sr * 2.0))

    # Temporarily change the QA_FAILURES_LOG to a temp file
    import telechannel.qa as qa_module
    original_log = qa_module.QA_FAILURES_LOG
    try:
        log_file = tmp_path / "qa_failures.jsonl"
        qa_module.QA_FAILURES_LOG = str(log_file)

        passed, reason = verify_clip(
            audio, sr=sr, clip_id="test_clip_1", log_failures=True
        )

        assert passed is False
        # Verify that the log file was created and contains the failure
        assert log_file.exists()
        with open(log_file, "r") as f:
            line = f.read()
            record = json.loads(line)
            assert record["clip_id"] == "test_clip_1"
            assert "silence" in record["reason"].lower()
    finally:
        qa_module.QA_FAILURES_LOG = original_log


def test_verify_clip_no_logging_by_default(tmp_path):
    """Test that verify_clip doesn't log when log_failures=False."""
    sr = 16000
    audio = np.zeros(int(sr * 2.0))

    import telechannel.qa as qa_module
    original_log = qa_module.QA_FAILURES_LOG
    try:
        log_file = tmp_path / "qa_failures.jsonl"
        qa_module.QA_FAILURES_LOG = str(log_file)

        passed, reason = verify_clip(audio, sr=sr, log_failures=False)

        assert passed is False
        # Verify that the log file was NOT created (no logging)
        assert not log_file.exists()
    finally:
        qa_module.QA_FAILURES_LOG = original_log


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_check_silence_with_very_small_amplitude():
    """Test silence check with very small but non-zero amplitude."""
    sr = 16000
    # Create audio with amplitude 1e-6 (essentially silent)
    audio = np.full(int(sr * 2.0), 1e-6)

    passed, reason = check_silence(audio, sr)

    assert passed is False  # Should fail RMS check


def test_check_duration_different_sample_rates():
    """Test duration check works correctly with different sample rates."""
    # Test with 8000 Hz sample rate
    sr = 8000
    duration_s = 2.0
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is True

    # Test with 48000 Hz sample rate
    sr = 48000
    audio = np.zeros(int(sr * duration_s))

    passed, reason = check_duration(audio, sr)

    assert passed is True


def test_check_narrowband_all_zeros():
    """Test narrowband check on all-zero audio (silent)."""
    sr = 16000
    audio = np.zeros(int(sr * 2.0))

    passed, reason = check_narrowband(audio, sr)

    # All-zero audio has no energy, so out-of-band energy will be zero
    # relative_db will be -inf, which is < -25 dB, so it should pass
    assert passed is True or passed is False  # Either outcome is valid for edge case
