"""
tests for app/engine_mock.py — decision-logic correctness (master plan §6).

These verify the contract the REAL engine must also satisfy: EMA smoothing,
2-consecutive-window alert rule, and alert onset ~2-4 s after clone entry
(given the mock's synthetic/real window model).
"""

import numpy as np
import pytest

from app.engine_mock import (
    ALERT_THRESHOLD,
    HOP_S,
    MockBackend,
    SR,
    AlertStateMachine,
    WINDOW_S,
)

CLONE_ENTRY = 22.0


def stream_scores(backend, total_s):
    """Score a synthetic 90 s call in 0.5 s hops, like the real engine would."""
    scores = []
    t = 0.0
    rng = np.random.default_rng(3)
    while t < total_s:
        window_len = int(WINDOW_S * SR)
        # window slides with hop; final windows may overrun total_s — fine
        dummy_audio = rng.standard_normal(int(WINDOW_S * SR)) * 0.01
        scores.append((t, backend.score_window(dummy_audio, SR, t_start_s=t)))
        t += HOP_S
    return scores


class TestMockBackend:
    def test_real_voice_windows_score_low(self):
        backend = MockBackend(seed=1)
        audio = np.zeros(int(WINDOW_S * SR))
        for t in (0.0, 5.0, 10.0, 18.0):
            assert backend.score_window(audio, SR, t_start_s=t) < 0.4

    def test_clone_windows_score_high(self):
        backend = MockBackend(seed=1)
        audio = np.zeros(int(WINDOW_S * SR))
        for t in (25.0, 40.0, 60.0, 85.0):
            assert backend.score_window(audio, SR, t_start_s=t) > 0.6

    def test_deterministic_given_seed(self):
        a1, a2 = MockBackend(seed=5), MockBackend(seed=5)
        audio = np.zeros(int(WINDOW_S * SR))
        assert a1.score_window(audio, SR, t_start_s=30.0) == a2.score_window(audio, SR, t_start_s=30.0)


class TestAlertStateMachine:
    def test_no_alert_on_single_high_window(self):
        sm = AlertStateMachine()
        # one spike above threshold must NOT alert (kills false alarms)
        assert sm.update(0.95) == "warn"
        assert sm.update(0.1) == "normal"

    def test_alert_after_two_consecutive_high(self):
        sm = AlertStateMachine()
        sm.update(0.9)
        assert sm.update(0.9) == "alert"

    def test_ema_smoothing_lags_raw(self):
        sm = AlertStateMachine()
        sm.update(0.1)
        sm.update(0.9)
        # after one high raw, EMA is between the two, not at the raw value
        assert 0.1 < sm.ema < 0.9

    def test_recovery_resets_consecutive_count(self):
        sm = AlertStateMachine()
        sm.update(0.9)
        sm.update(0.9)          # alert
        sm.update(0.05)         # low -> reset
        assert sm.update(0.9) == "warn"   # needs 2 again


class TestDemoCallABehavior:
    """Contract for the paired experiment (Doc 4 §4.6): A alerts, N doesn't."""

    def test_call_A_alerts_after_clone_entry(self):
        backend = MockBackend()
        sm = AlertStateMachine()
        scores = stream_scores(backend, total_s=90.0)
        first_alert_t = None
        for t, raw in scores:
            if sm.update(raw) == "alert" and first_alert_t is None:
                first_alert_t = t
                break
        assert first_alert_t is not None, "Call A must alert"
        # alert ~3-4 s after clone entry (master plan / Doc 4 §4.2):
        # 22 s entry + 2 s window + ~1-2 s of consecutive-rule lag
        onset_delay = first_alert_t + WINDOW_S - CLONE_ENTRY
        assert 2.0 <= onset_delay <= 6.0, f"alert onset {onset_delay}s after clone entry"

    def test_call_N_never_alerts(self):
        # Call N: identical script but no clone — mock models this as a stream
        # with clone_entry_s past the end (i.e. no synthetic speech at all).
        backend = MockBackend(clone_entry_s=1e9)
        sm = AlertStateMachine()
        for _t, raw in stream_scores(backend, total_s=90.0):
            assert sm.update(raw) != "alert"

    def test_threshold_enforced(self):
        assert ALERT_THRESHOLD == pytest.approx(0.6)
