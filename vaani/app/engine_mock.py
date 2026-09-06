"""
engine_mock.py — Mock score engine for the Task 4 UI skeleton.

Implements the master plan §6 decision logic exactly as the real engine will:
    per-window score -> EMA smoothing -> ALERT when the smoothed score exceeds
    the threshold for 2+ consecutive windows.

Module B swap point: `ScoreBackend` is the only seam. Once Module B Task 2
delivers a loadable checkpoint, implement `OnnxBackend` (or similar) returning
real per-window probabilities — everything downstream (EMA, alert state machine,
WebSocket payloads, UI) is unchanged.

Score model for the mock: synthetic-speech windows get moderately high scores
with jitter; real-voice windows get low scores. The demo scripts' clone entry
time (assets/raw/call_scripts.json, segment 2 start = 22 s) is used to emulate
"risk spikes exactly when the cloned voice speaks" (master plan §6).
"""

from typing import Optional

import numpy as np

WINDOW_S = 2.0       # scoring window length (Doc 4: 2 s windows)
HOP_S = 0.5          # refresh every 0.5 s (master plan §2)
EMA_ALPHA = 0.7      # EMA smoothing factor; fast enough that a single
                     # spike decays below threshold within one hop (the
                     # 2-consecutive rule would otherwise be defeated by
                     # the EMA's own memory), slow enough to smooth jitter.
ALERT_THRESHOLD = 0.6
CONSECUTIVE_REQUIRED = 2  # master plan §6: 2+ consecutive windows
CLONE_ENTRY_S = 22.0      # call_scripts.json: segment 2 starts at 22 s

SR = 16000  # engine's native rate (pipeline default)


class ScoreBackend:
    """Interface for per-window scoring. Mock implementation below."""

    def score_window(self, audio: np.ndarray, sr: int) -> float:
        raise NotImplementedError


class MockBackend(ScoreBackend):
    """Position-aware mock backend (deterministic; rng-seeded jitter only)."""

    def __init__(self, clone_entry_s: float = CLONE_ENTRY_S, seed: int = 7):
        self.clone_entry_s = clone_entry_s
        self.rng = np.random.default_rng(seed)

    def score_window(self, audio: np.ndarray, sr: int, t_start_s: float) -> float:
        mid = t_start_s + len(audio) / sr / 2.0
        if mid >= self.clone_entry_s:
            base = 0.85
        elif mid >= self.clone_entry_s - 0.25:
            # transition window straddling the clone entry: partial risk
            base = 0.55
        else:
            base = 0.12
        return float(np.clip(self.rng.normal(base, 0.05), 0.0, 1.0))


class AlertStateMachine:
    """
    EMA + 2-consecutive-window alert rule (master plan §6).

    Feed `update(raw_score)` per window; returns ("normal"|"warn"|"alert").
    """

    def __init__(self,
                 threshold: float = ALERT_THRESHOLD,
                 consecutive_required: int = CONSECUTIVE_REQUIRED,
                 ema_alpha: float = EMA_ALPHA):
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self.alpha = ema_alpha
        self.ema: Optional[float] = None
        self.consecutive_high = 0

    def update(self, raw_score: float) -> str:
        if self.ema is None:
            self.ema = raw_score
        else:
            self.ema = self.alpha * raw_score + (1 - self.alpha) * self.ema
        if self.ema >= self.threshold:
            self.consecutive_high += 1
        else:
            self.consecutive_high = 0
        if self.consecutive_high >= self.consecutive_required:
            return "alert"
        if self.consecutive_high > 0:
            return "warn"
        return "normal"
