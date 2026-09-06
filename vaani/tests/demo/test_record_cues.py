"""
tests for assets/scripts/record_cues.py — Module C Task 3 cue recorder.
"""

import sys
from pathlib import Path

import numpy as np

VAANI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(VAANI_ROOT / "assets" / "scripts"))

from record_cues import record_cues  # noqa: E402


class TestRecordCues:
    def test_call_A_first_alert_matches_clone_entry(self):
        audio = np.zeros(int(90 * 16000))
        sheet = record_cues(audio, "call_A")
        assert sheet["clone_entry_s"] == 22.0
        assert sheet["first_alert_t"] is not None
        onset_delay = sheet["first_alert_t"] + sheet["window_s"] - sheet["clone_entry_s"]
        assert 2.0 <= onset_delay <= 6.0

    def test_call_N_never_alerts(self):
        audio = np.zeros(int(90 * 16000))
        sheet = record_cues(audio, "call_N")
        assert sheet["clone_entry_s"] is None
        assert sheet["first_alert_t"] is None

    def test_cues_keyed_by_timestamp_string(self):
        audio = np.zeros(int(10 * 16000))
        sheet = record_cues(audio, "call_A")
        assert "0.0" in sheet["cues"]
        assert all(0.0 <= v <= 1.0 for v in sheet["cues"].values())
