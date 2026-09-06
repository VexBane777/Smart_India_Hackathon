"""
record_cues.py — Module C Task 3: record per-window mock scores to a
`{timestamp: score}` JSON cue sheet, for pitch-deck timing overlays.

Deviation from the plan text ("Modify vaani.engine to output ... when
--record-cues is passed"): no `vaani.engine` module exists anywhere in this
repo (checked — Module B's scoring lives in `models/`, the streaming engine
is `app/engine_mock.py` per Module C Task 4). This script reuses
`app.engine_mock.MockBackend`/`AlertStateMachine` directly instead, which is
the same mock-backend swap point Task 4 already established: once Module B
ships a real backend, this script's `--record-cues` behavior moves with it
with a one-line backend swap, same as `app/server.py`.

Usage:
    python assets/scripts/record_cues.py --call call_A \\
        --input assets/demo/call_A_whatsapp.wav \\
        --out assets/demo/call_A_whatsapp.cues.json
"""

import argparse
import json
import sys
import wave
from pathlib import Path

import numpy as np

VAANI_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(VAANI_ROOT))

from app.engine_mock import (  # noqa: E402
    HOP_S,
    SR,
    WINDOW_S,
    AlertStateMachine,
    MockBackend,
    clone_entry_for_call,
)


def _read_wav_mono_float(path: Path, target_sr: int = SR) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float64) / 32768.0
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = np.interp(
            np.linspace(0.0, len(audio) - 1.0, n_out),
            np.arange(len(audio), dtype=np.float64),
            audio,
        )
    return audio


def record_cues(audio: np.ndarray, call_key: str) -> dict:
    """Score `audio` in WINDOW_S-long, HOP_S-hop windows; return a cue sheet."""
    clone_entry_s = clone_entry_for_call(call_key)
    backend = MockBackend(clone_entry_s=clone_entry_s)
    sm = AlertStateMachine()

    window_n = int(WINDOW_S * SR)
    hop_n = int(HOP_S * SR)
    cues = {}
    first_alert_t = None
    t = 0.0
    start = 0
    while start < len(audio):
        window = audio[start:start + window_n]
        raw = backend.score_window(window, SR, t_start_s=t)
        state = sm.update(raw)
        cues[f"{t:.1f}"] = round(raw, 4)
        if state == "alert" and first_alert_t is None:
            first_alert_t = t
        t += HOP_S
        start += hop_n

    return {
        "call": call_key,
        "backend": "mock",
        "window_s": WINDOW_S,
        "hop_s": HOP_S,
        "clone_entry_s": clone_entry_s,
        "first_alert_t": first_alert_t,
        "cues": cues,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--call", required=True, help="call key, e.g. call_A")
    p.add_argument("--input", required=True, type=Path, help="WAV to score")
    p.add_argument("--out", required=True, type=Path, help="cues JSON output path")
    args = p.parse_args(argv)

    audio = _read_wav_mono_float(args.input)
    cue_sheet = record_cues(audio, args.call)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cue_sheet, indent=2) + "\n", encoding="utf-8")
    print(
        f"wrote {args.out} ({len(cue_sheet['cues'])} windows, "
        f"first_alert_t={cue_sheet['first_alert_t']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
