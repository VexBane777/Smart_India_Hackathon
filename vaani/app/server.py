"""
server.py — FastAPI audio streamer + mock score feed (Module C Task 4 Step 1).

Reads a WAV file (48 kHz raw demo asset), resamples to the engine's 16 kHz,
and pushes 0.5 s audio chunks over a WebSocket. Each chunk message carries a
per-window mock risk score + alert state so the Streamlit UI can be built and
rehearsed before Module B's real model lands.

Swap point for Module B: `MockBackend` in engine_mock.py is the only mock;
replace it with the real ONNX/checkpoint backend and this file is unchanged.

Endpoints:
    GET  /api/calls              -> list of available demo calls
    GET  /api/health             -> {"status": "ok", "backend": "mock"}
    WS   /ws/stream/{call_key}   -> JSON messages:
        {"type": "meta", "sr": ..., "chunk_samples": ..., "duration_s": ...,
         "backend": "mock"}
        {"type": "chunk", "t": <sec>, "audio_b64": <16-bit PCM>,
         "raw_score": <float>, "ema": <float>, "state": "normal|warn|alert"}
        {"type": "end"}

Real-time pacing is optional (`?pace=0` streams as fast as possible for
tests; default paces to wall clock so the demo behaves like a live call).
"""

import asyncio
import base64
import json
import wave
from pathlib import Path

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.engine_mock import HOP_S, SR, MockBackend

ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
RAW_DIR = ASSETS_DIR / "raw"
MANIFEST_PATH = ASSETS_DIR / "manifest.json"

CHUNK_S = HOP_S  # 0.5 s chunks, per Task 4 Step 1

app = FastAPI(title="VAANI demo streamer (mock backend)")


def _load_mono_float(path: Path, target_sr: int = SR) -> tuple[np.ndarray, int]:
    """Load WAV as mono float64 [-1, 1], resampled to target_sr (linear)."""
    with wave.open(str(path), "rb") as w:
        sr = w.getframerate()
        nch = w.getnchannels()
        sw = w.getsampwidth()
        if sw != 2:
            raise ValueError(f"expected 16-bit WAV, got sample width {sw}")
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if nch > 1:
        pcm = pcm.reshape(-1, nch).mean(axis=1)
    audio = pcm.astype(np.float64) / 32768.0
    if sr != target_sr:
        n_out = int(round(len(audio) * target_sr / sr))
        audio = np.interp(
            np.linspace(0.0, len(audio) - 1.0, n_out),
            np.arange(len(audio), dtype=np.float64),
            audio,
        )
    return audio, target_sr


def available_calls() -> dict:
    """Map call_key -> {"path", "duration_s"} for WAVs in assets/raw."""
    calls = {}
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for key, entry in manifest.get("assets", {}).items():
            p = ASSETS_DIR.parent / entry["path"]
            if p.exists():
                calls[key] = {"path": str(p)}
        return calls
    # fallback: any *_raw.wav next to this script's asset tree
    for p in sorted(RAW_DIR.glob("call_*_raw.wav")):
        calls[p.name.replace("_raw.wav", "")] = {"path": str(p)}
    return calls


@app.get("/api/health")
async def health():
    return {"status": "ok", "backend": "mock", "note": "Module B swap pending"}


@app.get("/api/calls")
async def list_calls():
    return available_calls()


@app.websocket("/ws/stream/{call_key}")
async def stream_call(ws: WebSocket, call_key: str, pace: float = 1.0):
    """
    Stream `call_key` as 0.5 s chunks with mock scores. `pace` scales the
    wall-clock delay (1.0 = realtime, 0 = as fast as possible, for tests).
    """
    await ws.accept()
    calls = available_calls()
    if call_key not in calls:
        await ws.send_json({"type": "error", "detail": f"unknown call {call_key!r}"})
        await ws.close()
        return

    audio, sr = _load_mono_float(Path(calls[call_key]["path"]), SR)
    backend = MockBackend()
    chunk_n = int(CHUNK_S * SR)
    from app.engine_mock import AlertStateMachine

    state_machine = AlertStateMachine()

    await ws.send_json({
        "type": "meta",
        "sr": sr,
        "chunk_samples": chunk_n,
        "duration_s": round(len(audio) / sr, 3),
        "backend": "mock",
        "window_s": 2.0,
        "hop_s": HOP_S,
    })

    delay = CHUNK_S * pace
    t = 0.0
    for start in range(0, len(audio), chunk_n):
        chunk = audio[start:start + chunk_n]
        # Only score full windows (2 s = 4 chunks accumulate) — emulate the
        # real engine's first-score latency of ~2.5 s (window + hop budget).
        raw = backend.score_window(chunk, sr, t_start_s=t)
        state = state_machine.update(raw)
        await ws.send_json({
            "type": "chunk",
            "t": round(t, 3),
            "audio_b64": base64.b64encode(
                (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
            ).decode("ascii"),
            "raw_score": round(raw, 4),
            "ema": round(state_machine.ema, 4),
            "state": state,
        })
        t += CHUNK_S
        if delay > 0:
            await asyncio.sleep(delay)

    await ws.send_json({"type": "end"})
    await ws.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
