"""
pipeline.py — the laptop side of the phone's scoring pipeline, stage by stage.

Why this exists
---------------
The console's job is to make a number *explainable*: given a verdict, which
window produced it, what did the features look like, what did the logits say,
what did the smoothing and the threshold rule do, and does the integration API
agree? So this module is deliberately written as an explicit sequence of named
stages that each publish an event, rather than one opaque `score(file)` call.

Nothing here is re-implemented:
  * `model_training/features.py`   — the numpy port of the phone's Dart extractor
  * `backend/model_backend.py`     — the deployed ONNX artifact
  * `vaani/app/engine_mock.py`     — the EMA + 2-consecutive-window rule the
                                     phone (`RiskScoreProvider`) and the
                                     integration API both use
so a laptop number is comparable to a phone number by construction. The one
thing this file owns is *windowing* (3 s window, 0.5 s hop), which it mirrors
from `lib/services/audio_service.dart`'s file-scan path, and the diff logic
against the backend.
"""
from __future__ import annotations

import asyncio
import json
import math
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

# Same convention backend/main.py already uses for the vaani import: put the
# sibling package on sys.path rather than duplicating code across the two.
_VOICE_GUARD_DIR = Path(__file__).resolve().parents[1]
for _p in (_VOICE_GUARD_DIR / "backend", _VOICE_GUARD_DIR / "model_training",
           Path(__file__).resolve().parents[2] / "vaani"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import features  # noqa: E402
from app.engine_mock import ALERT_THRESHOLD, AlertStateMachine  # noqa: E402
from model_backend import ModelUnavailable, OnnxModelBackend  # noqa: E402
from monitor import MonitorLog  # noqa: E402

# --- pipeline constants, mirrored from the phone ------------------------------
SR = 16000                    # AudioProcessor.sampleRate
WINDOW_SAMPLES = 48000        # AudioProcessor.chunkSamples == 3 s
DEFAULT_HOP_SAMPLES = 8000    # audio_service.dart:203 — 0.5 s sliding hop
SILENCE_RMS = 50.0 / 32768.0  # audio_service.dart:65 — RMS gate, normalized domain

# Verdict bands, from lib/utils/constants.dart (defaultThresholdLow/High).
BAND_LOW = 0.30
BAND_HIGH = 0.70

DIFF_TOLERANCE = 0.10  # |local - backend| at or below this counts as agreement


def verdict_for(score: float) -> str:
    """Same band vocabulary the phone's `ShadRiskMeter` and the API use."""
    if score < BAND_LOW:
        return "VERIFIED_HUMAN"
    if score < BAND_HIGH:
        return "SUSPICIOUS"
    return "AI_DETECTED"


def rms(pcm: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(pcm)))) if len(pcm) else 0.0


def dbfs(amplitude: float) -> float:
    """[0,1] amplitude -> dBFS, floored so silence is a number, not -inf."""
    return round(20.0 * math.log10(max(float(amplitude), 1e-9)), 2)


def resample_linear(pcm: np.ndarray, sr_in: int, sr_out: int = SR) -> np.ndarray:
    """Linear-interpolation resample — the same method as Dart's resampleLinear
    and `vaani/app/server.py`'s `_load_mono_float`. Resampling differently on
    the laptop would show up as a score difference and be blamed on the model.
    """
    if sr_in == sr_out or len(pcm) == 0:
        return pcm
    n_out = int(round(len(pcm) * sr_out / sr_in))
    return np.interp(
        np.linspace(0.0, len(pcm) - 1.0, n_out),
        np.arange(len(pcm), dtype=np.float64),
        pcm,
    )


def _decode_audio(path: Path) -> tuple[np.ndarray, int, dict[str, Any]]:
    """Read an audio file to mono float64 [-1, 1]; soundfile first, stdlib `wave` as fallback.

    Why a cascade rather than one decoder: the repo's own test assets are not
    all 16-bit PCM — `model_training/test_assets/ai_clone_test_clip.wav` is
    IEEE-float, and stdlib `wave` cannot open a float WAV at all (it raises
    "unknown format: 3"). The phone's `AudioProcessor.decodePcm16Wav` rejects
    float files outright, so this is a *real* difference between the two paths.
    Rather than hide it, the decoder and the sample format are recorded in the
    metadata and logged, so a laptop score is never quietly reported as if the
    device could have produced it from the same file.
    """
    try:
        import soundfile as sf  # optional dependency; present in this repo's Python env
    except Exception:
        sf = None
    soundfile_error = "soundfile not installed"
    if sf is not None:
        try:
            data, sr_in = sf.read(str(path), dtype="float64", always_2d=True)
            info = sf.info(str(path))
            audio = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
            meta: dict[str, Any] = {
                "decoder": "soundfile", "format": info.format, "subtype": info.subtype,
                "sr_in": int(sr_in), "channels": int(data.shape[1]),
                "frames": int(info.frames), "duration_s": round(info.duration, 3),
                "device_parity": info.subtype == "PCM_16",
            }
            return audio, int(sr_in), meta
        except Exception as exc:
            soundfile_error = f"{type(exc).__name__}: {exc}"

    with wave.open(str(path), "rb") as w:
        sr_in, n_ch, width, n_frames = (w.getframerate(), w.getnchannels(),
                                        w.getsampwidth(), w.getnframes())
        raw = np.frombuffer(w.readframes(n_frames), dtype=np.int16)
    if width != 2:
        raise ValueError(
            f"{path.name}: not 16-bit PCM (sample width {width}) and soundfile could not read it "
            f"either ({soundfile_error}). The on-device decoder only accepts 16-bit PCM WAV."
        )
    audio = raw.astype(np.float64) / 32768.0
    if n_ch > 1:
        audio = audio.reshape(-1, n_ch).mean(axis=1)
    meta = {
        "decoder": "wave (stdlib)", "format": "WAV", "subtype": "PCM_16",
        "sr_in": int(sr_in), "channels": int(n_ch), "frames": int(n_frames),
        "duration_s": round(n_frames / sr_in, 3) if sr_in else 0.0,
        "device_parity": True,
    }
    return audio, int(sr_in), meta


def load_wav_mono(path: Path | str, target_sr: int = SR) -> tuple[np.ndarray, dict[str, Any]]:
    """Decode an audio file to mono float64 [-1, 1] at `target_sr`, with provenance.

    `meta["device_parity"]` is True only when the file is 16-bit PCM — the one
    format `AudioProcessor.decodePcm16Wav` accepts. A False there means the
    phone's own file-scan button would have refused this file, so any
    laptop-vs-phone comparison for it is not like-for-like. That is logged.
    """
    path = Path(path)
    audio, sr_in, meta = _decode_audio(path)
    meta.update({
        "path": str(path), "file": path.name, "size_bytes": path.stat().st_size,
        "samples_mono": int(len(audio)), "sr_out": target_sr,
    })
    resampled = resample_linear(audio, sr_in, target_sr)
    meta["samples_16k"] = int(len(resampled))
    return resampled, meta


def iter_windows(pcm: np.ndarray, window: int = WINDOW_SAMPLES, hop: int = DEFAULT_HOP_SAMPLES,
                 pad_short: bool = False):
    """Yield `(index, start_sample, t_start_s, chunk)` for each 3 s window.

    A trailing partial window is dropped, not zero-padded: `dataset.py`'s
    windowing contract says the app "scores a 3 s window every second and never
    pads", and a zero-padded tail would be scored as silence — which this model
    reads as highly synthetic.

    `pad_short` handles the *other* case, and mirrors `audio_service.dart:204-206`
    exactly: a clip shorter than one window is front-padded with digital silence
    so the phone's "Test with audio file" path can still score it
    (`tts_elevenlabs_sample.wav` in this repo is 2.32 s, so without this it would
    produce no score at all). The runner logs that it happened, because silence
    padding is precisely what this model mis-reads — so that score deserves
    suspicion, and hiding the padding would hide the reason.
    """
    if len(pcm) < window:
        if not pad_short or len(pcm) == 0:
            return
        yield 0, 0, 0.0, np.concatenate([np.zeros(window - len(pcm), dtype=pcm.dtype), pcm])
        return
    index, start = 0, 0
    while start + window <= len(pcm):
        yield index, start, start / SR, pcm[start:start + window]
        index += 1
        start += hop


def extract_window_features(chunk: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """Run `features.py` (the numpy port of the phone's Dart extractor) on one window.

    Module-level and free of state so it can be handed to `asyncio.to_thread`:
    it is CPU-bound and would otherwise stall the loop the monitor endpoints are
    served from while a window is being processed.
    """
    sequence = features.extract_lfcc_sequence(chunk)
    scalars = features.extract_scalars(chunk)
    return sequence.tolist(), scalars.tolist()


# ------------------------------------------------------------------- the diff
@dataclass
class DiffResult:
    """Local ONNX score vs the integration API's score, for one window.

    Two axes, deliberately kept separate because they mean different things:

    * `band_mismatch` — the two scores land in different verdict bands. This is
      the diagnostically loud case: one says VERIFIED HUMAN, the other says AI
      DETECTED, for the same audio.
    * `agree` — the numeric scores are within `tolerance`. A small
      band-free difference in smoothing is uninteresting; a large one is not.
    """

    local_raw: float
    local_ema: float
    backend_raw: Optional[float]
    backend_ema: Optional[float]
    verdict_local: str
    verdict_backend: Optional[str]
    delta_raw: Optional[float]
    delta_ema: Optional[float]
    agree: Optional[bool]
    band_mismatch: Optional[bool]
    flags: list[str] = field(default_factory=list)
    error: Optional[str] = None
    backend_mode: Optional[str] = None
    backend_reports: Optional[str] = None
    latency_ms: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_raw": self.local_raw,
            "local_ema": self.local_ema,
            "backend_raw": self.backend_raw,
            "backend_ema": self.backend_ema,
            "verdict_local": self.verdict_local,
            "verdict_backend": self.verdict_backend,
            "delta_raw": self.delta_raw,
            "delta_ema": self.delta_ema,
            "agree": self.agree,
            "band_mismatch": self.band_mismatch,
            "flags": list(self.flags),
            "error": self.error,
            "backend_mode": self.backend_mode,
            "backend_reports": self.backend_reports,
            "latency_ms": self.latency_ms,
        }


def compute_diff(
    *,
    local_raw: float,
    local_ema: float,
    backend: Optional[dict[str, Any]],
    error: Optional[str] = None,
    backend_mode: Optional[str] = None,
    tolerance: float = DIFF_TOLERANCE,
) -> DiffResult:
    """Pure function so the flag rules are testable without running any service."""
    verdict_local = verdict_for(local_ema)
    if backend is None:
        kind = "backend_timeout" if error and "timeout" in error.lower() else "backend_unavailable"
        return DiffResult(
            local_raw=round(local_raw, 4), local_ema=round(local_ema, 4),
            backend_raw=None, backend_ema=None, verdict_local=verdict_local, verdict_backend=None,
            delta_raw=None, delta_ema=None, agree=None, band_mismatch=None,
            flags=[kind], error=error, backend_mode=backend_mode,
        )

    backend_ema = float(backend.get("riskScore", 0.0))
    backend_raw_value = backend.get("rawScore")
    backend_raw = float(backend_raw_value) if backend_raw_value is not None else None
    verdict_backend = str(backend.get("verdict") or verdict_for(backend_ema))
    delta_ema = round(backend_ema - local_ema, 4)
    delta_raw = None if backend_raw is None else round(backend_raw - local_raw, 4)

    # Compare raw-vs-raw when the backend reports its raw score, because that is
    # the scorer output before either side's EMA has had a chance to differ.
    # Falling back to EMA-vs-EMA keeps the comparison meaningful against an
    # older backend that does not return `rawScore`.
    if backend_raw is not None:
        reference, local_reference = backend_raw, local_raw
    else:
        reference, local_reference = backend_ema, local_ema
    agree = abs(reference - local_reference) <= tolerance
    band_mismatch = verdict_backend != verdict_local

    flags: list[str] = ["agree" if agree else "diverge"]
    if band_mismatch:
        flags.append("band_mismatch")
    if verdict_backend != verdict_for(backend_ema):
        # The backend's state-machine verdict disagrees with its own band
        # mapping — worth surfacing, because it means the two are not the same
        # rule and the comparison is apples-to-oranges.
        flags.append("backend_verdict_vs_band")
    reports = backend.get("modelBackend")
    flags.append(f"backend_{reports}" if reports else "backend_unlabelled")

    return DiffResult(
        local_raw=round(local_raw, 4), local_ema=round(local_ema, 4),
        backend_raw=backend_raw, backend_ema=round(backend_ema, 4),
        verdict_local=verdict_local, verdict_backend=verdict_backend,
        delta_raw=delta_raw, delta_ema=delta_ema, agree=agree, band_mismatch=band_mismatch,
        flags=flags, backend_mode=backend_mode, backend_reports=reports,
        latency_ms=backend.get("latencyMs"),
    )


# ------------------------------------------------------------- backend client
class BackendError(RuntimeError):
    """A backend call that did not produce a usable score. Never swallowed silently."""

    def __init__(self, kind: str, message: str):
        super().__init__(message)
        self.kind = kind


def _post_json_sync(url: str, payload: dict, api_key: str, timeout: float) -> dict:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-API-Key": api_key},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:400]
        raise BackendError("http_error", f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        kind = "timeout" if "timed out" in reason.lower() else "unreachable"
        raise BackendError(kind, reason) from exc
    except json.JSONDecodeError as exc:
        raise BackendError("bad_json", str(exc)) from exc


def _get_json_sync(url: str, api_key: str, timeout: float) -> dict:
    request = urllib.request.Request(url, method="GET", headers={"X-API-Key": api_key})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise BackendError("http_error", f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = str(getattr(exc, "reason", exc))
        kind = "timeout" if "timed out" in reason.lower() else "unreachable"
        raise BackendError(kind, reason) from exc


class BackendClient:
    """Calls the integration API with the *same window* the local model scored.

    `mode` decides which comparison the diff shows, and is always logged:

    * `"model"` (default) — send `lfcc_sequence`+`scalars` so the backend scores
      with the same ONNX artifact. Any difference is therefore a genuine
      pipeline/transport/state bug, not a scorer difference.
    * `"heuristic"` — send only the 60-d pooled `lfcc`+prosody, i.e. exactly what
      `lib/services/api_service.dart` sends. Differences here are expected and
      are the point: they show how far the legacy SDK path sits from the model.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8001", api_key: str = "vg_demo_key",
                 timeout: float = 3.0, mode: str = "model"):
        if mode not in ("model", "heuristic"):
            raise ValueError(f"mode must be 'model' or 'heuristic', got {mode!r}")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.mode = mode

    def analyze_payload(self, *, lfcc_sequence: list[list[float]], scalars: Sequence[float],
                        session_id: str) -> dict:
        """The exact body this client sends — exposed so tests and the UI can show it."""
        payload: dict[str, Any] = {
            "session_id": session_id,
            "metadata": {"source": "laptop_monitor"},
        }
        if self.mode == "model":
            payload["lfcc_sequence"] = lfcc_sequence
            payload["scalars"] = list(scalars)
        else:
            pooled = np.asarray(lfcc_sequence, dtype=np.float64).mean(axis=0).tolist()
            payload["lfcc"] = pooled
            payload["prosody"] = {
                "pauseRatio": float(scalars[0]) if len(scalars) > 0 else 0.0,
                "energyVar": float(scalars[1]) if len(scalars) > 1 else 0.0,
                "zcrVar": float(scalars[2]) if len(scalars) > 2 else 0.0,
            }
        return payload

    async def analyze(self, *, lfcc_sequence: list[list[float]], scalars: Sequence[float],
                      session_id: str) -> dict:
        payload = self.analyze_payload(lfcc_sequence=lfcc_sequence, scalars=scalars,
                                       session_id=session_id)
        # urllib is blocking; the console must keep streaming while the API
        # thinks, so the call runs in a thread and the loop stays responsive.
        return await asyncio.to_thread(
            _post_json_sync, f"{self.base_url}/v1/analyze-chunk", payload, self.api_key, self.timeout
        )

    async def reset(self, session_id: str) -> bool:
        try:
            await asyncio.to_thread(
                _post_json_sync, f"{self.base_url}/v1/reset/{session_id}", {}, self.api_key, self.timeout
            )
            return True
        except BackendError:
            return False

    async def model_info(self) -> Optional[dict]:
        try:
            return await asyncio.to_thread(
                _get_json_sync, f"{self.base_url}/v1/model-info", self.api_key, self.timeout
            )
        except BackendError:
            return None


# ------------------------------------------------------------------- the run
@dataclass
class RunnerConfig:
    """Everything an operator or an agent can change mid-run."""

    window_samples: int = WINDOW_SAMPLES
    hop_samples: int = DEFAULT_HOP_SAMPLES
    speed: float = 1.0            # 1.0 = real time, 0 = as fast as the CPU allows
    threshold: float = ALERT_THRESHOLD
    loop: bool = False
    backend_enabled: bool = True


class PipelineRunner:
    """Drives decode -> window -> features -> ONNX -> EMA -> decision -> backend diff.

    Designed to be watched: one event per stage per window, a live `state` block
    for `/monitor.json`, and pause/step/resume controls so a reader can freeze on
    a single window and inspect it without the next one overwriting the numbers.
    """

    def __init__(self, monitor: MonitorLog, model: OnnxModelBackend,
                 backend: BackendClient, config: Optional[RunnerConfig] = None):
        self.monitor = monitor
        self.model = model
        self.backend = backend
        self.config = config or RunnerConfig()
        self._task: Optional[asyncio.Task] = None
        self._running = False            # the loop is advancing windows
        self._paused = False
        self._pending_steps = 0
        self._wake = asyncio.Event()
        self._sm: Optional[AlertStateMachine] = None
        self._session_id = ""
        self._source: dict[str, Any] = {}
        self._stop_requested = False
        self._tally: dict[str, Any] = self._empty_tally()
        self._backend_error_count = 0
        self._window_index = 0
        self._e2e_ms: list[float] = []

    # ------------------------------------------------------------- controls
    @property
    def is_active(self) -> bool:
        return self._task is not None and not self._task.done()

    def pause(self) -> None:
        self._paused = True
        self._running = False
        self.monitor.log("control", "paused — the next window will not start",
                         severity="warn", paused=True)
        self.publish_state()

    def resume(self) -> None:
        self._paused = False
        self._running = True
        self._wake.set()
        self.monitor.log("control", "resumed")
        self.publish_state()

    def step(self) -> None:
        """Advance exactly one window while paused."""
        self._paused = True
        self._running = False
        self._pending_steps += 1
        self._wake.set()
        self.monitor.log("control", "single-step: one window", pending_steps=self._pending_steps)

    def set_threshold(self, value: float) -> None:
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("threshold must be within [0, 1]")
        old = self.config.threshold
        self.config.threshold = value
        if self._sm is not None:
            self._sm.threshold = value  # takes effect on the very next window
        self.monitor.log("control", f"alert threshold {old:.3f} -> {value:.3f}",
                         severity="warn", old=old, new=value)
        self.publish_state()

    def set_hop(self, samples: int) -> None:
        samples = int(samples)
        if not 1 <= samples <= self.config.window_samples:
            raise ValueError("hop must be between 1 and the window size")
        old, self.config.hop_samples = self.config.hop_samples, samples
        self.monitor.log("control",
                         f"hop {old} -> {samples} samples ({old / SR:.2f}s -> {samples / SR:.2f}s)",
                         severity="warn", old=old, new=samples)

    def set_speed(self, speed: float) -> None:
        old, self.config.speed = self.config.speed, float(speed)
        self.monitor.log("control", f"pacing speed {old} -> {self.config.speed}",
                         severity="warn", old=old, new=self.config.speed)

    def set_loop(self, loop: bool) -> None:
        self.config.loop = bool(loop)
        self.monitor.log("control", f"loop -> {self.config.loop}", loop=self.config.loop)

    def set_backend(self, enabled: bool, mode: Optional[str] = None) -> None:
        self.config.backend_enabled = bool(enabled)
        if mode:
            if mode not in ("model", "heuristic"):
                raise ValueError("backend mode must be 'model' or 'heuristic'")
            self.backend.mode = mode
        self.monitor.log("control",
                         f"backend comparison -> enabled={self.config.backend_enabled} mode={self.backend.mode}",
                         enabled=self.config.backend_enabled, mode=self.backend.mode)
        self.publish_state()

    def stop(self) -> None:
        self._stop_requested = True
        self._running = True
        self._wake.set()
        self.monitor.log("control", "stop requested — finishing the current window", severity="warn")

    @staticmethod
    def _empty_tally() -> dict[str, Any]:
        return {
            "windows_scored": 0, "windows_silent": 0, "compared": 0, "agree": 0,
            "diverge": 0, "band_mismatch": 0, "backend_errors": 0,
            "sum_abs_delta_raw": 0.0, "alerts": 0,
        }

    # ---------------------------------------------------------------- state
    def publish_state(self, **extra: Any) -> None:
        """Mirror the runner's live numbers into `/monitor.json`'s state block."""
        tally = dict(self._tally)
        compared = tally["compared"]
        tally["agree_pct"] = round(100.0 * tally["agree"] / compared, 1) if compared else None
        tally["mean_abs_delta_raw"] = (
            round(tally["sum_abs_delta_raw"] / compared, 4) if compared else None
        )
        tally["e2e_p50_ms"] = self._percentile(self._e2e_ms, 50)
        tally["e2e_p95_ms"] = self._percentile(self._e2e_ms, 95)
        self.monitor.set_state(
            runner={
                "active": self.is_active,
                "paused": self._paused,
                "pending_steps": self._pending_steps,
                "window_index": self._window_index,
                "source": self._source,
                "config": {
                    "window_samples": self.config.window_samples,
                    "window_s": round(self.config.window_samples / SR, 3),
                    "hop_samples": self.config.hop_samples,
                    "hop_s": round(self.config.hop_samples / SR, 3),
                    "speed": self.config.speed,
                    "threshold": self.config.threshold,
                    "loop": self.config.loop,
                    "backend_enabled": self.config.backend_enabled,
                    "backend_mode": self.backend.mode,
                },
                "tally": tally,
            },
            **extra,
        )

    @staticmethod
    def _percentile(values: Sequence[float], pct: float) -> Optional[float]:
        if not values:
            return None
        ordered = sorted(values)
        idx = min(len(ordered) - 1, max(0, int(round((pct / 100.0) * (len(ordered) - 1)))))
        return round(ordered[idx], 3)

    # ------------------------------------------------------------- lifecycle
    def start(self, wav_path: Path | str) -> asyncio.Task:
        """Begin a paced run over `wav_path`. Replaces any run already going."""
        if self.is_active and self._task is not None:
            self._task.cancel()
        self._stop_requested = False
        self._paused = False
        self._running = True
        self._pending_steps = 0
        self._wake.set()
        self._tally = self._empty_tally()
        self._backend_error_count = 0
        self._e2e_ms = []
        self._window_index = 0
        self._task = asyncio.create_task(self._run(Path(wav_path)))
        return self._task

    async def _run(self, path: Path) -> None:
        own_task = asyncio.current_task()
        self._session_id = f"laptop-{int(time.time())}"
        self._sm = AlertStateMachine(threshold=self.config.threshold)
        # Clear any stale reading from a previous file immediately — otherwise
        # the gauge keeps showing the last window's score (e.g. stuck at 99%)
        # while this one is still decoding, which reads as live audio playing.
        self.publish_state(local={"scored": False, "reason": "loading"}, window={}, diff=None)
        try:
            decode_started = time.perf_counter()
            pcm, meta = await asyncio.to_thread(load_wav_mono, path, SR)
            decode_ms = round((time.perf_counter() - decode_started) * 1000, 3)
            self._source = meta
            self.monitor.log("source_open", f"opened {meta['file']}", **meta)
            self.monitor.log(
                "decode",
                f"decoded {meta['samples_mono']} mono samples @ {meta['sr_in']} Hz in {decode_ms} ms "
                f"(decoder={meta['decoder']}, format={meta['format']}/{meta['subtype']})",
                decode_ms=decode_ms, decoder=meta["decoder"], format=meta["format"],
                subtype=meta["subtype"], device_parity=meta["device_parity"],
            )
            if not meta["device_parity"]:
                # Worth saying out loud: the phone's file-scan decoder accepts
                # 16-bit PCM only, so it would have refused this file. The
                # numbers below are still real model numbers — they are just not
                # numbers the device could have produced from this same file.
                self.monitor.log(
                    "decode",
                    f"{meta['subtype']} is not 16-bit PCM — the phone's decodePcm16Wav would refuse "
                    "this file, so laptop-vs-device parity does not hold for it",
                    severity="warn", subtype=meta["subtype"], device_parity=False,
                )
            if meta["sr_in"] != SR:
                self.monitor.log(
                    "resample",
                    f"{meta['sr_in']} Hz -> {SR} Hz (linear; the phone's decode path resamples the same way)",
                    severity="warn", sr_in=meta["sr_in"], sr_out=SR,
                    samples_in=meta["samples_mono"], samples_out=meta["samples_16k"],
                )
            await self._prepare_backend_session()
            await self._run_windows(pcm)
        except asyncio.CancelledError:  # replaced by a newer run
            self.monitor.log("run_end", "run cancelled", severity="warn")
            raise
        except Exception as exc:  # a broken run must be visible, never silent
            self.monitor.log("error", f"run failed: {type(exc).__name__}: {exc}",
                             severity="error", exception=type(exc).__name__)
        finally:
            self._running = False
            if self._task is own_task:
                # This run truly ended (finished, stopped, or errored) rather
                # than being superseded by a newer `start()` — safe to blank
                # the gauge back to "no audio" without racing a fresher run's
                # first score. If it *was* superseded, that other task already
                # owns `local`/`window`/`diff`; don't clobber it.
                self.publish_state(local={"scored": False, "reason": "no_active_run"},
                                    window={}, diff=None)
            else:
                self.publish_state()

    async def _prepare_backend_session(self) -> None:
        """Reset the backend's per-call state and ask which artifact it is running."""
        if not self.config.backend_enabled:
            self.monitor.log("backend_call", "backend comparison disabled — local model only",
                             severity="warn", enabled=False)
            return
        ok = await self.backend.reset(self._session_id)
        self.monitor.log(
            "backend_call",
            f"reset backend session {self._session_id!r}: {'ok' if ok else 'unreachable (continuing)'}",
            severity="info" if ok else "warn",
            endpoint=f"POST /v1/reset/{self._session_id}", mode=self.backend.mode, reset=ok,
        )
        info = await self.backend.model_info()
        self.monitor.log(
            "backend_call",
            "backend model-info: "
            + (f"loaded={info.get('loaded')} sha={info.get('sha256_short')}" if info
               else "unavailable — backend may be down"),
            severity="info" if info else "warn",
            endpoint="GET /v1/model-info",
            backend_model_loaded=(info or {}).get("loaded"),
            backend_model_sha=(info or {}).get("sha256_short"),
        )

    async def _run_windows(self, pcm: np.ndarray) -> None:
        windows = list(iter_windows(pcm, self.config.window_samples, self.config.hop_samples,
                                    pad_short=True))
        if not windows:
            self.monitor.log(
                "run_end",
                f"audio too short: {len(pcm)} samples is under one "
                f"{self.config.window_samples / SR:.1f} s window — nothing to score",
                severity="error",
            )
            return
        self.monitor.log(
            "run_start",
            f"{len(windows)} scoring windows queued (hop {self.config.hop_samples / SR:.2f} s, "
            f"speed {self.config.speed})",
            windows=len(windows), loop=self.config.loop, speed=self.config.speed,
            hop_samples=self.config.hop_samples, threshold=self.config.threshold,
            window_ms=int(self.config.window_samples / SR * 1000),
        )
        if len(pcm) < self.config.window_samples:
            padded_ms = int((self.config.window_samples - len(pcm)) / SR * 1000)
            self.monitor.log(
                "window",
                f"audio is only {len(pcm) / SR:.2f} s — front-padded with {padded_ms} ms of digital "
                "silence to reach one window, exactly as the phone's file-scan path does. This model "
                "reads silence as synthetic, so treat the score below with suspicion",
                severity="warn", padded_ms=padded_ms, audio_s=round(len(pcm) / SR, 3),
            )
        # Loops until `loop` is turned off or a stop is requested. Each pass gets
        # a fresh AlertStateMachine and backend session, so EMA and
        # consecutive-window state never leak from one pass into the next.
        pass_index = 0
        while True:
            for index, start_sample, t_start, chunk in windows:
                await self._wait_turn()
                if self._stop_requested:
                    self.monitor.log("run_end", f"stopped by operator at window {index}",
                                     severity="warn", window_index=index)
                    return
                self._window_index = index
                await self._process_window(index, start_sample, t_start, chunk)
                if self.config.speed > 0:
                    await asyncio.sleep((self.config.hop_samples / SR) / self.config.speed)
            pass_index += 1
            self.monitor.log(
                "run_end",
                f"pass {pass_index} complete — all {len(windows)} windows processed",
                counts=dict(self._tally),
            )
            if not self.config.loop or self._stop_requested:
                return
            self.monitor.log("run_start", "looping: fresh session, restarting at window 0",
                             severity="warn", loop=True, pass_index=pass_index + 1)
            self._sm = AlertStateMachine(threshold=self.config.threshold)
            self._tally = self._empty_tally()
            self._session_id = f"laptop-{int(time.time())}"
            await self._prepare_backend_session()

    async def _wait_turn(self) -> None:
        """Block while paused, honouring single-step requests."""
        while not self._running and not self._stop_requested:
            if self._pending_steps > 0:
                self._pending_steps -= 1
                self.monitor.log("control", "single-step window starting",
                                 pending_steps=self._pending_steps)
                self.publish_state()
                return
            self._wake.clear()
            await self._wake.wait()

    # ------------------------------------------------ one window, every stage
    async def _process_window(self, index: int, start_sample: int, t_start: float,
                              chunk: np.ndarray) -> None:
        """Publish one event per stage for a single window, in pipeline order."""
        window_started = time.perf_counter()
        t_end = t_start + len(chunk) / SR
        level = rms(chunk)
        peak = float(np.max(np.abs(chunk))) if len(chunk) else 0.0
        silent = level < SILENCE_RMS

        self.monitor.log(
            "window",
            f"window {index} t={t_start:.2f}-{t_end:.2f}s rms={dbfs(level)} dBFS "
            f"peak={dbfs(peak)} dBFS" + ("  [SILENT — silence gate will skip scoring]" if silent else ""),
            severity="warn" if silent else "info",
            t=t_start, window_index=index, start_sample=start_sample,
            end_sample=start_sample + len(chunk), t_end=round(t_end, 3),
            rms=round(level, 6), rms_dbfs=dbfs(level), peak_dbfs=dbfs(peak),
            silent=silent, silence_gate=SILENCE_RMS,
        )

        if silent:
            # Mirrors audio_service.dart: a window below the RMS gate is not
            # scored at all. Publishing a score here would invent a number for
            # audio nobody spoke into — and this model reads silence as highly
            # synthetic, so the fake number would be a loud one.
            self._tally["windows_silent"] += 1
            self.monitor.log("window", f"window {index}: below the silence gate — not scored",
                             severity="warn", t=t_start, window_index=index)
            self.publish_state(local={"scored": False, "reason": "below_silence_gate"})
            return

        if not self.model.loaded:
            self.monitor.log(
                "error",
                f"deployed model is not loaded ({self.model.load_error}) — cannot score window {index}",
                severity="error", t=t_start, window_index=index,
            )
            return

        features_started = time.perf_counter()
        sequence, scalars = await asyncio.to_thread(extract_window_features, chunk)
        features_ms = round((time.perf_counter() - features_started) * 1000, 3)
        self.monitor.log(
            "features",
            f"{len(sequence)} LFCC frames x {len(sequence[0]) if sequence else 0} coeffs, "
            f"{len(scalars)} scalars in {features_ms} ms",
            t=t_start, window_index=index, n_frames=len(sequence),
            n_lfcc=len(sequence[0]) if sequence else 0,
            scalars=[round(s, 6) for s in scalars], features_ms=features_ms,
            lfcc_mean=round(float(np.mean(sequence)), 6) if sequence else None,
            lfcc_std=round(float(np.std(sequence)), 6) if sequence else None,
        )

        result = self.model.score_sequence(sequence, scalars)
        self.monitor.log("infer", f"onnx session.run in {result.infer_ms} ms",
                         t=t_start, window_index=index, infer_ms=result.infer_ms,
                         real_fake_logits=[round(v, 6) for v in result.real_fake_logits],
                         attack_type_logits=[round(v, 6) for v in result.attack_type_logits])
        self.monitor.log("softmax",
                         f"p_real={result.p_real:.4f} p_fake={result.p_fake:.4f} "
                         f"attack={result.attack_type} ({result.attack_confidence:.3f})",
                         t=t_start, window_index=index, p_real=round(result.p_real, 6),
                         p_fake=round(result.p_fake, 6), attack_type=result.attack_type,
                         attack_confidence=round(result.attack_confidence, 6))

        raw = round(result.p_fake, 6)
        self.monitor.log("score", f"raw window score = {raw:.4f}",
                         t=t_start, window_index=index, raw_score=raw)

        previous_ema = self._sm.ema
        state = self._sm.update(raw)
        ema = round(self._sm.ema, 6)
        previous_label = "seed" if previous_ema is None else f"{previous_ema:.4f}"
        self.monitor.log("ema", f"ema {previous_label} -> {ema:.4f} (alpha={self._sm.alpha})",
                         t=t_start, window_index=index, alpha=self._sm.alpha,
                         prev_ema=None if previous_ema is None else round(previous_ema, 6), ema=ema)
        self.monitor.log("decision",
                         f"threshold={self._sm.threshold:.2f} "
                         f"consecutive_high={self._sm.consecutive_high} -> {state}",
                         t=t_start, window_index=index, threshold=self._sm.threshold,
                         consecutive_high=self._sm.consecutive_high, state=state,
                         consecutive_required=self._sm.consecutive_required)
        verdict = verdict_for(ema)
        self.monitor.log("verdict", f"{verdict} (bands {BAND_LOW:.2f}/{BAND_HIGH:.2f})",
                         t=t_start, window_index=index, verdict=verdict, ema=ema,
                         band_low=BAND_LOW, band_high=BAND_HIGH)
        if state == "alert":
            self._tally["alerts"] += 1
            self.monitor.log("alert",
                             f"ALERT at t={t_start:.2f}s — ema={ema:.4f} for "
                             f"{self._sm.consecutive_high} consecutive windows",
                             severity="warn", t=t_start, window_index=index, ema=ema)
        self._tally["windows_scored"] += 1

        diff: Optional[DiffResult] = None
        if self.config.backend_enabled:
            diff = await self._backend_diff(sequence, scalars, raw, ema, t_start, index)

        e2e_ms = round((time.perf_counter() - window_started) * 1000, 3)
        self._e2e_ms.append(e2e_ms)
        self.monitor.log("window_done",
                         f"window {index} done in {e2e_ms} ms "
                         f"(features {features_ms} ms + infer {result.infer_ms} ms)",
                         t=t_start, window_index=index, e2e_ms=e2e_ms,
                         features_ms=features_ms, infer_ms=result.infer_ms)
        self.publish_state(
            window={"index": index, "t_start_s": round(t_start, 3), "t_end_s": round(t_end, 3),
                    "rms_dbfs": dbfs(level), "peak_dbfs": dbfs(peak), "silent": False,
                    "e2e_ms": e2e_ms},
            local={"scored": True, "raw_score": raw, "ema": ema, "state": state,
                   "verdict": verdict, "threshold": self._sm.threshold,
                   "consecutive_high": self._sm.consecutive_high,
                   "attack_type": result.attack_type,
                   "attack_confidence": round(result.attack_confidence, 4),
                   "infer_ms": result.infer_ms, "features_ms": features_ms},
            diff=None if diff is None else diff.to_dict(),
        )

    async def _backend_diff(self, sequence: list[list[float]], scalars: list[float],
                            raw: float, ema: float, t_start: float,
                            index: int) -> DiffResult:
        """Score the same window through the integration API and diff the two."""
        started = time.perf_counter()
        try:
            payload = await self.backend.analyze(lfcc_sequence=sequence, scalars=scalars,
                                                 session_id=self._session_id)
        except BackendError as exc:
            self._backend_error_count += 1
            self._tally["backend_errors"] += 1
            # Throttled on purpose: a dead backend would otherwise emit an error
            # per window and bury the very log the reader is trying to use.
            if self._backend_error_count == 1 or self._backend_error_count % 10 == 0:
                throttle_note = ("" if self._backend_error_count == 1 else
                                 f" [throttled; {self._backend_error_count} failures so far]")
                self.monitor.log(
                    "backend_call",
                    f"backend call failed ({exc.kind}) — local model unaffected{throttle_note}",
                    severity="error" if self._backend_error_count == 1 else "warn",
                    t=t_start, window_index=index, kind=exc.kind, error=str(exc),
                    failure_count=self._backend_error_count,
                    endpoint="POST /v1/analyze-chunk", mode=self.backend.mode,
                )
            diff = compute_diff(local_raw=raw, local_ema=ema, backend=None, error=str(exc),
                                backend_mode=self.backend.mode)
            self._update_diff_tally(diff)
            self.monitor.log("diff", f"no comparison possible: {diff.flags[0]}", severity="warn",
                             t=t_start, window_index=index, **diff.to_dict())
            return diff

        http_ms = round((time.perf_counter() - started) * 1000, 3)
        self.monitor.log(
            "backend_call",
            f"POST /v1/analyze-chunk -> {payload.get('verdict')} "
            f"raw={payload.get('rawScore')} ema={payload.get('riskScore')} "
            f"(http {http_ms} ms, backend {payload.get('latencyMs')} ms, "
            f"scorer={payload.get('modelBackend')})",
            t=t_start, window_index=index, http_ms=http_ms, mode=self.backend.mode,
            endpoint="POST /v1/analyze-chunk",
            backend_scorer=payload.get("modelBackend"),
            backend_raw=payload.get("rawScore"), backend_ema=payload.get("riskScore"),
            backend_verdict=payload.get("verdict"), backend_state=payload.get("state"),
            backend_latency_ms=payload.get("latencyMs"),
            backend_model_version=payload.get("modelVersion"),
        )
        diff = compute_diff(local_raw=raw, local_ema=ema, backend=payload,
                            backend_mode=self.backend.mode)
        self._update_diff_tally(diff)
        backend_raw_label = "n/a" if diff.backend_raw is None else f"{diff.backend_raw:.4f}"
        self.monitor.log(
            "diff",
            f"local {diff.local_raw:.4f} vs backend {backend_raw_label} -> {' '.join(diff.flags)}",
            severity="warn" if diff.band_mismatch else "info",
            t=t_start, window_index=index, **diff.to_dict(),
        )
        return diff

    def _update_diff_tally(self, diff: DiffResult) -> None:
        self._tally["compared"] += 1
        if diff.agree:
            self._tally["agree"] += 1
        elif diff.agree is False:
            self._tally["diverge"] += 1
        if diff.band_mismatch:
            self._tally["band_mismatch"] += 1
        if diff.delta_raw is not None:
            self._tally["sum_abs_delta_raw"] += abs(diff.delta_raw)