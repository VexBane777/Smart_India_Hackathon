"""
VoiceGuard Integration Layer — FastAPI demo backend
Satisfies SIH requirement 5: REST API + SDK for banking/enterprise/telecom embedding.
On-device inference is primary; this shows the productization path.
Payload is LFCC features only — never raw audio (privacy story).

Two scoring paths, both labelled in every response:

* `modelBackend="onnx"` — the caller sends the full per-frame window
  (`lfcc_sequence` 184x60 + `scalars` 6) and this API scores it with the
  deployed artifact `assets/models/voice_detector.onnx` via `model_backend.py`.
  This is the same file, and the same math, the phone runs.
* `modelBackend="heuristic"` — the original path: a 60-d mean-pooled `lfcc`
  plus 3 prosody values through `_heuristic`. Kept because the SDK contract
  documented in README.md and used by `lib/services/api_service.dart` sends
  that shape; nothing that worked before this change stops working.

Which one produced a given number is never inferred from the request shape
alone: the response says so explicitly, `GET /v1/model-info` reports whether
the model is loaded (and why not, if it isn't), and every call is written to
the shared diagnostic log — see `monitor.py` and `GET /monitor.json`.

Run:  pip install -r requirements.txt
      uvicorn main:app --reload --port 8001
Docs: http://localhost:8001/docs          (port 8001, matching README.md and
                                           ApiService.baseUrl — the "8000" that
                                           used to be in this docstring was stale)
"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time, math

# Reuse the canonical EMA + 2-consecutive-window alert policy (master plan
# §6) instead of a second, independently-tuned decision rule: same
# ALERT_THRESHOLD/EMA_ALPHA/CONSECUTIVE_REQUIRED as the Streamlit demo's
# engine_mock.AlertStateMachine and the Flutter app's RiskScoreProvider. Both
# scoring paths below share this one rule, so `rawScore` differences between
# them are scorer differences — never decision-policy differences.
_VAANI_DIR = Path(__file__).resolve().parents[2] / "vaani"
if str(_VAANI_DIR) not in sys.path:
    sys.path.insert(0, str(_VAANI_DIR))
from app.engine_mock import AlertStateMachine  # noqa: E402
from signaling import signaling_router  # noqa: E402
from model_backend import ModelUnavailable, OnnxModelBackend  # noqa: E402
from monitor import MonitorLog, attach_monitor_routes  # noqa: E402

_monitor = MonitorLog(title="voiceguard-backend", echo_stdout=True)
_model = OnnxModelBackend()

_monitor.set_state(
    service="voiceguard-api",
    model={"loaded": _model.loaded, "path": str(_model.model_path)},
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the deployed model at startup and record what actually happened.

    A failed load is not fatal and not hidden: the API keeps serving the
    heuristic path, and both `/v1/model-info` and the monitor log say
    `loaded=false` with the reason, so no caller can mistake a fallback number
    for a model number.
    """
    ok = _model.load()
    info = _model.identity()
    if ok:
        _monitor.log(
            "model_load",
            f"deployed ONNX loaded in {info['load_ms']} ms",
            sha256_short=info["sha256_short"],
            size_bytes=info.get("size_bytes"),
            mtime=info.get("mtime"),
            inputs=info["input_names"],
            outputs=info["output_names"],
            providers=info["providers"],
        )
    else:
        _monitor.log(
            "model_load",
            "ONNX model NOT loaded — serving the heuristic scorer",
            severity="error",
            reason=_model.load_error,
            path=str(_model.model_path),
        )
    _monitor.set_state(model={"loaded": _model.loaded, "path": str(_model.model_path),
                              "sha256_short": _model.sha256_short,
                              "load_error": _model.load_error})
    yield
    _monitor.log("shutdown", "backend stopping")


app = FastAPI(
    title="VoiceGuard Integration API",
    version="0.2.0",
    description="Embeddable voice-cloning risk scoring for banking apps, contact centers, and telecom platforms. Accepts LFCC/prosody features (not raw audio). Scores with the deployed ONNX detector when the full window is supplied, otherwise with the legacy heuristic.",
    lifespan=lifespan,
)

app.include_router(signaling_router)
attach_monitor_routes(app, _monitor, tag="Monitor")

API_KEY = "vg_demo_key"

# Per-session alert state (EMA + consecutive-window tracking) — keyed by
# the caller-supplied session_id so a stream of analyze-chunk calls for one
# call behaves like one continuous engine run, matching how the Streamlit
# demo's WebSocket stream drives a single AlertStateMachine per call.
#
# Callers aren't required to supply session_id (it defaults to "default"),
# and nothing calls /v1/reset automatically — so without expiry, a session's
# alert state (e.g. two consecutive high-EMA windows) would otherwise leak
# into the next, unrelated call that reuses the same id, and _SESSIONS would
# grow without bound as distinct ids accumulate. SESSION_TTL_SECONDS bounds
# both: a session idle longer than the TTL is treated as a new call.
SESSION_TTL_SECONDS = 120
_SESSIONS: dict[str, AlertStateMachine] = {}
_LAST_SEEN: dict[str, float] = {}


def _evict_stale_sessions() -> None:
    now = time.monotonic()
    stale = [sid for sid, t in _LAST_SEEN.items() if now - t > SESSION_TTL_SECONDS]
    for sid in stale:
        _SESSIONS.pop(sid, None)
        _LAST_SEEN.pop(sid, None)


def _session(session_id: str) -> AlertStateMachine:
    _evict_stale_sessions()
    _LAST_SEEN[session_id] = time.monotonic()
    sm = _SESSIONS.get(session_id)
    if sm is None:
        sm = AlertStateMachine()
        _SESSIONS[session_id] = sm
    return sm

class ProsodyIn(BaseModel):
    pauseRatio: float = 0.2
    energyVar: float = 0.01
    zcrVar: float = 0.01

class AnalyzeIn(BaseModel):
    """One analysis window. Two mutually-exclusive shapes, both accepted:

    * **model path** — `lfcc_sequence` (184x60) + `scalars` (6): the complete
      per-frame window the deployed ONNX detector expects, scored for real.
    * **legacy / SDK path** — `lfcc` (60 mean-pooled) + `prosody`: unchanged
      since this API was written, and what `lib/services/api_service.dart`
      sends. Scored by `_heuristic`.

    `lfcc` is optional-with-default purely so the model path can omit it; the
    handler rejects a request that supplies neither, so the field is still
    effectively required.
    """

    lfcc: List[float] = Field(
        default_factory=list,
        description="60 mean-pooled LFCC coefficients (legacy/SDK path)",
    )
    lfcc_sequence: Optional[List[List[float]]] = Field(
        None,
        description="Per-frame LFCC window, 184 frames x 60 coeffs (model path). Requires `scalars`.",
    )
    scalars: Optional[List[float]] = Field(
        None,
        description="6 scalars [pauseRatio, energyVar, zcrVar, jitter_local, shimmer_local, hnr_db] (model path)",
    )
    prosody: Optional[ProsodyIn] = None
    metadata: Optional[dict] = None
    session_id: str = Field("default", description="Groups chunks from one call so EMA/consecutive-window state carries across requests")

class AnalyzeOut(BaseModel):
    riskScore: float
    verdict: str
    confidence: float
    latencyMs: float
    modelVersion: str = "unknown"
    # --- additive diagnostics (2026-09-15) — every field below is new, and a
    # caller written against the original 5-field response is unaffected. The
    # point is that a reader can tell *which scorer* produced a number without
    # having to ask, which is what makes the two paths comparable.
    rawScore: Optional[float] = Field(None, description="Per-window score before EMA smoothing")
    state: Optional[str] = Field(None, description="Alert state machine output: normal | warn | alert")
    modelBackend: str = Field("heuristic", description="Which scorer ran: onnx | heuristic")
    attackType: Optional[str] = Field(None, description="tts | vc — only when the model ran")
    attackConfidence: Optional[float] = None
    inferMs: Optional[float] = Field(None, description="ONNX session.run time (model path only)")

class AlertIn(BaseModel):
    callerId: Optional[str] = None
    riskScore: float
    verdict: str
    ts: Optional[str] = None
    channel: Optional[str] = "in-app"

def _heuristic(lfcc: List[float], prosody: Optional[ProsodyIn]) -> float:
    if not lfcc: return 0.15
    high = lfcc[int(len(lfcc)*0.6):]
    mean_h = sum(high)/len(high) if high else 0
    var_h = sum((v-mean_h)**2 for v in high)/len(high) if high else 0
    pause = prosody.pauseRatio if prosody else 0.2
    raw = max(0, min(1, var_h*0.9 + pause*0.25 + abs(lfcc[0])*0.05))
    return 0.08 + raw*0.78

def _verdict(state: str) -> str:
    # Maps AlertStateMachine's normal/warn/alert to the API's public verdict
    # vocabulary. "alert" requires 2+ consecutive high-EMA windows, so a
    # single noisy chunk can no longer flip this to AI_DETECTED.
    if state == "alert": return "AI_DETECTED"
    if state == "warn": return "SUSPICIOUS"
    return "VERIFIED_HUMAN"

def _check_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key. Use 'vg_demo_key' for demo.")

@app.get("/health")
def health(): return {"status": "ok", "service": "voiceguard-api", "version": "0.1.0"}

@app.post("/v1/analyze-chunk", response_model=AnalyzeOut, tags=["Scoring"])
def analyze_chunk(body: AnalyzeIn, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    t0 = time.perf_counter()

    attack_type: Optional[str] = None
    attack_conf: Optional[float] = None
    infer_ms: Optional[float] = None

    if body.lfcc_sequence is not None or body.scalars is not None:
        # ---- deployed-model path -------------------------------------------
        if body.lfcc_sequence is None or body.scalars is None:
            raise HTTPException(status_code=422, detail="the model path needs both `lfcc_sequence` and `scalars`")
        if not _model.loaded:
            # Deliberately NOT falling back to the heuristic here: the caller
            # asked for a model score, so quietly returning a different
            # scorer's number under a `modelBackend=onnx` label would be the
            # exact dishonesty this field exists to prevent.
            raise HTTPException(
                status_code=503,
                detail=f"ONNX model unavailable ({_model.load_error or 'not loaded'}). "
                       "Omit `lfcc_sequence` to score with the heuristic path, or see /v1/model-info.",
            )
        try:
            result = _model.score_sequence(body.lfcc_sequence, body.scalars)
        except ValueError as exc:  # wrong-shaped window -> clear 422, not an ort crash
            raise HTTPException(status_code=422, detail=str(exc))
        raw = max(0.0, min(1.0, result.p_fake))
        backend_name = "onnx"
        model_version = f"voice_detector.onnx@{_model.sha256_short}"
        attack_type, attack_conf, infer_ms = result.attack_type, round(result.attack_confidence, 4), result.infer_ms
    else:
        # ---- legacy / SDK heuristic path (unchanged behaviour) --------------
        if not body.lfcc:
            raise HTTPException(
                status_code=422,
                detail="supply `lfcc` (60 mean-pooled values), or the model path's `lfcc_sequence`+`scalars`",
            )
        raw = max(0.0, min(1.0, _heuristic(body.lfcc, body.prosody)))
        backend_name = "heuristic"
        model_version = "heuristic-v0.2"

    sm = _session(body.session_id)
    state = sm.update(raw)
    score = sm.ema
    latency = (time.perf_counter() - t0) * 1000
    verdict = _verdict(state)
    # confidence is distance from threshold
    conf = abs(score - 0.5) * 2
    out = AnalyzeOut(
        riskScore=round(score, 4),
        verdict=verdict,
        confidence=round(conf, 3),
        latencyMs=round(latency, 2),
        modelVersion=model_version,
        rawScore=round(raw, 4),
        state=state,
        modelBackend=backend_name,
        attackType=attack_type,
        attackConfidence=attack_conf,
        inferMs=infer_ms,
    )
    # One monitor line per request, so an agent watching this API sees the same
    # numbers the caller got (and prosody only, never the 11k-float sequence:
    # it is the diagnostic payload, not a transcript of what crossed the wire).
    _monitor.log(
        "analyze_chunk",
        f"{backend_name} window scored: raw={raw:.4f} ema={score:.4f} -> {verdict}",
        severity="error" if backend_name == "onnx" and not _model.loaded else "info",
        session_id=body.session_id,
        backend=backend_name,
        raw_score=out.rawScore,
        ema=out.riskScore,
        state=state,
        verdict=verdict,
        consecutive_high=sm.consecutive_high,
        threshold=sm.threshold,
        latency_ms=out.latencyMs,
        infer_ms=infer_ms,
        n_lfcc=len(body.lfcc) if body.lfcc else None,
        n_frames=len(body.lfcc_sequence) if body.lfcc_sequence else None,
        scalars=body.scalars,
        attack_type=attack_type,
        attack_confidence=attack_conf,
        caller_id=(body.metadata or {}).get("callerId"),
    )
    _monitor.set_state(
        scoring={
            "backend": backend_name,
            "model_version": model_version,
            "session_id": body.session_id,
            "raw_score": out.rawScore,
            "ema": out.riskScore,
            "state": state,
            "verdict": verdict,
            "confidence": out.confidence,
            "threshold": sm.threshold,
            "consecutive_high": sm.consecutive_high,
            "latency_ms": out.latencyMs,
            "infer_ms": infer_ms,
            "attack_type": attack_type,
            "attack_confidence": attack_conf,
        },
        sessions=len(_SESSIONS),
    )
    return out

@app.post("/v1/reset/{session_id}", tags=["Scoring"])
def reset_session(session_id: str, x_api_key: Optional[str] = Header(None)):
    """Clear EMA/consecutive-window state for a session — call at the start of each new call."""
    _check_key(x_api_key)
    _SESSIONS.pop(session_id, None)
    _LAST_SEEN.pop(session_id, None)
    # Logged, not just done: a start-of-call reset silently discarding an EMA
    # is exactly the kind of invisible state change that makes a live score
    # impossible to explain after the fact.
    _monitor.log("session_reset", f"reset EMA/alert state for session {session_id!r}",
                 session_id=session_id)
    return {"reset": session_id}

class ControlIn(BaseModel):
    action: str = Field(..., description="reload-model | clear-monitor")

@app.post("/v1/control", tags=["Monitor"])
def control(body: ControlIn, x_api_key: Optional[str] = Header(None)):
    """The only mutating endpoint here — explicit, and it logs itself.

    `reload-model` re-reads `assets/models/voice_detector.onnx` from disk, so a
    diagnostics session can drop in a new artifact and watch the scores change
    without restarting the API.
    """
    _check_key(x_api_key)
    if body.action == "reload-model":
        ok = _model.reload()
        info = _model.identity()
        _monitor.log("model_reload", f"model reload -> loaded={ok}",
                     severity="info" if ok else "error",
                     sha256_short=info["sha256_short"], load_ms=info["load_ms"],
                     reason=info["load_error"])
        _monitor.set_state(model={"loaded": _model.loaded, "path": str(_model.model_path),
                                  "sha256_short": _model.sha256_short,
                                  "load_error": _model.load_error})
        return {"action": body.action, "loaded": ok, "model": info}
    if body.action == "clear-monitor":
        _monitor.clear()
        _monitor.log("monitor_clear", "diagnostic log cleared via /v1/control")
        return {"action": body.action, "cleared": True}
    raise HTTPException(status_code=400, detail=f"unknown action {body.action!r}; use reload-model or clear-monitor")

@app.get("/v1/model-info", tags=["Meta"])
def model_info():
    """Which artifact is scoring, and if it isn't loaded, why not."""
    return _model.identity()

@app.post("/v1/alert", tags=["Alerting"])
def alert(body: AlertIn, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    # In production this fans out to SMS/email/SIEM. Here we log + ack.
    print(f"[ALERT] caller={body.callerId} verdict={body.verdict} risk={body.riskScore} ts={body.ts}")
    _monitor.log("alert", f"alert accepted for {body.callerId or 'unknown caller'}: {body.verdict}",
                 caller_id=body.callerId, verdict=body.verdict, risk_score=body.riskScore,
                 channel=body.channel, ts=body.ts)
    return {"accepted": True, "forwardedTo": ["in-app", "webhook"], "note": "SMS/email leg mocked for demo — see README."}

@app.get("/", tags=["Meta"])
def root():
    return {
        "message": "VoiceGuard API — see /docs for OpenAPI",
        "health": "/health",
        "analyze": "POST /v1/analyze-chunk",
        "alert": "POST /v1/alert",
        "model": "GET /v1/model-info",
        "monitor": {
            "snapshot": "GET /monitor.json",
            "text": "GET /monitor.txt",
            "tail": "GET /logs.ndjson?since=<seq>",
            "stream": "GET /events",
            "websocket": "/ws/monitor",
            "control": "POST /v1/control",
        },
    }
