"""
VoiceGuard Integration Layer — FastAPI demo backend
Satisfies SIH requirement 5: REST API + SDK for banking/enterprise/telecom embedding.
On-device inference is primary; this shows the productization path.
Payload is LFCC features only — never raw audio (privacy story).

Run:  pip install fastapi uvicorn pydantic
      uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time, math

# Reuse the canonical EMA + 2-consecutive-window alert policy (master plan
# §6) instead of a second, independently-tuned decision rule: same
# ALERT_THRESHOLD/EMA_ALPHA/CONSECUTIVE_REQUIRED as the Streamlit demo's
# engine_mock.AlertStateMachine, and the same one the Flutter app's
# RiskScoreProvider mirrors. Only the raw per-window score generator
# (`_heuristic` below) differs, since neither side has a real trained
# model yet.
_VAANI_DIR = Path(__file__).resolve().parents[2] / "vaani"
if str(_VAANI_DIR) not in sys.path:
    sys.path.insert(0, str(_VAANI_DIR))
from app.engine_mock import AlertStateMachine  # noqa: E402
from signaling import signaling_router

app = FastAPI(
    title="VoiceGuard Integration API",
    version="0.1.0",
    description="Embeddable voice-cloning risk scoring for banking apps, contact centers, and telecom platforms. Accepts LFCC/prosody features (not raw audio).",
)

app.include_router(signaling_router)

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
    lfcc: List[float] = Field(..., description="60 LFCC coefficients (mean-pooled)")
    prosody: Optional[ProsodyIn] = None
    metadata: Optional[dict] = None
    session_id: str = Field("default", description="Groups chunks from one call so EMA/consecutive-window state carries across requests")

class AnalyzeOut(BaseModel):
    riskScore: float
    verdict: str
    confidence: float
    latencyMs: float
    modelVersion: str = "mobilenetv3-small-int8-demo-v0.1"

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
    raw = max(0, min(1, _heuristic(body.lfcc, body.prosody)))
    sm = _session(body.session_id)
    state = sm.update(raw)
    score = sm.ema
    latency = (time.perf_counter()-t0)*1000
    verdict = _verdict(state)
    # confidence is distance from threshold
    conf = abs(score - 0.5)*2
    return AnalyzeOut(riskScore=round(score,4), verdict=verdict, confidence=round(conf,3), latencyMs=round(latency,2))

@app.post("/v1/reset/{session_id}", tags=["Scoring"])
def reset_session(session_id: str, x_api_key: Optional[str] = Header(None)):
    """Clear EMA/consecutive-window state for a session — call at the start of each new call."""
    _check_key(x_api_key)
    _SESSIONS.pop(session_id, None)
    _LAST_SEEN.pop(session_id, None)
    return {"reset": session_id}

@app.post("/v1/alert", tags=["Alerting"])
def alert(body: AlertIn, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    # In production this fans out to SMS/email/SIEM. Here we log + ack.
    print(f"[ALERT] caller={body.callerId} verdict={body.verdict} risk={body.riskScore} ts={body.ts}")
    return {"accepted": True, "forwardedTo": ["in-app", "webhook"], "note": "SMS/email leg mocked for demo — see README."}

@app.get("/", tags=["Meta"])
def root():
    return {"message": "VoiceGuard API — see /docs for OpenAPI", "health": "/health", "analyze": "POST /v1/analyze-chunk", "alert": "POST /v1/alert"}
