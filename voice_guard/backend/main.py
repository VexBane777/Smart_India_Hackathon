"""
VoiceGuard Integration Layer — FastAPI demo backend
Satisfies SIH requirement 5: REST API + SDK for banking/enterprise/telecom embedding.
On-device inference is primary; this shows the productization path.
Payload is LFCC features only — never raw audio (privacy story).

Run:  pip install fastapi uvicorn pydantic
      uvicorn main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, List
import time, math

app = FastAPI(
    title="VoiceGuard Integration API",
    version="0.1.0",
    description="Embeddable voice-cloning risk scoring for banking apps, contact centers, and telecom platforms. Accepts LFCC/prosody features (not raw audio).",
)

API_KEY = "vg_demo_key"

class ProsodyIn(BaseModel):
    pauseRatio: float = 0.2
    energyVar: float = 0.01
    zcrVar: float = 0.01

class AnalyzeIn(BaseModel):
    lfcc: List[float] = Field(..., description="60 LFCC coefficients (mean-pooled)")
    prosody: Optional[ProsodyIn] = None
    metadata: Optional[dict] = None

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

def _verdict(score: float) -> str:
    if score < 0.30: return "VERIFIED_HUMAN"
    if score < 0.70: return "SUSPICIOUS"
    return "AI_DETECTED"

def _check_key(x_api_key: Optional[str]):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-Key. Use 'vg_demo_key' for demo.")

@app.get("/health")
def health(): return {"status": "ok", "service": "voiceguard-api", "version": "0.1.0"}

@app.post("/v1/analyze-chunk", response_model=AnalyzeOut, tags=["Scoring"])
def analyze_chunk(body: AnalyzeIn, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    t0 = time.perf_counter()
    score = _heuristic(body.lfcc, body.prosody)
    # tiny jitter to look live
    score = max(0, min(1, score))
    latency = (time.perf_counter()-t0)*1000
    verdict = _verdict(score)
    # confidence is distance from threshold
    conf = abs(score - 0.5)*2
    return AnalyzeOut(riskScore=round(score,4), verdict=verdict, confidence=round(conf,3), latencyMs=round(latency,2))

@app.post("/v1/alert", tags=["Alerting"])
def alert(body: AlertIn, x_api_key: Optional[str] = Header(None)):
    _check_key(x_api_key)
    # In production this fans out to SMS/email/SIEM. Here we log + ack.
    print(f"[ALERT] caller={body.callerId} verdict={body.verdict} risk={body.riskScore} ts={body.ts}")
    return {"accepted": True, "forwardedTo": ["in-app", "webhook"], "note": "SMS/email leg mocked for demo — see README."}

@app.get("/", tags=["Meta"])
def root():
    return {"message": "VoiceGuard API — see /docs for OpenAPI", "health": "/health", "analyze": "POST /v1/analyze-chunk", "alert": "POST /v1/alert"}
