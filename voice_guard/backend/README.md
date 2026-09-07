# VoiceGuard Integration API — Banking / Enterprise / Telecom

> SIH Requirement 5: *Platform & integration layer — REST API + SDK embeddable in banking apps, contact centers, and telecom infrastructure.*

The Android POC runs inference **on-device** (TFLite) for latency. This backend is the **judge-facing integration artifact** showing how a bank or telco would embed VoiceGuard without needing real telecom access.

## Quick start

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8001
# docs → http://localhost:8001/docs
```

Android emulator reaches it at `http://10.0.2.2:8001` (see `ApiService.baseUrl`). Port 8001,
not 8000, so this can run alongside the Streamlit demo's FastAPI streamer
(`vaani/app/server.py`), which already owns 8000.

## Auth

`X-API-Key: vg_demo_key` (demo). Production would use per-tenant keys / mTLS.

## Endpoints

### `POST /v1/analyze-chunk`
LFCC + prosody features → risk score. **Never raw audio** (privacy story).

```json
{
  "lfcc": [60 floats],
  "prosody": { "pauseRatio": 0.21, "energyVar": 0.013, "zcrVar": 0.008 },
  "metadata": { "callerId": "+91...", "locale": "en-IN", "ts": "2026-09-07T..." },
  "session_id": "call-123"
}
```
→ `{ "riskScore": 0.83, "verdict": "AI_DETECTED", "confidence": 0.66, "latencyMs": 1.2 }`

`session_id` groups chunks from one call so EMA + consecutive-window state
carries across requests, same decision policy as `vaani/app/engine_mock.py`'s
`AlertStateMachine` (master plan §6) — `AI_DETECTED` only fires after 2+
consecutive high-risk windows, not a single noisy chunk. Call
`POST /v1/reset/{session_id}` at the start of each new call.

### `POST /v1/alert`
Webhook stub a downstream system subscribes to. In production fans out to SMS/email/SIEM.

```json
{ "callerId": "+91...", "riskScore": 0.83, "verdict": "AI_DETECTED", "ts": "..." }
```

### `GET /health` / `GET /`

## SDK sketch (what a bank app integrates)

```dart
final api = ApiService(baseUrl: 'https://voiceguard.yourbank.internal', apiKey: '...');
final res = await api.analyzeChunk(lfcc: lfcc, prosody: prosody, callerId: callerId);
if (res['verdict'] == 'AI_DETECTED') await api.sendAlert(callerId: callerId, riskScore: res['riskScore'], verdict: res['verdict']);
```

## Privacy

Same guarantees as the on-device path: only features + metadata cross the wire, raw PCM never leaves the device / never touches storage.
