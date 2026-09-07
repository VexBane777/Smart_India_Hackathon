# VoiceGuard — Real-Time AI Voice Cloning Detector (SIH POC)

> Every feature maps to one of the 5 SIH capability areas — see table below.

| SIH Requirement | Where it lives in this repo |
|---|---|
| 1. Multi-layer voice authenticity (spectral + prosody + cross-session) | `lib/utils/audio_processor.dart` (LFCC 60 + prosody), `lib/services/tflite_service*` |
| 2. Real-time risk scoring + configurable thresholds | `lib/providers/risk_score_provider.dart` (EMA), Settings sensitivity slider |
| 3. Alerting & user interaction | `android/.../OverlayService.kt` + `lib/services/notification_service.dart` + recommended-action copy |
| 4. Privacy & compliance (DPDP Act) | In-memory PCM only, `backend` accepts features not raw audio, in-app notices |
| 5. Platform & integration layer | `backend/main.py` (`POST /v1/analyze-chunk`, `POST /v1/alert`, OpenAPI `/docs`) |

## Run

### Android POC (primary)
```bash
cd voice_guard
flutter pub get
flutter run            # needs Android device/emulator; set as Default Dialer
```
Grant Phone/Mic + overlay permission in Settings. Use **Live Call → Demo Call** to see the risk meter animate without a real call.

### Backend integration artifact (judge-facing)
```bash
cd voice_guard/backend
pip install -r requirements.txt   # fastapi uvicorn pydantic numpy
uvicorn main:app --reload --port 8001  # docs at http://localhost:8001/docs
# Android emulator reaches it at http://10.0.2.2:8001
```

### Web preview (for desktop browser verification — heuristic scorer only)
```bash
flutter build web
cd build/web && python -m http.server 8766
```

## Architecture
```
Flutter (Home/Call/Logs/Settings) ──MethodChannel/EventChannel──► Android (InCallService, AudioRecord 16kHz MIC, TFLite, Overlay)
                                                              └─► FastAPI (/v1/analyze-chunk, /v1/alert) — SDK path
```

## Known limitations (stated in demo)
- Android 10+ blocks `VOICE_CALL` source — POC uses `MIC` + speakerphone; production needs telecom-side media forking / VoIP integration.
- iOS out of scope (CallKit + VoIP future work).
- Model is heuristic until `assets/models/voice_detector.tflite` (MobileNetV3-Small INT8) is dropped in.

## SIH Demo Checklist
- [ ] Installs, becomes Default Dialer, InCallService fires
- [ ] LFCC+prosody <50ms/chunk, inference <20ms, E2E <500ms
- [ ] Risk meter green/yellow/red + overlay + local notification at threshold
- [ ] Real voice → green, ElevenLabs sample → red (English + Hindi)
- [ ] Logs show metadata only — no raw audio persisted
- [ ] Backend `/docs` loads, `backend/README.md` frames enterprise story
