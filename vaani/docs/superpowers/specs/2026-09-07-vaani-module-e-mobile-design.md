# VAANI Module E (Mobile / Android) — Design

**Owner:** TBD · **Status:** Approved — ready for implementation plan · **Created:** 2026-09-07

## 0. Why this exists

The master plan's Non-Negotiable #6 ("the demo never depends on the venue") and
deliverable #6 ("edge deployment... laptop CPU and Raspberry Pi 5") already
establish VAANI as an edge-first system. This module extends that story to a
phone: a real Android app a judge can hold, import a voice-note or recorded
call into, or hold up to a live speakerphone call, and watch the same
detection pipeline the desktop Streamlit demo runs — gauge, spectrogram, risk
curve, occlusion highlights, bank HOLD→OTP→release, hash-chained audit log —
run natively on-device.

## 1. Hard constraint: what Android will not let us do

Modern Android does not allow a third-party app to tap another app's live
call or VoIP audio (WhatsApp, Teams, the native dialer) without root or an
Accessibility-service screen-capture hack that is unstable across OEMs/OS
versions and Play-Store-hostile. This is an OS privacy boundary, not a gap in
our engineering. Module E therefore supports exactly two capture paths:

1. **File import** — the user imports an audio file (a WhatsApp-exported
   voice note, a call-recorder app's output, any WAV/M4A/OGG file) via the
   Android Storage Access Framework picker.
2. **Live mic capture** — the app records its own microphone in real time
   (e.g., held next to a speakerphone call). This captures whatever audio
   reaches the phone's mic, not the remote app's internal audio stream.

No other capture mode is in scope for Module E.

## 2. Phasing

No trained model or ONNX export exists anywhere in this repo yet — Module B's
current backend (`app/engine_mock.py`) is a rule-based mock, not real
inference. Module E is split so that phone-side product work does not block
on Module B, but the app can never present a mock score as if it were a real
detection:

- **Phase 1 — buildable now.** App shell, both capture paths, the
  windowing/mel pipeline, full UI parity (gauge, spectrogram, risk curve,
  occlusion highlights, bank modal, audit log), wired to a `StubScorer`. Every
  stub-derived score is visibly labeled in the UI ("simulated — real model
  pending") so it can never be mistaken for a real result on stage.
- **Phase 2 — gated on Module B.** The moment Module B exports a trained
  TinyCNN to ONNX, swap `StubScorer` → `OnnxScorer` at one call site. No other
  code changes. This mirrors the exact `MockBackend` → real-backend swap point
  `app/server.py` already establishes for the desktop demo.

Phase 2 has no independent timeline here — it starts whenever Module B's
ONNX export lands, however soon or late that is.

## 3. Architecture

```
[file import]  OR  [phone mic]
        → normalize to mono 16 kHz PCM
        → Kotlin DSP bridge (platform channel):
             2 s windows, 0.5 s hop, mel-spectrogram
             (mirrors telechannel's mel step so the two platforms
              extract features identically)
        → Scorer (StubScorer Phase 1 / OnnxScorer Phase 2)
        → per-window probability
        → EMA smoothing, α=0.7 (ported line-for-line from
          engine_mock.py's decision logic — same constant, same
          reasoning: at α=0.3 the EMA's memory defeats the
          2-window rule)
        → call-level state machine: ALERT only if smoothed score
          stays above threshold for 2+ consecutive windows
        → UI: gauge · spectrogram · risk curve · occlusion
          highlights
        → bank sim: transfer → HOLD → simulated OTP → release
        → hash-chained audit log (Dart port of audit_log.py's
          SHA-256 chain — same tamper-evidence guarantee)
```

**Stack:** Flutter (Dart) app shell + UI; a small native Kotlin module
(`android/app/.../DspBridge.kt`) for FFT/mel-spectrogram extraction, exposed
to Dart via a MethodChannel; `onnxruntime` Flutter plugin for Phase 2
inference. Android-only — no iOS target, no server dependency, no network
call anywhere in the pipeline (matches §1.4's on-device, no-cloud
non-negotiable).

## 4. Components

- **`mobile/lib/capture/`** — file import (SAF picker + audio decode) and
  live mic capture (`AudioRecord` via platform channel), both normalized to
  mono 16 kHz PCM.
- **`mobile/android/.../DspBridge.kt`** — native windowing + mel-spectrogram
  extraction; the one piece of real DSP math, kept out of Dart deliberately.
- **`mobile/lib/scoring/`** — a `Scorer` interface with `StubScorer` (Phase 1)
  and `OnnxScorer` (Phase 2, loads the exported model as a bundled asset).
  Both feed the same `DecisionEngine` (EMA + 2-window state machine) so
  thresholds and timing are identical to the desktop demo regardless of which
  scorer is active.
- **`mobile/lib/ui/`** — gauge, spectrogram view, risk-curve chart,
  occlusion-highlight overlay, bank HOLD→OTP→release modal, audit-log viewer.
- **`mobile/lib/audit/`** — `AuditEntry`/`AuditLog`, a Dart port of
  `audit_log.py`'s SHA-256 hash chain (`verify_chain()` detects tampering,
  returns the first broken sequence number — same contract as the Python
  version).

## 5. Data flow & error handling

Phase 2 data flow: capture → PCM → DSP bridge → mel tensor → `OnnxScorer` →
probability → EMA → state machine → UI + bank modal + audit entry. Phase 1 is
identical except `StubScorer` replaces the ONNX call.

- Unsupported or corrupt import files reject with a clear on-screen message —
  never a silent zero score.
- Mic permission denial shows a blocking explainer, not a crash.
- If `OnnxScorer` fails to load its model asset (missing file, incompatible
  opset), the app falls back to `StubScorer` with a loud, persistent warning
  banner rather than crashing — the same "fail gracefully, never claim more
  than is true" non-negotiable (§1.3 of the master plan) applied on mobile.

## 6. Testing

- Dart unit tests for `DecisionEngine` (alert timing, EMA behavior) and
  `AuditLog` (chain integrity, tamper detection) — mirrors the coverage
  `tests/demo/` already has for the Python equivalents.
- A unit test asserting `StubScorer` output is never presented without the
  "simulated" UI label, and that swapping to `OnnxScorer` requires touching
  exactly one call site (the swap-point contract, tested the way
  `test_server.py::test_call_N_control_never_alerts` tests a similar contract
  today).
- `integration_test` widget tests: file import → gauge/curve movement;
  HOLD → OTP → release flow. No network, no cloud, deterministic fixtures —
  same discipline as `tests/demo/test_offline.py`.

## 7. Master plan integration

Module E is **fully woven into `00_MASTER_PLAN.md`**, not a side document:

- **§8 Timeline:** add a Module E row to the Week 1 table (app shell, both
  capture paths, full UI, `StubScorer`, clearly labeled) and to the Weeks
  2–12 table (Phase 2 `OnnxScorer` swap, timed to whenever Module B's ONNX
  export lands — not pinned to a specific week, since it depends on Module B
  finishing, not on a calendar date).
- **§9 Demo script:** add a beat showing the phone running the identical
  pipeline alongside the laptop — reinforces "the exact live pipeline,
  disclosed" across two devices instead of one.
- **§13 Deliverables Checklist:** add "Mobile: Android app (Flutter),
  Phase 1 stub-scored + Phase 2 real-model swap point" as its own checklist
  line.

These are the exact edits the implementation plan should make to
`00_MASTER_PLAN.md`; this spec does not modify that generated file directly.

## 8. Out of scope

- iOS. Not requested; Android-only per this spec.
- Any live capture of another app's call/VoIP audio (WhatsApp, Teams, native
  dialer) — blocked by the OS, not attempted even as a stretch goal.
- Training or shipping the mobile model itself — Module E consumes whatever
  ONNX artifact Module B produces; it does not train or quantize models.
- A server/network fallback path — deliberately not built, to keep the phone
  demo's "no cloud, no Wi-Fi dependency" story true end-to-end.
