<!--
VAANI documentation suite v1.0 — generated 2026-09-03
Document: Doc 4 — Demo Asset Production Spec · Owner: Product Lead · Status: Build-ready
This file is GENERATED. Edit generate_vaani_docs.py and rerun instead.
-->
# Doc 4 — Demo Asset Production Spec

**Owner:** Product Lead · **Status:** Build-ready · **Suite:** v1.0 · **Generated:** 2026-09-03
**Depends on:** Doc 6 Form A signed before any recording; Doc 1 pipeline for channel application

---

## 4.1 What this builds
The pre-recorded 90 s call that Demo Slide 5 streams through the live pipeline — plus a
**matched negative control** (same script, real voice, must NOT alert) and a stress
variant. Together: a paired experiment judges can't argue with.

## 4.2 Call A — primary script ("The Vendor Payment")
Two-party call, ~90 s. Speaker 1: employee (REAL, team member 1). Speaker 2: "CFO"
(RVC clone of team member 2's voice; a third member performs the lines, RVC converts).
Fictional CFO name only.

| Time | Speaker | Lines (Hinglish — deliberate, matches the hi-en training label) |
|---|---|---|
| 0:00–0:22 | Employee (REAL) | Phone pickup. "Hello? Haan bolo… vendor payment? Aaj hi nikalna hai? Theek hai, main dekh leta hoon… have you told sir? Okay, put him on." |
| 0:22–0:55 | "CFO" (CLONE) | "Namaste. I'll be quick — I'm in a board meeting. We need the Q2 vendor settlement released today. Forty lakh. New remittance account — details on WhatsApp. Please don't loop in accounts, I've cleared it with audit. Confidential until the announcement." |
| 0:55–1:10 | Employee (REAL) | "Okay… but sir, normally these go through the portal? Should I at least mark finance?" |
| 1:10–1:30 | "CFO" (CLONE) | "No time for the portal. I'm authorizing verbally — you know my voice. Do it now, I'll confirm by email tonight." → typing sounds → transfer attempt → HOLD → OTP modal |

Scammer lines carry the classic social-engineering fingerprint: urgency · authority ·
secrecy · new account · out-of-band details. ~45 s continuous synthetic speech
guarantees the alert math (2 s windows, 0.5 s hop, 2-consecutive rule → alert ~3–4 s
after clone entry).

## 4.3 Voice sources
- Real speakers: record segments as isolated tracks (clean assembly, controlled levels).
- Clone donor (member 2): **10 min clean reference audio**, varied text, quiet room,
  consistent mic distance. Form A must cover cloning AND, separately opt-in, public
  posting of video containing the synthesized voice.
- Casting rule: the performing member should be within ~a musical fourth of the donor's
  pitch — RVC quality collapses across large pitch gaps.
- Train RVC on any 4050 (~30–60 min); generate 2–3 takes; pick the most convincing.

## 4.4 Recording session ($0, ~2 h)
Phone-earbud mics acceptable (often better than laptop mics — fan noise). Audacity,
48 kHz mono WAV. ~15 cm, 30–45° off-axis (kills plosives — the popping "p"s). No gain
riding, no compression — raw takes. Peaks ~−6 dBFS (digital loudness, 0 = clipping
ceiling), never touch ceiling. Smallest room, curtains drawn, fan/AC off; record 30 s
room tone. **3 takes per segment** — nobody's first take is their best.

## 4.5 Assembly & channel application
1. One track per speaker; splice at natural turn boundaries; gaps 100–300 ms; overlap
   ≤200 ms (VAD needs clean turns).
2. Master 48 kHz WAV → 16 kHz mono.
3. Run through TeleChannel recipes: `whatsapp` (primary), `volte` (backup), keep clean
   for debugging. Moderate parameter end (SNR 18–20 dB). **Pick by measured call-level
   alert behavior, not taste**; record the choice in the manifest.
4. Optional CC0 ringtone (Freesound) at 0:00.

## 4.6 The library
| Asset | Content | Must do |
|---|---|---|
| Call A | script above, whatsapp channel | alert ≤4 s after clone entry; no alert 0:00–0:22 |
| Call A′ | same script, volte | same behavior |
| Call B | different pair + accent (e.g., Tamil-English) | answers "does it work on another accent?" |
| Call N | **identical script, donor's REAL voice** | **no alert, full run** — the paired-experiment evidence |
| Call S | Call A, noisy recipe (~10 dB SNR) | alert still fires — demo robustness |

## 4.7 Hold-out rule (leakage trap)
Demo assets **never enter any training split** — manifest rows carry `split: demo`.
If the model trains on the demo file, the stage demo becomes a rehearsed trick; a sharp
judge asking "was this in your training set?" ends the pitch.

## 4.8 QA gates
Duration 85–92 s (A/A′/B) · peaks ≤−3 dBFS, zero samples at ±1.0 · VAD coverage 55–85% ·
≥20 s continuous synthetic speech · ≥15 s leading real speech · transcription spot-check
≥95% vs script (a member listens and verifies) · engine behavior: A/A′/B/S alert, N no
alert, all logged.

## 4.9 Timestamp verification (feeds Doc 3)
```
python -m vaani.engine --mode file --input assets/demo/callA_whatsapp.wav --record-cues
→ assets/demo/callA_whatsapp.cues.json  (alert onset, per-window scores)
```
Paste **measured** times into the Doc 3 cue sheet. Same discipline as every other number.

## 4.10 Done means
All five assets pass QA · cues.json for A/A′/B/S · consent forms (donor cloning + posting
opt-ins, all speakers) filed in the register (Doc 6) · raw takes archived · backup video
recorded from Call A.

---

*Part of the VAANI documentation suite — regenerate with `python generate_vaani_docs.py`. Placeholders marked [M] must be replaced by measured values before use in the pitch.*
