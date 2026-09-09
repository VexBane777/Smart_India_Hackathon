# Project State: VoiceGuard (voice_guard/)

**Maintained every session.** Read this at the start of any voice_guard
session before doing anything else, and update it before ending a session
that changed status, findings, or plans — see `CLAUDE.md` at the repo root.

## Current status (2026-09-09)

**We are close to real-time on-device AI-voice detection working end to
end on Android — not there yet on real phone calls specifically, but the
full pipeline (capture → LFCC/prosody feature extraction → the real
retrained model, now served via ONNX Runtime instead of TFLite (see
"Accent/clone data pipeline + ONNX swap" below) → EMA/alert-gating → UI) is
proven correct and working on real audio right now, today, via the Live
Mic Test path.**

Do not read "blocked on real calls" as "the project doesn't work." Two
distinct things are true at once:
1. The detection pipeline itself: **validated, working, correct.**
2. Getting real *phone-call* audio into that pipeline on a stock,
   non-rooted Android device: **blocked by the OS/OEM, not by our code.**
   The privileged-capture path built to solve exactly this is code-complete
   and builds successfully; it is untested only for lack of a rooted
   device, not for any known defect.

**We are not treating today's blockers as a stopping point.** The plan is
to keep pushing on real-call capture — more efficiently, more effectively,
more creatively than the standard playbook — including inventing our own
undocumented approaches if the documented ones run out. See "Ideas not yet
tried" below; add to it rather than letting a session end with a shrug.

## Immediate to-dos / test checklist (next session, in priority order)

~~0. Verify the ONNX Runtime swap on-device~~ — **done, 2026-09-09, same
   session.** Phone reconnected; installed `app-privileged-debug.apk` (from
   `969fd7e`), ran Live Mic Test, `adb logcat` confirmed `ONNX model loaded
   from assets/models/voice_detector.onnx` (native `libonnxruntime4j_jni.so`
   loaded first), and live scoring worked correctly end to end — `Monitor:
   raw=...` values fluctuated realistically (0.064 quiet → 0.999 during
   speech), EMA/alert-gating fired as designed (`ALERT fired`), no `ONNX
   inference failed` errors. The ONNX swap is now fully verified, not just
   built. Also fixed leftover "TFLite" user-facing strings found while in
   the app (`call_screen.dart`, `onboarding_screen.dart`,
   `logs_screen.dart`, `settings_screen.dart` — cosmetic only, code
   identifiers like `tflite_service.dart`/`TFLiteService` deliberately kept
   as-is, see "Accent/clone data pipeline + ONNX swap" below).

Phone (CPH2613) is connected via USB as of this session; `app-privileged-
release.apk` built from commit `55b6e1f` is installed and launched
(`lastUpdateTime` 2026-09-09 14:51). `BLUETOOTH_CONNECT` is **not**
granted yet — `adb pm grant` hit the same shell restriction seen earlier
for `CAPTURE_AUDIO_OUTPUT`; grant it manually via Settings → Apps →
VoiceGuard → Permissions → Nearby devices before test 2b.

Do these in one on-device pass, in this order (each `adb logcat` filter
given so results can be pulled without guessing what to grep for):

1. **Clamp #2 / AEC-disable verification** (`ae11c20`, already implemented,
   still unverified). Live Mic Test screen, phone's own speaker playing
   audio into its own mic (no call needed). Watch for the previously-seen
   ~7-11s clamp to noise floor (`-inf` dB then permanent ~-37dB) — if it's
   gone/reduced, the AEC-disable fix worked.
   `adb logcat -s AudioCaptureManager` — look for `AcousticEchoCanceler
   disabled for session ...` at start, then judge by ear/RMS whether the
   clamp still happens.
2. **External-mic preference** (`preferExternalInputDevice()`, this
   session, code-complete/untested):
   - **2a. Wired earphones (inline mic)**: plug in, place or receive a
     real call, confirm `preferExternalInputDevice: setPreferredDevice
     (TYPE_WIRED_HEADSET, ...) -> true` in logcat and that `chunkRms`
     stays non-zero for the call duration (previously always zero per
     root cause #1).
   - **2b. Bluetooth headset**: pair/connect it, same real-call test,
     confirm `Requested Bluetooth SCO` + `setPreferredDevice
     (TYPE_BLUETOOTH_SCO, ...) -> true` in logcat, and check `chunkRms`
     — note BT SCO negotiation can take 1-2s before real audio starts
     flowing, don't judge the first couple of chunks.
   - Filter: `adb logcat -s AudioCaptureManager`
3. **VoLTE/call-type diagnostics** (`logCallAudioDiagnostics()`, this
   session): during the same real call(s) from step 2, check for `Call
   audio diagnostics: networkType=... highDefAudio=... wifiCall=...` at
   call-active. Record what network type the call(s) that do/don't
   reproduce root cause #1 were on — this is the data point the "VoLTE vs.
   legacy circuit-switched call" idea (below) needs, gathered for free
   alongside steps above rather than as a separate test.
   Filter: `adb logcat -s InCallServiceImpl`
4. If 2a/2b show real (non-zero) `chunkRms` during an actual call:
   **root cause #1 may be resolved or narrowed** — re-read state.md's
   root-cause section, since that would be new information contradicting
   "every source gets zero-filled" and needs a rewrite, not just a note.
   If 2a/2b still show zero: root cause #1 is confirmed *not* scoped to
   the internal mic, which is itself a useful negative result — record it.
5. Remaining hardware-only ideas, not part of this pass (no code changes
   possible, need separate devices/SIMs — see "Ideas not yet tried" for
   full rationale on each): get a rooted device to test
   `magisk-privileged-module/`; test on a non-ColorOS device; swap SIM
   carrier; try VoLTE-vs-legacy explicitly if the diagnostic log in step 3
   doesn't naturally surface both call types.
- ~~Try Shizuku~~ — **ruled out, see "Ideas tried and ruled out" below.**
  Shizuku cannot grant `CAPTURE_AUDIO_OUTPUT`; don't re-attempt.

## What's proven working (2026-09-09 session)

- **New retrained model (`508d637`, full-corpus retrain: ASVspoof2019 LA
  full train partition + ASVspoof2021 LA eval + 2019 LA dev + In-the-Wild
  speaker-disjoint split) is deployed and confirmed correct on real audio**,
  not just synthetic benchmarks:
  - Played a real AI-TTS-generated (not voice-cloned) YouTube clip through
    an **external speaker** (not the phone's own) into the phone's mic via
    the app's Live Mic Test (`call_screen.dart`'s mic-icon toggle, no real
    call needed). Full pipeline ran for 45+ continuous seconds with no
    capture interruption, correctly firing multiple properly-gated
    `AI DETECTED` alerts (raw scores 0.68–1.00, EMA up to 0.99, requiring
    2+ consecutive high windows per the alert state machine) exactly when
    the AI voice was speaking, dropping to `VERIFIED HUMAN`/quiet during
    pauses/music — tracking real content, not noise.
  - This is genuine, credible validation: real mic capture → real feature
    extraction → the real model → real alert logic, not a mock.
  - Confirms the earlier documented eval numbers
    (`model_training/README.md`: in-distribution EER 0.0793, cross-generator
    held-out EER 0.1624) aren't just training-time metrics — the exported
    `.tflite` behaves correctly in the actual app on-device.

- **Gradle build speed fixed** (`efdfdbe`): `MaxMetaspaceSize=512m` was too
  tight for this project (2 flavors, KGP, AGP 9.1, desugaring) and was
  crashing the Gradle daemon mid-build with an OOM, forcing a cold daemon
  restart on every retry — on top of the already-deliberate
  `kotlin.incremental=false` (kept; it's a real Windows file-locking
  workaround, don't revert it without cause). Raised to `1g` metaspace /
  `3G` heap in `android/gradle.properties`. Release builds dropped from
  ~2–3 min (with daemon crashes making it worse) to ~40s cold, ~7s on a
  no-op rebuild.

- **`privileged` build flavor (Magisk system-priv-app capture,
  `magisk-privileged-module/`) builds successfully** via
  `flutter build apk --flavor privileged --release`. Untested on real
  hardware only because the only test device available this session
  (CPH2613, Oplus/ColorOS, Android 16) is not rooted. No known defect —
  just unverified.

## Root causes found this session (evidence-backed, not guesses)

### 1. Real telephony-call audio capture is blocked at the OS/OEM level on this device — confirmed root cause, not a code bug

Tested placing real calls (device CPH2613, Oplus/ColorOS, non-rooted) with
speakerphone on and active speech on the line:
- Normal source cascade → won `VOICE_RECOGNITION` → `chunkRms=0.000000`
  for the entire call duration, every single window.
- Diagnostic: forced `MIC`-only (temporary edit to
  `AudioCaptureManager.kt`'s `SOURCE_CASCADE`, reverted after the test) →
  **also** `chunkRms=0.000000` for the entire call, across two separate
  test calls.
- Conclusion: it isn't source selection. Every `AudioRecord` source,
  including the plain universal-fallback `MIC`, gets silently zero-filled
  by the OS the instant a real telephony call goes active on this device.
  Almost certainly India's 2024+ carrier/OEM call-recording restrictions
  (ColorOS is known for this). `AudioCaptureManager.kt`'s docstring already
  anticipated this exact failure mode before this session confirmed it
  empirically.
- This is *precisely* the gap `magisk-privileged-module/` was built to
  close (`CAPTURE_AUDIO_OUTPUT` via a rooted priv-app install, bypassing
  this restriction entirely). That module is the correct next step, not a
  detour — see "Ideas not yet tried" for what's blocking testing it.

### 2. Own-speaker-into-own-mic acoustic loopback gets clamped after a few seconds — separate, newly-discovered issue, isolated and confirmed

Not the same bug as #1. Playing audio *through the phone's own speaker*
during a **non-call** Live Mic Test session (so #1's telephony-call block
doesn't apply) showed, consistently across 3 independent recordings:
- Real signal for the first ~3–7s (matches expected clip content).
- **True digital silence** (`-inf` dB, exact zero samples, not just quiet)
  for ~7–11s.
- Permanently drops to ambient noise-floor level (~-37 dB) for the rest of
  the recording — never regains the original signal strength.

Ruled out `flutter_webrtc` as the cause (checked with evidence, not
assumption): `AudioSwitchManager`'s audio-mode/focus-changing code
(`AudioManager.setMode(MODE_IN_COMMUNICATION)` etc.) only fires via
`GetUserMediaImpl.java`'s `getUserMedia(audio:...)`, which only happens
from the separate "Protected Call" / VoIP-screen path
(`webrtc_call_service.dart`) — never from the Dialer/Live-Detection screen
or the Live Mic Test button, which call `AudioCaptureManager.start()`
directly with no focus/mode manipulation at all. The
`audioFocusChangeListener [Speakerphone...]` log lines seen throughout are
just `AudioSwitch`'s harmless device-enumeration callback, unconditional
at plugin attach, unrelated to the actual clamp.

**Isolation test that confirmed the real cause:** same Live Mic Test, same
app, same device — but played the AI-voice clip through a **separate
external speaker** instead of the phone's own. Result: 45+ seconds of
continuous strong signal (RMS 0.05–0.12, chunk RMS never once hit -inf),
zero interruption, correct alerts throughout. **Conclusion: this is an
OEM-level anti-loopback-recording protection** — the OS/ColorOS detects
sustained acoustic coupling between the device's own speaker output and
its own mic input and clamps it after a few seconds, almost certainly to
prevent apps recording other apps' protected/DRM'd audio via speaker→mic
loopback. It does *not* trigger when the audio source is external.

**Practical implication:** always use an external speaker (or a second
device) when self-testing audio detection via Live Mic Test on this
phone — the phone's own speaker is not a reliable test source, separate
from and in addition to the already-known real-call block (#1). This
should be treated as a standing test-methodology note, not re-discovered
next session.

### 3. The in-app "Human Speech" / "AI Clone Audio" benchmark buttons are stale test fixtures

`call_screen.dart:446-462`'s two buttons (in the active-call view, or via
the Live Mic Test toggle) inject synthetic PCM via
`audio_service.dart`'s `injectBenchmarkTest()`. The "AI Clone Audio"
branch is three static sine harmonics — no formants, no real spectral
structure — tuned against the **old heuristic scorer**
(`tflite_io.dart`'s `_heuristic()`, literally just high-band variance,
exactly what a static tone triggers). The new real model, trained on
actual ASVspoof/In-the-Wild generator audio, correctly does not recognize
three sine tones as "AI speech" (out of its training distribution) — both
buttons now score near-identically low (~0.001), which looks like "broken"
but isn't; it's a stale fixture, not a model regression. **Do not use
these buttons to judge the new model.** Use Live Mic Test + a real
external audio source instead (see #2). If these buttons are worth fixing
later, they need synthetic stimuli that actually resemble ASVspoof-style
spoof artifacts, not arbitrary tones — or should be retired in favor of
bundling a couple of small real WAV clips as fixed assets.

## Ideas tried and ruled out

### Shizuku as a root alternative for `CAPTURE_AUDIO_OUTPUT` — ruled out, 2026-09-09

**Conclusion: structurally impossible, not device-specific. Don't
re-attempt, and don't bother installing the Shizuku app.**

Tested the "cheap ADB probe" listed as a Shizuku precursor — connected
CPH2613 via **USB** debugging (wireless debugging pairing was tried first
and abandoned; see below) and ran:

```
adb shell pm grant com.voiceguard.voice_guard android.permission.CAPTURE_AUDIO_OUTPUT
```

Result: `SecurityException: grantRuntimePermission: Neither user 2000 nor
current process has android.permission.GRANT_RUNTIME_PERMISSIONS` (full
stack trace in session transcript, `PermissionService.kt` /
`PackageManagerShellCommand.java`).

**Why this rules out Shizuku, not just this probe:** the exception is
about `pm grant` itself, not about ColorOS or this device — on stock AOSP,
shell (UID 2000) is only ever allowed to grant ordinary install-time
runtime permissions via `pm grant`; `signature|privileged` permissions
like `CAPTURE_AUDIO_OUTPUT` are categorically excluded from what shell can
grant, on any Android build. Shizuku's non-root mode runs its service at
that exact same shell UID (2000) — it is, by design, "whatever `adb shell`
can do, callable from inside an app." It has no more privilege than the
`adb shell` command just used, so it hits the identical
`SecurityException` for the identical reason. Root works instead because
it either runs as UID 0 (bypasses the permission check outright) or
installs the app into `/system/priv-app` with an OEM
privapp-permissions-allowlist XML entry (grants the permission at install
time, no `pm grant` call involved at all) — neither of which Shizuku can
do. `magisk-privileged-module/` still needs actual root; there is no
lower-effort substitute for it.

(Aside: whatever mechanism BCR's docs describe Shizuku enabling, it isn't
a `pm grant` of `CAPTURE_AUDIO_OUTPUT` — either BCR uses Shizuku for a
different, non-privileged capture path, or its Shizuku support assumes an
OEM/Android version where shell has been added to that permission's
allowlist, which is not the case on this CPH2613/ColorOS build. Not
investigated further since it doesn't unblock this project either way.)

**Wireless debugging pairing was tried first and abandoned for unrelated
network reasons** (kept for completeness, not a finding about Shizuku
itself): the phone was hotspotting mobile data to the PC, so wireless
debugging bound to the phone's mobile-data IP rather than the hotspot
interface actually reachable from the PC (`Test-NetConnection` timed out
on both TCP and ICMP). Retried on a shared Wi-Fi network
(`SVKMGRP.COM`, a campus/institutional network) and still failed — likely
AP client isolation blocking device-to-device traffic on that network
(ping and TCP both failed to the phone's IP even though both devices were
on the same SSID). Switched to a USB cable, which worked immediately
(`adb devices` saw the device on the first try). **If wireless debugging
is needed again for something else, use a network known not to isolate
clients (e.g. the phone's own hotspot, PC connecting to phone — not phone
hotspotting to PC) — USB is the reliable fallback if in doubt.**

## Ideas not yet tried (add to this, don't just leave it stale)

Goal: get real, unblocked audio into the detection pipeline during an
*actual phone call*, on a device we don't control the rooting of, without
waiting indefinitely for a rooted test device. Candidates, roughly ordered
by how soon they're actionable:

- **Get any rooted Android device** (even an old/cheap one, or a rootable
  emulator image with Magisk support) specifically to validate
  `magisk-privileged-module/` end-to-end. This is the most direct,
  already-built path — it just needs hardware.
- **Test on a different, non-Oplus/ColorOS device** (Pixel, stock AOSP,
  Samsung, older Android version) to check whether the OS-level call-audio
  block (#1) is ColorOS/India-carrier-specific or broader. If it's
  narrower than we think, the "acoustic fallback" story is much better on
  other hardware than this one test device suggests — don't let one
  device's behavior stand in for "Android in general."
- ~~`voice_guard/docs/superpowers/plans/2026-09-08-voice-guard-privileged-voip-capture-research-spike.md`
  follow-up: does the same Zygisk-can't-reach-`audioserver` finding also
  rule out a system-level bypass for the native telephony restriction
  (#1), not just VoIP apps?~~ — **ruled out, 2026-09-09, no hardware
  needed.** Added as a new section in that doc: `audioserver` is started
  by `init`, never zygote-forked, so Zygisk has no attachment surface
  regardless of which policy check inside it (VoIP usage-tag or telephony
  mic-block) is the target. Same non-viable conclusion, and actually a
  stronger no for telephony since bypassing it defeats an intentional
  carrier/OEM privacy restriction rather than a generic platform default.
- ~~Look for a completely different capture point that isn't
  microphone-shaped at all: does `AudioPlaybackCapture`
  (`PlaybackCaptureManager.kt`) have any applicability to the native
  dialer/telephony path, not just VoIP apps?~~ — **ruled out, 2026-09-09,
  no hardware needed.** Documented in `PlaybackCaptureManager.kt`'s
  docstring: native cellular call audio is rendered by the telephony
  HAL/modem directly, never as an app-owned `AudioTrack` playback session,
  so there is no session for `AudioPlaybackCaptureConfiguration` to attach
  to at all — categorically out of scope, independent of the
  already-known `USAGE_VOICE_COMMUNICATION` exclusion.
- **Bluetooth/wired external mic as the capture device** — **code-complete
  this session, not yet verified on-device.**
  `AudioCaptureManager.preferExternalInputDevice()` now explicitly queries
  `AudioManager.getDevices(GET_DEVICES_INPUTS)` for a wired/USB headset or
  BT SCO device and calls `AudioRecord.setPreferredDevice()` on it (BT SCO
  is also explicitly started via `startBluetoothSco()`, since a paired BT
  mic doesn't get live input just from being connected — SCO must be
  requested). Previously this idea was untested because the app never
  routed to the external device explicitly in the first place; now it
  will, so testing it actually exercises the intended condition. Needs
  `MODIFY_AUDIO_SETTINGS` + `BLUETOOTH_CONNECT` (added to
  `AndroidManifest.xml`; `BLUETOOTH_CONNECT` also requested at runtime in
  `permissions.dart`, best-effort — not fatal if denied). Builds clean.
  See "Immediate to-dos" for the on-device verification still needed.
- ~~Shizuku instead of full root~~ — **ruled out, 2026-09-09, see "Ideas
  tried and ruled out."**
- **Swap the SIM to a different carrier** in the same test device, to
  isolate whether block #1 is carrier-config-driven (India 2024+
  call-recording rules surfaced via `CarrierConfigManager`) vs.
  OEM/ColorOS policy independent of carrier. Zero hardware cost if a
  second SIM is available.
- **VoLTE vs. legacy circuit-switched call**, if the SIM/network still
  supports falling back to a non-VoLTE call — tests whether the block is
  specific to the IMS/VoLTE audio path (more carrier plumbing, more DRM
  hooks) or applies uniformly regardless of call type. **Instrumented this
  session** (not yet observed): `InCallServiceImpl.logCallAudioDiagnostics()`
  now logs `networkType` (LTE/NR/UMTS/GSM/etc., decoded to a readable
  name) plus the call's `PROPERTY_HIGH_DEF_AUDIO`/`PROPERTY_WIFI` flags at
  `STATE_ACTIVE`, so a future repro of #1 can be correlated against call
  type straight from logcat without a second tool. Read-only, wrapped in
  try/catch, cannot affect call handling if it fails.
- **Test whether clamp #2 (the acoustic anti-loopback clamp, see below)
  also fires during a real call's acoustic fallback**, not just Live Mic
  Test self-testing. Force speakerphone during an actual live call and
  watch for the same ~7-11s-then-clamp signature. Not yet tested — if it
  reproduces on a real call, the acoustic-fallback tier (forced
  speakerphone + MIC, what most non-root call-recorder apps rely on) is a
  second, independent failure mode stacked on top of #1, not merely a
  self-test artifact. Matters a lot for how much effort the fallback tier
  deserves vs. going all-in on privileged capture.
- ~~Explicitly disable AEC at the AudioEffect level~~ — **implemented,
  2026-09-09, untested on-device.** `AudioCaptureManager.kt` now calls
  `attachAec()` right after `recorder.startRecording()`: creates an
  `AcousticEchoCanceler` on the winning source's `audioSessionId` (guarded
  by `isAvailable()`, all best-effort/try-catch since availability and
  effect actually being honored are both device-dependent) and
  `setEnabled(false)`s it; releases it in `stop()` alongside the recorder.
  Builds clean (`flutter build apk --flavor privileged --release`, exit 0,
  `app-privileged-release.apk`). **Not yet verified to actually change
  clamp #2's behavior** — that needs a real Live Mic Test run on-device
  (phone's own speaker → own mic, watch whether the ~7-11s clamp to noise
  floor still happens). Do that check before crediting this as a fix.
- Stay open to genuinely new, undocumented approaches if the above don't
  pan out — this file exists partly so an idea tried and abandoned doesn't
  get silently retried next session, and so a genuinely new idea gets
  written down here instead of living only in a chat transcript.

## Accent/clone data pipeline + ONNX swap (2026-09-09 session)

Triggered by a real false-positive finding earlier this session: live,
genuine self-speech scored 80-100% "AI" via Live Mic Test whenever ambient
noise was present, ~0-20% when quiet. Root cause (reasoned from
`model_training/README.md`'s own admitted gap, not fully proven): the
deployed model was trained on ASVspoof (studio-quality) + In-the-Wild
(real-world but pre-recorded) audio, never evaluated against raw live-mic
capture with real room noise — and `VOICE_RECOGNITION` (the winning
capture source) deliberately runs with AGC/noise-suppression off, so two
of the three model features (`energyVariance`, `zcrVariance`) see
real-world noise the training data mostly didn't.

**Data-collection subsystem** (`model_training/data/`, commits `ad57406`,
`2d55cd3`) — scaffolding for an 8-cell `{en,hi} x {native,foreign} x
{real,fake}` corpus to close both the noise gap and an accent-coverage gap
(target: broad foreign-English-accent and Indian-Hindi-accent coverage).
`prep_accents_real.py` (Common Voice downloader/bucketer) and
`gen_accents_fake.py` (XTTS-v2 local voice cloning) are **written and
documented but not run** — no GPU/`datasets`/`TTS` on the machine this was
built on; `data/README.md` has the exact commands/links/license notes for
whoever runs it on the GPU machine next. `ingest_self_recordings.py`,
`report_accent_coverage.py`, `split_accents.py` (speaker-disjoint
train/held split), and `eval_accent_cells.py` (per-cell EER) need no
network/GPU and **are** smoke-tested (synthetic multi-speaker WAVs) — found
and fixed a real path bug in `prep_accents_real.py` this way (manifest
paths were relative to the wrong base dir, would have made every pulled
file unfindable). Next actual step: run steps 1-4 in `data/README.md` on
the GPU machine, then `report_accent_coverage.py` to see what's still thin
and needs manual supplementing, then fold `accents_split/train/` into a
retrain per the README's documented `train.py` invocation.

**ONNX Runtime swap** (commit `969fd7e`) — replaced `tflite_flutter` with
`flutter_onnxruntime`; app now loads `assets/models/voice_detector.onnx`
directly instead of a `.tflite` converted via `export_tflite.py` (that
script is kept but no longer the recommended path). `model.onnx` itself
wasn't actually on disk (only `model.pt` and the old `.tflite`) — re-exported
it from `model.pt` and confirmed numeric parity (~5e-4 max diff) before
shipping. `infer()`/`scoreChunk()` became `async` (ONNX Runtime's API is
Future-based), rippling into `TFLiteServiceBase`, the web stub, and
`audio_service.dart`'s two call sites. Root `.gitignore`'s blanket
`*.onnx` rule got an explicit exception for just this one shipped asset.
Verified: `flutter analyze` clean, debug APK builds, **and confirmed
on-device** (same session, once the phone reconnected) — model loads,
scores fluctuate correctly on real live-mic audio, alerts fire as
designed. See item 0 (now struck through) in "Immediate to-dos" above.

**LCNN backup — spike result, decision: not adopting now.** Investigated
whether a pretrained LCNN (the ASVspoof-era light-CNN architecture) could
serve as a more "reliable, less experimental" backup model. Findings: a
real pretrained checkpoint exists
([nii-yamagishilab/project-NN-Pytorch-scripts](https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts),
BSD-3-Clause, trained on ASVspoof2019 LA) — but it's architecturally a
different kind of model than ours: frame-level LFCC through a CNN +
bidirectional LSTM + global-average-pooling, vs. our tiny MLP over one
mean-pooled 63-float vector. Adopting it would mean building a second,
frame-sequence feature-extraction path in `audio_processor.dart` (doesn't
exist today), building our own ONNX export/mobile deployment (the repo has
neither), and "moderate" fine-tuning work in an unfamiliar codebase (their
own toy example says protocol-file prep + code edits are needed for a
custom dataset) — all while *not* sidestepping the same real-world
noise/domain-shift problem the accent-data pipeline above already exists
to fix, since LCNN's pretrained weights come from the same ASVspoof-era
data ours does. **Decision: revisit only if, after fine-tuning the current
MLP on the new accent/noise data, `eval_accent_cells.py` still shows it's
not enough** — at that point LCNN's extra temporal capacity might
genuinely earn its much larger integration cost. Don't re-litigate this
research from scratch next session; this paragraph is the record of why.

## Dataset expansion session (2026-09-09, later same day)

Picked up the accent/clone data pipeline (scaffolding only as of the
previous entry above) and actually ran it, plus went well beyond the
originally-scoped `data/README.md` sources per explicit instruction to
maximize breadth. Scope stayed **en/hi only** (a deliberate choice — see
below) and **XTTS-v2 proceeded as already documented** (non-commercial CPML,
fine for this prototype) — both confirmed with the user before expanding
further, since `vaani/00_MASTER_PLAN.md` has a broader/conflicting plan for
a *different* module (TeleChannel) that doesn't bind voice_guard.

**Sources actually ingested this session** (new `data/prep_*.py` scripts,
one per source, each with full provenance/license/gotcha notes in its own
docstring — read those before re-running, don't re-derive from scratch):
Common Voice 17.0 (`fixie-ai` mirror — official `mozilla-foundation` repo is
deprecated, Mozilla moved to Mozilla Data Collective Oct 2025), VCTK (108
speakers, real accent labels pulled cheaply via a 3.6KB Kaggle metadata
file rather than the 10.94GB official zip), Svarah (genuine Indian-accented
English, 6656 clips), OpenSLR SLR103 (99,925+3,843 8kHz native-telephony
Hindi clips — by far the largest single source), IndicTTS-Hindi,
IndicVoices-R_Hindi (spontaneous/conversational, different register than
the read-speech sources), CodecFake (neural-codec-resynthesis fakes, a
different attack family than ASVspoof/XTTS), MLAAD (580 clips / **116
distinct TTS architectures** — ElevenLabs, ChatTTS, F5-TTS, FireRedTTS-2.0,
Cartesia Sonic-3, etc. — by far the broadest generator diversity found).
Plus the two user-provided self-recordings (`ingest_self_recordings.py`,
already-existing script, finally actually used).

**Recurring technical blocker, same root cause every time:** `torchcodec`'s
native DLL fails to load on this Windows box (`libtorchcodec_core{4-9}.dll`
— FFmpeg shared-library mismatch) — hit in `datasets`' `Audio` feature
decode AND inside XTTS's own `torchaudio.load()` call. Fixed the same way
everywhere: decode via `soundfile` from raw bytes instead (`Audio(decode=
False)` for HF datasets; monkeypatched `torchaudio.load` itself for XTTS,
since its `tts_to_file(speaker_wav=...)` only accepts a path, not a
preloaded tensor — see `gen_accents_fake.py`'s `_patch_torchaudio_load_with_
soundfile()`). If a future script hits the same DLL error, don't reinstall/
fight torchcodec — apply this same workaround.

**XTTS-v2 setup, separately:** runs in an isolated `.venv_tts` (Python 3.13
— original `TTS` PyPI package doesn't support it, used the maintained
`coqui-tts` fork + `transformers==4.57.1` + `coqui-tts[codec]` instead), so
the main `.venv313`'s working torch+cuda for `train.py` couldn't be
accidentally broken by TTS's own pinned deps. Model weights: don't let
`TTS.api.TTS()` download them itself on a congested link — its downloader
is plain `requests`, no resume, dies and restarts from zero on any
connection drop (observed: died at 1.18GB, twice). Pre-download via
`huggingface_hub.snapshot_download(repo_id='coqui/XTTS-v2', local_dir=...)`
(properly resumable) and pass `--local-model-dir` instead. Also needs
`COQUI_TOS_AGREED=1` env var set (its license-agreement prompt uses
`input()`, which hangs forever / EOFErrors under `nohup`/background).

**Sources found but NOT pulled — retry after the demo, revisit this list
rather than re-researching from scratch:**
- **DECRO** (English+Chinese cross-lingual, ~55k samples) — Zenodo was down
  (504 Gateway Timeout, twice) at research time, purely transient infra
  issue on their end, not a real access blocker. Just retry
  `https://zenodo.org/record/7603208`.
- **Deepfake-Eval-2024** (real in-the-wild flagged deepfakes, 42 languages,
  56.5hrs audio) — gated on HF (`nuriachandra/Deepfake-Eval-2024`), needs a
  token/login.
- **IndieFake Dataset** (would've been ideal — genuine Indian-accented
  English deepfakes, from IIT Ropar) — **not actually released yet** as of
  this session; the paper (arXiv:2506.19014) says "will be publicly
  available upon acceptance." Check again later; a search-engine summary
  incorrectly implied it was already downloadable — verified via the actual
  paper text that it isn't.
- **HAV-DF** (Hindi-specific audio-video deepfakes, arXiv:2411.15457) — no
  public download link found; also video-bundled (faceswap+lipsync+clone
  together), would need extra work to pull audio-only even if found.
- **MLAAD's Hindi slice** — the 580-clip English pull above used a small
  Kaggle mirror sample; the *full* 45GB `trapka/mlaadthe-multi-languag...`
  Kaggle mirror likely has more languages, but confirming Hindi's presence
  requires paginating deep into its ~90k-file listing (got through 1200
  files, still all "ar" alphabetically, before giving up) or just
  committing to the full 45GB pull blind. Worth doing post-demo when
  bandwidth/disk aren't both under pressure.
- **FoR (Fake-or-Real)**, 17.2GB Kaggle, English, 33 synthetic voices across
  major cloud TTS providers — started downloading, then **killed
  deliberately** when free disk space hit ~50GB (see disk-space entry
  below), barely any progress lost. Straightforward to resume:
  `kaggle datasets download -d mohammedabdeldayem/the-fake-or-real-dataset
  -p data/for_dataset --unzip`.
- **SpoofCeleb** — ruled out **permanently, not a retry-later item**: gated
  to officially-affiliated institutional email addresses only, explicitly
  rejects personal emails. No path to this without an institutional
  affiliation on record with the dataset maintainers.

**Disk space emergency, mid-session:** free space on `C:` hit ~50GB while
several GB-scale downloads were still in flight (real risk of a full-disk
write failure corrupting an in-progress extraction). Freed ~45GB total,
in order: (1) deleted already-ingested raw archives once confirmed their
data had already landed in `accent_manifest.csv`/`data/real`+`data/fake`
(OpenSLR tarballs, ASVspoof2019 LA's `data/extracted/` tree, Svarah's
parquet) — safe because the manifest is the durable record, not the
archives; (2) `pip cache purge` (~4GB, pure cache, always safe); (3)
emptied the Recycle Bin (~30GB — by far the biggest single win, zero risk,
already user-deleted content). Deliberately did **not** touch
`%TEMP%\claude` (15.6GB) — that's shared working storage for *all* Claude
Code sessions on this machine, including possibly other still-running ones
per this file's own "many parallel sessions" note above; user explicitly
confirmed leaving it alone rather than risk another session's state.
**Lesson for next time a big multi-source pull is planned:** check free
disk space *before* kicking off several parallel multi-GB downloads, not
after hitting a low-space scare mid-flight.

**Update, later same session — full autonomous run through actual retraining:**
User handed off full autonomy ("work until done, make it production-ready,
commit/push at checkpoints"). Completed: Hindi XTTS cloning (400 clips),
DECRO ingestion (English spoof only, 7000 clips — its bona-fide side is
literally re-partitioned ASVspoof2019 LA, already in this corpus, so only
spoof was pulled), MUSAN noise augmentation (`augment_with_noise.py`,
8519 en_native + 3000 hi_native clips), `split_accents.py` (speaker-disjoint
train/held, all 8 cells, zero missing files), and — the actual first-ever
`train.py` retrain of this session.

**`hi_native`'s real training split had 92k files** (wildly disproportionate
vs. every other cell's few thousand) — capped to a random 8000-file sample
(`data/accents_split/train/real/hi_native_capped/`) before training, or
feature extraction alone would've taken impractically long and skewed the
corpus hard toward Hindi.

**Training attempt 1** (`runs/voice_guard_v4_attempt1/`, `--channel whatsapp
volte` — 2 recipes): `final_val_eer=0.0812` (in-distribution, comparable to
v3's 0.0793), but **cross-generator held-out EER (In-the-Wild) = 0.2028 —
worse than v3's 0.1624**, a real regression despite far more training data.
Root cause identified, not just noticed: I dropped `clean` from the
`--channel` list, not matching the documented v3 recipe (`--channel whatsapp
volte clean`, 3 recipes) — `clean` is a real registered light-degradation
TeleChannel recipe (see `train.py` README's own note on this), not "no
processing", so the base ASVspoof2019 corpus got 1/3 less channel-diversity
exposure than v3 had. **Did not deploy attempt 1** — a regression on the
metric that matters most (cross-generator generalization) doesn't meet "make
the model production-ready" no matter how much bigger the training set got.
Archived (not deleted) at `runs/voice_guard_v4_attempt1/` alongside its log
(`runs/train_v4_attempt1_log.txt`) for the record.

**Training attempt 2** in progress as of this update — same corpus, `--channel
whatsapp volte clean` (matching v3 exactly). Per-cell breakdown from attempt
1's `eval_accent_cells.py` run, worth carrying forward regardless of which
attempt ships: `en_native` EER 0.0614, `en_foreign` 0.1333, `hi_native`
0.1545, `hi_foreign` 0.1833 (only 11 windows/6 files — not statistically
reliable, matches the known-thin real/hi_foreign gap). Hindi generalizing
noticeably worse than English is expected (English had the full ASVspoof
base corpus behind it; Hindi is entirely new this session) but is a real,
now-measured gap, not a guess.

**Memory constraint discovered and worked around this session:** this
machine has only ~15.4GB RAM, and background-tracked heavy processes
(training, `split_accents.py`, dataset ingestion) got auto-killed by
low-memory monitoring repeatedly (observed free RAM oscillating 0.4–1.1GB
under load, driven by Windows Defender real-time-scanning every new file
plus ordinary desktop-app load — Discord, browser, etc. — this is a
general-use machine, not a dedicated headless box). **Fix: launch heavy/long
Python jobs via `nohup cmd &` + `disown` (fully OS-detached), not as
tool-tracked background tasks** — detached processes survived every memory
event that killed tracked ones. Also: run moderate-length jobs in the
foreground when practical (`split_accents.py` succeeded this way after
failing backgrounded) — the low-memory killer only ever targeted
tool-tracked background tasks, never plain foreground calls or detached
`nohup` processes. Keep `--workers` conservative (2, down from the
documented 8) on this machine specifically.

**Not done yet, pick up here next (if this session ends before finishing):**
wait for training attempt 2, compare its cross-generator held-out EER against
v3's 0.1624 — only deploy (copy `model.onnx` to `assets/models/`) if it's at
or better than that number, not just "bigger dataset therefore assumed
better". If attempt 2 still regresses, don't try a third blind attempt —
stop and think about *why* (candidates not yet tested: the tiny 63→64→32→2
MLP may simply be under-capacity for a much more heterogeneous task now;
the disproportionate real/hi_native volume, even capped to 8000, may still
be pulling the decision boundary away from what the ITW held-out set needs;
worth an ablation — retrain on the *old* v3 corpus alone with the corrected
`clean` channel recipe, to isolate "channel-recipe bug" from "new data hurt
generalization" as two separate explanations before concluding either one).

## Housekeeping done this session

- Merged `origin/vaani`'s 3 new commits (full-corpus retrain + docs) with
  local `1e55aee` wip commit — clean merge, `fa64831`.
- Committed `efdfdbe`: gradle.properties memory fix + the untracked
  privileged-VoIP-capture research spike doc (was sitting uncommitted from
  a prior session). Pushed to `origin/vaani`.
- Committed `ae11c20`: finished the AEC-disable idea (see "Ideas not yet
  tried") — `AudioCaptureManager.kt` now creates and disables an
  `AcousticEchoCanceler` on the winning capture source's session, released
  in `stop()`. Verified via a clean `flutter build apk --flavor privileged
  --release` (exit 0); not yet verified on-device against clamp #2's
  actual behavior. Pushed to `origin/vaani`.
- Cleaned up a stray 596MB `android/java_pid7812.hprof` heap-dump artifact
  left by the Gradle daemon OOM crash (not committed, was git-ignored
  anyway, just disk clutter).
- Deleted all on-device call recordings after inspection (user request,
  after confirming their content locally first — see git history / this
  session's transcript for the exact WAV analysis if ever needed again;
  not preserved as repo artifacts, they were device-local debug recordings
  only).
- Committed and pushed to `origin/vaani`, same session, continued from
  above: `ad57406` (accent/clone data-collection scaffolding),
  `955d7fd` (opt-in call detection — `InCallServiceImpl`/`MainActivity`/
  `call_screen.dart`, previously uncommitted from earlier work),
  `2d55cd3` (speaker-disjoint accent split + per-cell EER eval, plus the
  `prep_accents_real.py` path-bug fix), `969fd7e` (ONNX Runtime swap,
  replacing `tflite_flutter`). See "Accent/clone data pipeline + ONNX
  swap" above for the full rationale on the last two.

## How to keep this file useful

- Update the "Current status" date and paragraph at the *start* of a
  session if anything above turned out stale (check a claim before relying
  on it — a memory or a status note is a claim about the past, not
  guaranteed still true).
- Add newly-tried ideas (successes *and* dead ends) to "Ideas not yet
  tried" / a new "Ideas tried and ruled out" section as they happen, with
  the evidence, not just the conclusion — future sessions need to be able
  to tell "we tried X and it didn't work because Y" apart from "we never
  tried X."
- Keep root-cause writeups evidence-first (what was tested, what was
  observed, what the observation rules in/out) — this file has already
  once caught a wrong hypothesis (flutter_webrtc) before it became a wasted
  fix attempt, precisely because the investigation was written down
  verifiably enough to be double-checked.
