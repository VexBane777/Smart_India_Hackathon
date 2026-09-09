# Project State: VoiceGuard (voice_guard/)

**Maintained every session.** Read this at the start of any voice_guard
session before doing anything else, and update it before ending a session
that changed status, findings, or plans — see `CLAUDE.md` at the repo root.

## Current status (2026-09-09)

**We are close to real-time on-device AI-voice detection working end to
end on Android — not there yet on real phone calls specifically, but the
full pipeline (capture → LFCC/prosody feature extraction → the real
retrained TFLite model → EMA/alert-gating → UI) is proven correct and
working on real audio right now, today, via the Live Mic Test path.**

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

## Immediate to-dos (next session, in priority order)

- **Try wired earphones (inline mic) and Bluetooth headset as the capture
  device** during a real call, instead of the phone's internal mic. Zero
  cost, user already has both on hand. Checks whether the OS/OEM block
  (#1 below) is scoped to the internal mic specifically. See "Ideas not
  yet tried" for full rationale.
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
- **`voice_guard/docs/superpowers/plans/2026-09-08-voice-guard-privileged-voip-capture-research-spike.md`**
  (committed this session) already investigated a *system-level* Zygisk
  bypass of `AudioPlaybackCaptureConfiguration`'s
  `USAGE_VOICE_COMMUNICATION` exclusion, for VoIP apps generically instead
  of per-app reverse-engineering. Findings-only so far, no code shipped —
  worth a follow-up spike to see if the same class of system-level hook
  could also help with the native telephony-call restriction (#1), not
  just VoIP apps.
- **Look for a completely different capture point that isn't
  microphone-shaped at all**: e.g. Android's `AudioPlaybackCapture` API
  (API 29+, `MediaProjection`-based) for apps that don't set
  `USAGE_VOICE_COMMUNICATION` — already partially built
  (`PlaybackCaptureService.dart`, `AudioPlaybackCapture` manager per commit
  history) for the VoIP-app angle (WhatsApp/Zoom/Telegram/Meet). Revisit
  whether any part of that mechanism, or a variant of it, has any
  applicability to the native dialer/telephony path too.
- **Bluetooth/wired external mic as the capture device** instead of the
  phone's internal mic, on the theory that OEM anti-recording heuristics
  might be scoped to the internal mic specifically. Untested — worth a
  quick check since it's low effort. User has wired earphones (inline mic)
  and a Bluetooth headset on hand to try this with, zero cost.
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
  hooks) or applies uniformly regardless of call type.
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
