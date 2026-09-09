# Research spike: system-level bypass of the AudioPlaybackCapture USAGE_VOICE_COMMUNICATION exclusion

**Status: findings only — no code shipped from this spike.** Scope and
framing agreed with the user in `2026-09-08` debugging session (see
`AudioCaptureManager.kt` and `PlaybackCaptureManager.kt` docstrings for the
underlying app-level root cause this spike investigates a bypass for).

## The question this spike set out to answer

Could a **Zygisk module hooking `system_server`'s audio-policy code**
relax the `USAGE_VOICE_COMMUNICATION` exclusion from
`AudioPlaybackCaptureConfiguration` generically, for every third-party VoIP
app (WhatsApp/Zoom/Telegram/Meet) at once — instead of reverse-engineering
each app's own obfuscated, frequently-updated audio pipeline individually?

## Finding: the enforcement point is not in `system_server` — this rules out the originally scoped approach

Read `AudioPlaybackCaptureConfiguration.Builder.addMatchingUsage()` directly
from AOSP source
([platform_frameworks_base, media/java/android/media/AudioPlaybackCaptureConfiguration.java](https://github.com/aosp-mirror/platform_frameworks_base/blob/master/media/java/android/media/AudioPlaybackCaptureConfiguration.java)):

```java
public @NonNull Builder addMatchingUsage(@AttributeUsage int usage) {
    Preconditions.checkState(
        mUsageMatchType != MATCH_TYPE_EXCLUSIVE,
        ERROR_MESSAGE_MISMATCHED_RULES);
    mAudioMixingRuleBuilder.addRule(
        new AudioAttributes.Builder().setUsage(usage).build(),
        AudioMixingRule.RULE_MATCH_ATTRIBUTE_USAGE);
    mUsageMatchType = MATCH_TYPE_INCLUSIVE;
    return this;
}
```

This method does **no validation** of the usage value — it accepts
`USAGE_VOICE_COMMUNICATION` without complaint (consistent with
`PlaybackCaptureManager.kt` never having crashed when it called this with
that exact value — see the cleanup made alongside this spike). It just
builds an `AudioMixingRule` descriptor and hands it downstream.

The actual restriction — "only `USAGE_UNKNOWN`/`USAGE_GAME`/`USAGE_MEDIA`
tracks are ever eligible to be captured" — is documented behavior
([developer.android.com/media/platform/av-capture](https://developer.android.com/media/platform/av-capture))
but is **enforced natively**, inside `AudioPolicyManager`/`AudioFlinger`,
which run in the **`audioserver` process** — a native daemon started
directly by `init` from its own `.rc` script at boot, not forked from
zygote.

This matters because **Zygisk only injects into zygote-forked processes**
— regular app processes and `system_server` (this is exactly the mechanism
LSPosed uses to hook `system_server`'s Java methods, confirmed via
[LSPosed's own docs/source](https://github.com/LSPosed/LSPosed)). It has
**no mechanism to reach `audioserver`**, since that process is never forked
from zygote at all. The originally scoped approach — "hook `system_server`
via Zygisk" — targets the wrong process for this specific enforcement
point and would not work even if fully implemented.

## What would actually be required, and why it's a materially bigger undertaking than originally scoped

To move or bypass this check would mean patching the native audio-policy
code running inside `audioserver` itself. On a rooted device the only
realistic mechanisms are:

- **Replacing/patching the `audioserver` binary or its native libraries**
  (e.g. `libaudiopolicymanager.so`) via a Magisk module overlay. This is a
  boot-critical system daemon — audioserver crashing takes down the whole
  device's audio system. It would need to be rebuilt/patched per Android
  version and CPU architecture, with no existing open-source reference
  implementation (unlike Track A's `CAPTURE_AUDIO_OUTPUT`/priv-app path,
  which BCR already proves out in production).
- **Runtime ptrace-based injection into the already-running `audioserver`
  process** (e.g. via a persistent Frida server running as root). Frida
  *can* attach to arbitrary native processes, including system daemons,
  since it doesn't depend on the zygote-fork hook point Zygisk uses. But
  running a persistent Frida server on a consumer device is a research/
  debugging technique, not a shippable production mechanism — it's
  trivially fingerprintable, adds a large stability/security surface to a
  core system daemon, and every stock Android security patch is free to
  change the exact functions being hooked without notice.

Both options are categorically riskier and higher-maintenance than
anything in Track A: no stable reference implementation exists (nothing
like BCR for this specific problem), the blast radius of getting it wrong
is total loss of device audio (not just this app breaking), and the
patch surface changes with every OEM/Android security update rather than
being a one-time integration.

## Recommendation: do not proceed

Given:
1. The original Zygisk/`system_server` approach is confirmed non-viable —
   it targets a process that structurally cannot reach the actual
   enforcement point.
2. The remaining options (native `audioserver` binary patching, or a
   persistent root-level Frida injection) are boot-critical-risk,
   version-fragile, and have no stable reference implementation to build
   on or maintain against.
3. Track A (the cellular-call privileged-capture tier via
   `magisk-privileged-module/`) already delivers the higher-value win —
   real telephony call audio — using a proven, actively-maintained
   open-source architecture (BCR) with a bounded, well-understood
   integration surface (one priv-app permission grant, not a system
   daemon patch).

**Do not pursue further engineering work on system-level VoIP capture
bypass.** Third-party VoIP apps (WhatsApp/Zoom/Telegram/Meet) should stay
on the existing acoustic (MIC + speakerphone) fallback path — the same
constraint already accepted for cellular calls on non-rooted devices.
If this is revisited in the future, the next real option to evaluate would
be per-app reverse engineering (hooking each VoIP app's own audio pipeline
individually via Zygisk, which *can* reach app processes) — but that was
explicitly out of scope for this spike and trades one large, open-ended
maintenance burden (a native daemon patch) for four smaller but
still-ongoing ones (one per target app, breaking on every app update).

## Follow-up (2026-09-09): does this also help with root cause #1 (native telephony-call zero-fill)?

state.md's "Ideas not yet tried" asked whether the same class of
system-level hook investigated above could also help with the native
telephony-call capture restriction (root cause #1: every `AudioRecord`
source, including plain `MIC`, gets silently zero-filled by the OS once a
real cellular call goes active — see `AudioCaptureManager.kt` docstring).
Answer, reachable from the same evidence gathered above without needing
hardware: **no, for the same structural reason, and it's actually a
stronger no than for VoIP apps.**

The zero-fill/mic-block behavior for an active telephony call is enforced
by the same class of component as the `AudioPlaybackCaptureConfiguration`
usage-tag restriction — native audio-policy code inside `audioserver`
(recording-concurrency/privacy policy that mutes or zero-fills
non-privileged `AudioRecord` reads while a call owns the mic), not
anything in `system_server`. `audioserver` is started directly by `init`
from its own native `.rc` script and is never forked from zygote, so
Zygisk — whose entire mechanism is hooking the zygote fork point — has no
attachment surface to it, exactly as found above for the VoIP-capture
question. The only mechanisms that could reach this enforcement point are
the same two identified above (patching `audioserver`'s native libraries
via a Magisk overlay, or a persistent root-level Frida injection), with
the same boot-critical blast radius and no stable reference
implementation to build from.

It's a *stronger* no than the VoIP case for one additional reason: even if
the mic-block were bypassed at the `audioserver` level, root cause #1 is
about a **carrier/OEM privacy restriction on recording live call audio**
(India 2024+ call-recording rules, ColorOS-enforced) — bypassing it is
squarely the scenario that restriction exists to prevent, unlike the VoIP
case where the restriction is a generic platform default with no
carrier/regulatory backing specific to this data. This spike's
recommendation is unchanged and now covers both questions: **do not
pursue system-level `audioserver`-level bypasses for either VoIP playback
capture or native telephony capture.** `magisk-privileged-module/`
(`CAPTURE_AUDIO_OUTPUT` via priv-app install, proven out by BCR) remains
the only validated path to real call audio on a rooted device, because it
works *with* the platform's own privileged-permission model rather than
against `audioserver`'s enforcement.
