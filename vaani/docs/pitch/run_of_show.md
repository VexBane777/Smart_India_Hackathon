<!-- Hand-written 2026-09-07 — not part of generate_vaani_docs.py's output set. -->
# VAANI — Run of Show (phone + laptop, two-screen demo)

Choreographic integration of `voice_guard` (Android, the live-alert
moment) and the Streamlit app (laptop, the consequence/evidence trail).
Zero new code, zero network dependency between the two devices — see
`00_MASTER_PLAN.md`'s non-negotiable #6 ("the demo never depends on the
venue"). Both devices run fully offline, independently, and are only
sequenced by a human on stage.

Maps onto the master plan's 90s beat sheet (`00_MASTER_PLAN.md` §9)
without changing it — just assigns each beat to a screen.

## Setup, before judges arrive

1. Phone: VoiceGuard installed, set as Default Dialer, overlay + mic
   permission granted, airplane mode OFF only long enough to confirm the
   app launches, then back ON (§6 non-negotiable — no live network on
   stage). Demo Call preloaded (`Live Call → Demo Call`, per
   `voice_guard/README.md`).
2. Phone mirrored to the projector: `scrcpy` over USB (free, no
   Wi-Fi/network cast needed) — test this specific laptop+phone pairing
   beforehand, USB cables and drivers are exactly the kind of thing that
   fails silently on a different machine than the one you tested on.
   Fallback if `scrcpy` misbehaves on the day: a phone document-camera
   stand, or just hold the phone up to a mic'd presenter camera.
3. Laptop: Streamlit app running locally (`streamlit run app.py` from
   `vaani/`), airplane mode ON, model preloaded, browser tab already
   open and warmed up (first Streamlit load can be visibly slow — don't
   let that happen live).
4. Both devices: brightness/volume checked against actual room
   conditions, not your desk.

## The 90 seconds

| T | Beat | Screen | Notes |
|---|---|---|---|
| 0:00–0:10 | Hook: Arup $25M + Indian advisory | (speaker, no screen) | verified facts only — master plan §2 |
| 0:10–0:25 | "Pre-recorded call, streamed through the exact live pipeline — disclosed" | phone (projected) | say this BEFORE playing audio, not after |
| 0:25–0:55 | Real voice (gauge low) → clone enters (gauge climbs) → 2 occlusion highlights | **phone** | this is voice_guard's one job — the risk meter animating live, in-app overlay firing, on real hardware |
| 0:55–1:10 | Transfer attempted → HOLD → OTP step-up modal | **cut to laptop** | Streamlit's `bank_modal.py` flow — say "simulated; production plugs into the bank's existing MFA" during the cut, covers the screen-switch |
| 1:10–1:30 | Audit log + metrics slide (EER table, measured latency, calibration) | **laptop** | `audit_log.py`'s hash chain + the pitch deck's lockdown numbers |

**The cut at 0:55 is the only handoff in the whole demo** — one clean
transition, called out verbally ("...and here's what happens on the
bank's side") so it reads as a deliberate scene change, not a recovery
from something going wrong.

## Fallback order (master plan §9, unchanged)

live app (both screens) → backup video (screen-recorded full run,
played instead if either device misbehaves) → annotated screenshots.
Rehearse the *fallback* too, not just the happy path — know exactly
which slide/video segment to jump to if the phone segment specifically
fails (most likely single point of failure: USB mirroring, not the app
itself).

## Rehearsal checklist (do this on the actual stage-day machine + phone)

- [ ] 3 full run-throughs, timed against the 90s budget above
- [ ] `scrcpy` (or chosen mirroring method) tested on the exact laptop
      that will be used, not a different one
- [ ] Backup video recorded from a clean run (screen recording, e.g. OBS
      or Windows Game Bar `Win+Alt+R`, saved as `.mp4`) — see
      `state.md`'s 2026-09-06 note: this is "Task 8," not yet attempted
- [ ] Airplane mode / cable-pull tested on both devices literally, not
      just via `tests/demo/test_offline.py`'s socket-mocked version
- [ ] Latency numbers measured during a rehearsal, fed into
      `docs/pitch/lockdown.json` Step 3
- [ ] OTP modal's disclosed on-screen constant re-checked (never
      actually sent anywhere — say this explicitly if a judge asks)
