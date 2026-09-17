# VoIP sub-project 1: Tailscale-backed signaling backbone — Design

**Date:** 2026-09-17
**Status:** Design approved by user in conversation. Not yet an implementation
plan (see "Next step" below).
**Relationship to other tracks:** this is **sub-project 1 of 3** decomposing
`Voip.md`'s original 6-phase production plan into an ordered set of smaller
specs, per the user's decision to prioritize a working demo across real
distance now, with a documented upgrade path to the full production design
later. Sub-project 2 (native 48kHz→16kHz audio resampling,
`RemoteAudioTap.kt`) and sub-project 3 (client call-state wiring + in-call AI
alert UX) are named here for context only — each gets its own
brainstorm → spec → plan cycle, not detailed in this document.

## 1. Motivation

`Voip.md`'s forensic gap analysis identifies the current signaling path as
hardcoded to `10.0.2.2:8001` (emulator loopback) with no NAT traversal,
meaning two real devices on different networks cannot connect at all. The
plan's own fix (Phase 2) was a self-hosted `coturn` TURN relay behind a
public VPS with HMAC-signed ephemeral credentials.

Conversation with the user established constraints that change the right
answer here:

- Exactly 2 physical devices, which must be able to call across **genuine
  distance** (different networks/carriers), not just the same LAN.
- **No third party in the data or metadata path** — ruled out managed
  TURN/signaling SaaS (Twilio, PubNub, etc.) explicitly.
- Self-hosting a VPS (Oracle/AWS/GCP/Azure free tier) was evaluated and is
  technically sound (WebRTC's DTLS-SRTP means even a relay you don't trust
  can't read audio content), but every major free tier requires credit-card
  identity verification, and running/hardening `coturn` plus cloud firewall
  configuration is avoidable work for a demo.
- Fully decentralized peer discovery (Jami/Tox-style DHT) was considered and
  explicitly ruled out as disproportionate engineering effort for
  "implement right away."
- **Decision:** use **Tailscale** (WireGuard mesh VPN) as the NAT-traversal
  layer instead of coturn/STUN/TURN. Both devices join the same free-tier
  "tailnet" (login via Google/GitHub/Microsoft/Apple — no credit card),
  each gets a stable `100.x.x.x` IP reachable from the other regardless of
  network, and Tailscale's own DERP relay (E2E-encrypted, same guarantee as
  DTLS-SRTP) is the fallback when direct WireGuard P2P can't establish. This
  eliminates coturn, the VPS, and all cloud firewall work from this
  sub-project's scope.

## 2. Architecture

```
Phone A (Tailscale: 100.x.x.a)              Phone B (Tailscale: 100.x.x.b)
        |                                            |
        |  WS ws://<signaling-host-tailscale-ip>:8001/v1/signal/<room_id>
        |------------------> signaling.py <----------|
        |     (SDP offer/answer + ICE candidates only, relayed verbatim)
        |                                            |
        \_________________ WebRTC PeerConnection ___________________/
                  (DTLS-SRTP audio, P2P over Tailscale mesh;
                   falls back to Tailscale DERP relay if needed —
                   still E2E encrypted either way)
```

- Tailscale handles NAT traversal at the network layer — this replaces
  coturn/STUN/TURN entirely for this sub-project. `backend/signaling.py`'s
  existing design (room-based, 2-peer cap, dumb relay — see its own
  docstring) is already correct and needs **no protocol change**; it only
  needs to run somewhere reachable over the tailnet instead of localhost.
- `WebRtcCallService._iceServers`
  (`lib/services/webrtc_call_service.dart:44-48`) keeps the public Google
  STUN entry as a harmless no-cost fallback but gets **no TURN entry** —
  none is needed.
- WebRTC's own ICE candidate gathering is expected to surface the
  Tailscale interface's `100.x.x.x` address as a usable host candidate
  automatically, since it's a real (virtual) network interface on the
  device. This needs on-device verification (§6) — it's the one piece of
  this design that's a reasonable-confidence assumption rather than a
  confirmed fact.

## 3. Components & changes

| File | Change |
|---|---|
| `backend/signaling.py` | **No code change.** Operational change only: run it on a machine joined to the tailnet (see §4), reachable at `<tailscale-ip>:8001` instead of `10.0.2.2`. |
| `lib/providers/settings_provider.dart` | Add persisted `signalingHost` (String) and `signalingPort` (int) fields via `SharedPreferences`, defaulting to the current hardcoded `10.0.2.2:8001` so emulator/dev workflows keep working unchanged. |
| `lib/screens/settings_screen.dart` | Add a text field to enter/edit the signaling host (the Tailscale IP of wherever `signaling.py` runs). |
| `lib/services/signaling_service.dart` | **No code change** — `SignalingService.connect` (line 21) already takes `host`/`port` parameters; only the caller needs to source them from settings instead of the default. |
| `lib/screens/protected_call_screen.dart` | Read `signalingHost`/`signalingPort` from `SettingsProvider` instead of relying on `SignalingService.connect`'s hardcoded default. |
| `lib/services/webrtc_call_service.dart` | **No change** — `_iceServers` stays public-STUN-only (line 44-48). Explicitly out of scope: SDP/track/call-state logic, which is sub-project 3. |

Explicitly **not built** in this sub-project (present in `Voip.md`'s
original Phase 2, cut for the reasons in §1): `coturn`, any cloud VPS,
the `/v1/webrtc/ice-servers` HMAC credential endpoint.

## 4. Where `signaling.py` runs

Two viable options; recommending the first as the default for the demo:

- **Recommended: a laptop joined to the tailnet.** One `uvicorn` command,
  no mobile runtime, easy to restart/inspect logs during the demo. The
  laptop just needs the Tailscale app installed and logged into the same
  account as both phones.
- **Alternative: one of the two phones itself**, via Termux running Python.
  Removes the laptop dependency entirely, but adds Android
  background-process reliability concerns (the OS can kill a backgrounded
  Termux process) that are out of scope to solve here. Worth revisiting
  only if a laptop genuinely isn't available at demo time.

## 5. NAT traversal & fallback behavior

- Tailscale attempts direct WireGuard P2P first (UDP hole-punching across
  CGNAT/mobile carrier NAT); if that fails, it transparently falls back to
  Tailscale's own DERP relay. Both paths are Noise-protocol encrypted
  end-to-end — the DERP relay operator (Tailscale) cannot read traffic,
  matching the privacy property already established for coturn/DTLS-SRTP
  in the earlier conversation.
- **Known risk, not expected to block the demo:** if WebRTC's ICE gathering
  does not surface the Tailscale interface as a usable candidate on a given
  Android build, direct P2P audio over the tailnet won't establish. Given
  Tailscale exposes a normal interface with a real IP, this is
  low-probability, but it's the one assumption in this design that needs
  on-device confirmation before relying on it (§6). Mitigation if it does
  fail: revisit self-hosted coturn (§7) — not a redesign of this
  sub-project, an escalation to the deferred one.

## 6. Testing / verification plan

1. Install the official Tailscale Android app on both phones, log into the
   same tailnet, confirm each can `ping` the other's `100.x.x.x` address
   with the phones on **different** networks (e.g. one on home Wi-Fi, one
   on cellular data) — this is the actual scenario the original plan's
   CGNAT problem describes, so it's the right thing to validate first,
   before any app code changes.
2. Run `signaling.py` on the chosen host (§4), confirm both phones'
   `SignalingService` can open a WebSocket to
   `ws://<tailscale-ip>:8001/v1/signal/<room_id>`.
3. Re-run the existing `backend/test_signaling.py` — protocol is unchanged,
   should pass without modification.
4. Manual end-to-end: two phones on different networks, dial through
   `protected_call_screen.dart`, confirm the call connects and audio flows.
   This is the sub-project's actual acceptance criterion.

## 7. Explicitly deferred (post-demo / production upgrade note)

Per the user's request for "a distinct note for upgradation ... post-demo":

- **Self-hosted `coturn` + `/v1/webrtc/ice-servers` HMAC endpoint**
  (`Voip.md` Phase 2) — needed if scaling beyond what Tailscale's free tier
  comfortably handles, or for a real public release where requiring end
  users to install a VPN app before making a call is not acceptable
  (Tailscale-on-both-devices is a reasonable ask for a demo between 2
  devices you control, not for arbitrary end users).
- **Android Telecom `ConnectionService`** (`Voip.md` Phase 4) — lock-screen
  incoming call UI, audio-focus arbitration with carrier calls.
- **Full user directory / auth** beyond a shared room code (`Voip.md`
  Phase 1.1's stateful registration/session-state-machine design).
- **`signaling.py` hardening** — it is currently a dumb, unauthenticated
  relay trusting `room_id` as its only access control. Reasonable for a
  demo between 2 known, Tailscale-authenticated devices; not reasonable at
  production scale once arbitrary end users (and therefore TURN) enter the
  picture.
- **FCM background call wake-up**, full Phase 6 hardware/impairment test
  matrix (Bluetooth route switching, packet-loss injection via TeleChannel
  Module A).

## Next step

This spec covers sub-project 1 only. Once reviewed, the next step is the
`superpowers:writing-plans` skill to produce the implementation plan for
this sub-project's concrete tasks (SettingsProvider fields + settings UI,
wiring `protected_call_screen.dart` to read them, standing up
`signaling.py` on the tailnet host, and the on-device Tailscale
verification in §6). Sub-projects 2 and 3 get their own
brainstorm → spec → plan cycles after this one ships and is demo-verified.
