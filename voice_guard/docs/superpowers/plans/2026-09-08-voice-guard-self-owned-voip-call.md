# Self-Owned VoIP Call (WebRTC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give VoiceGuard its own peer-to-peer, VoiceGuard-to-VoiceGuard call mode where the app owns both the uplink and downlink PCM natively — no OS mic-during-call restriction to fight, because there is nothing to intercept: the audio never leaves the app's own code path.

**Architecture:** Two VoiceGuard installs place a direct WebRTC call to each other. A tiny WebSocket signaling relay (added to the existing FastAPI `backend/main.py`, no new server stack) exchanges SDP offers/answers and ICE candidates between the two devices; once connected, WebRTC's own `getUserMedia`-equivalent (`flutter_webrtc`'s local audio track) gives the app direct access to both the outgoing mic audio and the incoming remote audio track's raw frames, which feed the existing `AudioService.ingestBytes()` → TFLite scoring pipeline exactly like every other capture path in this app. **Scope decision:** this is an in-app "VAANI Protected Call" feature between two VoiceGuard users, not a replacement for the existing dial-a-real-number flow and not a full SIP/PSTN gateway — see Global Constraints for why.

**Tech Stack:** `flutter_webrtc` (Flutter/Dart binding over Google's WebRTC native libraries), `web_socket_channel` (Dart WebSocket client), FastAPI `websockets` support (already a `uvicorn[standard]` transitive dependency per `backend/requirements.txt`) for the signaling relay.

**Spec:** No separate spec doc exists — this plan's own Architecture section *is* the spec, derived from the debugging investigation earlier in this conversation. Read `docs/superpowers/plans/2026-09-08-voice-guard-audioplaybackcapture.md`'s "Spec" section first — it documents the underlying platform restriction (carrier-call mic capture blocked at the OS level, confirmed by direct WAV sample analysis) that makes this heavier plan necessary at all for a "real live call" demo scenario.

## Global Constraints

- **Scope is peer-to-peer app-to-app calling, not PSTN interconnection.** Building an actual SIP trunk into the real phone network (so a VoiceGuard user could call an arbitrary landline/mobile number and still own the audio) requires a paid SIP trunk provider, carrier-grade signaling, and telecom compliance work far outside a hackathon demo's scope. This plan's "self-owned call" is a call between two devices that both have VoiceGuard installed — pitch it to judges as "VAANI's own protected calling channel," not as a general phone-call replacement.
- **This is an in-app call, not a Telecom-framework call.** Do not register a `PhoneAccount`/`ConnectionService` for this feature and do not route it through `InCallServiceImpl` (that class stays dedicated to real carrier calls, per the existing `voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/InCallServiceImpl.kt`). Reinventing Telecom integration on top of WebRTC would add real complexity (call-in-progress OS UI, audio focus arbitration, notification requirements) for no benefit here, since there's no interop requirement with the dialer/contacts app for a VoiceGuard-to-VoiceGuard call.
- **Reuse the existing scoring pipeline.** `AudioService.ingestBytes()` / `AudioService.scoreStream` (in `lib/services/audio_service.dart`) is the single point where PCM enters TFLite scoring across every capture path already in this app (real calls, the Live Mic self-test, and the AudioPlaybackCapture plan). This plan must feed it the same way — do not build a parallel scoring path.
- **Signaling server has no auth beyond what's already in `backend/main.py`** (the demo's `X-API-Key: vg_demo_key` header). This is fine for a hackathon demo on a controlled network; note in code comments that call-routing/room-matching has no user identity verification and is not production-ready.
- **No STUN/TURN infrastructure dependency for the demo.** Both devices are expected to be on the same local Wi-Fi network at demo time (judges' table). Use only ICE **host candidates** (same-subnet direct connection) as the primary path; include a public STUN server (`stun:stun.l.google.com:19302`) as a non-critical fallback for NAT traversal so the feature still has a chance to work if the two devices end up on different subnets, but do not make the demo's success depend on it.
- **`flutter_webrtc` requires additional native permissions/config already partially present** (mic, which VoiceGuard already declares) — verify camera permission is NOT requested if this stays audio-only (do not add `<uses-permission android:name="android.permission.CAMERA"/>` — this is a voice product, adding an unused camera permission is a real Play-Store-review and privacy-review red flag even though "Play Store legal isn't a priority" was said elsewhere in this project's history for a different, narrower context — don't over-generalize that into requesting permissions with no corresponding feature).

---

## File Structure

New files:
- `backend/signaling.py` — WebSocket signaling relay (`/v1/signal/{room_id}`), added as a FastAPI router included from `backend/main.py`. Kept in its own file since `main.py` is the REST scoring API and this is a structurally different concern (stateful WebSocket relay vs. stateless per-request scoring).
- `android/app/src/main/kotlin/com/voiceguard/voice_guard/... ` — **none needed.** `flutter_webrtc` ships its own Android/iOS platform implementation; this plan does not touch native Kotlin code, unlike the AudioPlaybackCapture plan.
- `lib/services/signaling_service.dart` — Dart WebSocket client wrapping the relay's message protocol (join room, send/receive offer/answer/candidate).
- `lib/services/webrtc_call_service.dart` — owns the `RTCPeerConnection`, local/remote audio tracks, and bridges remote-audio PCM frames into `AudioService.ingestBytes()`.
- `lib/screens/protected_call_screen.dart` — new screen: enter a room code (shared verbally between the two demo phones, or QR-coded), connect, show the same `RiskMeter` UI pattern as the other capture modes.

Modified files:
- `backend/main.py` — `app.include_router(signaling_router)`.
- `backend/requirements.txt` — no new entry needed (`uvicorn[standard]` already pulls in `websockets`); add a comment noting this.
- `pubspec.yaml` — add `flutter_webrtc` and `web_socket_channel` dependencies.
- `lib/main.dart` — provide `WebRtcCallService`, add navigation entry.

---

## Task 1: Signaling relay on the existing FastAPI backend

**Files:**
- Create: `backend/signaling.py`
- Modify: `backend/main.py`
- Test: `backend/test_signaling.py`

**Interfaces:**
- Produces: `signaling_router: fastapi.APIRouter` with one WebSocket route `/v1/signal/{room_id}` that relays any JSON message it receives to the *other* client currently connected to the same `room_id` (exactly two clients per room; a third connection to an occupied room is rejected).

- [ ] **Step 1: Write the failing test**

Create `backend/test_signaling.py`:

```python
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_two_clients_relay_messages_to_each_other():
    with client.websocket_connect("/v1/signal/room1") as caller, \
         client.websocket_connect("/v1/signal/room1") as callee:
        caller.send_json({"type": "offer", "sdp": "fake-sdp"})
        msg = callee.receive_json()
        assert msg == {"type": "offer", "sdp": "fake-sdp"}

        callee.send_json({"type": "answer", "sdp": "fake-answer"})
        msg = caller.receive_json()
        assert msg == {"type": "answer", "sdp": "fake-answer"}

def test_third_client_to_occupied_room_is_rejected():
    with client.websocket_connect("/v1/signal/room2"):
        with client.websocket_connect("/v1/signal/room2"):
            with pytest.raises(Exception):
                with client.websocket_connect("/v1/signal/room2") as third:
                    third.receive_json()
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd voice_guard/backend && pytest test_signaling.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'signaling'` (imported transitively once `main.py` tries to include it in Step 3) or a collection error, since `signaling_router` doesn't exist yet.

- [ ] **Step 3: Implement the relay**

Create `backend/signaling.py`:

```python
"""
WebSocket signaling relay for VoiceGuard's peer-to-peer "Protected Call"
feature (see docs/superpowers/plans/2026-09-08-voice-guard-self-owned-voip-call.md).

Exactly two clients per room_id: whatever one sends (SDP offer/answer, ICE
candidates) is relayed verbatim to the other. No message inspection, no
persistence — this is a dumb pipe. Room capacity of 2 keeps the demo's
threat model simple: no third party can join and MITM without knowing (and
occupying before the second caller) the shared room_id.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

signaling_router = APIRouter()

_rooms: dict[str, list[WebSocket]] = {}


@signaling_router.websocket("/v1/signal/{room_id}")
async def signal(websocket: WebSocket, room_id: str):
    await websocket.accept()
    peers = _rooms.setdefault(room_id, [])
    if len(peers) >= 2:
        await websocket.close(code=4000, reason="room full")
        return
    peers.append(websocket)
    try:
        while True:
            msg = await websocket.receive_json()
            for peer in peers:
                if peer is not websocket:
                    await peer.send_json(msg)
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in peers:
            peers.remove(websocket)
        if not peers:
            _rooms.pop(room_id, None)
```

- [ ] **Step 4: Wire the router into the app**

In `backend/main.py`, add near the top (after the existing imports, before `app = FastAPI(...)`):

```python
from signaling import signaling_router
```

And immediately after `app = FastAPI(...)` is constructed:

```python
app.include_router(signaling_router)
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd voice_guard/backend && pytest test_signaling.py -v`
Expected: `2 passed`

- [ ] **Step 6: Commit**

```bash
git add backend/signaling.py backend/main.py backend/test_signaling.py
git commit -m "feat(voice_guard): add WebSocket signaling relay for protected calls"
```

---

## Task 2: Dart SignalingService

**Files:**
- Create: `lib/services/signaling_service.dart`
- Modify: `pubspec.yaml` — add `web_socket_channel: ^3.0.1`
- Test: `test/signaling_service_test.dart`

**Interfaces:**
- Consumes: the `/v1/signal/{room_id}` WebSocket endpoint from Task 1.
- Produces: `SignalingService` class with:
  - `Future<void> connect(String roomId, {String host = '10.0.2.2', int port = 8001})`
  - `void send(Map<String, dynamic> message)`
  - `Stream<Map<String, dynamic>> get messages`
  - `Future<void> close()`

- [ ] **Step 1: Add the dependency**

In `pubspec.yaml`, under `dependencies:`, add:

```yaml
  web_socket_channel: ^3.0.1
```

Run: `cd voice_guard && flutter pub get`

- [ ] **Step 2: Write the failing test**

Create `test/signaling_service_test.dart`:

```dart
import 'dart:async';
import 'dart:convert';
import 'package:flutter_test/flutter_test.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:voice_guard/services/signaling_service.dart';

void main() {
  test('send encodes the message as JSON onto the underlying sink', () async {
    final sentFrames = <String>[];
    final local = StreamController<String>();
    final channel = _FakeChannel(local, sentFrames);

    final service = SignalingService.withChannel(channel);
    service.send({'type': 'offer', 'sdp': 'x'});

    expect(sentFrames, [jsonEncode({'type': 'offer', 'sdp': 'x'})]);
  });

  test('messages stream decodes incoming JSON frames', () async {
    final local = StreamController<String>();
    final sentFrames = <String>[];
    final channel = _FakeChannel(local, sentFrames);
    final service = SignalingService.withChannel(channel);

    final received = <Map<String, dynamic>>[];
    final sub = service.messages.listen(received.add);

    local.add(jsonEncode({'type': 'answer', 'sdp': 'y'}));
    await Future.delayed(Duration.zero);

    expect(received, [{'type': 'answer', 'sdp': 'y'}]);
    await sub.cancel();
  });
}

class _FakeChannel implements WebSocketChannel {
  final StreamController<String> _incoming;
  final List<String> sentFrames;
  _FakeChannel(this._incoming, this.sentFrames);

  @override
  Stream get stream => _incoming.stream;

  @override
  WebSocketSink get sink => _FakeSink(sentFrames);

  @override
  dynamic noSuchMethod(Invocation invocation) => null;
}

class _FakeSink implements WebSocketSink {
  final List<String> sentFrames;
  _FakeSink(this.sentFrames);
  @override
  void add(dynamic data) => sentFrames.add(data as String);
  @override
  dynamic noSuchMethod(Invocation invocation) => null;
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd voice_guard && flutter test test/signaling_service_test.dart`
Expected: FAIL — `Error: Not found: 'package:voice_guard/services/signaling_service.dart'`

- [ ] **Step 4: Write the implementation**

Create `lib/services/signaling_service.dart`:

```dart
import 'dart:async';
import 'dart:convert';
import 'package:web_socket_channel/web_socket_channel.dart';

/// Thin client for the room-based WebSocket relay in backend/signaling.py.
/// See docs/superpowers/plans/2026-09-08-voice-guard-self-owned-voip-call.md
/// for why this exists: it's the "who's calling whom" handshake for the
/// peer-to-peer WebRTC call — SDP offers/answers and ICE candidates travel
/// over this channel, media never does.
class SignalingService {
  final WebSocketChannel _channel;
  final _messagesCtrl = StreamController<Map<String, dynamic>>.broadcast();

  SignalingService.withChannel(this._channel) {
    _channel.stream.listen((frame) {
      final decoded = jsonDecode(frame as String) as Map<String, dynamic>;
      _messagesCtrl.add(decoded);
    });
  }

  factory SignalingService.connect(String roomId, {String host = '10.0.2.2', int port = 8001}) {
    final uri = Uri.parse('ws://$host:$port/v1/signal/$roomId');
    return SignalingService.withChannel(WebSocketChannel.connect(uri));
  }

  Stream<Map<String, dynamic>> get messages => _messagesCtrl.stream;

  void send(Map<String, dynamic> message) {
    _channel.sink.add(jsonEncode(message));
  }

  Future<void> close() async {
    await _channel.sink.close();
    await _messagesCtrl.close();
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd voice_guard && flutter test test/signaling_service_test.dart`
Expected: `00:0X +2: All tests passed!`

- [ ] **Step 6: Commit**

```bash
git add pubspec.yaml pubspec.lock lib/services/signaling_service.dart test/signaling_service_test.dart
git commit -m "feat(voice_guard): add Dart signaling client for protected calls"
```

---

## Task 3: WebRtcCallService — peer connection + audio bridging

**Files:**
- Modify: `pubspec.yaml` — add `flutter_webrtc: ^0.12.5`
- Create: `lib/services/webrtc_call_service.dart`
- Test: `test/webrtc_call_service_test.dart`

**Interfaces:**
- Consumes: `SignalingService` (Task 2), `AudioService.ingestBytes(Uint8List bytes)` (existing, `lib/services/audio_service.dart`).
- Produces: `WebRtcCallService` class with:
  - `Future<void> startCall(String roomId, {required bool isCaller})`
  - `Future<void> endCall()`
  - `Stream<RTCPeerConnectionState> get connectionState`

- [ ] **Step 1: Add the dependency**

In `pubspec.yaml`, under `dependencies:`, add:

```yaml
  flutter_webrtc: ^0.12.5
```

Run: `cd voice_guard && flutter pub get`

- [ ] **Step 2: Write the failing test**

`flutter_webrtc`'s native peer-connection objects cannot run under `flutter_test` (no platform channels backing them in a pure Dart test environment) — this is consistent with how this codebase already treats native-only Android glue (`AudioCaptureManager`, `InCallServiceImpl` have no Dart-side unit tests; verification for those is manual, on-device, per the existing plan's Task 6 pattern). Write the one piece of this class that *is* pure Dart logic and testable: the SDP-offer/answer routing decision based on `isCaller`.

Create `test/webrtc_call_service_test.dart`:

```dart
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/webrtc_call_service.dart';

void main() {
  test('caller role sends an offer-type signal first, callee does not', () {
    expect(WebRtcCallService.initialSignalType(isCaller: true), 'offer');
    expect(WebRtcCallService.initialSignalType(isCaller: false), null);
  });
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `cd voice_guard && flutter test test/webrtc_call_service_test.dart`
Expected: FAIL — `Error: Not found: 'package:voice_guard/services/webrtc_call_service.dart'`

- [ ] **Step 4: Write the implementation**

Create `lib/services/webrtc_call_service.dart`:

```dart
import 'dart:async';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'audio_service.dart';
import 'signaling_service.dart';

/// Owns the WebRTC peer connection for VoiceGuard's "Protected Call" mode.
/// See docs/superpowers/plans/2026-09-08-voice-guard-self-owned-voip-call.md:
/// this is the self-owned-audio path — both local and remote PCM stay
/// inside this process's own WebRTC stack, so there is no OS mic-during-a-
/// real-call restriction to hit (there is no "real call" from the
/// telephony stack's point of view; it's just this app talking to itself
/// on the network).
class WebRtcCallService {
  final AudioService audioService;
  final SignalingService signaling;
  RTCPeerConnection? _pc;
  MediaStream? _localStream;
  final _connectionStateCtrl = StreamController<RTCPeerConnectionState>.broadcast();

  WebRtcCallService({required this.audioService, required this.signaling}) {
    signaling.messages.listen(_onSignalingMessage);
  }

  Stream<RTCPeerConnectionState> get connectionState => _connectionStateCtrl.stream;

  /// Pure decision logic (see test): only the caller sends the first SDP
  /// offer; the callee waits for one to arrive over signaling.
  static String? initialSignalType({required bool isCaller}) =>
      isCaller ? 'offer' : null;

  static const _iceServers = {
    'iceServers': [
      {'urls': 'stun:stun.l.google.com:19302'},
    ],
  };

  Future<void> startCall(String roomId, {required bool isCaller}) async {
    _pc = await createPeerConnection(_iceServers);
    _pc!.onConnectionState = (state) => _connectionStateCtrl.add(state);
    _pc!.onIceCandidate = (candidate) {
      signaling.send({
        'type': 'candidate',
        'candidate': candidate.candidate,
        'sdpMid': candidate.sdpMid,
        'sdpMLineIndex': candidate.sdpMLineIndex,
      });
    };
    _pc!.onTrack = (event) {
      if (event.track.kind == 'audio') {
        // flutter_webrtc surfaces remote audio via the platform's own audio
        // output (it plays automatically) rather than raw PCM callbacks.
        // Scoring the remote party's voice from a WebRTC track requires an
        // audio sink/interceptor — see Task 4, which wires this up via
        // flutter_webrtc's RTCAudioSink once the track exists here.
      }
    };

    _localStream = await navigator.mediaDevices.getUserMedia({
      'audio': true,
      'video': false,
    });
    for (final track in _localStream!.getAudioTracks()) {
      await _pc!.addTrack(track, _localStream!);
    }

    if (initialSignalType(isCaller: isCaller) == 'offer') {
      final offer = await _pc!.createOffer();
      await _pc!.setLocalDescription(offer);
      signaling.send({'type': 'offer', 'sdp': offer.sdp});
    }
  }

  Future<void> _onSignalingMessage(Map<String, dynamic> msg) async {
    final pc = _pc;
    if (pc == null) return;
    switch (msg['type']) {
      case 'offer':
        await pc.setRemoteDescription(RTCSessionDescription(msg['sdp'] as String, 'offer'));
        final answer = await pc.createAnswer();
        await pc.setLocalDescription(answer);
        signaling.send({'type': 'answer', 'sdp': answer.sdp});
        break;
      case 'answer':
        await pc.setRemoteDescription(RTCSessionDescription(msg['sdp'] as String, 'answer'));
        break;
      case 'candidate':
        await pc.addCandidate(RTCIceCandidate(
          msg['candidate'] as String,
          msg['sdpMid'] as String?,
          msg['sdpMLineIndex'] as int?,
        ));
        break;
    }
  }

  Future<void> endCall() async {
    await _localStream?.dispose();
    await _pc?.close();
    _pc = null;
    _localStream = null;
  }

  void dispose() {
    _connectionStateCtrl.close();
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `cd voice_guard && flutter test test/webrtc_call_service_test.dart`
Expected: `00:0X +1: All tests passed!`

- [ ] **Step 6: Commit**

```bash
git add pubspec.yaml pubspec.lock lib/services/webrtc_call_service.dart test/webrtc_call_service_test.dart
git commit -m "feat(voice_guard): add WebRTC peer connection service (signaling + local audio only)"
```

---

## Task 4: Remote-audio scoring via RTCAudioSink

**Files:**
- Modify: `lib/services/webrtc_call_service.dart`

**Interfaces:**
- Consumes: `AudioService.ingestBytes(Uint8List bytes)` (existing).
- Produces: remote-party audio now reaches `AudioService`'s scoring pipeline, same as every other capture path in this app.

**Why this is its own task:** Task 3's `onTrack` handler has a comment flagging that remote audio isn't actually being routed into scoring yet — `flutter_webrtc` plays remote audio through the OS automatically, but getting the raw PCM frames out for our own scoring requires attaching an `RTCAudioSink` to the remote track. This is a distinct, separately-verifiable piece of behavior (Task 3 is "can two devices connect and hear each other," Task 4 is "is the far end's voice actually reaching the model") and should not be silently folded into Task 3, where a reviewer could reasonably approve "calls connect" while this part is still a stub.

- [ ] **Step 1: Attach an RTCAudioSink to the remote track**

In `lib/services/webrtc_call_service.dart`, add the import:

```dart
import 'dart:typed_data';
```

Replace the `_pc!.onTrack = (event) { ... }` body from Task 3 with:

```dart
    _pc!.onTrack = (event) {
      if (event.track.kind != 'audio') return;
      final sink = RTCAudioSink(event.track);
      sink.onData((data) {
        // data.buffer is 16-bit PCM per flutter_webrtc's RTCAudioSink
        // contract; AudioService.ingestBytes expects exactly this shape
        // (see lib/services/audio_service.dart:24, the same format every
        // other capture path in this app already produces).
        audioService.ingestBytes(Uint8List.fromList(data.buffer));
      });
      _remoteAudioSink = sink;
    };
```

Add the field this references, near `MediaStream? _localStream;`:

```dart
  RTCAudioSink? _remoteAudioSink;
```

- [ ] **Step 2: Dispose the sink on call end**

In `endCall()`, add before `await _localStream?.dispose();`:

```dart
    _remoteAudioSink?.dispose();
    _remoteAudioSink = null;
```

- [ ] **Step 3: Verify the project builds**

Run: `cd voice_guard && flutter build apk --debug`
Expected: `✓ Built build\app\outputs\flutter-apk\app-debug.apk`. If `RTCAudioSink`'s exact API (constructor signature, `onData` callback payload shape) differs from what's assumed here, this step will surface it as a compile error — check the installed `flutter_webrtc` version's actual `RTCAudioSink` class (in `~/.pub-cache/hosted/pub.dev/flutter_webrtc-<version>/lib/src/native/rtc_audio_sink.dart` or the equivalent for whatever platform channel implementation gets resolved) and adjust the field/method names in Step 1 to match precisely — the version pinned in Task 3 (`^0.12.5`) is the plan's best information at write time, not a guarantee the API hasn't shifted.

- [ ] **Step 4: Commit**

```bash
git add lib/services/webrtc_call_service.dart
git commit -m "feat(voice_guard): route remote WebRTC audio into the TFLite scoring pipeline"
```

---

## Task 5: Protected Call screen

**Files:**
- Create: `lib/screens/protected_call_screen.dart`
- Modify: `lib/main.dart` — provide `WebRtcCallService`, add navigation entry.

**Interfaces:**
- Consumes: `SignalingService.connect()` (Task 2), `WebRtcCallService` (Tasks 3-4), `RiskScoreProvider`/`RiskMeter` (existing).
- Produces: a reachable screen where the demo presenter types a shared room code and taps Connect.

- [ ] **Step 1: Write the screen**

Create `lib/screens/protected_call_screen.dart`:

```dart
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import '../services/signaling_service.dart';
import '../services/webrtc_call_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/risk_meter.dart';

class ProtectedCallScreen extends StatefulWidget {
  const ProtectedCallScreen({super.key});
  @override
  State<ProtectedCallScreen> createState() => _ProtectedCallScreenState();
}

class _ProtectedCallScreenState extends State<ProtectedCallScreen> {
  final _roomController = TextEditingController();
  WebRtcCallService? _call;
  RTCPeerConnectionState _state = RTCPeerConnectionState.RTCPeerConnectionStateNew;

  Future<void> _connect({required bool isCaller}) async {
    final roomId = _roomController.text.trim();
    if (roomId.isEmpty) return;
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();
    risk.reset();
    audio.clearBuffer();
    audio.startScoring();

    final signaling = SignalingService.connect(roomId);
    final call = WebRtcCallService(audioService: audio, signaling: signaling);
    call.connectionState.listen((s) {
      if (mounted) setState(() => _state = s);
    });
    await call.startCall(roomId, isCaller: isCaller);
    setState(() => _call = call);
  }

  Future<void> _hangUp() async {
    final audio = context.read<AudioService>();
    await _call?.endCall();
    audio.stopScoring();
    setState(() => _call = null);
  }

  @override
  void dispose() {
    _call?.dispose();
    _roomController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.0;

    return Scaffold(
      appBar: AppBar(title: const Text('Protected Call (VAANI-to-VAANI)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          const Text(
            'Both phones must have VoiceGuard installed and share the same '
            'room code, entered on the same Wi-Fi network.',
            style: TextStyle(fontSize: 12),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _roomController,
            decoration: const InputDecoration(labelText: 'Room code'),
          ),
          const SizedBox(height: 12),
          if (_call == null)
            Row(children: [
              Expanded(child: ElevatedButton(onPressed: () => _connect(isCaller: true), child: const Text('Call'))),
              const SizedBox(width: 8),
              Expanded(child: ElevatedButton(onPressed: () => _connect(isCaller: false), child: const Text('Answer'))),
            ])
          else ...[
            Text('State: $_state'),
            const SizedBox(height: 20),
            RiskMeter(score: score),
            const SizedBox(height: 20),
            ElevatedButton(onPressed: _hangUp, child: const Text('Hang Up')),
          ],
        ]),
      ),
    );
  }
}
```

- [ ] **Step 2: Add a navigation entry point**

In `lib/main.dart`, alongside the quick-action card added in the AudioPlaybackCapture plan's Task 5 (or, if that plan hasn't been implemented yet, alongside the existing `'Live Call'` card), add:

```dart
              _QuickActionCard(
                icon: Icons.shield_outlined,
                title: 'Protected Call',
                subtitle: 'VAANI-to-VAANI',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const ProtectedCallScreen()),
                ),
              ),
```

Add `import 'screens/protected_call_screen.dart';` at the top of `lib/main.dart`.

(As in the sibling plan's Task 5 Step 4: confirm the exact quick-action widget class name against the current source before copying this in.)

- [ ] **Step 3: Manual verification on two devices**

Run the backend: `cd voice_guard/backend && uvicorn main:app --host 0.0.0.0 --port 8001`. Confirm both test phones can reach the backend host's IP on port 8001 (same Wi-Fi network; `adb shell ip addr` or the backend host machine's LAN IP, not `10.0.2.2` — that address is only valid from an Android *emulator*, not a physical device; `SignalingService.connect()`'s default `host` parameter will need to be overridden to the actual LAN IP for physical-device testing).

Install the built APK on both phones. On phone A: enter room code `demo1`, tap **Call**. On phone B: enter `demo1`, tap **Answer**. Confirm:
1. Both phones show `RTCPeerConnectionState.RTCPeerConnectionStateConnected` within a few seconds.
2. Speaking into phone A is audibly heard on phone B (and vice versa) — this confirms the WebRTC audio path itself works, independent of VoiceGuard's own scoring.
3. The risk meter on **each** phone moves in response to the **other** phone's speech (not its own) — this is the actual point of the feature; if a phone's meter reacts to its own user's voice, `RTCAudioSink` is attached to the wrong track (local instead of remote) and Task 4 needs revisiting.

- [ ] **Step 4: Commit**

```bash
git add lib/screens/protected_call_screen.dart lib/main.dart
git commit -m "feat(voice_guard): add Protected Call screen for VAANI-to-VAANI WebRTC calls"
```

---

## Self-Review Notes

- **Spec coverage:** signaling relay (Task 1), Dart signaling client (Task 2), peer connection + local audio (Task 3), remote audio scoring bridge (Task 4 — deliberately split out, see its own rationale note), UI + two-device verification (Task 5).
- **Placeholder scan:** no TBD/TODO markers introduced.
- **Type consistency:** `WebRtcCallService(audioService:, signaling:)` constructor shape is used identically in Task 5's `_connect()`; `SignalingService.connect(roomId, {host, port})` factory signature from Task 2 matches Task 5's manual-verification instructions (which explicitly call out that the default `host` needs overriding for physical-device testing — this is flagged rather than silently assumed to be a placeholder gap).
- **Known open risk, not a placeholder:** `RTCAudioSink`'s exact API is asserted from the plan author's best knowledge of `flutter_webrtc` at plan-writing time, not verified against the installed package version — Task 4 Step 3 explicitly directs the implementer to check the real installed source if the assumed shape doesn't compile, rather than papering over a mismatch.
