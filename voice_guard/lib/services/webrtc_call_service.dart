import 'dart:async';
import 'package:flutter/services.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'audio_service.dart';
import 'signaling_service.dart';

/// Native bridge for tapping raw PCM off the remote WebRTC audio track.
/// flutter_webrtc has no Dart-level `RTCAudioSink`-equivalent API (verified
/// absent across its entire published history — see
/// android/app/src/main/kotlin/com/voiceguard/voice_guard/RemoteAudioTap.kt
/// for the native mechanism this actually uses:
/// org.webrtc.AudioTrack.addSink(), which flutter_webrtc's own Dart/Java
/// layer never wraps but which is real and public on the native track
/// object underneath).
const _audioTapChannel = MethodChannel('com.voiceguard/webrtc_audio_tap');
const _audioTapEventChannel = EventChannel('com.voiceguard/webrtc_audio_tap_stream');

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
  StreamSubscription? _remoteAudioSub;
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
        _attachRemoteAudioTap(event.track.id!);
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

  /// Asks the native side (RemoteAudioTap.kt) to attach a raw-PCM sink to
  /// the remote track with this id, then routes whatever it emits into the
  /// same scoring pipeline every other capture path in this app already
  /// uses.
  Future<void> _attachRemoteAudioTap(String trackId) async {
    final attached = await _audioTapChannel.invokeMethod<bool>('attach', {'trackId': trackId}) ?? false;
    if (!attached) return;
    _remoteAudioSub = _audioTapEventChannel.receiveBroadcastStream().listen((data) {
      audioService.ingestBytes(Uint8List.fromList(data as List<int>));
    });
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
    await _remoteAudioSub?.cancel();
    _remoteAudioSub = null;
    await _audioTapChannel.invokeMethod('detach');
    await _localStream?.dispose();
    await _pc?.close();
    _pc = null;
    _localStream = null;
  }

  void dispose() {
    _connectionStateCtrl.close();
  }
}
