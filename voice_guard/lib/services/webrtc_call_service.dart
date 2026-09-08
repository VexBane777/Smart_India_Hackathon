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
        // audio sink/interceptor — see Task 4, which wires this up once the
        // track exists here.
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
