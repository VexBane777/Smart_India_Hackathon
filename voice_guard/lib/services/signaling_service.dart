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
