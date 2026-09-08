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
