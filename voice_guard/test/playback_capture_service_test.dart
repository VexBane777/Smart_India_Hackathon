import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/playback_capture_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('com.voiceguard/calls');

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('requestConsent returns the native result', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      expect(call.method, 'requestPlaybackCaptureConsent');
      return true;
    });
    final service = PlaybackCaptureService();
    expect(await service.requestConsent(), true);
  });

  test('consentResult stream emits pushed onPlaybackCaptureConsent events', () async {
    final service = PlaybackCaptureService();
    final events = <bool>[];
    final sub = service.consentResult.listen(events.add);

    final handler = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    final call = MethodCall('onPlaybackCaptureConsent', {'granted': true});
    final data = const StandardMethodCodec().encodeMethodCall(call);
    await handler.handlePlatformMessage('com.voiceguard/calls', data, (_) {});

    await Future.delayed(Duration.zero);
    expect(events, [true]);
    await sub.cancel();
  });
}
