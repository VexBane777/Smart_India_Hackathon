import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/capture/audio_capture_bridge.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('start() emits 0.5s-ish chunks of the expected sample count',
      (tester) async {
    final bridge = AudioCaptureBridge();
    final granted = await bridge.requestPermission();
    expect(granted, isTrue, reason: 'test device must pre-grant RECORD_AUDIO');

    final chunks = <int>[];
    final sub = bridge.start().listen((chunk) => chunks.add(chunk.length));
    await Future.delayed(const Duration(seconds: 2));
    await bridge.stop();
    await sub.cancel();

    expect(chunks.length, greaterThanOrEqualTo(2));
    for (final len in chunks) {
      expect(len, closeTo(8000, 800)); // 0.5s @ 16kHz, +/-10% jitter allowed
    }
  });
}
