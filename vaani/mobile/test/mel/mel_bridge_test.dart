import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/mel/mel_bridge.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('vaani/mel');

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('computeMelDb marshals pcm/sr and returns the mocked matrix', () async {
    Map<String, dynamic>? capturedArgs;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      capturedArgs = Map<String, dynamic>.from(call.arguments as Map);
      return [
        [1.0, 2.0],
        [3.0, 4.0],
      ];
    });

    final bridge = MelBridge();
    final result = await bridge.computeMelDb(Float32List.fromList([0.1, 0.2]), 16000);

    expect(capturedArgs!['sr'], 16000);
    expect((capturedArgs!['pcm'] as List).length, 2);
    expect(result, [
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });
}
