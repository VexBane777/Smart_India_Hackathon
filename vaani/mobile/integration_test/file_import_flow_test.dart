import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/capture/audio_decode_bridge.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('decodeFile returns ~1s of 16kHz mono PCM for the tone fixture',
      (tester) async {
    final bytes = await rootBundle.load('assets/test_fixtures/tone_1s_16k.wav');
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tone_1s_16k.wav');
    await file.writeAsBytes(bytes.buffer.asUint8List());

    final bridge = AudioDecodeBridge();
    final pcm = await bridge.decodeFile(file.path);

    // Allow +/- one decoder frame of slack around the expected 16000 samples.
    expect(pcm.length, greaterThan(15000));
    expect(pcm.length, lessThan(17000));
    expect(pcm.reduce((a, b) => a.abs() > b.abs() ? a : b).abs(), greaterThan(0.01));
  });
}
