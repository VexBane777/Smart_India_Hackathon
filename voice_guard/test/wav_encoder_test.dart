import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';
import 'package:voice_guard/utils/wav_encoder.dart';

void main() {
  test('encodePcm16Mono round-trips through AudioProcessor.decodePcm16Wav', () {
    final samples = List<double>.generate(1600, (i) => 0.5 * math.sin(2 * math.pi * 220 * i / 16000));

    final wavBytes = WavEncoder.encodePcm16Mono(samples, sampleRate: 16000);
    final decoded = AudioProcessor.decodePcm16Wav(wavBytes);

    expect(decoded, isNotNull);
    expect(decoded!.sampleRate, 16000);
    expect(decoded.samples.length, samples.length);
    for (var i = 0; i < samples.length; i++) {
      // PCM16 quantization: within one quantization step of the original.
      expect((decoded.samples[i] - samples[i]).abs(), lessThan(1.0 / 32768.0 * 1.5));
    }
  });

  test('encodePcm16Mono produces a canonical 44-byte header', () {
    final bytes = WavEncoder.encodePcm16Mono([0.0, 0.1, -0.1], sampleRate: 16000);
    expect(bytes.length, 44 + 3 * 2);
    expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(bytes.sublist(8, 12)), 'WAVE');
    expect(String.fromCharCodes(bytes.sublist(12, 16)), 'fmt ');
    expect(String.fromCharCodes(bytes.sublist(36, 40)), 'data');
  });
}
