import 'dart:typed_data';

import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';

/// Minimal big/little-endian writers for building WAV bytes in tests.
void _le16(BytesBuilder b, int v) {
  b.addByte(v & 0xFF);
  b.addByte((v >> 8) & 0xFF);
}

void _le32(BytesBuilder b, int v) {
  b.addByte(v & 0xFF);
  b.addByte((v >> 8) & 0xFF);
  b.addByte((v >> 16) & 0xFF);
  b.addByte((v >> 24) & 0xFF);
}

/// Builds a canonical PCM WAV. [channels] > 1 interleaves [samples] in
/// channel-major then frame-major order (L0, R0, L1, R1...).
/// If [extensible] is true, emits a WAVE_FORMAT_EXTENSIBLE 'fmt ' chunk with
/// a PCM subformat GUID instead of a plain PCM one.
Uint8List _buildWav({
  required int sampleRate,
  required int channels,
  required List<int> samples,
  bool extensible = false,
}) {
  final data = BytesBuilder();
  for (final s in samples) {
    _le16(data, s & 0xFFFF);
  }
  final dataBytes = data.toBytes();

  final fmtSize = extensible ? 40 : 16;
  final out = BytesBuilder();
  out.add('RIFF'.codeUnits);
  _le32(out, 4 + (8 + fmtSize) + (8 + dataBytes.length));
  out.add('WAVE'.codeUnits);

  out.add('fmt '.codeUnits);
  _le32(out, fmtSize);
  _le16(out, extensible ? 0xFFFE : 1);
  _le16(out, channels);
  _le32(out, sampleRate);
  _le32(out, sampleRate * channels * 2); // byte rate
  _le16(out, channels * 2); // block align
  _le16(out, 16); // bits per sample
  if (extensible) {
    _le16(out, 22); // cbSize
    _le16(out, 16); // valid bits
    _le32(out, 1); // channel mask: front-left
    // PCM subformat GUID: Data1=00000001, Data2=0000, Data3=0010,
    // Data4=8000-00aa00389b71 (16 bytes total).
    _le16(out, 1);
    _le16(out, 0);
    _le16(out, 0);
    _le16(out, 0x0010);
    _le16(out, 0x8000);
    out.add(const [0, 0xAA, 0, 0x38, 0x9B, 0x71]);
  }

  out.add('data'.codeUnits);
  _le32(out, dataBytes.length);
  out.add(dataBytes);
  return out.toBytes();
}

void main() {
  group('AudioProcessor.decodePcm16Wav', () {
    test('decodes mono 16-bit PCM at the file sample rate', () {
      final samples = <int>[0, 1000, -1000, 32767, -32768];
      final wav = _buildWav(sampleRate: 16000, channels: 1, samples: samples);

      final decoded = AudioProcessor.decodePcm16Wav(wav);
      expect(decoded, isNotNull);
      expect(decoded!.sampleRate, 16000);
      expect(decoded.samples.length, samples.length);
      expect(decoded.samples[0], closeTo(0.0, 1e-6));
      expect(decoded.samples[1], closeTo(1000 / 32768.0, 1e-6));
      expect(decoded.samples[4], closeTo(-32768 / 32768.0, 1e-6));
    });

    test('averages stereo channels to mono', () {
      // 3 frames of stereo: (1, -1), (1000, 2000), (-1000, 1000)
      final samples = <int>[1, -1, 1000, 2000, -1000, 1000];
      final wav = _buildWav(sampleRate: 16000, channels: 2, samples: samples);

      final decoded = AudioProcessor.decodePcm16Wav(wav);
      expect(decoded!.samples.length, 3);
      expect(decoded.samples[0], closeTo(0.0, 1e-6));
      expect(decoded.samples[1], closeTo(1500 / 32768.0, 1e-6));
      expect(decoded.samples[2], closeTo(0.0, 1e-6));
    });

    test('handles WAVE_FORMAT_EXTENSIBLE with a PCM subformat', () {
      final samples = <int>[0, 1000, 2000];
      final wav = _buildWav(
        sampleRate: 48000,
        channels: 1,
        samples: samples,
        extensible: true,
      );

      final decoded = AudioProcessor.decodePcm16Wav(wav);
      expect(decoded, isNotNull);
      expect(decoded!.sampleRate, 48000);
      expect(decoded.samples[2], closeTo(2000 / 32768.0, 1e-6));
    });

    test('resamples a 48 kHz decode to the model-native 16 kHz', () {
      final n = 4800; // 0.1 s at 48 kHz
      final samples = List<int>.generate(n, (i) => (i * 100) & 0xFFFF);
      final wav = _buildWav(sampleRate: 48000, channels: 1, samples: samples);

      final decoded = AudioProcessor.decodePcm16Wav(wav)!;
      final resampled = AudioProcessor.resampleLinear(
          decoded.samples, decoded.sampleRate, AudioProcessor.sampleRate);
      expect(resampled.length, (n / 3).round());
      expect(resampled, isNot(contains(double.nan)));
    });

    test('resampleLinear is identity for matching rates', () {
      final x = <double>[0.1, 0.2, 0.3];
      expect(AudioProcessor.resampleLinear(x, 16000, 16000), x);
    });

    test('rejects non-RIFF, non-PCM, and truncated input', () {
      expect(AudioProcessor.decodePcm16Wav(Uint8List.fromList(List.filled(44, 0))), isNull);
      expect(AudioProcessor.decodePcm16Wav(Uint8List.fromList(List.filled(4, 0))), isNull);

      // float WAV (audioFormat == 3): decode must reject.
      final data = BytesBuilder();
      data.add('RIFF'.codeUnits);
      _le32(data, 40);
      data.add('WAVE'.codeUnits);
      data.add('fmt '.codeUnits);
      _le32(data, 16);
      _le16(data, 3); // IEEE float
      _le16(data, 1);
      _le32(data, 16000);
      _le32(data, 32000);
      _le16(data, 2);
      _le16(data, 32);
      data.add('data'.codeUnits);
      _le32(data, 0);
      expect(AudioProcessor.decodePcm16Wav(data.toBytes()), isNull);
    });
  });
}