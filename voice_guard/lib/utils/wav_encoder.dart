import 'dart:typed_data';

/// Encodes normalized [-1, 1] samples as a canonical 16-bit PCM mono .wav —
/// the inverse of AudioProcessor.decodePcm16Wav. Used for the Protected
/// Call forensic dump (Voip.md Phase 5).
class WavEncoder {
  static Uint8List encodePcm16Mono(List<double> samples, {int sampleRate = 16000}) {
    const bitsPerSample = 16;
    const channels = 1;
    final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
    final blockAlign = channels * bitsPerSample ~/ 8;
    final dataSize = samples.length * 2;

    final buffer = ByteData(44 + dataSize);
    void writeAscii(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        buffer.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    writeAscii(0, 'RIFF');
    buffer.setUint32(4, 36 + dataSize, Endian.little);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    buffer.setUint32(16, 16, Endian.little); // fmt chunk size
    buffer.setUint16(20, 1, Endian.little); // audioFormat = PCM
    buffer.setUint16(22, channels, Endian.little);
    buffer.setUint32(24, sampleRate, Endian.little);
    buffer.setUint32(28, byteRate, Endian.little);
    buffer.setUint16(32, blockAlign, Endian.little);
    buffer.setUint16(34, bitsPerSample, Endian.little);
    writeAscii(36, 'data');
    buffer.setUint32(40, dataSize, Endian.little);

    for (var i = 0; i < samples.length; i++) {
      final clamped = samples[i].clamp(-1.0, 1.0);
      final intSample = (clamped * 32767.0).round().clamp(-32768, 32767);
      buffer.setInt16(44 + i * 2, intSample, Endian.little);
    }

    return buffer.buffer.asUint8List();
  }
}
