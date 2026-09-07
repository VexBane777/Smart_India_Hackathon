import 'dart:async';
import 'dart:typed_data';
import '../utils/audio_processor.dart';
import 'tflite_service.dart';

import 'dart:math' as math;

/// Buffers raw PCM16 frames from the native EventChannel and emits risk scores.
/// Raw PCM is held only in RAM and discarded after feature extraction.
class AudioService {
  final TFLiteService tflite;
  AudioService(this.tflite);

  final List<double> _buffer = [];
  Timer? _timer;
  final _scoreCtrl = StreamController<double>.broadcast();
  final _pcmCtrl = StreamController<List<double>>.broadcast();
  final _rmsCtrl = StreamController<double>.broadcast();

  Stream<double> get scoreStream => _scoreCtrl.stream;
  Stream<List<double>> get pcmStream => _pcmCtrl.stream;
  Stream<double> get rmsStream => _rmsCtrl.stream;

  void ingestBytes(Uint8List bytes) {
    final samples = AudioProcessor.pcm16ToDouble(bytes);
    if (samples.isEmpty) return;
    _buffer.addAll(samples);
    // keep only last 5s to bound memory
    const maxSamples = 80000;
    if (_buffer.length > maxSamples) {
      _buffer.removeRange(0, _buffer.length - maxSamples);
    }

    // Compute RMS and downsampled waveform bars
    double sum = 0;
    for (final s in samples) {
      sum += s * s;
    }
    final rms = math.sqrt(sum / samples.length);
    _rmsCtrl.add(rms);

    final step = math.max(1, samples.length ~/ 28);
    final wave = <double>[];
    for (int i = 0; i < samples.length && wave.length < 28; i += step) {
      wave.add(samples[i]);
    }
    _pcmCtrl.add(wave);
  }

  void startScoring({Duration interval = const Duration(seconds: 1)}) {
    _timer?.cancel();
    _timer = Timer.periodic(interval, (_) {
      if (_buffer.length < 4800) return; // need ~0.3s minimum
      final chunk = _buffer.length >= 48000
          ? _buffer.sublist(_buffer.length - 48000)
          : List<double>.from(_buffer);
      final score = tflite.scoreChunk(chunk);
      _scoreCtrl.add(score);
    });
  }

  void stopScoring() {
    _timer?.cancel();
    _timer = null;
  }

  void clearBuffer() => _buffer.clear();

  void injectBenchmarkTest({required bool isAiVoice}) {
    final chunk = List<double>.generate(16000, (i) {
      final t = i / 16000.0;
      if (isAiVoice) {
        // Static mechanical waveform typical of vocoders
        return 0.25 * math.sin(2 * math.pi * 180 * t) +
            0.20 * math.sin(2 * math.pi * 360 * t) +
            0.15 * math.sin(2 * math.pi * 540 * t);
      } else {
        // Dynamic human speech with natural frequency modulation and envelope
        final f0 = 120.0 + 15.0 * math.sin(2 * math.pi * 3.5 * t);
        final env = math.max(0.0, math.sin(2 * math.pi * 1.2 * t));
        return env * (0.35 * math.sin(2 * math.pi * f0 * t) +
            0.25 * math.sin(2 * math.pi * (2 * f0) * t) +
            0.05 * (math.Random(i).nextDouble() - 0.5));
      }
    });
    final score = tflite.scoreChunk(chunk);
    _scoreCtrl.add(score);
    final wave = chunk.take(28).toList();
    _pcmCtrl.add(wave);
    _rmsCtrl.add(isAiVoice ? 0.35 : 0.25);
  }

  void dispose() {
    _timer?.cancel();
    _scoreCtrl.close();
    _pcmCtrl.close();
    _rmsCtrl.close();
  }
}

