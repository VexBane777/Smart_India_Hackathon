import 'dart:async';
import 'package:flutter/foundation.dart';
import '../utils/audio_processor.dart';
import 'tflite_service.dart';

import 'dart:math' as math;

/// Result of a calibration capture — mean/std of raw (pre-EMA) model
/// scores over N consecutive 3s windows of the user's own voice.
class CalibrationSample {
  final double mean;
  final double std;
  final int windowsCaptured;
  const CalibrationSample({required this.mean, required this.std, required this.windowsCaptured});

  factory CalibrationSample.fromScores(List<double> scores) {
    if (scores.isEmpty) {
      return const CalibrationSample(mean: 0.0, std: 0.0, windowsCaptured: 0);
    }
    final mean = scores.reduce((a, b) => a + b) / scores.length;
    final variance = scores.map((s) => (s - mean) * (s - mean)).reduce((a, b) => a + b) / scores.length;
    return CalibrationSample(mean: mean, std: math.sqrt(variance), windowsCaptured: scores.length);
  }
}

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
  final _signalCtrl = StreamController<bool>.broadcast();

  Stream<double> get scoreStream => _scoreCtrl.stream;
  Stream<List<double>> get pcmStream => _pcmCtrl.stream;
  Stream<double> get rmsStream => _rmsCtrl.stream;
  /// Whether the most recently *scored* window carried real signal, as
  /// opposed to silence/OS zero-filled reads (see AudioCaptureManager's
  /// docstring — some OEM builds zero-fill AudioRecord reads once the real
  /// call audio path takes over rather than erroring). Consumers should show
  /// "no voice detected" instead of trusting a score during a false stretch,
  /// since a window of exact/near-zero PCM deterministically collapses to
  /// the same LFCC features every time and would otherwise look like the
  /// model froze.
  Stream<bool> get hasSignalStream => _signalCtrl.stream;

  // Matches AudioCaptureManager's own SILENCE_RMS_THRESHOLD (50.0 on the
  // int16 PCM domain) converted to this class's normalized [-1, 1] domain.
  static const double _silenceRmsThreshold = 50.0 / 32768.0;

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
    _timer = Timer.periodic(interval, (_) async {
      // Require a full 3s window before scoring at all — a partial window
      // (e.g. the first ~0.3-1s right after clearBuffer(), often containing
      // only a dial tone/ring beep or a transient) produced spurious
      // high-confidence verdicts in testing (an "AI DETECTED" alert fired a
      // second into a call before any real speech had arrived).
      if (_buffer.length < 48000) return;
      final chunk = _buffer.sublist(_buffer.length - 48000);

      double sumSq = 0;
      for (final s in chunk) { sumSq += s * s; }
      final chunkRms = math.sqrt(sumSq / chunk.length);
      debugPrint('Monitor: chunkRms=${chunkRms.toStringAsFixed(6)} (int16-equiv=${(chunkRms * 32768).toStringAsFixed(1)}) '
          'threshold=${_silenceRmsThreshold.toStringAsFixed(6)} bufLen=${_buffer.length}');
      if (chunkRms < _silenceRmsThreshold) {
        // Silent/zero-filled window: skip scoring rather than feed the model
        // a degenerate all-floor input that would deterministically repeat
        // whatever score silence happens to map to.
        _signalCtrl.add(false);
        return;
      }
      _signalCtrl.add(true);

      final score = await tflite.scoreChunk(chunk);
      _scoreCtrl.add(score);
    });
  }

  void stopScoring() {
    _timer?.cancel();
    _timer = null;
  }

  void clearBuffer() => _buffer.clear();

  /// Collects `windows` consecutive raw scores from scoreStream, without
  /// touching RiskScoreProvider (deliberately — see plan Global
  /// Constraints: calibration must never pollute call logs/risk history).
  /// Caller is responsible for having already started native capture
  /// (`CallService.startCallDetection()`) and `startScoring()` — this
  /// method only *listens*, it doesn't start capture, mirroring how
  /// call_screen.dart's Live Mic Test already separates those concerns.
  Future<CalibrationSample> captureCalibrationSample({
    int windows = 8,
    Duration perWindowTimeout = const Duration(seconds: 2),
  }) async {
    final scores = <double>[];
    final sub = scoreStream.listen(scores.add);
    try {
      final deadline = DateTime.now().add(perWindowTimeout * windows);
      while (scores.length < windows && DateTime.now().isBefore(deadline)) {
        await Future.delayed(const Duration(milliseconds: 200));
      }
    } finally {
      await sub.cancel();
    }
    return CalibrationSample.fromScores(scores);
  }

  Future<void> injectBenchmarkTest({required bool isAiVoice}) async {
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
    final score = await tflite.scoreChunk(chunk);
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
    _signalCtrl.close();
  }
}

