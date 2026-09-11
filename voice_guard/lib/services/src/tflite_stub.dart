import 'package:flutter/foundation.dart';
import '../../utils/audio_processor.dart';

/// Web stub — heuristic scorer only, no native TFLite (dart:ffi unavailable on web).
class TFLiteService {
  bool _ready = false;
  bool get isReady => _ready;

  Future<void> init() async {
    debugPrint('TFLite stub (web): using heuristic scorer');
    _ready = false;
  }

  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars) async =>
      (_heuristic(lfccSequence, scalars), null, 0.0);

  double _heuristic(List<List<double>> lfccSequence, List<double> scalars) {
    // Kept independent from tflite_io.dart's copy per this project's
    // existing convention (tflite_stub.dart has never imported from
    // tflite_io.dart) — identical body, duplicated deliberately.
    if (lfccSequence.isEmpty) return 0.15;
    final pooled = List<double>.filled(60, 0);
    for (final frame in lfccSequence) {
      for (int i = 0; i < 60; i++) { pooled[i] += frame[i]; }
    }
    for (int i = 0; i < 60; i++) { pooled[i] /= lfccSequence.length; }
    final high = pooled.sublist((pooled.length * 0.6).floor());
    final meanH = high.reduce((a, b) => a + b) / high.length;
    final varH = high.map((v) => (v - meanH) * (v - meanH)).reduce((a, b) => a + b) / high.length;
    final pauseRatio = scalars.isNotEmpty ? scalars[0] : 0.2;
    double raw = (varH * 0.9 + pauseRatio * 0.25 + (pooled[0].abs() * 0.05)).clamp(0.0, 1.0);
    raw = 0.08 + raw * 0.78;
    return raw;
  }

  Future<(double, String?, double)> scoreChunk(List<double> pcm) {
    final lfccSequence = AudioProcessor.extractLfccSequence(pcm);
    final scalars = AudioProcessor.extractScalars(pcm);
    return infer(lfccSequence, scalars);
  }

  void dispose() {}
}
