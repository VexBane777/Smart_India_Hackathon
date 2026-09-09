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

  Future<double> infer(List<double> lfcc, List<double> prosody) async => _heuristic(lfcc, prosody);

  double _heuristic(List<double> lfcc, List<double> prosody) {
    if (lfcc.isEmpty) return 0.15;
    final high = lfcc.sublist((lfcc.length * 0.6).floor());
    final meanH = high.reduce((a, b) => a + b) / high.length;
    final varH = high.map((v) => (v - meanH) * (v - meanH)).reduce((a, b) => a + b) / high.length;
    final pauseRatio = prosody.isNotEmpty ? prosody[0] : 0.2;
    double raw = (varH * 0.9 + pauseRatio * 0.25 + (lfcc[0].abs() * 0.05)).clamp(0.0, 1.0);
    raw = 0.08 + raw * 0.78;
    return raw;
  }

  Future<double> scoreChunk(List<double> pcm) {
    final lfcc = AudioProcessor.extractLfcc(pcm);
    final prosody = AudioProcessor.extractProsody(pcm);
    return infer(lfcc, prosody);
  }

  void dispose() {}
}
