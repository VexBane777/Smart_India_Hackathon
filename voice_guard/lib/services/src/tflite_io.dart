import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../../utils/audio_processor.dart';

class TFLiteService {
  Interpreter? _interpreter;
  bool _ready = false;
  List<int>? _inputShape;
  bool get isReady => _ready;

  Future<void> init() async {
    try {
      _interpreter = await Interpreter.fromAsset('models/voice_detector.tflite');
      _inputShape = _interpreter!.getInputTensor(0).shape;
      _ready = true;
      debugPrint('TFLite model loaded, input shape: $_inputShape');
    } catch (e) {
      debugPrint('TFLite model not found — using heuristic scorer: $e');
      _ready = false;
    }
  }

  double infer(List<double> lfcc, List<double> prosody) {
    if (_ready && _interpreter != null) {
      try {
        final input = [...lfcc, ...prosody];
        final dim = _inputShape!.last;
        final trimmed = input.sublist(0, math.min(dim, input.length));
        while (trimmed.length < dim) { trimmed.add(0); }
        final inputTensor = [trimmed.map((e) => e.toDouble()).toList()];
        final output = List.filled(1 * 2, 0.0).reshape([1, 2]);
        _interpreter!.run(inputTensor, output);
        final logits = (output[0] as List<double>);
        final maxL = logits.reduce(math.max);
        final exps = logits.map((v) => math.exp(v - maxL)).toList();
        final sum = exps.reduce((a, b) => a + b);
        final probs = exps.map((e) => e / sum).toList();
        return probs[1].clamp(0.0, 1.0);
      } catch (e) {
        debugPrint('TFLite inference failed: $e');
      }
    }
    return _heuristic(lfcc, prosody);
  }

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

  double scoreChunk(List<double> pcm) {
    final lfcc = AudioProcessor.extractLfcc(pcm);
    final prosody = AudioProcessor.extractProsody(pcm);
    return infer(lfcc, prosody);
  }

  void dispose() { _interpreter?.close(); }
}
