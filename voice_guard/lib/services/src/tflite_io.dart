import 'dart:math' as math;
import 'package:flutter/foundation.dart';
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';
import '../../utils/audio_processor.dart';

/// Runs the exported ONNX model (see model_training/README.md's "Feature
/// contract" — 66-float [...lfcc, ...prosody, ...physio] in, (1,2) raw-logit "logits"
/// out, softmax applied here) via flutter_onnxruntime. Replaces the
/// previous tflite_flutter path; kept this file's name and the
/// TFLiteService class name to avoid touching every call site
/// (audio_service.dart, main.dart, providers) for what's purely an
/// inference-backend swap, not an API change — infer()/scoreChunk() did
/// have to become async (ONNX Runtime's session.run is Future-based),
/// which is why TFLiteServiceBase and the web stub changed too.
class TFLiteService {
  OrtSession? _session;
  bool _ready = false;
  bool get isReady => _ready;

  static const int nLfcc = AudioProcessor.nLfcc;

  Future<void> init() async {
    final ort = OnnxRuntime();
    try {
      _session = await ort.createSessionFromAsset('assets/models/voice_detector.onnx');
      _ready = true;
      debugPrint('ONNX model loaded from assets/models/voice_detector.onnx');
    } catch (e) {
      debugPrint('ONNX model not found — using heuristic scorer: $e');
      _ready = false;
    }
  }

  Future<(double, String?, double)> infer(List<List<double>> lfccSequence, List<double> scalars) async {
    final session = _session;
    if (_ready && session != null) {
      try {
        final flatSeq = lfccSequence.expand((frame) => frame).toList();
        final seqInput = await OrtValue.fromList(
          flatSeq.map((e) => e.toDouble()).toList(),
          [1, lfccSequence.length, nLfcc],
        );
        final scalarInput = await OrtValue.fromList(
          scalars.map((e) => e.toDouble()).toList(),
          [1, scalars.length],
        );
        final outputs = await session.run({'lfcc_sequence': seqInput, 'scalars': scalarInput});

        final realFakeOut = (await outputs['real_fake_logits']!.asFlattenedList())
            .map((e) => (e as num).toDouble()).toList();
        final attackOut = (await outputs['attack_type_logits']!.asFlattenedList())
            .map((e) => (e as num).toDouble()).toList();

        final realFakeProb = _softmaxSecond(realFakeOut);
        final attackProbs = _softmax(attackOut);
        final attackLabel = attackProbs[1] > attackProbs[0] ? 'vc' : 'tts';
        final attackConfidence = attackProbs.reduce(math.max);

        return (realFakeProb, attackLabel, attackConfidence);
      } catch (e) {
        debugPrint('ONNX inference failed: $e');
      }
    }
    final heuristic = _heuristic(lfccSequence, scalars);
    return (heuristic, null, 0.0); // heuristic fallback never claims an attack type
  }

  double _softmaxSecond(List<double> logits) {
    final probs = _softmax(logits);
    return probs[1].clamp(0.0, 1.0);
  }

  List<double> _softmax(List<double> logits) {
    final maxL = logits.reduce(math.max);
    final exps = logits.map((v) => math.exp(v - maxL)).toList();
    final sum = exps.reduce((a, b) => a + b);
    return exps.map((e) => e / sum).toList();
  }

  double _heuristic(List<List<double>> lfccSequence, List<double> scalars) {
    if (lfccSequence.isEmpty) return 0.15;
    final pooled = List<double>.filled(nLfcc, 0);
    for (final frame in lfccSequence) {
      for (int i = 0; i < nLfcc; i++) { pooled[i] += frame[i]; }
    }
    for (int i = 0; i < nLfcc; i++) { pooled[i] /= lfccSequence.length; }
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

  void dispose() { _session?.close(); }
}
