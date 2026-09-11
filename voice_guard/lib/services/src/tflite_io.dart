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

  static const int _inputDim = 66; // must match model_training's INPUT_DIM (model.py)

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

  Future<double> infer(List<double> lfcc, List<double> prosody, List<double> physio) async {
    final session = _session;
    if (_ready && session != null) {
      try {
        final input = [...lfcc, ...prosody, ...physio];
        final trimmed = input.sublist(0, math.min(_inputDim, input.length));
        while (trimmed.length < _inputDim) { trimmed.add(0); }

        final inputValue = await OrtValue.fromList(
          trimmed.map((e) => e.toDouble()).toList(),
          [1, _inputDim],
        );
        final outputs = await session.run({'features': inputValue});
        // Output tensor shape is [1, 2]; asFlattenedList() ignores shape
        // and always returns a plain 1D list, so no reshape needed here.
        final outList = (await outputs['logits']!.asFlattenedList())
            .map((e) => (e as num).toDouble())
            .toList();

        final sumOut = outList[0] + outList[1];
        if ((sumOut - 1.0).abs() < 0.05 && outList[0] >= 0 && outList[1] >= 0) {
          return outList[1].clamp(0.0, 1.0);
        }
        final maxL = outList.reduce(math.max);
        final exps = outList.map((v) => math.exp(v - maxL)).toList();
        final sum = exps.reduce((a, b) => a + b);
        final probs = exps.map((e) => e / sum).toList();
        return probs[1].clamp(0.0, 1.0);
      } catch (e) {
        debugPrint('ONNX inference failed: $e');
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

  Future<double> scoreChunk(List<double> pcm) {
    final lfcc = AudioProcessor.extractLfcc(pcm);
    final prosody = AudioProcessor.extractProsody(pcm);
    final physio = AudioProcessor.extractPhysio(pcm);
    return infer(lfcc, prosody, physio);
  }

  void dispose() { _session?.close(); }
}
