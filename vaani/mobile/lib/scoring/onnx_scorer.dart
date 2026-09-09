import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:onnxruntime/onnxruntime.dart';
import 'scorer.dart';

/// Phase 2 real-inference scorer. Loads a bundled ONNX model asset and
/// runs it against MelBridge's mel-spectrogram output. This is the single
/// swap point named in the design spec (§2, §6): everywhere else in the
/// app depends only on the `Scorer` interface, never on this class
/// directly, so replacing StubScorer with OnnxScorer at HomeScreen's one
/// call site is the entire Phase 2 rollout.
///
/// NOTE for whoever wires in the real TinyCNN export: verify its expected
/// input tensor shape (n_mels, hop, n_fft) matches MelBridge's (48, 160,
/// 512) before assuming this class's mel input is compatible as-is — the
/// spec explicitly flags this as unverified until a real model exists.
/// The zero-filled [1, 48, 10] tensor below is a plumbing placeholder
/// only, not a real feature pipeline (see plan Task 15, Step 6).
class OnnxScorer implements Scorer {
  OnnxScorer({required this.modelAssetPath});

  final String modelAssetPath;
  OrtSession? _session;

  @override
  String get backendLabel => 'onnx (on-device)';

  Future<void> load() async {
    OrtEnv.instance.init();
    final rawBytes = await _loadAsset(modelAssetPath);
    _session = OrtSession.fromBuffer(rawBytes, OrtSessionOptions());
  }

  Future<Uint8List> _loadAsset(String path) async {
    final data = await rootBundle.load(path);
    return data.buffer.asUint8List();
  }

  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    final session = _session;
    if (session == null) {
      throw StateError('OnnxScorer.load() must be awaited before scoreWindow()');
    }
    // Mel extraction happens in WindowPipeline via MelBridge; real Phase 2
    // wiring (feeding MelBridge's actual mel output into this tensor,
    // confirming shape compatibility with Module B's real exported model)
    // is follow-up work for whoever picks this up once Module B ships —
    // this identity-model test only exercises the ONNX plumbing with a
    // placeholder mel window of zeros shaped [1, 48, 10].
    final inputData = Float32List(48 * 10);
    final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, [1, 48, 10]);
    final outputs = session.run(OrtRunOptions(), {'mel': inputTensor});
    final result = (outputs.first?.value as List).first as double;
    inputTensor.release();
    return result.clamp(0.0, 1.0);
  }
}
