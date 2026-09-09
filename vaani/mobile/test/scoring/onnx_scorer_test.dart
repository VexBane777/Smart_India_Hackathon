import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/scoring/onnx_scorer.dart';
import 'package:mobile/scoring/scorer.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('OnnxScorer implements the same Scorer interface as StubScorer', () {
    final scorer = OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx');
    expect(scorer, isA<Scorer>());
    expect(scorer.backendLabel, isNot(contains('simulated')));
  });

  test('scoreWindow returns a value in [0, 1] for the dummy identity model',
      () async {
    final scorer = OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx');
    await scorer.load();
    final audio = Float32List(32000); // 2s @ 16kHz, silence
    final score = scorer.scoreWindow(audio, 16000, 0.0);
    expect(score, inInclusiveRange(0.0, 1.0));
  });

  test('swapping HomeScreen from StubScorer to OnnxScorer requires only the constructor call',
      () {
    // Documents the swap-point contract (spec §2/§6): both scorers satisfy
    // `Scorer`, so any code written against the interface (WindowPipeline,
    // Task 9) needs no changes when the concrete type changes.
    Scorer makeScorer(bool usePhase2) => usePhase2
        ? OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx')
        : StubScorerForContractCheck();
    expect(makeScorer(true), isA<Scorer>());
    expect(makeScorer(false), isA<Scorer>());
  });
}

class StubScorerForContractCheck implements Scorer {
  @override
  String get backendLabel => 'stub';
  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) => 0.0;
}
