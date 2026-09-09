import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/mel/mel_bridge.dart';
import 'package:mobile/pipeline/window_pipeline.dart';
import 'package:mobile/scoring/scorer.dart';

class _FakeScorer implements Scorer {
  final List<double> calls = [];
  @override
  String get backendLabel => 'fake';
  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    calls.add(tStartS);
    return 0.9;
  }
}

class _FakeMel implements MelBridgeLike {
  @override
  Future<List<List<double>>> computeMelDb(Float32List pcm, int sr) async => [
        [0.0]
      ];
}

void main() {
  test('first FrameResult only fires after 2s (4 hop chunks) accumulate', () async {
    final scorer = _FakeScorer();
    final pipeline = WindowPipeline(mel: _FakeMel());
    final controller = StreamController<Float32List>();
    final results = <FrameResult>[];
    final sub = pipeline.process(controller.stream, scorer: scorer).listen(results.add);

    for (var i = 0; i < 3; i++) {
      controller.add(Float32List(8000)); // 0.5s each, sr assumed 16000
      await Future.delayed(Duration.zero);
    }
    expect(results, isEmpty, reason: 'only 1.5s buffered so far');

    controller.add(Float32List(8000)); // 4th chunk -> 2.0s
    await Future.delayed(Duration.zero);
    expect(results.length, 1);
    expect(scorer.calls.single, closeTo(0.0, 1e-9));

    await controller.close();
    await sub.cancel();
  });

  test('state machine and ema are threaded across windows', () async {
    final pipeline = WindowPipeline(mel: _FakeMel());
    final controller = StreamController<Float32List>();
    final results = <FrameResult>[];
    final sub = pipeline
        .process(controller.stream, scorer: _FakeScorer())
        .listen(results.add);

    for (var i = 0; i < 8; i++) {
      controller.add(Float32List(8000));
      await Future.delayed(Duration.zero);
    }
    expect(results.length, 5); // chunks 4,5,6,7,8 each complete a new window
    expect(results.last.state.toString(), contains('alert')); // score 0.9 repeatedly
    await controller.close();
    await sub.cancel();
  });
}
