import 'dart:async';
import 'dart:typed_data';
import '../decision/decision_engine.dart';
import '../mel/mel_bridge.dart';
import '../scoring/scorer.dart';
import 'frame_result.dart';
export 'frame_result.dart';

const int _sampleRate = 16000;
const int _windowSamples = _sampleRate * 2; // 2.0s

/// Assembles 0.5s hop chunks (from file import or mic capture) into a 2s
/// sliding window, scores each completed window, and threads the result
/// through DecisionEngine. Mirrors server.py's "only score full windows...
/// first-score latency of ~2.5s" behavior.
class WindowPipeline {
  WindowPipeline({required MelBridgeLike mel}) : _mel = mel;

  final MelBridgeLike _mel;
  final List<double> _buffer = [];
  final DecisionEngine _engine = DecisionEngine();
  double _elapsedS = 0.0;
  bool _first = true;

  Stream<FrameResult> process(Stream<Float32List> hopChunks,
      {required Scorer scorer}) async* {
    await for (final chunk in hopChunks) {
      _buffer.addAll(chunk);
      if (_buffer.length < _windowSamples) continue;

      final window = Float32List.fromList(
          _buffer.sublist(_buffer.length - _windowSamples));
      final tStart = _first ? 0.0 : _elapsedS - 1.5; // window start = now - 2s + 0.5s hop already counted
      final raw = scorer.scoreWindow(window, _sampleRate, tStart);
      final state = _engine.update(raw);
      final melDb = await _mel.computeMelDb(window, _sampleRate);

      yield FrameResult(
        t: _elapsedS,
        rawScore: raw,
        ema: _engine.ema!,
        state: state,
        melDb: melDb,
        audioWindow: window,
      );

      _elapsedS += 0.5;
      _first = false;
      // Keep only the last window's worth of samples to bound memory.
      if (_buffer.length > _windowSamples) {
        _buffer.removeRange(0, _buffer.length - _windowSamples);
      }
    }
  }
}
