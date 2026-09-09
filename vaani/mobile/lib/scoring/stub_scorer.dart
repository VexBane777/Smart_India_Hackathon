import 'dart:math';
import 'dart:typed_data';
import 'scorer.dart';

/// Ported from app/engine_mock.py's MockBackend. `cloneEntryS == null`
/// means "no known clone position in this audio" — the honest default for
/// arbitrary user-imported files, which always scores low. Passing a
/// value reproduces the canned clone-entry-at-t demo behavior.
class StubScorer implements Scorer {
  StubScorer({this.cloneEntryS, int seed = 7}) : _rng = _SeededGaussian(seed);

  final double? cloneEntryS;
  final _SeededGaussian _rng;

  @override
  String get backendLabel => 'stub (simulated — real model pending)';

  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    final mid = tStartS + audio.length / sr / 2.0;
    double base;
    if (cloneEntryS == null) {
      base = 0.12;
    } else if (mid >= cloneEntryS!) {
      base = 0.85;
    } else if (mid >= cloneEntryS! - 0.25) {
      base = 0.55;
    } else {
      base = 0.12;
    }
    final sampled = _rng.next(base, 0.05);
    return sampled.clamp(0.0, 1.0);
  }
}

/// Deterministic seeded Gaussian sampler (Box-Muller), since dart:math's
/// Random has no built-in normal distribution. Determinism only needs to
/// hold within one StubScorer instance's lifetime (tests seed explicitly);
/// exact parity with numpy's RNG stream is not required.
class _SeededGaussian {
  _SeededGaussian(int seed) : _random = Random(seed);
  final Random _random;
  double? _spare;

  double next(double mean, double stdDev) {
    if (_spare != null) {
      final v = _spare!;
      _spare = null;
      return mean + stdDev * v;
    }
    double u, v, s;
    do {
      u = _random.nextDouble() * 2 - 1;
      v = _random.nextDouble() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);
    final mul = sqrt(-2.0 * log(s) / s);
    _spare = v * mul;
    return mean + stdDev * (u * mul);
  }
}
