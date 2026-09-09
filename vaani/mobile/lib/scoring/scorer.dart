import 'dart:typed_data';

/// One call site in WindowPipeline (Task 9) chooses which implementation
/// to use — StubScorer (Phase 1) or OnnxScorer (Phase 2, Task 15). Both
/// implement this same interface, so swapping is a one-line change.
abstract class Scorer {
  double scoreWindow(Float32List audio, int sr, double tStartS);

  /// Shown verbatim in the UI (gauge banner) so the app never implies real
  /// inference while it isn't running one. Honest numbers only.
  String get backendLabel;
}
