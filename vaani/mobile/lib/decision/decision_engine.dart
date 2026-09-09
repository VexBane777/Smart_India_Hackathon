/// Ported from app/engine_mock.py's AlertStateMachine (master plan §6):
/// per-window score -> EMA smoothing -> alert only after 2+ consecutive
/// windows stay above threshold. Constants match the Python original
/// exactly so both platforms behave identically.
enum AlertState { normal, warn, alert }

class DecisionEngine {
  DecisionEngine({
    this.threshold = 0.6,
    this.consecutiveRequired = 2,
    this.emaAlpha = 0.7,
  });

  final double threshold;
  final int consecutiveRequired;
  final double emaAlpha;

  double? ema;
  int _consecutiveHigh = 0;

  AlertState update(double rawScore) {
    ema = ema == null ? rawScore : emaAlpha * rawScore + (1 - emaAlpha) * ema!;
    if (ema! >= threshold) {
      _consecutiveHigh += 1;
    } else {
      _consecutiveHigh = 0;
    }
    if (_consecutiveHigh >= consecutiveRequired) return AlertState.alert;
    if (_consecutiveHigh > 0) return AlertState.warn;
    return AlertState.normal;
  }
}
