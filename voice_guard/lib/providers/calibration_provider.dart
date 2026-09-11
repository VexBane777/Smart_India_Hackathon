import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Persists a per-device "voice baseline" (mean/std of this user's own raw
/// model scores, captured via a short calibration recording) and turns it
/// into an adjusted alert threshold. See
/// voice_guard/docs/CRITICAL-entity-vs-style-confound.md §4, item 1: this
/// does NOT change what the model outputs (no feature/model changes at
/// all) — it only shifts *where the alert line is drawn* for this specific
/// speaker, so a consistently low-or-high-variance talker isn't judged
/// against a population-wide threshold that doesn't fit them.
class CalibrationProvider extends ChangeNotifier {
  // Approximates the population mean of the *current* deployed model's raw
  // score on genuine, ambient-noise-present human speech. Matches
  // RiskScoreProvider's own pre-calibration default EMA seed
  // (risk_score_provider.dart:22, `ema ?? 0.15`) deliberately, so
  // calibration and the rest of the app agree on what "typical real
  // speech" looks like absent any per-speaker information. If the shipped
  // model is retrained (e.g. voice_guard_v9_noisefix_final or later), this
  // constant should be re-derived from that model's own mean raw score on
  // a held-out real-speech set — it is a property of the model, not of
  // this calibration mechanism.
  static const double populationMean = 0.15;
  static const double minThreshold = 0.35;
  static const double maxThreshold = 0.90;

  double _baselineMean = 0.0;
  double _baselineStd = 0.0;
  bool _isCalibrated = false;

  double get baselineMean => _baselineMean;
  double get baselineStd => _baselineStd;
  bool get isCalibrated => _isCalibrated;

  Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    _baselineMean = p.getDouble('calibration_baselineMean') ?? 0.0;
    _baselineStd = p.getDouble('calibration_baselineStd') ?? 0.0;
    _isCalibrated = p.getBool('calibration_isCalibrated') ?? false;
    notifyListeners();
  }

  Future<void> setBaseline(double mean, double std) async {
    _baselineMean = mean;
    _baselineStd = std;
    _isCalibrated = true;
    final p = await SharedPreferences.getInstance();
    await p.setDouble('calibration_baselineMean', mean);
    await p.setDouble('calibration_baselineStd', std);
    await p.setBool('calibration_isCalibrated', true);
    notifyListeners();
  }

  Future<void> clearBaseline() async {
    _baselineMean = 0.0;
    _baselineStd = 0.0;
    _isCalibrated = false;
    final p = await SharedPreferences.getInstance();
    await p.remove('calibration_baselineMean');
    await p.remove('calibration_baselineStd');
    await p.setBool('calibration_isCalibrated', false);
    notifyListeners();
  }

  /// Instance convenience wrapper around the static, directly-testable core.
  double effectiveThreshold(double sensitivity) => computeThreshold(
        sensitivity: sensitivity,
        isCalibrated: _isCalibrated,
        baselineMean: _baselineMean,
        populationMean: populationMean,
      );

  /// Pure function, no I/O — deliberately static so it's testable without
  /// constructing SharedPreferences. Shifts the alert threshold by exactly
  /// how far this speaker's own baseline sits from the assumed population
  /// mean, so a speaker whose baseline is naturally elevated needs a
  /// proportionally higher raw/EMA score to trip an alert, and vice versa.
  /// This does NOT rescale the score itself (see plan header) — only where
  /// the line is drawn — deliberately, so raw/EMA scores stay comparable
  /// across the app (logs, UI, `RiskScoreProvider.history`) regardless of
  /// calibration state.
  static double computeThreshold({
    required double sensitivity,
    required bool isCalibrated,
    required double baselineMean,
    required double populationMean,
  }) {
    if (!isCalibrated) return sensitivity;
    final adjusted = sensitivity + (baselineMean - populationMean);
    return adjusted.clamp(minThreshold, maxThreshold);
  }
}
