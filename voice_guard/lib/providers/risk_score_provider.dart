import 'package:flutter/foundation.dart';
import '../models/risk_score.dart';

class RiskScoreProvider extends ChangeNotifier {
  RiskScore? _current;
  final List<RiskScore> _history = [];
  double _ema = 0.15; // smoothed score
  static const double _alpha = 0.35;

  RiskScore? get current => _current;
  List<RiskScore> get history => List.unmodifiable(_history);
  double get ema => _ema;

  void update(double rawScore, {List<double>? prosody}) {
    // EMA smoothing to prevent UI flicker
    _ema = _alpha * rawScore + (1 - _alpha) * _ema;
    final rs = RiskScore(score: _ema, timestamp: DateTime.now(), prosody: prosody);
    _current = rs;
    _history.add(rs);
    if (_history.length > 200) _history.removeAt(0);
    notifyListeners();
  }

  void reset() {
    _current = null;
    _ema = 0.15;
    // keep history for logs/chart
    notifyListeners();
  }

  void clearHistory() {
    _history.clear();
    _current = null;
    _ema = 0.15;
    notifyListeners();
  }
}
