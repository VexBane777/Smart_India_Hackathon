import 'package:flutter/foundation.dart';
import '../models/risk_score.dart';

class RiskScoreProvider extends ChangeNotifier {
  RiskScore? _current;
  final List<RiskScore> _history = [];
  double _ema = 0.15; // smoothed score
  int _consecutiveHigh = 0;
  String _state = 'normal'; // normal | warn | alert

  // Mirrors vaani/app/engine_mock.py's AlertStateMachine (master plan §6):
  // EMA smoothing, then "alert" only after 2+ consecutive windows over
  // threshold — a single spike must never fire an overlay/notification.
  static const double _alpha = 0.7;
  static const int _consecutiveRequired = 2;

  RiskScore? get current => _current;
  List<RiskScore> get history => List.unmodifiable(_history);
  double get ema => _ema;
  String get state => _state;
  bool get isAlert => _state == 'alert';

  void update(double rawScore, {List<double>? prosody, double alertThreshold = 0.6}) {
    // EMA smoothing to prevent UI flicker
    _ema = _alpha * rawScore + (1 - _alpha) * _ema;
    if (_ema >= alertThreshold) {
      _consecutiveHigh++;
    } else {
      _consecutiveHigh = 0;
    }
    _state = _consecutiveHigh >= _consecutiveRequired
        ? 'alert'
        : (_consecutiveHigh > 0 ? 'warn' : 'normal');
    final rs = RiskScore(score: _ema, timestamp: DateTime.now(), prosody: prosody);
    _current = rs;
    _history.add(rs);
    if (_history.length > 200) _history.removeAt(0);
    notifyListeners();
  }

  void reset() {
    _current = null;
    _ema = 0.15;
    _consecutiveHigh = 0;
    _state = 'normal';
    // keep history for logs/chart
    notifyListeners();
  }

  void clearHistory() {
    _history.clear();
    _current = null;
    _ema = 0.15;
    _consecutiveHigh = 0;
    _state = 'normal';
    notifyListeners();
  }
}
