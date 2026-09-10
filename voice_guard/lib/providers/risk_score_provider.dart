import 'package:flutter/foundation.dart';
import '../models/risk_score.dart';
import '../models/call_log.dart';

class RiskScoreProvider extends ChangeNotifier {
  RiskScore? _current;
  final List<RiskScore> _history = [];
  double? _ema; // smoothed score; null until the first update() seeds it
  int _consecutiveHigh = 0;
  String _state = 'normal'; // normal | warn | alert
  bool _hasSignal = true;
  String? _captureSource; // e.g. "VOICE_CALL", "MIC" — from getCaptureStatus

  // Mirrors vaani/app/engine_mock.py's AlertStateMachine (master plan §6):
  // EMA smoothing, then "alert" only after 2+ consecutive windows over
  // threshold — a single spike must never fire an overlay/notification.
  static const double _alpha = 0.7;
  static const int _consecutiveRequired = 2;

  RiskScore? get current => _current;
  List<RiskScore> get history => List.unmodifiable(_history);
  double get ema => _ema ?? 0.15;
  String get state => _state;
  bool get isAlert => _state == 'alert';
  bool get hasSignal => _hasSignal;
  String? get captureSource => _captureSource;
  // AudioCaptureManager's SOURCE_CASCADE tries VOICE_CALL first; it only wins
  // when CAPTURE_AUDIO_OUTPUT is actually granted, which only happens on the
  // rooted/privileged-flavor install path (see magisk-privileged-module/).
  // Everyone else falls through to VOICE_RECOGNITION/MIC.
  bool get isPrivilegedCapture => _captureSource == 'VOICE_CALL';

  void setHasSignal(bool v) {
    if (v == _hasSignal) return;
    _hasSignal = v;
    notifyListeners();
  }

  void setCaptureSource(String? source) {
    if (source == _captureSource) return;
    _captureSource = source;
    notifyListeners();
  }

  void update(double rawScore, {List<double>? prosody, double alertThreshold = 0.6}) {
    // EMA smoothing to prevent UI flicker. Seed from the first raw score
    // (matches engine_mock.py's AlertStateMachine.update) instead of
    // blending it against an arbitrary starting value.
    _ema = _ema == null ? rawScore : _alpha * rawScore + (1 - _alpha) * _ema!;
    if (_ema! >= alertThreshold) {
      _consecutiveHigh++;
    } else {
      _consecutiveHigh = 0;
    }
    _state = _consecutiveHigh >= _consecutiveRequired
        ? 'alert'
        : (_consecutiveHigh > 0 ? 'warn' : 'normal');
    final rs = RiskScore(score: _ema!, timestamp: DateTime.now(), prosody: prosody);
    _current = rs;
    _history.add(rs);
    if (_history.length > 200) _history.removeAt(0);
    notifyListeners();
  }

  void reset() {
    _current = null;
    _ema = null;
    _consecutiveHigh = 0;
    _state = 'normal';
    _hasSignal = true;
    // keep history for logs/chart
    notifyListeners();
  }

  final List<CallLog> _callLogs = [
    CallLog(
      id: 'log-001',
      timestamp: DateTime.now().subtract(const Duration(hours: 2, minutes: 12)),
      number: '+91 98450 12891',
      riskScore: 0.94,
      verdict: Verdict.detected,
      duration: const Duration(minutes: 1, seconds: 14),
      recordingPath: '/sandboxed/recordings/call_20260910_12891_threat.wav',
    ),
    CallLog(
      id: 'log-002',
      timestamp: DateTime.now().subtract(const Duration(hours: 5, minutes: 40)),
      number: '+91 80234 56789',
      riskScore: 0.08,
      verdict: Verdict.verified,
      duration: const Duration(minutes: 4, seconds: 32),
      recordingPath: '/sandboxed/recordings/call_20260910_56789_clean.wav',
    ),
    CallLog(
      id: 'log-003',
      timestamp: DateTime.now().subtract(const Duration(days: 1, hours: 3)),
      number: 'WhatsApp Audio (VoIP)',
      riskScore: 0.72,
      verdict: Verdict.detected,
      duration: const Duration(seconds: 48),
      recordingPath: '/sandboxed/recordings/voip_whatsapp_clone.wav',
    ),
    CallLog(
      id: 'log-004',
      timestamp: DateTime.now().subtract(const Duration(days: 1, hours: 8)),
      number: '+91 91234 56780',
      riskScore: 0.42,
      verdict: Verdict.suspicious,
      duration: const Duration(minutes: 2, seconds: 15),
      recordingPath: null,
    ),
    CallLog(
      id: 'log-005',
      timestamp: DateTime.now().subtract(const Duration(days: 2, hours: 1)),
      number: '+91 98765 43210',
      riskScore: 0.06,
      verdict: Verdict.verified,
      duration: const Duration(minutes: 5, seconds: 20),
      recordingPath: null,
    ),
    CallLog(
      id: 'log-006',
      timestamp: DateTime.now().subtract(const Duration(days: 3)),
      number: 'Live Acoustic Mic Test',
      riskScore: 0.04,
      verdict: Verdict.verified,
      duration: const Duration(seconds: 25),
      recordingPath: null,
    ),
  ];
  List<CallLog> get callLogs => List.unmodifiable(_callLogs);

  void addCallLog(CallLog log) {
    _callLogs.insert(0, log);
    notifyListeners();
  }

  void clearHistory() {
    _history.clear();
    _callLogs.clear();
    _current = null;
    _ema = null;
    _consecutiveHigh = 0;
    _state = 'normal';
    notifyListeners();
  }
}
