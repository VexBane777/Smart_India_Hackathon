import 'dart:async';
import 'package:flutter/foundation.dart';
import '../models/call_state.dart';

class CallStateProvider extends ChangeNotifier {
  CallState _state = const CallState();
  Timer? _ticker;

  CallState get state => _state;

  void setStatus(CallStatus s, {String? number}) {
    if (s == CallStatus.active) {
      _state = CallState(status: s, number: number ?? _state.number, startedAt: DateTime.now());
      _startTicker();
    } else if (s == CallStatus.idle || s == CallStatus.disconnected) {
      _stopTicker();
      _state = CallState(status: s, number: _state.number);
      if (s == CallStatus.disconnected) {
        // briefly hold disconnected then reset to idle
        Future.delayed(const Duration(seconds: 2), () {
          _state = const CallState();
          notifyListeners();
        });
      }
    } else {
      _state = _state.copyWith(status: s, number: number);
    }
    notifyListeners();
  }

  void _startTicker() {
    _ticker?.cancel();
    _ticker = Timer.periodic(const Duration(seconds: 1), (_) {
      if (_state.startedAt != null) {
        _state = _state.copyWith(elapsed: DateTime.now().difference(_state.startedAt!));
        notifyListeners();
      }
    });
  }

  void _stopTicker() {
    _ticker?.cancel();
    _ticker = null;
  }

  @override
  void dispose() {
    _ticker?.cancel();
    super.dispose();
  }

  String get elapsedLabel {
    final d = _state.elapsed;
    final m = d.inMinutes.remainder(60).toString().padLeft(2, '0');
    final s = d.inSeconds.remainder(60).toString().padLeft(2, '0');
    return '$m:$s';
  }
}
