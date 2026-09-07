import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:provider/provider.dart';
import 'package:flutter/material.dart';
import '../providers/risk_score_provider.dart';
import '../providers/settings_provider.dart';
import 'call_service.dart';
import 'audio_service.dart';
import 'tflite_service.dart';
import 'notification_service.dart';

/// Binds native audio stream → LFCC/prosody → TFLite → risk score → UI + alerts.
/// Raw PCM is processed in RAM and discarded after feature extraction.
class AudioPipeline {
  final BuildContext context;
  final CallService calls;
  final TFLiteService tflite;
  final AudioService audio;
  final NotificationService notifications;

  StreamSubscription<Uint8List>? _sub;
  StreamSubscription<double>? _scoreSub;
  bool _running = false;

  AudioPipeline({
    required this.context,
    required this.calls,
    required this.tflite,
    required this.audio,
    required this.notifications,
  });

  Future<void> start() async {
    if (_running) return;
    _running = true;
    final settings = context.read<SettingsProvider>();
    if (!settings.protectionEnabled) return;

    audio.startScoring();
    _scoreSub = audio.scoreStream.listen((raw) async {
      if (!context.mounted) return;
      final riskProvider = context.read<RiskScoreProvider>();
      final settings = context.read<SettingsProvider>();
      riskProvider.update(raw, alertThreshold: settings.sensitivity);
      final cur = riskProvider.current;
      if (cur == null) return;
      if (!context.mounted) return;
      // Gate on the shared EMA + 2-consecutive-window alert state (master
      // plan §6), not a single sample crossing threshold — matches the
      // decision logic in vaani/app/engine_mock.py's AlertStateMachine.
      if (riskProvider.isAlert) {
        // cur.label is derived from RiskScore's own fixed 0.30/0.70 bands,
        // which don't track settings.sensitivity (the alertThreshold above)
        // — an alert can fire at ema=0.65 while cur.label still says
        // "SUSPICIOUS". The fact that an alert fired at all means the verdict
        // is AI DETECTED by definition, so say that instead of cur.label.
        const alertVerdict = 'AI DETECTED';
        if (settings.overlayEnabled) {
          try { await calls.showOverlay(riskScore: cur.score, verdict: alertVerdict); } catch (_) {}
        }
        if (!context.mounted) return;
        if (settings.soundEnabled) {
          try { await notifications.showRiskAlert(score: cur.score, verdict: alertVerdict); } catch (_) {}
        }
      }
    });

    // Try native stream; fall back to mock if plugin missing (web/desktop demo)
    try {
      _sub = calls.audioStream.listen(
        (bytes) => audio.ingestBytes(bytes),
        onError: (_) => _startMock(),
        cancelOnError: false,
      );
      // kick native capture
      try { await calls.startCallDetection(); } catch (_) {}
      // native stream established — mock fallback only via onError
    } catch (_) {
      _startMock();
    }
    debugPrint('AudioPipeline started');
  }

  void _startMock() {
    debugPrint('AudioPipeline: native stream unavailable — mock data not used for real calls (demo only via Live Call screen)');
  }

  Future<void> stop() async {
    _running = false;
    await _sub?.cancel(); _sub = null;
    await _scoreSub?.cancel(); _scoreSub = null;
    audio.stopScoring();
    try { await calls.stopCallDetection(); } catch (_) {}
    try { await calls.hideOverlay(); } catch (_) {}
    debugPrint('AudioPipeline stopped');
  }
}
