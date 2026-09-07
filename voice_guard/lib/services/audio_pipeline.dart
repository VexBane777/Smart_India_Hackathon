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
      riskProvider.update(raw);
      final cur = riskProvider.current;
      if (cur == null) return;
      if (!context.mounted) return;
      final settings = context.read<SettingsProvider>();
      final threshold = settings.sensitivity;
      if (cur.score > threshold) {
        if (settings.overlayEnabled) {
          try { await calls.showOverlay(riskScore: cur.score, verdict: cur.label); } catch (_) {}
        }
        if (!context.mounted) return;
        if (settings.soundEnabled) {
          try { await notifications.showRiskAlert(score: cur.score, verdict: cur.label); } catch (_) {}
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
