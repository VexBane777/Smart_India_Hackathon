import 'package:flutter/material.dart';
import 'package:flutter_local_notifications/flutter_local_notifications.dart';

/// Secondary alert channel (problem-statement requirement: multi-channel alerting).
/// Fires a local notification + can be extended to call backend /v1/alert.
class NotificationService {
  final _plugin = FlutterLocalNotificationsPlugin();
  bool _inited = false;

  Future<void> init() async {
    const android = AndroidInitializationSettings('@mipmap/ic_launcher');
    const settings = InitializationSettings(android: android);
    await _plugin.initialize(settings);
    _inited = true;
  }

  Future<void> showRiskAlert({required double score, required String verdict, String? caller}) async {
    if (!_inited) await init();
    const details = NotificationDetails(
      android: AndroidNotificationDetails(
        'vaani_alerts', 'Vaani Alerts',
        channelDescription: 'High-risk voice cloning detections',
        importance: Importance.high, priority: Priority.high,
        colorized: true, color: Color(0xFFC62828),
      ),
    );
    await _plugin.show(
      9001, '⚠️ $verdict — ${(score * 100).round()}% risk',
      caller != null ? 'Caller $caller — ${ _advice(verdict)}' : _advice(verdict),
      details,
    );
  }

  String _advice(String v) {
    if (v == 'AI DETECTED') return 'Do NOT share OTP. Call back on a known number.';
    if (v == 'SUSPICIOUS') return 'Verify identity via a second channel.';
    return 'Voice verified — low risk.';
  }
}
