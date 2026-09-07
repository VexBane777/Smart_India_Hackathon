import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../providers/settings_provider.dart';
import '../widgets/risk_meter.dart';
import '../widgets/waveform_visualizer.dart';
import '../utils/constants.dart';
import '../models/call_state.dart';

class CallScreen extends StatefulWidget {
  const CallScreen({super.key});
  @override
  State<CallScreen> createState() => _CallScreenState();
}

class _CallScreenState extends State<CallScreen> {
  Timer? _demoTimer;
  double _demoScore = 0.12;

  @override
  void initState() {
    super.initState();
    // Demo mode: animate risk for judges without a real call
    _demoTimer = Timer.periodic(const Duration(milliseconds: 900), (_) {
      if (!mounted) return;
      final callActive = context.read<CallStateProvider>().state.isActive;
      if (!callActive) return;
      // oscillate demo
      _demoScore = (_demoScore + 0.07) % 1.0;
      // occasionally spike to red for demo
      final spike = (_demoScore > 0.85) ? 0.92 : _demoScore;
      context.read<RiskScoreProvider>().update(spike.clamp(0.05, 0.95));
      final threshold = context.read<SettingsProvider>().sensitivity;
      if (spike > threshold && context.read<SettingsProvider>().overlayEnabled) {
        // overlay would be triggered via CallService in real flow
      }
    });
  }

  @override
  void dispose() {
    _demoTimer?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final call = context.watch<CallStateProvider>();
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.12;
    final isActive = call.state.isActive;
    final verdict = risk?.label ?? AppConstants.verdictFor(score);
    final color = risk?.color ?? AppConstants.colorFor(score);

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text('Live Call', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
        actions: [
          if (!isActive)
            Padding(
              padding: const EdgeInsets.only(right: 12),
              child: FilledButton.icon(
                onPressed: () {
                  context.read<CallStateProvider>().setStatus(CallStatus.active, number: '+91 98XXXX XX10');
                  context.read<RiskScoreProvider>().reset();
                },
                icon: const Icon(Icons.play_arrow, size: 18),
                label: const Text('Demo Call', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
                style: FilledButton.styleFrom(backgroundColor: AppColors.primary, padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8)),
              ),
            ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Caller card
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 12)]),
            child: Row(children: [
              CircleAvatar(radius: 26, backgroundColor: AppColors.primary.withValues(alpha: 0.12), child: const Icon(Icons.person, color: AppColors.primary)),
              const SizedBox(width: 12),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(call.state.number ?? 'No active call', style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800)),
                const SizedBox(height: 2),
                Text(isActive ? 'Ongoing • ${call.elapsedLabel}' : 'Tap "Demo Call" to simulate a live call',
                    style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.6))),
              ])),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(color: isActive ? AppColors.verifiedBg : const Color(0xFFEEEEEE), borderRadius: BorderRadius.circular(999)),
                child: Row(mainAxisSize: MainAxisSize.min, children: [
                  Container(width: 7, height: 7, decoration: BoxDecoration(color: isActive ? AppColors.verified : Colors.grey, shape: BoxShape.circle)),
                  const SizedBox(width: 6),
                  Text(isActive ? 'ACTIVE' : 'IDLE', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, color: isActive ? AppColors.verified : Colors.black54)),
                ]),
              ),
            ]),
          ),
          const SizedBox(height: 14),
          RiskMeter(score: score),
          const SizedBox(height: 14),
          // Verdict banner
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: color.withValues(alpha: 0.10), borderRadius: BorderRadius.circular(12), border: Border.all(color: color.withValues(alpha: 0.25))),
            child: Row(children: [
              Icon(verdict == 'AI DETECTED' ? Icons.warning_rounded : verdict == 'SUSPICIOUS' ? Icons.error_outline : Icons.verified_user_rounded, color: color),
              const SizedBox(width: 10),
              Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                Text(verdict, style: TextStyle(fontSize: 13, fontWeight: FontWeight.w800, color: color)),
                const SizedBox(height: 2),
                Text(_adviceFor(verdict), style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.65), height: 1.3)),
              ])),
            ]),
          ),
          const SizedBox(height: 14),
          // Waveform
          Container(
            padding: const EdgeInsets.all(14),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Row(children: [
                const Text('Live Audio', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
                const Spacer(),
                Text(isActive ? '16 kHz • LFCC 60 • Prosody' : 'Idle — demo animates on Active',
                    style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.5))),
              ]),
              const SizedBox(height: 10),
              WaveformVisualizer(color: color),
              const SizedBox(height: 8),
              Text('Features extracted on-device every 1s (3s window) → TFLite risk score. Raw PCM discarded after inference.',
                  style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.5), height: 1.3)),
            ]),
          ),
          const SizedBox(height: 14),
          if (isActive)
            SizedBox(
              width: double.infinity,
              child: OutlinedButton.icon(
                onPressed: () {
                  context.read<CallStateProvider>().setStatus(CallStatus.disconnected);
                  Future.delayed(const Duration(milliseconds: 300), () {
                    if (context.mounted) context.read<CallStateProvider>().setStatus(CallStatus.idle);
                  });
                },
                icon: const Icon(Icons.call_end, color: AppColors.detected),
                label: const Text('End Demo Call', style: TextStyle(color: AppColors.detected, fontWeight: FontWeight.w700)),
                style: OutlinedButton.styleFrom(side: const BorderSide(color: AppColors.detected), padding: const EdgeInsets.symmetric(vertical: 12)),
              ),
            ),
          if (score > 0.70)
            Container(
              margin: const EdgeInsets.only(top: 12),
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(color: AppColors.detectedBg, borderRadius: BorderRadius.circular(12), border: Border.all(color: AppColors.detected.withValues(alpha: 0.3))),
              child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                const Row(children: [Icon(Icons.warning_amber_rounded, color: AppColors.detected, size: 18), SizedBox(width: 6), Text('Recommended Action', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: AppColors.detected))]),
                const SizedBox(height: 6),
                Text('• Do NOT share OTP, passwords, or authorize transfers on this call.\n• Hang up and call back on a known number.\n• Escalate to supervisor / trigger MFA via the app.',
                    style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.7), height: 1.4)),
              ]),
            ),
        ],
      ),
    );
  }

  String _adviceFor(String verdict) {
    switch (verdict) {
      case 'AI DETECTED':
        return 'High likelihood of synthetic voice — do not disclose sensitive info. Call back on a verified number.';
      case 'SUSPICIOUS':
        return 'Unusual vocal traits detected — verify identity via a second channel before acting.';
      default:
        return 'Voice traits consistent with a live human speaker at this moment.';
    }
  }
}
