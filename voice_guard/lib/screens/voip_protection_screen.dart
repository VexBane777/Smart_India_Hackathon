import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/playback_capture_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/risk_meter.dart';

class VoipProtectionScreen extends StatefulWidget {
  const VoipProtectionScreen({super.key});
  @override
  State<VoipProtectionScreen> createState() => _VoipProtectionScreenState();
}

class _VoipProtectionScreenState extends State<VoipProtectionScreen> {
  bool _capturing = false;
  StreamSubscription<bool>? _consentSub;

  @override
  void initState() {
    super.initState();
    final capture = context.read<PlaybackCaptureService>();
    _consentSub = capture.consentResult.listen(_onConsent);
  }

  Future<void> _onConsent(bool granted) async {
    if (!granted) return;
    final capture = context.read<PlaybackCaptureService>();
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();
    risk.reset();
    audio.clearBuffer();
    audio.startScoring();
    final started = await capture.startCapture();
    if (mounted) setState(() => _capturing = started);
  }

  Future<void> _start() async {
    final capture = context.read<PlaybackCaptureService>();
    await capture.requestConsent();
    // startCapture() runs from _onConsent once the system dialog resolves.
  }

  Future<void> _stop() async {
    final capture = context.read<PlaybackCaptureService>();
    final audio = context.read<AudioService>();
    await capture.stopCapture();
    audio.stopScoring();
    if (mounted) setState(() => _capturing = false);
  }

  @override
  void dispose() {
    _consentSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.0;

    return Scaffold(
      appBar: AppBar(title: const Text('VoIP Call Protection')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          const Text(
            'Analyzes the other side\'s voice on a WhatsApp, Telegram, or '
            'Zoom call while it plays. This does not analyze your own '
            'microphone input.',
            style: TextStyle(fontSize: 12),
          ),
          const SizedBox(height: 20),
          RiskMeter(score: score),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _capturing ? _stop : _start,
            child: Text(_capturing ? 'Stop Protection' : 'Start Protection'),
          ),
        ]),
      ),
    );
  }
}
