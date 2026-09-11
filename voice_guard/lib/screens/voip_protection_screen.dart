import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../services/playback_capture_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/shad_risk_meter.dart';
import '../widgets/shad_card.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_button.dart';
import '../design/tokens.dart';

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
      backgroundColor: ShadTokens.background,
      appBar: AppBar(
        backgroundColor: ShadTokens.surface,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: const Text(
          'VoIP & Conference Protection',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.3,
            color: ShadTokens.foreground,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.all(ShadTokens.space4),
        children: [
          // ── App Explanation Card ──
          ShadCard(
            padding: const EdgeInsets.all(ShadTokens.space4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    ShadBadge(
                      label: _capturing ? 'PLAYBACK CAPTURE ACTIVE' : 'STANDBY',
                      variant: _capturing ? ShadBadgeVariant.verified : ShadBadgeVariant.secondary,
                      showDot: true,
                    ),
                    const ShadBadge(
                      label: 'MEDIA PROJECTION',
                      variant: ShadBadgeVariant.outline,
                    ),
                  ],
                ),
                const SizedBox(height: ShadTokens.space3),
                const Text(
                  'Incoming Caller Playback Capture',
                  style: TextStyle(fontSize: 15, fontWeight: FontWeight.w700, color: ShadTokens.cardFg),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Analyzes the other participant\'s voice on WhatsApp, Telegram, Zoom, or Google Meet as it plays through the device speaker. Excludes your own microphone to prevent false positives.',
                  style: TextStyle(fontSize: 12, color: ShadTokens.muted, height: 1.4),
                ),
                const SizedBox(height: ShadTokens.space3),
                Wrap(
                  spacing: 6,
                  runSpacing: 6,
                  children: const [
                    ShadBadge(label: 'WhatsApp', variant: ShadBadgeVariant.secondary),
                    ShadBadge(label: 'Telegram', variant: ShadBadgeVariant.secondary),
                    ShadBadge(label: 'Zoom', variant: ShadBadgeVariant.secondary),
                    ShadBadge(label: 'Google Meet', variant: ShadBadgeVariant.secondary),
                    ShadBadge(label: 'Teams', variant: ShadBadgeVariant.secondary),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: ShadTokens.space4),

          // ── Central Risk Meter ──
          Center(
            child: ShadRiskMeter(
              score: score,
              size: 210,
              showLegend: true,
            ),
          ),
          const SizedBox(height: ShadTokens.space5),

          // ── Main Action Button ──
          ShadButton(
            onTap: _capturing ? _stop : _start,
            variant: _capturing ? ShadButtonVariant.destructive : ShadButtonVariant.primary,
            size: ShadButtonSize.lg,
            fullWidth: true,
            icon: Icon(_capturing ? LucideIcons.stopCircle : LucideIcons.play),
            text: _capturing ? 'Stop VoIP Protection' : 'Engage VoIP Audio Defense',
          ),
          const SizedBox(height: ShadTokens.space4),
          const Center(
            child: Text(
              'Uses Android AudioPlaybackCapture API • Android 10+ compatible',
              style: TextStyle(fontSize: 11, color: ShadTokens.muted),
            ),
          ),
        ],
      ),
    );
  }
}
