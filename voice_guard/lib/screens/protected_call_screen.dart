import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'package:path_provider/path_provider.dart';
import '../services/signaling_service.dart';
import '../services/webrtc_call_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../providers/settings_provider.dart';
import '../widgets/shad_risk_meter.dart';
import '../widgets/shad_alert_banner.dart';
import '../widgets/shad_card.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_button.dart';
import '../widgets/shad_input.dart';
import '../design/tokens.dart';
import '../models/call_log.dart';
import '../models/risk_score.dart';
import '../utils/wav_encoder.dart';

class ProtectedCallScreen extends StatefulWidget {
  const ProtectedCallScreen({super.key});
  @override
  State<ProtectedCallScreen> createState() => _ProtectedCallScreenState();
}

class _ProtectedCallScreenState extends State<ProtectedCallScreen> {
  final _roomController = TextEditingController();
  WebRtcCallService? _call;
  RTCPeerConnectionState _state = RTCPeerConnectionState.RTCPeerConnectionStateNew;
  StreamSubscription<double>? _scoreSub;

  Future<void> _connect({required bool isCaller}) async {
    final roomId = _roomController.text.trim();
    if (roomId.isEmpty) return;
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();
    risk.reset();
    audio.clearBuffer();
    audio.startScoring();

    final settings = context.read<SettingsProvider>();
    _scoreSub?.cancel();
    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = risk.isAlert;
      risk.update(score, alertThreshold: settings.sensitivity);
      if (!wasAlert && risk.isAlert) {
        HapticFeedback.heavyImpact();
        await _dumpForensicAudio(audio, risk);
      }
    });

    final signaling = SignalingService.connect(
      roomId,
      host: settings.signalingHost,
      port: settings.signalingPort,
    );
    final call = WebRtcCallService(audioService: audio, signaling: signaling);
    call.connectionState.listen((s) {
      if (mounted) setState(() => _state = s);
    });
    await call.startCall(roomId, isCaller: isCaller);
    setState(() => _call = call);
  }

  Future<void> _hangUp() async {
    final audio = context.read<AudioService>();
    await _call?.endCall();
    await _scoreSub?.cancel();
    audio.stopScoring();
    setState(() => _call = null);
  }

  Future<void> _dumpForensicAudio(AudioService audio, RiskScoreProvider risk) async {
    try {
      final samples = audio.snapshotBuffer();
      final wavBytes = WavEncoder.encodePcm16Mono(samples, sampleRate: 16000);
      final dir = await getApplicationDocumentsDirectory();
      final path = '${dir.path}/forensic_${DateTime.now().millisecondsSinceEpoch}.wav';
      await File(path).writeAsBytes(wavBytes);
      risk.addCallLog(CallLog(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.now(),
        number: 'Protected Call: ${_roomController.text.trim()}',
        riskScore: risk.current?.score ?? 0.0,
        verdict: Verdict.detected,
        recordingPath: path,
      ));
    } catch (e) {
      debugPrint('Protected Call: forensic dump failed: $e');
    }
  }

  @override
  void dispose() {
    _call?.dispose();
    _scoreSub?.cancel();
    _roomController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.0;
    final isCallActive = _call != null;

    return Scaffold(
      backgroundColor: ShadTokens.background,
      appBar: AppBar(
        backgroundColor: ShadTokens.surface,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: const Text(
          'Protected Call (VAANI-to-VAANI)',
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
          // ── Instructions Card ──
          ShadCard(
            padding: const EdgeInsets.all(ShadTokens.space4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    ShadBadge(
                      label: isCallActive ? 'WEBRTC ACTIVE' : 'PEER-TO-PEER ENCRYPTED',
                      variant: isCallActive ? ShadBadgeVariant.verified : ShadBadgeVariant.secondary,
                      showDot: isCallActive,
                    ),
                    const ShadBadge(
                      label: 'DTLS-SRTP',
                      variant: ShadBadgeVariant.outline,
                    ),
                  ],
                ),
                const SizedBox(height: ShadTokens.space3),
                const Text(
                  'End-to-End Encrypted Voice Channel',
                  style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: ShadTokens.cardFg),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Both parties enter the same room identifier while on the same local network or signaling channel. Live audio is intercepted and analyzed for AI cloning in real-time.',
                  style: TextStyle(fontSize: 12, color: ShadTokens.muted, height: 1.4),
                ),
              ],
            ),
          ),
          const SizedBox(height: ShadTokens.space4),

          // ── Input & Action Card ──
          if (!isCallActive) ...[
            ShadCard(
              padding: const EdgeInsets.all(ShadTokens.space4),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Join or Create Room',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: ShadTokens.cardFg),
                  ),
                  const SizedBox(height: 8),
                  ShadInput(
                    controller: _roomController,
                    placeholder: 'Enter 4-digit room code (e.g. 1024)',
                    prefix: const Icon(LucideIcons.hash, size: 16, color: ShadTokens.muted),
                  ),
                  const SizedBox(height: ShadTokens.space4),
                  Row(
                    children: [
                      Expanded(
                        child: ShadButton(
                          onTap: () => _connect(isCaller: true),
                          variant: ShadButtonVariant.primary,
                          size: ShadButtonSize.md,
                          icon: const Icon(LucideIcons.phoneOutgoing, size: 15),
                          text: 'Initiate Call',
                        ),
                      ),
                      const SizedBox(width: ShadTokens.space2),
                      Expanded(
                        child: ShadButton(
                          onTap: () => _connect(isCaller: false),
                          variant: ShadButtonVariant.outline,
                          size: ShadButtonSize.md,
                          icon: const Icon(LucideIcons.phoneIncoming, size: 15),
                          text: 'Answer Room',
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ] else ...[
            // ── Live Connected Call State ──
            ShadCard(
              padding: const EdgeInsets.all(ShadTokens.space4),
              child: Row(
                children: [
                  Container(
                    width: 44,
                    height: 44,
                    decoration: BoxDecoration(
                      color: ShadTokens.verifiedBg,
                      borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                    ),
                    child: const Icon(LucideIcons.lock, color: ShadTokens.verified, size: 20),
                  ),
                  const SizedBox(width: ShadTokens.space3),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Room: ${_roomController.text.trim()}',
                          style: const TextStyle(fontSize: 15, fontWeight: FontWeight.w800, color: ShadTokens.cardFg),
                        ),
                        const SizedBox(height: 2),
                        Text(
                          'Peer Status: ${_state.name.replaceAll('RTCPeerConnectionState', '')}',
                          style: const TextStyle(fontSize: 11, color: ShadTokens.muted, fontWeight: FontWeight.w600),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: ShadTokens.space4),

            ShadAlertBanner(state: context.watch<RiskScoreProvider>().state),
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

            // ── Hangup Button ──
            ShadButton(
              onTap: _hangUp,
              variant: ShadButtonVariant.destructive,
              size: ShadButtonSize.lg,
              fullWidth: true,
              icon: const Icon(LucideIcons.phoneOff, size: 18),
              text: 'Hang Up Protected Call',
            ),
          ],
        ],
      ),
    );
  }
}
