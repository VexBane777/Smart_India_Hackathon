import 'dart:async';
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../providers/settings_provider.dart';
import '../providers/calibration_provider.dart';
import '../services/call_service.dart';
import '../services/audio_service.dart';
import '../services/notification_service.dart';
import '../widgets/shad_risk_meter.dart';
import '../widgets/shad_waveform.dart';
import '../widgets/shad_glass_card.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_button.dart';
import '../design/tokens.dart';
import '../models/call_state.dart';
import '../models/call_log.dart';
import '../models/risk_score.dart';

class CallScreen extends StatefulWidget {
  const CallScreen({super.key});
  @override
  State<CallScreen> createState() => _CallScreenState();
}

class _CallScreenState extends State<CallScreen> {
  String _dialNumber = '';
  bool _speakerphoneOn = false;
  bool _micMuted = false;
  bool _liveMicActive = false;
  bool _detectionActive = false;
  bool _fileScanActive = false;
  bool _isDefaultDialer = false;
  DateTime? _callStartTime;
  List<double> _liveWaveform = const [];
  StreamSubscription<double>? _scoreSub;
  StreamSubscription<List<double>>? _pcmSub;
  StreamSubscription<bool>? _signalSub;
  StreamSubscription<(String?, double)>? _attackTypeSub;
  Timer? _captureStatusTimer;
  CaptureStatus? _captureStatus;
  String? _attackType;
  double _attackConfidence = 0.0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      _bindPipeline();
      _checkDefaultDialer();
    });
  }

  Future<void> _checkDefaultDialer() async {
    final calls = context.read<CallService>();
    final isDef = await calls.isDefaultDialer();
    if (mounted) setState(() => _isDefaultDialer = isDef);
  }

  Future<void> _requestDefaultDialer() async {
    final calls = context.read<CallService>();
    await calls.setAsDefaultDialer();
    await Future.delayed(const Duration(milliseconds: 1200));
    final isDef = await calls.isDefaultDialer();
    if (mounted) {
      setState(() => _isDefaultDialer = isDef);
      if (isDef) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            backgroundColor: ShadTokens.verified,
            content: Text('Vaani is now your default phone dialer!'),
          ),
        );
      }
    }
  }

  void _bindPipeline() {
    final audio = context.read<AudioService>();
    final riskProvider = context.read<RiskScoreProvider>();
    final settings = context.read<SettingsProvider>();
    final calls = context.read<CallService>();
    final notifs = context.read<NotificationService>();
    final calibration = context.read<CalibrationProvider>();

    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = riskProvider.isAlert;
      // Per-speaker calibration shifts where the alert line is drawn; when
      // uncalibrated this is just settings.sensitivity (see
      // CalibrationProvider.computeThreshold).
      final effectiveThreshold = calibration.effectiveThreshold(settings.sensitivity);
      riskProvider.update(score, alertThreshold: effectiveThreshold);
      final cur = riskProvider.current;
      debugPrint('Monitor: raw=${score.toStringAsFixed(3)} ema=${cur?.score.toStringAsFixed(3)} '
          'state=${riskProvider.state} label=${cur?.label} threshold=${effectiveThreshold.toStringAsFixed(3)}');
      if (!wasAlert && riskProvider.isAlert) {
        debugPrint('Monitor: ALERT fired — ema=${cur?.score.toStringAsFixed(3)} '
            'threshold=${effectiveThreshold.toStringAsFixed(3)} sensitivity=${settings.sensitivity}');
      }
      if (cur != null && cur.score > effectiveThreshold) {
        if (settings.overlayEnabled) {
          try { await calls.showOverlay(riskScore: cur.score, verdict: cur.label); } catch (_) {}
        }
        if (settings.soundEnabled) {
          try { await notifs.showRiskAlert(score: cur.score, verdict: cur.label); } catch (_) {}
        }
      }
    });

    _pcmSub = audio.pcmStream.listen((samples) {
      if (!mounted) return;
      setState(() => _liveWaveform = samples);
    });

    _signalSub = audio.hasSignalStream.listen((hasSignal) {
      if (!mounted) return;
      riskProvider.setHasSignal(hasSignal);
    });

    _attackTypeSub = audio.attackTypeStream.listen((event) {
      if (!mounted) return;
      final (attackType, attackConfidence) = event;
      setState(() {
        _attackType = attackType;
        _attackConfidence = attackConfidence;
      });
    });
  }

  @override
  void dispose() {
    _scoreSub?.cancel();
    _pcmSub?.cancel();
    _signalSub?.cancel();
    _attackTypeSub?.cancel();
    _captureStatusTimer?.cancel();
    super.dispose();
  }

  void _startCaptureStatusPolling() {
    _captureStatusTimer?.cancel();
    final calls = context.read<CallService>();
    _captureStatusTimer = Timer.periodic(const Duration(seconds: 2), (_) async {
      final status = await calls.getCaptureStatus();
      if (!mounted) return;
      setState(() => _captureStatus = status);
      context.read<RiskScoreProvider>().setCaptureSource(status.source);
    });
  }

  void _stopCaptureStatusPolling() {
    _captureStatusTimer?.cancel();
    _captureStatusTimer = null;
    if (mounted) setState(() => _captureStatus = null);
  }

  void _onDigitPress(String digit) {
    if (_dialNumber.length < 15) {
      setState(() => _dialNumber += digit);
    }
  }

  void _onBackspace() {
    if (_dialNumber.isNotEmpty) {
      setState(() => _dialNumber = _dialNumber.substring(0, _dialNumber.length - 1));
    }
  }

  Future<void> _startOutgoingCall() async {
    if (_dialNumber.trim().isEmpty) return;
    final calls = context.read<CallService>();
    final callState = context.read<CallStateProvider>();

    _callStartTime = DateTime.now();
    callState.setStatus(CallStatus.dialing, number: _dialNumber);

    final placed = await calls.placeCall(_dialNumber);
    if (!placed) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not initiate carrier call. Please verify phone permissions.')),
        );
      }
    }
  }

  Future<void> _toggleLiveMic() async {
    if (_liveMicActive) {
      await _endCall();
    } else {
      final calls = context.read<CallService>();
      final audio = context.read<AudioService>();
      final callState = context.read<CallStateProvider>();
      final risk = context.read<RiskScoreProvider>();

      _callStartTime = DateTime.now();
      setState(() => _liveMicActive = true);
      risk.reset();
      audio.clearBuffer();
      audio.startScoring();
      callState.setStatus(CallStatus.active, number: 'Live Acoustic Scanner');
      await calls.startCallDetection();
      _startCaptureStatusPolling();
    }
  }

  /// Tests the ONNX scoring path on a user-picked .wav file, bypassing the
  /// acoustic speaker->mic loop entirely. This is the deterministic demo path
  /// (Module E spec's audio_decode_bridge): the same model, the same feature
  /// extraction, the same gauge — just the file's own PCM, no room acoustics.
  Future<void> _startFileScan() async {
    final calls = context.read<CallService>();
    final audio = context.read<AudioService>();
    final callState = context.read<CallStateProvider>();
    final risk = context.read<RiskScoreProvider>();

    final bytes = await calls.pickAudioForTest();
    if (!mounted) return;
    if (bytes == null || bytes.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('No WAV file selected.')),
      );
      return;
    }

    // Tear down any live capture/scoring so the file scan is the only
    // source feeding the gauge.
    audio.stopScoring();
    audio.stopAudioFileScoring();
    await calls.stopCallDetection();
    _stopCaptureStatusPolling();

    _callStartTime = DateTime.now();
    setState(() {
      _liveMicActive = false;
      _detectionActive = false;
      _fileScanActive = true;
    });
    callState.setStatus(CallStatus.active, number: 'Audio file scan');
    risk.reset();
    audio.clearBuffer();

    final err = await audio.scanAudioFile(bytes);
    if (!mounted) return;
    setState(() => _fileScanActive = false);
    if (err == 'cancelled') {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Scan cancelled.')),
      );
    } else if (err != null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Could not scan file: $err')),
      );
      callState.setStatus(CallStatus.disconnected);
    } else {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('File scan complete.')),
      );
    }
  }

  Future<void> _toggleDetection() async {
    final calls = context.read<CallService>();
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();

    if (_detectionActive) {
      await calls.stopCallDetection();
      _stopCaptureStatusPolling();
      audio.stopScoring();
      setState(() => _detectionActive = false);
    } else {
      risk.reset();
      audio.clearBuffer();
      audio.startScoring();
      await calls.startCallDetection();
      _startCaptureStatusPolling();
      setState(() => _detectionActive = true);
    }
  }

  Future<void> _endCall() async {
    final calls = context.read<CallService>();
    final audio = context.read<AudioService>();
    final callState = context.read<CallStateProvider>();
    final risk = context.read<RiskScoreProvider>();

    final duration = _callStartTime != null
        ? DateTime.now().difference(_callStartTime!)
        : Duration.zero;
    _callStartTime = null;

    _stopCaptureStatusPolling();
    setState(() {
      _liveMicActive = false;
      _speakerphoneOn = false;
      _micMuted = false;
      _detectionActive = false;
      _fileScanActive = false;
    });
    audio.stopScoring();
    audio.stopAudioFileScoring();
    await calls.endCall();
    await calls.stopCallDetection();
    await calls.hideOverlay();

    await Future.delayed(const Duration(milliseconds: 300));
    final recPath = await calls.getLastRecordingPath();
    final curRisk = risk.current;
    final score = curRisk?.score ?? 0.0;
    final verdict = curRisk?.verdict ??
        (score >= 0.70
            ? Verdict.detected
            : (score >= 0.30 ? Verdict.suspicious : Verdict.verified));

    final log = CallLog(
      id: DateTime.now().millisecondsSinceEpoch.toString(),
      timestamp: DateTime.now(),
      number: callState.state.number ?? (_dialNumber.isNotEmpty ? _dialNumber : 'Live Acoustic Scan'),
      riskScore: score,
      verdict: verdict,
      duration: duration,
      recordingPath: recPath,
    );
    risk.addCallLog(log);

    callState.setStatus(CallStatus.disconnected);
    Future.delayed(const Duration(milliseconds: 600), () {
      if (mounted) callState.setStatus(CallStatus.idle);
    });
  }

  Future<void> _toggleSpeakerphone() async {
    final calls = context.read<CallService>();
    final next = !_speakerphoneOn;
    final res = await calls.toggleSpeakerphone(next);
    setState(() => _speakerphoneOn = res);
  }

  Future<void> _toggleMicMute() async {
    final calls = context.read<CallService>();
    final next = !_micMuted;
    final res = await calls.toggleMicMute(next);
    setState(() => _micMuted = res);
  }

  /// The AppBar's red control is labeled "Stop Live Monitor", so it must only
  /// ever STOP things — never start them (Codex review on PR #8: during a
  /// carrier call with `_liveMicActive == false` it used to fall through to
  /// `_toggleLiveMic`'s start path, overwriting the displayed call state with
  /// "Live Acoustic Scanner" and starting capture).
  Future<void> _onAppBarStopTap() async {
    // 1) Live Mic self-test session — tear it down entirely; _endCall also
    //    writes the call log and restores the dialpad view.
    if (_liveMicActive) {
      await _endCall();
      return;
    }
    // 2) Audio-file scan session — same teardown as live mic.
    if (_fileScanActive) {
      await _endCall();
      return;
    }
    // 2) Carrier call with the opt-in acoustic monitor armed — stop it
    //    (detection, scoring, capture-status polling) but keep the call.
    if (_detectionActive) {
      await _toggleDetection();
      return;
    }
    // 3) Nothing armed by the user, but the telecom callback in main.dart may
    //    still be driving ambient scoring for this call. Shut that down and
    //    dismiss any live risk overlay so the monitor is fully quiet.
    final audio = context.read<AudioService>();
    audio.stopScoring();
    try {
      await context.read<CallService>().hideOverlay();
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    final call = context.watch<CallStateProvider>();
    final riskProvider = context.watch<RiskScoreProvider>();
    final risk = riskProvider.current;
    final score = risk?.score ?? 0.0;
    final isCallInProgress = call.state.isActive || call.state.isDialing || call.state.isIncoming || _liveMicActive;
    final verdict = risk?.label ?? ShadRiskMeter.verdictFor(score);
    final color = risk?.color ?? ShadRiskMeter.colorFor(score);
    final scoringHasSignal = riskProvider.hasSignal;

    return Scaffold(
      backgroundColor: const Color(0xFF09090B),
      appBar: isCallInProgress
          ? AppBar(
              backgroundColor: Colors.transparent,
              elevation: 0,
              scrolledUnderElevation: 0,
              title: Text(
                'Acoustic Defense Monitor',
                style: GoogleFonts.inter(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  color: Colors.white,
                ),
              ),
              actions: [
                IconButton(
                  tooltip: 'Stop Live Monitor',
                  icon: Container(
                    padding: const EdgeInsets.all(7),
                    decoration: BoxDecoration(
                      color: const Color(0xFFFF6268),
                      borderRadius: BorderRadius.circular(10),
                    ),
                    child: const Icon(LucideIcons.square, size: 16, color: Colors.white),
                  ),
                  onPressed: _onAppBarStopTap,
                ),
                const SizedBox(width: 8),
              ],
            )
          : null,
      body: SafeArea(
        child: isCallInProgress
            ? _buildActiveCallView(
                call: call,
                score: score,
                verdict: verdict,
                color: color,
                scoringHasSignal: scoringHasSignal,
              )
            : _buildDialpadView(),
      ),
    );
  }

  // ── VIEW 1: ACTIVE CALL & REAL-TIME INFERENCE ──
  Widget _buildActiveCallView({
    required CallStateProvider call,
    required double score,
    required String verdict,
    required Color color,
    required bool scoringHasSignal,
  }) {
    final audio = context.read<AudioService>();
    final callerLabel = call.state.number ?? (_dialNumber.isNotEmpty ? _dialNumber : 'Live Acoustic Scanner');

    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      children: [
        // ── Caller Banner Card ──
        ShadGlassCard(
          padding: const EdgeInsets.all(ShadTokens.space4),
          child: Row(
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                ),
                child: Icon(LucideIcons.user, color: color, size: 24),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      callerLabel,
                      style: const TextStyle(
                        fontSize: 15,
                        fontWeight: FontWeight.w800,
                        letterSpacing: -0.3,
                        color: ShadTokens.foreground,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 3),
                    Wrap(
                      spacing: 6,
                      runSpacing: 4,
                      crossAxisAlignment: WrapCrossAlignment.center,
                      children: [
                        Text(
                          call.state.isDialing ? 'Dialing...' : 'Live • ${call.elapsedLabel}',
                          style: const TextStyle(fontSize: 11, color: ShadTokens.muted, fontWeight: FontWeight.w500),
                        ),
                        ShadBadge(
                          label: scoringHasSignal ? 'VOICE DETECTED' : 'SILENCE',
                          variant: scoringHasSignal ? ShadBadgeVariant.verified : ShadBadgeVariant.secondary,
                          showDot: true,
                        ),
                        if (_captureStatus?.source != null)
                          ShadBadge(
                            label: _captureStatus!.source!,
                            variant: ShadBadgeVariant.outline,
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Carrier Detection Opt-in (if normal call) ──
        if (!_liveMicActive && !_fileScanActive && call.state.isActive) ...[
          ShadGlassCard(
            padding: const EdgeInsets.all(ShadTokens.space3),
            tintColor: _detectionActive ? const Color(0xFFF0FDF4) : Colors.white,
            child: Row(
              children: [
                Icon(
                  _detectionActive ? LucideIcons.shieldCheck : LucideIcons.shield,
                  color: _detectionActive ? ShadTokens.verified : ShadTokens.muted,
                  size: 20,
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        _detectionActive ? 'Acoustic Monitoring Running' : 'Acoustic Monitoring Idle',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w700,
                          color: _detectionActive ? ShadTokens.verified : ShadTokens.foreground,
                        ),
                      ),
                      Text(
                        _detectionActive
                            ? 'Speakerphone engaged so mic can capture remote caller audio.'
                            : 'Normal call audio. Turn on to score incoming voice.',
                        style: const TextStyle(fontSize: 10, color: ShadTokens.muted),
                      ),
                    ],
                  ),
                ),
                ShadButton(
                  onTap: _toggleDetection,
                  variant: _detectionActive ? ShadButtonVariant.outline : ShadButtonVariant.primary,
                  size: ShadButtonSize.sm,
                  text: _detectionActive ? 'Stop' : 'Start Monitor',
                ),
              ],
            ),
          ),
          const SizedBox(height: 16),
        ],

        // ── Central High-Impact Risk Meter ──
        Center(
          child: ShadRiskMeter(
            score: score,
            size: 200,
            showLegend: true,
          ),
        ),
        const SizedBox(height: 16),

        // ── Security Advisory Banner ──
        ShadGlassCard(
          padding: const EdgeInsets.all(ShadTokens.space4),
          tintColor: ShadRiskMeter.bgFor(score),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Icon(
                verdict == 'AI DETECTED'
                    ? LucideIcons.alertTriangle
                    : (verdict == 'SUSPICIOUS' ? LucideIcons.alertCircle : LucideIcons.shieldCheck),
                color: color,
                size: 18,
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Text(
                          verdict,
                          style: TextStyle(fontSize: 13, fontWeight: FontWeight.w800, color: color),
                        ),
                        if (_attackTypeSubLabel(verdict) case final String subLabel) ...[
                          const SizedBox(width: 6),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                            decoration: BoxDecoration(
                              color: color.withValues(alpha: 0.15),
                              borderRadius: BorderRadius.circular(6),
                            ),
                            child: Text(
                              subLabel,
                              style: TextStyle(fontSize: 10, fontWeight: FontWeight.w700, color: color),
                            ),
                          ),
                        ],
                      ],
                    ),
                    const SizedBox(height: 2),
                    Text(
                      _adviceFor(verdict),
                      style: const TextStyle(fontSize: 11, color: ShadTokens.foreground, height: 1.35),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Real-time Waveform Visualizer ──
        ShadGlassCard(
          padding: const EdgeInsets.all(ShadTokens.space4),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: const [
                  Text(
                    'Microphone Telephony Stream',
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: ShadTokens.foreground),
                  ),
                  ShadBadge(label: '16 kHz PCM', variant: ShadBadgeVariant.outline),
                ],
              ),
              const SizedBox(height: 12),
              ShadWaveform(samples: _liveWaveform, color: color, height: 48),
            ],
          ),
        ),
        const SizedBox(height: 16),

        // ── Benchmark Test Injection (Judges / SIH Verification) ──
        ShadGlassCard(
          padding: const EdgeInsets.all(ShadTokens.space3),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text(
                'Model Validation & Injections',
                style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: ShadTokens.muted),
              ),
              const SizedBox(height: 8),
              Row(
                children: [
                  Expanded(
                    child: ShadButton(
                      onTap: () => audio.injectBenchmarkTest(isAiVoice: false),
                      variant: ShadButtonVariant.outline,
                      size: ShadButtonSize.sm,
                      icon: const Icon(LucideIcons.checkCircle2, color: ShadTokens.verified, size: 14),
                      text: 'Human Voice',
                    ),
                  ),
                  const SizedBox(width: 8),
                  Expanded(
                    child: ShadButton(
                      onTap: () => audio.injectBenchmarkTest(isAiVoice: true),
                      variant: ShadButtonVariant.outline,
                      size: ShadButtonSize.sm,
                      icon: const Icon(LucideIcons.alertTriangle, color: ShadTokens.detected, size: 14),
                      text: 'AI Clone Clip',
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
        const SizedBox(height: 24),

        // ── In-Call Action Dock ──
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceEvenly,
          children: [
            _circleAction(
              icon: _speakerphoneOn ? LucideIcons.volume2 : LucideIcons.volume1,
              label: _speakerphoneOn ? 'Speaker On' : 'Speaker Off',
              active: _speakerphoneOn,
              onTap: _toggleSpeakerphone,
            ),
            // End Call
            GestureDetector(
              onTap: _endCall,
              child: Container(
                width: 60,
                height: 60,
                decoration: const BoxDecoration(
                  color: ShadTokens.destructive,
                  shape: BoxShape.circle,
                ),
                child: const Icon(LucideIcons.phoneOff, color: Colors.white, size: 24),
              ),
            ),
            _circleAction(
              icon: _micMuted ? LucideIcons.micOff : LucideIcons.mic,
              label: _micMuted ? 'Muted' : 'Mic On',
              active: !_micMuted,
              onTap: _toggleMicMute,
            ),
          ],
        ),
        const SizedBox(height: 16),
      ],
    );
  }

  // ── VIEW 2: INTERACTIVE DIALPAD ──
  // ── VIEW 2: INTERACTIVE DIALPAD (FIGMA MAIN-1.PNG) ──
  Widget _buildDialpadView() {
    return Column(
      children: [
        const SizedBox(height: 16),
        // ── Top Pill Header: Zero-Trust Interception | ✓ Active ──
        Center(
          child: GestureDetector(
            onTap: _isDefaultDialer ? null : _requestDefaultDialer,
            child: Container(
              padding: const EdgeInsets.only(left: 18, right: 6, top: 6, bottom: 6),
              decoration: BoxDecoration(
                color: const Color(0xFF18181B),
                borderRadius: BorderRadius.circular(999),
                border: Border.all(color: const Color(0xFF27272A)),
              ),
              child: Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Zero-Trust Interception',
                    style: GoogleFonts.inter(
                      fontSize: 13,
                      fontWeight: FontWeight.w500,
                      color: const Color(0xFFD4D4D8),
                    ),
                  ),
                  const SizedBox(width: 14),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                    decoration: BoxDecoration(
                      color: const Color(0xFF131315),
                      borderRadius: BorderRadius.circular(999),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Icon(
                          _isDefaultDialer ? LucideIcons.check : LucideIcons.shieldAlert,
                          size: 12,
                          color: _isDefaultDialer ? const Color(0xFF10B981) : const Color(0xFFF59E0B),
                        ),
                        const SizedBox(width: 4),
                        Text(
                          _isDefaultDialer ? 'Active' : 'Set Default',
                          style: GoogleFonts.inter(
                            fontSize: 11,
                            fontWeight: FontWeight.w700,
                            color: _isDefaultDialer ? const Color(0xFF10B981) : const Color(0xFFF59E0B),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),

        // ── Number Display Box ──
        Expanded(
          flex: 2,
          child: Container(
            alignment: Alignment.center,
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Text(
                  _dialNumber.isEmpty ? '+91 98450 ...' : _dialNumber,
                  style: GoogleFonts.inter(
                    fontSize: _dialNumber.length > 12 ? 26 : 32,
                    fontWeight: FontWeight.w700,
                    color: _dialNumber.isEmpty ? const Color(0xFF71717A) : Colors.white,
                    letterSpacing: 1.2,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 6),
                Text(
                  'On-Device Inference Active',
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w400,
                    color: const Color(0xFF71717A),
                  ),
                ),
              ],
            ),
          ),
        ),

        // ── Dialpad Grid (0-9, *, #) ──
        Expanded(
          flex: 6,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 36),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                _dialRow(['1', '2', '3'], ['', 'ABC', 'DEF']),
                _dialRow(['4', '5', '6'], ['GHI', 'JKL', 'MNO']),
                _dialRow(['7', '8', '9'], ['PQRS', 'TUV', 'WXYZ']),
                _dialRow(['*', '0', '#'], ['', '+', '']),
              ],
            ),
          ),
        ),

        // ── Action Bar: Live Mic, File Scan, Call, Backspace ──
        Padding(
          padding: const EdgeInsets.only(bottom: 28, left: 36, right: 36),
          child: Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              // Live Mic mode button
              InkWell(
                onTap: _toggleLiveMic,
                borderRadius: BorderRadius.circular(32),
                child: Container(
                  width: 62,
                  height: 62,
                  decoration: const BoxDecoration(
                    color: Color(0xFF1C1C1E),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(LucideIcons.mic, size: 22, color: Colors.white),
                ),
              ),

              // "Test with audio file" — scores a picked .wav through the
              // same ONNX path with NO acoustic loop (deterministic demo).
              InkWell(
                onTap: _startFileScan,
                borderRadius: BorderRadius.circular(32),
                child: Container(
                  width: 62,
                  height: 62,
                  decoration: const BoxDecoration(
                    color: Color(0xFF1C1C1E),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(LucideIcons.fileAudio2, size: 22, color: Color(0xFF9AA8BB)),
                ),
              ),

              // Large Call Button (Emerald Green - Figma #00C274)
              GestureDetector(
                onTap: _startOutgoingCall,
                child: Container(
                  width: 68,
                  height: 68,
                  decoration: const BoxDecoration(
                    color: Color(0xFF00C274),
                    shape: BoxShape.circle,
                    boxShadow: [
                      BoxShadow(
                        color: Color(0x3300C274),
                        blurRadius: 14,
                        offset: Offset(0, 4),
                      ),
                    ],
                  ),
                  child: const Icon(LucideIcons.phone, color: Colors.white, size: 28),
                ),
              ),

              // Backspace button
              InkWell(
                onTap: _onBackspace,
                onLongPress: () => setState(() => _dialNumber = ''),
                borderRadius: BorderRadius.circular(32),
                child: Container(
                  width: 62,
                  height: 62,
                  decoration: const BoxDecoration(
                    color: Color(0xFF1C1C1E),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(LucideIcons.delete, size: 22, color: Color(0xFF8E9192)),
                ),
              ),
            ],
          ),
        ),
      ],
    );
  }

  Widget _dialRow(List<String> digits, List<String> subs) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        for (int i = 0; i < 3; i++) _dialKey(digits[i], subs[i]),
      ],
    );
  }

  Widget _dialKey(String digit, String sub) {
    return InkWell(
      onTap: () => _onDigitPress(digit),
      onLongPress: digit == '0' ? () => _onDigitPress('+') : null,
      borderRadius: BorderRadius.circular(40),
      child: Container(
        width: 70,
        height: 70,
        decoration: const BoxDecoration(
          color: Color(0xFF1C1C1E),
          shape: BoxShape.circle,
        ),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Text(
              digit,
              style: GoogleFonts.inter(
                fontSize: 26,
                fontWeight: FontWeight.w600,
                color: Colors.white,
              ),
            ),
            if (sub.isNotEmpty)
              Text(
                sub,
                style: GoogleFonts.inter(
                  fontSize: 9,
                  fontWeight: FontWeight.w600,
                  color: const Color(0xFF71717A),
                  letterSpacing: 1.0,
                ),
              ),
          ],
        ),
      ),
    );
  }

  Widget _circleAction({
    required IconData icon,
    required String label,
    required bool active,
    required VoidCallback onTap,
  }) {
    return Column(
      children: [
        InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(26),
          child: Container(
            width: 50,
            height: 50,
            decoration: BoxDecoration(
              color: active ? ShadTokens.primary : const Color(0xFF27272A),
              shape: BoxShape.circle,
              border: Border.all(
                color: active ? ShadTokens.primary : ShadTokens.border,
              ),
            ),
            child: Icon(
              icon,
              // Inactive controls sit on a dark zinc circle, so their icons
              // must stay white; the old Colors.white background +
              // ShadTokens.foreground (white) icon rendered a blank circle
              // (Codex review on PR #8).
              color: active ? ShadTokens.primaryFg : ShadTokens.foreground,
              size: 22,
            ),
          ),
        ),
        const SizedBox(height: 6),
        Text(
          label,
          style: const TextStyle(fontSize: 10, fontWeight: FontWeight.w600, color: ShadTokens.muted),
        ),
      ],
    );
  }

  /// Sub-label shown alongside "AI DETECTED" — never standalone, never
  /// below the confidence threshold (see
  /// docs/superpowers/specs/2026-09-11-attack-type-differentiator-design.md §6).
  String? _attackTypeSubLabel(String verdict) {
    const confidenceThreshold = 0.70; // draft value from the design spec §6, tune during on-device testing
    if (verdict != 'AI DETECTED' || _attackType == null || _attackConfidence < confidenceThreshold) {
      return null;
    }
    switch (_attackType) {
      case 'tts':
        return 'TTS';
      case 'vc':
        return 'VOICE CLONE';
      default:
        return 'SYNTHETIC';
    }
  }

  String _adviceFor(String verdict) {
    switch (verdict) {
      case 'AI DETECTED':
        return 'High probability of synthetic/cloned speech. Do NOT share OTP or passwords. Hang up and verify through an independent channel.';
      case 'SUSPICIOUS':
        return 'Vocal anomalies or acoustic distortions detected. Request verification before sharing credentials.';
      default:
        return 'Acoustic parameters match natural human vocal tract dynamics.';
    }
  }
}
