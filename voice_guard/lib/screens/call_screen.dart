import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../providers/settings_provider.dart';
import '../services/call_service.dart';
import '../services/audio_service.dart';
import '../services/notification_service.dart';
import '../widgets/risk_meter.dart';
import '../widgets/waveform_visualizer.dart';
import '../utils/constants.dart';
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
  bool _isDefaultDialer = false;
  DateTime? _callStartTime;
  List<double> _liveWaveform = const [];
  StreamSubscription<double>? _scoreSub;
  StreamSubscription<List<double>>? _pcmSub;
  StreamSubscription<bool>? _signalSub;
  Timer? _captureStatusTimer;
  CaptureStatus? _captureStatus;

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
            backgroundColor: Color(0xFF2E7D32),
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

    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = riskProvider.isAlert;
      riskProvider.update(score);
      final cur = riskProvider.current;
      debugPrint('Monitor: raw=${score.toStringAsFixed(3)} ema=${cur?.score.toStringAsFixed(3)} '
          'state=${riskProvider.state} label=${cur?.label}');
      if (!wasAlert && riskProvider.isAlert) {
        debugPrint('Monitor: ALERT fired — ema=${cur?.score.toStringAsFixed(3)} sensitivity=${settings.sensitivity}');
      }
      if (cur != null && cur.score > settings.sensitivity) {
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
  }

  @override
  void dispose() {
    _scoreSub?.cancel();
    _pcmSub?.cancel();
    _signalSub?.cancel();
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

    // Detection (forced speaker + capture) is opt-in — see _toggleDetection —
    // so placing a call no longer auto-starts it. This lets the call route
    // normally (including to an already-connected headset) until the user
    // explicitly asks to monitor it.
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
      // Stop live mic and finalize recording + log
      await _endCall();
    } else {
      // Start live mic test
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

  /// Toggles detection (forced-speaker acoustic capture) for a real, already-
  /// connected call. Kept separate from _toggleLiveMic, which is the no-call
  /// self-test path. Off by default so a normal call — including one routed
  /// to a connected BT/wired headset — behaves like any other call until the
  /// user explicitly asks to monitor it.
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
    });
    audio.stopScoring();
    await calls.endCall();
    await calls.stopCallDetection();
    await calls.hideOverlay();

    // Small delay to allow AudioCaptureManager to flush and close WAV file
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

  @override
  Widget build(BuildContext context) {
    final call = context.watch<CallStateProvider>();
    final riskProvider = context.watch<RiskScoreProvider>();
    final risk = riskProvider.current;
    final score = risk?.score ?? 0.0;
    final isCallInProgress = call.state.isActive || call.state.isDialing || call.state.isIncoming || _liveMicActive;
    final verdict = risk?.label ?? AppConstants.verdictFor(score);
    final color = risk?.color ?? AppConstants.colorFor(score);
    final scoringHasSignal = riskProvider.hasSignal;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Text(isCallInProgress ? 'Active Call Detection' : 'Dialer & Live Detection',
            style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
        actions: [
          IconButton(
            tooltip: _liveMicActive ? 'Stop Live Mic' : 'Start Live Mic Test',
            icon: Icon(_liveMicActive ? Icons.mic : Icons.mic_none, color: _liveMicActive ? AppColors.detected : AppColors.primary),
            onPressed: _toggleLiveMic,
          ),
        ],
      ),
      body: isCallInProgress
          ? _buildActiveCallView(call: call, score: score, verdict: verdict, color: color, scoringHasSignal: scoringHasSignal)
          : _buildDialpadView(),
    );
  }

  // ---- VIEW 1: ACTIVE CALL & REAL-TIME INFERENCE ----
  Widget _buildActiveCallView({
    required CallStateProvider call,
    required double score,
    required String verdict,
    required Color color,
    required bool scoringHasSignal,
  }) {
    final audio = context.read<AudioService>();
    return ListView(
      padding: const EdgeInsets.all(16),
      children: [
        // Caller card
        Container(
          padding: const EdgeInsets.all(16),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 12)],
          ),
          child: Row(children: [
            CircleAvatar(
              radius: 26,
              backgroundColor: color.withValues(alpha: 0.15),
              child: Icon(Icons.person, color: color, size: 28),
            ),
            const SizedBox(width: 12),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(call.state.number ?? (_dialNumber.isNotEmpty ? _dialNumber : 'Active Call'),
                  style: const TextStyle(fontSize: 17, fontWeight: FontWeight.w800)),
              const SizedBox(height: 2),
              Text(call.state.isDialing ? 'Dialing...' : 'Connected • ${call.elapsedLabel}',
                  style: TextStyle(fontSize: 12, color: Colors.black.withValues(alpha: 0.6))),
            ])),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(color: color.withValues(alpha: 0.15), borderRadius: BorderRadius.circular(999)),
              child: Row(mainAxisSize: MainAxisSize.min, children: [
                Container(width: 8, height: 8, decoration: BoxDecoration(color: color, shape: BoxShape.circle)),
                const SizedBox(width: 6),
                Text(verdict == 'AI DETECTED' ? 'ALERT' : 'LIVE',
                    style: TextStyle(fontSize: 11, fontWeight: FontWeight.w800, color: color)),
              ]),
            ),
          ]),
        ),
        const SizedBox(height: 14),

        // Detection is opt-in for real calls (not the Live Mic self-test):
        // starting it forces the speaker (needed for the mic to overhear the
        // far end) and overrides a connected headset for the call's duration.
        if (!_liveMicActive && call.state.isActive) ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: _detectionActive ? AppColors.primary.withValues(alpha: 0.08) : Colors.white,
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: _detectionActive ? AppColors.primary.withValues(alpha: 0.3) : Colors.black12),
            ),
            child: Row(children: [
              Icon(_detectionActive ? Icons.shield : Icons.shield_outlined,
                  color: _detectionActive ? AppColors.primary : Colors.black54, size: 22),
              const SizedBox(width: 10),
              Expanded(
                child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                  Text(_detectionActive ? 'Detection running' : 'Detection off',
                      style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
                  Text(
                    _detectionActive
                        ? 'Speaker forced on so the mic can monitor this call.'
                        : 'Call audio routes normally (including to a connected headset).',
                    style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.6)),
                  ),
                ]),
              ),
              TextButton(
                onPressed: _toggleDetection,
                style: TextButton.styleFrom(
                  backgroundColor: _detectionActive ? Colors.black12 : AppColors.primary,
                  foregroundColor: _detectionActive ? Colors.black87 : Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  minimumSize: Size.zero,
                  tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                ),
                child: Text(_detectionActive ? 'Stop' : 'Start Detection',
                    style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700)),
              ),
            ]),
          ),
          const SizedBox(height: 14),
        ],

        if (_captureStatus?.source == 'VOICE_CALL') ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.10),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.green.withValues(alpha: 0.3)),
            ),
            child: Row(children: [
              const Icon(Icons.verified_user_outlined, color: Colors.green, size: 22),
              const SizedBox(width: 10),
              const Expanded(
                child: Text(
                  'Privileged capture active — real call audio, not the acoustic (mic) fallback.',
                  style: TextStyle(fontSize: 11, color: Colors.black87, height: 1.3, fontWeight: FontWeight.w600),
                ),
              ),
            ]),
          ),
          const SizedBox(height: 14),
        ],

        if (_captureStatus != null && !_captureStatus!.hasSignal) ...[
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.orange.withValues(alpha: 0.10),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: Colors.orange.withValues(alpha: 0.3)),
            ),
            child: Row(children: [
              const Icon(Icons.mic_off_outlined, color: Colors.orange, size: 22),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  'No live audio reaching the mic (source: ${_captureStatus!.source ?? 'unknown'}). '
                  'Turn on speakerphone so the other side\'s voice can be heard by the mic.',
                  style: const TextStyle(fontSize: 11, color: Colors.black87, height: 1.3),
                ),
              ),
            ]),
          ),
          const SizedBox(height: 14),
        ],

        // Live Risk Meter (TFLite Inference)
        RiskMeter(score: score),
        if (!scoringHasSignal) ...[
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
            decoration: BoxDecoration(
              color: Colors.blueGrey.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(10),
            ),
            child: Row(mainAxisAlignment: MainAxisAlignment.center, children: [
              const Icon(Icons.volume_off_outlined, size: 16, color: Colors.blueGrey),
              const SizedBox(width: 8),
              Text('Quiet — score paused until voice resumes (last: ${(score * 100).round()}%)',
                  style: const TextStyle(fontSize: 11, color: Colors.blueGrey, fontWeight: FontWeight.w600)),
            ]),
          ),
        ],
        const SizedBox(height: 14),

        // Live Verdict Banner
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(
            color: color.withValues(alpha: 0.10),
            borderRadius: BorderRadius.circular(12),
            border: Border.all(color: color.withValues(alpha: 0.25)),
          ),
          child: Row(children: [
            Icon(verdict == 'AI DETECTED' ? Icons.warning_rounded : verdict == 'SUSPICIOUS' ? Icons.error_outline : Icons.verified_user_rounded, color: color, size: 28),
            const SizedBox(width: 10),
            Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
              Text(verdict, style: TextStyle(fontSize: 14, fontWeight: FontWeight.w800, color: color)),
              const SizedBox(height: 2),
              Text(_adviceFor(verdict), style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.7), height: 1.3)),
            ])),
          ]),
        ),
        const SizedBox(height: 14),

        // Live Audio Waveform (Driven by real 16kHz microphone stream)
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(mainAxisAlignment: MainAxisAlignment.spaceBetween, children: [
              const Text('Live Microphone Audio Stream', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
              Text('16 kHz • TFLite', style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.5))),
            ]),
            const SizedBox(height: 10),
            WaveformVisualizer(samples: _liveWaveform, color: color),
            const SizedBox(height: 8),
            Text('Evaluating 60-band spectral filterbanks + prosody every 1s on-device via TFLite.',
                style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.55))),
          ]),
        ),
        const SizedBox(height: 16),

        // Quick Benchmark Injection (For instant testing without phone calls)
        Container(
          padding: const EdgeInsets.all(12),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.black12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('Benchmark Model Verification', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: AppColors.textSecondary)),
            const SizedBox(height: 8),
            Row(children: [
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => audio.injectBenchmarkTest(isAiVoice: false),
                  icon: const Icon(Icons.check_circle_outline, color: AppColors.verified, size: 16),
                  label: const Text('Human Speech', style: TextStyle(fontSize: 11, color: AppColors.verified)),
                  style: OutlinedButton.styleFrom(side: const BorderSide(color: AppColors.verified)),
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: OutlinedButton.icon(
                  onPressed: () => audio.injectBenchmarkTest(isAiVoice: true),
                  icon: const Icon(Icons.warning_amber_rounded, color: AppColors.detected, size: 16),
                  label: const Text('AI Clone Audio', style: TextStyle(fontSize: 11, color: AppColors.detected)),
                  style: OutlinedButton.styleFrom(side: const BorderSide(color: AppColors.detected)),
                ),
              ),
            ]),
          ]),
        ),
        const SizedBox(height: 20),

        // In-Call Action Bar (Speakerphone, Mute, End Call)
        Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
          _circleAction(
            icon: _speakerphoneOn ? Icons.volume_up : Icons.volume_down,
            label: _speakerphoneOn ? 'Speaker On' : 'Speaker Off',
            color: _speakerphoneOn ? AppColors.primary : Colors.black54,
            bg: _speakerphoneOn ? AppColors.primary.withValues(alpha: 0.12) : const Color(0xFFF0F0F0),
            onTap: _toggleSpeakerphone,
          ),
          _circleAction(
            icon: Icons.call_end,
            label: 'End Call',
            color: Colors.white,
            bg: AppColors.detected,
            size: 64,
            onTap: _endCall,
          ),
          _circleAction(
            icon: _micMuted ? Icons.mic_off : Icons.mic,
            label: _micMuted ? 'Muted' : 'Mic On',
            color: _micMuted ? Colors.black54 : AppColors.primary,
            bg: _micMuted ? const Color(0xFFF0F0F0) : AppColors.primary.withValues(alpha: 0.12),
            onTap: _toggleMicMute,
          ),
        ]),
        const SizedBox(height: 20),
      ],
    );
  }

  // ---- VIEW 2: INTERACTIVE DIALPAD ----
  Widget _buildDialpadView() {
    return Column(
      children: [
        if (!_isDefaultDialer)
          Container(
            margin: const EdgeInsets.fromLTRB(16, 12, 16, 4),
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            decoration: BoxDecoration(
              color: AppColors.primary.withValues(alpha: 0.08),
              borderRadius: BorderRadius.circular(12),
              border: Border.all(color: AppColors.primary.withValues(alpha: 0.3)),
            ),
            child: Row(
              children: [
                const Icon(Icons.shield_outlined, color: AppColors.primary, size: 22),
                const SizedBox(width: 10),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Set Vaani as Default Phone App',
                        style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.primary),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Enables native in-call detection & audio recording',
                        style: TextStyle(fontSize: 10, color: Colors.black54),
                      ),
                    ],
                  ),
                ),
                TextButton(
                  onPressed: _requestDefaultDialer,
                  style: TextButton.styleFrom(
                    backgroundColor: AppColors.primary,
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                    minimumSize: Size.zero,
                    tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  ),
                  child: const Text('Set Default', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700)),
                ),
              ],
            ),
          ),

        // Number display box
        Expanded(
          flex: 2,
          child: Container(
            alignment: Alignment.center,
            padding: const EdgeInsets.symmetric(horizontal: 24),
            child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
              Text(
                _dialNumber.isEmpty ? 'Enter Number' : _dialNumber,
                style: TextStyle(
                  fontSize: _dialNumber.length > 10 ? 28 : 34,
                  fontWeight: FontWeight.w800,
                  color: _dialNumber.isEmpty ? Colors.black38 : AppColors.textPrimary,
                  letterSpacing: 1.5,
                ),
                maxLines: 1,
                overflow: TextOverflow.ellipsis,
              ),
              const SizedBox(height: 8),
              Text('Vaani will monitor audio via on-device TFLite',
                  style: TextStyle(fontSize: 12, color: Colors.black.withValues(alpha: 0.5))),
            ]),
          ),
        ),

        // Dialpad Grid (0-9, *, #)
        Expanded(
          flex: 6,
          child: Container(
            padding: const EdgeInsets.symmetric(horizontal: 28),
            child: Column(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
              _dialRow(['1', '2', '3'], ['', 'ABC', 'DEF']),
              _dialRow(['4', '5', '6'], ['GHI', 'JKL', 'MNO']),
              _dialRow(['7', '8', '9'], ['PQRS', 'TUV', 'WXYZ']),
              _dialRow(['*', '0', '#'], ['', '+', '']),
            ]),
          ),
        ),

        // Bottom action row: Live Mic Test, Call Button, Backspace
        Padding(
          padding: const EdgeInsets.only(bottom: 24, left: 32, right: 32),
          child: Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
            // Live Mic mode button
            IconButton(
              tooltip: 'Start Live Mic Acoustic Test',
              icon: const Icon(Icons.mic, size: 28, color: AppColors.primary),
              onPressed: _toggleLiveMic,
            ),

            // Big Green Call Button
            GestureDetector(
              onTap: _startOutgoingCall,
              child: Container(
                width: 68,
                height: 68,
                decoration: const BoxDecoration(
                  color: Color(0xFF2E7D32),
                  shape: BoxShape.circle,
                  boxShadow: [BoxShadow(color: Colors.black26, blurRadius: 10, offset: Offset(0, 4))],
                ),
                child: const Icon(Icons.phone, color: Colors.white, size: 32),
              ),
            ),

            // Backspace button
            IconButton(
              tooltip: 'Delete',
              icon: const Icon(Icons.backspace_outlined, size: 28, color: Colors.black54),
              onPressed: _onBackspace,
              onLongPress: () => setState(() => _dialNumber = ''),
            ),
          ]),
        ),
      ],
    );
  }

  Widget _dialRow(List<String> digits, List<String> subs) {
    return Row(mainAxisAlignment: MainAxisAlignment.spaceEvenly, children: [
      for (int i = 0; i < 3; i++)
        _dialKey(digits[i], subs[i]),
    ]);
  }

  Widget _dialKey(String digit, String sub) {
    return InkWell(
      onTap: () => _onDigitPress(digit),
      onLongPress: digit == '0' ? () => _onDigitPress('+') : null,
      borderRadius: BorderRadius.circular(40),
      child: Container(
        width: 72,
        height: 72,
        decoration: BoxDecoration(
          color: Colors.white,
          shape: BoxShape.circle,
          boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 6)],
        ),
        child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
          Text(digit, style: const TextStyle(fontSize: 26, fontWeight: FontWeight.w700, color: AppColors.textPrimary)),
          if (sub.isNotEmpty)
            Text(sub, style: const TextStyle(fontSize: 9, fontWeight: FontWeight.w600, color: Colors.black45, letterSpacing: 1.0)),
        ]),
      ),
    );
  }

  Widget _circleAction({
    required IconData icon,
    required String label,
    required Color color,
    required Color bg,
    double size = 52,
    required VoidCallback onTap,
  }) {
    return Column(children: [
      InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(size / 2),
        child: Container(
          width: size,
          height: size,
          decoration: BoxDecoration(color: bg, shape: BoxShape.circle),
          child: Icon(icon, color: color, size: size * 0.48),
        ),
      ),
      const SizedBox(height: 6),
      Text(label, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w600, color: Colors.black54)),
    ]);
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

