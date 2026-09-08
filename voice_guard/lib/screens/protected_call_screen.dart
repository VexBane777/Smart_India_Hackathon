import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import '../services/signaling_service.dart';
import '../services/webrtc_call_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/risk_meter.dart';

class ProtectedCallScreen extends StatefulWidget {
  const ProtectedCallScreen({super.key});
  @override
  State<ProtectedCallScreen> createState() => _ProtectedCallScreenState();
}

class _ProtectedCallScreenState extends State<ProtectedCallScreen> {
  final _roomController = TextEditingController();
  WebRtcCallService? _call;
  RTCPeerConnectionState _state = RTCPeerConnectionState.RTCPeerConnectionStateNew;

  Future<void> _connect({required bool isCaller}) async {
    final roomId = _roomController.text.trim();
    if (roomId.isEmpty) return;
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();
    risk.reset();
    audio.clearBuffer();
    audio.startScoring();

    final signaling = SignalingService.connect(roomId);
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
    audio.stopScoring();
    setState(() => _call = null);
  }

  @override
  void dispose() {
    _call?.dispose();
    _roomController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.0;

    return Scaffold(
      appBar: AppBar(title: const Text('Protected Call (VAANI-to-VAANI)')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          const Text(
            'Both phones must have VoiceGuard installed and share the same '
            'room code, entered on the same Wi-Fi network.',
            style: TextStyle(fontSize: 12),
          ),
          const SizedBox(height: 12),
          TextField(
            controller: _roomController,
            decoration: const InputDecoration(labelText: 'Room code'),
          ),
          const SizedBox(height: 12),
          if (_call == null)
            Row(children: [
              Expanded(child: ElevatedButton(onPressed: () => _connect(isCaller: true), child: const Text('Call'))),
              const SizedBox(width: 8),
              Expanded(child: ElevatedButton(onPressed: () => _connect(isCaller: false), child: const Text('Answer'))),
            ])
          else ...[
            Text('State: $_state'),
            const SizedBox(height: 20),
            RiskMeter(score: score),
            const SizedBox(height: 20),
            ElevatedButton(onPressed: _hangUp, child: const Text('Hang Up')),
          ],
        ]),
      ),
    );
  }
}
