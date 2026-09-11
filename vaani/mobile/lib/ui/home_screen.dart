import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import '../capture/audio_capture_bridge.dart';
import '../capture/audio_decode_bridge.dart';
import '../capture/audio_playback_bridge.dart';
import '../decision/decision_engine.dart';
import '../mel/mel_bridge.dart';
import '../pipeline/window_pipeline.dart';
import '../scoring/stub_scorer.dart';
import 'gauge.dart';
import 'occlusion_overlay.dart';
import 'playback_control.dart';
import 'risk_curve.dart';
import 'spectrogram_view.dart';

/// Demo clone-entry time for the bundled call_A_whatsapp.wav asset,
/// matching assets/raw/call_scripts.json's segment 2 start (22.0s) � see
/// app/engine_mock.py's CLONE_ENTRY_S. Only used for the bundled canned
/// demo; real imported/mic audio always uses cloneEntryS: null (honest
/// default � no known ground truth).
const double _demoCloneEntryS = 22.0;

/// Phase 1 HomeScreen (Module E, minus the bank HOLD/OTP/release panel).
/// NEW (this session): real-time audio playback during file analysis with
/// mute toggle, so the user hears what the model is testing.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, this.chunkDelay = const Duration(milliseconds: 500)});

  final Duration chunkDelay;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _decodeBridge = AudioDecodeBridge();
  final _micBridge = AudioCaptureBridge();
  final _playbackBridge = AudioPlaybackBridge();
  FrameResult? _latest;
  final List<double> _emaHistory = [];
  StreamSubscription<FrameResult>? _sub;

  // Playback state
  bool _isPlaying = false;
  bool _isMuted = false;
  bool _isProcessing = false;

  Future<void> _runBundledDemo() async {
    final bytes = await rootBundle.load('assets/demo/call_A_whatsapp.wav');
    final dir = await getTemporaryDirectory();
    final file = await File('/call_A_whatsapp.wav').writeAsBytes(bytes.buffer.asUint8List());
    await _runFile(file.path, cloneEntryS: _demoCloneEntryS);
  }

  Future<void> _importFile() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (result == null || result.files.single.path == null) return;
    await _runFile(result.files.single.path!, cloneEntryS: null);
  }

  Future<void> _runFile(String path, {required double? cloneEntryS}) async {
    if (_isProcessing) return;

    Float32List pcm;
    try {
      pcm = await _decodeBridge.decodeFile(path);
    } on AudioDecodeException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not read file: $e')));
      }
      return;
    }

    setState(() {
      _isProcessing = true;
      _isPlaying = true;
      _latest = null;
      _emaHistory.clear();
    });

    // Start playback BEFORE processing so audio plays as it's analyzed
    final playbackStarted = await _playbackBridge.start();
    if (!playbackStarted) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Could not start audio playback')),
        );
      }
    }

    final pipeline = WindowPipeline(mel: MelBridge());
    final scorer = StubScorer(cloneEntryS: cloneEntryS);
    const hopSamples = 8000; // 0.5s at 16kHz
    final controller = StreamController<Float32List>();

    await _sub?.cancel();
    _sub = pipeline.process(controller.stream, scorer: scorer).listen((r) {
      setState(() {
        _latest = r;
        _emaHistory.add(r.ema);
      });
    });

    try {
      // Process audio in 0.5s chunks, playing each chunk as it's analyzed
      for (var i = 0; i + hopSamples <= pcm.length; i += hopSamples) {
        final chunk = Float32List.sublistView(pcm, i, i + hopSamples);

        // Play this chunk through the speaker
        if (!_isMuted) {
          await _playbackBridge.playChunk(chunk);
        }

        // Send chunk to analyzer
        controller.add(chunk);

        // Wait to maintain real-time pacing (1x speed = listen as it plays)
        await Future.delayed(widget.chunkDelay);
      }
    } finally {
      await controller.close();
      await _playbackBridge.stop();
      setState(() {
        _isPlaying = false;
        _isProcessing = false;
      });
    }
  }

  Future<void> _toggleMute() async {
    if (!_isPlaying) return;
    final newMute = !_isMuted;
    await _playbackBridge.setMuted(newMute);
    setState(() => _isMuted = newMute);
  }

  Future<void> _startMic() async {
    final granted = await _micBridge.requestPermission();
    if (!granted) return;
    final pipeline = WindowPipeline(mel: MelBridge());
    final scorer = StubScorer(cloneEntryS: null);
    await _sub?.cancel();
    _sub = pipeline.process(_micBridge.start(), scorer: scorer).listen((r) {
      setState(() {
        _latest = r;
        _emaHistory.add(r.ema);
      });
    });
  }

  Future<void> _stopMic() async {
    await _micBridge.stop();
    await _sub?.cancel();
  }

  @override
  void dispose() {
    _sub?.cancel();
    _playbackBridge.stop();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = _latest?.state ?? AlertState.normal;
    return Scaffold(
      appBar: AppBar(title: const Text('VAANI (mobile)')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Playback control bar
            PlaybackControl(
              isPlaying: _isPlaying,
              isMuted: _isMuted,
              isProcessing: _isProcessing,
              onMuteToggle: _toggleMute,
            ),
            const SizedBox(height: 16),

            // Action buttons
            Wrap(spacing: 8, children: [
              ElevatedButton(
                onPressed: _isProcessing ? null : _runBundledDemo,
                child: const Text('Run bundled demo call'),
              ),
              ElevatedButton(
                onPressed: _isProcessing ? null : _importFile,
                child: const Text('Import audio file'),
              ),
              ElevatedButton(
                onPressed: _isProcessing ? null : _startMic,
                child: const Text('Start mic capture'),
              ),
              OutlinedButton(
                onPressed: _stopMic,
                child: const Text('Stop mic capture'),
              ),
            ]),
            const SizedBox(height: 16),

            // Risk gauge
            RiskGauge(
              ema: _latest?.ema,
              state: state,
              raw: _latest?.rawScore,
              backendLabel: 'stub (simulated � real model pending)',
            ),
            const SizedBox(height: 16),

            // Spectrogram
            SpectrogramView(melDb: _latest?.melDb ?? const []),
            const SizedBox(height: 16),

            // Risk curve
            RiskCurve(emaHistory: _emaHistory, threshold: 0.6),
            const SizedBox(height: 16),

            // Occlusion overlay
            const OcclusionOverlay(highlights: [], windowDurationS: 2.0),
            const SizedBox(height: 16),

            // Status text
            Text(
              _isProcessing
                  ? 'Analyzing... playing through speaker'
                  : 'Ready � import a file or use the mic',
              style: TextStyle(
                color: _isProcessing ? Colors.green : Color(0xFF94A3B8),
                fontSize: 13,
                fontWeight: FontWeight.w500,
              ),
            ),
            const SizedBox(height: 16),

            const Text(
              'Bank transfer HOLD -> OTP -> release simulation is deferred to '
              'after the demo (Module E Task 12).',
              style: TextStyle(color: Color(0xFF94A3B8), fontSize: 12),
            ),
          ],
        ),
      ),
    );
  }
}
