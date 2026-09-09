import 'dart:async';
import 'dart:io';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import '../capture/audio_capture_bridge.dart';
import '../capture/audio_decode_bridge.dart';
import '../decision/decision_engine.dart';
import '../mel/mel_bridge.dart';
import '../pipeline/window_pipeline.dart';
import '../scoring/stub_scorer.dart';
import 'gauge.dart';
import 'occlusion_overlay.dart';
import 'risk_curve.dart';
import 'spectrogram_view.dart';

/// Demo clone-entry time for the bundled call_A_whatsapp.wav asset,
/// matching assets/raw/call_scripts.json's segment 2 start (22.0s) — see
/// app/engine_mock.py's CLONE_ENTRY_S. Only used for the bundled canned
/// demo; real imported/mic audio always uses cloneEntryS: null (honest
/// default — no known ground truth).
const double _demoCloneEntryS = 22.0;

/// Phase 1 HomeScreen (Module E, minus the bank HOLD/OTP/release panel —
/// deliberately deferred to after the demo; see plan Task 12). Everything
/// else from the plan's Task 13 is wired: file import, mic capture, the
/// bundled canned demo, gauge, spectrogram, risk curve, occlusion overlay.
class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key, this.chunkDelay = const Duration(milliseconds: 500)});

  /// Test-only override so integration tests don't have to wait in real
  /// time for a canned demo file to "play" hop-by-hop.
  final Duration chunkDelay;

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _decodeBridge = AudioDecodeBridge();
  final _micBridge = AudioCaptureBridge();
  FrameResult? _latest;
  final List<double> _emaHistory = [];
  StreamSubscription<FrameResult>? _sub;

  Future<void> _runBundledDemo() async {
    final bytes = await rootBundle.load('assets/demo/call_A_whatsapp.wav');
    final dir = await getTemporaryDirectory();
    final file = await File('${dir.path}/call_A_whatsapp.wav').writeAsBytes(bytes.buffer.asUint8List());
    await _runFile(file.path, cloneEntryS: _demoCloneEntryS);
  }

  Future<void> _importFile() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (result == null || result.files.single.path == null) return;
    await _runFile(result.files.single.path!, cloneEntryS: null);
  }

  Future<void> _runFile(String path, {required double? cloneEntryS}) async {
    Float32List pcm;
    try {
      pcm = await _decodeBridge.decodeFile(path);
    } on AudioDecodeException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not read file: $e')));
      }
      return;
    }
    final pipeline = WindowPipeline(mel: MelBridge());
    final scorer = StubScorer(cloneEntryS: cloneEntryS);
    const hopSamples = 8000;
    final controller = StreamController<Float32List>();
    await _sub?.cancel();
    _sub = pipeline.process(controller.stream, scorer: scorer).listen((r) {
      setState(() {
        _latest = r;
        _emaHistory.add(r.ema);
      });
    });
    for (var i = 0; i + hopSamples <= pcm.length; i += hopSamples) {
      controller.add(Float32List.sublistView(pcm, i, i + hopSamples));
      await Future.delayed(widget.chunkDelay);
    }
    await controller.close();
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
            Wrap(spacing: 8, children: [
              ElevatedButton(onPressed: _runBundledDemo, child: const Text('Run bundled demo call')),
              ElevatedButton(onPressed: _importFile, child: const Text('Import audio file')),
              ElevatedButton(onPressed: _startMic, child: const Text('Start mic capture')),
              OutlinedButton(onPressed: _stopMic, child: const Text('Stop mic capture')),
            ]),
            const SizedBox(height: 16),
            RiskGauge(
              ema: _latest?.ema,
              state: state,
              raw: _latest?.rawScore,
              backendLabel: 'stub (simulated — real model pending)',
            ),
            const SizedBox(height: 16),
            SpectrogramView(melDb: _latest?.melDb ?? const []),
            const SizedBox(height: 16),
            RiskCurve(emaHistory: _emaHistory, threshold: 0.6),
            const SizedBox(height: 16),
            const OcclusionOverlay(highlights: [], windowDurationS: 2.0),
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
