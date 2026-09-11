import 'dart:async';
import 'package:flutter/foundation.dart';
import '../utils/audio_processor.dart';
import 'tflite_service.dart';

import 'dart:math' as math;

/// Result of a calibration capture — mean/std of raw (pre-EMA) model
/// scores over N consecutive 3s windows of the user's own voice.
class CalibrationSample {
  final double mean;
  final double std;
  final int windowsCaptured;
  const CalibrationSample({required this.mean, required this.std, required this.windowsCaptured});

  factory CalibrationSample.fromScores(List<double> scores) {
    if (scores.isEmpty) {
      return const CalibrationSample(mean: 0.0, std: 0.0, windowsCaptured: 0);
    }
    final mean = scores.reduce((a, b) => a + b) / scores.length;
    final variance = scores.map((s) => (s - mean) * (s - mean)).reduce((a, b) => a + b) / scores.length;
    return CalibrationSample(mean: mean, std: math.sqrt(variance), windowsCaptured: scores.length);
  }
}

/// Buffers raw PCM16 frames from the native EventChannel and emits risk scores.
/// Raw PCM is held only in RAM and discarded after feature extraction.
class AudioService {
  final TFLiteService tflite;
  AudioService(this.tflite);

  final List<double> _buffer = [];
  Timer? _timer;
  bool _scoring = false;
  bool _fileScanning = false;
  final _scoreCtrl = StreamController<double>.broadcast();
  final _pcmCtrl = StreamController<List<double>>.broadcast();
  final _rmsCtrl = StreamController<double>.broadcast();
  final _signalCtrl = StreamController<bool>.broadcast();
  final _attackTypeCtrl = StreamController<(String?, double)>.broadcast();

  Stream<double> get scoreStream => _scoreCtrl.stream;
  Stream<List<double>> get pcmStream => _pcmCtrl.stream;
  Stream<double> get rmsStream => _rmsCtrl.stream;
  /// (attackType, confidence) alongside scoreStream — attackType is 'tts',
  /// 'vc', or null (heuristic fallback / no ONNX session loaded). UI layers
  /// decide their own confidence-gating threshold (see call_screen.dart);
  /// this stream reports the raw model output, unfiltered.
  Stream<(String?, double)> get attackTypeStream => _attackTypeCtrl.stream;
  /// Whether the most recently *scored* window carried real signal, as
  /// opposed to silence/OS zero-filled reads (see AudioCaptureManager's
  /// docstring — some OEM builds zero-fill AudioRecord reads once the real
  /// call audio path takes over rather than erroring). Consumers should show
  /// "no voice detected" instead of trusting a score during a false stretch,
  /// since a window of exact/near-zero PCM deterministically collapses to
  /// the same LFCC features every time and would otherwise look like the
  /// model froze.
  Stream<bool> get hasSignalStream => _signalCtrl.stream;

  // Matches AudioCaptureManager's own SILENCE_RMS_THRESHOLD (50.0 on the
  // int16 PCM domain) converted to this class's normalized [-1, 1] domain.
  static const double _silenceRmsThreshold = 50.0 / 32768.0;

  void ingestBytes(Uint8List bytes) {
    final samples = AudioProcessor.pcm16ToDouble(bytes);
    if (samples.isEmpty) return;
    _buffer.addAll(samples);
    // keep only last 5s to bound memory
    const maxSamples = 80000;
    if (_buffer.length > maxSamples) {
      _buffer.removeRange(0, _buffer.length - maxSamples);
    }

    // Compute RMS and downsampled waveform bars
    double sum = 0;
    for (final s in samples) {
      sum += s * s;
    }
    final rms = math.sqrt(sum / samples.length);
    _rmsCtrl.add(rms);

    final step = math.max(1, samples.length ~/ 28);
    final wave = <double>[];
    for (int i = 0; i < samples.length && wave.length < 28; i += step) {
      wave.add(samples[i]);
    }
    _pcmCtrl.add(wave);
  }

  void startScoring({Duration interval = const Duration(seconds: 1)}) {
    _timer?.cancel();
    _timer = Timer.periodic(interval, (_) async {
      // Require a full 3s window before scoring at all — a partial window
      // (e.g. the first ~0.3-1s right after clearBuffer(), often containing
      // only a dial tone/ring beep or a transient) produced spurious
      // high-confidence verdicts in testing (an "AI DETECTED" alert fired a
      // second into a call before any real speech had arrived).
      if (_buffer.length < 48000) return;
      // A previous tick's scoreChunk() (feature extraction + inference) is
      // still in flight — skip this tick instead of piling up concurrent
      // isolate spawns / ONNX runs on top of it.
      if (_scoring) return;
      final chunk = _buffer.sublist(_buffer.length - 48000);

      double sumSq = 0;
      for (final s in chunk) { sumSq += s * s; }
      final chunkRms = math.sqrt(sumSq / chunk.length);
      final (peakDb, noiseFloorDb) = _levelDb(chunk);
      debugPrint('Monitor: chunkRms=${chunkRms.toStringAsFixed(6)} (int16-equiv=${(chunkRms * 32768).toStringAsFixed(1)}) '
          'peak=${peakDb.toStringAsFixed(1)}dBFS noise=${noiseFloorDb.toStringAsFixed(1)}dBFS '
          'threshold=${_silenceRmsThreshold.toStringAsFixed(6)} bufLen=${_buffer.length}');
      if (chunkRms < _silenceRmsThreshold) {
        // Silent/zero-filled window: skip scoring rather than feed the model
        // a degenerate all-floor input that would deterministically repeat
        // whatever score silence happens to map to.
        _signalCtrl.add(false);
        return;
      }
      _signalCtrl.add(true);

      _scoring = true;
      try {
        final (score, attackType, attackConfidence) = await tflite.scoreChunk(chunk);
        _scoreCtrl.add(score);
        _attackTypeCtrl.add((attackType, attackConfidence));
      } finally {
        _scoring = false;
      }
    });
  }

  void stopScoring() {
    _timer?.cancel();
    _timer = null;
  }

  void clearBuffer() => _buffer.clear();

  /// Cancels an in-flight [scanAudioFile] loop (checked between windows).
  void stopAudioFileScoring() => _fileScanning = false;

  /// (peakDbFS, noiseFloorDbFS) for a scoring window — cheap strided scan
  /// (every 8th sample; 10th percentile as a noise-floor proxy). Pure
  /// instrumentation for the Monitor/AudioScan logs; never used in the
  /// scoring decision itself.
  (double, double) _levelDb(List<double> chunk) {
    double peak = 0;
    final mags = <double>[];
    for (int i = 0; i < chunk.length; i += 8) {
      final a = chunk[i] < 0 ? -chunk[i] : chunk[i];
      if (a > peak) peak = a;
      mags.add(a);
    }
    mags.sort();
    final noiseFloor = mags[(mags.length * 0.10).floor().clamp(0, mags.length - 1)];
    return (
      20 * math.log(peak + 1e-9) / math.ln10,
      20 * math.log(noiseFloor + 1e-9) / math.ln10,
    );
  }

  /// Scores a 16-bit PCM WAV file through the EXACT same path live audio
  /// takes — `AudioProcessor.extractFeaturesIsolate` -> the same ONNX
  /// [TFLiteService.scoreChunk], 3s windows, the same RMS silence gate —
  /// sliding the window at 0.5 s so a played clip actually moves the needle
  /// during its duration instead of one score per tick.
  ///
  /// The only difference from a real call is the source: the file is decoded
  /// in RAM and never replayed through a speaker, i.e. there is no acoustic
  /// loop for the model to choke on. This is the deterministic demo path
  /// (Module E spec's audio_decode_bridge) — for scoring a user-selected
  /// .wav byte-for-byte like the laptop does.
  ///
  /// Returns null on success, a human-readable error string on failure, or
  /// 'cancelled'. Call [stopAudioFileScoring] from another isolate to cancel.
  Future<String?> scanAudioFile(Uint8List wavBytes) async {
    if (_fileScanning) return 'scan already running';
    final decoded = AudioProcessor.decodePcm16Wav(wavBytes);
    if (decoded == null) return 'only 16-bit PCM .wav files are supported';
    var pcm = decoded.samples;
    if (decoded.sampleRate != AudioProcessor.sampleRate) {
      pcm = AudioProcessor.resampleLinear(pcm, decoded.sampleRate, AudioProcessor.sampleRate);
    }
    if (pcm.isEmpty) return 'empty audio';

    _fileScanning = true;
    const win = AudioProcessor.chunkSamples; // 48000 == 3s @ 16k
    const hop = 8000; // 0.5s sliding hop
    if (pcm.length < win) {
      pcm = [...List<double>.filled(win - pcm.length, 0.0), ...pcm];
    }
    final lastStart = pcm.length - win;
    debugPrint('AudioScan: ${pcm.length} samples (~${(pcm.length / AudioProcessor.sampleRate).toStringAsFixed(1)}s), '
        '${(lastStart ~/ hop) + 1} scoring windows');
    int i = 0;
    while (i <= lastStart && _fileScanning) {
      final chunk = pcm.sublist(i, i + win);
      final windowT = i / AudioProcessor.sampleRate;

      double sumSq = 0;
      for (final s in chunk) { sumSq += s * s; }
      final rms = math.sqrt(sumSq / chunk.length);
      _rmsCtrl.add(rms);
      final (peakDb, noiseFloorDb) = _levelDb(chunk);
      debugPrint('AudioScan: t=${windowT.toStringAsFixed(1)}s '
          'rms=${rms.toStringAsFixed(4)} peak=${peakDb.toStringAsFixed(1)}dBFS noise=${noiseFloorDb.toStringAsFixed(1)}dBFS');

      if (rms >= _silenceRmsThreshold) {
        _signalCtrl.add(true);
        try {
          final (score, attackType, attackConfidence) = await tflite.scoreChunk(chunk);
          _scoreCtrl.add(score);
          _attackTypeCtrl.add((attackType, attackConfidence));
        } catch (e) {
          debugPrint('AudioScan: scoreChunk failed: $e');
        }
      } else {
        _signalCtrl.add(false);
        debugPrint('AudioScan: t=${windowT.toStringAsFixed(1)}s silent — skipped scoring');
      }

      final step = math.max(1, chunk.length ~/ 28);
      final wave = <double>[];
      for (int j = 0; j < chunk.length && wave.length < 28; j += step) {
        wave.add(chunk[j]);
      }
      _pcmCtrl.add(wave);

      await Future.delayed(const Duration(milliseconds: 40));
      i += hop;
    }
    _fileScanning = false;
    debugPrint('AudioScan: finished at t=${(i / AudioProcessor.sampleRate).toStringAsFixed(1)}s');
    return i > lastStart ? null : 'cancelled';
  }

  /// Collects `windows` consecutive raw scores from scoreStream, without
  /// touching RiskScoreProvider (deliberately — see plan Global
  /// Constraints: calibration must never pollute call logs/risk history).
  /// Caller is responsible for having already started native capture
  /// (`CallService.startCallDetection()`) and `startScoring()` — this
  /// method only *listens*, it doesn't start capture, mirroring how
  /// call_screen.dart's Live Mic Test already separates those concerns.
  Future<CalibrationSample> captureCalibrationSample({
    int windows = 8,
    Duration perWindowTimeout = const Duration(seconds: 2),
  }) async {
    final scores = <double>[];
    final sub = scoreStream.listen(scores.add);
    try {
      final deadline = DateTime.now().add(perWindowTimeout * windows);
      while (scores.length < windows && DateTime.now().isBefore(deadline)) {
        await Future.delayed(const Duration(milliseconds: 200));
      }
    } finally {
      await sub.cancel();
    }
    return CalibrationSample.fromScores(scores);
  }

  Future<void> injectBenchmarkTest({required bool isAiVoice}) async {
    final chunk = List<double>.generate(16000, (i) {
      final t = i / 16000.0;
      if (isAiVoice) {
        // Static mechanical waveform typical of vocoders
        return 0.25 * math.sin(2 * math.pi * 180 * t) +
            0.20 * math.sin(2 * math.pi * 360 * t) +
            0.15 * math.sin(2 * math.pi * 540 * t);
      } else {
        // Dynamic human speech with natural frequency modulation and envelope
        final f0 = 120.0 + 15.0 * math.sin(2 * math.pi * 3.5 * t);
        final env = math.max(0.0, math.sin(2 * math.pi * 1.2 * t));
        return env * (0.35 * math.sin(2 * math.pi * f0 * t) +
            0.25 * math.sin(2 * math.pi * (2 * f0) * t) +
            0.05 * (math.Random(i).nextDouble() - 0.5));
      }
    });
    final (score, attackType, attackConfidence) = await tflite.scoreChunk(chunk);
    _scoreCtrl.add(score);
    _attackTypeCtrl.add((attackType, attackConfidence));
    final wave = chunk.take(28).toList();
    _pcmCtrl.add(wave);
    _rmsCtrl.add(isAiVoice ? 0.35 : 0.25);
  }

  void dispose() {
    _timer?.cancel();
    _scoreCtrl.close();
    _pcmCtrl.close();
    _rmsCtrl.close();
    _signalCtrl.close();
    _attackTypeCtrl.close();
  }
}

