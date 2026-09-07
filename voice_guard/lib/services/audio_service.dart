import 'dart:async';
import 'dart:typed_data';
import '../utils/audio_processor.dart';
import 'tflite_service.dart';

/// Buffers raw PCM16 frames from the native EventChannel and emits risk scores.
/// Raw PCM is held only in RAM and discarded after feature extraction.
class AudioService {
  final TFLiteService tflite;
  AudioService(this.tflite);

  final List<double> _buffer = [];
  Timer? _timer;
  final _scoreCtrl = StreamController<double>.broadcast();
  final _pcmCtrl = StreamController<List<double>>.broadcast();

  Stream<double> get scoreStream => _scoreCtrl.stream;
  Stream<List<double>> get pcmStream => _pcmCtrl.stream;

  void ingestBytes(Uint8List bytes) {
    final samples = AudioProcessor.pcm16ToDouble(bytes);
    _buffer.addAll(samples);
    // keep only last 5s to bound memory
    const maxSamples = 80000;
    if (_buffer.length > maxSamples) {
      _buffer.removeRange(0, _buffer.length - maxSamples);
    }
    _pcmCtrl.add(List<double>.from(_buffer.take(512)));
  }

  void startScoring({Duration interval = const Duration(seconds: 1)}) {
    _timer?.cancel();
    _timer = Timer.periodic(interval, (_) {
      if (_buffer.length < 4800) return; // need ~0.3s minimum
      final chunk = _buffer.length >= 48000
          ? _buffer.sublist(_buffer.length - 48000)
          : List<double>.from(_buffer);
      final score = tflite.scoreChunk(chunk);
      _scoreCtrl.add(score);
    });
  }

  void stopScoring() {
    _timer?.cancel();
    _timer = null;
  }

  void clearBuffer() => _buffer.clear();

  void dispose() {
    _timer?.cancel();
    _scoreCtrl.close();
    _pcmCtrl.close();
  }
}
