import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Abstraction over MelBridge so WindowPipeline (Task 9) can accept either
/// the real native bridge or a test fake.
abstract class MelBridgeLike {
  Future<List<List<double>>> computeMelDb(Float32List pcm, int sr);
}

/// Dart side of MainActivity.kt's "vaani/mel" MethodChannel (Task 5).
class MelBridge implements MelBridgeLike {
  static const _channel = MethodChannel('vaani/mel');

  @override
  Future<List<List<double>>> computeMelDb(Float32List pcm, int sr) async {
    final raw = await _channel.invokeMethod<List<dynamic>>('computeMelDb', {
      'pcm': pcm.toList(),
      'sr': sr,
    });
    return raw!
        .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
        .toList();
  }
}
