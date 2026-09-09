import 'dart:typed_data';
import 'package:flutter/services.dart';

class AudioDecodeException implements Exception {
  AudioDecodeException(this.message);
  final String message;
  @override
  String toString() => message;
}

/// Dart side of AudioDecodeBridge.kt's "vaani/decode" MethodChannel (Task 7).
/// Used for both file-import (arbitrary user audio) and the bundled canned
/// demo asset.
class AudioDecodeBridge {
  static const _channel = MethodChannel('vaani/decode');

  Future<Float32List> decodeFile(String path) async {
    try {
      final raw = await _channel.invokeMethod<List<dynamic>>('decodeFile', {
        'path': path,
      });
      return Float32List.fromList(
          raw!.map((v) => (v as num).toDouble()).toList());
    } on PlatformException catch (e) {
      throw AudioDecodeException(e.message ?? 'failed to decode $path');
    }
  }
}
