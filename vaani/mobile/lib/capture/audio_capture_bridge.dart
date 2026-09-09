import 'dart:async';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

class AudioCaptureBridge {
  static const _channel = EventChannel('vaani/mic');
  StreamSubscription? _sub;

  Future<bool> requestPermission() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }

  Stream<Float32List> start() {
    return _channel.receiveBroadcastStream().map((event) {
      final list = (event as List).map((v) => (v as num).toDouble()).toList();
      return Float32List.fromList(list);
    });
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
  }
}
