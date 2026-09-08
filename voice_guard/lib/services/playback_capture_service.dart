import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'call_service.dart';

/// Dart-side wrapper for the native AudioPlaybackCapture flow (see
/// PlaybackCaptureManager.kt's docstring for why this path exists: it
/// captures a VoIP app's call-playback audio via the public MediaProjection
/// API, as the alternative to mic capture on a real cellular call, which
/// Android blocks for non-privileged apps).
class PlaybackCaptureService {
  static const _method = MethodChannel('com.voiceguard/calls');
  final _consentCtrl = StreamController<bool>.broadcast();
  final CallService? _callService;

  PlaybackCaptureService({CallService? callService}) : _callService = callService {
    _method.setMethodCallHandler(_onMethodCall);
  }

  Future<dynamic> _onMethodCall(MethodCall call) async {
    if (call.method == 'onPlaybackCaptureConsent') {
      final args = call.arguments is Map ? call.arguments as Map : null;
      _consentCtrl.add(args?['granted'] as bool? ?? false);
      return null;
    }
    return _callService?.handleMethodCall(call);
  }

  Stream<bool> get consentResult => _consentCtrl.stream;

  Future<bool> requestConsent() async {
    try {
      final res = await _method.invokeMethod<bool>('requestPlaybackCaptureConsent');
      return res ?? false;
    } catch (e) {
      debugPrint('requestPlaybackCaptureConsent failed: $e');
      return false;
    }
  }

  Future<bool> startCapture() async {
    try {
      final res = await _method.invokeMethod<bool>('startPlaybackCapture');
      return res ?? false;
    } catch (e) {
      debugPrint('startPlaybackCapture failed: $e');
      return false;
    }
  }

  Future<void> stopCapture() async {
    try {
      await _method.invokeMethod('stopPlaybackCapture');
    } catch (e) {
      debugPrint('stopPlaybackCapture failed: $e');
    }
  }

  void dispose() => _consentCtrl.close();
}
