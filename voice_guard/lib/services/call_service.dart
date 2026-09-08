import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

class CaptureStatus {
  final String? source;
  final bool hasSignal;
  const CaptureStatus({required this.source, required this.hasSignal});
}

class CallService {
  static const _method = MethodChannel('com.voiceguard/calls');
  static const _event = EventChannel('com.voiceguard/audio_stream');

  Stream<Uint8List>? _audioStream;

  Future<bool> isDefaultDialer() async {
    try {
      final v = await _method.invokeMethod<bool>('isDefaultDialer');
      return v ?? false;
    } on MissingPluginException {
      return false;
    } catch (_) {
      return false;
    }
  }

  Future<void> setAsDefaultDialer() async {
    try {
      await _method.invokeMethod('setAsDefaultDialer');
    } catch (e) {
      debugPrint('setAsDefaultDialer failed: $e');
    }
  }

  void setCallStateCallback(void Function(String status, String? number) cb) {
    _method.setMethodCallHandler((call) async {
      if (call.method == 'onCallStateChanged') {
        final args = call.arguments is Map ? call.arguments as Map : null;
        final status = args?['status'] as String? ?? 'idle';
        final number = args?['number'] as String?;
        cb(status, number);
      }
    });
  }

  Future<bool> placeCall(String number) async {
    try {
      final res = await _method.invokeMethod<bool>('placeCall', {'number': number});
      return res ?? false;
    } catch (e) {
      debugPrint('placeCall failed: $e');
      return false;
    }
  }

  Future<void> endCall() async {
    try {
      await _method.invokeMethod('endCall');
    } catch (e) {
      debugPrint('endCall failed: $e');
    }
  }

  Future<bool> toggleSpeakerphone(bool enable) async {
    try {
      final res = await _method.invokeMethod<bool>('toggleSpeakerphone', {'enable': enable});
      return res ?? false;
    } catch (e) {
      debugPrint('toggleSpeakerphone failed: $e');
      return false;
    }
  }

  Future<bool> toggleMicMute(bool muted) async {
    try {
      final res = await _method.invokeMethod<bool>('toggleMicMute', {'muted': muted});
      return res ?? false;
    } catch (e) {
      debugPrint('toggleMicMute failed: $e');
      return false;
    }
  }

  Future<void> startCallDetection() async {
    try {
      await _method.invokeMethod('startCallDetection');
    } catch (e) {
      debugPrint('startCallDetection failed: $e');
    }
  }

  Future<void> stopCallDetection() async {
    try {
      await _method.invokeMethod('stopCallDetection');
    } catch (e) {
      debugPrint('stopCallDetection failed: $e');
    }
  }

  Future<String?> getLastRecordingPath() async {
    try {
      final path = await _method.invokeMethod<String>('getLastRecordingPath');
      return path;
    } catch (e) {
      debugPrint('getLastRecordingPath failed: $e');
      return null;
    }
  }

  /// Reports what AudioCaptureManager is actually reading right now: which
  /// AudioSource cascade step won, and whether recent reads carried real
  /// signal or the OS is silently zero-filling them (see AudioCaptureManager's
  /// docstring for why that distinction matters on a live cellular call).
  Future<CaptureStatus> getCaptureStatus() async {
    try {
      final res = await _method.invokeMethod<Map>('getCaptureStatus');
      return CaptureStatus(
        source: res?['source'] as String?,
        hasSignal: res?['hasSignal'] as bool? ?? false,
      );
    } catch (_) {
      return const CaptureStatus(source: null, hasSignal: false);
    }
  }

  Future<bool> hasOverlayPermission() async {
    try {
      final v = await _method.invokeMethod<bool>('hasOverlayPermission');
      return v ?? false;
    } catch (_) {
      return false;
    }
  }

  Future<void> requestOverlayPermission() async {
    try {
      await _method.invokeMethod('requestOverlayPermission');
    } catch (e) {
      debugPrint('requestOverlayPermission failed: $e');
    }
  }

  Future<void> showOverlay({required double riskScore, required String verdict}) async {
    try {
      await _method.invokeMethod('showOverlay', {'riskScore': riskScore, 'verdict': verdict});
    } catch (e) {
      debugPrint('showOverlay failed: $e');
    }
  }

  Future<void> hideOverlay() async {
    try {
      await _method.invokeMethod('hideOverlay');
    } catch (_) {}
  }

  Stream<Uint8List> get audioStream {
    _audioStream ??= _event.receiveBroadcastStream().map((e) {
      if (e is Uint8List) return e;
      if (e is List<int>) return Uint8List.fromList(e);
      return Uint8List(0);
    }).handleError((e) => debugPrint('audioStream error: $e'));
    return _audioStream!;
  }

  // For demo without native: emit silence.
  Stream<Uint8List> get mockAudioStream =>
      Stream.periodic(const Duration(milliseconds: 200), (_) => Uint8List(3200));
}
