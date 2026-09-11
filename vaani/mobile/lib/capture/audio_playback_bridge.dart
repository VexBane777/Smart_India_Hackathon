import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Dart side of AudioPlaybackBridge.kt's "vaani/playback" MethodChannel.
///
/// Plays 16kHz mono 16-bit PCM audio through the phone's speaker in real time
/// while the model analyzes it. Supports mute/unmute so scoring continues but
/// audio is silenced.
///
/// Usage:
/// ```dart
/// final playback = AudioPlaybackBridge();
/// await playback.start();
/// await playback.playChunk(chunk); // 16-bit PCM samples
/// await playback.setMuted(true);   // silence playback
/// await playback.stop();
/// ```
class AudioPlaybackBridge {
  static const _channel = MethodChannel('vaani/playback');

  /// Start the AudioTrack for streaming playback.
  Future<bool> start() async {
    try {
      return await _channel.invokeMethod<bool>('start') ?? false;
    } on PlatformException catch (e) {
      throw PlatformException(
        code: 'playback_start_failed',
        message: e.message,
      );
    }
  }

  /// Play a chunk of 16-bit PCM mono samples.
  /// [samples] is a Float32List of audio in [-1, 1] — will be converted to
  /// 16-bit PCM internally.
  Future<void> playChunk(Float32List samples) async {
    try {
      // Convert float [-1, 1] to 16-bit PCM int samples
      final pcm16 = samples.map((s) => (s * 32767.0).round()).toList();
      await _channel.invokeMethod('playChunk', {'samples': pcm16});
    } on PlatformException catch (e) {
      throw PlatformException(
        code: 'playback_play_failed',
        message: e.message,
      );
    }
  }

  /// Mute or unmute playback. Scoring continues regardless.
  Future<void> setMuted(bool muted) async {
    try {
      await _channel.invokeMethod('setMuted', {'muted': muted});
    } on PlatformException catch (e) {
      throw PlatformException(
        code: 'playback_mute_failed',
        message: e.message,
      );
    }
  }

  /// Stop playback and release the AudioTrack.
  Future<void> stop() async {
    try {
      await _channel.invokeMethod('stop');
    } on PlatformException catch (e) {
      throw PlatformException(
        code: 'playback_stop_failed',
        message: e.message,
      );
    }
  }

  /// Whether playback is currently active.
  Future<bool> get isPlaying async {
    try {
      return await _channel.invokeMethod<bool>('isPlaying') ?? false;
    } on PlatformException {
      return false;
    }
  }

  /// Whether playback is currently muted.
  Future<bool> get isMuted async {
    try {
      return await _channel.invokeMethod<bool>('isMuted') ?? false;
    } on PlatformException {
      return false;
    }
  }
}
