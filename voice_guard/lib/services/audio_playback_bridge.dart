import 'package:flutter/services.dart';

/// Dart side of MainActivity.kt's "com.voiceguard/playback" MethodChannel.
///
/// Plays 16kHz mono 16-bit PCM audio through the phone's speaker in real time
/// while [AudioService.scanAudioFile] analyzes it, so the user can hear what
/// the model is testing. Supports mute/unmute so scoring continues but audio
/// is silenced. Ported from vaani/mobile's AudioPlaybackBridge.
///
/// Usage:
/// ```dart
/// final playback = AudioPlaybackBridge();
/// await playback.start();
/// await playback.playChunk(chunk); // List<int> 16-bit PCM samples
/// await playback.setMuted(true);   // silence playback, scoring continues
/// await playback.stop();
/// ```
class AudioPlaybackBridge {
  static const _channel = MethodChannel('com.voiceguard/playback');

  /// Start the AudioTrack for streaming playback.
  Future<bool> start() async {
    try {
      return await _channel.invokeMethod<bool>('start') ?? false;
    } on PlatformException {
      return false;
    }
  }

  /// Play a chunk of audio. [samples] are floats in [-1, 1]; converted to
  /// 16-bit PCM internally.
  Future<void> playChunk(List<double> samples) async {
    try {
      final pcm16 = samples.map((s) => (s * 32767.0).round()).toList();
      await _channel.invokeMethod('playChunk', {'samples': pcm16});
    } on PlatformException {
      // Best-effort — a dropped playback chunk must never interrupt scoring.
    }
  }

  /// Mute or unmute playback. Scoring continues regardless.
  Future<void> setMuted(bool muted) async {
    try {
      await _channel.invokeMethod('setMuted', {'muted': muted});
    } on PlatformException {
      // Best-effort.
    }
  }

  /// Stop playback and release the AudioTrack.
  Future<void> stop() async {
    try {
      await _channel.invokeMethod('stop');
    } on PlatformException {
      // Best-effort.
    }
  }

  Future<bool> get isPlaying async {
    try {
      return await _channel.invokeMethod<bool>('isPlaying') ?? false;
    } on PlatformException {
      return false;
    }
  }

  Future<bool> get isMuted async {
    try {
      return await _channel.invokeMethod<bool>('isMuted') ?? false;
    } on PlatformException {
      return false;
    }
  }
}
