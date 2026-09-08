import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'app.dart';
import 'providers/settings_provider.dart';
import 'providers/call_state_provider.dart';
import 'providers/risk_score_provider.dart';
import 'services/tflite_service.dart';

import 'services/audio_service.dart';
import 'services/call_service.dart';
import 'services/notification_service.dart';
import 'services/playback_capture_service.dart';
import 'models/call_state.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = SettingsProvider();
  await settings.load();
  final tflite = TFLiteService();
  tflite.init();

  final calls = CallService();
  final audio = AudioService(tflite);
  final notifications = NotificationService();
  final callStateProvider = CallStateProvider();
  final riskScoreProvider = RiskScoreProvider();

  // Listen to native Android telecom call state changes
  calls.setCallStateCallback((status, number) {
    if (status == 'active') {
      audio.clearBuffer();
      audio.startScoring();
      callStateProvider.setStatus(CallStatus.active, number: number);
    } else if (status == 'dialing') {
      callStateProvider.setStatus(CallStatus.dialing, number: number);
    } else if (status == 'incoming') {
      callStateProvider.setStatus(CallStatus.incoming, number: number);
    } else if (status == 'holding') {
      callStateProvider.setStatus(CallStatus.holding, number: number);
    } else if (status == 'disconnected') {
      audio.stopScoring();
      callStateProvider.setStatus(CallStatus.disconnected, number: number);
    } else {
      callStateProvider.setStatus(CallStatus.idle);
    }
  });

  // Pipe real audio bytes from Android EventChannel to AudioService (also
  // used by PlaybackCaptureManager on the VoIP-protection path, which
  // pushes bytes through the same event channel/eventSink).
  calls.audioStream.listen((bytes) {
    audio.ingestBytes(bytes);
  }, onError: (e) {
    debugPrint('AudioStream listener: $e');
  });

  // Single registered handler for the shared com.voiceguard/calls
  // MethodChannel — delegates onCallStateChanged to CallService. Must be
  // constructed after calls.setCallStateCallback() above.
  final playbackCapture = PlaybackCaptureService(callService: calls);

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: settings),
        ChangeNotifierProvider.value(value: callStateProvider),
        ChangeNotifierProvider.value(value: riskScoreProvider),
        Provider<TFLiteService>.value(value: tflite),
        Provider<CallService>.value(value: calls),
        Provider<AudioService>.value(value: audio),
        Provider<NotificationService>.value(value: notifications),
        Provider<PlaybackCaptureService>.value(value: playbackCapture),
      ],
      child: const VaaniApp(),
    ),
  );
}
