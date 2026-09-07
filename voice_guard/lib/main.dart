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

  // Listen to native Android telecom call state changes
  calls.setCallStateCallback((status, number) {
    if (status == 'active') {
      callStateProvider.setStatus(CallStatus.active, number: number);
    } else if (status == 'dialing') {
      callStateProvider.setStatus(CallStatus.dialing, number: number);
    } else if (status == 'incoming') {
      callStateProvider.setStatus(CallStatus.incoming, number: number);
    } else if (status == 'holding') {
      callStateProvider.setStatus(CallStatus.holding, number: number);
    } else if (status == 'disconnected') {
      callStateProvider.setStatus(CallStatus.disconnected, number: number);
    } else {
      callStateProvider.setStatus(CallStatus.idle);
    }
  });

  // Pipe real audio bytes from Android EventChannel to AudioService
  calls.audioStream.listen((bytes) {
    audio.ingestBytes(bytes);
  }, onError: (e) {
    debugPrint('AudioStream listener: $e');
  });

  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: settings),
        ChangeNotifierProvider.value(value: callStateProvider),
        ChangeNotifierProvider(create: (_) => RiskScoreProvider()),
        Provider<TFLiteService>.value(value: tflite),
        Provider<CallService>.value(value: calls),
        Provider<AudioService>.value(value: audio),
        Provider<NotificationService>.value(value: notifications),
      ],
      child: const VaaniApp(),
    ),
  );
}
