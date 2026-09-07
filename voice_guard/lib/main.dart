import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'app.dart';
import 'providers/settings_provider.dart';
import 'providers/call_state_provider.dart';
import 'providers/risk_score_provider.dart';
import 'services/tflite_service.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final settings = SettingsProvider();
  await settings.load();
  final tflite = TFLiteService();
  // Don't block startup on model load.
  tflite.init();
  runApp(
    MultiProvider(
      providers: [
        ChangeNotifierProvider.value(value: settings),
        ChangeNotifierProvider(create: (_) => CallStateProvider()),
        ChangeNotifierProvider(create: (_) => RiskScoreProvider()),
        Provider<TFLiteService>.value(value: tflite),
      ],
      child: const VaaniApp(),
    ),
  );
}
