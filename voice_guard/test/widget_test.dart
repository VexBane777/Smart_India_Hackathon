import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:voice_guard/app.dart';
import 'package:voice_guard/providers/call_state_provider.dart';
import 'package:voice_guard/providers/risk_score_provider.dart';
import 'package:voice_guard/providers/settings_provider.dart';
import 'package:voice_guard/services/tflite_service.dart';

void main() {
  testWidgets('Vaani app builds without crashing', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({'onboardingDone': true});
    final settings = SettingsProvider();
    await settings.load();
    final tflite = TFLiteService();

    await tester.pumpWidget(
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
    await tester.pumpAndSettle();
    // Home AppBar title after the PR #8 Figma redesign ("SYSTEM STATUS",
    // formerly "Vaani")
    expect(find.text('SYSTEM STATUS'), findsOneWidget);
  });

  testWidgets('Providers wire correctly', (WidgetTester tester) async {
    SharedPreferences.setMockInitialValues({});
    final settings = SettingsProvider();
    await settings.load();
    expect(settings, isA<SettingsProvider>());
    expect(settings.protectionEnabled, isTrue);
  });
}
