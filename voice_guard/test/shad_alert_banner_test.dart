import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/design/tokens.dart';
import 'package:voice_guard/widgets/shad_alert_banner.dart';

void main() {
  testWidgets('normal state shows Verified Human in the verified color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'normal')));
    expect(find.text('Verified Human'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.verifiedBg);
  });

  testWidgets('warn state shows Suspicious in the suspicious color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'warn')));
    expect(find.text('Suspicious'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.suspiciousBg);
  });

  testWidgets('alert state shows Synthetic Clone Detected in the detected color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'alert')));
    expect(find.text('Synthetic Clone Detected'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.detectedBg);
  });
}
