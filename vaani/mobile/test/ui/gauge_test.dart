import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/decision/decision_engine.dart';
import 'package:mobile/ui/gauge.dart';

void main() {
  testWidgets('alert state shows the ALERT label and backend banner', (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: RiskGauge(ema: 0.92, state: AlertState.alert, raw: 0.95, backendLabel: 'stub (simulated)'),
    ));
    expect(find.textContaining('ALERT'), findsOneWidget);
    expect(find.textContaining('stub (simulated)'), findsOneWidget);
    expect(find.textContaining('92%'), findsOneWidget);
  });

  testWidgets('null ema renders 0% without crashing', (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: RiskGauge(ema: null, state: AlertState.normal, backendLabel: 'stub'),
    ));
    expect(find.textContaining('0%'), findsOneWidget);
  });
}
