import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/ui/risk_curve.dart';

void main() {
  testWidgets('renders a chart for non-empty history', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: RiskCurve(emaHistory: [0.1, 0.2, 0.9], threshold: 0.6),
    ));
    expect(find.byType(RiskCurve), findsOneWidget);
  });

  testWidgets('empty history does not crash', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: RiskCurve(emaHistory: [], threshold: 0.6)));
    expect(tester.takeException(), isNull);
  });
}
