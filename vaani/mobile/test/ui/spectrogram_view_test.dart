import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/ui/spectrogram_view.dart';

void main() {
  testWidgets('renders a CustomPaint for a non-empty mel matrix', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: SpectrogramView(melDb: List.generate(48, (_) => [1.0, 2.0, 3.0])),
    ));
    expect(find.byType(CustomPaint), findsWidgets);
  });

  testWidgets('shows a waiting message for an empty mel matrix', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SpectrogramView(melDb: [])));
    expect(find.textContaining('Waiting'), findsOneWidget);
  });
}
