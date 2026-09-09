import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/ui/home_screen.dart';

/// Bank HOLD->OTP->release is Module E Task 12, deliberately deferred to
/// after the demo — this integration test only exercises the pipeline
/// through the alert, not a bank transfer.
void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('running the bundled demo call eventually shows ALERT',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: HomeScreen(chunkDelay: Duration.zero),
    ));
    await tester.tap(find.text('Run bundled demo call'));
    await tester.pump();

    // The demo call's clone segment starts at 22s of audio; with
    // chunkDelay: Duration.zero the pipeline races through hop chunks as
    // fast as the test scheduler allows rather than in real time.
    await tester.pumpAndSettle(const Duration(seconds: 5));

    expect(find.textContaining('ALERT'), findsOneWidget);
  });
}
