import 'package:flutter_test/flutter_test.dart';

import 'package:mobile/main.dart';

void main() {
  testWidgets('VaaniApp renders the home screen with capture controls', (tester) async {
    await tester.pumpWidget(const VaaniApp());

    expect(find.text('VAANI (mobile)'), findsOneWidget);
    expect(find.text('Run bundled demo call'), findsOneWidget);
    expect(find.text('Import audio file'), findsOneWidget);
    expect(find.text('Start mic capture'), findsOneWidget);
  });
}
