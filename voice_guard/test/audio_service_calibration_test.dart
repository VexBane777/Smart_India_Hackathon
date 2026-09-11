import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/audio_service.dart';

void main() {
  group('CalibrationSample.fromScores', () {
    test('computes mean and population std over collected raw scores', () {
      final sample = CalibrationSample.fromScores([0.10, 0.20, 0.12, 0.18]);
      expect(sample.windowsCaptured, 4);
      expect(sample.mean, closeTo(0.15, 1e-9));
      // population std of [0.10,0.20,0.12,0.18] around mean 0.15
      expect(sample.std, closeTo(0.0412310563, 1e-6));
    });

    test('empty input yields a zero sample rather than throwing', () {
      final sample = CalibrationSample.fromScores(const []);
      expect(sample.windowsCaptured, 0);
      expect(sample.mean, 0.0);
      expect(sample.std, 0.0);
    });
  });
}
