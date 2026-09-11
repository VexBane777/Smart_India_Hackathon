import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/providers/calibration_provider.dart';

void main() {
  group('CalibrationProvider.computeThreshold', () {
    test('uncalibrated: returns sensitivity unchanged', () {
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: false,
        baselineMean: 0.0,
        populationMean: 0.15,
      );
      expect(t, 0.60);
    });

    test('speaker baseline above population mean raises the threshold', () {
      // A naturally "busier"-sounding speaker (higher raw scores at
      // baseline) needs a higher bar before being flagged.
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.30, // 0.15 above the assumed population mean
        populationMean: 0.15,
      );
      expect(t, closeTo(0.75, 1e-9));
    });

    test('speaker baseline below population mean lowers the threshold', () {
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.05,
        populationMean: 0.15,
      );
      expect(t, closeTo(0.50, 1e-9));
    });

    test('clamps to [minThreshold, maxThreshold] so calibration can never '
        'disable detection or make it impossible to alert', () {
      final low = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.99,
        populationMean: 0.15,
      );
      expect(low, CalibrationProvider.maxThreshold);

      final high = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: -0.5, // pathological, shouldn't occur, but must clamp
        populationMean: 0.15,
      );
      expect(high, CalibrationProvider.minThreshold);
    });
  });
}
