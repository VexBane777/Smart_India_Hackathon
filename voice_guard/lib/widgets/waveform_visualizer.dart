import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../utils/constants.dart';

class WaveformVisualizer extends StatelessWidget {
  final List<double> samples; // normalized -1..1 or 0..1
  final Color color;
  const WaveformVisualizer({super.key, this.samples = const [], this.color = AppColors.primary});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: const Size(double.infinity, 56),
      painter: _WavePainter(samples: samples, color: color),
    );
  }
}

class _WavePainter extends CustomPainter {
  final List<double> samples;
  final Color color;
  _WavePainter({required this.samples, required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final bg = Paint()..color = Colors.black.withValues(alpha: 0.04);
    final r = RRect.fromRectAndRadius(Offset.zero & size, const Radius.circular(12));
    canvas.drawRRect(r, bg);

    final barPaint = Paint()..color = color.withValues(alpha: 0.85)..strokeCap = StrokeCap.round..strokeWidth = 3;
    final data = samples.isEmpty
        ? List.generate(28, (i) => 0.12 + 0.18 * math.sin(i * 0.9) + 0.08 * math.cos(i * 1.7))
        : samples;

    final n = data.length.clamp(1, 48);
    final step = size.width / (n + 1);
    for (int i = 0; i < n; i++) {
      final v = data[i].abs().clamp(0.06, 1.0);
      final h = v * (size.height * 0.82);
      final x = step * (i + 1);
      final y0 = (size.height - h) / 2;
      canvas.drawLine(Offset(x, y0), Offset(x, y0 + h), barPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _WavePainter old) => old.samples != samples || old.color != color;
}
