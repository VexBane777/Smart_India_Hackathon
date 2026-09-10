import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// Real-time animated acoustic waveform visualizer.
class ShadWaveform extends StatelessWidget {
  final List<double> samples;
  final Color color;
  final double height;
  final int barCount;

  const ShadWaveform({
    super.key,
    this.samples = const [],
    this.color = ShadTokens.primary,
    this.height = 56,
    this.barCount = 36,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      height: height,
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 6),
      decoration: BoxDecoration(
        color: ShadTokens.secondary.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
        border: Border.all(color: ShadTokens.border),
      ),
      child: CustomPaint(
        size: Size(double.infinity, height - 12),
        painter: _WaveformPainter(
          samples: samples,
          color: color,
          barCount: barCount,
        ),
      ),
    );
  }
}

class _WaveformPainter extends CustomPainter {
  final List<double> samples;
  final Color color;
  final int barCount;

  _WaveformPainter({
    required this.samples,
    required this.color,
    required this.barCount,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final barPaint = Paint()
      ..color = color
      ..strokeCap = StrokeCap.round
      ..strokeWidth = 3.0;

    final hasRealSamples = samples.isNotEmpty;
    final totalBars = barCount.clamp(16, 64);
    final step = size.width / totalBars;

    for (int i = 0; i < totalBars; i++) {
      double amp;
      if (hasRealSamples) {
        final sampleIdx = (i * samples.length ~/ totalBars).clamp(0, samples.length - 1);
        amp = samples[sampleIdx].abs().clamp(0.04, 1.0);
      } else {
        // Idle breathing oscillation
        final phase = i / totalBars;
        amp = (0.10 + 0.12 * math.sin(phase * 2 * math.pi)).clamp(0.04, 0.4);
      }

      final barH = (amp * size.height).clamp(4.0, size.height);
      final x = step * i + step / 2;
      final y0 = (size.height - barH) / 2;

      canvas.drawLine(
        Offset(x, y0),
        Offset(x, y0 + barH),
        barPaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _WaveformPainter old) =>
      old.samples != samples || old.color != color || old.barCount != barCount;
}