import 'dart:math' as math;
import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// shadcn/ui-styled circular risk meter for real-time AI voice clone scoring.
class ShadRiskMeter extends StatelessWidget {
  final double score; // 0.0 to 1.0
  final bool animate;
  final double size;
  final bool showLegend;

  const ShadRiskMeter({
    super.key,
    required this.score,
    this.animate = true,
    this.size = 200,
    this.showLegend = true,
  });

  static Color colorFor(double s) {
    if (s < 0.30) return ShadTokens.verified;
    if (s < 0.70) return ShadTokens.suspicious;
    return ShadTokens.detected;
  }

  static Color bgFor(double s) {
    if (s < 0.30) return ShadTokens.verifiedBg;
    if (s < 0.70) return ShadTokens.suspiciousBg;
    return ShadTokens.detectedBg;
  }

  static Color borderFor(double s) {
    if (s < 0.30) return ShadTokens.verifiedBorder;
    if (s < 0.70) return ShadTokens.suspiciousBorder;
    return ShadTokens.detectedBorder;
  }

  static String verdictFor(double s) {
    if (s < 0.30) return 'VERIFIED HUMAN';
    if (s < 0.70) return 'SUSPICIOUS';
    return 'AI DETECTED';
  }

  @override
  Widget build(BuildContext context) {
    final pct = score.clamp(0.0, 1.0);
    final color = colorFor(pct);
    final bg = bgFor(pct);
    final borderColor = borderFor(pct);
    final verdict = verdictFor(pct);

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 18),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(ShadTokens.radius2xl),
        border: Border.all(color: borderColor, width: 1.2),
        boxShadow: ShadTokens.shadowMd,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SizedBox(
            width: size,
            height: size,
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Circular Progress Indicator
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: pct),
                  duration: animate ? ShadTokens.normal : Duration.zero,
                  curve: Curves.easeOutCubic,
                  builder: (context, value, child) {
                    return CustomPaint(
                      size: Size(size, size),
                      painter: _RiskArcPainter(
                        percent: value,
                        color: color,
                        trackColor: color.withValues(alpha: 0.12),
                        strokeWidth: size * 0.075,
                      ),
                    );
                  },
                ),
                // Center metric text
                Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0.0, end: pct),
                      duration: animate ? ShadTokens.normal : Duration.zero,
                      curve: Curves.easeOutCubic,
                      builder: (context, value, child) {
                        return Text(
                          '${(value * 100).round()}%',
                          style: TextStyle(
                            fontSize: size * 0.22,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1.0,
                            color: color,
                            fontFeatures: const [FontFeature.tabularFigures()],
                          ),
                        );
                      },
                    ),
                    Text(
                      'AI RISK SCORE',
                      style: TextStyle(
                        fontSize: size * 0.055,
                        fontWeight: FontWeight.w700,
                        letterSpacing: 1.5,
                        color: color.withValues(alpha: 0.85),
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(height: 14),
          // Verdict Badge
          AnimatedContainer(
            duration: ShadTokens.fast,
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
            decoration: BoxDecoration(
              color: color,
              borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
              boxShadow: [
                BoxShadow(
                  color: color.withValues(alpha: 0.35),
                  blurRadius: 8,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: Text(
              verdict,
              style: const TextStyle(
                color: Colors.white,
                fontSize: 12,
                fontWeight: FontWeight.w800,
                letterSpacing: 0.6,
              ),
            ),
          ),
          if (showLegend) ...[
            const SizedBox(height: 12),
            _buildLegend(),
          ],
        ],
      ),
    );
  }

  Widget _buildLegend() {
    Widget pill(Color c, String text) => Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 7,
              height: 7,
              decoration: BoxDecoration(color: c, shape: BoxShape.circle),
            ),
            const SizedBox(width: 4),
            Text(
              text,
              style: const TextStyle(
                fontSize: 10,
                color: ShadTokens.muted,
                fontWeight: FontWeight.w600,
              ),
            ),
          ],
        );

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        pill(ShadTokens.verified, '0–30%'),
        const SizedBox(width: 12),
        pill(ShadTokens.suspicious, '31–70%'),
        const SizedBox(width: 12),
        pill(ShadTokens.detected, '71–100%'),
      ],
    );
  }
}

class _RiskArcPainter extends CustomPainter {
  final double percent;
  final Color color;
  final Color trackColor;
  final double strokeWidth;

  _RiskArcPainter({
    required this.percent,
    required this.color,
    required this.trackColor,
    required this.strokeWidth,
  });

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height / 2);
    final radius = (size.width - strokeWidth) / 2;

    // Track
    final trackPaint = Paint()
      ..color = trackColor
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    // We draw an arc of 270 degrees (3/4 circle), open at the bottom
    const startAngle = 0.75 * math.pi;
    const sweepAngle = 1.5 * math.pi;

    canvas.drawArc(
      Rect.fromCircle(center: center, radius: radius),
      startAngle,
      sweepAngle,
      false,
      trackPaint,
    );

    // Active progress
    if (percent > 0.0) {
      final activePaint = Paint()
        ..color = color
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      canvas.drawArc(
        Rect.fromCircle(center: center, radius: radius),
        startAngle,
        sweepAngle * percent,
        false,
        activePaint,
      );
    }
  }

  @override
  bool shouldRepaint(covariant _RiskArcPainter old) =>
      old.percent != percent ||
      old.color != color ||
      old.trackColor != trackColor ||
      old.strokeWidth != strokeWidth;
}