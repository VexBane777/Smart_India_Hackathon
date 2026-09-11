import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../design/tokens.dart';
import 'shad_glass_card.dart';

/// A sleek, minimalist speedometer graph that visualizes total acoustic scans
/// and the exact proportion of Human Verified vs. Threats Blocked with
/// precision arc ticks, animated needle, and frosted glass aesthetics.
class ShadSpeedometer extends StatefulWidget {
  final int totalScans;
  final int verifiedCount;
  final int threatCount;
  final double? liveRisk;
  final VoidCallback? onTap;

  const ShadSpeedometer({
    super.key,
    required this.totalScans,
    required this.verifiedCount,
    required this.threatCount,
    this.liveRisk,
    this.onTap,
  });

  @override
  State<ShadSpeedometer> createState() => _ShadSpeedometerState();
}

class _ShadSpeedometerState extends State<ShadSpeedometer>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _controller.forward();
  }

  @override
  void didUpdateWidget(covariant ShadSpeedometer oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.totalScans != widget.totalScans ||
        oldWidget.verifiedCount != widget.verifiedCount ||
        oldWidget.threatCount != widget.threatCount ||
        oldWidget.liveRisk != widget.liveRisk) {
      _controller.forward(from: 0.0);
    }
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final total = widget.totalScans > 0 ? widget.totalScans : 1;
    final verifiedRatio = (widget.verifiedCount / total).clamp(0.0, 1.0);
    final threatRatio = (widget.threatCount / total).clamp(0.0, 1.0);

    // Needle position: indicates live risk if scoring, or the threat ratio (scaled for visible gauge deflection)
    final double targetNeedle = widget.liveRisk != null
        ? widget.liveRisk!.clamp(0.05, 0.95)
        : (threatRatio > 0 ? (threatRatio * 3.0).clamp(0.12, 0.90) : 0.10);

    return ShadGlassCard(
      padding: const EdgeInsets.fromLTRB(20, 16, 20, 18),
      onTap: widget.onTap,
      child: Column(
        children: [
          // Header / Tag
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Row(
                children: [
                  Container(
                    width: 28,
                    height: 28,
                    decoration: BoxDecoration(
                      color: ShadTokens.secondary,
                      borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                    ),
                    child: const Icon(
                      LucideIcons.gauge,
                      size: 15,
                      color: ShadTokens.foreground,
                    ),
                  ),
                  const SizedBox(width: 10),
                  const Text(
                    'Acoustic Telemetry Gauge',
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.2,
                      color: ShadTokens.foreground,
                    ),
                  ),
                ],
              ),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                decoration: BoxDecoration(
                  color: ShadTokens.secondary,
                  borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
                  border: Border.all(color: ShadTokens.border),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Container(
                      width: 5,
                      height: 5,
                      decoration: const BoxDecoration(
                        color: ShadTokens.verified,
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 5),
                    const Text(
                      'Live Ratio',
                      style: TextStyle(
                        fontSize: 10,
                        fontWeight: FontWeight.w700,
                        color: ShadTokens.muted,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 10),

          // Speedometer Custom Canvas
          AnimatedBuilder(
            animation: _animation,
            builder: (context, child) {
              return SizedBox(
                width: double.infinity,
                height: 195,
                child: CustomPaint(
                  painter: _SpeedometerPainter(
                    progress: _animation.value,
                    verifiedRatio: verifiedRatio,
                    threatRatio: threatRatio,
                    needleTarget: targetNeedle,
                    totalScans: widget.totalScans,
                  ),
                ),
              );
            },
          ),

          const SizedBox(height: 6),

          // Proportional Breakdown Legend Cards
          Row(
            children: [
              // Human Verified Proportion
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFF0FDF4),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                    border: Border.all(
                      color: const Color(0xFF86EFAC).withValues(alpha: 0.6),
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: Color(0xFF10B981),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Human Verified',
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: Color(0xFF047857),
                              ),
                            ),
                            Text(
                              '${widget.verifiedCount} (${(verifiedRatio * 100).toInt()}%)',
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w800,
                                color: Color(0xFF065F46),
                                fontFeatures: [FontFeature.tabularFigures()],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 10),
              // Threats Blocked Proportion
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
                  decoration: BoxDecoration(
                    color: const Color(0xFFFFF1F2),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                    border: Border.all(
                      color: const Color(0xFFFECDD3).withValues(alpha: 0.8),
                    ),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: Color(0xFFEF4444),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            const Text(
                              'Threats Blocked',
                              style: TextStyle(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: Color(0xFFB91C1C),
                              ),
                            ),
                            Text(
                              '${widget.threatCount} (${(threatRatio * 100).toInt()}%)',
                              style: const TextStyle(
                                fontSize: 13,
                                fontWeight: FontWeight.w800,
                                color: Color(0xFF991B1B),
                                fontFeatures: [FontFeature.tabularFigures()],
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class _SpeedometerPainter extends CustomPainter {
  final double progress;
  final double verifiedRatio;
  final double threatRatio;
  final double needleTarget;
  final int totalScans;

  _SpeedometerPainter({
    required this.progress,
    required this.verifiedRatio,
    required this.threatRatio,
    required this.needleTarget,
    required this.totalScans,
  });

  // Classic sports gauge arch: starts at 155° (bottom left) and sweeps 230° around the top to 25° (bottom right)
  static const double _startAngle = 155 * (math.pi / 180);
  static const double _sweepAngle = 230 * (math.pi / 180);

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.85);
    final radius = math.min(size.width * 0.40, size.height * 0.58);
    const strokeWidth = 12.0;

    // 1. Outer subtle track
    final trackPaint = Paint()
      ..color = const Color(0xFFE4E4E7).withValues(alpha: 0.65)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    final arcRect = Rect.fromCircle(center: center, radius: radius);
    canvas.drawArc(arcRect, _startAngle, _sweepAngle, false, trackPaint);

    // 2. Tick marks along the arc
    final tickPaint = Paint()
      ..color = const Color(0xFFA1A1AA)
      ..strokeCap = StrokeCap.round;

    const numTicks = 20;
    for (int i = 0; i <= numTicks; i++) {
      final tickProgress = i / numTicks;
      final angle = _startAngle + tickProgress * _sweepAngle;
      final isMajor = i % 5 == 0;
      final tickLength = isMajor ? 7.0 : 4.0;
      tickPaint.strokeWidth = isMajor ? 1.8 : 1.0;
      tickPaint.color = isMajor ? const Color(0xFF71717A) : const Color(0xFFD4D4D8);

      final outer = Offset(
        center.dx + (radius + strokeWidth * 0.5 + 4) * math.cos(angle),
        center.dy + (radius + strokeWidth * 0.5 + 4) * math.sin(angle),
      );
      final inner = Offset(
        center.dx + (radius + strokeWidth * 0.5 + 4 + tickLength) * math.cos(angle),
        center.dy + (radius + strokeWidth * 0.5 + 4 + tickLength) * math.sin(angle),
      );
      canvas.drawLine(outer, inner, tickPaint);
    }

    // 3. Proportional Arcs
    final verifiedSweep = _sweepAngle * verifiedRatio * progress;
    final threatSweep = _sweepAngle * threatRatio * progress;

    // Human Verified Arc (Emerald Green)
    if (verifiedRatio > 0) {
      final verifiedPaint = Paint()
        ..color = const Color(0xFF10B981)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = threatRatio > 0 ? StrokeCap.butt : StrokeCap.round;

      canvas.drawArc(arcRect, _startAngle, verifiedSweep, false, verifiedPaint);
    }

    // Threats Blocked Arc (Alert Red)
    if (threatRatio > 0) {
      final threatStart = _startAngle + verifiedSweep;
      final threatPaint = Paint()
        ..color = const Color(0xFFEF4444)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      canvas.drawArc(arcRect, threatStart, threatSweep, false, threatPaint);
    }

    // 4. Center Digital Readout: Total Scans
    final textSpan = TextSpan(
      children: [
        TextSpan(
          text: '$totalScans\n',
          style: const TextStyle(
            fontSize: 32,
            fontWeight: FontWeight.w900,
            letterSpacing: -1.2,
            height: 1.0,
            color: Color(0xFF09090B),
            fontFeatures: [FontFeature.tabularFigures()],
          ),
        ),
        const TextSpan(
          text: 'TOTAL SCANS',
          style: TextStyle(
            fontSize: 10,
            fontWeight: FontWeight.w800,
            letterSpacing: 1.2,
            height: 1.6,
            color: Color(0xFF71717A),
          ),
        ),
      ],
    );

    final textPainter = TextPainter(
      text: textSpan,
      textAlign: TextAlign.center,
      textDirection: TextDirection.ltr,
    );
    textPainter.layout();

    final textOffset = Offset(
      center.dx - (textPainter.width / 2),
      center.dy - radius * 0.65 - (textPainter.height / 2),
    );
    textPainter.paint(canvas, textOffset);

    // 5. Speedometer Needle
    final needleAngle = _startAngle + (_sweepAngle * needleTarget * progress);
    final needleLength = radius * 0.78;
    final needleTip = Offset(
      center.dx + needleLength * math.cos(needleAngle),
      center.dy + needleLength * math.sin(needleAngle),
    );

    // Needle shadow
    final needleShadowPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.12)
      ..strokeWidth = 2.5
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(
      Offset(center.dx, center.dy + 2),
      Offset(needleTip.dx, needleTip.dy + 2),
      needleShadowPaint,
    );

    // Needle blade
    final needlePaint = Paint()
      ..color = const Color(0xFF18181B)
      ..strokeWidth = 2.4
      ..strokeCap = StrokeCap.round;
    canvas.drawLine(center, needleTip, needlePaint);

    // Needle tip highlight point
    final needleTipDotPaint = Paint()
      ..color = needleTarget > 0.60 ? const Color(0xFFEF4444) : const Color(0xFF10B981)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(needleTip, 3.2, needleTipDotPaint);

    // 6. Center Hub
    final hubOuterPaint = Paint()
      ..color = Colors.white
      ..style = PaintingStyle.fill
      ..maskFilter = const MaskFilter.blur(BlurStyle.solid, 2);
    canvas.drawCircle(center, 12, hubOuterPaint);

    final hubRingPaint = Paint()
      ..color = const Color(0xFFD4D4D8)
      ..style = PaintingStyle.stroke
      ..strokeWidth = 1.5;
    canvas.drawCircle(center, 12, hubRingPaint);

    final hubCorePaint = Paint()
      ..color = const Color(0xFF18181B)
      ..style = PaintingStyle.fill;
    canvas.drawCircle(center, 6, hubCorePaint);
  }

  @override
  bool shouldRepaint(covariant _SpeedometerPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.verifiedRatio != verifiedRatio ||
        oldDelegate.threatRatio != threatRatio ||
        oldDelegate.needleTarget != needleTarget ||
        oldDelegate.totalScans != totalScans;
  }
}
