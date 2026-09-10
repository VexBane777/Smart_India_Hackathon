import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../design/tokens.dart';
import 'shad_glass_card.dart';

/// A world-class Apple/shadcn minimalist half-donut chart displaying total analyzed calls
/// with an elegant arc, centered digital readout, segmented distribution bar, and monochromatic legends.
class ShadHalfDonutChart extends StatefulWidget {
  final int totalCount;
  final int humanCount;
  final int aiCount;
  final VoidCallback? onTap;

  const ShadHalfDonutChart({
    super.key,
    this.totalCount = 148,
    this.humanCount = 138,
    this.aiCount = 10,
    this.onTap,
  });

  @override
  State<ShadHalfDonutChart> createState() => _ShadHalfDonutChartState();
}

class _ShadHalfDonutChartState extends State<ShadHalfDonutChart>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _animation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 900),
    );
    _animation = CurvedAnimation(
      parent: _controller,
      curve: Curves.easeOutCubic,
    );
    _controller.forward();
  }

  @override
  void didUpdateWidget(covariant ShadHalfDonutChart oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (oldWidget.totalCount != widget.totalCount ||
        oldWidget.humanCount != widget.humanCount ||
        oldWidget.aiCount != widget.aiCount) {
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
    final total = widget.totalCount > 0 ? widget.totalCount : (widget.humanCount + widget.aiCount);
    final safeTotal = total > 0 ? total : 1;
    final humanRatio = (widget.humanCount / safeTotal).clamp(0.0, 1.0);
    final aiRatio = (widget.aiCount / safeTotal).clamp(0.0, 1.0);

    return ShadGlassCard(
      padding: const EdgeInsets.fromLTRB(20, 20, 20, 18),
      onTap: widget.onTap,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          // ── Half-Donut Arc with Centered Overlay Readout ──
          SizedBox(
            width: double.infinity,
            height: 155,
            child: Stack(
              alignment: Alignment.center,
              children: [
                // Animated Custom Paint Arc
                Positioned.fill(
                  child: AnimatedBuilder(
                    animation: _animation,
                    builder: (context, child) {
                      return CustomPaint(
                        painter: _AppleHalfDonutPainter(
                          progress: _animation.value,
                          humanRatio: humanRatio,
                          aiRatio: aiRatio,
                        ),
                      );
                    },
                  ),
                ),

                // Centered Metric Display
                Positioned(
                  bottom: 10,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      // Big bold count with Inter font
                      Text(
                        '$total',
                        style: GoogleFonts.inter(
                          fontSize: 38,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -1.5,
                          height: 1.0,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      const SizedBox(height: 3),
                      Text(
                        'CALLS ANALYZED',
                        style: GoogleFonts.inter(
                          fontSize: 10,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 1.4,
                          color: ShadTokens.muted,
                        ),
                      ),
                      const SizedBox(height: 6),
                      // Pill status badge
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0xFFF4F4F5),
                          borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
                          border: Border.all(color: ShadTokens.border),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Container(
                              width: 6,
                              height: 6,
                              decoration: const BoxDecoration(
                                color: Color(0xFF10B981),
                                shape: BoxShape.circle,
                              ),
                            ),
                            const SizedBox(width: 5),
                            Text(
                              '${(humanRatio * 100).toInt()}% Authentic',
                              style: GoogleFonts.inter(
                                fontSize: 10,
                                fontWeight: FontWeight.w600,
                                color: ShadTokens.foreground,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            ),
          ),

          const SizedBox(height: 18),

          // ── Segmented Distribution Bar (Apple Storage / Battery Style) ──
          ClipRRect(
            borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
            child: Container(
              height: 6,
              width: double.infinity,
              color: const Color(0xFFF4F4F5),
              child: Row(
                children: [
                  Expanded(
                    flex: (humanRatio * 1000).toInt().clamp(1, 1000),
                    child: Container(color: const Color(0xFF10B981)),
                  ),
                  if (aiRatio > 0) const SizedBox(width: 2),
                  if (aiRatio > 0)
                    Expanded(
                      flex: (aiRatio * 1000).toInt().clamp(1, 1000),
                      child: Container(color: const Color(0xFFEF4444)),
                    ),
                ],
              ),
            ),
          ),

          const SizedBox(height: 14),

          // ── Monochromatic Legend Metric Cards ──
          Row(
            children: [
              // Human Proportion Card
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: const Color(0xFFF4F4F5),
                          borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                          border: Border.all(color: ShadTokens.border),
                        ),
                        child: const Icon(
                          LucideIcons.userCheck,
                          size: 16,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'Human',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w500,
                                color: ShadTokens.muted,
                              ),
                            ),
                            const SizedBox(height: 1),
                            Text(
                              '${widget.humanCount} (${(humanRatio * 100).toInt()}%)',
                              style: GoogleFonts.inter(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: ShadTokens.foreground,
                              ),
                            ),
                          ],
                        ),
                      ),
                      Container(
                        width: 6,
                        height: 6,
                        decoration: const BoxDecoration(
                          color: Color(0xFF10B981),
                          shape: BoxShape.circle,
                        ),
                      ),
                    ],
                  ),
                ),
              ),
              const SizedBox(width: 10),
              // AI Voice Proportion Card
              Expanded(
                child: Container(
                  padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                  decoration: BoxDecoration(
                    color: Colors.white,
                    borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: Row(
                    children: [
                      Container(
                        width: 32,
                        height: 32,
                        decoration: BoxDecoration(
                          color: const Color(0xFFF4F4F5),
                          borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                          border: Border.all(color: ShadTokens.border),
                        ),
                        child: const Icon(
                          LucideIcons.bot,
                          size: 16,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      const SizedBox(width: 10),
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              'AI Voice',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w500,
                                color: ShadTokens.muted,
                              ),
                            ),
                            const SizedBox(height: 1),
                            Text(
                              '${widget.aiCount} (${(aiRatio * 100).toInt()}%)',
                              style: GoogleFonts.inter(
                                fontSize: 13,
                                fontWeight: FontWeight.w700,
                                color: const Color(0xFFEF4444),
                              ),
                            ),
                          ],
                        ),
                      ),
                      Container(
                        width: 6,
                        height: 6,
                        decoration: const BoxDecoration(
                          color: Color(0xFFEF4444),
                          shape: BoxShape.circle,
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

class _AppleHalfDonutPainter extends CustomPainter {
  final double progress;
  final double humanRatio;
  final double aiRatio;

  _AppleHalfDonutPainter({
    required this.progress,
    required this.humanRatio,
    required this.aiRatio,
  });

  // 180° semi-circle arc spanning from left (pi) to right (2*pi / 0)
  static const double _startAngle = math.pi;
  static const double _sweepAngle = math.pi;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.98);
    final radius = math.min(size.width * 0.40, 110.0);
    const strokeWidth = 16.0;

    final arcRect = Rect.fromCircle(center: center, radius: radius);

    // 1. Subtle Background track
    final trackPaint = Paint()
      ..color = const Color(0xFFE4E4E7)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(arcRect, _startAngle, _sweepAngle, false, trackPaint);

    // 2. Human Segment (Emerald Green)
    final humanSweep = _sweepAngle * humanRatio * progress;
    if (humanRatio > 0 && humanSweep > 0) {
      final humanPaint = Paint()
        ..color = const Color(0xFF10B981)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = aiRatio > 0 ? StrokeCap.round : StrokeCap.round;

      canvas.drawArc(arcRect, _startAngle, humanSweep, false, humanPaint);
    }

    // 3. AI Voice Segment (Red)
    final aiSweep = _sweepAngle * aiRatio * progress;
    if (aiRatio > 0 && aiSweep > 0) {
      const gap = 0.04; // small clean gap in radians
      final aiStart = _startAngle + humanSweep + gap;
      final adjustedSweep = math.max(0.0, aiSweep - gap);

      final aiPaint = Paint()
        ..color = const Color(0xFFEF4444)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      if (adjustedSweep > 0) {
        canvas.drawArc(arcRect, aiStart, adjustedSweep, false, aiPaint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant _AppleHalfDonutPainter oldDelegate) {
    return oldDelegate.progress != progress ||
        oldDelegate.humanRatio != humanRatio ||
        oldDelegate.aiRatio != aiRatio;
  }
}
