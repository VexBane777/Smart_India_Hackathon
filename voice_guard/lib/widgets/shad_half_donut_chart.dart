import 'dart:math' as math;
import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

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

    return GestureDetector(
      onTap: widget.onTap,
      child: Container(
        color: Colors.transparent,
        padding: const EdgeInsets.symmetric(vertical: 8),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            // ── Half-Donut Arc with Centered Digital Readout ──
            SizedBox(
              width: double.infinity,
              height: 165,
              child: Stack(
                alignment: Alignment.center,
                children: [
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
                    bottom: 12,
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          '$total',
                          style: GoogleFonts.inter(
                            fontSize: 48,
                            fontWeight: FontWeight.w800,
                            letterSpacing: -1.8,
                            height: 1.0,
                            color: Colors.white,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          'TOTAL CELLS SCANNED',
                          style: GoogleFonts.inter(
                            fontSize: 10,
                            fontWeight: FontWeight.w700,
                            letterSpacing: 1.2,
                            color: const Color(0xFF8E9192),
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            const SizedBox(height: 16),

            // ── Legend Below Arc: ● 138 AUTHENTIC | ● 10 AI CLONES ──
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  // Left: Authentic
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: Color(0xFF21D4B2),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${widget.humanCount}',
                        style: GoogleFonts.inter(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: const Color(0xFF21D4B2),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'AUTHENTIC',
                        style: GoogleFonts.inter(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                          color: const Color(0xFF8E9192),
                        ),
                      ),
                    ],
                  ),

                  // Right: AI Clones
                  Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Container(
                        width: 8,
                        height: 8,
                        decoration: const BoxDecoration(
                          color: Color(0xFFFF6268),
                          shape: BoxShape.circle,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        '${widget.aiCount}',
                        style: GoogleFonts.inter(
                          fontSize: 22,
                          fontWeight: FontWeight.w800,
                          color: const Color(0xFFFF6268),
                        ),
                      ),
                      const SizedBox(width: 8),
                      Text(
                        'AI CLONES',
                        style: GoogleFonts.inter(
                          fontSize: 11,
                          fontWeight: FontWeight.w700,
                          letterSpacing: 0.8,
                          color: const Color(0xFF8E9192),
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ],
        ),
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

  static const double _startAngle = math.pi;
  static const double _sweepAngle = math.pi;

  @override
  void paint(Canvas canvas, Size size) {
    final center = Offset(size.width / 2, size.height * 0.96);
    final radius = math.min(size.width * 0.38, 115.0);
    const strokeWidth = 20.0;

    final arcRect = Rect.fromCircle(center: center, radius: radius);

    // 1. Subtle Background track
    final trackPaint = Paint()
      ..color = const Color(0xFF1C1C1E)
      ..style = PaintingStyle.stroke
      ..strokeWidth = strokeWidth
      ..strokeCap = StrokeCap.round;

    canvas.drawArc(arcRect, _startAngle, _sweepAngle, false, trackPaint);

    // 2. Human Segment (Teal - #21D4B2)
    final humanSweep = _sweepAngle * humanRatio * progress;
    if (humanRatio > 0 && humanSweep > 0) {
      final humanPaint = Paint()
        ..color = const Color(0xFF21D4B2)
        ..style = PaintingStyle.stroke
        ..strokeWidth = strokeWidth
        ..strokeCap = StrokeCap.round;

      canvas.drawArc(arcRect, _startAngle, humanSweep, false, humanPaint);
    }

    // 3. AI Voice Segment (Coral - #FF6268)
    final aiSweep = _sweepAngle * aiRatio * progress;
    if (aiRatio > 0 && aiSweep > 0) {
      const gap = 0.05;
      final aiStart = _startAngle + humanSweep + gap;
      final adjustedSweep = math.max(0.0, aiSweep - gap);

      final aiPaint = Paint()
        ..color = const Color(0xFFFF6268)
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
