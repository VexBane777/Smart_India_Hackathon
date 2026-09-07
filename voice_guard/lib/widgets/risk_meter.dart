import 'package:flutter/material.dart';
import 'package:percent_indicator/circular_percent_indicator.dart';
import '../utils/constants.dart';

class RiskMeter extends StatelessWidget {
  final double score; // 0..1
  final bool animate;
  const RiskMeter({super.key, required this.score, this.animate = true});

  @override
  Widget build(BuildContext context) {
    final pct = score.clamp(0.0, 1.0);
    final color = AppConstants.colorFor(pct);
    final verdict = AppConstants.verdictFor(pct);
    final bg = AppConstants.bgFor(pct);

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Column(
        children: [
          CircularPercentIndicator(
            radius: 78,
            lineWidth: 12,
            percent: pct,
            animation: animate,
            animationDuration: 600,
            circularStrokeCap: CircularStrokeCap.round,
            backgroundColor: Colors.black.withValues(alpha: 0.06),
            progressColor: color,
            center: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                Text('${(pct * 100).round()}%',
                    style: TextStyle(fontSize: 34, fontWeight: FontWeight.w800, color: color)),
                const SizedBox(height: 2),
                Text('RISK', style: TextStyle(fontSize: 11, letterSpacing: 1.6, color: color.withValues(alpha: 0.8), fontWeight: FontWeight.w700)),
              ],
            ),
          ),
          const SizedBox(height: 14),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
            decoration: BoxDecoration(color: color, borderRadius: BorderRadius.circular(999)),
            child: Text(verdict,
                style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w800, letterSpacing: 0.6)),
          ),
          const SizedBox(height: 8),
          _legend(),
        ],
      ),
    );
  }

  Widget _legend() {
    Widget dot(Color c, String l) => Row(mainAxisSize: MainAxisSize.min, children: [
          Container(width: 8, height: 8, decoration: BoxDecoration(color: c, shape: BoxShape.circle)),
          const SizedBox(width: 5),
          Text(l, style: const TextStyle(fontSize: 10, color: Colors.black54, fontWeight: FontWeight.w600)),
        ]);
    return Row(mainAxisAlignment: MainAxisAlignment.center, children: [
      dot(AppColors.verified, '0-30%'),
      const SizedBox(width: 10),
      dot(AppColors.suspicious, '31-70%'),
      const SizedBox(width: 10),
      dot(AppColors.detected, '71-100%'),
    ]);
  }
}
