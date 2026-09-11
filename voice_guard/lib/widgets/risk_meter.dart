import 'package:flutter/material.dart';
import 'shad_risk_meter.dart';

export 'shad_risk_meter.dart';

/// Legacy adapter for RiskMeter, wrapping ShadRiskMeter.
class RiskMeter extends StatelessWidget {
  final double score;
  final bool animate;

  const RiskMeter({
    super.key,
    required this.score,
    this.animate = true,
  });

  @override
  Widget build(BuildContext context) {
    return ShadRiskMeter(
      score: score,
      animate: animate,
      size: 190,
      showLegend: true,
    );
  }
}
