import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// Clean metric / stat card displaying key telemetry numbers.
class ShadStatCard extends StatelessWidget {
  final Widget icon;
  final String label;
  final dynamic value; // int or String
  final Color color;
  final String? subtitle;

  const ShadStatCard({
    super.key,
    required this.icon,
    required this.label,
    required this.value,
    this.color = ShadTokens.primary,
    this.subtitle,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.all(ShadTokens.space3),
      decoration: BoxDecoration(
        color: ShadTokens.card,
        borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
        border: Border.all(color: ShadTokens.border),
        boxShadow: ShadTokens.shadowSm,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        mainAxisSize: MainAxisSize.min,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Container(
                padding: const EdgeInsets.all(6),
                decoration: BoxDecoration(
                  color: color.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                ),
                child: IconTheme.merge(
                  data: IconThemeData(color: color, size: 16),
                  child: icon,
                ),
              ),
              if (subtitle != null)
                Text(
                  subtitle!,
                  style: TextStyle(
                    fontSize: 10,
                    fontWeight: FontWeight.w600,
                    color: color,
                  ),
                ),
            ],
          ),
          const SizedBox(height: ShadTokens.space2),
          Text(
            '$value',
            style: const TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
              color: ShadTokens.cardFg,
              fontFeatures: [FontFeature.tabularFigures()],
            ),
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w500,
              color: ShadTokens.muted,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ],
      ),
    );
  }
}