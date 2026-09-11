import 'package:flutter/material.dart';
import '../design/tokens.dart';

class StatusIndicator extends StatelessWidget {
  final bool active;
  const StatusIndicator({super.key, required this.active});

  @override
  Widget build(BuildContext context) {
    final color = active ? ShadTokens.verified : ShadTokens.muted;
    final bg = active ? ShadTokens.verifiedBg : ShadTokens.secondary;
    final borderColor = active ? ShadTokens.verifiedBorder : ShadTokens.border;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 5),
      decoration: BoxDecoration(
        color: bg,
        borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
        border: Border.all(color: borderColor),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Container(
            width: 7,
            height: 7,
            decoration: BoxDecoration(
              color: color,
              shape: BoxShape.circle,
              boxShadow: active
                  ? [BoxShadow(color: color.withValues(alpha: 0.5), blurRadius: 4)]
                  : null,
            ),
          ),
          const SizedBox(width: 6),
          Text(
            active ? 'System Protected' : 'Protection Paused',
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.2,
              color: active ? ShadTokens.verified : ShadTokens.muted,
            ),
          ),
        ],
      ),
    );
  }
}

class ShieldIcon extends StatelessWidget {
  final bool active;
  final double size;
  const ShieldIcon({super.key, required this.active, this.size = 56});

  @override
  Widget build(BuildContext context) {
    final color = active ? ShadTokens.verified : ShadTokens.muted;
    final bg = active ? ShadTokens.verifiedBg : ShadTokens.secondary;
    final borderColor = active ? ShadTokens.verifiedBorder : ShadTokens.border;

    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        color: bg,
        shape: BoxShape.circle,
        border: Border.all(color: borderColor, width: 1.5),
        boxShadow: ShadTokens.shadowSm,
      ),
      child: Icon(
        Icons.shield_rounded,
        size: size * 0.52,
        color: color,
      ),
    );
  }
}
