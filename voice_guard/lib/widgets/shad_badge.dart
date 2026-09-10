import 'package:flutter/material.dart';
import '../design/tokens.dart';

enum ShadBadgeVariant {
  defaultBadge,
  secondary,
  outline,
  destructive,
  verified,
  suspicious,
  info,
}

class ShadBadge extends StatelessWidget {
  final String label;
  final ShadBadgeVariant variant;
  final Widget? icon;
  final bool showDot;
  final Color? customColor;
  final Color? customBg;

  const ShadBadge({
    super.key,
    required this.label,
    this.variant = ShadBadgeVariant.defaultBadge,
    this.icon,
    this.showDot = false,
    this.customColor,
    this.customBg,
  });

  Color get _bg {
    if (customBg != null) return customBg!;
    switch (variant) {
      case ShadBadgeVariant.defaultBadge:
        return ShadTokens.primary;
      case ShadBadgeVariant.secondary:
        return ShadTokens.secondary;
      case ShadBadgeVariant.outline:
        return Colors.transparent;
      case ShadBadgeVariant.destructive:
        return ShadTokens.destructiveBg;
      case ShadBadgeVariant.verified:
        return ShadTokens.verifiedBg;
      case ShadBadgeVariant.suspicious:
        return ShadTokens.suspiciousBg;
      case ShadBadgeVariant.info:
        return ShadTokens.infoBg;
    }
  }

  Color get _fg {
    if (customColor != null) return customColor!;
    switch (variant) {
      case ShadBadgeVariant.defaultBadge:
        return ShadTokens.primaryFg;
      case ShadBadgeVariant.secondary:
        return ShadTokens.secondaryFg;
      case ShadBadgeVariant.outline:
        return ShadTokens.foreground;
      case ShadBadgeVariant.destructive:
        return ShadTokens.destructive;
      case ShadBadgeVariant.verified:
        return ShadTokens.verified;
      case ShadBadgeVariant.suspicious:
        return ShadTokens.suspicious;
      case ShadBadgeVariant.info:
        return ShadTokens.info;
    }
  }

  Border? get _border {
    switch (variant) {
      case ShadBadgeVariant.outline:
        return Border.all(color: ShadTokens.border, width: 1);
      case ShadBadgeVariant.destructive:
        return Border.all(color: ShadTokens.destructiveBorder, width: 1);
      case ShadBadgeVariant.verified:
        return Border.all(color: ShadTokens.verifiedBorder, width: 1);
      case ShadBadgeVariant.suspicious:
        return Border.all(color: ShadTokens.suspiciousBorder, width: 1);
      case ShadBadgeVariant.info:
        return Border.all(color: ShadTokens.infoBorder, width: 1);
      default:
        return null;
    }
  }

  @override
  Widget build(BuildContext context) {
    final fg = _fg;

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
      decoration: BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
        border: _border,
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          if (showDot) ...[
            Container(
              width: 6,
              height: 6,
              margin: const EdgeInsets.only(right: 5),
              decoration: BoxDecoration(
                color: fg,
                shape: BoxShape.circle,
              ),
            ),
          ],
          if (icon != null) ...[
            IconTheme.merge(
              data: IconThemeData(color: fg, size: 12),
              child: icon!,
            ),
            const SizedBox(width: 4),
          ],
          Text(
            label,
            style: TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.3,
              color: fg,
              height: 1.1,
            ),
          ),
        ],
      ),
    );
  }
}
