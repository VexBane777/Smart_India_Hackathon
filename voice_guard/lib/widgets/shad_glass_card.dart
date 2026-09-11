import 'dart:ui';
import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// A premium Frosted Glass (Glassmorphism) container with backdrop blur,
/// subtle translucent gradient fill, crisp border, and ambient shadow.
class ShadGlassCard extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;
  final EdgeInsetsGeometry? margin;
  final double? width;
  final double? height;
  final double borderRadius;
  final VoidCallback? onTap;
  final Color? tintColor;
  final double opacity;
  final double blur;
  final Border? customBorder;

  const ShadGlassCard({
    super.key,
    required this.child,
    this.padding,
    this.margin,
    this.width,
    this.height,
    this.borderRadius = ShadTokens.radiusXl,
    this.onTap,
    this.tintColor,
    this.opacity = 0.72,
    this.blur = 16.0,
    this.customBorder,
  });

  @override
  Widget build(BuildContext context) {
    final baseColor = tintColor ?? Colors.white;

    Widget cardContent = ClipRRect(
      borderRadius: BorderRadius.circular(borderRadius),
      child: BackdropFilter(
        filter: ImageFilter.blur(sigmaX: blur, sigmaY: blur),
        child: Container(
          width: width,
          height: height,
          padding: padding ?? const EdgeInsets.all(ShadTokens.space4),
          decoration: BoxDecoration(
            color: baseColor.withValues(alpha: opacity),
            borderRadius: BorderRadius.circular(borderRadius),
            border: customBorder ??
                Border.all(
                  color: Colors.white.withValues(alpha: 0.75),
                  width: 1.2,
                ),
            boxShadow: [
              BoxShadow(
                color: const Color(0xFF0F172A).withValues(alpha: 0.04),
                blurRadius: 20,
                offset: const Offset(0, 8),
              ),
              BoxShadow(
                color: Colors.white.withValues(alpha: 0.6),
                blurRadius: 1,
                offset: const Offset(0, 1),
              ),
            ],
          ),
          child: child,
        ),
      ),
    );

    if (margin != null) {
      cardContent = Padding(padding: margin!, child: cardContent);
    }

    if (onTap != null) {
      return Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(borderRadius),
          splashColor: ShadTokens.primary.withValues(alpha: 0.05),
          highlightColor: ShadTokens.primary.withValues(alpha: 0.03),
          child: cardContent,
        ),
      );
    }

    return cardContent;
  }
}
