import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// shadcn/ui-styled Card — surface with subtle border, rounded-lg, and ambient shadow.
class ShadCard extends StatelessWidget {
  final Widget? child;
  final EdgeInsetsGeometry? padding;
  final VoidCallback? onTap;
  final bool selected;
  final Color? backgroundColor;
  final Color? borderColor;

  const ShadCard({
    super.key,
    this.child,
    this.padding,
    this.onTap,
    this.selected = false,
    this.backgroundColor,
    this.borderColor,
  });

  @override
  Widget build(BuildContext context) {
    final effectiveBorderColor = borderColor ??
        (selected ? ShadTokens.primary : ShadTokens.border);

    Widget box = Container(
      padding: padding ?? const EdgeInsets.all(ShadTokens.space4),
      decoration: BoxDecoration(
        color: backgroundColor ?? ShadTokens.card,
        borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
        border: Border.all(
          color: effectiveBorderColor,
          width: selected ? 1.5 : 1.0,
        ),
        boxShadow: ShadTokens.shadowSm,
      ),
      child: child,
    );

    if (onTap != null) {
      return Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onTap,
          borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
          child: box,
        ),
      );
    }
    return box;
  }
}

/// Card header with title, optional description, and optional trailing widget.
class ShadCardHeader extends StatelessWidget {
  final Widget? title;
  final Widget? description;
  final Widget? trailing;
  final EdgeInsetsGeometry? padding;

  const ShadCardHeader({
    super.key,
    this.title,
    this.description,
    this.trailing,
    this.padding,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: padding ?? const EdgeInsets.only(bottom: ShadTokens.space3),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                if (title != null)
                  DefaultTextStyle.merge(
                    style: const TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      color: ShadTokens.cardFg,
                      letterSpacing: -0.2,
                    ),
                    child: title!,
                  ),
                if (description != null) ...[
                  const SizedBox(height: 3),
                  DefaultTextStyle.merge(
                    style: const TextStyle(
                      fontSize: 12,
                      fontWeight: FontWeight.w400,
                      color: ShadTokens.muted,
                      height: 1.35,
                    ),
                    child: description!,
                  ),
                ],
              ],
            ),
          ),
          if (trailing != null) ...[
            const SizedBox(width: ShadTokens.space2),
            trailing!,
          ],
        ],
      ),
    );
  }
}

/// Card footer with subtle top spacing or actions row.
class ShadCardFooter extends StatelessWidget {
  final Widget child;
  final EdgeInsetsGeometry? padding;

  const ShadCardFooter({
    super.key,
    required this.child,
    this.padding,
  });

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: padding ?? const EdgeInsets.only(top: ShadTokens.space3),
      child: child,
    );
  }
}