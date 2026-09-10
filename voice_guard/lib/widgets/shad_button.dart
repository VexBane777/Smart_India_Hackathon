import 'package:flutter/material.dart';
import '../design/tokens.dart';

enum ShadButtonVariant { primary, secondary, outline, ghost, destructive, verified }
enum ShadButtonSize { sm, md, lg, icon }

class ShadButton extends StatefulWidget {
  final VoidCallback? onTap;
  final Widget? child;
  final String? text;
  final Widget? icon;
  final Widget? trailingIcon;
  final ShadButtonVariant variant;
  final ShadButtonSize size;
  final bool loading;
  final bool fullWidth;

  const ShadButton({
    super.key,
    required this.onTap,
    this.child,
    this.text,
    this.icon,
    this.trailingIcon,
    this.variant = ShadButtonVariant.primary,
    this.size = ShadButtonSize.md,
    this.loading = false,
    this.fullWidth = false,
  }) : assert(child != null || text != null || icon != null, 'Provide child, text, or icon');

  @override
  State<ShadButton> createState() => _ShadButtonState();
}

class _ShadButtonState extends State<ShadButton> {
  bool _pressed = false;

  Color get _bg {
    if (widget.onTap == null) {
      return ShadTokens.mutedBg;
    }
    switch (widget.variant) {
      case ShadButtonVariant.primary:
        return ShadTokens.primary;
      case ShadButtonVariant.secondary:
        return ShadTokens.secondary;
      case ShadButtonVariant.outline:
      case ShadButtonVariant.ghost:
        return Colors.transparent;
      case ShadButtonVariant.destructive:
        return ShadTokens.destructive;
      case ShadButtonVariant.verified:
        return ShadTokens.verified;
    }
  }

  Color get _fg {
    if (widget.onTap == null) {
      return ShadTokens.mutedFg;
    }
    switch (widget.variant) {
      case ShadButtonVariant.primary:
        return ShadTokens.primaryFg;
      case ShadButtonVariant.secondary:
        return ShadTokens.secondaryFg;
      case ShadButtonVariant.outline:
      case ShadButtonVariant.ghost:
        return ShadTokens.foreground;
      case ShadButtonVariant.destructive:
        return ShadTokens.destructiveFg;
      case ShadButtonVariant.verified:
        return ShadTokens.verifiedFg;
    }
  }

  Border? get _border {
    if (widget.variant == ShadButtonVariant.outline) {
      return Border.all(
        color: widget.onTap == null ? ShadTokens.mutedFg : ShadTokens.border,
        width: 1.0,
      );
    }
    return null;
  }

  EdgeInsetsGeometry get _padding {
    if (widget.size == ShadButtonSize.icon) {
      return const EdgeInsets.all(8);
    }
    switch (widget.size) {
      case ShadButtonSize.sm:
        return const EdgeInsets.symmetric(horizontal: 10, vertical: 6);
      case ShadButtonSize.md:
        return const EdgeInsets.symmetric(horizontal: 14, vertical: 10);
      case ShadButtonSize.lg:
        return const EdgeInsets.symmetric(horizontal: 20, vertical: 14);
      case ShadButtonSize.icon:
        return const EdgeInsets.all(8);
    }
  }

  double get _fontSize {
    switch (widget.size) {
      case ShadButtonSize.sm:
        return 12.0;
      case ShadButtonSize.md:
        return 13.0;
      case ShadButtonSize.lg:
        return 15.0;
      case ShadButtonSize.icon:
        return 13.0;
    }
  }

  @override
  Widget build(BuildContext context) {
    final fg = _fg;
    Widget content;

    if (widget.loading) {
      content = SizedBox(
        width: _fontSize + 2,
        height: _fontSize + 2,
        child: CircularProgressIndicator(
          strokeWidth: 2,
          color: fg,
        ),
      );
    } else {
      final items = <Widget>[];
      if (widget.icon != null) {
        items.add(IconTheme.merge(
          data: IconThemeData(color: fg, size: _fontSize + 4),
          child: widget.icon!,
        ));
      }
      if (widget.text != null) {
        if (items.isNotEmpty) items.add(const SizedBox(width: 6));
        items.add(Text(
          widget.text!,
          style: TextStyle(
            fontSize: _fontSize,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.1,
            color: fg,
          ),
        ));
      } else if (widget.child != null) {
        if (items.isNotEmpty) items.add(const SizedBox(width: 6));
        items.add(DefaultTextStyle.merge(
          style: TextStyle(
            fontSize: _fontSize,
            fontWeight: FontWeight.w600,
            letterSpacing: -0.1,
            color: fg,
          ),
          child: widget.child!,
        ));
      }
      if (widget.trailingIcon != null) {
        items.add(const SizedBox(width: 6));
        items.add(IconTheme.merge(
          data: IconThemeData(color: fg, size: _fontSize + 4),
          child: widget.trailingIcon!,
        ));
      }

      if (items.length == 1) {
        content = items.first;
      } else {
        content = Row(
          mainAxisSize: MainAxisSize.min,
          mainAxisAlignment: MainAxisAlignment.center,
          children: items,
        );
      }
    }

    Widget container = AnimatedContainer(
      duration: ShadTokens.fast,
      curve: Curves.easeOut,
      transform: Matrix4.diagonal3Values(_pressed ? 0.97 : 1.0, _pressed ? 0.97 : 1.0, 1.0),
      padding: _padding,
      decoration: BoxDecoration(
        color: _bg,
        borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
        border: _border,
        boxShadow: (widget.variant == ShadButtonVariant.primary && widget.onTap != null && !_pressed)
            ? ShadTokens.shadowSm
            : null,
      ),
      child: Center(
        widthFactor: widget.fullWidth ? null : 1.0,
        child: content,
      ),
    );

    return Listener(
      onPointerDown: widget.onTap != null ? (_) => setState(() => _pressed = true) : null,
      onPointerUp: widget.onTap != null
          ? (_) {
              setState(() => _pressed = false);
              widget.onTap?.call();
            }
          : null,
      onPointerCancel: widget.onTap != null ? (_) => setState(() => _pressed = false) : null,
      child: MouseRegion(
        cursor: widget.onTap != null ? SystemMouseCursors.click : SystemMouseCursors.forbidden,
        child: container,
      ),
    );
  }
}