import 'package:flutter/material.dart';
import '../design/tokens.dart';
import 'shad_button.dart';

class ShadDialog extends StatelessWidget {
  final String? title;
  final String? description;
  final Widget? content;
  final List<Widget>? actions;

  const ShadDialog({
    super.key,
    this.title,
    this.description,
    this.content,
    this.actions,
  });

  static Future<T?> show<T>({
    required BuildContext context,
    String? title,
    String? description,
    Widget? content,
    List<Widget>? actions,
  }) {
    return showDialog<T>(
      context: context,
      barrierColor: Colors.black.withValues(alpha: 0.5),
      builder: (_) => ShadDialog(
        title: title,
        description: description,
        content: content,
        actions: actions,
      ),
    );
  }

  static Future<bool?> confirm({
    required BuildContext context,
    required String title,
    required String description,
    String confirmLabel = 'Confirm',
    String cancelLabel = 'Cancel',
    bool isDestructive = false,
  }) {
    return show<bool>(
      context: context,
      title: title,
      description: description,
      actions: [
        ShadButton(
          onTap: () => Navigator.of(context).pop(false),
          variant: ShadButtonVariant.outline,
          text: cancelLabel,
          size: ShadButtonSize.sm,
        ),
        ShadButton(
          onTap: () => Navigator.of(context).pop(true),
          variant: isDestructive ? ShadButtonVariant.destructive : ShadButtonVariant.primary,
          text: confirmLabel,
          size: ShadButtonSize.sm,
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Dialog(
      backgroundColor: ShadTokens.card,
      elevation: 0,
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(ShadTokens.radiusXl),
        side: const BorderSide(color: ShadTokens.border),
      ),
      insetPadding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
      child: Padding(
        padding: const EdgeInsets.all(ShadTokens.space5),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (title != null) ...[
              Text(
                title!,
                style: const TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w700,
                  letterSpacing: -0.3,
                  color: ShadTokens.cardFg,
                ),
              ),
              const SizedBox(height: 4),
            ],
            if (description != null) ...[
              Text(
                description!,
                style: const TextStyle(
                  fontSize: 13,
                  color: ShadTokens.muted,
                  height: 1.4,
                ),
              ),
              const SizedBox(height: ShadTokens.space3),
            ],
            if (content != null) ...[
              content!,
              const SizedBox(height: ShadTokens.space4),
            ],
            if (actions != null && actions!.isNotEmpty) ...[
              const SizedBox(height: ShadTokens.space2),
              Row(
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  for (int i = 0; i < actions!.length; i++) ...[
                    if (i > 0) const SizedBox(width: ShadTokens.space2),
                    actions![i],
                  ],
                ],
              ),
            ],
          ],
        ),
      ),
    );
  }
}