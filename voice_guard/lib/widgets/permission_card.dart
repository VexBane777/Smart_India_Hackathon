import 'package:flutter/material.dart';
import '../design/tokens.dart';
import 'shad_card.dart';
import 'shad_button.dart';

class PermissionCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final bool granted;
  final VoidCallback onAction;
  final String actionLabel;

  const PermissionCard({
    super.key,
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.granted,
    required this.onAction,
    required this.actionLabel,
  });

  @override
  Widget build(BuildContext context) {
    return ShadCard(
      padding: const EdgeInsets.all(ShadTokens.space3),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
              border: Border.all(color: ShadTokens.border),
            ),
            child: Icon(
              icon,
              color: granted ? ShadTokens.foreground : ShadTokens.muted,
              size: 18,
            ),
          ),
          const SizedBox(width: ShadTokens.space3),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisSize: MainAxisSize.min,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 13,
                    fontWeight: FontWeight.w700,
                    color: ShadTokens.cardFg,
                  ),
                  maxLines: 1,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 3),
                Text(
                  subtitle,
                  style: const TextStyle(
                    fontSize: 11,
                    color: ShadTokens.muted,
                    height: 1.3,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
              ],
            ),
          ),
          const SizedBox(width: ShadTokens.space2),
          ShadButton(
            onTap: onAction,
            variant: granted ? ShadButtonVariant.outline : ShadButtonVariant.primary,
            size: ShadButtonSize.sm,
            text: actionLabel,
          ),
        ],
      ),
    );
  }
}
