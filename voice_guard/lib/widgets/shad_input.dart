import 'package:flutter/material.dart';
import '../design/tokens.dart';

class ShadInput extends StatelessWidget {
  final TextEditingController? controller;
  final String? placeholder;
  final String? label;
  final Widget? prefix;
  final Widget? suffix;
  final ValueChanged<String>? onChanged;
  final ValueChanged<String>? onSubmitted;
  final TextInputType? keyboardType;
  final bool autofocus;
  final bool readOnly;
  final int maxLines;

  const ShadInput({
    super.key,
    this.controller,
    this.placeholder,
    this.label,
    this.prefix,
    this.suffix,
    this.onChanged,
    this.onSubmitted,
    this.keyboardType,
    this.autofocus = false,
    this.readOnly = false,
    this.maxLines = 1,
  });

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      mainAxisSize: MainAxisSize.min,
      children: [
        if (label != null) ...[
          Text(
            label!,
            style: const TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: ShadTokens.foreground,
            ),
          ),
          const SizedBox(height: 6),
        ],
        Container(
          decoration: BoxDecoration(
            color: ShadTokens.surface,
            borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
            border: Border.all(color: ShadTokens.border),
          ),
          padding: const EdgeInsets.symmetric(horizontal: 12),
          child: Row(
            children: [
              if (prefix != null) ...[
                prefix!,
                const SizedBox(width: 8),
              ],
              Expanded(
                child: TextField(
                  controller: controller,
                  onChanged: onChanged,
                  onSubmitted: onSubmitted,
                  keyboardType: keyboardType,
                  autofocus: autofocus,
                  readOnly: readOnly,
                  maxLines: maxLines,
                  style: const TextStyle(
                    fontSize: 14,
                    color: ShadTokens.foreground,
                  ),
                  decoration: InputDecoration(
                    hintText: placeholder,
                    hintStyle: const TextStyle(
                      fontSize: 13,
                      color: ShadTokens.muted,
                    ),
                    isDense: true,
                    border: InputBorder.none,
                    contentPadding: const EdgeInsets.symmetric(vertical: 12),
                  ),
                ),
              ),
              if (suffix != null) ...[
                const SizedBox(width: 8),
                suffix!,
              ],
            ],
          ),
        ),
      ],
    );
  }
}
