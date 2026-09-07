import 'package:flutter/material.dart';
import '../utils/constants.dart';

class StatusIndicator extends StatelessWidget {
  final bool active;
  const StatusIndicator({super.key, required this.active});

  @override
  Widget build(BuildContext context) {
    final color = active ? AppColors.verified : Colors.grey;
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
      decoration: BoxDecoration(
        color: active ? AppColors.verifiedBg : const Color(0xFFEEEEEE),
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: color.withValues(alpha: 0.35)),
      ),
      child: Row(mainAxisSize: MainAxisSize.min, children: [
        Container(
          width: 10, height: 10,
          decoration: BoxDecoration(
            color: color,
            shape: BoxShape.circle,
            boxShadow: active ? [BoxShadow(color: color.withValues(alpha: 0.5), blurRadius: 6)] : null,
          ),
        ),
        const SizedBox(width: 8),
        Text(active ? 'Protection Active' : 'Protection Inactive',
            style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: active ? AppColors.verified : Colors.black54)),
      ]),
    );
  }
}

class ShieldIcon extends StatelessWidget {
  final bool active;
  final double size;
  const ShieldIcon({super.key, required this.active, this.size = 64});

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size, height: size,
      decoration: BoxDecoration(
        color: active ? AppColors.verifiedBg : const Color(0xFFEEEEEE),
        shape: BoxShape.circle,
        border: Border.all(color: (active ? AppColors.verified : Colors.grey).withValues(alpha: 0.35), width: 2),
      ),
      child: Icon(Icons.shield_rounded, size: size * 0.52, color: active ? AppColors.verified : Colors.grey),
    );
  }
}
