import 'package:flutter/material.dart';
import '../utils/constants.dart';

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
    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(14),
        border: Border.all(color: granted ? AppColors.verified.withValues(alpha: 0.3) : Colors.black12),
        boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 10, offset: const Offset(0, 2))],
      ),
      child: Row(children: [
        Container(
          width: 44, height: 44,
          decoration: BoxDecoration(color: granted ? AppColors.verifiedBg : const Color(0xFFF5F5F5), borderRadius: BorderRadius.circular(10)),
          child: Icon(icon, color: granted ? AppColors.verified : Colors.black54),
        ),
        const SizedBox(width: 12),
        Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
          Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700)),
          const SizedBox(height: 2),
          Text(subtitle, style: const TextStyle(fontSize: 11, color: Colors.black54)),
          const SizedBox(height: 4),
          Row(children: [
            Icon(granted ? Icons.check_circle : Icons.error_outline, size: 14, color: granted ? AppColors.verified : AppColors.suspicious),
            const SizedBox(width: 4),
            Text(granted ? 'Granted' : 'Required', style: TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: granted ? AppColors.verified : AppColors.suspicious)),
          ]),
        ])),
        const SizedBox(width: 8),
        FilledButton(
          onPressed: onAction,
          style: FilledButton.styleFrom(backgroundColor: granted ? AppColors.verified : AppColors.primary, padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8)),
          child: Text(actionLabel, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
        ),
      ]),
    );
  }
}
