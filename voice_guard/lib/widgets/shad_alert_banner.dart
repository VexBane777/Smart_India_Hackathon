import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// In-call risk banner for Protected Call — Voip.md Phase 5: Emerald
/// ("Verified Human") -> Amber ("Suspicious") -> Crimson ("Synthetic Clone
/// Detected"), driven directly by [RiskScoreProvider.state].
class ShadAlertBanner extends StatelessWidget {
  final String state; // 'normal' | 'warn' | 'alert'
  const ShadAlertBanner({required this.state, super.key});

  static const _copy = {
    'normal': 'Verified Human',
    'warn': 'Suspicious',
    'alert': 'Synthetic Clone Detected',
  };

  Color _bg() => switch (state) {
        'alert' => ShadTokens.detectedBg,
        'warn' => ShadTokens.suspiciousBg,
        _ => ShadTokens.verifiedBg,
      };

  Color _fg() => switch (state) {
        'alert' => ShadTokens.detected,
        'warn' => ShadTokens.suspicious,
        _ => ShadTokens.verified,
      };

  Color _border() => switch (state) {
        'alert' => ShadTokens.detectedBorder,
        'warn' => ShadTokens.suspiciousBorder,
        _ => ShadTokens.verifiedBorder,
      };

  IconData _icon() => switch (state) {
        'alert' => Icons.gpp_bad,
        'warn' => Icons.warning_amber_rounded,
        _ => Icons.verified_user,
      };

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: _bg(),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _border()),
      ),
      child: Row(
        children: [
          Icon(_icon(), color: _fg(), size: 20),
          const SizedBox(width: 10),
          Text(
            _copy[state] ?? _copy['normal']!,
            style: TextStyle(color: _fg(), fontWeight: FontWeight.w800, fontSize: 14),
          ),
        ],
      ),
    );
  }
}
