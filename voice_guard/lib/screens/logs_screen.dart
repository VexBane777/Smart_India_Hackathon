import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import '../providers/risk_score_provider.dart';
import '../models/risk_score.dart';
import '../utils/constants.dart';

class LogsScreen extends StatelessWidget {
  const LogsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<RiskScoreProvider>();
    final logs = provider.history.reversed.toList();
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.white, elevation: 0,
        title: const Text('Call Logs', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
        actions: [
          if (logs.isNotEmpty)
            TextButton(onPressed: () => provider.clearHistory(), child: const Text('Clear', style: TextStyle(color: AppColors.detected, fontWeight: FontWeight.w700))),
        ],
      ),
      body: logs.isEmpty
          ? Center(child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.history, size: 48, color: Colors.black.withValues(alpha: 0.2)),
                const SizedBox(height: 10),
                Text('No calls yet', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.black.withValues(alpha: 0.6))),
                const SizedBox(height: 4),
                Text('Risk events from live calls will appear here.\nOnly metadata is stored — no raw audio.',
                    textAlign: TextAlign.center, style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.5), height: 1.4)),
                if (Navigator.canPop(context)) ...[
                  const SizedBox(height: 14),
                  OutlinedButton(onPressed: () => Navigator.pop(context), child: const Text('Back to Home')),
                ],
              ]))
          : ListView.separated(
              padding: const EdgeInsets.all(12),
              itemCount: logs.length,
              separatorBuilder: (_, _) => const SizedBox(height: 8),
              itemBuilder: (_, i) {
                final r = logs[i];
                return Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.black12)),
                  child: Row(children: [
                    Container(
                      width: 44, height: 44,
                      decoration: BoxDecoration(color: r.bg, borderRadius: BorderRadius.circular(10), border: Border.all(color: r.color.withValues(alpha: 0.3))),
                      child: Center(child: Text('${r.percent}%', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: r.color))),
                    ),
                    const SizedBox(width: 12),
                    Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Row(children: [
                        Container(padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3), decoration: BoxDecoration(color: r.color, borderRadius: BorderRadius.circular(999)), child: Text(r.label, style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.w800))),
                        const SizedBox(width: 6),
                        Text(DateFormat('dd MMM, hh:mm a').format(r.timestamp), style: const TextStyle(fontSize: 10, color: Colors.black54)),
                      ]),
                      const SizedBox(height: 4),
                      Text('Risk ${(r.score*100).toStringAsFixed(1)}% • ${r.prosody != null ? "prosody ✓" : "spectral"}',
                          style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.6))),
                    ])),
                    Icon(r.verdict == Verdict.detected ? Icons.warning_amber_rounded : r.verdict == Verdict.suspicious ? Icons.error_outline : Icons.verified_user_rounded, color: r.color, size: 20),
                  ]),
                );
              },
            ),
    );
  }
}
