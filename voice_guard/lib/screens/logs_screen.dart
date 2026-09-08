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
    final callLogs = provider.callLogs;
    final historyLogs = provider.history.reversed.toList();
    final hasLogs = callLogs.isNotEmpty || historyLogs.isNotEmpty;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: const Text('Call Logs & Recordings', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800)),
        actions: [
          if (hasLogs)
            TextButton(
              onPressed: () => provider.clearHistory(),
              child: const Text('Clear', style: TextStyle(color: AppColors.detected, fontWeight: FontWeight.w700)),
            ),
        ],
      ),
      body: !hasLogs
          ? Center(
              child: Column(mainAxisSize: MainAxisSize.min, children: [
                Icon(Icons.mic_none_outlined, size: 52, color: Colors.black.withValues(alpha: 0.2)),
                const SizedBox(height: 10),
                Text('No recorded calls yet',
                    style: TextStyle(fontSize: 14, fontWeight: FontWeight.w700, color: Colors.black.withValues(alpha: 0.6))),
                const SizedBox(height: 4),
                Text('Real calls and live acoustic scans are recorded to disk\nas standard WAV files and evaluated via TFLite.',
                    textAlign: TextAlign.center,
                    style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.5), height: 1.4)),
                if (Navigator.canPop(context)) ...[
                  const SizedBox(height: 14),
                  OutlinedButton(onPressed: () => Navigator.pop(context), child: const Text('Back to Home')),
                ],
              ]))
          : ListView(
              padding: const EdgeInsets.all(12),
              children: [
                if (callLogs.isNotEmpty) ...[
                  const Padding(
                    padding: EdgeInsets.only(left: 4, bottom: 8),
                    child: Text('Recorded Phone Calls & Scans', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.textSecondary)),
                  ),
                  for (final cl in callLogs) ...[
                    Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: Colors.white,
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: Colors.black12),
                      ),
                      child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                        Row(children: [
                          Container(
                            width: 44,
                            height: 44,
                            decoration: BoxDecoration(
                              color: AppConstants.colorFor(cl.riskScore).withValues(alpha: 0.12),
                              borderRadius: BorderRadius.circular(10),
                              border: Border.all(color: AppConstants.colorFor(cl.riskScore).withValues(alpha: 0.3)),
                            ),
                            child: Center(
                              child: Text(
                                '${(cl.riskScore * 100).toInt()}%',
                                style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: AppConstants.colorFor(cl.riskScore)),
                              ),
                            ),
                          ),
                          const SizedBox(width: 12),
                          Expanded(
                            child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                              Text(cl.number, style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w800)),
                              const SizedBox(height: 2),
                              Row(children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                                  decoration: BoxDecoration(
                                    color: AppConstants.colorFor(cl.riskScore),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    AppConstants.verdictFor(cl.riskScore),
                                    style: const TextStyle(color: Colors.white, fontSize: 9, fontWeight: FontWeight.w800),
                                  ),
                                ),
                                const SizedBox(width: 6),
                                Text(
                                  DateFormat('dd MMM, hh:mm a').format(cl.timestamp),
                                  style: const TextStyle(fontSize: 10, color: Colors.black54),
                                ),
                              ]),
                            ]),
                          ),
                          Icon(
                            cl.riskScore >= 0.71 ? Icons.warning_amber_rounded : Icons.verified_user_rounded,
                            color: AppConstants.colorFor(cl.riskScore),
                            size: 22,
                          ),
                        ]),
                        if (cl.recordingPath != null) ...[
                          const SizedBox(height: 8),
                          Container(
                            padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                            decoration: BoxDecoration(
                              color: const Color(0xFFF5F5F5),
                              borderRadius: BorderRadius.circular(8),
                            ),
                            child: Row(children: [
                              const Icon(Icons.audio_file, size: 16, color: AppColors.primary),
                              const SizedBox(width: 6),
                              Expanded(
                                child: Text(
                                  cl.recordingPath!.split(RegExp(r'[\\/]')).last,
                                  style: const TextStyle(fontSize: 10, fontFamily: 'monospace', color: Colors.black87),
                                  overflow: TextOverflow.ellipsis,
                                ),
                              ),
                              const Icon(Icons.check_circle, size: 14, color: Color(0xFF2E7D32)),
                            ]),
                          ),
                        ],
                      ]),
                    ),
                  ],
                  const SizedBox(height: 12),
                ],
                if (historyLogs.isNotEmpty) ...[
                  const Padding(
                    padding: EdgeInsets.only(left: 4, bottom: 8),
                    child: Text('Live Scoring Timeline', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700, color: AppColors.textSecondary)),
                  ),
                  for (final r in historyLogs)
                    Container(
                      margin: const EdgeInsets.only(bottom: 8),
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.black12)),
                      child: Row(children: [
                        Container(
                          width: 44,
                          height: 44,
                          decoration: BoxDecoration(color: r.bg, borderRadius: BorderRadius.circular(10), border: Border.all(color: r.color.withValues(alpha: 0.3))),
                          child: Center(child: Text('${r.percent}%', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: r.color))),
                        ),
                        const SizedBox(width: 12),
                        Expanded(
                          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                            Row(children: [
                              Container(
                                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                                decoration: BoxDecoration(color: r.color, borderRadius: BorderRadius.circular(999)),
                                child: Text(r.label, style: const TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.w800)),
                              ),
                              const SizedBox(width: 6),
                              Text(DateFormat('dd MMM, hh:mm a').format(r.timestamp), style: const TextStyle(fontSize: 10, color: Colors.black54)),
                            ]),
                            const SizedBox(height: 4),
                            Text('Risk ${(r.score * 100).toStringAsFixed(1)}% • ${r.prosody != null ? "prosody ✓" : "spectral"}',
                                style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.6))),
                          ]),
                        ),
                        Icon(r.verdict == Verdict.detected ? Icons.warning_amber_rounded : r.verdict == Verdict.suspicious ? Icons.error_outline : Icons.verified_user_rounded, color: r.color, size: 20),
                      ]),
                    ),
                ],
              ],
            ),
    );
  }
}
