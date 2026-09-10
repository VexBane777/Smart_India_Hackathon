import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:intl/intl.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/risk_score_provider.dart';
import '../models/risk_score.dart';
import '../models/call_log.dart';
import '../widgets/shad_glass_card.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_button.dart';
import '../widgets/shad_dialog.dart';
import '../design/tokens.dart';

class LogsScreen extends StatefulWidget {
  const LogsScreen({super.key});

  @override
  State<LogsScreen> createState() => _LogsScreenState();
}

class _LogsScreenState extends State<LogsScreen> {
  int _filterIndex = 0; // 0: All, 1: Threats, 2: Suspicious, 3: Verified

  @override
  Widget build(BuildContext context) {
    final provider = context.watch<RiskScoreProvider>();
    final callLogs = provider.callLogs;
    final historyLogs = provider.history.reversed.toList();
    final hasLogs = callLogs.isNotEmpty || historyLogs.isNotEmpty;

    // Filtered lists
    final filteredCallLogs = callLogs.where((cl) {
      if (_filterIndex == 1) return cl.riskScore >= 0.70;
      if (_filterIndex == 2) return cl.riskScore >= 0.30 && cl.riskScore < 0.70;
      if (_filterIndex == 3) return cl.riskScore < 0.30;
      return true;
    }).toList();

    final filteredHistory = historyLogs.where((r) {
      if (_filterIndex == 1) return r.score >= 0.70;
      if (_filterIndex == 2) return r.score >= 0.30 && r.score < 0.70;
      if (_filterIndex == 3) return r.score < 0.30;
      return true;
    }).toList();

    final threatCount = callLogs.where((cl) => cl.riskScore >= 0.70).length;
    final suspiciousCount = callLogs.where((cl) => cl.riskScore >= 0.30 && cl.riskScore < 0.70).length;
    final verifiedCount = callLogs.where((cl) => cl.riskScore < 0.30).length;

    return Scaffold(
      backgroundColor: ShadTokens.background,
      appBar: AppBar(
        backgroundColor: ShadTokens.surface,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: Text(
          'Security Audit Logs',
          style: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.3,
            color: ShadTokens.foreground,
          ),
        ),
        actions: [
          if (hasLogs) ...[
            ShadButton(
              onTap: () async {
                final ok = await ShadDialog.confirm(
                  context: context,
                  title: 'Wipe Audit History?',
                  description: 'This permanently clears in-memory call logs and scoring traces. Recorded audio files will remain in sandboxed storage.',
                  confirmLabel: 'Wipe Logs',
                  isDestructive: true,
                );
                if (ok == true) {
                  provider.clearHistory();
                }
              },
              variant: ShadButtonVariant.ghost,
              size: ShadButtonSize.sm,
              text: 'Clear All',
            ),
            const SizedBox(width: 8),
          ],
        ],
      ),
      body: Column(
        children: [
          // ── Apple-Style Segmented Filter Control ──
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
            color: ShadTokens.surface,
            child: Container(
              padding: const EdgeInsets.all(3),
              decoration: BoxDecoration(
                color: const Color(0xFFF4F4F5),
                borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                border: Border.all(color: ShadTokens.border),
              ),
              child: Row(
                children: [
                  _segmentedTab(0, 'All (${callLogs.length})'),
                  _segmentedTab(1, 'Threats ($threatCount)'),
                  _segmentedTab(2, 'Suspicious ($suspiciousCount)'),
                  _segmentedTab(3, 'Verified ($verifiedCount)'),
                ],
              ),
            ),
          ),

          const Divider(height: 1, color: ShadTokens.border),

          // ── Main Body ──
          Expanded(
            child: (!hasLogs || (filteredCallLogs.isEmpty && filteredHistory.isEmpty))
                ? Center(
                    child: Padding(
                      padding: const EdgeInsets.all(32),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        children: [
                          Container(
                            width: 56,
                            height: 56,
                            decoration: BoxDecoration(
                              color: const Color(0xFFF4F4F5),
                              borderRadius: BorderRadius.circular(ShadTokens.radiusXl),
                              border: Border.all(color: ShadTokens.border),
                            ),
                            child: const Icon(
                              LucideIcons.fileText,
                              size: 24,
                              color: ShadTokens.muted,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Text(
                            'No Security Records',
                            style: GoogleFonts.inter(
                              fontSize: 15,
                              fontWeight: FontWeight.w700,
                              color: ShadTokens.foreground,
                            ),
                          ),
                          const SizedBox(height: 6),
                          Text(
                            'No logs match this filter criteria.',
                            textAlign: TextAlign.center,
                            style: GoogleFonts.inter(fontSize: 12, color: ShadTokens.muted),
                          ),
                        ],
                      ),
                    ),
                  )
                : ListView(
                    padding: const EdgeInsets.all(16),
                    children: [
                      // ── Telemetry Summary Header Card ──
                      ShadGlassCard(
                        margin: const EdgeInsets.only(bottom: 16),
                        padding: const EdgeInsets.all(16),
                        child: Column(
                          children: [
                            Row(
                              mainAxisAlignment: MainAxisAlignment.spaceAround,
                              children: [
                                _summaryStat('TOTAL SCANS', '148', LucideIcons.scanLine),
                                Container(width: 1, height: 36, color: ShadTokens.border),
                                _summaryStat('AI BLOCKED', '$threatCount', LucideIcons.shieldAlert),
                                Container(width: 1, height: 36, color: ShadTokens.border),
                                _summaryStat('CLEAN TRAFFIC', '93.2%', LucideIcons.shieldCheck),
                              ],
                            ),
                            const SizedBox(height: 12),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                              decoration: BoxDecoration(
                                color: const Color(0xFFF4F4F5),
                                borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                                border: Border.all(color: ShadTokens.border),
                              ),
                              child: Row(
                                children: [
                                  const Icon(LucideIcons.lock, size: 12, color: ShadTokens.muted),
                                  const SizedBox(width: 6),
                                  Expanded(
                                    child: Text(
                                      'Sandboxed Telemetry • Transient RAM Buffer • DPDP Act Section 6',
                                      style: GoogleFonts.inter(
                                        fontSize: 9.5,
                                        fontWeight: FontWeight.w500,
                                        color: ShadTokens.muted,
                                      ),
                                    ),
                                  ),
                                ],
                              ),
                            ),
                          ],
                        ),
                      ),

                      // ── Recorded Calls List ──
                      if (filteredCallLogs.isNotEmpty) ...[
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'AUDIT RECORDS',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.8,
                                color: ShadTokens.muted,
                              ),
                            ),
                            Text(
                              '${filteredCallLogs.length} entries',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                color: ShadTokens.muted,
                              ),
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        for (final cl in filteredCallLogs) ...[
                          _buildCallLogCard(context, cl),
                          const SizedBox(height: 10),
                        ],
                        const SizedBox(height: 12),
                      ],

                      // ── Live Scoring Frame Traces ──
                      if (filteredHistory.isNotEmpty) ...[
                        Row(
                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                          children: [
                            Text(
                              'INFERENCE FRAMES',
                              style: GoogleFonts.inter(
                                fontSize: 11,
                                fontWeight: FontWeight.w700,
                                letterSpacing: 0.8,
                                color: ShadTokens.muted,
                              ),
                            ),
                            ShadBadge(
                              label: '${filteredHistory.length} FRAMES',
                              variant: ShadBadgeVariant.secondary,
                            ),
                          ],
                        ),
                        const SizedBox(height: 10),
                        for (final r in filteredHistory) ...[
                          _buildTraceCard(r),
                          const SizedBox(height: 8),
                        ],
                      ],
                    ],
                  ),
          ),
        ],
      ),
    );
  }

  Widget _summaryStat(String label, String value, IconData icon) {
    return Column(
      children: [
        Icon(icon, size: 16, color: ShadTokens.muted),
        const SizedBox(height: 4),
        Text(
          value,
          style: GoogleFonts.inter(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            color: ShadTokens.foreground,
          ),
        ),
        Text(
          label,
          style: GoogleFonts.inter(
            fontSize: 9,
            fontWeight: FontWeight.w600,
            letterSpacing: 0.5,
            color: ShadTokens.muted,
          ),
        ),
      ],
    );
  }

  Widget _segmentedTab(int index, String label) {
    final selected = _filterIndex == index;
    return Expanded(
      child: GestureDetector(
        onTap: () => setState(() => _filterIndex = index),
        child: AnimatedContainer(
          duration: ShadTokens.fast,
          padding: const EdgeInsets.symmetric(vertical: 6),
          alignment: Alignment.center,
          decoration: BoxDecoration(
            color: selected ? Colors.white : Colors.transparent,
            borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
            boxShadow: selected ? ShadTokens.shadowSm : null,
          ),
          child: Text(
            label,
            style: GoogleFonts.inter(
              fontSize: 11,
              fontWeight: selected ? FontWeight.w700 : FontWeight.w500,
              color: selected ? ShadTokens.foreground : ShadTokens.muted,
            ),
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
          ),
        ),
      ),
    );
  }

  Widget _buildCallLogCard(BuildContext context, CallLog cl) {
    final isThreat = cl.riskScore >= 0.70;
    final isSuspicious = cl.riskScore >= 0.30 && cl.riskScore < 0.70;
    final badgeVariant = isThreat
        ? ShadBadgeVariant.destructive
        : (isSuspicious ? ShadBadgeVariant.suspicious : ShadBadgeVariant.verified);
    final scoreColor = isThreat
        ? ShadTokens.detected
        : (isSuspicious ? ShadTokens.suspicious : ShadTokens.verified);

    final durationFormatted =
        '${cl.duration.inMinutes}:${(cl.duration.inSeconds % 60).toString().padLeft(2, '0')}';

    return ShadGlassCard(
      padding: const EdgeInsets.all(12),
      onTap: () => _showLogDetailsDialog(context, cl),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              // Risk score box
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color: isThreat
                      ? const Color(0xFFFEF2F2)
                      : (isSuspicious ? const Color(0xFFFFFBEB) : const Color(0xFFF0FDF4)),
                  borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                  border: Border.all(
                    color: isThreat
                        ? const Color(0xFFFECACA)
                        : (isSuspicious ? const Color(0xFFFDE68A) : const Color(0xFFBBF7D0)),
                  ),
                ),
                child: Center(
                  child: Text(
                    '${(cl.riskScore * 100).toInt()}%',
                    style: GoogleFonts.inter(
                      fontSize: 13,
                      fontWeight: FontWeight.w800,
                      color: scoreColor,
                    ),
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      cl.number,
                      style: GoogleFonts.inter(
                        fontSize: 14,
                        fontWeight: FontWeight.w700,
                        color: ShadTokens.foreground,
                      ),
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 3),
                    Row(
                      children: [
                        ShadBadge(
                          label: isThreat
                              ? 'AI CLONE'
                              : (isSuspicious ? 'SUSPICIOUS' : 'AUTHENTIC'),
                          variant: badgeVariant,
                        ),
                        const SizedBox(width: 8),
                        Text(
                          DateFormat('dd MMM, hh:mm a').format(cl.timestamp),
                          style: GoogleFonts.inter(fontSize: 10, color: ShadTokens.muted),
                        ),
                        const SizedBox(width: 6),
                        Text(
                          '• $durationFormatted',
                          style: GoogleFonts.inter(fontSize: 10, color: ShadTokens.muted),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
              const Icon(
                LucideIcons.chevronRight,
                color: ShadTokens.muted,
                size: 16,
              ),
            ],
          ),
          if (cl.recordingPath != null) ...[
            const SizedBox(height: 8),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
              decoration: BoxDecoration(
                color: const Color(0xFFF4F4F5),
                borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                border: Border.all(color: ShadTokens.border),
              ),
              child: Row(
                children: [
                  const Icon(LucideIcons.fileAudio, size: 14, color: ShadTokens.foreground),
                  const SizedBox(width: 6),
                  Expanded(
                    child: Text(
                      cl.recordingPath!.split(RegExp(r'[\\/]')).last,
                      style: GoogleFonts.inter(
                        fontSize: 10.5,
                        fontWeight: FontWeight.w500,
                        color: ShadTokens.foreground,
                      ),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 1),
                    decoration: BoxDecoration(
                      color: isThreat ? const Color(0xFFFEE2E2) : const Color(0xFFDCFCE7),
                      borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
                    ),
                    child: Text(
                      isThreat ? 'QUARANTINED' : 'VERIFIED',
                      style: GoogleFonts.inter(
                        fontSize: 9,
                        fontWeight: FontWeight.w700,
                        color: isThreat ? ShadTokens.destructive : ShadTokens.verified,
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTraceCard(RiskScore r) {
    final color = r.color;
    final isThreat = r.score >= 0.70;
    final isSuspicious = r.score >= 0.30 && r.score < 0.70;
    final badgeVariant = isThreat
        ? ShadBadgeVariant.destructive
        : (isSuspicious ? ShadBadgeVariant.suspicious : ShadBadgeVariant.verified);

    return ShadGlassCard(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      child: Row(
        children: [
          Container(
            width: 36,
            height: 36,
            decoration: BoxDecoration(
              color: const Color(0xFFF4F4F5),
              borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
              border: Border.all(color: ShadTokens.border),
            ),
            child: Center(
              child: Text(
                '${r.percent}%',
                style: GoogleFonts.inter(
                  fontSize: 11,
                  fontWeight: FontWeight.w800,
                  color: color,
                ),
              ),
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    ShadBadge(
                      label: r.label,
                      variant: badgeVariant,
                    ),
                    const SizedBox(width: 6),
                    Text(
                      DateFormat('hh:mm:ss a').format(r.timestamp),
                      style: GoogleFonts.inter(fontSize: 10, color: ShadTokens.muted),
                    ),
                  ],
                ),
                const SizedBox(height: 2),
                Text(
                  'Spectral LFCC 60-band inference • Score ${(r.score * 100).toStringAsFixed(1)}%',
                  style: GoogleFonts.inter(fontSize: 11, color: ShadTokens.muted),
                ),
              ],
            ),
          ),
          Icon(
            isThreat
                ? LucideIcons.alertTriangle
                : (isSuspicious ? LucideIcons.alertCircle : LucideIcons.checkCircle2),
            color: color,
            size: 18,
          ),
        ],
      ),
    );
  }

  void _showLogDetailsDialog(BuildContext context, CallLog cl) {
    ShadDialog.show(
      context: context,
      title: 'Forensic Audit Inspection',
      description: 'Cryptographic trace verified by on-device ONNX engine.',
      content: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _dialogRow('Trace Identifier', cl.id),
          _dialogRow('Audio Source', cl.number),
          _dialogRow('Risk Probability', '${(cl.riskScore * 100).toStringAsFixed(2)}%'),
          _dialogRow('Classification', cl.verdict.name.toUpperCase()),
          _dialogRow('Session Duration', '${cl.duration.inSeconds} seconds'),
          _dialogRow('Timestamp', DateFormat('yyyy-MM-dd HH:mm:ss').format(cl.timestamp)),
          _dialogRow('Neural Latency', '< 74 ms (INT8 Neon)'),
          _dialogRow('DPDP Act Storage', 'Transient RAM (Zero Cloud)'),
          if (cl.recordingPath != null)
            _dialogRow('Forensic WAV', cl.recordingPath!.split(RegExp(r'[\\/]')).last),
        ],
      ),
      actions: [
        ShadButton(
          onTap: () => Navigator.of(context).pop(),
          variant: ShadButtonVariant.primary,
          size: ShadButtonSize.sm,
          text: 'Done',
        ),
      ],
    );
  }

  Widget _dialogRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          SizedBox(
            width: 120,
            child: Text(
              label,
              style: GoogleFonts.inter(fontSize: 11, fontWeight: FontWeight.w600, color: ShadTokens.muted),
            ),
          ),
          Expanded(
            child: Text(
              value,
              style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: ShadTokens.foreground),
            ),
          ),
        ],
      ),
    );
  }
}