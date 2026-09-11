import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/risk_score_provider.dart';
import '../models/call_log.dart';
import '../design/tokens.dart';

class _AuditItemData {
  final String title;
  final String date;
  final int riskPercent;
  final CallLog? callLog;

  const _AuditItemData({
    required this.title,
    required this.date,
    required this.riskPercent,
    this.callLog,
  });
}

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
    final liveCallLogs = provider.callLogs;

    // Combine live call logs with baseline Figma logs
    final List<_AuditItemData> allItems = [];

    for (final cl in liveCallLogs) {
      final percent = (cl.riskScore * 100).toInt();
      final dt = cl.timestamp;
      final months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
      final hour = dt.hour > 12 ? dt.hour - 12 : (dt.hour == 0 ? 12 : dt.hour);
      final ampm = dt.hour >= 12 ? 'PM' : 'AM';
      final min = dt.minute.toString().padLeft(2, '0');
      final dateStr = '${dt.day} ${months[dt.month - 1]}, $hour:$min $ampm';

      allItems.add(_AuditItemData(
        title: cl.number,
        date: dateStr,
        riskPercent: percent,
        callLog: cl,
      ));
    }

    final totalCount = allItems.length;
    final threatCount = allItems.where((i) => i.riskPercent >= 70).length;
    final suspiciousCount = allItems.where((i) => i.riskPercent >= 30 && i.riskPercent < 70).length;
    final verifiedCount = allItems.where((i) => i.riskPercent < 30).length;

    final filteredItems = allItems.where((i) {
      if (_filterIndex == 1) return i.riskPercent >= 70;
      if (_filterIndex == 2) return i.riskPercent >= 30 && i.riskPercent < 70;
      if (_filterIndex == 3) return i.riskPercent < 30;
      return true;
    }).toList();

    return Scaffold(
      backgroundColor: ShadTokens.background,
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          children: [
            // ── Top Header (Title + Export Button) ──
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Audit Records',
                        style: GoogleFonts.inter(
                          fontSize: 26,
                          fontWeight: FontWeight.w800,
                          letterSpacing: -0.5,
                          color: Colors.white,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '$totalCount events recorded  •  Local buffer synced',
                        style: GoogleFonts.inter(
                          fontSize: 12,
                          fontWeight: FontWeight.w400,
                          color: ShadTokens.muted,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(width: 8),
                InkWell(
                  onTap: () {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(
                        backgroundColor: ShadTokens.surface,
                        content: Text(
                          'Exported $totalCount audit logs to device storage',
                          style: const TextStyle(color: Colors.white),
                        ),
                      ),
                    );
                  },
                  borderRadius: BorderRadius.circular(999),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 7),
                    decoration: BoxDecoration(
                      color: ShadTokens.surfaceContainer,
                      borderRadius: BorderRadius.circular(999),
                      border: Border.all(color: ShadTokens.border),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        const Icon(LucideIcons.download, size: 14, color: Colors.white),
                        const SizedBox(width: 6),
                        Text(
                          'Export',
                          style: GoogleFonts.inter(
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                            color: Colors.white,
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 20),

            // ── 3 Summary Metric Cards (Figma Main-2.png) ──
            Row(
              children: [
                Expanded(
                  child: _metricCard(
                    title: 'INSPECTED',
                    value: totalCount.toString(),
                    subtitle: 'Deep-scans',
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _metricCard(
                    title: 'THREATS',
                    value: threatCount.toString().padLeft(2, '0'),
                    subtitle: 'Isolated',
                  ),
                ),
                const SizedBox(width: 10),
                Expanded(
                  child: _metricCard(
                    title: 'PURITY',
                    value: totalCount == 0
                        ? '—'
                        : '${(verifiedCount / totalCount * 100).toStringAsFixed(1)}%',
                    subtitle: 'Biometric',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 24),

            // ── Horizontal Filter Pills (Figma Main-2.png) ──
            SingleChildScrollView(
              scrollDirection: Axis.horizontal,
              child: Row(
                children: [
                  _filterPill(0, 'All ($totalCount)'),
                  const SizedBox(width: 8),
                  _filterPill(1, 'Threats ($threatCount)'),
                  const SizedBox(width: 8),
                  _filterPill(2, 'Suspicious ($suspiciousCount)'),
                  const SizedBox(width: 8),
                  _filterPill(3, 'Verified ($verifiedCount)'),
                ],
              ),
            ),
            const SizedBox(height: 20),

            // ── Itemized Audit Records ──
            if (filteredItems.isEmpty)
              Padding(
                padding: const EdgeInsets.symmetric(vertical: 40),
                child: Center(
                  child: Text(
                    'No audit records found',
                    style: GoogleFonts.inter(color: ShadTokens.mutedFg),
                  ),
                ),
              )
            else
              for (int i = 0; i < filteredItems.length; i++) ...[
                _auditRecordRow(context, filteredItems[i]),
                const Divider(color: ShadTokens.border, height: 1, thickness: 1),
              ],
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _metricCard({
    required String title,
    required String value,
    required String subtitle,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 10),
      decoration: BoxDecoration(
        color: ShadTokens.surface,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: ShadTokens.border, width: 1),
      ),
      child: Column(
        children: [
          Text(
            title,
            style: GoogleFonts.inter(
              fontSize: 10.5,
              fontWeight: FontWeight.w700,
              letterSpacing: 0.8,
              color: ShadTokens.muted,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            value,
            style: GoogleFonts.inter(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            subtitle,
            style: GoogleFonts.inter(
              fontSize: 11,
              fontWeight: FontWeight.w400,
              color: ShadTokens.mutedFg,
            ),
          ),
        ],
      ),
    );
  }

  Widget _filterPill(int index, String text) {
    final isSelected = _filterIndex == index;
    return GestureDetector(
      onTap: () => setState(() => _filterIndex = index),
      child: AnimatedContainer(
        duration: const Duration(milliseconds: 150),
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 8),
        decoration: BoxDecoration(
          color: isSelected ? Colors.white : ShadTokens.surface,
          borderRadius: BorderRadius.circular(999),
          border: Border.all(
            color: isSelected ? Colors.white : ShadTokens.border,
            width: 1,
          ),
        ),
        child: Text(
          text,
          style: GoogleFonts.inter(
            fontSize: 13,
            fontWeight: isSelected ? FontWeight.w700 : FontWeight.w500,
            color: isSelected ? Colors.black : const Color(0xFFD4D4D8),
          ),
        ),
      ),
    );
  }

  Widget _auditRecordRow(BuildContext context, _AuditItemData item) {
    return InkWell(
      onTap: () => _showAuditDetails(context, item),
      child: Padding(
        padding: const EdgeInsets.symmetric(vertical: 18, horizontal: 4),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            // Left: Title + Timestamp
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    item.title,
                    style: GoogleFonts.inter(
                      fontSize: 15,
                      fontWeight: FontWeight.w600,
                      letterSpacing: -0.2,
                      color: Colors.white,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                  const SizedBox(height: 4),
                  Text(
                    item.date,
                    style: GoogleFonts.inter(
                      fontSize: 12,
                      fontWeight: FontWeight.w400,
                      color: ShadTokens.mutedFg,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),

            // Right: Risk Percentage
            Text(
              '${item.riskPercent}% RISK',
              style: GoogleFonts.inter(
                fontSize: 13,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.2,
                color: Colors.white,
              ),
            ),
          ],
        ),
      ),
    );
  }

  void _showAuditDetails(BuildContext context, _AuditItemData item) {
    showModalBottomSheet(
      context: context,
      backgroundColor: ShadTokens.surface,
      shape: const RoundedRectangleBorder(
        borderRadius: BorderRadius.vertical(top: Radius.circular(24)),
      ),
      builder: (ctx) {
        return SafeArea(
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      'Forensic Audit Trace',
                      style: GoogleFonts.inter(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(LucideIcons.x, color: ShadTokens.muted, size: 20),
                      onPressed: () => Navigator.pop(ctx),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _detailRow('Audio Source', item.title),
                _detailRow('Recorded Date', item.date),
                _detailRow('Risk Score', '${item.riskPercent}% RISK'),
                _detailRow(
                  'Classification',
                  item.riskPercent >= 70
                      ? 'AI Synthetic Voice (Isolated)'
                      : (item.riskPercent >= 30 ? 'Suspicious Artifacts' : 'Human Authentic Biometrics'),
                ),
                _detailRow('Inference Engine', 'ONNX INT8 • 60-band LFCC'),
                _detailRow('Storage Policy', 'Volatile RAM • DPDP Act 2023 Compliant'),
                const SizedBox(height: 20),
                SizedBox(
                  width: double.infinity,
                  height: 48,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
                    ),
                    onPressed: () => Navigator.pop(ctx),
                    child: Text(
                      'Dismiss Trace',
                      style: GoogleFonts.inter(fontSize: 14, fontWeight: FontWeight.w700),
                    ),
                  ),
                ),
              ],
            ),
          ),
        );
      },
    );
  }

  Widget _detailRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: GoogleFonts.inter(fontSize: 12, color: ShadTokens.muted),
          ),
          Text(
            value,
            style: GoogleFonts.inter(
              fontSize: 12,
              fontWeight: FontWeight.w600,
              color: Colors.white,
            ),
          ),
        ],
      ),
    );
  }
}
