import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../models/risk_score.dart';
import '../widgets/shad_half_donut_chart.dart';
import '../widgets/shad_badge.dart';
import 'call_screen.dart';
import 'logs_screen.dart';
import 'protected_call_screen.dart';
import 'voip_protection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final callState = context.watch<CallStateProvider>().state;
    final riskProvider = context.watch<RiskScoreProvider>();
    final risk = riskProvider.current;
    final callLogs = riskProvider.callLogs;
    final scannedCount = callLogs.length;
    final humanCount = callLogs.where((c) => c.verdict == Verdict.verified).length;
    final aiCount = callLogs.where((c) => c.verdict != Verdict.verified).length;

    return Scaffold(
      backgroundColor: const Color(0xFF131315),
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: Row(
          children: [
            Container(
              width: 3,
              height: 16,
              decoration: BoxDecoration(
                color: const Color(0xFF9AA8BB),
                borderRadius: BorderRadius.circular(2),
              ),
            ),
            const SizedBox(width: 8),
            Text(
              'SYSTEM STATUS',
              style: GoogleFonts.inter(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 1.5,
                color: const Color(0xFF9AA8BB),
              ),
            ),
          ],
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 8),
        children: [
          // ── Active Call Floating Banner (Only shown when call is underway) ──
          if (!callState.isIdle) ...[
            Container(
              margin: const EdgeInsets.only(bottom: 16),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              decoration: BoxDecoration(
                color: const Color(0xFF18181B),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF27272A)),
              ),
              child: InkWell(
                onTap: () => Navigator.push(
                  context,
                  MaterialPageRoute(builder: (_) => const CallScreen()),
                ),
                child: Row(
                  children: [
                    Container(
                      width: 38,
                      height: 38,
                      decoration: BoxDecoration(
                        color: const Color(0xFF27272A),
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Icon(
                        callState.isActive ? LucideIcons.phoneCall : LucideIcons.phoneIncoming,
                        color: const Color(0xFF21D4B2),
                        size: 18,
                      ),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            callState.isActive ? 'Live Call Monitored' : 'Incoming Call',
                            style: GoogleFonts.inter(
                              fontSize: 13,
                              fontWeight: FontWeight.w700,
                              color: Colors.white,
                            ),
                          ),
                          Text(
                            callState.number ?? 'Carrier Call',
                            style: GoogleFonts.inter(
                              fontSize: 11,
                              color: const Color(0xFF8E9192),
                            ),
                          ),
                        ],
                      ),
                    ),
                    if (risk != null)
                      ShadBadge(
                        label: '${risk.percent}% RISK',
                        variant: risk.score > 0.7
                            ? ShadBadgeVariant.destructive
                            : (risk.score > 0.3
                                ? ShadBadgeVariant.suspicious
                                : ShadBadgeVariant.verified),
                      ),
                    const SizedBox(width: 6),
                    const Icon(
                      LucideIcons.chevronRight,
                      color: Color(0xFF8E9192),
                      size: 16,
                    ),
                  ],
                ),
              ),
            ),
          ],

          // ── Minimalist Half-Donut Chart: 148 CELLS SCANNED ──
          ShadHalfDonutChart(
            totalCount: scannedCount,
            humanCount: humanCount,
            aiCount: aiCount,
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const LogsScreen()),
            ),
          ),
          const SizedBox(height: 28),

          // ── Defense Vectors Header ──
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 6),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  'DEFENSE VECTORS',
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w700,
                    letterSpacing: 1.0,
                    color: const Color(0xFF71717A),
                  ),
                ),
                Text(
                  '4 Active',
                  style: GoogleFonts.inter(
                    fontSize: 12,
                    fontWeight: FontWeight.w500,
                    color: const Color(0xFF71717A),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 8),

          // ── Grouped Defense Vectors Card (Figma Main.png) ──
          Container(
            decoration: BoxDecoration(
              color: const Color(0xFF18181B),
              borderRadius: BorderRadius.circular(20),
              border: Border.all(color: const Color(0xFF27272A), width: 1),
            ),
            child: Column(
              children: [
                // Vector 1: Live Call Shield
                _vectorRow(
                  icon: LucideIcons.shieldCheck,
                  title: 'Live Call Shield',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const CallScreen()),
                  ),
                ),
                const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),

                // Vector 2: VoIP App Guard
                _vectorRow(
                  icon: LucideIcons.phone,
                  title: 'VoIP App Guard',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const VoipProtectionScreen()),
                  ),
                ),
                const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),

                // Vector 3: Encrypted Voice
                _vectorRow(
                  icon: LucideIcons.lock,
                  title: 'Encrypted Voice',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const ProtectedCallScreen()),
                  ),
                ),
                const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),

                // Vector 4: Forensic Evidence
                _vectorRow(
                  icon: LucideIcons.audioWaveform,
                  title: 'Forensic Evidence',
                  onTap: () => Navigator.push(
                    context,
                    MaterialPageRoute(builder: (_) => const LogsScreen()),
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 32),

          // ── Run Acoustic Forensic Scan Stadium Button (Figma Main.png) ──
          GestureDetector(
            onTap: () {
              Navigator.push(
                context,
                MaterialPageRoute(builder: (_) => const CallScreen()),
              );
            },
            child: Container(
              height: 54,
              width: double.infinity,
              decoration: BoxDecoration(
                color: Colors.white,
                borderRadius: BorderRadius.circular(999),
                boxShadow: const [
                  BoxShadow(
                    color: Colors.black26,
                    blurRadius: 10,
                    offset: Offset(0, 4),
                  ),
                ],
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Icon(LucideIcons.fingerprint, color: Colors.black, size: 22),
                  const SizedBox(width: 10),
                  Text(
                    'Run Acoustic Forensic Scan',
                    style: GoogleFonts.inter(
                      fontSize: 14,
                      fontWeight: FontWeight.w700,
                      letterSpacing: -0.2,
                      color: Colors.black,
                    ),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _vectorRow({
    required IconData icon,
    required String title,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(18),
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 16),
        child: Row(
          children: [
            Container(
              width: 42,
              height: 42,
              decoration: BoxDecoration(
                color: const Color(0xFF1C1C1E),
                shape: BoxShape.circle,
                border: Border.all(color: const Color(0xFF27272A), width: 1),
              ),
              child: Icon(icon, color: Colors.white, size: 19),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Text(
                title,
                style: GoogleFonts.inter(
                  fontSize: 15,
                  fontWeight: FontWeight.w600,
                  letterSpacing: -0.2,
                  color: Colors.white,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
