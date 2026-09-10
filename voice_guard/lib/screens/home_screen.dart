import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/shad_glass_card.dart';
import '../widgets/shad_glass_scaffold.dart';
import '../widgets/shad_half_donut_chart.dart';
import '../widgets/shad_badge.dart';
import '../design/tokens.dart';
import '../utils/constants.dart';
import 'call_screen.dart';
import 'logs_screen.dart';
import 'protected_call_screen.dart';
import 'voip_protection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final callState = context.watch<CallStateProvider>().state;
    final risk = context.watch<RiskScoreProvider>().current;

    return ShadGlassScaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: Row(
          children: [
            Container(
              width: 32,
              height: 32,
              decoration: BoxDecoration(
                color: ShadTokens.primary,
                borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                boxShadow: ShadTokens.shadowSm,
              ),
              child: const Icon(LucideIcons.activity, color: Colors.white, size: 17),
            ),
            const SizedBox(width: 10),
            const Text(
              AppConstants.appName,
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.5,
                color: ShadTokens.foreground,
              ),
            ),
          ],
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        children: [
          // ── Active Call Float Banner (Only shown when call is underway) ──
          if (!callState.isIdle) ...[
            ShadGlassCard(
              margin: const EdgeInsets.only(bottom: 16),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              tintColor: callState.isActive ? ShadTokens.primary : Colors.white,
              opacity: callState.isActive ? 0.92 : 0.80,
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
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(ShadTokens.radiusMd),
                      border: Border.all(color: ShadTokens.border),
                    ),
                    child: Icon(
                      callState.isActive
                          ? LucideIcons.phoneCall
                          : LucideIcons.phoneIncoming,
                      color: callState.isActive ? ShadTokens.primary : const Color(0xFF2563EB),
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
                          style: TextStyle(
                            fontSize: 13,
                            fontWeight: FontWeight.w700,
                            color: callState.isActive ? Colors.white : ShadTokens.foreground,
                          ),
                        ),
                        Text(
                          callState.number ?? 'Carrier Call',
                          style: TextStyle(
                            fontSize: 11,
                            color: callState.isActive ? Colors.white70 : ShadTokens.muted,
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
                  Icon(
                    LucideIcons.chevronRight,
                    color: callState.isActive ? Colors.white70 : ShadTokens.muted,
                    size: 16,
                  ),
                ],
              ),
            ),
          ],

          // ── Minimalist Half-Donut Chart: Human vs AI ──
          ShadHalfDonutChart(
            totalCount: 148,
            humanCount: 138,
            aiCount: 10,
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const LogsScreen()),
            ),
          ),
          const SizedBox(height: 24),

          // ── Monochromatic Capabilities Section Title ──
          const Padding(
            padding: EdgeInsets.only(left: 4, bottom: 12),
            child: Text(
              'Security Capabilities',
              style: TextStyle(
                fontSize: 12,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.6,
                color: ShadTokens.muted,
              ),
            ),
          ),

          // ── Capability 1: Live Call & Scanner ──
          ShadGlassCard(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const CallScreen()),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF4F4F5),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: const Icon(
                    LucideIcons.phoneCall,
                    color: ShadTokens.foreground,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Live Call & Mic Scanner',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.2,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Dialpad, real-time telemetry, & microphone clone test',
                        style: TextStyle(fontSize: 11, color: ShadTokens.muted),
                      ),
                    ],
                  ),
                ),
                const Icon(LucideIcons.chevronRight, size: 16, color: ShadTokens.muted),
              ],
            ),
          ),

          // ── Capability 2: VoIP Protection ──
          ShadGlassCard(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const VoipProtectionScreen()),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF4F4F5),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: const Icon(
                    LucideIcons.headphones,
                    color: ShadTokens.foreground,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'VoIP App Shield',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.2,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Playback capture for WhatsApp, Telegram, & Zoom',
                        style: TextStyle(fontSize: 11, color: ShadTokens.muted),
                      ),
                    ],
                  ),
                ),
                const Icon(LucideIcons.chevronRight, size: 16, color: ShadTokens.muted),
              ],
            ),
          ),

          // ── Capability 3: Protected Call ──
          ShadGlassCard(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const ProtectedCallScreen()),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF4F4F5),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: const Icon(
                    LucideIcons.shieldCheck,
                    color: ShadTokens.foreground,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Encrypted Protected Call',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.2,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'End-to-end encrypted WebRTC P2P voice channel',
                        style: TextStyle(fontSize: 11, color: ShadTokens.muted),
                      ),
                    ],
                  ),
                ),
                const Icon(LucideIcons.chevronRight, size: 16, color: ShadTokens.muted),
              ],
            ),
          ),

          // ── Capability 4: Audit Traces & Recordings ──
          ShadGlassCard(
            margin: const EdgeInsets.only(bottom: 24),
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 15),
            onTap: () => Navigator.push(
              context,
              MaterialPageRoute(builder: (_) => const LogsScreen()),
            ),
            child: Row(
              children: [
                Container(
                  width: 44,
                  height: 44,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF4F4F5),
                    borderRadius: BorderRadius.circular(ShadTokens.radiusLg),
                    border: Border.all(color: ShadTokens.border),
                  ),
                  child: const Icon(
                    LucideIcons.fileText,
                    color: ShadTokens.foreground,
                    size: 20,
                  ),
                ),
                const SizedBox(width: 14),
                const Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Audit Traces & Recordings',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w700,
                          letterSpacing: -0.2,
                          color: ShadTokens.foreground,
                        ),
                      ),
                      SizedBox(height: 2),
                      Text(
                        'Inspect forensic WAV recordings & inference logs',
                        style: TextStyle(fontSize: 11, color: ShadTokens.muted),
                      ),
                    ],
                  ),
                ),
                const Icon(LucideIcons.chevronRight, size: 16, color: ShadTokens.muted),
              ],
            ),
          ),
          const SizedBox(height: 8),
        ],
      ),
    );
  }
}
