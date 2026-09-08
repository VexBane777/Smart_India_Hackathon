import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import '../providers/call_state_provider.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/status_indicator.dart';
import '../utils/constants.dart';
import 'call_screen.dart';
import 'settings_screen.dart';
import 'logs_screen.dart';
import 'protected_call_screen.dart';
import 'voip_protection_screen.dart';

class HomeScreen extends StatelessWidget {
  const HomeScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final settings = context.watch<SettingsProvider>();
    final callState = context.watch<CallStateProvider>().state;
    final risk = context.watch<RiskScoreProvider>().current;

    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(
        backgroundColor: Colors.white,
        elevation: 0,
        title: Row(children: [
          Container(
            padding: const EdgeInsets.all(6),
            decoration: BoxDecoration(color: AppColors.primary, borderRadius: BorderRadius.circular(8)),
            child: const Icon(Icons.shield_rounded, color: Colors.white, size: 20),
          ),
          const SizedBox(width: 10),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text(AppConstants.appName, style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800, color: AppColors.textPrimary)),
            Text(AppConstants.tagline, style: const TextStyle(fontSize: 10, color: AppColors.textSecondary, fontWeight: FontWeight.w500)),
          ]),
        ]),
        actions: [
          IconButton(icon: const Icon(Icons.settings_outlined, color: AppColors.textPrimary), onPressed: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen()))),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          // Shield + status
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.06), blurRadius: 12)]),
            child: Column(children: [
              ShieldIcon(active: settings.protectionEnabled, size: 80),
              const SizedBox(height: 14),
              StatusIndicator(active: settings.protectionEnabled),
              const SizedBox(height: 12),
              Text(settings.protectionEnabled ? 'Your calls are being protected' : 'Protection is paused',
                  style: TextStyle(fontSize: 12, color: Colors.black.withValues(alpha: 0.6))),
              const SizedBox(height: 14),
              // Protection toggle
              Row(mainAxisAlignment: MainAxisAlignment.center, children: [
                const Text('Protection', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w600)),
                const SizedBox(width: 10),
                Switch(value: settings.protectionEnabled, activeThumbColor: AppColors.verified, onChanged: (v) => context.read<SettingsProvider>().setProtection(v)),
              ]),
            ]),
          ),
          const SizedBox(height: 14),
          // Live call banner
          if (!callState.isIdle)
            InkWell(
              onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const CallScreen())),
              child: Container(
                padding: const EdgeInsets.all(14),
                decoration: BoxDecoration(
                  color: callState.isActive ? AppColors.primary : Colors.white,
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: callState.isActive ? AppColors.primary : Colors.black12),
                ),
                child: Row(children: [
                  Icon(callState.isActive ? Icons.call : Icons.phone_in_talk, color: callState.isActive ? Colors.white : AppColors.primary),
                  const SizedBox(width: 12),
                  Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                    Text(callState.isActive ? 'Call in progress' : 'Incoming call',
                        style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: callState.isActive ? Colors.white : AppColors.textPrimary)),
                    Text(callState.number ?? 'Unknown', style: TextStyle(fontSize: 12, color: callState.isActive ? Colors.white70 : Colors.black54)),
                  ])),
                  if (risk != null)
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                      decoration: BoxDecoration(color: risk.color, borderRadius: BorderRadius.circular(999)),
                      child: Text('${risk.percent}%', style: const TextStyle(color: Colors.white, fontSize: 12, fontWeight: FontWeight.w800)),
                    ),
                  const SizedBox(width: 8),
                  Icon(Icons.chevron_right, color: callState.isActive ? Colors.white70 : Colors.black38),
                ]),
              ),
            ),
          if (!callState.isIdle) const SizedBox(height: 14),
          // Stats row
          Row(children: [
            _statCard(context, icon: Icons.verified_user_rounded, label: 'Calls Protected', value: '${context.watch<RiskScoreProvider>().history.length}', color: AppColors.verified),
            const SizedBox(width: 12),
            _statCard(context, icon: Icons.warning_amber_rounded, label: 'Threats Blocked', value: '${context.watch<RiskScoreProvider>().history.where((r) => r.score > 0.7).length}', color: AppColors.detected),
          ]),
          const SizedBox(height: 14),
          // Quick actions
          const Text('Quick Actions', style: TextStyle(fontSize: 13, fontWeight: FontWeight.w800)),
          const SizedBox(height: 8),
          Row(children: [
            Expanded(child: _actionBtn(context, icon: Icons.phone_in_talk, label: 'Live Call', subtitle: 'Risk meter', onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const CallScreen())))),
            const SizedBox(width: 10),
            Expanded(child: _actionBtn(context, icon: Icons.history, label: 'Call Logs', subtitle: 'History', onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const LogsScreen())))),
            const SizedBox(width: 10),
            Expanded(child: _actionBtn(context, icon: Icons.tune, label: 'Settings', subtitle: 'Tune alerts', onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const SettingsScreen())))),
          ]),
          const SizedBox(height: 10),
          Row(children: [
            Expanded(child: _actionBtn(context, icon: Icons.shield_outlined, label: 'Protected Call', subtitle: 'VAANI-to-VAANI', onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const ProtectedCallScreen())))),
            const SizedBox(width: 10),
            Expanded(child: _actionBtn(context, icon: Icons.videocam_outlined, label: 'VoIP Protection', subtitle: 'WhatsApp / Zoom', onTap: () => Navigator.push(context, MaterialPageRoute(builder: (_) => const VoipProtectionScreen())))),
          ]),
          const SizedBox(height: 16),
          // Privacy notice
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(color: const Color(0xFFE3F2FD), borderRadius: BorderRadius.circular(12), border: Border.all(color: AppColors.primary.withValues(alpha: 0.2))),
            child: Row(crossAxisAlignment: CrossAxisAlignment.start, children: [
              const Icon(Icons.lock_outline, size: 18, color: AppColors.primary),
              const SizedBox(width: 8),
              Expanded(child: Text('Privacy-first: raw audio is processed in memory and discarded. Only risk scores & call metadata are stored — DPDP Act aligned.',
                  style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.7), height: 1.35))),
            ]),
          ),
          const SizedBox(height: 12),
          // SIH traceability footer
          Text('SIH 2025-26 • Maps to all 5 problem-statement capability areas — see About in Settings',
              textAlign: TextAlign.center, style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.45))),
        ],
      ),
    );
  }

  Widget _statCard(BuildContext context, {required IconData icon, required String label, required String value, required Color color}) {
    return Expanded(
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), boxShadow: [BoxShadow(color: Colors.black.withValues(alpha: 0.04), blurRadius: 8)]),
        child: Row(children: [
          Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: color.withValues(alpha: 0.12), borderRadius: BorderRadius.circular(10)), child: Icon(icon, size: 18, color: color)),
          const SizedBox(width: 10),
          Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(value, style: const TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
            Text(label, style: const TextStyle(fontSize: 10, color: Colors.black54, fontWeight: FontWeight.w600)),
          ]),
        ]),
      ),
    );
  }

  Widget _actionBtn(BuildContext context, {required IconData icon, required String label, required String subtitle, required VoidCallback onTap}) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(14),
      child: Container(
        padding: const EdgeInsets.all(14),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
        child: Column(children: [
          Icon(icon, color: AppColors.primary),
          const SizedBox(height: 6),
          Text(label, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
          Text(subtitle, style: const TextStyle(fontSize: 10, color: Colors.black54)),
        ]),
      ),
    );
  }
}
