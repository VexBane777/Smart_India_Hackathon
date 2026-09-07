import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import '../services/call_service.dart';
import '../utils/constants.dart';
import '../utils/permissions.dart';

class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});
  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  bool _isDefaultDialer = false;
  bool _hasOverlay = false;
  bool _hasPhonePerms = false;

  @override
  void initState() {
    super.initState();
    _refresh();
  }

  Future<void> _refresh() async {
    final cs = CallService();
    final d = await cs.isDefaultDialer();
    final o = await cs.hasOverlayPermission();
    final p = await PermissionHelper.hasPhonePermissions();
    if (mounted) setState(() { _isDefaultDialer = d; _hasOverlay = o; _hasPhonePerms = p; });
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<SettingsProvider>();
    return Scaffold(
      backgroundColor: AppColors.surface,
      appBar: AppBar(backgroundColor: Colors.white, elevation: 0, title: const Text('Settings', style: TextStyle(fontSize: 16, fontWeight: FontWeight.w800))),
      body: ListView(padding: const EdgeInsets.all(16), children: [
        _section('Protection'),
        _toggleTile('${AppConstants.appName} Protection', 'Master switch for all detection', s.protectionEnabled, (v) => s.setProtection(v)),
        _toggleTile('In-Call Overlay Warning', 'Floating banner when AI voice is detected', s.overlayEnabled, (v) => s.setOverlay(v)),
        _toggleTile('Sound Alerts', 'Beep/vibration on high risk', s.soundEnabled, (v) => s.setSound(v)),
        const SizedBox(height: 14),
        _section('Sensitivity — Detection Threshold'),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Row(children: [
              const Text('Threshold', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w700)),
              const Spacer(),
              Container(padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4), decoration: BoxDecoration(color: AppColors.primary.withValues(alpha: 0.10), borderRadius: BorderRadius.circular(999)), child: Text(s.sensitivity.toStringAsFixed(2), style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: AppColors.primary))),
            ]),
            // ignore: deprecated_member_use
            Slider(value: s.sensitivity, min: 0.50, max: 0.90, divisions: 8, label: s.sensitivity.toStringAsFixed(2), activeColor: AppColors.primary, onChanged: (v) => s.setSensitivity(v)),
            Text('Lower = more sensitive (more alerts). Maps to "configurable thresholds per scenario" in the SIH brief.',
                style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.55))),
          ]),
        ),
        const SizedBox(height: 14),
        _section('Permissions & Dialer'),
        _permTile(icon: Icons.phone, title: 'Phone & Microphone', subtitle: 'Required to detect calls and capture audio', granted: _hasPhonePerms, onTap: () async { await PermissionHelper.requestPhonePermissions(); _refresh(); }),
        const SizedBox(height: 8),
        _permTile(icon: Icons.open_in_new, title: 'Display over other apps', subtitle: 'Required for in-call warning overlay', granted: _hasOverlay, onTap: () async { await CallService().requestOverlayPermission(); _refresh(); }),
        const SizedBox(height: 8),
        _permTile(icon: Icons.phone_in_talk, title: 'Default Dialer', subtitle: _isDefaultDialer ? '${AppConstants.appName} is your default dialer' : 'Set as default to intercept calls via InCallService', granted: _isDefaultDialer, onTap: () async { await CallService().setAsDefaultDialer(); _refresh(); }),
        const SizedBox(height: 14),
        _section('Privacy & Compliance'),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: const Color(0xFFE8F5E9), borderRadius: BorderRadius.circular(14), border: Border.all(color: AppColors.verified.withValues(alpha: 0.25))),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Row(children: [Icon(Icons.lock, size: 16, color: AppColors.verified), SizedBox(width: 6), Text('Privacy-First Design', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800, color: AppColors.verified))]),
            const SizedBox(height: 6),
            Text('• Raw PCM is processed in RAM and discarded — never written to disk.\n• Only {timestamp, caller, riskScore, verdict} is stored locally.\n• Inference runs on-device (TFLite); backend receives LFCC features only, not raw audio.\n• DPDP Act–aligned data minimization; no audio retention.',
                style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.7), height: 1.4)),
          ]),
        ),
        const SizedBox(height: 14),
        _section('Platform & Integration'),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            const Text('API for Enterprise / Banking Apps', style: TextStyle(fontSize: 12, fontWeight: FontWeight.w800)),
            const SizedBox(height: 6),
            Text('POST /v1/analyze-chunk  — LFCC + metadata → risk score\nPOST /v1/alert            — webhook for downstream escalation\nSee backend/README.md & OpenAPI docs. The Android app uses on-device inference for latency; the backend demonstrates the SDK path for judges.',
                style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.65), height: 1.4)),
          ]),
        ),
        const SizedBox(height: 14),
        _section('About'),
        Container(
          padding: const EdgeInsets.all(14),
          decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(14), border: Border.all(color: Colors.black12)),
          child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
            Text(AppConstants.appName, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w800)),
            Text(AppConstants.tagline, style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.6))),
            const SizedBox(height: 6),
            Text('Version ${AppConstants.version} • ${AppConstants.org}\nKnown limitation (Android 10+): VOICE_CALL source is blocked for non-system apps — POC uses MIC + speakerphone; production needs telecom-side media forking.',
                style: TextStyle(fontSize: 10, color: Colors.black.withValues(alpha: 0.55), height: 1.35)),
          ]),
        ),
        const SizedBox(height: 24),
      ]),
    );
  }

  Widget _section(String t) => Padding(padding: const EdgeInsets.only(bottom: 8, top: 4), child: Text(t, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w800, letterSpacing: 0.4, color: AppColors.textSecondary)));
  Widget _toggleTile(String title, String sub, bool value, ValueChanged<bool> onChanged) => Container(
        margin: const EdgeInsets.only(bottom: 8),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.black12)),
        child: SwitchListTile(value: value, onChanged: onChanged, title: Text(title, style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700)), subtitle: Text(sub, style: const TextStyle(fontSize: 11, color: Colors.black54)), activeThumbColor: AppColors.primary),
      );
  Widget _permTile({required IconData icon, required String title, required String subtitle, required bool granted, required VoidCallback onTap}) => Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: granted ? AppColors.verified.withValues(alpha: 0.3) : Colors.black12)),
        child: Row(children: [
          Container(padding: const EdgeInsets.all(8), decoration: BoxDecoration(color: granted ? AppColors.verifiedBg : const Color(0xFFF5F5F5), borderRadius: BorderRadius.circular(10)), child: Icon(icon, size: 18, color: granted ? AppColors.verified : Colors.black54)),
          const SizedBox(width: 10),
          Expanded(child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [Text(title, style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w700)), Text(subtitle, style: const TextStyle(fontSize: 10, color: Colors.black54), maxLines: 2)])),
          const SizedBox(width: 8),
          FilledButton(onPressed: onTap, style: FilledButton.styleFrom(backgroundColor: granted ? AppColors.verified : AppColors.primary, padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6)), child: Text(granted ? 'Granted' : 'Grant', style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700))),
        ]),
      );
}
