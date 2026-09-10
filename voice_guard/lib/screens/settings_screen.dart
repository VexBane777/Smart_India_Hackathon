import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/settings_provider.dart';
import '../services/call_service.dart';
import '../utils/constants.dart';
import '../utils/permissions.dart';
import '../widgets/permission_card.dart';
import '../widgets/shad_glass_card.dart';
import '../widgets/shad_glass_scaffold.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_toggle.dart';
import '../design/tokens.dart';

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
    if (mounted) {
      setState(() {
        _isDefaultDialer = d;
        _hasOverlay = o;
        _hasPhonePerms = p;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    final s = context.watch<SettingsProvider>();

    return ShadGlassScaffold(
      appBar: AppBar(
        backgroundColor: Colors.transparent,
        elevation: 0,
        scrolledUnderElevation: 0,
        title: const Text(
          'Settings & Engine Controls',
          style: TextStyle(
            fontSize: 16,
            fontWeight: FontWeight.w800,
            letterSpacing: -0.3,
            color: ShadTokens.foreground,
          ),
        ),
      ),
      body: ListView(
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
        children: [
          // ── Active Protection Controls ──
          _sectionHeader('Active Protection Controls'),
          ShadGlassCard(
            padding: const EdgeInsets.all(ShadTokens.space3),
            child: Column(
              children: [
                ShadToggleTile(
                  title: 'Master Protection Switch',
                  subtitle: s.protectionEnabled
                      ? 'Continuous neural scoring active'
                      : 'Protection paused',
                  leading: const Icon(LucideIcons.shieldCheck, size: 18, color: ShadTokens.foreground),
                  value: s.protectionEnabled,
                  onChanged: (v) => s.setProtection(v),
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: ShadTokens.space2),
                  child: Divider(color: ShadTokens.border, height: 1),
                ),
                ShadToggleTile(
                  title: 'In-Call Floating Warning Banner',
                  subtitle: 'Head-up overlay displayed over active carrier calls',
                  leading: const Icon(LucideIcons.layers, size: 18, color: ShadTokens.foreground),
                  value: s.overlayEnabled,
                  onChanged: (v) => s.setOverlay(v),
                ),
                const Padding(
                  padding: EdgeInsets.symmetric(horizontal: ShadTokens.space2),
                  child: Divider(color: ShadTokens.border, height: 1),
                ),
                ShadToggleTile(
                  title: 'Auditory & Haptic Alerts',
                  subtitle: 'Immediate audio tone and vibration on threat detection',
                  leading: const Icon(LucideIcons.volume2, size: 18, color: ShadTokens.foreground),
                  value: s.soundEnabled,
                  onChanged: (v) => s.setSound(v),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // ── Sensitivity & Alert Threshold ──
          _sectionHeader('Threat Detection Sensitivity'),
          ShadGlassCard(
            padding: const EdgeInsets.all(ShadTokens.space4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'Alert Trigger Threshold',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.w700, color: ShadTokens.cardFg),
                    ),
                    ShadBadge(
                      label: '${(s.sensitivity * 100).toInt()}% RISK',
                      variant: ShadBadgeVariant.defaultBadge,
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                SliderTheme(
                  data: SliderThemeData(
                    activeTrackColor: ShadTokens.primary,
                    inactiveTrackColor: ShadTokens.secondary,
                    thumbColor: ShadTokens.primary,
                    overlayColor: ShadTokens.primary.withValues(alpha: 0.10),
                    trackHeight: 4,
                  ),
                  child: Slider(
                    value: s.sensitivity,
                    min: 0.40,
                    max: 0.90,
                    divisions: 10,
                    label: '${(s.sensitivity * 100).toInt()}%',
                    onChanged: (v) => s.setSensitivity(v),
                  ),
                ),
                const SizedBox(height: 2),
                const Text(
                  'Lower values detect subtle distortions sooner. Higher values require stronger model certainty before raising an alarm.',
                  style: TextStyle(fontSize: 11, color: ShadTokens.muted, height: 1.35),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // ── Operating System Permissions ──
          _sectionHeader('Operating System Permissions'),
          PermissionCard(
            icon: LucideIcons.mic,
            title: 'Microphone & Phone Access',
            subtitle: 'Required to read audio stream and detect call state transitions',
            granted: _hasPhonePerms,
            actionLabel: _hasPhonePerms ? 'Granted' : 'Grant',
            onAction: () async {
              await PermissionHelper.requestPhonePermissions();
              _refresh();
            },
          ),
          const SizedBox(height: 8),
          PermissionCard(
            icon: LucideIcons.externalLink,
            title: 'Display Over Other Apps',
            subtitle: 'Required for floating in-call real-time risk overlay',
            granted: _hasOverlay,
            actionLabel: _hasOverlay ? 'Granted' : 'Grant',
            onAction: () async {
              await CallService().requestOverlayPermission();
              _refresh();
            },
          ),
          const SizedBox(height: 8),
          PermissionCard(
            icon: LucideIcons.phoneCall,
            title: 'Default Telecom Dialer Role',
            subtitle: _isDefaultDialer
                ? 'Vaani is bound to Android Telecom subsystem as Default Dialer'
                : 'Intercept real telephony calls directly via InCallService',
            granted: _isDefaultDialer,
            actionLabel: _isDefaultDialer ? 'Active' : 'Grant',
            onAction: () async {
              await CallService().setAsDefaultDialer();
              _refresh();
            },
          ),
          const SizedBox(height: 24),

          // ── Neural Engine Architecture ──
          _sectionHeader('Neural Engine & Acoustic Pipeline'),
          ShadGlassCard(
            padding: const EdgeInsets.all(ShadTokens.space4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _infoRow('Neural Framework', 'ONNX Runtime INT8 Engine'),
                _infoRow('Acoustic Features', '60-Band LFCC + Prosody Jitter/Shimmer'),
                _infoRow('Inference Latency', '< 80 ms per 1.0s hop window'),
                _infoRow('Audio Sampling', '16,000 Hz Mono 16-bit PCM'),
                _infoRow('Hardware Accelerator', 'ARMv8-A / NEON Vectorized'),
                _infoRow('Model Checksum', 'SHA256 • 881 KB INT8 Quantized'),
              ],
            ),
          ),
          const SizedBox(height: 24),

          // ── About & Statutory DPDP Act Compliance ──
          _sectionHeader('About Vaani & DPDP Act 2023 Compliance'),
          ShadGlassCard(
            padding: const EdgeInsets.all(ShadTokens.space4),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                _infoRow('Version', '1.0.0 (Production Release)'),
                _infoRow('Application ID', 'com.voiceguard.voice_guard'),
                _infoRow('National Initiative', AppConstants.org),
                _infoRow('Problem Category', 'AI Voice Clone & Deepfake Defense'),
                _infoRow('Architecture', 'Telecom InCallService + WebRTC P2P'),
                const Padding(
                  padding: EdgeInsets.symmetric(vertical: 12),
                  child: Divider(color: ShadTokens.border, height: 1),
                ),
                Row(
                  children: const [
                    Icon(LucideIcons.shieldCheck, size: 16, color: ShadTokens.foreground),
                    SizedBox(width: 8),
                    Text(
                      'DPDP Act 2023 Privacy by Design',
                      style: TextStyle(
                        fontSize: 12,
                        fontWeight: FontWeight.w700,
                        color: ShadTokens.foreground,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 10),
                _complianceItem(
                  'Volatile In-Memory Processing',
                  'Audio streams reside strictly in transient RAM buffers during live neural inference and are immediately recycled. Zero raw audio recordings are stored to disk without explicit user authorization.',
                ),
                const SizedBox(height: 8),
                _complianceItem(
                  'Zero Cloud Transmission',
                  '100% on-device AI inference via ONNX Runtime INT8. No raw voice bytes, metadata, or telemetry are transmitted to remote servers.',
                ),
                const SizedBox(height: 8),
                _complianceItem(
                  'DPDP Act Section 6 & 8 Compliance',
                  'Full adherence to India Digital Personal Data Protection Act: purpose limitation, data minimization, and anonymized cryptographic event logging.',
                ),
                const SizedBox(height: 8),
                _complianceItem(
                  'Right to Erasure & User Sovereignty',
                  'You retain complete control over audit records. You can permanently purge all stored forensic event traces at any time from the Audit Logs screen.',
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
        ],
      ),
    );
  }

  Widget _complianceItem(String title, String description) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w700,
            color: ShadTokens.foreground,
          ),
        ),
        const SizedBox(height: 2),
        Text(
          description,
          style: const TextStyle(
            fontSize: 10.5,
            color: ShadTokens.muted,
            height: 1.35,
          ),
        ),
      ],
    );
  }

  Widget _sectionHeader(String title) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, bottom: 8),
      child: Text(
        title.toUpperCase(),
        style: const TextStyle(
          fontSize: 11,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.6,
          color: ShadTokens.muted,
        ),
      ),
    );
  }

  Widget _infoRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 11, color: ShadTokens.muted)),
          Text(value, style: const TextStyle(fontSize: 11, fontWeight: FontWeight.w700, color: ShadTokens.foreground)),
        ],
      ),
    );
  }
}