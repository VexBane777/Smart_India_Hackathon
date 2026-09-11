import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/settings_provider.dart';
import '../services/call_service.dart';
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

    return Scaffold(
      backgroundColor: const Color(0xFF131315),
      body: SafeArea(
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
          children: [
            // ── Settings Title (Figma Main-3.png) ──
            Text(
              'Settings',
              style: GoogleFonts.inter(
                fontSize: 28,
                fontWeight: FontWeight.w800,
                letterSpacing: -0.5,
                color: Colors.white,
              ),
            ),
            const SizedBox(height: 28),

            // ── Section 1: PROTECTION ──
            _sectionLabel('PROTECTION'),
            const SizedBox(height: 10),
            Container(
              decoration: BoxDecoration(
                color: const Color(0xFF18181B),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF27272A), width: 1),
              ),
              child: Column(
                children: [
                  _switchTile(
                    title: 'Master Protection',
                    value: s.protectionEnabled,
                    onChanged: (v) => s.setProtection(v),
                  ),
                  const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),
                  _switchTile(
                    title: 'In-Call Banner',
                    value: s.overlayEnabled,
                    onChanged: (v) => s.setOverlay(v),
                  ),
                  const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),
                  _switchTile(
                    title: 'Haptic Alerts',
                    value: s.soundEnabled,
                    onChanged: (v) => s.setSound(v),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 28),

            // ── Section 2: PERMISSIONS ──
            _sectionLabel('PERMISSIONS'),
            const SizedBox(height: 10),
            Container(
              decoration: BoxDecoration(
                color: const Color(0xFF18181B),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF27272A), width: 1),
              ),
              child: Column(
                children: [
                  _navigationTile(
                    title: 'Microphone & Phone',
                    status: _hasPhonePerms ? 'Allowed' : 'Grant',
                    onTap: () async {
                      await PermissionHelper.requestPhonePermissions();
                      _refresh();
                    },
                  ),
                  const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),
                  _navigationTile(
                    title: 'Display Over Other Apps',
                    status: _hasOverlay ? 'Allowed' : 'Grant',
                    onTap: () async {
                      await CallService().requestOverlayPermission();
                      _refresh();
                    },
                  ),
                  const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),
                  _navigationTile(
                    title: 'Default Phone App',
                    status: _isDefaultDialer ? 'Active' : 'Grant',
                    onTap: () async {
                      await CallService().setAsDefaultDialer();
                      _refresh();
                    },
                  ),
                ],
              ),
            ),
            const SizedBox(height: 28),

            // ── Section 3: ABOUT & STATUTORY DPDP ACT ──
            _sectionLabel('COMPLIANCE & SYSTEM'),
            const SizedBox(height: 10),
            Container(
              decoration: BoxDecoration(
                color: const Color(0xFF18181B),
                borderRadius: BorderRadius.circular(16),
                border: Border.all(color: const Color(0xFF27272A), width: 1),
              ),
              child: Column(
                children: [
                  _navigationTile(
                    title: 'DPDP Act 2023 Compliance',
                    status: 'Verified',
                    onTap: () => _showDpdpComplianceModal(context),
                  ),
                  const Divider(color: Color(0xFF27272A), height: 1, thickness: 1),
                  _navigationTile(
                    title: 'Neural Engine Architecture',
                    status: 'ONNX INT8',
                    onTap: () => _showArchitectureModal(context),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 36),

            // ── Reset to Defaults Button (Figma Main-3.png) ──
            GestureDetector(
              onTap: () {
                s.resetToDefaults();
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(
                    backgroundColor: Color(0xFF18181B),
                    content: Text(
                      'Settings restored to defense defaults',
                      style: TextStyle(color: Colors.white),
                    ),
                  ),
                );
              },
              child: Container(
                height: 52,
                width: double.infinity,
                alignment: Alignment.center,
                decoration: BoxDecoration(
                  color: const Color(0xFF18181B),
                  borderRadius: BorderRadius.circular(14),
                  border: Border.all(color: const Color(0xFF27272A), width: 1),
                ),
                child: Text(
                  'Reset to Defaults',
                  style: GoogleFonts.inter(
                    fontSize: 15,
                    fontWeight: FontWeight.w600,
                    color: const Color(0xFFFF8C82), // Soft coral/peach
                  ),
                ),
              ),
            ),
            const SizedBox(height: 24),
          ],
        ),
      ),
    );
  }

  Widget _sectionLabel(String text) {
    return Padding(
      padding: const EdgeInsets.only(left: 4),
      child: Text(
        text,
        style: GoogleFonts.inter(
          fontSize: 12,
          fontWeight: FontWeight.w700,
          letterSpacing: 0.8,
          color: const Color(0xFF71717A),
        ),
      ),
    );
  }

  Widget _switchTile({
    required String title,
    required bool value,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 12),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Text(
              title,
              style: GoogleFonts.inter(
                fontSize: 15,
                fontWeight: FontWeight.w500,
                color: Colors.white,
              ),
              overflow: TextOverflow.ellipsis,
            ),
          ),
          const SizedBox(width: 8),
          Switch(
            value: value,
            onChanged: onChanged,
            activeThumbColor: const Color(0xFF131315),
            activeTrackColor: Colors.white,
            inactiveThumbColor: const Color(0xFF71717A),
            inactiveTrackColor: const Color(0xFF27272A),
            trackOutlineColor: WidgetStateProperty.all(Colors.transparent),
          ),
        ],
      ),
    );
  }

  Widget _navigationTile({
    required String title,
    required String status,
    required VoidCallback onTap,
  }) {
    return InkWell(
      onTap: onTap,
      child: Padding(
        padding: const EdgeInsets.symmetric(horizontal: 18, vertical: 18),
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Expanded(
              child: Text(
                title,
                style: GoogleFonts.inter(
                  fontSize: 15,
                  fontWeight: FontWeight.w500,
                  color: Colors.white,
                ),
                overflow: TextOverflow.ellipsis,
              ),
            ),
            const SizedBox(width: 8),
            Row(
              children: [
                Text(
                  status,
                  style: GoogleFonts.inter(
                    fontSize: 14,
                    fontWeight: FontWeight.w400,
                    color: const Color(0xFF8E9192),
                  ),
                ),
                const SizedBox(width: 6),
                const Icon(
                  LucideIcons.chevronRight,
                  size: 16,
                  color: Color(0xFF8E9192),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _showDpdpComplianceModal(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF18181B),
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
                      'DPDP Act 2023 Compliance',
                      style: GoogleFonts.inter(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(LucideIcons.x, color: Color(0xFF8E9192), size: 20),
                      onPressed: () => Navigator.pop(ctx),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _complianceItem(
                  'Volatile In-Memory Processing',
                  'Audio streams reside strictly in transient RAM buffers during live neural inference and are immediately wiped. Zero raw audio is stored on disk without consent.',
                ),
                _complianceItem(
                  'On-Device Data Sovereignty',
                  '100% on-device AI inference with zero cloud telemetry. No biometric data leaves the user device, adhering to Section 6 of the Digital Personal Data Protection Act 2023.',
                ),
                _complianceItem(
                  'Purpose Limitation & Auditing',
                  'Telemetry is solely utilized for real-time synthetic voice detection and threat defense isolation.',
                ),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  height: 46,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
                    ),
                    onPressed: () => Navigator.pop(ctx),
                    child: Text(
                      'Done',
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

  Widget _complianceItem(String title, String desc) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: GoogleFonts.inter(
              fontSize: 13,
              fontWeight: FontWeight.w700,
              color: Colors.white,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            desc,
            style: GoogleFonts.inter(
              fontSize: 11,
              color: const Color(0xFF8E9192),
              height: 1.35,
            ),
          ),
        ],
      ),
    );
  }

  void _showArchitectureModal(BuildContext context) {
    showModalBottomSheet(
      context: context,
      backgroundColor: const Color(0xFF18181B),
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
                      'Neural Engine Specs',
                      style: GoogleFonts.inter(
                        fontSize: 18,
                        fontWeight: FontWeight.w800,
                        color: Colors.white,
                      ),
                    ),
                    IconButton(
                      icon: const Icon(LucideIcons.x, color: Color(0xFF8E9192), size: 20),
                      onPressed: () => Navigator.pop(ctx),
                    ),
                  ],
                ),
                const SizedBox(height: 16),
                _specRow('Neural Engine', 'ONNX Runtime INT8 Quantized'),
                _specRow('Acoustic Features', '60-Band LFCC + Jitter/Shimmer Prosody'),
                _specRow('Inference Latency', '< 80 ms per 1.0s window'),
                _specRow('Sampling Rate', '16,000 Hz Mono 16-bit PCM'),
                _specRow('Hardware Acceleration', 'ARMv8-A / NEON SIMD Vectorized'),
                const SizedBox(height: 16),
                SizedBox(
                  width: double.infinity,
                  height: 46,
                  child: ElevatedButton(
                    style: ElevatedButton.styleFrom(
                      backgroundColor: Colors.white,
                      foregroundColor: Colors.black,
                      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(999)),
                    ),
                    onPressed: () => Navigator.pop(ctx),
                    child: Text(
                      'Close',
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

  Widget _specRow(String key, String val) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(key, style: GoogleFonts.inter(fontSize: 12, color: const Color(0xFF8E9192))),
          Text(val, style: GoogleFonts.inter(fontSize: 12, fontWeight: FontWeight.w600, color: Colors.white)),
        ],
      ),
    );
  }
}
