import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import '../providers/settings_provider.dart';
import '../utils/permissions.dart';
import '../services/call_service.dart';
import '../widgets/shad_card.dart';
import '../widgets/shad_badge.dart';
import '../widgets/shad_button.dart';
import '../design/tokens.dart';

class OnboardingScreen extends StatefulWidget {
  const OnboardingScreen({super.key});
  @override
  State<OnboardingScreen> createState() => _OnboardingScreenState();
}

class _OnboardingScreenState extends State<OnboardingScreen> {
  int _page = 0;
  final _ctrl = PageController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: ShadTokens.background,
      body: SafeArea(
        child: Column(
          children: [
            // Top Bar
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  Row(
                    children: [
                      Container(
                        width: 28,
                        height: 28,
                        decoration: BoxDecoration(
                          color: ShadTokens.primary,
                          borderRadius: BorderRadius.circular(ShadTokens.radiusSm),
                        ),
                        child: const Icon(LucideIcons.activity, color: Colors.white, size: 16),
                      ),
                      const SizedBox(width: 8),
                      const Text(
                        'VAANI ONBOARDING',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.w800,
                          letterSpacing: 0.5,
                          color: ShadTokens.foreground,
                        ),
                      ),
                    ],
                  ),
                  ShadBadge(
                    label: 'STEP ${_page + 1} OF 4',
                    variant: ShadBadgeVariant.outline,
                  ),
                ],
              ),
            ),

            // Carousel
            Expanded(
              child: PageView(
                controller: _ctrl,
                onPageChanged: (i) => setState(() => _page = i),
                children: [
                  _buildSlide(
                    icon: LucideIcons.shieldCheck,
                    color: ShadTokens.primary,
                    tag: 'REAL-TIME PROTECTION',
                    title: 'Stop Voice Cloning\nFraud In Real Time',
                    body: 'Modern Generative AI models can clone any target voice from just 3 seconds of reference audio. Vaani detects synthetic acoustic artifacts during live incoming calls and warns you instantly.',
                  ),
                  _buildSlide(
                    icon: LucideIcons.cpu,
                    color: ShadTokens.verified,
                    tag: 'NEURAL PIPELINE',
                    title: 'On-Device ONNX\nNeural Inference',
                    body: 'Incoming 16 kHz audio is processed through 60 LFCC filterbanks to capture synthesis jitter and vocal tract dynamics. Evaluated in under 80 ms entirely on-device without cloud round-trips.',
                  ),
                  _buildSlide(
                    icon: LucideIcons.lock,
                    color: ShadTokens.info,
                    tag: 'PRIVACY BY DESIGN',
                    title: 'Zero Storage &\nDPDP Act Aligned',
                    body: 'Raw audio PCM is processed in volatile RAM and immediately discarded. Never stored, never uploaded. Only anonymized risk scores and timestamps remain on your device.',
                  ),
                  _buildPermissionsSlide(),
                ],
              ),
            ),

            // Indicator Dots
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                4,
                (i) => AnimatedContainer(
                  duration: ShadTokens.fast,
                  width: i == _page ? 24 : 7,
                  height: 7,
                  margin: const EdgeInsets.symmetric(horizontal: 3),
                  decoration: BoxDecoration(
                    color: i == _page ? ShadTokens.primary : ShadTokens.border,
                    borderRadius: BorderRadius.circular(ShadTokens.radiusFull),
                  ),
                ),
              ),
            ),
            const SizedBox(height: 16),

            // Bottom Buttons
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
              child: Row(
                children: [
                  if (_page > 0) ...[
                    ShadButton(
                      onTap: () => _ctrl.previousPage(
                        duration: ShadTokens.normal,
                        curve: Curves.easeOut,
                      ),
                      variant: ShadButtonVariant.outline,
                      size: ShadButtonSize.lg,
                      text: 'Back',
                    ),
                    const SizedBox(width: 12),
                  ],
                  Expanded(
                    child: ShadButton(
                      onTap: () async {
                        if (_page < 3) {
                          _ctrl.nextPage(
                            duration: ShadTokens.normal,
                            curve: Curves.easeOut,
                          );
                          return;
                        }
                        final settingsProvider = context.read<SettingsProvider>();
                        await settingsProvider.setOnboardingDone(true);
                        if (!context.mounted) return;
                        Navigator.of(context).pop();
                      },
                      variant: ShadButtonVariant.primary,
                      size: ShadButtonSize.lg,
                      fullWidth: true,
                      text: _page < 3 ? 'Continue' : 'Enter Vaani Security',
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildSlide({
    required IconData icon,
    required Color color,
    required String tag,
    required String title,
    required String body,
  }) {
    return Padding(
      padding: const EdgeInsets.all(24),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 80,
            height: 80,
            decoration: BoxDecoration(
              color: color.withValues(alpha: 0.10),
              borderRadius: BorderRadius.circular(ShadTokens.radius2xl),
              border: Border.all(color: color.withValues(alpha: 0.25)),
            ),
            child: Icon(icon, size: 40, color: color),
          ),
          const SizedBox(height: 24),
          ShadBadge(
            label: tag,
            variant: ShadBadgeVariant.outline,
          ),
          const SizedBox(height: 12),
          Text(
            title,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.6,
              color: ShadTokens.foreground,
              height: 1.2,
            ),
          ),
          const SizedBox(height: 14),
          Text(
            body,
            textAlign: TextAlign.center,
            style: const TextStyle(
              fontSize: 13,
              color: ShadTokens.muted,
              height: 1.5,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPermissionsSlide() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            width: 72,
            height: 72,
            decoration: BoxDecoration(
              color: ShadTokens.secondary,
              borderRadius: BorderRadius.circular(ShadTokens.radius2xl),
              border: Border.all(color: ShadTokens.border),
            ),
            child: const Icon(LucideIcons.shieldAlert, size: 36, color: ShadTokens.primary),
          ),
          const SizedBox(height: 20),
          const ShadBadge(
            label: 'AUTHORIZATION REQUIRED',
            variant: ShadBadgeVariant.secondary,
          ),
          const SizedBox(height: 10),
          const Text(
            'Grant Defense Roles',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 22,
              fontWeight: FontWeight.w800,
              letterSpacing: -0.5,
              color: ShadTokens.foreground,
            ),
          ),
          const SizedBox(height: 12),
          const Text(
            'Vaani requires Microphone and Telecom permissions to monitor acoustic speech signals, and Display-Over-Other-Apps to show real-time alerts.',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 12,
              color: ShadTokens.muted,
              height: 1.45,
            ),
          ),
          const SizedBox(height: 20),
          ShadCard(
            padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 10),
            child: Row(
              children: [
                const Icon(LucideIcons.checkCircle2, color: ShadTokens.verified, size: 18),
                const SizedBox(width: 10),
                const Expanded(
                  child: Text(
                    'Microphone, Phone & Overlay Permissions',
                    style: TextStyle(fontSize: 12, fontWeight: FontWeight.w600, color: ShadTokens.cardFg),
                  ),
                ),
                ShadButton(
                  onTap: () async {
                    await PermissionHelper.requestPhonePermissions();
                    await CallService().requestOverlayPermission();
                  },
                  variant: ShadButtonVariant.primary,
                  size: ShadButtonSize.sm,
                  text: 'Authorize',
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}