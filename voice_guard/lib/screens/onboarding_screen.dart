import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/settings_provider.dart';
import '../utils/constants.dart';
import '../utils/permissions.dart';
import '../services/call_service.dart';

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
      backgroundColor: Colors.white,
      body: SafeArea(
        child: Column(children: [
          Expanded(
            child: PageView(
              controller: _ctrl,
              onPageChanged: (i) => setState(() => _page = i),
              children: [
                _slide(icon: Icons.shield_rounded, color: AppColors.primary, title: 'Stop Voice Cloning\nFraud in Real Time',
                    body: 'Generative AI can clone a voice from seconds of audio. ${AppConstants.appName} detects synthetic speech during live calls and warns you before you act.'),
                _slide(icon: Icons.graphic_eq, color: AppColors.verified, title: 'How it works',
                    body: '16 kHz audio → 60 LFCC features (linear filterbank preserves synthesis artifacts) + prosody (pause/pitch variance) → on-device TFLite model → risk score 0–100% with smoothing. Raw PCM never leaves RAM.'),
                _slide(icon: Icons.lock_rounded, color: AppColors.suspicious, title: 'Privacy-first & compliant',
                    body: 'No raw audio is stored. Only {timestamp, caller, risk, verdict} is logged. Inference is on-device; the /v1/analyze-chunk API accepts features only. DPDP Act aligned.'),
                _permissionsSlide(),
              ],
            ),
          ),
          Row(mainAxisAlignment: MainAxisAlignment.center, children: List.generate(4, (i) => Container(
            width: 8, height: 8, margin: const EdgeInsets.all(4),
            decoration: BoxDecoration(color: i == _page ? AppColors.primary : Colors.black12, shape: BoxShape.circle),
          ))),
          Padding(
            padding: const EdgeInsets.all(16),
            child: SizedBox(width: double.infinity, child: FilledButton(
              onPressed: () async {
                if (_page < 3) { _ctrl.nextPage(duration: const Duration(milliseconds: 300), curve: Curves.easeOut); return; }
                final settingsProvider = context.read<SettingsProvider>();
                await settingsProvider.setOnboardingDone(true);
                if (!context.mounted) return;
                Navigator.of(context).pop();
              },
              style: FilledButton.styleFrom(backgroundColor: AppColors.primary, padding: const EdgeInsets.symmetric(vertical: 14)),
              child: Text(_page < 3 ? 'Next' : 'Get Started', style: const TextStyle(fontWeight: FontWeight.w800)),
            )),
          ),
        ]),
      ),
    );
  }

  Widget _slide({required IconData icon, required Color color, required String title, required String body}) => Padding(
    padding: const EdgeInsets.all(24),
    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(color: color.withValues(alpha: 0.12), shape: BoxShape.circle), child: Icon(icon, size: 48, color: color)),
      const SizedBox(height: 20),
      Text(title, textAlign: TextAlign.center, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.w800)),
      const SizedBox(height: 12),
      Text(body, textAlign: TextAlign.center, style: TextStyle(fontSize: 13, color: Colors.black.withValues(alpha: 0.65), height: 1.5)),
    ]),
  );

  Widget _permissionsSlide() => Padding(
    padding: const EdgeInsets.all(24),
    child: Column(mainAxisAlignment: MainAxisAlignment.center, children: [
      Container(padding: const EdgeInsets.all(20), decoration: BoxDecoration(color: AppColors.detectedBg, shape: BoxShape.circle), child: const Icon(Icons.settings, size: 48, color: AppColors.detected)),
      const SizedBox(height: 16),
      const Text('Permissions needed', style: TextStyle(fontSize: 18, fontWeight: FontWeight.w800)),
      const SizedBox(height: 12),
      Text('Phone + Microphone to detect calls, overlay permission for in-call warnings, and Default Dialer role for InCallService.\n\nAndroid 10+ blocks VOICE_CALL capture — POC uses MIC + speakerphone (stated in demo); production uses telecom-side media forking.',
          textAlign: TextAlign.center, style: TextStyle(fontSize: 11, color: Colors.black.withValues(alpha: 0.65), height: 1.5)),
      const SizedBox(height: 16),
      OutlinedButton.icon(onPressed: () async { await PermissionHelper.requestPhonePermissions(); await CallService().requestOverlayPermission(); }, icon: const Icon(Icons.check_circle_outline), label: const Text('Grant permissions')),
    ]),
  );
}
