import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

class SettingsProvider extends ChangeNotifier {
  bool _protectionEnabled = true;
  bool _overlayEnabled = true;
  bool _soundEnabled = true;
  double _sensitivity = 0.60; // alert threshold, matches engine_mock.py's ALERT_THRESHOLD
  bool _onboardingDone = false;

  bool get protectionEnabled => _protectionEnabled;
  bool get overlayEnabled => _overlayEnabled;
  bool get soundEnabled => _soundEnabled;
  double get sensitivity => _sensitivity;
  bool get onboardingDone => _onboardingDone;

  Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    _protectionEnabled = p.getBool('protectionEnabled') ?? true;
    _overlayEnabled = p.getBool('overlayEnabled') ?? true;
    _soundEnabled = p.getBool('soundEnabled') ?? true;
    _sensitivity = p.getDouble('sensitivity') ?? 0.60;
    _onboardingDone = p.getBool('onboardingDone') ?? false;
    notifyListeners();
  }

  Future<void> setProtection(bool v) async {
    _protectionEnabled = v;
    (await SharedPreferences.getInstance()).setBool('protectionEnabled', v);
    notifyListeners();
  }

  Future<void> setOverlay(bool v) async {
    _overlayEnabled = v;
    (await SharedPreferences.getInstance()).setBool('overlayEnabled', v);
    notifyListeners();
  }

  Future<void> setSound(bool v) async {
    _soundEnabled = v;
    (await SharedPreferences.getInstance()).setBool('soundEnabled', v);
    notifyListeners();
  }

  Future<void> setSensitivity(double v) async {
    _sensitivity = v;
    (await SharedPreferences.getInstance()).setDouble('sensitivity', v);
    notifyListeners();
  }

  Future<void> setOnboardingDone(bool v) async {
    _onboardingDone = v;
    (await SharedPreferences.getInstance()).setBool('onboardingDone', v);
    notifyListeners();
  }
}
