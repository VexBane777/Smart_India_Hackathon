// voice_guard/test/settings_provider_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:voice_guard/providers/settings_provider.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  setUp(() {
    SharedPreferences.setMockInitialValues({});
  });

  test('signalingHost/signalingPort default to the emulator loopback values', () async {
    final settings = SettingsProvider();
    await settings.load();
    expect(settings.signalingHost, '10.0.2.2');
    expect(settings.signalingPort, 8001);
  });

  test('setSignalingHost persists and updates the getter', () async {
    final settings = SettingsProvider();
    await settings.load();
    await settings.setSignalingHost('100.101.102.5');
    expect(settings.signalingHost, '100.101.102.5');

    final reloaded = SettingsProvider();
    await reloaded.load();
    expect(reloaded.signalingHost, '100.101.102.5');
  });

  test('setSignalingPort persists and updates the getter', () async {
    final settings = SettingsProvider();
    await settings.load();
    await settings.setSignalingPort(9000);
    expect(settings.signalingPort, 9000);

    final reloaded = SettingsProvider();
    await reloaded.load();
    expect(reloaded.signalingPort, 9000);
  });
}
