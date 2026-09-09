import 'package:permission_handler/permission_handler.dart';

class PermissionHelper {
  static Future<bool> requestPhonePermissions() async {
    final statuses = await [
      Permission.phone,
      Permission.microphone,
      // Needed to enumerate/prefer a Bluetooth SCO input device and start
      // SCO routing — see AudioCaptureManager.preferExternalInputDevice().
      // Not fatal if denied (older Android, or user declines): that path
      // just falls back to whatever device is already default-routed.
      Permission.bluetoothConnect,
    ].request();
    // Bluetooth permission is best-effort only; phone+mic are the hard
    // requirement for the app to function at all.
    return statuses[Permission.phone]!.isGranted &&
        statuses[Permission.microphone]!.isGranted;
  }

  static Future<bool> hasPhonePermissions() async {
    final phone = await Permission.phone.status;
    final mic = await Permission.microphone.status;
    return phone.isGranted && mic.isGranted;
  }

  static Future<bool> hasNotificationPermission() async {
    final s = await Permission.notification.status;
    return s.isGranted;
  }

  static Future<void> requestNotificationPermission() async {
    await Permission.notification.request();
  }

  static Future<void> openSettings() async {
    await openAppSettings();
  }
}
