import 'package:permission_handler/permission_handler.dart';

class PermissionHelper {
  static Future<bool> requestPhonePermissions() async {
    final statuses = await [
      Permission.phone,
      Permission.microphone,
    ].request();
    return statuses.values.every((s) => s.isGranted);
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
