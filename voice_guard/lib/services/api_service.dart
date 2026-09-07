import 'dart:convert';
import 'package:http/http.dart' as http;

/// Thin client for the optional integration-layer backend (Phase 8).
/// The on-device path is primary; this demonstrates the enterprise SDK story.
class ApiService {
  final String baseUrl;
  final String apiKey;
  ApiService({this.baseUrl = 'http://10.0.2.2:8001', this.apiKey = 'vg_demo_key'});

  Future<Map<String, dynamic>?> analyzeChunk({
    required List<double> lfcc,
    required List<double> prosody,
    String? callerId,
    String? locale,
  }) async {
    try {
      final r = await http.post(
        Uri.parse('$baseUrl/v1/analyze-chunk'),
        headers: {'Content-Type': 'application/json', 'X-API-Key': apiKey},
        body: jsonEncode({
          'lfcc': lfcc,
          'prosody': {'pauseRatio': prosody.isNotEmpty ? prosody[0] : 0, 'energyVar': prosody.length > 1 ? prosody[1] : 0, 'zcrVar': prosody.length > 2 ? prosody[2] : 0},
          'metadata': {'callerId': callerId, 'locale': locale ?? 'en-IN', 'ts': DateTime.now().toIso8601String()},
        }),
      ).timeout(const Duration(seconds: 3));
      if (r.statusCode == 200) return jsonDecode(r.body) as Map<String, dynamic>;
    } catch (_) {}
    return null;
  }

  Future<bool> sendAlert({required String callerId, required double riskScore, required String verdict}) async {
    try {
      final r = await http.post(
        Uri.parse('$baseUrl/v1/alert'),
        headers: {'Content-Type': 'application/json', 'X-API-Key': apiKey},
        body: jsonEncode({'callerId': callerId, 'riskScore': riskScore, 'verdict': verdict, 'ts': DateTime.now().toIso8601String()}),
      ).timeout(const Duration(seconds: 3));
      return r.statusCode == 200 || r.statusCode == 202;
    } catch (_) {
      return false;
    }
  }
}
