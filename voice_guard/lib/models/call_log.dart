import 'risk_score.dart';

class CallLog {
  final String id;
  final DateTime timestamp;
  final String number;
  final double riskScore;
  final Verdict verdict;
  final Duration duration;

  CallLog({
    required this.id,
    required this.timestamp,
    required this.number,
    required this.riskScore,
    required this.verdict,
    this.duration = Duration.zero,
  });

  Map<String, dynamic> toJson() => {
        'id': id,
        'timestamp': timestamp.toIso8601String(),
        'number': number,
        'riskScore': riskScore,
        'verdict': verdict.name,
        'durationMs': duration.inMilliseconds,
      };

  factory CallLog.fromJson(Map<String, dynamic> j) => CallLog(
        id: j['id'] as String,
        timestamp: DateTime.parse(j['timestamp'] as String),
        number: j['number'] as String,
        riskScore: (j['riskScore'] as num).toDouble(),
        verdict: Verdict.values.byName(j['verdict'] as String),
        duration: Duration(milliseconds: (j['durationMs'] as num?)?.toInt() ?? 0),
      );
}
