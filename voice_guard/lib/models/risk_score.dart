import 'package:flutter/material.dart';
import '../utils/constants.dart';

enum Verdict { verified, suspicious, detected }

class RiskScore {
  final double score; // 0..1 (fakeProb)
  final Verdict verdict;
  final DateTime timestamp;
  final List<double>? prosody; // [pauseRatio, energyVar, zcrVar]

  RiskScore({
    required this.score,
    required this.timestamp,
    this.prosody,
  }) : verdict = _verdict(score);

  static Verdict _verdict(double s) {
    if (s < 0.30) return Verdict.verified;
    if (s < 0.70) return Verdict.suspicious;
    return Verdict.detected;
  }

  String get label {
    switch (verdict) {
      case Verdict.verified: return 'VERIFIED HUMAN';
      case Verdict.suspicious: return 'SUSPICIOUS';
      case Verdict.detected: return 'AI DETECTED';
    }
  }

  Color get color {
    switch (verdict) {
      case Verdict.verified: return AppColors.verified;
      case Verdict.suspicious: return AppColors.suspicious;
      case Verdict.detected: return AppColors.detected;
    }
  }

  Color get bg {
    switch (verdict) {
      case Verdict.verified: return AppColors.verifiedBg;
      case Verdict.suspicious: return AppColors.suspiciousBg;
      case Verdict.detected: return AppColors.detectedBg;
    }
  }

  int get percent => (score * 100).round();
}
