import 'package:flutter/material.dart';

class AppColors {
  static const primary = Color(0xFF1A73E8);
  static const primaryDark = Color(0xFF0D47A1);
  static const verified = Color(0xFF2E7D32);
  static const verifiedBg = Color(0xFFE8F5E9);
  static const suspicious = Color(0xFFF9A825);
  static const suspiciousBg = Color(0xFFFFF8E1);
  static const detected = Color(0xFFC62828);
  static const detectedBg = Color(0xFFFFEBEE);
  static const surface = Color(0xFFF8F9FA);
  static const cardBg = Colors.white;
  static const textPrimary = Color(0xFF212121);
  static const textSecondary = Color(0xFF757575);
}

class AppConstants {
  static const appName = 'Vaani';
  static const tagline = 'Real-Time AI Voice Cloning Detector';
  static const org = 'Smart India Hackathon 2025-26';
  static const version = '0.1.0 (POC)';

  // Audio
  static const sampleRate = 16000;
  static const chunkDurationSec = 3;
  static const chunkSamples = sampleRate * chunkDurationSec; // 48000
  static const hopDurationSec = 1;
  static const nLfcc = 60;
  static const fftSize = 1024;
  static const hopLength = 256;
  static const nFilterBanks = 513;

  // Thresholds (configurable via Settings)
  static const defaultThresholdLow = 0.30;
  static const defaultThresholdHigh = 0.70;
  static const defaultSensitivity = 0.70;

  // Risk bands
  static String verdictFor(double score) {
    if (score < 0.30) return 'VERIFIED HUMAN';
    if (score < 0.70) return 'SUSPICIOUS';
    return 'AI DETECTED';
  }

  static Color colorFor(double score) {
    if (score < 0.30) return AppColors.verified;
    if (score < 0.70) return AppColors.suspicious;
    return AppColors.detected;
  }

  static Color bgFor(double score) {
    if (score < 0.30) return AppColors.verifiedBg;
    if (score < 0.70) return AppColors.suspiciousBg;
    return AppColors.detectedBg;
  }
}
