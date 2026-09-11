import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Comprehensive design tokens inspired by shadcn/ui (Zinc palette).
/// Provides high contrast, crisp borders, subtle shadows, and disciplined spacing.
class ShadTokens {
  ShadTokens._();

  // ── Zinc / Neutral Palette ───────────────────────────────────────────
  static const Color white = Color(0xFFFFFFFF);
  static const Color black = Color(0xFF000000);

  // Backgrounds & Surfaces (Figma Obsidian / Dark Zinc)
  static const Color background = Color(0xFF131315);
  static const Color surface = Color(0xFF18181B);
  static const Color surfaceContainer = Color(0xFF1C1C1E);
  static const Color foreground = Color(0xFFFFFFFF);

  // Cards & Popovers
  static const Color card = Color(0xFF18181B);
  static const Color cardFg = Color(0xFFFFFFFF);
  static const Color popover = Color(0xFF18181B);
  static const Color popoverFg = Color(0xFFFFFFFF);

  // Primary (White) & Secondary (Zinc-800)
  static const Color primary = Color(0xFFFFFFFF);
  static const Color primaryFg = Color(0xFF000000);
  static const Color secondary = Color(0xFF27272A);
  static const Color secondaryFg = Color(0xFFFFFFFF);

  // Muted & Accents (Figma Slate/Zinc)
  static const Color muted = Color(0xFF8E9192);
  static const Color mutedFg = Color(0xFF71717A);
  static const Color mutedBg = Color(0xFF1C1C1E);
  static const Color accent = Color(0xFF27272A);
  static const Color accentFg = Color(0xFFFFFFFF);

  // Borders & Inputs
  static const Color border = Color(0xFF27272A);
  static const Color input = Color(0xFF27272A);
  static const Color ring = Color(0xFF21D4B2);

  // Destructive (Red-500 / Coral)
  static const Color destructive = Color(0xFFFF6268);
  static const Color destructiveFg = Color(0xFFFFFFFF);
  static const Color destructiveBg = Color(0xFF2A1517);
  static const Color destructiveBorder = Color(0xFF521B21);

  // ── VoiceGuard Semantic Accents (Figma Palette) ─────────────────────
  // Authentic Voice (Cyan/Teal - #21D4B2)
  static const Color teal = Color(0xFF21D4B2);
  static const Color verified = Color(0xFF21D4B2);
  static const Color verifiedFg = Color(0xFF000000);
  static const Color verifiedBg = Color(0xFF0D2824);
  static const Color verifiedBorder = Color(0xFF154F46);

  // Coral / AI Clones (#FF6268)
  static const Color coral = Color(0xFFFF6268);
  static const Color detected = Color(0xFFFF6268);
  static const Color detectedFg = Color(0xFFFFFFFF);
  static const Color detectedBg = Color(0xFF2C1618);
  static const Color detectedBorder = Color(0xFF5A1E24);

  // Suspicious (Amber-500)
  static const Color suspicious = Color(0xFFF59E0B);
  static const Color suspiciousFg = Color(0xFF000000);
  static const Color suspiciousBg = Color(0xFF2B200E);
  static const Color suspiciousBorder = Color(0xFF523C13);

  // Call Button Active Green (#00C274 / #10B981)
  static const Color callGreen = Color(0xFF00C274);

  // Info (Blue-500)
  static const Color info = Color(0xFF3B82F6);
  static const Color infoBg = Color(0xFF132338);
  static const Color infoBorder = Color(0xFF1E3A5F);

  // ── Radii ────────────────────────────────────────────────────────────
  static const double radiusSm = 6.0;
  static const double radiusMd = 10.0;
  static const double radiusLg = 14.0;
  static const double radiusXl = 18.0;
  static const double radius2xl = 24.0;
  static const double radiusFull = 999.0;

  // ── Elevation & Shadows ──────────────────────────────────────────────
  static List<BoxShadow> shadowSm = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.04),
      blurRadius: 3,
      offset: const Offset(0, 1),
    ),
  ];

  static List<BoxShadow> shadowMd = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.06),
      blurRadius: 10,
      offset: const Offset(0, 3),
    ),
  ];

  static List<BoxShadow> shadowLg = [
    BoxShadow(
      color: Colors.black.withValues(alpha: 0.08),
      blurRadius: 20,
      offset: const Offset(0, 6),
    ),
  ];

  // ── Spacing Scale (8pt grid) ────────────────────────────────────────
  static const double space1 = 4.0;
  static const double space2 = 8.0;
  static const double space3 = 12.0;
  static const double space4 = 16.0;
  static const double space5 = 20.0;
  static const double space6 = 24.0;
  static const double space8 = 32.0;
  static const double space10 = 40.0;

  // ── Durations ────────────────────────────────────────────────────────
  static const Duration fast = Duration(milliseconds: 150);
  static const Duration normal = Duration(milliseconds: 250);
  static const Duration slow = Duration(milliseconds: 400);

  // ── Text Styles (Apple SF Pro / Inter typography) ──────────────────────
  static TextStyle get h1 => GoogleFonts.inter(
        fontSize: 24,
        fontWeight: FontWeight.w800,
        letterSpacing: -0.6,
        color: foreground,
        height: 1.2,
      );

  static TextStyle get h2 => GoogleFonts.inter(
        fontSize: 18,
        fontWeight: FontWeight.w700,
        letterSpacing: -0.4,
        color: foreground,
        height: 1.25,
      );

  static TextStyle get h3 => GoogleFonts.inter(
        fontSize: 15,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.2,
        color: foreground,
        height: 1.3,
      );

  static TextStyle get body => GoogleFonts.inter(
        fontSize: 13,
        fontWeight: FontWeight.w400,
        color: foreground,
        height: 1.4,
      );

  static TextStyle get bodyMuted => GoogleFonts.inter(
        fontSize: 12,
        fontWeight: FontWeight.w400,
        color: muted,
        height: 1.4,
      );

  static TextStyle get caption => GoogleFonts.inter(
        fontSize: 11,
        fontWeight: FontWeight.w500,
        color: muted,
        height: 1.3,
      );

  static TextStyle get mono => GoogleFonts.jetBrainsMono(
        fontSize: 12,
        fontFeatures: const [FontFeature.tabularFigures()],
        fontWeight: FontWeight.w600,
        color: foreground,
      );
}

/// Type alias so that both ShadTheme and ShadTokens refer to the same tokens.
typedef ShadTheme = ShadTokens;