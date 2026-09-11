import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

/// Comprehensive design tokens inspired by shadcn/ui (Zinc palette).
/// Provides high contrast, crisp borders, subtle shadows, and disciplined spacing.
class ShadTokens {
  ShadTokens._();

  // ── Zinc / Neutral Palette ───────────────────────────────────────────
  static const Color white = Color(0xFFFFFFFF);
  static const Color black = Color(0xFF000000);

  // Backgrounds & Surfaces
  static const Color background = Color(0xFFFAFAFA);
  static const Color surface = Color(0xFFFFFFFF);
  static const Color foreground = Color(0xFF09090B);

  // Cards & Popovers
  static const Color card = Color(0xFFFFFFFF);
  static const Color cardFg = Color(0xFF09090B);
  static const Color popover = Color(0xFFFFFFFF);
  static const Color popoverFg = Color(0xFF09090B);

  // Primary (Zinc-900) & Secondary (Zinc-100)
  static const Color primary = Color(0xFF18181B);
  static const Color primaryFg = Color(0xFFFAFAFA);
  static const Color secondary = Color(0xFFF4F4F5);
  static const Color secondaryFg = Color(0xFF18181B);

  // Muted & Accents
  static const Color muted = Color(0xFF71717A);
  static const Color mutedFg = Color(0xFFA1A1AA);
  static const Color mutedBg = Color(0xFFF4F4F5);
  static const Color accent = Color(0xFFF4F4F5);
  static const Color accentFg = Color(0xFF18181B);

  // Borders & Inputs
  static const Color border = Color(0xFFE4E4E7);
  static const Color input = Color(0xFFE4E4E7);
  static const Color ring = Color(0xFF18181B);

  // Destructive (Red-600)
  static const Color destructive = Color(0xFFDC2626);
  static const Color destructiveFg = Color(0xFFFFFFFF);
  static const Color destructiveBg = Color(0xFFFEF2F2);
  static const Color destructiveBorder = Color(0xFFFECACA);

  // ── VoiceGuard Semantic Accents (DPDP & Risk Detection) ─────────────
  // Verified Human (Green-600)
  static const Color verified = Color(0xFF16A34A);
  static const Color verifiedFg = Color(0xFFFFFFFF);
  static const Color verifiedBg = Color(0xFFF0FDF4);
  static const Color verifiedBorder = Color(0xFFBBF7D0);

  // Suspicious (Amber-600)
  static const Color suspicious = Color(0xFFD97706);
  static const Color suspiciousFg = Color(0xFFFFFFFF);
  static const Color suspiciousBg = Color(0xFFFFFBEB);
  static const Color suspiciousBorder = Color(0xFFFDE68A);

  // AI Detected (Red-600)
  static const Color detected = Color(0xFFDC2626);
  static const Color detectedFg = Color(0xFFFFFFFF);
  static const Color detectedBg = Color(0xFFFEF2F2);
  static const Color detectedBorder = Color(0xFFFECACA);

  // Info (Blue-600)
  static const Color info = Color(0xFF2563EB);
  static const Color infoBg = Color(0xFFEFF6FF);
  static const Color infoBorder = Color(0xFFBFDBFE);

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