import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'tokens.dart';

export 'tokens.dart';

/// Helper to configure Flutter's MaterialApp ThemeData with shadcn Zinc aesthetic.
class ShadThemeHelper {
  ShadThemeHelper._();

  static ThemeData get lightTheme {
    return ThemeData(
      useMaterial3: true,
      brightness: Brightness.light,
      textTheme: GoogleFonts.interTextTheme(),
      fontFamily: GoogleFonts.inter().fontFamily,
      scaffoldBackgroundColor: ShadTokens.background,
      colorScheme: const ColorScheme.light(
        surface: ShadTokens.surface,
        onSurface: ShadTokens.foreground,
        primary: ShadTokens.primary,
        onPrimary: ShadTokens.primaryFg,
        secondary: ShadTokens.secondary,
        onSecondary: ShadTokens.secondaryFg,
        error: ShadTokens.destructive,
        onError: ShadTokens.destructiveFg,
        outline: ShadTokens.border,
      ),
      appBarTheme: AppBarTheme(
        backgroundColor: ShadTokens.surface,
        foregroundColor: ShadTokens.foreground,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        titleTextStyle: ShadTokens.h2,
        iconTheme: const IconThemeData(color: ShadTokens.foreground, size: 20),
      ),
      dividerTheme: const DividerThemeData(
        color: ShadTokens.border,
        thickness: 1,
        space: 1,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: ShadTokens.surface,
        indicatorColor: ShadTokens.secondary,
        elevation: 0,
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              color: ShadTokens.primary,
            );
          }
          return const TextStyle(
            fontSize: 11,
            fontWeight: FontWeight.w500,
            color: ShadTokens.muted,
          );
        }),
        iconTheme: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const IconThemeData(color: ShadTokens.primary, size: 22);
          }
          return const IconThemeData(color: ShadTokens.muted, size: 22);
        }),
      ),
    );
  }
}