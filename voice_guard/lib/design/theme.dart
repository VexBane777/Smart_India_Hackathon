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
      brightness: Brightness.dark,
      textTheme: GoogleFonts.interTextTheme(ThemeData.dark().textTheme),
      fontFamily: GoogleFonts.inter().fontFamily,
      scaffoldBackgroundColor: ShadTokens.background,
      colorScheme: const ColorScheme.dark(
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
      appBarTheme: const AppBarTheme(
        backgroundColor: Colors.transparent,
        foregroundColor: ShadTokens.foreground,
        elevation: 0,
        scrolledUnderElevation: 0,
        centerTitle: false,
        iconTheme: IconThemeData(color: ShadTokens.foreground, size: 20),
      ),
      dividerTheme: const DividerThemeData(
        color: ShadTokens.border,
        thickness: 1,
        space: 1,
      ),
      navigationBarTheme: NavigationBarThemeData(
        backgroundColor: ShadTokens.background,
        indicatorColor: Colors.transparent,
        elevation: 0,
        labelTextStyle: WidgetStateProperty.resolveWith((states) {
          if (states.contains(WidgetState.selected)) {
            return const TextStyle(
              fontSize: 11,
              fontWeight: FontWeight.w700,
              color: Colors.white,
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
            return const IconThemeData(color: Colors.white, size: 22);
          }
          return const IconThemeData(color: ShadTokens.muted, size: 22);
        }),
      ),
    );
  }
}