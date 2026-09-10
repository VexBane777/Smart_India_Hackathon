import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:lucide_icons_flutter/lucide_icons.dart';
import 'providers/settings_provider.dart';
import 'screens/home_screen.dart';
import 'screens/call_screen.dart';
import 'screens/logs_screen.dart';
import 'screens/settings_screen.dart';
import 'screens/onboarding_screen.dart';
import 'design/theme.dart';
import 'utils/constants.dart';

class VaaniApp extends StatelessWidget {
  const VaaniApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: AppConstants.appName,
      debugShowCheckedModeBanner: false,
      theme: ShadThemeHelper.lightTheme,
      home: const RootNav(),
    );
  }
}

class RootNav extends StatefulWidget {
  const RootNav({super.key});
  @override
  State<RootNav> createState() => _RootNavState();
}

class _RootNavState extends State<RootNav> {
  int _idx = 0;
  final _pages = const [
    HomeScreen(),
    CallScreen(),
    LogsScreen(),
    SettingsScreen(),
  ];

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final done = context.read<SettingsProvider>().onboardingDone;
      if (!done) {
        Navigator.of(context).push(
          MaterialPageRoute(builder: (_) => const OnboardingScreen()),
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: ShadTokens.background,
      body: _pages[_idx],
      bottomNavigationBar: Container(
        decoration: const BoxDecoration(
          color: ShadTokens.surface,
          border: Border(top: BorderSide(color: ShadTokens.border, width: 1.0)),
        ),
        child: NavigationBar(
          selectedIndex: _idx,
          backgroundColor: ShadTokens.surface,
          elevation: 0,
          height: 64,
          onDestinationSelected: (i) => setState(() => _idx = i),
          destinations: const [
            NavigationDestination(
              icon: Icon(LucideIcons.gauge, size: 20),
              selectedIcon: Icon(LucideIcons.gauge, size: 20, color: ShadTokens.primary),
              label: 'Overview',
            ),
            NavigationDestination(
              icon: Icon(LucideIcons.phoneCall, size: 20),
              selectedIcon: Icon(LucideIcons.phoneCall, size: 20, color: ShadTokens.primary),
              label: 'Live Call',
            ),
            NavigationDestination(
              icon: Icon(LucideIcons.fileText, size: 20),
              selectedIcon: Icon(LucideIcons.fileText, size: 20, color: ShadTokens.primary),
              label: 'Audit Logs',
            ),
            NavigationDestination(
              icon: Icon(LucideIcons.slidersHorizontal, size: 20),
              selectedIcon: Icon(LucideIcons.slidersHorizontal, size: 20, color: ShadTokens.primary),
              label: 'Settings',
            ),
          ],
        ),
      ),
    );
  }
}
