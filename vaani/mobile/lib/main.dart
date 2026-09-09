import 'package:flutter/material.dart';
import 'ui/home_screen.dart';

void main() => runApp(const VaaniApp());

class VaaniApp extends StatelessWidget {
  const VaaniApp({super.key});
  @override
  Widget build(BuildContext context) => const MaterialApp(home: HomeScreen());
}
