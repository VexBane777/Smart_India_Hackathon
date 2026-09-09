import 'package:flutter/material.dart';

/// Displays which time ranges of the current window drove the alarm
/// (master plan §6, item 2: occlusion sensitivity). This widget only
/// renders whatever ranges it's given — computing them requires a real
/// model and is out of Module E's scope; Phase 1 always passes an empty
/// list, which must render cleanly.
class OcclusionOverlay extends StatelessWidget {
  const OcclusionOverlay({
    super.key,
    required this.highlights,
    required this.windowDurationS,
  });

  final List<(double start, double end)> highlights;
  final double windowDurationS;

  @override
  Widget build(BuildContext context) {
    if (highlights.isEmpty) {
      return const Text('No occlusion highlights available yet.',
          style: TextStyle(color: Color(0xFF94A3B8)));
    }
    return Wrap(
      spacing: 8,
      children: [
        for (final (start, end) in highlights)
          Chip(label: Text('${start.toStringAsFixed(1)}s–${end.toStringAsFixed(1)}s')),
      ],
    );
  }
}
