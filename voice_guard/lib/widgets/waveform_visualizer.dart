import 'package:flutter/material.dart';
import '../design/tokens.dart';
import 'shad_waveform.dart';

export 'shad_waveform.dart';

/// Legacy adapter for WaveformVisualizer, wrapping ShadWaveform.
class WaveformVisualizer extends StatelessWidget {
  final List<double> samples;
  final Color color;

  const WaveformVisualizer({
    super.key,
    this.samples = const [],
    this.color = ShadTokens.primary,
  });

  @override
  Widget build(BuildContext context) {
    return ShadWaveform(
      samples: samples,
      color: color,
      height: 56,
      barCount: 36,
    );
  }
}
