import 'package:flutter/material.dart';

/// Visual-aid mel spectrogram (matches spectrogram.py's framing: "a visual
/// aid, not the model's feature extractor").
class SpectrogramView extends StatelessWidget {
  const SpectrogramView({super.key, required this.melDb});
  final List<List<double>> melDb;

  @override
  Widget build(BuildContext context) {
    if (melDb.isEmpty || melDb.first.isEmpty) {
      return const Center(child: Text('Waiting for audio...'));
    }
    return SizedBox(
      height: 160,
      child: CustomPaint(
        painter: _MelPainter(melDb),
        child: Container(),
      ),
    );
  }
}

class _MelPainter extends CustomPainter {
  _MelPainter(this.melDb);
  final List<List<double>> melDb;

  @override
  void paint(Canvas canvas, Size size) {
    final nMels = melDb.length;
    final nFrames = melDb.first.length;
    double minV = double.infinity, maxV = double.negativeInfinity;
    for (final row in melDb) {
      for (final v in row) {
        if (v < minV) minV = v;
        if (v > maxV) maxV = v;
      }
    }
    final range = (maxV - minV).abs() < 1e-9 ? 1.0 : maxV - minV;
    final cellW = size.width / nFrames;
    final cellH = size.height / nMels;
    final paint = Paint();
    for (var m = 0; m < nMels; m++) {
      for (var f = 0; f < nFrames; f++) {
        final t = (melDb[m][f] - minV) / range;
        paint.color = Color.lerp(Colors.black, Colors.deepOrangeAccent, t)!;
        canvas.drawRect(
          Rect.fromLTWH(f * cellW, (nMels - 1 - m) * cellH, cellW + 1, cellH + 1),
          paint,
        );
      }
    }
  }

  @override
  bool shouldRepaint(covariant _MelPainter oldDelegate) => oldDelegate.melDb != melDb;
}
