import 'dart:typed_data';
import '../decision/decision_engine.dart';

class FrameResult {
  FrameResult({
    required this.t,
    required this.rawScore,
    required this.ema,
    required this.state,
    required this.melDb,
    required this.audioWindow,
  });

  final double t;
  final double rawScore;
  final double ema;
  final AlertState state;
  final List<List<double>> melDb;
  final Float32List audioWindow;
}
