import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/scoring/stub_scorer.dart';

void main() {
  final audio = Float32List(16000); // 1 s of silence @ 16 kHz, content unused

  test('backendLabel discloses this is not a real model', () {
    final scorer = StubScorer();
    expect(scorer.backendLabel, contains('simulated'));
  });

  test('with no clone entry, score stays low regardless of position', () {
    final scorer = StubScorer(cloneEntryS: null);
    for (final t in [0.0, 10.0, 50.0]) {
      final s = scorer.scoreWindow(audio, 16000, t);
      expect(s, inInclusiveRange(0.0, 0.4));
    }
  });

  test('score rises once the window midpoint passes cloneEntryS', () {
    final scorer = StubScorer(cloneEntryS: 22.0);
    final before = scorer.scoreWindow(audio, 16000, 10.0); // mid=10.5
    final after = scorer.scoreWindow(audio, 16000, 22.0); // mid=22.5
    expect(before, lessThan(0.4));
    expect(after, greaterThan(0.6));
  });

  test('scores are always clamped to [0, 1]', () {
    final scorer = StubScorer(cloneEntryS: 22.0, seed: 1);
    for (final t in [0.0, 21.9, 22.0, 100.0]) {
      final s = scorer.scoreWindow(audio, 16000, t);
      expect(s, inInclusiveRange(0.0, 1.0));
    }
  });
}
