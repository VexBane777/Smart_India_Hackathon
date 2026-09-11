import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';

/// Deliberately NOISE-FREE (unlike audio_processor_parity_test.dart's
/// physio fixture): LFCC coefficients are computed per-frame from raw
/// spectral content, so any per-sample noise lands directly in every
/// frame's values — unlike physio's frame-*aggregate* statistics, which
/// average noise-realization differences out. A noisy fixture made
/// Python/Dart per-frame LFCC values diverge by several units on ~half
/// the coefficients purely from numpy's and Dart's PRNGs producing
/// different noise sequences (confirmed by inspection, not a real
/// algorithm bug) — this noise-free signal sidesteps that and lets this
/// test hold LFCC to a tight, meaningful tolerance. See
/// model_training/dump_sequence_parity_fixture.py's matching docstring.
List<double> _fixtureSignal() {
  const sampleRate = 16000;
  const n = sampleRate * 3;
  final sig = List<double>.filled(n, 0);
  double phase = 0;
  for (int i = 0; i < n; i++) {
    final t = i / sampleRate;
    final f0 = 140.0 + 6.0 * math.sin(2 * math.pi * 4.0 * t);
    phase += 2 * math.pi * f0 / sampleRate;
    final s = 0.6 * math.sin(phase) + 0.25 * math.sin(2 * phase) + 0.1 * math.sin(3 * phase);
    sig[i] = s;
  }
  final maxAbs = sig.map((v) => v.abs()).reduce(math.max);
  return sig.map((v) => v / maxAbs).toList();
}

void main() {
  test('extractLfccSequence matches Python within tolerance', () {
    final pcm = _fixtureSignal();
    final seq = AudioProcessor.extractLfccSequence(pcm);

    // Values from `python model_training/dump_sequence_parity_fixture.py`.
    const pythonNFrames = 184;
    const List<double> pythonFirstFrame = [
      -432.48292947018217, 37.700119336700006, 22.594291577725162, 16.598292946302017,
      13.215411104851981, 10.874749910143635, 9.055892754274968, 7.5346625164285435,
      6.20817041846245, 5.020324792046944, 3.943877498306142, 2.9625975209608124,
      2.069985442202409, 1.2620614700490231, 0.5385839704177067, -0.10090365186925716,
      -0.6555769903877677, -1.1258758538449092, -1.5121212624721212, -1.816288137895434,
      -2.0407733643377397, -2.18967297951546, -2.2676748556991857, -2.2809887219701994,
      -2.236336036354961, -2.1416282264759334, -2.0050382368037423, -1.8354980258647688,
      -1.6418482149593534, -1.4332128489644718, -1.2182321326572219, -1.005363506664369,
      -0.8022068961947444, -0.6157730279843389, -0.4519111059933337, -0.3155766371223036,
      -0.21036858508281453, -0.1388194495414355, -0.1020435898579797, -0.10006219536721302,
      -0.13155763832801085, -0.1942359225832301, -0.28467471565203295, -0.3987166770452291,
      -0.5313922975654034, -0.677330002972456, -0.8307300222005495, -0.985772026133675,
      -1.136613652131431, -1.2777741291177804, -1.4041311965681347, -1.5112598429512112,
      -1.5954042238118817, -1.6537541284173083, -1.6843738313549126, -1.6864047140920035,
      -1.6599401809593195, -1.606150248195727, -1.5270998913594203, -1.4257994484516658,
    ];
    const List<double> pythonLastFrame = [
      -400.41145592965046, 34.94407003511812, 18.9867242946888, 13.330364058945305,
      10.320760294640518, 8.34724877429046, 6.885844058417938, 5.708219120045075,
      4.708438091968826, 3.8267398207871985, 3.033000154007267, 2.3080090395357953,
      1.6430641994274011, 1.0321995195319034, 0.47399572725268385, -0.03277033835687404,
      -0.48722134847036297, -0.8892426422720084, -1.237822476097549, -1.533104874959539,
      -1.7749806152290082, -1.9646271668722417, -2.103299919123509, -2.1935148770686586,
      -2.237991656227042, -2.240573327310245, -2.2052858724708027, -2.1370601363746236,
      -2.0408854986792533, -1.9223833768208882, -1.7870432346635177, -1.640687878348713,
      -1.4887871933010075, -1.3368489832688601, -1.1898101847609779, -1.0523814355672299,
      -0.9285173103122166, -0.8217373187455326, -0.7346763782293882, -0.669399228165621,
      -0.6270304125385924, -0.6080731876182824, -0.612115287511054, -0.6381575396303758,
      -0.6843882102242135, -0.7485210789497415, -0.8276277752814439, -0.9184803865498595,
      -1.0174284067938975, -1.1207374136939483, -1.2244954650775752, -1.3249372427755304,
      -1.4183642197063933, -1.5014430697677443, -1.5711248030473592, -1.6249071690457155,
      -1.660740167830293, -1.6772444662769646, -1.6735940324065532, -1.6496859421461643,
    ];

    expect(seq.length, pythonNFrames);
    for (int i = 0; i < 60; i++) {
      expect(seq.first[i], closeTo(pythonFirstFrame[i], 0.05));
      expect(seq.last[i], closeTo(pythonLastFrame[i], 0.05));
    }
  });

  test('extractScalars matches Python within tolerance', () {
    final pcm = _fixtureSignal();
    final scalars = AudioProcessor.extractScalars(pcm);

    // Values from `python model_training/dump_sequence_parity_fixture.py`.
    const List<double> pythonScalars = [
      0.0, 0.00022107440099744456, 8.04486971037114e-07,
      0.0062373120715525565, 0.020694925647400914, 31.418730639804423,
    ];
    expect(scalars.length, 6);
    for (int i = 0; i < 6; i++) {
      // Index 5 (hnrDb) is in dB, coarser tolerance — matches this
      // project's existing convention for HNR comparisons.
      final tolerance = i == 5 ? 2.0 : 0.02;
      expect(scalars[i], closeTo(pythonScalars[i], tolerance));
    }
  });
}
