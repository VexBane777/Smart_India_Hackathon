# Physiological voice-quality features — jitter/shimmer/HNR (remediation track 2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add three genuine physiological voice-quality features — jitter
(local, %), shimmer (local, %), harmonic-to-noise ratio (dB) — to the
model's input, alongside the existing 60 LFCC + 3 prosody scalars, so the
model has *some* signal that isn't purely a delivery-style statistic (see
`voice_guard/docs/CRITICAL-entity-vs-style-confound.md` §3-4).

**Architecture:** Autocorrelation-based pitch-period tracking over the same
3s/16kHz window already used for LFCC/prosody, computed independently and
identically in `model_training/features.py` (numpy) and
`lib/utils/audio_processor.dart` (hand-rolled, matching this project's
existing convention — see `features.py`'s own docstring: "MUST stay
numerically equivalent to the Dart extractor"). The feature vector grows
from 63 to 66 floats, in a new fixed concatenation order
`[...lfcc(60), ...prosody(3), ...physio(3)]`. This ripples through
`model.py::INPUT_DIM`, `tflite_io.dart`'s `_inputDim`, and every place that
currently assumes 63 (found via grep: `train.py:197`'s ONNX-export dummy
tensor, `model_training/README.md`'s documented feature contract).
`model_training/dataset.py`'s `extract_features` call site needs no change
— it already passes through whatever `features.py::extract_features`
returns.

**Tech Stack:** Python/numpy (training side), Dart (on-device side), PyTorch
(model), ONNX Runtime (mobile inference, unchanged runtime — only the input
tensor's last dim changes).

**Spec:** `voice_guard/docs/CRITICAL-entity-vs-style-confound.md` §4, item 2.

## Global Constraints

- Dart and Python implementations MUST be numerically equivalent (existing
  project rule, `features.py:1-14`) — Task 2 adds the first-ever automated
  cross-language parity check for this file pair; there is currently none
  (verified: no such test exists anywhere in the repo as of this plan).
- No retrain is "production-ready" without beating the current deployed
  model's noise-FPR AND ITW-held-out-EER within statistical noise, per the
  hard-won lesson logged throughout `voice_guard/state.md` (the whole
  "attempt1 through v9_noisefix" saga) — Task 5 enforces this explicitly,
  do not skip the gate.
- The acceptance criterion for *this specific track* is not "EER improved"
  — it's a **smaller** style/entity confound, i.e. a weaker correlation
  between the model's fake-probability and the prosody-style features on
  real speech, per the CRITICAL doc's own measured baseline (25.1% vs.
  12.9% for `energyVariance`, etc.) — Task 5 reproduces that exact analysis
  against the new model as the primary pass/fail bar, EER is secondary.
- This is additive to the existing 63-d vector, not a replacement — do not
  remove LFCC or prosody. It is compatible with, and does not block, track
  3 (frame-level sequence model, already scoped in
  `docs/superpowers/plans/2026-09-11-frame-level-sequence-model-plan.md`,
  §2.4 notes this same track 2 could later be folded in as per-frame
  channels — out of scope here, this plan ships the standalone scalar
  version first).

---

### Task 1: Jitter/shimmer/HNR in `features.py` (Python side, TDD)

**Files:**
- Modify: `model_training/features.py`
- Create: `model_training/test_features.py`

**Interfaces:**
- Produces: `extract_physio(pcm: np.ndarray) -> np.ndarray` (shape `(3,)`,
  order `[jitter_local, shimmer_local, hnr_db]`); modifies
  `extract_features` to return shape `(66,)` instead of `(63,)`.

- [ ] **Step 1: Write the failing tests**

```python
# model_training/test_features.py
"""Tests for features.py, including the jitter/shimmer/HNR addition
(remediation track 2, see docs/CRITICAL-entity-vs-style-confound.md §4)."""
from __future__ import annotations

import numpy as np

from features import (
    N_LFCC,
    SAMPLE_RATE,
    extract_features,
    extract_physio,
    extract_prosody,
)


def _synthetic_voiced_tone(f0: float = 150.0, duration_s: float = 3.0,
                            jitter_frac: float = 0.0, amp_wobble_frac: float = 0.0,
                            noise_amp: float = 0.0, seed: int = 0) -> np.ndarray:
    """A deterministic near-periodic tone standing in for voiced speech:
    a fundamental + 2 harmonics, optional per-cycle period jitter and
    amplitude wobble (shimmer), plus optional additive white noise (for
    HNR). Used because it gives known-shape expectations (near-zero
    jitter/shimmer, high HNR when noise_amp=0) without needing a real
    speech corpus checked into the repo."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * SAMPLE_RATE)
    out = np.zeros(n)
    t = 0.0
    i = 0
    base_period = 1.0 / f0
    while i < n:
        period = base_period * (1.0 + jitter_frac * (rng.random() - 0.5) * 2)
        amp = 1.0 + amp_wobble_frac * (rng.random() - 0.5) * 2
        n_samples_cycle = max(1, int(round(period * SAMPLE_RATE)))
        cycle_t = np.arange(n_samples_cycle) / SAMPLE_RATE
        cycle = amp * (0.6 * np.sin(2 * np.pi * f0 * cycle_t)
                        + 0.3 * np.sin(2 * np.pi * 2 * f0 * cycle_t)
                        + 0.1 * np.sin(2 * np.pi * 3 * f0 * cycle_t))
        end = min(n, i + n_samples_cycle)
        out[i:end] = cycle[: end - i]
        i = end
    if noise_amp > 0:
        out = out + rng.normal(0, noise_amp, size=n)
    return (out / (np.abs(out).max() + 1e-9)).astype(np.float64)


def test_extract_physio_shape_and_order():
    pcm = _synthetic_voiced_tone()
    physio = extract_physio(pcm)
    assert physio.shape == (3,)


def test_clean_periodic_tone_has_low_jitter_and_shimmer_high_hnr():
    pcm = _synthetic_voiced_tone(jitter_frac=0.0, amp_wobble_frac=0.0, noise_amp=0.0)
    jitter, shimmer, hnr = extract_physio(pcm)
    assert jitter < 0.02
    assert shimmer < 0.02
    assert hnr > 15.0


def test_jittery_tone_has_higher_jitter_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(jitter_frac=0.0))
    jittery = extract_physio(_synthetic_voiced_tone(jitter_frac=0.08, seed=1))
    assert jittery[0] > clean[0]


def test_wobbly_amplitude_has_higher_shimmer_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(amp_wobble_frac=0.0))
    wobbly = extract_physio(_synthetic_voiced_tone(amp_wobble_frac=0.10, seed=2))
    assert wobbly[1] > clean[1]


def test_noisy_tone_has_lower_hnr_than_clean_tone():
    clean = extract_physio(_synthetic_voiced_tone(noise_amp=0.0))
    noisy = extract_physio(_synthetic_voiced_tone(noise_amp=0.15, seed=3))
    assert noisy[2] < clean[2]


def test_silence_returns_zeros_not_nan_or_exception():
    pcm = np.zeros(SAMPLE_RATE * 3, dtype=np.float64)
    physio = extract_physio(pcm)
    assert np.all(np.isfinite(physio))
    assert np.array_equal(physio, np.zeros(3))


def test_short_buffer_returns_zeros():
    pcm = np.zeros(100, dtype=np.float64)
    physio = extract_physio(pcm)
    assert np.array_equal(physio, np.zeros(3))


def test_extract_features_is_now_66_dimensional_in_documented_order():
    pcm = _synthetic_voiced_tone()
    feats = extract_features(pcm)
    assert feats.shape == (66,)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    np.testing.assert_allclose(feats[N_LFCC:N_LFCC + 3], prosody, rtol=1e-5)
    np.testing.assert_allclose(feats[N_LFCC + 3:N_LFCC + 6], physio, rtol=1e-5)
```

- [ ] **Step 2: Run tests to verify they fail**

Run (from `model_training/`, with `.venv313` active per this project's
existing convention):
`python -m pytest test_features.py -v`
Expected: FAIL — `extract_physio` does not exist, `extract_features` still
returns shape `(63,)`.

- [ ] **Step 3: Implement pitch tracking + jitter/shimmer/HNR in `features.py`**

Add to `model_training/features.py`, after the existing `extract_prosody`
function and before `extract_features`:

```python
# --- Physiological voice-quality features (remediation track 2, see
# docs/CRITICAL-entity-vs-style-confound.md §4 item 2) ---
#
# Standard autocorrelation-based pitch tracking (the same family of
# algorithm Praat uses for jitter/shimmer/HNR), computed directly on the
# 3s/16kHz window — no new dependency, pure numpy. This is a deliberate
# simplification vs. full pitch-synchronous waveform matching: amplitude
# for shimmer is taken as frame RMS rather than per-cycle peak amplitude,
# which is standard practice when a lighter-weight implementation is
# preferred over a full Praat-style pitch-marking pipeline, and is exactly
# mirrored on the Dart side (see audio_processor.dart) so both sides stay
# numerically equivalent to each other, which is the actual invariant this
# project depends on (not exact agreement with Praat itself).
_PITCH_FRAME_LEN = 480  # 30ms @ 16kHz
_PITCH_HOP = 160  # 10ms @ 16kHz
_MIN_F0 = 75.0  # Hz, typical human voice floor
_MAX_F0 = 500.0  # Hz, typical human voice ceiling
_MIN_LAG = int(round(SAMPLE_RATE / _MAX_F0))  # 32 samples
_MAX_LAG = int(round(SAMPLE_RATE / _MIN_F0))  # 213 samples
_VOICING_THRESHOLD = 0.30  # normalized autocorrelation peak


def _pitch_track(pcm: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (periods_s, amplitudes, autocorr_peaks, voiced_mask) — one
    entry per frame, aligned; unvoiced frames carry period=0/amp=0/r=0 and
    voiced_mask=False, filtered out by callers."""
    n = len(pcm)
    if n < _PITCH_FRAME_LEN:
        return (np.empty(0), np.empty(0), np.empty(0), np.empty(0, dtype=bool))
    n_frames = 1 + (n - _PITCH_FRAME_LEN) // _PITCH_HOP
    periods = np.zeros(n_frames)
    amps = np.zeros(n_frames)
    peaks = np.zeros(n_frames)
    voiced = np.zeros(n_frames, dtype=bool)
    for i in range(n_frames):
        start = i * _PITCH_HOP
        frame = pcm[start:start + _PITCH_FRAME_LEN]
        amps[i] = np.sqrt(np.mean(frame ** 2))
        energy0 = np.dot(frame, frame)
        if energy0 <= 1e-12:
            continue
        best_lag, best_r = 0, 0.0
        for lag in range(_MIN_LAG, min(_MAX_LAG, _PITCH_FRAME_LEN - 1) + 1):
            a, b = frame[:-lag], frame[lag:]
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            if denom <= 1e-12:
                continue
            r = np.dot(a, b) / denom
            if r > best_r:
                best_r, best_lag = r, lag
        peaks[i] = best_r
        if best_r >= _VOICING_THRESHOLD and best_lag > 0:
            voiced[i] = True
            periods[i] = best_lag / SAMPLE_RATE
    return periods, amps, peaks, voiced


def extract_physio(pcm: np.ndarray) -> np.ndarray:
    """3s PCM float buffer -> [jitter_local, shimmer_local, hnr_db].
    Zeros if fewer than 2 consecutive voiced frames are found (silence,
    noise, or too-short input) — mirrors extract_prosody's degenerate-input
    convention of returning zeros rather than raising or returning NaN."""
    periods, amps, peaks, voiced = _pitch_track(pcm)
    if voiced.sum() < 2:
        return np.zeros(3, dtype=np.float64)

    # Jitter/shimmer: only over PAIRS of consecutive (hop-adjacent) voiced
    # frames — a voiced frame next to an unvoiced one contributes no pair,
    # same convention Praat uses (skip across unvoiced gaps).
    pair_mask = voiced[:-1] & voiced[1:]
    if pair_mask.sum() < 1:
        return np.zeros(3, dtype=np.float64)

    p0, p1 = periods[:-1][pair_mask], periods[1:][pair_mask]
    a0, a1 = amps[:-1][pair_mask], amps[1:][pair_mask]

    mean_period = (p0 + p1).mean() / 2
    jitter_local = np.mean(np.abs(p1 - p0)) / mean_period if mean_period > 1e-12 else 0.0

    mean_amp = (a0 + a1).mean() / 2
    shimmer_local = np.mean(np.abs(a1 - a0)) / mean_amp if mean_amp > 1e-12 else 0.0

    voiced_peaks = np.clip(peaks[voiced], 0.0, 0.999999)
    hnr_db = float(np.mean(10 * np.log10(voiced_peaks / (1 - voiced_peaks) + 1e-12)))

    return np.array([jitter_local, shimmer_local, hnr_db], dtype=np.float64)
```

- [ ] **Step 4: Wire `extract_physio` into `extract_features`**

```python
def extract_features(pcm: np.ndarray) -> np.ndarray:
    """Full 66-d feature vector: 60 LFCC + 3 prosody + 3 physio, matching
    TFLiteService.infer's `[...lfcc, ...prosody, ...physio]` concatenation
    order exactly."""
    lfcc = extract_lfcc(pcm)
    prosody = extract_prosody(pcm)
    physio = extract_physio(pcm)
    return np.concatenate([lfcc, prosody, physio]).astype(np.float32)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest test_features.py -v`
Expected: PASS (9 tests). If `test_clean_periodic_tone_has_low_jitter...`
or the HNR/shimmer-direction tests are borderline, adjust the synthetic
fixture's `jitter_frac`/`amp_wobble_frac`/`noise_amp` magnitudes (not the
production thresholds) until the *direction* of each effect is unambiguous
— these are sanity checks on the algorithm's correctness, not tuned
thresholds for the real classifier.

- [ ] **Step 6: Run the existing dataset-integrity suite to check nothing downstream silently broke**

Run: `python -m pytest test_dataset_integrity.py test_pipeline_smoke.py -v`
Expected: same pass/fail pattern as before this change (the 3
`test_no_shortcut_in_accent_cell` failures documented in `state.md` as
pre-existing/expected are fine; anything newly broken is not).

- [ ] **Step 7: Commit**

```bash
git add model_training/features.py model_training/test_features.py
git commit -m "feat(voice_guard): add jitter/shimmer/HNR physiological features"
```

---

### Task 2: Mirror in `audio_processor.dart` + first-ever Dart/Python parity test

**Files:**
- Modify: `lib/utils/audio_processor.dart`
- Create: `test/audio_processor_parity_test.dart`
- Create: `model_training/dump_parity_fixture.py` (one-off generator, not
  part of the training pipeline — run once to produce the fixture values
  below, then the script itself stays in the repo so the fixture is
  regeneratable if the algorithm ever changes on the Python side)

**Interfaces:**
- Produces: `AudioProcessor.extractPhysio(List<double> pcm) -> List<double>`
  (length 3, order `[jitterLocal, shimmerLocal, hnrDb]`), mirroring
  `extract_physio` exactly.

- [ ] **Step 1: Generate the parity fixture from the Python implementation**

```python
# model_training/dump_parity_fixture.py
"""One-off: prints a deterministic PCM buffer (as a Dart-literal-ready CSV
of samples) and its Python-computed physio features, so the Dart parity
test (test/audio_processor_parity_test.dart) can assert against a value
independently produced by the already-tested Python implementation. Rerun
and re-paste into the Dart test only if extract_physio's algorithm changes
on the Python side — the whole point is these two must never silently
drift apart (see features.py's top-of-file docstring)."""
import numpy as np
from features import extract_physio, SAMPLE_RATE

rng = np.random.default_rng(42)
n = SAMPLE_RATE * 3
t = np.arange(n) / SAMPLE_RATE
# A fixed, realistic-ish synthetic voiced signal: wandering F0 (natural
# vibrato-like drift) + mild amplitude envelope + a little broadband noise,
# chosen to be unambiguously "voiced" for both implementations' voicing
# threshold rather than a pathological edge case.
f0 = 140.0 + 6.0 * np.sin(2 * np.pi * 4.0 * t)
phase = 2 * np.pi * np.cumsum(f0) / SAMPLE_RATE
sig = 0.6 * np.sin(phase) + 0.25 * np.sin(2 * phase) + 0.1 * np.sin(3 * phase)
sig = sig * (0.9 + 0.1 * np.sin(2 * np.pi * 1.5 * t))
sig = sig + rng.normal(0, 0.02, size=n)
sig = (sig / np.abs(sig).max()).astype(np.float64)

physio = extract_physio(sig)
print("jitter_local =", repr(float(physio[0])))
print("shimmer_local =", repr(float(physio[1])))
print("hnr_db =", repr(float(physio[2])))
# Print a compact way to reconstruct the identical signal in Dart, rather
# than dumping 48000 floats: the generator above is fully deterministic
# given (seed=42, n, formula), so the Dart test reconstructs it with the
# same closed-form expression instead of a literal array.
```

Run: `python model_training/dump_parity_fixture.py` and record the three
printed values — they go into the Dart test in Step 3 below as the
expected values (the actual run's output takes precedence over any
placeholder if they differ, since this step's whole purpose is to produce
ground truth from the already-tested Python side).

- [ ] **Step 2: Write the failing Dart parity test**

```dart
// test/audio_processor_parity_test.dart
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';

/// Reconstructs the exact same deterministic signal as
/// model_training/dump_parity_fixture.py (same seed=42 noise draw is NOT
/// reproduced here — see note below — everything else is the identical
/// closed-form signal), so this test can assert Dart's extractPhysio
/// matches Python's to a reasonable tolerance without shipping a 48000-
/// sample fixture file. The noise term uses Dart's own PRNG seeded
/// identically (`math.Random(42)`); Dart's PRNG algorithm differs from
/// numpy's, so the exact per-sample noise won't match numpy's —
/// acceptable because jitter/shimmer/HNR are frame-aggregate statistics
/// over a *mostly periodic* signal, robust to the specific noise draw at
/// this SNR. Tolerance below is set accordingly (looser than the LFCC/
/// prosody parity this project doesn't yet test either, but nonzero
/// tolerance is the honest choice here, not a hidden bug).
List<double> _fixtureSignal() {
  const sampleRate = 16000;
  const n = sampleRate * 3;
  final rng = math.Random(42);
  final sig = List<double>.filled(n, 0);
  double phase = 0;
  for (int i = 0; i < n; i++) {
    final t = i / sampleRate;
    final f0 = 140.0 + 6.0 * math.sin(2 * math.pi * 4.0 * t);
    phase += 2 * math.pi * f0 / sampleRate;
    double s = 0.6 * math.sin(phase) + 0.25 * math.sin(2 * phase) + 0.1 * math.sin(3 * phase);
    s *= 0.9 + 0.1 * math.sin(2 * math.pi * 1.5 * t);
    s += (rng.nextDouble() - 0.5) * 2 * 0.02;
    sig[i] = s;
  }
  final maxAbs = sig.map((v) => v.abs()).reduce(math.max);
  return sig.map((v) => v / maxAbs).toList();
}

void main() {
  test('extractPhysio matches Python features.py::extract_physio within tolerance', () {
    final pcm = _fixtureSignal();
    final physio = AudioProcessor.extractPhysio(pcm);
    expect(physio.length, 3);

    // Values from `python model_training/dump_parity_fixture.py` — PASTE
    // THE ACTUAL PRINTED VALUES FROM STEP 1 HERE, replacing these three.
    const pythonJitter = 0.0; // REPLACE
    const pythonShimmer = 0.0; // REPLACE
    const pythonHnrDb = 0.0; // REPLACE

    expect(physio[0], closeTo(pythonJitter, 0.02));
    expect(physio[1], closeTo(pythonShimmer, 0.02));
    expect(physio[2], closeTo(pythonHnrDb, 2.0)); // dB, coarser tolerance
  });
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `flutter test test/audio_processor_parity_test.dart`
Expected: FAIL — `AudioProcessor.extractPhysio` doesn't exist yet (and the
placeholder `0.0` expected values are wrong regardless).

- [ ] **Step 4: Implement `extractPhysio` in `audio_processor.dart`**

Add to `lib/utils/audio_processor.dart`, mirroring `features.py`'s
`_pitch_track`/`extract_physio` exactly (same frame len/hop/lag
range/voicing threshold as constants):

```dart
  // --- Physiological voice-quality features (remediation track 2) ---
  static const int _pitchFrameLen = 480; // 30ms @ 16kHz
  static const int _pitchHop = 160; // 10ms @ 16kHz
  static const double _minF0 = 75.0;
  static const double _maxF0 = 500.0;
  static final int _minLag = (sampleRate / _maxF0).round(); // 32
  static final int _maxLag = (sampleRate / _minF0).round(); // 213
  static const double _voicingThreshold = 0.30;

  /// [periods(s), amplitudes, autocorrPeaks, voicedMask] — one entry per
  /// frame, mirrors features.py::_pitch_track exactly.
  static (List<double>, List<double>, List<double>, List<bool>) _pitchTrack(List<double> pcm) {
    final n = pcm.length;
    if (n < _pitchFrameLen) return (const [], const [], const [], const []);
    final nFrames = 1 + (n - _pitchFrameLen) ~/ _pitchHop;
    final periods = List<double>.filled(nFrames, 0);
    final amps = List<double>.filled(nFrames, 0);
    final peaks = List<double>.filled(nFrames, 0);
    final voiced = List<bool>.filled(nFrames, false);
    for (int i = 0; i < nFrames; i++) {
      final start = i * _pitchHop;
      final frame = pcm.sublist(start, start + _pitchFrameLen);
      double sumSq = 0;
      for (final v in frame) { sumSq += v * v; }
      amps[i] = math.sqrt(sumSq / _pitchFrameLen);
      if (sumSq <= 1e-12) continue;
      int bestLag = 0;
      double bestR = 0.0;
      final maxLagForFrame = math.min(_maxLag, _pitchFrameLen - 1);
      for (int lag = _minLag; lag <= maxLagForFrame; lag++) {
        double dotAB = 0, dotAA = 0, dotBB = 0;
        for (int k = 0; k < _pitchFrameLen - lag; k++) {
          final a = frame[k], b = frame[k + lag];
          dotAB += a * b; dotAA += a * a; dotBB += b * b;
        }
        final denom = math.sqrt(dotAA * dotBB);
        if (denom <= 1e-12) continue;
        final r = dotAB / denom;
        if (r > bestR) { bestR = r; bestLag = lag; }
      }
      peaks[i] = bestR;
      if (bestR >= _voicingThreshold && bestLag > 0) {
        voiced[i] = true;
        periods[i] = bestLag / sampleRate;
      }
    }
    return (periods, amps, peaks, voiced);
  }

  /// 3s PCM buffer -> [jitterLocal, shimmerLocal, hnrDb]. Mirrors
  /// features.py::extract_physio exactly — see that function's docstring
  /// for the algorithm and its deliberate simplifications.
  static List<double> extractPhysio(List<double> pcm) {
    final (periods, amps, peaks, voiced) = _pitchTrack(pcm);
    final voicedCount = voiced.where((v) => v).length;
    if (voicedCount < 2) return [0, 0, 0];

    final pairIdx = <int>[];
    for (int i = 0; i < voiced.length - 1; i++) {
      if (voiced[i] && voiced[i + 1]) pairIdx.add(i);
    }
    if (pairIdx.isEmpty) return [0, 0, 0];

    double sumAbsPeriodDiff = 0, sumMeanPeriod = 0;
    double sumAbsAmpDiff = 0, sumMeanAmp = 0;
    for (final i in pairIdx) {
      final p0 = periods[i], p1 = periods[i + 1];
      sumAbsPeriodDiff += (p1 - p0).abs();
      sumMeanPeriod += (p0 + p1) / 2;
      final a0 = amps[i], a1 = amps[i + 1];
      sumAbsAmpDiff += (a1 - a0).abs();
      sumMeanAmp += (a0 + a1) / 2;
    }
    final meanPeriod = sumMeanPeriod / pairIdx.length;
    final meanAmp = sumMeanAmp / pairIdx.length;
    final jitterLocal = meanPeriod > 1e-12 ? (sumAbsPeriodDiff / pairIdx.length) / meanPeriod : 0.0;
    final shimmerLocal = meanAmp > 1e-12 ? (sumAbsAmpDiff / pairIdx.length) / meanAmp : 0.0;

    double sumHnr = 0;
    for (int i = 0; i < voiced.length; i++) {
      if (!voiced[i]) continue;
      final r = peaks[i].clamp(0.0, 0.999999);
      sumHnr += 10 * math.log(r / (1 - r) + 1e-12) / math.ln10;
    }
    final hnrDb = sumHnr / voicedCount;

    return [jitterLocal, shimmerLocal, hnrDb];
  }
```

- [ ] **Step 5: Paste the real Python-computed values into the Dart test, then run it**

Replace the three `REPLACE` placeholders in
`test/audio_processor_parity_test.dart` with the actual output from Step 1.

Run: `flutter test test/audio_processor_parity_test.dart`
Expected: PASS. If it fails, the discrepancy is either a real algorithm
mismatch (fix the Dart or Python side to match — re-read both
implementations side by side, line by line) or a tolerance that's too
tight for the PRNG-divergence reason documented in the test's own comment
(widen tolerance, but only after confirming by inspection that the
algorithms genuinely match — do not widen tolerance to paper over a real
bug).

- [ ] **Step 6: Commit**

```bash
git add lib/utils/audio_processor.dart test/audio_processor_parity_test.dart model_training/dump_parity_fixture.py
git commit -m "feat(voice_guard): mirror jitter/shimmer/HNR in Dart, add Dart/Python parity test"
```

---

### Task 3: Wire the 66-d vector through the model and inference path

**Files:**
- Modify: `model_training/model.py:14` (`INPUT_DIM`)
- Modify: `model_training/train.py:197` (dummy ONNX-export tensor)
- Modify: `lib/services/src/tflite_io.dart:20` (`_inputDim`), `:38` (concatenation)
- Modify: `lib/services/audio_service.dart` (`scoreChunk`)
- Modify: `model_training/README.md` (feature-contract doc)

- [ ] **Step 1: Add a shape-safety assert to `VoiceGuardMLP` (this plan's grounding found a real latent bug here)**

`model.py`'s `VoiceGuardMLP.__init__` currently defaults `input_dim` to the
module-level `INPUT_DIM` constant but is never passed an explicit
`input_dim` from `train.py` — it silently trusts the caller's `norm_mean`/
`norm_std` shapes to agree with whatever `INPUT_DIM` happens to be. Fix
both the constant and add a real guard so this can't silently drift again:

```python
INPUT_DIM = 66  # 60 LFCC + 3 prosody + 3 physio (jitter/shimmer/HNR), must match audio_processor.dart exactly
```

In `FixedNormalize.__init__`, add:
```python
        assert mean.shape == std.shape, f"mean/std shape mismatch: {mean.shape} vs {std.shape}"
```

In `VoiceGuardMLP.__init__`, right after `self.normalize = FixedNormalize(...)`, add:
```python
        actual_dim = self.normalize.mean.shape[0]
        assert actual_dim == input_dim, (
            f"input_dim={input_dim} but norm_mean/norm_std have {actual_dim} entries — "
            "train.py must pass input_dim= explicitly if features.py's output width changes"
        )
```

- [ ] **Step 2: Make `train.py` pass `input_dim` explicitly rather than relying on the default**

In `train.py`, the `VoiceGuardMLP(...)` construction currently reads:
```python
    model = VoiceGuardMLP(norm_mean=mean, norm_std=std, hidden_dims=tuple(args.hidden_dims)).to(device)
```
Change to:
```python
    model = VoiceGuardMLP(
        input_dim=X_train.shape[1], norm_mean=mean, norm_std=std, hidden_dims=tuple(args.hidden_dims)
    ).to(device)
```
This makes the model's input width always derive from whatever
`features.py` actually produced for this run, not a constant that has to
be remembered to update — the Step 1 assert then catches any future
mismatch immediately instead of producing a model that silently trains on
misaligned normalization stats.

Also fix the hardcoded dummy-tensor fallback a few lines down:
```python
    dummy = torch.from_numpy(X_val[:1]) if len(X_val) else torch.zeros(1, 63)
```
to:
```python
    dummy = torch.from_numpy(X_val[:1]) if len(X_val) else torch.zeros(1, X_train.shape[1])
```

- [ ] **Step 3: Update `tflite_io.dart`**

```dart
  static const int _inputDim = 66; // must match model_training's INPUT_DIM (model.py)
```

Change `infer`'s signature and the app's one call site
(`scoreChunk` in the same file) to thread physio through:

```dart
  Future<double> infer(List<double> lfcc, List<double> prosody, List<double> physio) async {
    ...
        final input = [...lfcc, ...prosody, ...physio];
    ...
  }

  double _heuristic(List<double> lfcc, List<double> prosody) {
    // unchanged — the heuristic fallback path deliberately doesn't need
    // physio; it's a coarse fallback for when no ONNX model loaded at all.
```

```dart
  Future<double> scoreChunk(List<double> pcm) {
    final lfcc = AudioProcessor.extractLfcc(pcm);
    final prosody = AudioProcessor.extractProsody(pcm);
    final physio = AudioProcessor.extractPhysio(pcm);
    return infer(lfcc, prosody, physio);
  }
```

Check for any other call site of `infer(` besides `scoreChunk` (`grep -rn
"\.infer(" lib/`) and update it the same way — if none, this is the only
one.

- [ ] **Step 4: Update `model_training/README.md`'s feature contract**

Change the "Feature contract" paragraph (currently at `README.md:56-64`) to
describe 66 floats in order
`[...lfcc(60), ...prosody(3), ...physio(3)]`, naming the physio features
explicitly (`jitter_local`, `shimmer_local`, `hnr_db`) and cross-referencing
this plan file and the CRITICAL doc, same as the existing paragraph
cross-references `audio_processor.dart`.

- [ ] **Step 5: Run the full Python test suite**

Run: `python -m pytest model_training/ -v` (from `voice_guard/`, or `cd
model_training && python -m pytest -v`)
Expected: all `test_features.py` tests still pass (Task 1), and nothing in
`test_dataset_integrity.py`/`test_pipeline_smoke.py` newly fails because of
the dimension change — those tests operate on shapes derived from
`extract_features`'s actual output, not a hardcoded 63, so they should
adapt automatically; if any hardcode 63, fix them here.

- [ ] **Step 6: `flutter analyze` and `flutter test`**

Run: `flutter analyze && flutter test`
Expected: clean analyze, all tests (including Task 2's parity test) pass.
Note this does NOT yet verify on-device — the shipped `.onnx` still has 63
inputs until Task 4/5 retrain and redeploy; this step only verifies the
Dart code compiles and unit-tests correctly against the new function
signatures.

- [ ] **Step 7: Commit**

```bash
git add model_training/model.py model_training/train.py model_training/README.md lib/services/src/tflite_io.dart lib/services/audio_service.dart
git commit -m "feat(voice_guard): wire 66-d jitter/shimmer/HNR feature vector through model and inference path"
```

---

### Task 4: Retrain and gate on the confound-reduction criterion

**Files:** none (this is a training-run + analysis task, not a code
change) — outputs land in a new `model_training/runs/voice_guard_v10_physio/`

- [ ] **Step 1: Preflight the corpus (unchanged corpus, just confirming nothing regressed given the new feature dim)**

Run the same `check_corpus.py` preflight this project always runs before
committing to a training run (see `voice_guard/state.md`'s repeated
"Run this before train.py, not after"), against whatever corpus
configuration the currently-deployed model
(`runs/voice_guard_v9_noisefix_final`) was trained on — read that run's
directory for its exact invocation before choosing arguments here, don't
guess.

- [ ] **Step 2: Retrain with the exact same recipe as the currently-deployed model, only the feature vector changed**

This isolates "did adding physio features help" as the single variable —
matching this project's own hard-learned discipline (see `state.md`'s
"attempt1" postmortem: an unintentional recipe change confounded every
early result this project produced). Use `train.py` with the same
`--channel`, `--real`/`--fake`/`--real-clean`/`--fake-clean` args, weight
decay, label smoothing, and `--save-every-epoch-checkpoints` as v9_noisefix
(again: read that run's actual invocation, don't assume it matches the
README's documented command per `state.md`'s own documented caveat that the
two have diverged before).

```bash
python train.py \
  --real <same as v9_noisefix> --fake <same as v9_noisefix> \
  --real-clean <same as v9_noisefix> --fake-clean <same as v9_noisefix> \
  --channel whatsapp volte none \
  --weight-decay 1e-4 --label-smoothing 0.05 \
  --save-every-epoch-checkpoints \
  --out runs/voice_guard_v10_physio --epochs 25
```

Launch detached (`nohup ... &` + `disown`) per this project's own documented
memory-constraint workaround (`state.md`, "Memory constraint discovered and
worked around this session") if running on the same machine — do not
assume more RAM is available now than was measured before.

- [ ] **Step 3: Select the best checkpoint by noise-FPR, same metric v9_noisefix was gated on**

```bash
python select_best_checkpoint.py \
  --run runs/voice_guard_v10_physio \
  --real data/real_noise_aug_split/held/en_native data/real_noise_aug_split/held/hi_native
```
(Check `select_best_checkpoint.py --help` for its exact current flags before
running — this plan describes the metric to select on, not a guessed CLI
surface for a script this plan didn't modify.)

- [ ] **Step 4: THE actual acceptance test — reproduce the CRITICAL doc's median-split analysis against the new model**

This is the primary pass/fail bar for this entire track (see Global
Constraints) — write a small one-off script (not part of the permanent
pipeline, since it's a diagnostic, not a repeated operation):

```python
# model_training/measure_confound.py
"""Reproduces the median-split analysis from
docs/CRITICAL-entity-vs-style-confound.md §2 against a given checkpoint,
to check whether adding physio features shrank the style/entity confound.
Usage: python measure_confound.py --model runs/voice_guard_v10_physio/model.pt --real data/real_itw_held"""
import argparse
from pathlib import Path

import numpy as np
import torch

from dataset import build_examples
from features import N_LFCC
from model import VoiceGuardMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--real", type=Path, nargs="+", required=True)
    args = ap.parse_args()

    examples = build_examples(args.real, [], channel_recipes=[None])
    X = np.stack([e.features for e in examples])

    state = torch.load(args.model, map_location="cpu", weights_only=True)
    input_dim = state["normalize.mean"].shape[0]
    model = VoiceGuardMLP(input_dim=input_dim)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(X)), dim=-1)[:, 1].numpy()

    prosody_start = N_LFCC
    names = ["energyVariance", "pauseRatio", "zcrVariance"]
    # matches extract_prosody's [pauseRatio, energyVariance, zcrVariance] order
    idx = {"pauseRatio": prosody_start + 0, "energyVariance": prosody_start + 1, "zcrVariance": prosody_start + 2}
    for name in names:
        col = X[:, idx[name]]
        median = np.median(col)
        low_mean = probs[col < median].mean()
        high_mean = probs[col >= median].mean()
        print(f"{name}: low-half mean fake-prob={low_mean:.3f}  high-half={high_mean:.3f}  gap={abs(high_mean - low_mean):.3f}")


if __name__ == "__main__":
    main()
```

Run against both the OLD deployed model and the NEW v10_physio checkpoint,
on the same `data/real_itw_held` set the CRITICAL doc used:
```bash
python measure_confound.py --model runs/voice_guard_v9_noisefix_final/model.pt --real data/real_itw_held
python measure_confound.py --model runs/voice_guard_v10_physio/checkpoints/<selected>.pt --real data/real_itw_held
```

**Pass condition:** the new model's gap (`|high_mean - low_mean|`) for at
least `energyVariance` and `pauseRatio` is *meaningfully* smaller than the
old model's measured 12.2-point and 8.9-point gaps (25.1%-12.9% and
14.7%-23.6% respectively, from the CRITICAL doc §2) — not just numerically
smaller by noise-level margin. If unsure whether a shrink is real or noise,
reuse `eval_stats.py::bootstrap_eer_ci`'s bootstrap approach adapted to
this gap statistic, per this project's own established practice
(`state.md`'s "Replication check failed, bootstrap CIs built" section) —
do not eyeball a single point estimate given this project's own documented
history of chasing noise.

- [ ] **Step 5: Also check ITW EER and noise-FPR didn't regress**

```bash
python eval_held_out_dirs.py --model runs/voice_guard_v10_physio/checkpoints/<selected>.pt \
  --real data/real_itw_held --fake data/fake_itw_held --baseline-eer 0.1538
```
(0.1538 is the current correctly-re-measured v3/v9-family baseline per
`state.md`'s bootstrap-CI section — re-confirm this is still the right
comparison number for whatever checkpoint is actually deployed by the time
this task runs, don't copy it blindly if `state.md` has moved on.)

Record all three numbers (confound gaps, ITW EER, noise-FPR) — this is what
Task 5 documents.

---

### Task 5: Deploy gate + documentation

**Files:**
- Modify (conditionally): `assets/models/voice_detector.onnx`
- Modify: `voice_guard/state.md`

- [ ] **Step 1: Only replace the shipped model if Task 4's pass condition holds**

If the confound gap shrank meaningfully (Task 4 Step 4) AND ITW EER/noise-
FPR didn't regress beyond noise (Task 4 Step 5): copy
`runs/voice_guard_v10_physio/model.onnx` (or re-export the selected
checkpoint via the same `torch.onnx.export` pattern `train.py` already
uses, if the selected checkpoint isn't epoch-final) to
`assets/models/voice_detector.onnx`.

If it did NOT pass: do not replace the shipped model. Say so plainly (per
this project's own repeatedly-stated preference for honest negative
results over quiet omission — see `state.md` throughout, e.g. the
"attempt1/2/ablation" section's explicit "did not deploy" calls).

- [ ] **Step 2: On-device verification if a phone is available**

Same discipline as every prior model swap in this project (`state.md`'s
"Verify on-device before demo" pattern): install a build with the new
`.onnx`, run Live Mic Test with a deliberately monotone/controlled human
reading (the exact register that triggered the original 60-80s false
positive in the CRITICAL doc's §1) and confirm it now scores lower than
before. If no phone is available, say so explicitly rather than presenting
this as verified.

- [ ] **Step 3: Update `state.md` and the CRITICAL doc**

In `voice_guard/state.md`, add a dated section with: what was implemented,
the Task 4 measured numbers (old vs. new confound gaps, ITW EER, noise-FPR),
whether the model was deployed, and on-device verification status.

In `voice_guard/docs/CRITICAL-entity-vs-style-confound.md` §4, update item
2's status from "not yet started" to reflect the outcome (implemented +
measured result, whether or not it was enough to fully resolve the
confound — track 3 remains the most-likely-to-fully-resolve option
regardless of how this track's numbers land, per that section's own
original ranking).

- [ ] **Step 4: Commit**

```bash
git add voice_guard/state.md voice_guard/docs/CRITICAL-entity-vs-style-confound.md
# add assets/models/voice_detector.onnx too, only if Step 1 deployed it
git commit -m "feat(voice_guard): retrain with jitter/shimmer/HNR, gate on confound-reduction criterion"
```
