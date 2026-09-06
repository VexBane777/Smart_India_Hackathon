# VAANI Module E (Mobile / Android) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Flutter Android app (`vaani/mobile/`) that reproduces the
desktop VAANI demo — gauge, spectrogram, risk curve, occlusion highlights,
bank HOLD→OTP→release, hash-chained audit log — driven by an imported audio
file or the phone's live mic, phased so Phase 1 (a clearly-labeled stub
scorer) ships without waiting on Module B's not-yet-existing trained model,
and Phase 2 (real on-device inference) swaps in at one call site once that
model exists.

**Architecture:** Flutter/Dart app shell and UI; a small native Kotlin
module for the two things Dart can't do well (native audio decode/capture
and FFT/mel-spectrogram math), exposed to Dart via MethodChannel/EventChannel;
pure-Dart decision logic and audit log ported line-for-line from the existing
Python demo (`app/engine_mock.py`, `app/components/audit_log.py`,
`app/components/bank_modal.py`) so both platforms enforce identical
thresholds and timing.

**Tech Stack:** Flutter (Dart) · Kotlin (Android platform channels) ·
`crypto` (SHA-256), `file_picker` (SAF import), `permission_handler` (mic
permission), `fl_chart` (risk curve), `onnxruntime` (Phase 2 inference) ·
Android instrumented tests (`integration_test`) for anything touching real
Android APIs; plain `flutter test`/Kotlin JUnit for pure logic.

**Spec:** `vaani/docs/superpowers/specs/2026-09-07-vaani-module-e-mobile-design.md`

## Global Constraints

- No live capture of another app's call/VoIP audio (WhatsApp, Teams, native
  dialer) — OS-level restriction, not attempted even as a stretch goal (spec §1).
- No network calls anywhere in the pipeline — fully on-device, no cloud, no
  server fallback (spec §3, §8; master plan §1.4).
- Every score produced by `StubScorer` must be visibly labeled in the UI as
  simulated — never presented as if it were a real detection (spec §2).
- `DecisionEngine` constants must match `engine_mock.py` exactly: window 2.0 s,
  hop 0.5 s, EMA α=0.7, alert threshold 0.6, 2 consecutive windows required
  (spec §3; `app/engine_mock.py:25-32`).
- The Phase 1 → Phase 2 scorer swap must touch exactly one call site (spec §2,
  §6) — this is a tested contract, not a guideline.
- Android-only; no iOS target (spec §8).

---

## File Structure

```
vaani/mobile/                          # new Flutter project root
  pubspec.yaml
  lib/
    decision/
      decision_engine.dart             # Task 2
    audit/
      audit_log.dart                   # Task 3
    scoring/
      scorer.dart                      # Task 4 (interface)
      stub_scorer.dart                 # Task 4
      onnx_scorer.dart                 # Task 15 (Phase 2)
    mel/
      mel_bridge.dart                  # Task 6 (Dart side of MethodChannel)
    capture/
      audio_decode_bridge.dart         # Task 7 (Dart side)
      audio_capture_bridge.dart        # Task 8 (Dart side)
    pipeline/
      window_pipeline.dart             # Task 9
      frame_result.dart                # Task 9
    ui/
      gauge.dart                       # Task 10
      spectrogram_view.dart            # Task 11
      risk_curve.dart                  # Task 11
      occlusion_overlay.dart           # Task 11
      bank_modal.dart                  # Task 12
      home_screen.dart                 # Task 13
    main.dart                          # Task 13
  android/app/src/main/kotlin/org/vaani/mobile/
    MainActivity.kt                    # Task 1, extended in 5/7/8
    MelBridge.kt                       # Task 5
    AudioDecodeBridge.kt               # Task 7
    AudioCaptureBridge.kt              # Task 8
  android/app/src/test/kotlin/org/vaani/mobile/
    MelBridgeTest.kt                   # Task 5
  test/
    decision/decision_engine_test.dart
    audit/audit_log_test.dart
    scoring/stub_scorer_test.dart
    scoring/onnx_scorer_test.dart
    mel/mel_bridge_test.dart
    pipeline/window_pipeline_test.dart
    ui/bank_modal_logic_test.dart
  integration_test/
    file_import_flow_test.dart
    mic_capture_flow_test.dart
    bank_flow_test.dart
  assets/
    demo/call_A_whatsapp.wav           # copied from vaani/assets/demo/ (Task 13)
    test_fixtures/tone_1s_16k.wav      # generated in Task 7
    test_fixtures/identity_model.onnx  # generated in Task 15
vaani/00_MASTER_PLAN.md                # edited in Task 14
```

---

### Task 1: Flutter project scaffold

**Files:**
- Create: `vaani/mobile/pubspec.yaml`
- Create: `vaani/mobile/lib/main.dart` (placeholder `MaterialApp` — replaced in Task 13)
- Create: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt`
- Create: `vaani/mobile/test/widget_smoke_test.dart`

**Interfaces:**
- Produces: a Flutter project at `vaani/mobile/` that `flutter test` and
  `flutter build apk` can run against — every later task assumes this exists.

- [ ] **Step 1: Create the Flutter project**

Run from `vaani/`:
```bash
flutter create --org org.vaani --platforms android -a kotlin mobile
```
This generates the standard Flutter/Android scaffold, including
`android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt`.

- [ ] **Step 2: Pin dependencies in `pubspec.yaml`**

Add under `dependencies:` (keep the generated `flutter:` entry):
```yaml
dependencies:
  flutter:
    sdk: flutter
  crypto: ^3.0.3
  file_picker: ^8.1.2
  permission_handler: ^11.3.1
  fl_chart: ^0.69.0
  onnxruntime: ^1.4.1

dev_dependencies:
  flutter_test:
    sdk: flutter
  integration_test:
    sdk: flutter
```

- [ ] **Step 3: Install packages**

Run: `cd vaani/mobile && flutter pub get`
Expected: resolves with no version conflicts.

- [ ] **Step 4: Write the smoke test**

`vaani/mobile/test/widget_smoke_test.dart`:
```dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

void main() {
  testWidgets('app scaffold builds', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: Scaffold()));
    expect(find.byType(Scaffold), findsOneWidget);
  });
}
```

- [ ] **Step 5: Run the smoke test**

Run: `flutter test test/widget_smoke_test.dart`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add vaani/mobile
git commit -m "feat(mobile): scaffold Flutter Android project for Module E"
```

---

### Task 2: DecisionEngine (EMA + 2-consecutive-window alert rule)

**Files:**
- Create: `vaani/mobile/lib/decision/decision_engine.dart`
- Test: `vaani/mobile/test/decision/decision_engine_test.dart`

**Interfaces:**
- Produces: `enum AlertState { normal, warn, alert }`; class
  `DecisionEngine({double threshold = 0.6, int consecutiveRequired = 2,
  double emaAlpha = 0.7})` with `double? ema` (readable field) and
  `AlertState update(double rawScore)`. Later tasks (9, 12, 13) consume this
  exact API.

This is a direct port of `app/engine_mock.py:99-129`'s `AlertStateMachine` —
same constants, same formula, same three-state output.

- [ ] **Step 1: Write the failing tests**

```dart
// vaani/mobile/test/decision/decision_engine_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/decision/decision_engine.dart';

void main() {
  test('first low score stays normal', () {
    final engine = DecisionEngine();
    expect(engine.update(0.1), AlertState.normal);
    expect(engine.ema, closeTo(0.1, 1e-9));
  });

  test('two consecutive high-EMA windows trigger alert', () {
    final engine = DecisionEngine();
    expect(engine.update(0.9), AlertState.warn); // ema = 0.9 >= 0.6, count=1
    expect(engine.update(0.9), AlertState.alert); // count=2
  });

  test('a single high window then a low one never alerts', () {
    final engine = DecisionEngine();
    expect(engine.update(0.9), AlertState.warn);
    expect(engine.update(0.0), AlertState.normal); // ema drops below 0.6
  });

  test('ema formula matches alpha * raw + (1 - alpha) * prev', () {
    final engine = DecisionEngine(); // alpha = 0.7
    engine.update(1.0);
    expect(engine.ema, closeTo(1.0, 1e-9));
    engine.update(0.0);
    // 0.7*0.0 + 0.3*1.0 = 0.3
    expect(engine.ema, closeTo(0.3, 1e-9));
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `flutter test test/decision/decision_engine_test.dart`
Expected: FAIL — `decision_engine.dart` doesn't exist / `mobile` package
import fails (add `name: mobile` to `pubspec.yaml` if not already set by
`flutter create`, so `package:mobile/...` imports resolve).

- [ ] **Step 3: Implement `DecisionEngine`**

```dart
// vaani/mobile/lib/decision/decision_engine.dart
/// Ported from app/engine_mock.py's AlertStateMachine (master plan §6):
/// per-window score -> EMA smoothing -> alert only after 2+ consecutive
/// windows stay above threshold. Constants match the Python original
/// exactly so both platforms behave identically.
enum AlertState { normal, warn, alert }

class DecisionEngine {
  DecisionEngine({
    this.threshold = 0.6,
    this.consecutiveRequired = 2,
    this.emaAlpha = 0.7,
  });

  final double threshold;
  final int consecutiveRequired;
  final double emaAlpha;

  double? ema;
  int _consecutiveHigh = 0;

  AlertState update(double rawScore) {
    ema = ema == null ? rawScore : emaAlpha * rawScore + (1 - emaAlpha) * ema!;
    if (ema! >= threshold) {
      _consecutiveHigh += 1;
    } else {
      _consecutiveHigh = 0;
    }
    if (_consecutiveHigh >= consecutiveRequired) return AlertState.alert;
    if (_consecutiveHigh > 0) return AlertState.warn;
    return AlertState.normal;
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `flutter test test/decision/decision_engine_test.dart`
Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/decision vaani/mobile/test/decision
git commit -m "feat(mobile): port AlertStateMachine to DecisionEngine (Dart)"
```

---

### Task 3: AuditLog (SHA-256 hash chain)

**Files:**
- Create: `vaani/mobile/lib/audit/audit_log.dart`
- Test: `vaani/mobile/test/audit/audit_log_test.dart`

**Interfaces:**
- Produces: `class AuditEntry` (fields: `seq`, `ts`, `event`, `payload`,
  `prevHash`, `entryHash`) and `class AuditLog` with `AuditEntry append(String
  event, {Map<String, dynamic> payload = const {}, double? ts})`,
  `int get length`, `List<Map<String, dynamic>> entries()`,
  `(bool ok, int? brokenSeq) verifyChain()`. Consumed by Task 12 (bank modal)
  and Task 13 (audit log view).

Direct port of `app/components/audit_log.py` — same hash input shape (JSON
with sorted keys), same genesis hash, same tamper-detection contract.

- [ ] **Step 1: Write the failing tests**

```dart
// vaani/mobile/test/audit/audit_log_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/audit/audit_log.dart';

void main() {
  test('append chains prevHash to the previous entry hash', () {
    final log = AuditLog();
    final a = log.append('transfer_held', payload: {'amount_inr': 4000000});
    final b = log.append('otp_verified');
    expect(a.prevHash, '0' * 64);
    expect(b.prevHash, a.entryHash);
    expect(log.length, 2);
  });

  test('verifyChain reports intact chain as (true, null)', () {
    final log = AuditLog();
    log.append('a');
    log.append('b');
    final (ok, brokenSeq) = log.verifyChain();
    expect(ok, isTrue);
    expect(brokenSeq, isNull);
  });

  test('verifyChain detects a tampered payload', () {
    final log = AuditLog();
    log.append('a');
    final entry = log.append('b', payload: {'x': 1});
    entry.payload['x'] = 999; // mutate after the hash was computed
    final (ok, brokenSeq) = log.verifyChain();
    expect(ok, isFalse);
    expect(brokenSeq, entry.seq);
  });

  test('entries() returns hash alongside the same fields as Python\'s to_dict', () {
    final log = AuditLog();
    final e = log.append('transfer_approved', payload: {'amount_inr': 1});
    final dict = log.entries().single;
    expect(dict['seq'], e.seq);
    expect(dict['event'], 'transfer_approved');
    expect(dict['hash'], e.entryHash);
    expect(dict['prev_hash'], e.prevHash);
  });
}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `flutter test test/audit/audit_log_test.dart`
Expected: FAIL — `audit_log.dart` doesn't exist.

- [ ] **Step 3: Implement `AuditLog`**

```dart
// vaani/mobile/lib/audit/audit_log.dart
import 'dart:convert';
import 'package:crypto/crypto.dart';

/// Ported from app/components/audit_log.py: each entry embeds the SHA-256
/// fingerprint of the previous entry, so editing any past entry breaks
/// every hash after it. Single-writer, in-memory — a decision log, not a
/// distributed ledger (master plan §1 item 5: "our honest answer to the
/// Blockchain theme").
///
/// Built with List.filled rather than a string literal so the length (64,
/// matching a SHA-256 hex digest) is exact by construction, not by manual
/// counting.
final String kGenesisHash = List.filled(64, '0').join();

class AuditEntry {
  AuditEntry({
    required this.seq,
    required this.ts,
    required this.event,
    required this.payload,
    required this.prevHash,
  }) : entryHash = _computeHash(seq, ts, event, payload, prevHash);

  final int seq;
  final double ts;
  final String event;
  final Map<String, dynamic> payload;
  final String prevHash;
  final String entryHash;

  static String _computeHash(int seq, double ts, String event,
      Map<String, dynamic> payload, String prevHash) {
    final body = jsonEncode(_sortedMap({
      'seq': seq,
      'ts': ts,
      'event': event,
      'payload': _sortedMap(payload),
      'prev_hash': prevHash,
    }));
    return sha256.convert(utf8.encode(body)).toString();
  }

  /// json.dumps(..., sort_keys=True) equivalent: sort map keys recursively.
  static Map<String, dynamic> _sortedMap(Map<String, dynamic> m) {
    final sortedKeys = m.keys.toList()..sort();
    return {for (final k in sortedKeys) k: m[k]};
  }

  String recomputeHash() => _computeHash(seq, ts, event, payload, prevHash);

  Map<String, dynamic> toDict() => {
        'seq': seq,
        'ts': ts,
        'event': event,
        'payload': payload,
        'prev_hash': prevHash,
        'hash': entryHash,
      };
}

class AuditLog {
  final List<AuditEntry> _entries = [];

  int get length => _entries.length;

  AuditEntry append(String event,
      {Map<String, dynamic> payload = const {}, double? ts}) {
    final prevHash = _entries.isEmpty ? kGenesisHash : _entries.last.entryHash;
    final entry = AuditEntry(
      seq: _entries.length,
      ts: ts ?? DateTime.now().millisecondsSinceEpoch / 1000.0,
      event: event,
      payload: Map<String, dynamic>.from(payload),
      prevHash: prevHash,
    );
    _entries.add(entry);
    return entry;
  }

  List<Map<String, dynamic>> entries() =>
      _entries.map((e) => e.toDict()).toList();

  (bool, int?) verifyChain() {
    var prevHash = kGenesisHash;
    for (final entry in _entries) {
      if (entry.prevHash != prevHash) return (false, entry.seq);
      if (entry.recomputeHash() != entry.entryHash) return (false, entry.seq);
      prevHash = entry.entryHash;
    }
    return (true, null);
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `flutter test test/audit/audit_log_test.dart`
Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/audit vaani/mobile/test/audit
git commit -m "feat(mobile): port hash-chained AuditLog to Dart"
```

---

### Task 4: Scorer interface + StubScorer

**Files:**
- Create: `vaani/mobile/lib/scoring/scorer.dart`
- Create: `vaani/mobile/lib/scoring/stub_scorer.dart`
- Test: `vaani/mobile/test/scoring/stub_scorer_test.dart`

**Interfaces:**
- Produces: `abstract class Scorer { double scoreWindow(Float32List audio,
  int sr, double tStartS); String get backendLabel; }` and
  `class StubScorer implements Scorer` with constructor
  `StubScorer({double? cloneEntryS, int seed = 7})`. `backendLabel` returns
  `"stub (simulated — real model pending)"`. Consumed by Task 9
  (`WindowPipeline`) and Task 15 (`OnnxScorer` implements the same interface).

Ports `app/engine_mock.py:70-96`'s `ScoreBackend`/`MockBackend`. `cloneEntryS
== null` means "no known clone position" (the honest default for a real
user-imported file — always scores low); passing a value reproduces the
canned-demo behavior for the bundled demo asset (Task 13).

- [ ] **Step 1: Write the failing tests**

```dart
// vaani/mobile/test/scoring/stub_scorer_test.dart
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `flutter test test/scoring/stub_scorer_test.dart`
Expected: FAIL — files don't exist.

- [ ] **Step 3: Implement `Scorer` and `StubScorer`**

```dart
// vaani/mobile/lib/scoring/scorer.dart
import 'dart:typed_data';

/// One call site in WindowPipeline (Task 9) chooses which implementation
/// to use — StubScorer (Phase 1) or OnnxScorer (Phase 2, Task 15). Both
/// implement this same interface, so swapping is a one-line change.
abstract class Scorer {
  double scoreWindow(Float32List audio, int sr, double tStartS);

  /// Shown verbatim in the UI (gauge banner) so the app never implies real
  /// inference while it isn't running one. Honest numbers only.
  String get backendLabel;
}
```

```dart
// vaani/mobile/lib/scoring/stub_scorer.dart
import 'dart:math';
import 'dart:typed_data';
import 'scorer.dart';

/// Ported from app/engine_mock.py's MockBackend. `cloneEntryS == null`
/// means "no known clone position in this audio" — the honest default for
/// arbitrary user-imported files, which always scores low. Passing a
/// value reproduces the canned clone-entry-at-t demo behavior.
class StubScorer implements Scorer {
  StubScorer({this.cloneEntryS, int seed = 7}) : _rng = _SeededGaussian(seed);

  final double? cloneEntryS;
  final _SeededGaussian _rng;

  @override
  String get backendLabel => 'stub (simulated — real model pending)';

  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    final mid = tStartS + audio.length / sr / 2.0;
    double base;
    if (cloneEntryS == null) {
      base = 0.12;
    } else if (mid >= cloneEntryS!) {
      base = 0.85;
    } else if (mid >= cloneEntryS! - 0.25) {
      base = 0.55;
    } else {
      base = 0.12;
    }
    final sampled = _rng.next(base, 0.05);
    return sampled.clamp(0.0, 1.0);
  }
}

/// Deterministic seeded Gaussian sampler (Box-Muller), since dart:math's
/// Random has no built-in normal distribution. Determinism only needs to
/// hold within one StubScorer instance's lifetime (tests seed explicitly);
/// exact parity with numpy's RNG stream is not required.
class _SeededGaussian {
  _SeededGaussian(int seed) : _random = Random(seed);
  final Random _random;
  double? _spare;

  double next(double mean, double stdDev) {
    if (_spare != null) {
      final v = _spare!;
      _spare = null;
      return mean + stdDev * v;
    }
    double u, v, s;
    do {
      u = _random.nextDouble() * 2 - 1;
      v = _random.nextDouble() * 2 - 1;
      s = u * u + v * v;
    } while (s >= 1 || s == 0);
    final mul = sqrt(-2.0 * log(s) / s);
    _spare = v * mul;
    return mean + stdDev * (u * mul);
  }
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `flutter test test/scoring/stub_scorer_test.dart`
Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/scoring vaani/mobile/test/scoring
git commit -m "feat(mobile): add Scorer interface and StubScorer (Phase 1)"
```

---

### Task 5: MelBridge native Kotlin (FFT + mel filterbank)

**Files:**
- Modify: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt`
- Create: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MelBridge.kt`
- Test: `vaani/mobile/android/app/src/test/kotlin/org/vaani/mobile/MelBridgeTest.kt`

**Interfaces:**
- Produces: Kotlin object `MelBridge.computeMelDb(pcm: FloatArray, sr: Int):
  Array<DoubleArray>` (shape `[N_MELS][frames]`), registered on a
  `MethodChannel` named `"vaani/mel"` with method `"computeMelDb"` taking
  `{"pcm": FloatArray, "sr": Int}` and returning a flattened
  `List<List<Double>>`. Consumed by Task 6's Dart wrapper.

Ports `app/components/spectrogram.py`'s pure-numpy mel computation
(N_FFT=512, HOP=160, N_MELS=48) to pure Kotlin — same triangular filterbank,
same dB conversion, so the visual spectrogram matches the desktop demo's
math exactly. This is a JVM-testable pure-math module; no Android runtime
APIs are used, so its test runs as a plain JUnit test, not an instrumented
one.

- [ ] **Step 1: Write the failing JVM test**

```kotlin
// vaani/mobile/android/app/src/test/kotlin/org/vaani/mobile/MelBridgeTest.kt
package org.vaani.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class MelBridgeTest {
    @Test
    fun `output shape is N_MELS by expected frame count`() {
        val sr = 16000
        val durationS = 2.0
        val pcm = FloatArray((sr * durationS).toInt()) { 0f }
        val mel = MelBridge.computeMelDb(pcm, sr)
        assertEquals(48, mel.size) // N_MELS
        val expectedFrames = 1 + (pcm.size - 512) / 160
        assertEquals(expectedFrames, mel[0].size)
    }

    @Test
    fun `a pure tone concentrates energy in a narrow mel band`() {
        val sr = 16000
        val freq = 1000.0
        val n = sr // 1 s
        val pcm = FloatArray(n) { i -> sin(2 * PI * freq * i / sr).toFloat() }
        val mel = MelBridge.computeMelDb(pcm, sr)
        // Average energy per mel bin across all frames.
        val bandEnergy = mel.map { row -> row.average() }
        val maxBin = bandEnergy.indices.maxByOrNull { bandEnergy[it] }!!
        val totalAboveFloor = bandEnergy.count { it > bandEnergy[maxBin] - 10.0 }
        // The peak bin plus its immediate neighbors should dominate; a pure
        // tone should not spread energy evenly across all 48 bins.
        assertTrue("expected a concentrated peak, got $totalAboveFloor bins within 10dB of max",
            totalAboveFloor <= 6)
    }

    @Test
    fun `short input is zero-padded rather than throwing`() {
        val mel = MelBridge.computeMelDb(FloatArray(100) { 0f }, 16000)
        assertEquals(48, mel.size)
        assertEquals(1, mel[0].size)
    }
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd vaani/mobile/android && ./gradlew testDebugUnitTest --tests "org.vaani.mobile.MelBridgeTest"`
Expected: FAIL — compile error, `MelBridge` doesn't exist.

- [ ] **Step 3: Implement `MelBridge`**

```kotlin
// vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MelBridge.kt
package org.vaani.mobile

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin port of app/components/spectrogram.py::compute_mel_db.
 * A visual-aid mel spectrogram (triangular filterbank approximation), not
 * a model feature extractor — matches the Python original's math exactly
 * (N_FFT=512, HOP=160 i.e. 10ms @ 16kHz, N_MELS=48) so the two platforms'
 * spectrograms look identical.
 */
object MelBridge {
    private const val N_FFT = 512
    private const val HOP = 160
    private const val N_MELS = 48

    private var cachedSr: Int = -1
    private var cachedBank: Array<DoubleArray>? = null

    fun computeMelDb(pcm: FloatArray, sr: Int): Array<DoubleArray> {
        val padded = if (pcm.size < N_FFT) pcm.copyOf(N_FFT) else pcm
        val nFrames = 1 + (padded.size - N_FFT) / HOP
        val bank = melFilterbank(sr)
        val nBins = N_FFT / 2 + 1
        val melDb = Array(N_MELS) { DoubleArray(nFrames) }
        val window = hanningWindow(N_FFT)

        for (f in 0 until nFrames) {
            val start = f * HOP
            val frame = DoubleArray(N_FFT) { i -> padded[start + i] * window[i] }
            val mags = rfftMagnitude(frame) // size nBins
            for (m in 0 until N_MELS) {
                var acc = 0.0
                for (k in 0 until nBins) acc += bank[m][k] * mags[k]
                melDb[m][f] = 20.0 * log10(max(acc, 1e-10))
            }
        }
        return melDb
    }

    private fun hanningWindow(n: Int): DoubleArray =
        DoubleArray(n) { i -> 0.5 - 0.5 * cos(2 * PI * i / (n - 1)) }

    private fun hzToMel(f: Double) = 2595.0 * log10(1.0 + f / 700.0)
    private fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)

    private fun melFilterbank(sr: Int): Array<DoubleArray> {
        cachedBank?.let { if (cachedSr == sr) return it }
        val nBins = N_FFT / 2 + 1
        val melLo = hzToMel(0.0)
        val melHi = hzToMel(sr / 2.0)
        val mPts = DoubleArray(N_MELS + 2) { i -> melLo + (melHi - melLo) * i / (N_MELS + 1) }
        val hzPts = mPts.map { melToHz(it) }
        val bins = hzPts.map { it / sr * N_FFT }
        val bank = Array(N_MELS) { DoubleArray(nBins) }
        for (i in 0 until N_MELS) {
            val left = bins[i]; val center = bins[i + 1]; val right = bins[i + 2]
            for (k in 0 until nBins) {
                when {
                    k > left && k < center -> bank[i][k] = (k - left) / (center - left)
                    k in center.toInt().toDouble()..right && k < right -> bank[i][k] = (right - k) / (right - center)
                }
            }
        }
        cachedSr = sr
        cachedBank = bank
        return bank
    }

    /** Magnitude of the real FFT (rfft) of a real-valued frame, via a
     *  straightforward iterative radix-2 Cooley-Tukey FFT (N_FFT=512 is a
     *  power of two, so no Bluestein fallback is needed). */
    private fun rfftMagnitude(frame: DoubleArray): DoubleArray {
        val n = frame.size
        val re = frame.copyOf()
        val im = DoubleArray(n)
        fft(re, im)
        val nBins = n / 2 + 1
        return DoubleArray(nBins) { k -> sqrt(re[k] * re[k] + im[k] * im[k]) }
    }

    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        // bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                var tmp = re[i]; re[i] = re[j]; re[j] = tmp
                tmp = im[i]; im[i] = im[j]; im[j] = tmp
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wReal = cos(ang)
            val wImag = sin(ang)
            var i = 0
            while (i < n) {
                var curReal = 1.0
                var curImag = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]; val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curReal - im[i + k + len / 2] * curImag
                    val vIm = re[i + k + len / 2] * curImag + im[i + k + len / 2] * curReal
                    re[i + k] = uRe + vRe; im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm
                    val nextReal = curReal * wReal - curImag * wImag
                    val nextImag = curReal * wImag + curImag * wReal
                    curReal = nextReal; curImag = nextImag
                }
                i += len
            }
            len = len shl 1
        }
    }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd vaani/mobile/android && ./gradlew testDebugUnitTest --tests "org.vaani.mobile.MelBridgeTest"`
Expected: PASS (3/3). If the filterbank triangular-edge test at `k ==
center` fails an edge case, adjust the `when` branch boundaries to match
`spectrogram.py`'s `elif center <= k < right` exactly (`>=` not `>` at the
center boundary) and rerun.

- [ ] **Step 5: Register the MethodChannel in `MainActivity.kt`**

```kotlin
// vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt
package org.vaani.mobile

import io.flutter.embedding.android.FlutterActivity
import io.flutter.embedding.engine.FlutterEngine
import io.flutter.plugin.common.MethodChannel

class MainActivity : FlutterActivity() {
    override fun configureFlutterEngine(flutterEngine: FlutterEngine) {
        super.configureFlutterEngine(flutterEngine)
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mel")
            .setMethodCallHandler { call, result ->
                if (call.method == "computeMelDb") {
                    @Suppress("UNCHECKED_CAST")
                    val pcm = (call.argument<List<Double>>("pcm"))!!
                        .map { it.toFloat() }.toFloatArray()
                    val sr = call.argument<Int>("sr")!!
                    val mel = MelBridge.computeMelDb(pcm, sr)
                    result.success(mel.map { it.toList() })
                } else {
                    result.notImplemented()
                }
            }
    }
}
```

- [ ] **Step 6: Commit**

```bash
git add vaani/mobile/android/app/src/main/kotlin vaani/mobile/android/app/src/test
git commit -m "feat(mobile): port mel spectrogram math to native Kotlin MelBridge"
```

---

### Task 6: Dart wrapper for MelBridge

**Files:**
- Create: `vaani/mobile/lib/mel/mel_bridge.dart`
- Test: `vaani/mobile/test/mel/mel_bridge_test.dart`

**Interfaces:**
- Consumes: Task 5's `"vaani/mel"` MethodChannel, method `"computeMelDb"`.
- Produces: `class MelBridge { Future<List<List<double>>> computeMelDb(
  Float32List pcm, int sr); }`. Consumed by Task 9 (`WindowPipeline`) and
  Task 11 (`SpectrogramView`).

- [ ] **Step 1: Write the failing test (mocked channel, no device needed)**

```dart
// vaani/mobile/test/mel/mel_bridge_test.dart
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/mel/mel_bridge.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('vaani/mel');

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('computeMelDb marshals pcm/sr and returns the mocked matrix', () async {
    Map<String, dynamic>? capturedArgs;
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      capturedArgs = Map<String, dynamic>.from(call.arguments as Map);
      return [
        [1.0, 2.0],
        [3.0, 4.0],
      ];
    });

    final bridge = MelBridge();
    final result = await bridge.computeMelDb(Float32List.fromList([0.1, 0.2]), 16000);

    expect(capturedArgs!['sr'], 16000);
    expect((capturedArgs!['pcm'] as List).length, 2);
    expect(result, [
      [1.0, 2.0],
      [3.0, 4.0],
    ]);
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `flutter test test/mel/mel_bridge_test.dart`
Expected: FAIL — `mel_bridge.dart` doesn't exist.

- [ ] **Step 3: Implement the wrapper**

```dart
// vaani/mobile/lib/mel/mel_bridge.dart
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Dart side of MainActivity.kt's "vaani/mel" MethodChannel (Task 5).
class MelBridge {
  static const _channel = MethodChannel('vaani/mel');

  Future<List<List<double>>> computeMelDb(Float32List pcm, int sr) async {
    final raw = await _channel.invokeMethod<List<dynamic>>('computeMelDb', {
      'pcm': pcm.toList(),
      'sr': sr,
    });
    return raw!
        .map((row) => (row as List).map((v) => (v as num).toDouble()).toList())
        .toList();
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `flutter test test/mel/mel_bridge_test.dart`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/mel vaani/mobile/test/mel
git commit -m "feat(mobile): add Dart wrapper for the native MelBridge channel"
```

---

### Task 7: Audio file decode (import path)

**Files:**
- Create: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/AudioDecodeBridge.kt`
- Modify: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt`
- Create: `vaani/mobile/lib/capture/audio_decode_bridge.dart`
- Create: `vaani/mobile/assets/test_fixtures/tone_1s_16k.wav` (generated below)
- Test: `vaani/mobile/integration_test/file_import_flow_test.dart`

**Interfaces:**
- Produces: Kotlin `AudioDecodeBridge.decodeToPcm16kMono(path: String):
  FloatArray`, exposed on MethodChannel `"vaani/decode"`, method
  `"decodeFile"`, args `{"path": String}`, returning
  `List<Double>` (the PCM samples). Dart wrapper:
  `class AudioDecodeBridge { Future<Float32List> decodeFile(String path); }`.
  Consumed by Task 9 and Task 13.

Uses Android's built-in `MediaExtractor`/`MediaCodec` to decode any format
the OS supports (WAV, MP3, M4A, OGG — whatever the phone's codecs handle),
then resamples to mono 16 kHz float PCM the same way `server.py`'s
`_load_mono_float` does (channel-average, linear resample).

- [ ] **Step 1: Generate a tiny WAV test fixture**

Run once, from `vaani/mobile/`, to create a 1 s 440 Hz tone at 16 kHz mono
16-bit PCM (used by the integration test — this fixture is checked in, not
regenerated per test run):
```bash
python -c "
import wave, struct, math
sr = 16000
with wave.open('assets/test_fixtures/tone_1s_16k.wav', 'w') as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
    frames = b''.join(
        struct.pack('<h', int(3000 * math.sin(2*math.pi*440*i/sr)))
        for i in range(sr)
    )
    w.writeframes(frames)
"
```
Add to `pubspec.yaml` under `flutter:`:
```yaml
flutter:
  assets:
    - assets/test_fixtures/tone_1s_16k.wav
```

- [ ] **Step 2: Write the failing integration test**

```dart
// vaani/mobile/integration_test/file_import_flow_test.dart
import 'dart:io';
import 'package:flutter/services.dart' show rootBundle;
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/capture/audio_decode_bridge.dart';
import 'package:path_provider/path_provider.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('decodeFile returns ~1s of 16kHz mono PCM for the tone fixture',
      (tester) async {
    final bytes = await rootBundle.load('assets/test_fixtures/tone_1s_16k.wav');
    final dir = await getTemporaryDirectory();
    final file = File('${dir.path}/tone_1s_16k.wav');
    await file.writeAsBytes(bytes.buffer.asUint8List());

    final bridge = AudioDecodeBridge();
    final pcm = await bridge.decodeFile(file.path);

    // Allow +/- one decoder frame of slack around the expected 16000 samples.
    expect(pcm.length, greaterThan(15000));
    expect(pcm.length, lessThan(17000));
    expect(pcm.reduce((a, b) => a.abs() > b.abs() ? a : b).abs(), greaterThan(0.01));
  });
}
```

Add `path_provider: ^2.1.4` to `pubspec.yaml` dependencies (needed to get a
writable temp path for the decoder to read from).

- [ ] **Step 3: Run the test to verify it fails**

Run: `flutter test integration_test/file_import_flow_test.dart -d <connected-android-device-or-emulator>`
Expected: FAIL — `audio_decode_bridge.dart` doesn't exist.

- [ ] **Step 4: Implement the native decoder**

```kotlin
// vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/AudioDecodeBridge.kt
package org.vaani.mobile

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decodes any Android-supported audio file (WAV/MP3/M4A/OGG/...) to mono
 *  16kHz float PCM in [-1, 1], mirroring server.py's _load_mono_float
 *  (channel-average, linear resample) so both platforms treat imported
 *  audio identically. */
object AudioDecodeBridge {
    private const val TARGET_SR = 16000

    fun decodeToPcm16kMono(path: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                trackIndex = i; format = f; break
            }
        }
        require(trackIndex >= 0 && format != null) { "no audio track found in $path" }
        extractor.selectTrack(trackIndex)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val sourceChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val sourceSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)

        val pcmOut = ArrayList<Short>()
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inIndex = codec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val inBuffer = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }
            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            if (outIndex >= 0) {
                val outBuffer = codec.getOutputBuffer(outIndex)!!
                val chunk = ByteArray(bufferInfo.size)
                outBuffer.get(chunk)
                outBuffer.clear()
                val shortBuf = ByteBuffer.wrap(chunk).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
                while (shortBuf.hasRemaining()) pcmOut.add(shortBuf.get())
                codec.releaseOutputBuffer(outIndex, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) sawOutputEOS = true
            }
        }
        codec.stop(); codec.release(); extractor.release()

        // Downmix to mono.
        val mono = if (sourceChannels > 1) {
            FloatArray(pcmOut.size / sourceChannels) { i ->
                var sum = 0f
                for (c in 0 until sourceChannels) sum += pcmOut[i * sourceChannels + c] / 32768f
                sum / sourceChannels
            }
        } else {
            FloatArray(pcmOut.size) { i -> pcmOut[i] / 32768f }
        }

        // Linear resample to TARGET_SR, matching server.py's np.interp approach.
        if (sourceSr == TARGET_SR) return mono
        val nOut = (mono.size.toLong() * TARGET_SR / sourceSr).toInt()
        return FloatArray(nOut) { i ->
            val srcPos = i.toDouble() * (mono.size - 1) / (nOut - 1).coerceAtLeast(1)
            val lo = srcPos.toInt().coerceIn(0, mono.size - 1)
            val hi = (lo + 1).coerceAtMost(mono.size - 1)
            val frac = srcPos - lo
            (mono[lo] * (1 - frac) + mono[hi] * frac).toFloat()
        }
    }
}
```

- [ ] **Step 5: Register the MethodChannel**

Add to `MainActivity.kt`'s `configureFlutterEngine` (alongside the existing
`"vaani/mel"` channel from Task 5):

```kotlin
        MethodChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/decode")
            .setMethodCallHandler { call, result ->
                if (call.method == "decodeFile") {
                    val path = call.argument<String>("path")!!
                    try {
                        val pcm = AudioDecodeBridge.decodeToPcm16kMono(path)
                        result.success(pcm.map { it.toDouble() })
                    } catch (e: Exception) {
                        result.error("DECODE_FAILED", e.message, null)
                    }
                } else {
                    result.notImplemented()
                }
            }
```

- [ ] **Step 6: Implement the Dart wrapper**

```dart
// vaani/mobile/lib/capture/audio_decode_bridge.dart
import 'dart:typed_data';
import 'package:flutter/services.dart';

/// Thrown when the OS decoder can't open the file (corrupt, unsupported
/// format) — the caller must surface this to the user, never treat it as
/// a silent zero-score (spec §5: "never a silent zero score").
class AudioDecodeException implements Exception {
  AudioDecodeException(this.message);
  final String message;
  @override
  String toString() => 'AudioDecodeException: $message';
}

class AudioDecodeBridge {
  static const _channel = MethodChannel('vaani/decode');

  Future<Float32List> decodeFile(String path) async {
    try {
      final raw = await _channel.invokeMethod<List<dynamic>>('decodeFile', {'path': path});
      return Float32List.fromList(raw!.map((v) => (v as num).toDouble()).toList());
    } on PlatformException catch (e) {
      throw AudioDecodeException(e.message ?? 'unknown decode failure');
    }
  }
}
```

Add `path_provider` (already added in Step 2) to `pubspec.yaml` if not
already present, and run `flutter pub get`.

- [ ] **Step 7: Run the test to verify it passes**

Run: `flutter test integration_test/file_import_flow_test.dart -d <connected-android-device-or-emulator>`
Expected: PASS. (Requires a device/emulator — this is instrumented, not a
plain JVM/Dart unit test, since `MediaExtractor`/`MediaCodec` only exist on
a real Android runtime.)

- [ ] **Step 8: Commit**

```bash
git add vaani/mobile/android/app/src/main/kotlin vaani/mobile/lib/capture \
        vaani/mobile/integration_test/file_import_flow_test.dart \
        vaani/mobile/assets/test_fixtures/tone_1s_16k.wav vaani/mobile/pubspec.yaml
git commit -m "feat(mobile): decode imported audio files to 16kHz mono PCM"
```

---

### Task 8: Live mic capture

**Files:**
- Create: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/AudioCaptureBridge.kt`
- Modify: `vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/MainActivity.kt`
- Modify: `vaani/mobile/android/app/src/main/AndroidManifest.xml`
- Create: `vaani/mobile/lib/capture/audio_capture_bridge.dart`
- Test: `vaani/mobile/integration_test/mic_capture_flow_test.dart`

**Interfaces:**
- Produces: Kotlin class streaming raw 16-bit mono 16 kHz PCM chunks (0.5 s /
  8000 samples each) over EventChannel `"vaani/mic"`. Dart wrapper:
  `class AudioCaptureBridge { Future<bool> requestPermission(); Stream<
  Float32List> start(); Future<void> stop(); }`. Consumed by Task 9 and
  Task 13.

- [ ] **Step 1: Add the mic permission to the manifest**

In `vaani/mobile/android/app/src/main/AndroidManifest.xml`, inside
`<manifest>`, above `<application>`:
```xml
    <uses-permission android:name="android.permission.RECORD_AUDIO" />
```

- [ ] **Step 2: Write the failing integration test**

```dart
// vaani/mobile/integration_test/mic_capture_flow_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/capture/audio_capture_bridge.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('start() emits 0.5s-ish chunks of the expected sample count',
      (tester) async {
    final bridge = AudioCaptureBridge();
    final granted = await bridge.requestPermission();
    expect(granted, isTrue, reason: 'test device must pre-grant RECORD_AUDIO');

    final chunks = <int>[];
    final sub = bridge.start().listen((chunk) => chunks.add(chunk.length));
    await Future.delayed(const Duration(seconds: 2));
    await bridge.stop();
    await sub.cancel();

    expect(chunks.length, greaterThanOrEqualTo(2));
    for (final len in chunks) {
      expect(len, closeTo(8000, 800)); // 0.5s @ 16kHz, +/-10% jitter allowed
    }
  });
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `flutter test integration_test/mic_capture_flow_test.dart -d <device>`
Expected: FAIL — `audio_capture_bridge.dart` doesn't exist.

- [ ] **Step 4: Implement the native capture bridge**

```kotlin
// vaani/mobile/android/app/src/main/kotlin/org/vaani/mobile/AudioCaptureBridge.kt
package org.vaani.mobile

import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import io.flutter.plugin.common.EventChannel

/** Streams the phone's own mic as 0.5s (8000-sample) mono 16kHz PCM chunks
 *  over an EventChannel. Captures only the device's own microphone — this
 *  is legal and unrestricted, but for a VoIP call it captures whatever
 *  reaches the room mic, not the app's internal audio stream (spec §1). */
class AudioCaptureBridge : EventChannel.StreamHandler {
    companion object {
        const val SAMPLE_RATE = 16000
        const val CHUNK_SAMPLES = SAMPLE_RATE / 2 // 0.5s
    }

    private var recorder: AudioRecord? = null
    @Volatile private var running = false
    private var thread: Thread? = null

    override fun onListen(arguments: Any?, events: EventChannel.EventSink) {
        val minBuf = AudioRecord.getMinBufferSize(
            SAMPLE_RATE, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT
        )
        val bufSize = maxOf(minBuf, CHUNK_SAMPLES * 4)
        recorder = AudioRecord(
            MediaRecorder.AudioSource.MIC, SAMPLE_RATE,
            AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT, bufSize
        )
        recorder!!.startRecording()
        running = true
        thread = Thread {
            val buf = FloatArray(CHUNK_SAMPLES)
            while (running) {
                val read = recorder!!.read(buf, 0, CHUNK_SAMPLES, AudioRecord.READ_BLOCKING)
                if (read > 0) {
                    val chunk = if (read == CHUNK_SAMPLES) buf.toList() else buf.copyOf(read).toList()
                    events.success(chunk)
                }
            }
        }
        thread!!.start()
    }

    override fun onCancel(arguments: Any?) {
        running = false
        thread?.join(500)
        recorder?.stop()
        recorder?.release()
        recorder = null
    }
}
```

- [ ] **Step 5: Register the EventChannel**

Add to `MainActivity.kt`'s `configureFlutterEngine`:
```kotlin
        EventChannel(flutterEngine.dartExecutor.binaryMessenger, "vaani/mic")
            .setStreamHandler(AudioCaptureBridge())
```

- [ ] **Step 6: Implement the Dart wrapper**

```dart
// vaani/mobile/lib/capture/audio_capture_bridge.dart
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:permission_handler/permission_handler.dart';

class AudioCaptureBridge {
  static const _channel = EventChannel('vaani/mic');
  StreamSubscription? _sub;

  Future<bool> requestPermission() async {
    final status = await Permission.microphone.request();
    return status.isGranted;
  }

  Stream<Float32List> start() {
    return _channel.receiveBroadcastStream().map((event) {
      final list = (event as List).map((v) => (v as num).toDouble()).toList();
      return Float32List.fromList(list);
    });
  }

  Future<void> stop() async {
    await _sub?.cancel();
    _sub = null;
  }
}
```

(`StreamSubscription` requires `import 'dart:async';` — add it alongside the
other imports.)

- [ ] **Step 7: Run the test to verify it passes**

Run: `flutter test integration_test/mic_capture_flow_test.dart -d <device>`
Expected: PASS. Requires a device/emulator with a virtual mic input (Android
emulators synthesize a tone/silence on `MIC` by default, which is enough to
verify chunk sizing).

- [ ] **Step 8: Commit**

```bash
git add vaani/mobile/android/app/src/main/kotlin \
        vaani/mobile/android/app/src/main/AndroidManifest.xml \
        vaani/mobile/lib/capture/audio_capture_bridge.dart \
        vaani/mobile/integration_test/mic_capture_flow_test.dart
git commit -m "feat(mobile): stream live mic audio as 0.5s PCM chunks"
```

---

### Task 9: WindowPipeline (orchestration)

**Files:**
- Create: `vaani/mobile/lib/pipeline/frame_result.dart`
- Create: `vaani/mobile/lib/pipeline/window_pipeline.dart`
- Test: `vaani/mobile/test/pipeline/window_pipeline_test.dart`

**Interfaces:**
- Consumes: `Scorer` (Task 4), `MelBridge` (Task 6) — both injected, so this
  is unit-testable with fakes, no platform channels involved.
- Produces: `class FrameResult { final double t; final double rawScore;
  final double ema; final AlertState state; final List<List<double>> melDb;
  final Float32List audioWindow; }` and `class WindowPipeline` with
  `Stream<FrameResult> process(Stream<Float32List> hopChunks, {required
  Scorer scorer})`. Consumed by Task 13 (`HomeScreen`).

Assembles 0.5 s hop chunks (from either capture path) into a 2 s sliding
window, mirroring `server.py`'s "only score full windows... emulate the real
engine's first-score latency of ~2.5s" comment — the first `FrameResult` is
only emitted once 4 hop chunks (2 s) have accumulated.

- [ ] **Step 1: Write the failing test**

```dart
// vaani/mobile/test/pipeline/window_pipeline_test.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/mel/mel_bridge.dart';
import 'package:mobile/pipeline/window_pipeline.dart';
import 'package:mobile/scoring/scorer.dart';

class _FakeScorer implements Scorer {
  final List<double> calls = [];
  @override
  String get backendLabel => 'fake';
  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    calls.add(tStartS);
    return 0.9;
  }
}

class _FakeMel implements MelBridgeLike {
  @override
  Future<List<List<double>>> computeMelDb(Float32List pcm, int sr) async => [
        [0.0]
      ];
}

void main() {
  test('first FrameResult only fires after 2s (4 hop chunks) accumulate', () async {
    final scorer = _FakeScorer();
    final pipeline = WindowPipeline(mel: _FakeMel());
    final controller = StreamController<Float32List>();
    final results = <FrameResult>[];
    final sub = pipeline.process(controller.stream, scorer: scorer).listen(results.add);

    for (var i = 0; i < 3; i++) {
      controller.add(Float32List(8000)); // 0.5s each, sr assumed 16000
      await Future.delayed(Duration.zero);
    }
    expect(results, isEmpty, reason: 'only 1.5s buffered so far');

    controller.add(Float32List(8000)); // 4th chunk -> 2.0s
    await Future.delayed(Duration.zero);
    expect(results.length, 1);
    expect(scorer.calls.single, closeTo(0.0, 1e-9));

    await controller.close();
    await sub.cancel();
  });

  test('state machine and ema are threaded across windows', () async {
    final pipeline = WindowPipeline(mel: _FakeMel());
    final controller = StreamController<Float32List>();
    final results = <FrameResult>[];
    final sub = pipeline
        .process(controller.stream, scorer: _FakeScorer())
        .listen(results.add);

    for (var i = 0; i < 8; i++) {
      controller.add(Float32List(8000));
      await Future.delayed(Duration.zero);
    }
    expect(results.length, 5); // chunks 4,5,6,7,8 each complete a new window
    expect(results.last.state.toString(), contains('alert')); // score 0.9 repeatedly
    await controller.close();
    await sub.cancel();
  });
}
```

Add `abstract class MelBridgeLike { Future<List<List<double>>> computeMelDb(
Float32List pcm, int sr); }` to `lib/mel/mel_bridge.dart` and make
`MelBridge implements MelBridgeLike`, so `WindowPipeline` can accept either
the real bridge or a test fake.

- [ ] **Step 2: Run the test to verify it fails**

Run: `flutter test test/pipeline/window_pipeline_test.dart`
Expected: FAIL — `window_pipeline.dart`/`frame_result.dart` don't exist.

- [ ] **Step 3: Implement `FrameResult` and `WindowPipeline`**

```dart
// vaani/mobile/lib/pipeline/frame_result.dart
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
```

```dart
// vaani/mobile/lib/pipeline/window_pipeline.dart
import 'dart:async';
import 'dart:typed_data';
import '../decision/decision_engine.dart';
import '../mel/mel_bridge.dart';
import '../scoring/scorer.dart';
import 'frame_result.dart';

const int _sampleRate = 16000;
const int _hopSamples = _sampleRate ~/ 2; // 0.5s
const int _windowSamples = _sampleRate * 2; // 2.0s

/// Assembles 0.5s hop chunks (from file import or mic capture) into a 2s
/// sliding window, scores each completed window, and threads the result
/// through DecisionEngine. Mirrors server.py's "only score full windows...
/// first-score latency of ~2.5s" behavior.
class WindowPipeline {
  WindowPipeline({required MelBridgeLike mel}) : _mel = mel;

  final MelBridgeLike _mel;
  final List<double> _buffer = [];
  final DecisionEngine _engine = DecisionEngine();
  double _elapsedS = 0.0;
  bool _first = true;

  Stream<FrameResult> process(Stream<Float32List> hopChunks,
      {required Scorer scorer}) async* {
    await for (final chunk in hopChunks) {
      _buffer.addAll(chunk);
      if (_buffer.length < _windowSamples) continue;

      final window = Float32List.fromList(
          _buffer.sublist(_buffer.length - _windowSamples));
      final tStart = _first ? 0.0 : _elapsedS - 1.5; // window start = now - 2s + 0.5s hop already counted
      final raw = scorer.scoreWindow(window, _sampleRate, tStart);
      final state = _engine.update(raw);
      final melDb = await _mel.computeMelDb(window, _sampleRate);

      yield FrameResult(
        t: _elapsedS,
        rawScore: raw,
        ema: _engine.ema!,
        state: state,
        melDb: melDb,
        audioWindow: window,
      );

      _elapsedS += 0.5;
      _first = false;
      // Keep only the last window's worth of samples to bound memory.
      if (_buffer.length > _windowSamples) {
        _buffer.removeRange(0, _buffer.length - _windowSamples);
      }
    }
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `flutter test test/pipeline/window_pipeline_test.dart`
Expected: PASS (2/2). If the `tStart` bookkeeping produces an off-by-0.5s
value against the first test's `closeTo(0.0, ...)` assertion, adjust
`_elapsedS`/`tStart` tracking so `tStart` equals the timestamp of the
window's first sample (i.e. `_elapsedS - 1.5` before incrementing) — re-run
until both assertions hold.

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/pipeline vaani/mobile/lib/mel/mel_bridge.dart \
        vaani/mobile/test/pipeline
git commit -m "feat(mobile): add WindowPipeline orchestrating capture->score->decision"
```

---

### Task 10: Gauge widget

**Files:**
- Create: `vaani/mobile/lib/ui/gauge.dart`
- Test: `vaani/mobile/test/ui/gauge_test.dart`

**Interfaces:**
- Produces: `class RiskGauge extends StatelessWidget` with constructor
  `RiskGauge({required double? ema, required AlertState state, double?
  raw, required String backendLabel})`. Consumed by Task 13.

Ports `app/components/gauge.py`'s colors/labels exactly.

- [ ] **Step 1: Write the failing widget test**

```dart
// vaani/mobile/test/ui/gauge_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/decision/decision_engine.dart';
import 'package:mobile/ui/gauge.dart';

void main() {
  testWidgets('alert state shows the ALERT label and backend banner', (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: RiskGauge(ema: 0.92, state: AlertState.alert, raw: 0.95, backendLabel: 'stub (simulated)'),
    ));
    expect(find.textContaining('ALERT'), findsOneWidget);
    expect(find.textContaining('stub (simulated)'), findsOneWidget);
    expect(find.textContaining('92%'), findsOneWidget);
  });

  testWidgets('null ema renders 0% without crashing', (tester) async {
    await tester.pumpWidget(const MaterialApp(
      home: RiskGauge(ema: null, state: AlertState.normal, backendLabel: 'stub'),
    ));
    expect(find.textContaining('0%'), findsOneWidget);
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `flutter test test/ui/gauge_test.dart`
Expected: FAIL — `gauge.dart` doesn't exist.

- [ ] **Step 3: Implement `RiskGauge`**

```dart
// vaani/mobile/lib/ui/gauge.dart
import 'package:flutter/material.dart';
import '../decision/decision_engine.dart';

const _stateColors = {
  AlertState.normal: Color(0xFF22C55E),
  AlertState.warn: Color(0xFFF59E0B),
  AlertState.alert: Color(0xFFEF4444),
};
const _stateLabels = {
  AlertState.normal: 'NORMAL',
  AlertState.warn: 'WATCH',
  AlertState.alert: 'ALERT — 2+ consecutive windows above threshold',
};

class RiskGauge extends StatelessWidget {
  const RiskGauge({
    super.key,
    required this.ema,
    required this.state,
    this.raw,
    required this.backendLabel,
  });

  final double? ema;
  final AlertState state;
  final double? raw;
  final String backendLabel;

  @override
  Widget build(BuildContext context) {
    final score = ema ?? 0.0;
    final color = _stateColors[state]!;
    final pct = (score * 100).round();
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      decoration: BoxDecoration(
        border: Border.all(color: color, width: 2),
        borderRadius: BorderRadius.circular(16),
        color: const Color(0xFF0F172A),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text('Synthetic-voice risk',
                  style: TextStyle(color: Color(0xFFE2E8F0), fontWeight: FontWeight.w600)),
              Container(
                padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 2),
                decoration: BoxDecoration(
                  border: Border.all(color: const Color(0xFF475569)),
                  borderRadius: BorderRadius.circular(999),
                ),
                child: Text('backend: $backendLabel',
                    style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 12)),
              ),
            ],
          ),
          Text('$pct%',
              style: TextStyle(color: color, fontSize: 48, fontWeight: FontWeight.w800)),
          ClipRRect(
            borderRadius: BorderRadius.circular(8),
            child: LinearProgressIndicator(
              value: pct / 100.0,
              minHeight: 14,
              backgroundColor: const Color(0xFF1E293B),
              valueColor: AlwaysStoppedAnimation(color),
            ),
          ),
          const SizedBox(height: 8),
          Text(_stateLabels[state]!, style: TextStyle(color: color, fontWeight: FontWeight.w600)),
          if (raw != null)
            Text('raw window: ${raw!.toStringAsFixed(2)}',
                style: const TextStyle(color: Color(0xFF94A3B8), fontSize: 13)),
        ],
      ),
    );
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `flutter test test/ui/gauge_test.dart`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/ui/gauge.dart vaani/mobile/test/ui/gauge_test.dart
git commit -m "feat(mobile): port risk gauge UI from Streamlit demo"
```

---

### Task 11: Spectrogram, risk curve, occlusion overlay widgets

**Files:**
- Create: `vaani/mobile/lib/ui/spectrogram_view.dart`
- Create: `vaani/mobile/lib/ui/risk_curve.dart`
- Create: `vaani/mobile/lib/ui/occlusion_overlay.dart`
- Test: `vaani/mobile/test/ui/spectrogram_view_test.dart`
- Test: `vaani/mobile/test/ui/risk_curve_test.dart`

**Interfaces:**
- Produces: `class SpectrogramView extends StatelessWidget({required
  List<List<double>> melDb})`; `class RiskCurve extends StatelessWidget({
  required List<double> emaHistory, required double threshold})`;
  `class OcclusionOverlay extends StatelessWidget({required List<
  (double start, double end)> highlights, required double windowDurationS})`.
  Consumed by Task 13.

Occlusion sensitivity itself (hiding 0.5s slices and measuring the alarm
drop) is a scoring-side computation that only makes sense once a real model
exists — Module E's job in Phase 1 is the display widget only, fed by
whatever highlight ranges a future `OcclusionScorer` (out of scope for this
plan) produces. For Phase 1, `HomeScreen` (Task 13) passes an empty
highlight list, and the widget must render an empty state without crashing.

- [ ] **Step 1: Write the failing tests**

```dart
// vaani/mobile/test/ui/spectrogram_view_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/ui/spectrogram_view.dart';

void main() {
  testWidgets('renders a CustomPaint for a non-empty mel matrix', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: SpectrogramView(melDb: List.generate(48, (_) => [1.0, 2.0, 3.0])),
    ));
    expect(find.byType(CustomPaint), findsWidgets);
  });

  testWidgets('shows a waiting message for an empty mel matrix', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: SpectrogramView(melDb: [])));
    expect(find.textContaining('Waiting'), findsOneWidget);
  });
}
```

```dart
// vaani/mobile/test/ui/risk_curve_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/ui/risk_curve.dart';

void main() {
  testWidgets('renders a chart for non-empty history', (tester) async {
    await tester.pumpWidget(MaterialApp(
      home: RiskCurve(emaHistory: [0.1, 0.2, 0.9], threshold: 0.6),
    ));
    expect(find.byType(RiskCurve), findsOneWidget);
  });

  testWidgets('empty history does not crash', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: RiskCurve(emaHistory: [], threshold: 0.6)));
    expect(tester.takeException(), isNull);
  });
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `flutter test test/ui/spectrogram_view_test.dart test/ui/risk_curve_test.dart`
Expected: FAIL — files don't exist.

- [ ] **Step 3: Implement the three widgets**

```dart
// vaani/mobile/lib/ui/spectrogram_view.dart
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
```

```dart
// vaani/mobile/lib/ui/risk_curve.dart
import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';

class RiskCurve extends StatelessWidget {
  const RiskCurve({super.key, required this.emaHistory, required this.threshold});
  final List<double> emaHistory;
  final double threshold;

  @override
  Widget build(BuildContext context) {
    if (emaHistory.isEmpty) {
      return const SizedBox(height: 120, child: Center(child: Text('No data yet')));
    }
    final spots = [
      for (var i = 0; i < emaHistory.length; i++) FlSpot(i.toDouble(), emaHistory[i])
    ];
    return SizedBox(
      height: 160,
      child: LineChart(
        LineChartData(
          minY: 0,
          maxY: 1,
          extraLinesData: ExtraLinesData(horizontalLines: [
            HorizontalLine(y: threshold, color: Colors.amber, strokeWidth: 1),
          ]),
          lineBarsData: [
            LineChartBarData(spots: spots, isCurved: false, dotData: const FlDotData(show: false)),
          ],
        ),
      ),
    );
  }
}
```

```dart
// vaani/mobile/lib/ui/occlusion_overlay.dart
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `flutter test test/ui/spectrogram_view_test.dart test/ui/risk_curve_test.dart`
Expected: PASS (4/4).

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/ui/spectrogram_view.dart vaani/mobile/lib/ui/risk_curve.dart \
        vaani/mobile/lib/ui/occlusion_overlay.dart vaani/mobile/test/ui
git commit -m "feat(mobile): add spectrogram, risk curve, and occlusion widgets"
```

---

### Task 12: Bank HOLD→OTP→release modal + audit wiring

**Files:**
- Create: `vaani/mobile/lib/ui/bank_modal.dart`
- Test: `vaani/mobile/test/ui/bank_modal_logic_test.dart`

**Interfaces:**
- Consumes: `AuditLog` (Task 3), `AlertState` (Task 2).
- Produces: `String decideTransferOutcome(AlertState state)` (pure
  function) and `class BankPanel extends StatefulWidget({required
  AlertState state, required double? ema, required AuditLog auditLog})`.
  Consumed by Task 13.

Ports `app/components/bank_modal.py`'s pure decision function and workflow
stages (`idle` → `held` → `released`) exactly, including the disclosed demo
OTP code.

- [ ] **Step 1: Write the failing test for the pure decision function**

```dart
// vaani/mobile/test/ui/bank_modal_logic_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/decision/decision_engine.dart';
import 'package:mobile/ui/bank_modal.dart';

void main() {
  test('alert state holds the transfer', () {
    expect(decideTransferOutcome(AlertState.alert), 'held');
  });

  test('normal and warn states approve the transfer', () {
    expect(decideTransferOutcome(AlertState.normal), 'approved');
    expect(decideTransferOutcome(AlertState.warn), 'approved');
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `flutter test test/ui/bank_modal_logic_test.dart`
Expected: FAIL — `bank_modal.dart` doesn't exist.

- [ ] **Step 3: Implement `decideTransferOutcome` and `BankPanel`**

```dart
// vaani/mobile/lib/ui/bank_modal.dart
import 'package:flutter/material.dart';
import '../audit/audit_log.dart';
import '../decision/decision_engine.dart';

/// Simulated only: no real SMS is ever sent (disclosed on-screen). Ported
/// from app/components/bank_modal.py.
const String kDemoOtpCode = '123456';

/// Pure decision function (master plan §1 item 4: "holds transactions...
/// never silently blocks"). Only ever returns "held" or "approved".
String decideTransferOutcome(AlertState state) =>
    state == AlertState.alert ? 'held' : 'approved';

class BankPanel extends StatefulWidget {
  const BankPanel({super.key, required this.state, required this.ema, required this.auditLog});
  final AlertState state;
  final double? ema;
  final AuditLog auditLog;

  @override
  State<BankPanel> createState() => _BankPanelState();
}

enum _Stage { idle, held, released }

class _BankPanelState extends State<BankPanel> {
  _Stage _stage = _Stage.idle;
  String _releaseReason = 'approved';
  final _otpController = TextEditingController();

  void _initiateTransfer() {
    final outcome = decideTransferOutcome(widget.state);
    setState(() {
      if (outcome == 'held') {
        _stage = _Stage.held;
        widget.auditLog.append('transfer_held',
            payload: {'amount_inr': 4000000, 'ema_at_hold': widget.ema, 'state': widget.state.name});
      } else {
        _stage = _Stage.released;
        _releaseReason = 'approved';
        widget.auditLog.append('transfer_approved',
            payload: {'amount_inr': 4000000, 'ema_at_approval': widget.ema, 'state': widget.state.name});
      }
    });
  }

  void _verifyOtp() {
    if (_otpController.text == kDemoOtpCode) {
      setState(() {
        _stage = _Stage.released;
        _releaseReason = 'otp_verified';
      });
      widget.auditLog.append('otp_verified', payload: {'amount_inr': 4000000});
    } else {
      widget.auditLog.append('otp_failed', payload: {'attempted': _otpController.text});
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Incorrect code — transfer remains held.')),
      );
    }
  }

  void _cancel() {
    setState(() => _stage = _Stage.idle);
    widget.auditLog.append('transfer_cancelled', payload: {'amount_inr': 4000000});
  }

  void _reset() {
    setState(() {
      _stage = _Stage.idle;
      _otpController.clear();
    });
  }

  @override
  Widget build(BuildContext context) {
    final (chainOk, brokenSeq) = widget.auditLog.verifyChain();
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Mock bank: vendor transfer', style: TextStyle(fontWeight: FontWeight.bold)),
        const Text(
          'Simulation only. OTP is never actually sent — shown on-screen for '
          'the demo (real bulk SMS needs TRAI registration, out of scope).',
          style: TextStyle(fontSize: 12, color: Color(0xFF94A3B8)),
        ),
        const SizedBox(height: 8),
        if (_stage == _Stage.idle)
          ElevatedButton(
            onPressed: _initiateTransfer,
            child: const Text('Initiate Transfer (₹40,00,000 → new remittance account)'),
          ),
        if (_stage == _Stage.held) ...[
          const Text('🛑 TRANSACTION HELD — synthetic-voice risk detected on this call.',
              style: TextStyle(color: Colors.red)),
          TextField(
            controller: _otpController,
            maxLength: 6,
            decoration: const InputDecoration(labelText: 'Enter 6-digit code sent to your device'),
          ),
          Text('(Demo code: $kDemoOtpCode)', style: const TextStyle(fontSize: 12)),
          Row(children: [
            ElevatedButton(onPressed: _verifyOtp, child: const Text('Verify code')),
            const SizedBox(width: 8),
            OutlinedButton(onPressed: _cancel, child: const Text('Cancel transfer')),
          ]),
        ],
        if (_stage == _Stage.released) ...[
          Text(
            _releaseReason == 'otp_verified'
                ? '✅ Transfer released after step-up verification.'
                : '✅ Transfer approved — no synthetic-voice risk detected.',
            style: const TextStyle(color: Colors.green),
          ),
          ElevatedButton(onPressed: _reset, child: const Text('Reset demo')),
        ],
        const SizedBox(height: 8),
        ExpansionTile(
          title: Text('Audit log (${widget.auditLog.length} entries, hash-chained)'),
          children: [
            Text(chainOk
                ? '✅ Chain verified — no tampering detected.'
                : '⚠️ Chain broken at entry $brokenSeq — tampering detected.'),
            for (final e in widget.auditLog.entries())
              Text('${e['seq']}: ${e['event']}', style: const TextStyle(fontSize: 12)),
          ],
        ),
      ],
    );
  }
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `flutter test test/ui/bank_modal_logic_test.dart`
Expected: PASS (2/2).

- [ ] **Step 5: Commit**

```bash
git add vaani/mobile/lib/ui/bank_modal.dart vaani/mobile/test/ui/bank_modal_logic_test.dart
git commit -m "feat(mobile): port bank HOLD/OTP/release workflow and audit wiring"
```

---

### Task 13: HomeScreen wiring + end-to-end integration test

**Files:**
- Create: `vaani/mobile/lib/ui/home_screen.dart`
- Modify: `vaani/mobile/lib/main.dart`
- Modify: `vaani/mobile/pubspec.yaml` (bundle the demo asset)
- Create: `vaani/mobile/assets/demo/call_A_whatsapp.wav` (copy step below)
- Test: `vaani/mobile/integration_test/bank_flow_test.dart`

**Interfaces:**
- Produces: the full app screen — file-import button, mic-capture
  toggle, `RiskGauge`, `SpectrogramView`, `RiskCurve`, `OcclusionOverlay`,
  `BankPanel`, wired to a `WindowPipeline` + `StubScorer`. This is the
  terminal integration point for Tasks 2–12.

- [ ] **Step 1: Copy the existing demo asset into the mobile project**

```bash
mkdir -p vaani/mobile/assets/demo
cp vaani/assets/demo/call_A_whatsapp.wav vaani/mobile/assets/demo/call_A_whatsapp.wav
```

Add to `pubspec.yaml` under `flutter: assets:`:
```yaml
    - assets/demo/call_A_whatsapp.wav
```

- [ ] **Step 2: Write the failing end-to-end integration test**

```dart
// vaani/mobile/integration_test/bank_flow_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';
import 'package:mobile/ui/home_screen.dart';

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('running the bundled demo call eventually shows ALERT and holds the transfer',
      (tester) async {
    await tester.pumpWidget(const MaterialApp(home: HomeScreen()));
    await tester.tap(find.text('Run bundled demo call'));
    await tester.pump();

    // The demo call's clone segment starts at 22s; pump enough frames for
    // the pipeline to reach and pass it. Real time, not pumpAndSettle,
    // since WindowPipeline runs on a real Stream over real decoded audio.
    await tester.pump(const Duration(seconds: 26));
    await tester.pumpAndSettle(const Duration(seconds: 1));

    expect(find.textContaining('ALERT'), findsOneWidget);

    await tester.tap(find.textContaining('Initiate Transfer'));
    await tester.pumpAndSettle();
    expect(find.textContaining('TRANSACTION HELD'), findsOneWidget);

    await tester.enterText(find.byType(TextField), '123456');
    await tester.tap(find.text('Verify code'));
    await tester.pumpAndSettle();
    expect(find.textContaining('released after step-up verification'), findsOneWidget);
  });
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `flutter test integration_test/bank_flow_test.dart -d <device>`
Expected: FAIL — `home_screen.dart` doesn't exist.

- [ ] **Step 4: Implement `HomeScreen`**

```dart
// vaani/mobile/lib/ui/home_screen.dart
import 'dart:async';
import 'dart:typed_data';
import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';
import '../audit/audit_log.dart';
import '../capture/audio_capture_bridge.dart';
import '../capture/audio_decode_bridge.dart';
import '../decision/decision_engine.dart';
import '../mel/mel_bridge.dart';
import '../pipeline/frame_result.dart';
import '../pipeline/window_pipeline.dart';
import '../scoring/stub_scorer.dart';
import 'bank_modal.dart';
import 'gauge.dart';
import 'occlusion_overlay.dart';
import 'risk_curve.dart';
import 'spectrogram_view.dart';

/// Demo clone-entry time for the bundled call_A_whatsapp.wav asset,
/// matching assets/raw/call_scripts.json's segment 2 start (22.0s) — see
/// app/engine_mock.py's CLONE_ENTRY_S. Only used for the bundled canned
/// demo; real imported/mic audio always uses cloneEntryS: null (honest
/// default — no known ground truth).
const double _demoCloneEntryS = 22.0;

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});
  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final _auditLog = AuditLog();
  final _decodeBridge = AudioDecodeBridge();
  final _micBridge = AudioCaptureBridge();
  FrameResult? _latest;
  final List<double> _emaHistory = [];
  StreamSubscription<FrameResult>? _sub;

  Future<void> _runBundledDemo() async {
    final bytes = await rootBundle.load('assets/demo/call_A_whatsapp.wav');
    final dir = await getTemporaryDirectory();
    final file = await File('${dir.path}/call_A_whatsapp.wav').writeAsBytes(bytes.buffer.asUint8List());
    await _runFile(file.path, cloneEntryS: _demoCloneEntryS);
  }

  Future<void> _importFile() async {
    final result = await FilePicker.platform.pickFiles(type: FileType.audio);
    if (result == null || result.files.single.path == null) return;
    await _runFile(result.files.single.path!, cloneEntryS: null);
  }

  Future<void> _runFile(String path, {required double? cloneEntryS}) async {
    Float32List pcm;
    try {
      pcm = await _decodeBridge.decodeFile(path);
    } on AudioDecodeException catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(SnackBar(content: Text('Could not read file: $e')));
      }
      return;
    }
    final pipeline = WindowPipeline(mel: MelBridge());
    final scorer = StubScorer(cloneEntryS: cloneEntryS);
    final hopSamples = 8000;
    final controller = StreamController<Float32List>();
    await _sub?.cancel();
    _sub = pipeline.process(controller.stream, scorer: scorer).listen((r) {
      setState(() {
        _latest = r;
        _emaHistory.add(r.ema);
      });
    });
    for (var i = 0; i + hopSamples <= pcm.length; i += hopSamples) {
      controller.add(Float32List.sublistView(pcm, i, i + hopSamples));
      await Future.delayed(const Duration(milliseconds: 500));
    }
    await controller.close();
  }

  Future<void> _startMic() async {
    final granted = await _micBridge.requestPermission();
    if (!granted) return;
    final pipeline = WindowPipeline(mel: MelBridge());
    final scorer = StubScorer(cloneEntryS: null);
    await _sub?.cancel();
    _sub = pipeline.process(_micBridge.start(), scorer: scorer).listen((r) {
      setState(() {
        _latest = r;
        _emaHistory.add(r.ema);
      });
    });
  }

  Future<void> _stopMic() async {
    await _micBridge.stop();
    await _sub?.cancel();
  }

  @override
  void dispose() {
    _sub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final state = _latest?.state ?? AlertState.normal;
    return Scaffold(
      appBar: AppBar(title: const Text('VAANI (mobile)')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Wrap(spacing: 8, children: [
              ElevatedButton(onPressed: _runBundledDemo, child: const Text('Run bundled demo call')),
              ElevatedButton(onPressed: _importFile, child: const Text('Import audio file')),
              ElevatedButton(onPressed: _startMic, child: const Text('Start mic capture')),
              OutlinedButton(onPressed: _stopMic, child: const Text('Stop mic capture')),
            ]),
            const SizedBox(height: 16),
            RiskGauge(
              ema: _latest?.ema,
              state: state,
              raw: _latest?.rawScore,
              backendLabel: 'stub (simulated — real model pending)',
            ),
            const SizedBox(height: 16),
            SpectrogramView(melDb: _latest?.melDb ?? const []),
            const SizedBox(height: 16),
            RiskCurve(emaHistory: _emaHistory, threshold: 0.6),
            const SizedBox(height: 16),
            const OcclusionOverlay(highlights: [], windowDurationS: 2.0),
            const SizedBox(height: 16),
            BankPanel(state: state, ema: _latest?.ema, auditLog: _auditLog),
          ],
        ),
      ),
    );
  }
}
```

- [ ] **Step 5: Wire `main.dart`**

```dart
// vaani/mobile/lib/main.dart
import 'package:flutter/material.dart';
import 'ui/home_screen.dart';

void main() => runApp(const VaaniApp());

class VaaniApp extends StatelessWidget {
  const VaaniApp({super.key});
  @override
  Widget build(BuildContext context) => const MaterialApp(home: HomeScreen());
}
```

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `flutter test integration_test/bank_flow_test.dart -d <device>`
Expected: PASS. If the `Duration(seconds: 26)` real-time pump makes the test
too slow/flaky in CI, reduce the per-chunk `Duration(milliseconds: 500)`
delay in `_runFile` to `Duration.zero` behind a test-only constructor
parameter on `HomeScreen` (e.g. `HomeScreen({this.chunkDelay =
const Duration(milliseconds: 500)})`) and pass `chunkDelay: Duration.zero`
from the test — re-run until it passes reliably.

- [ ] **Step 7: Run the full test suite**

Run: `cd vaani/mobile && flutter test` (unit/widget) and
`flutter test integration_test -d <device>` (instrumented)
Expected: ALL PASS.

- [ ] **Step 8: Commit**

```bash
git add vaani/mobile/lib/ui/home_screen.dart vaani/mobile/lib/main.dart \
        vaani/mobile/pubspec.yaml vaani/mobile/assets/demo \
        vaani/mobile/integration_test/bank_flow_test.dart
git commit -m "feat(mobile): wire HomeScreen end-to-end (Phase 1 complete)"
```

---

### Task 14: Master plan integration edits

**Files:**
- Modify: `vaani/00_MASTER_PLAN.md`

**Interfaces:** none — documentation-only task.

- [ ] **Step 1: Add a Module E row to §8's Week 1 table**

In `00_MASTER_PLAN.md`, find the Week 1 table (starts `| Day | Work | Done
means |`). Add a row after the "Fri" row and before "Sat":

```markdown
| Fri (parallel) | Module E: Flutter Android app shell, file import + mic capture, full UI (gauge/spectrogram/risk curve/occlusion/bank/audit log) wired to a clearly-labeled stub scorer | Phone runs the identical pipeline as the laptop demo, labeled "simulated — real model pending" |
```

- [ ] **Step 2: Add a Module E row to the Weeks 2–12 table**

Find the table starting `| Phase | Weeks | Deliverables | GPUs | Done means
|`. Add a row after "Ship":

```markdown
| Mobile (Module E) | whenever Module B ships ONNX | swap StubScorer → OnnxScorer at one call site (mobile/lib/scoring) | none (CPU/NPU on-device) | phone app runs real on-device inference, same thresholds as desktop |
```

- [ ] **Step 3: Add a phone beat to §9's demo script**

Find the table starting `| T | Beat |`. Add a row after the `0:55–1:10` row:

```markdown
| 1:10–1:15 | Hand a phone to a judge running the identical pipeline on the bundled demo call — same gauge, same alert, same audit log |
```

(Renumber the following row's timing if the 90 s budget needs adjusting —
compress the `1:10–1:30` row to `1:15–1:30` to keep the total at 90 s.)

- [ ] **Step 4: Add a checklist line to §13**

Find the `## 13. Deliverables Checklist` list. Add:

```markdown
- [ ] Mobile: Android app (Flutter) — Module E, Phase 1 stub-scored + Phase 2 real-model swap point
```

- [ ] **Step 5: Verify the edits render correctly**

Run: `git diff vaani/00_MASTER_PLAN.md` and read through — confirm no
existing rows were accidentally altered, all four new entries above are
present, and the file's markdown tables still have matching column counts
per row (count `|` characters against a neighboring row if unsure).

- [ ] **Step 6: Commit**

```bash
git add vaani/00_MASTER_PLAN.md
git commit -m "docs: weave Module E (mobile) into master plan timeline, demo script, checklist"
```

---

### Task 15: OnnxScorer (Phase 2 contract, gated on Module B)

**Files:**
- Create: `vaani/mobile/lib/scoring/onnx_scorer.dart`
- Create: `vaani/mobile/assets/test_fixtures/identity_model.onnx` (generated below)
- Test: `vaani/mobile/test/scoring/onnx_scorer_test.dart`

**Interfaces:**
- Produces: `class OnnxScorer implements Scorer` with constructor
  `OnnxScorer({required String modelAssetPath})`. Implements the same
  `Scorer` interface as `StubScorer` (Task 4) — this is the entire Phase 2
  swap contract: `HomeScreen` (Task 13) changes exactly one line
  (`StubScorer(...)` → `OnnxScorer(modelAssetPath: ...)`) to go live once
  Module B ships a real ONNX export.

This task builds and tests the **plumbing** now, against a tiny dummy ONNX
model (not the real TinyCNN, which doesn't exist yet) — proving the
swap-point contract holds before it's ever exercised for real. Wiring in
the true TinyCNN model asset once Module B ships it, and confirming its
input tensor shape matches `MelBridge`'s output, is future work outside
this plan's scope (spec §2: "no independent timeline").

- [ ] **Step 1: Generate a tiny dummy ONNX model fixture**

Run once, from `vaani/mobile/` (requires `pip install torch onnx` in
whatever Python env is available — this is a one-time fixture generation
script, not part of the app):
```bash
python -c "
import torch
class Identity(torch.nn.Module):
    def forward(self, x):
        # x: [1, 48, T] mel-db window -> a single scalar in [0,1] via mean+sigmoid,
        # just enough to exercise the ONNX Runtime plumbing end-to-end.
        return torch.sigmoid(x.mean(dim=(1, 2)))
model = Identity()
dummy = torch.zeros(1, 48, 10)
torch.onnx.export(model, dummy, 'assets/test_fixtures/identity_model.onnx',
                   input_names=['mel'], output_names=['score'],
                   dynamic_axes={'mel': {2: 'frames'}})
"
```
Add to `pubspec.yaml` under `flutter: assets:`:
```yaml
    - assets/test_fixtures/identity_model.onnx
```

- [ ] **Step 2: Write the failing test**

```dart
// vaani/mobile/test/scoring/onnx_scorer_test.dart
import 'dart:typed_data';
import 'package:flutter_test/flutter_test.dart';
import 'package:mobile/scoring/onnx_scorer.dart';
import 'package:mobile/scoring/scorer.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();

  test('OnnxScorer implements the same Scorer interface as StubScorer', () {
    final scorer = OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx');
    expect(scorer, isA<Scorer>());
    expect(scorer.backendLabel, isNot(contains('simulated')));
  });

  test('scoreWindow returns a value in [0, 1] for the dummy identity model',
      () async {
    final scorer = OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx');
    await scorer.load();
    final audio = Float32List(32000); // 2s @ 16kHz, silence
    final score = scorer.scoreWindow(audio, 16000, 0.0);
    expect(score, inInclusiveRange(0.0, 1.0));
  });

  test('swapping HomeScreen from StubScorer to OnnxScorer requires only the constructor call',
      () {
    // Documents the swap-point contract (spec §2/§6): both scorers satisfy
    // `Scorer`, so any code written against the interface (WindowPipeline,
    // Task 9) needs no changes when the concrete type changes.
    Scorer makeScorer(bool usePhase2) => usePhase2
        ? OnnxScorer(modelAssetPath: 'assets/test_fixtures/identity_model.onnx')
        : StubScorerForContractCheck();
    expect(makeScorer(true), isA<Scorer>());
    expect(makeScorer(false), isA<Scorer>());
  });
}

class StubScorerForContractCheck implements Scorer {
  @override
  String get backendLabel => 'stub';
  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) => 0.0;
}
```

- [ ] **Step 3: Run the test to verify it fails**

Run: `flutter test test/scoring/onnx_scorer_test.dart -d <device>`
(ONNX Runtime's Flutter plugin loads a native `.so`, so this needs a real
device/emulator, not plain `flutter test` on host.)
Expected: FAIL — `onnx_scorer.dart` doesn't exist.

- [ ] **Step 4: Implement `OnnxScorer`**

```dart
// vaani/mobile/lib/scoring/onnx_scorer.dart
import 'dart:typed_data';
import 'package:onnxruntime/onnxruntime.dart';
import 'scorer.dart';

/// Phase 2 real-inference scorer. Loads a bundled ONNX model asset and
/// runs it against MelBridge's mel-spectrogram output. This is the single
/// swap point named in the design spec (§2, §6): everywhere else in the
/// app depends only on the `Scorer` interface, never on this class
/// directly, so replacing StubScorer with OnnxScorer at HomeScreen's one
/// call site is the entire Phase 2 rollout.
///
/// NOTE for whoever wires in the real TinyCNN export: verify its expected
/// input tensor shape (n_mels, hop, n_fft) matches MelBridge's (48, 160,
/// 512) before assuming this class's mel input is compatible as-is — the
/// spec explicitly flags this as unverified until a real model exists.
class OnnxScorer implements Scorer {
  OnnxScorer({required this.modelAssetPath});

  final String modelAssetPath;
  OrtSession? _session;

  @override
  String get backendLabel => 'onnx (on-device)';

  Future<void> load() async {
    OrtEnv.instance.init();
    final rawBytes = await _loadAsset(modelAssetPath);
    _session = OrtSession.fromBuffer(rawBytes, OrtSessionOptions());
  }

  Future<Uint8List> _loadAsset(String path) async {
    // Kept as a separate method so tests can override asset loading if the
    // bundled asset path changes; production path loads via rootBundle.
    final data = await OrtEnv.instance.loadAsset(path);
    return data;
  }

  @override
  double scoreWindow(Float32List audio, int sr, double tStartS) {
    final session = _session;
    if (session == null) {
      throw StateError('OnnxScorer.load() must be awaited before scoreWindow()');
    }
    // Mel extraction happens in WindowPipeline via MelBridge; OnnxScorer
    // expects `audio` to already be the raw 2s PCM window (matching
    // StubScorer's contract) and computes its own mel pass synchronously
    // via a cached MelBridge call is NOT done here to keep this a sync
    // interface — Task 9's WindowPipeline instead passes mel-derived
    // features once Phase 2 lands; until then this identity-model test
    // exercises the ONNX plumbing with a placeholder mel window of zeros
    // shaped [1, 48, 10], not a real feature pipeline.
    final inputData = Float32List(48 * 10);
    final inputTensor = OrtValueTensor.createTensorWithDataList(inputData, [1, 48, 10]);
    final outputs = session.run(OrtRunOptions(), {'mel': inputTensor});
    final result = (outputs.first?.value as List).first as double;
    inputTensor.release();
    return result.clamp(0.0, 1.0);
  }
}
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `flutter test test/scoring/onnx_scorer_test.dart -d <device>`
Expected: PASS (3/3). If `onnxruntime`'s exact API surface (`OrtEnv`,
`OrtSession.fromBuffer`, `loadAsset`) differs from what's written above
(the package has had breaking API changes across versions), check
`vaani/mobile/.dart_tool/package_config.json` for the resolved
`onnxruntime` version and adjust the calls to match that version's actual
API — the behavioral contract (implements `Scorer`, returns a value in
[0,1], loads a bundled asset) is what must hold, not these exact method
names.

- [ ] **Step 6: Document the real integration gap**

Add a code comment (already included above) making explicit that
`scoreWindow`'s placeholder zero-input is a plumbing test only — real
Phase 2 wiring (feeding `MelBridge`'s actual mel output into the tensor,
confirming shape compatibility with Module B's real exported model) is
follow-up work for whoever picks this up once Module B ships, not part of
this plan's deliverable.

- [ ] **Step 7: Commit**

```bash
git add vaani/mobile/lib/scoring/onnx_scorer.dart vaani/mobile/test/scoring/onnx_scorer_test.dart \
        vaani/mobile/assets/test_fixtures/identity_model.onnx vaani/mobile/pubspec.yaml
git commit -m "feat(mobile): add OnnxScorer Phase 2 swap-point contract (dummy model)"
```

---

## Self-Review Notes (for whoever executes this plan)

- **Spec coverage:** §1 (capture modes) → Tasks 6–8; §2 (phasing) → Tasks 4,
  15; §3 (architecture/stack) → Tasks 1, 5–9; §4 (components) → Tasks 2–13;
  §5 (data flow/error handling) → Tasks 7, 9, 13; §6 (testing) → every
  task's test step; §7 (master plan integration) → Task 14; §8 (out of
  scope) → respected throughout (no iOS target, no live call-audio tap
  attempted, no server fallback).
- **Known follow-up not in this plan:** Task 15 deliberately stops at a
  dummy-model plumbing test. Wiring the real TinyCNN ONNX export in and
  confirming its tensor shape against `MelBridge`'s output is real,
  non-trivial work that cannot be planned in detail until that export
  exists — attempting to pre-write it now would violate "no placeholders"
  by guessing at an interface that doesn't exist yet.
