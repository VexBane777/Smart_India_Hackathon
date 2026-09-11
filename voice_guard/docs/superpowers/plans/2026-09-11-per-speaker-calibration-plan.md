# Per-speaker relative calibration (remediation track 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user capture a brief on-device baseline of their own voice and
have subsequent calls scored against *their* typical raw-score distribution
instead of a fixed global alert threshold, so a consistently low-variance
speaker stops getting permanently false-flagged.

**Architecture:** No model or feature-extraction changes. A new
`CalibrationProvider` persists `baselineMean`/`baselineStd`/`isCalibrated` in
`SharedPreferences`. A calibration capture flow (reusing the existing native
audio-capture path already wired for Live Mic Test) records ~8-10s of the
user's own voice, scores each 3s window with the already-loaded ONNX model
via `AudioService.scoreStream` (which carries the **raw**, pre-EMA score —
confirmed at `lib/services/audio_service.dart:89-90`), and computes
mean/std. `call_screen.dart`'s existing `riskProvider.update(score,
alertThreshold: ...)` call (`RiskScoreProvider.update` already accepts an
`alertThreshold` override, default `0.6` — see
`lib/providers/risk_score_provider.dart:45`) is given a per-speaker-adjusted
threshold instead of the raw `settings.sensitivity` value whenever a
calibration exists. Uncalibrated behavior is byte-for-byte unchanged (this
is the explicit backward-compat requirement — see Task 2).

**Tech Stack:** Dart/Flutter, `provider` package, `shared_preferences`
(already a dependency — see `settings_provider.dart`).

**Spec:** `voice_guard/docs/CRITICAL-entity-vs-style-confound.md` §4, item 1.

## Global Constraints

- Must not change scoring/UI behavior for a user who has never calibrated —
  `effectiveThreshold` must equal `settings.sensitivity` exactly when
  `!isCalibrated`.
- Must not touch `model_training/` or the shipped `.onnx` — this track is
  Dart-only by design (per the CRITICAL doc, it's the "cheap, partial,
  immediate" tier, not a model change).
- Calibration capture must not pollute call logs / risk history — it must
  not call `riskProvider.update()` or write to `RiskScoreProvider.history`.
- Calibration is per-device, not per-contact — one baseline, used for every
  call, matching the CRITICAL doc's "voice baseline capture at call/session
  start" framing generalized to "once, reusable" (recalibration is
  supported via a reset button, not a per-call requirement).

---

### Task 1: `CalibrationProvider` — pure threshold-adjustment logic + persistence

**Files:**
- Create: `lib/providers/calibration_provider.dart`
- Test: `test/calibration_provider_test.dart`

**Interfaces:**
- Produces: `CalibrationProvider` (a `ChangeNotifier`) with:
  - `bool get isCalibrated`
  - `double get baselineMean`, `double get baselineStd`
  - `Future<void> load()` — reads persisted state, mirrors
    `SettingsProvider.load()`'s pattern exactly.
  - `Future<void> setBaseline(double mean, double std)` — persists and
    notifies.
  - `Future<void> clearBaseline()` — resets to uncalibrated, persists.
  - `double effectiveThreshold(double sensitivity)` — the pure function
    other tasks call; static logic factored out as
    `CalibrationProvider.computeThreshold(...)` so it's testable without a
    `SharedPreferences` instance.

- [ ] **Step 1: Write the failing test for the pure threshold function**

```dart
// test/calibration_provider_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/providers/calibration_provider.dart';

void main() {
  group('CalibrationProvider.computeThreshold', () {
    test('uncalibrated: returns sensitivity unchanged', () {
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: false,
        baselineMean: 0.0,
        populationMean: 0.15,
      );
      expect(t, 0.60);
    });

    test('speaker baseline above population mean raises the threshold', () {
      // A naturally "busier"-sounding speaker (higher raw scores at
      // baseline) needs a higher bar before being flagged.
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.30, // 0.15 above the assumed population mean
        populationMean: 0.15,
      );
      expect(t, closeTo(0.75, 1e-9));
    });

    test('speaker baseline below population mean lowers the threshold', () {
      final t = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.05,
        populationMean: 0.15,
      );
      expect(t, closeTo(0.50, 1e-9));
    });

    test('clamps to [minThreshold, maxThreshold] so calibration can never '
        'disable detection or make it impossible to alert', () {
      final low = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: 0.99,
        populationMean: 0.15,
      );
      expect(low, CalibrationProvider.maxThreshold);

      final high = CalibrationProvider.computeThreshold(
        sensitivity: 0.60,
        isCalibrated: true,
        baselineMean: -0.5, // pathological, shouldn't occur, but must clamp
        populationMean: 0.15,
      );
      expect(high, CalibrationProvider.minThreshold);
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `flutter test test/calibration_provider_test.dart`
Expected: FAIL — `lib/providers/calibration_provider.dart` does not exist yet.

- [ ] **Step 3: Implement `CalibrationProvider`**

```dart
// lib/providers/calibration_provider.dart
import 'package:flutter/foundation.dart';
import 'package:shared_preferences/shared_preferences.dart';

/// Persists a per-device "voice baseline" (mean/std of this user's own raw
/// model scores, captured via a short calibration recording) and turns it
/// into an adjusted alert threshold. See
/// voice_guard/docs/CRITICAL-entity-vs-style-confound.md §4, item 1: this
/// does NOT change what the model outputs (no feature/model changes at
/// all) — it only shifts *where the alert line is drawn* for this specific
/// speaker, so a consistently low-or-high-variance talker isn't judged
/// against a population-wide threshold that doesn't fit them.
class CalibrationProvider extends ChangeNotifier {
  // Approximates the population mean of the *current* deployed model's raw
  // score on genuine, ambient-noise-present human speech. Matches
  // RiskScoreProvider's own pre-calibration default EMA seed
  // (risk_score_provider.dart:22, `ema ?? 0.15`) deliberately, so
  // calibration and the rest of the app agree on what "typical real
  // speech" looks like absent any per-speaker information. If the shipped
  // model is retrained (e.g. voice_guard_v9_noisefix_final or later), this
  // constant should be re-derived from that model's own mean raw score on
  // a held-out real-speech set — it is a property of the model, not of
  // this calibration mechanism.
  static const double populationMean = 0.15;
  static const double minThreshold = 0.35;
  static const double maxThreshold = 0.90;

  double _baselineMean = 0.0;
  double _baselineStd = 0.0;
  bool _isCalibrated = false;

  double get baselineMean => _baselineMean;
  double get baselineStd => _baselineStd;
  bool get isCalibrated => _isCalibrated;

  Future<void> load() async {
    final p = await SharedPreferences.getInstance();
    _baselineMean = p.getDouble('calibration_baselineMean') ?? 0.0;
    _baselineStd = p.getDouble('calibration_baselineStd') ?? 0.0;
    _isCalibrated = p.getBool('calibration_isCalibrated') ?? false;
    notifyListeners();
  }

  Future<void> setBaseline(double mean, double std) async {
    _baselineMean = mean;
    _baselineStd = std;
    _isCalibrated = true;
    final p = await SharedPreferences.getInstance();
    await p.setDouble('calibration_baselineMean', mean);
    await p.setDouble('calibration_baselineStd', std);
    await p.setBool('calibration_isCalibrated', true);
    notifyListeners();
  }

  Future<void> clearBaseline() async {
    _baselineMean = 0.0;
    _baselineStd = 0.0;
    _isCalibrated = false;
    final p = await SharedPreferences.getInstance();
    await p.remove('calibration_baselineMean');
    await p.remove('calibration_baselineStd');
    await p.setBool('calibration_isCalibrated', false);
    notifyListeners();
  }

  /// Instance convenience wrapper around the static, directly-testable core.
  double effectiveThreshold(double sensitivity) => computeThreshold(
        sensitivity: sensitivity,
        isCalibrated: _isCalibrated,
        baselineMean: _baselineMean,
        populationMean: populationMean,
      );

  /// Pure function, no I/O — deliberately static so it's testable without
  /// constructing SharedPreferences. Shifts the alert threshold by exactly
  /// how far this speaker's own baseline sits from the assumed population
  /// mean, so a speaker whose baseline is naturally elevated needs a
  /// proportionally higher raw/EMA score to trip an alert, and vice versa.
  /// This does NOT rescale the score itself (see plan header) — only where
  /// the line is drawn — deliberately, so raw/EMA scores stay comparable
  /// across the app (logs, UI, `RiskScoreProvider.history`) regardless of
  /// calibration state.
  static double computeThreshold({
    required double sensitivity,
    required bool isCalibrated,
    required double baselineMean,
    required double populationMean,
  }) {
    if (!isCalibrated) return sensitivity;
    final adjusted = sensitivity + (baselineMean - populationMean);
    return adjusted.clamp(minThreshold, maxThreshold);
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `flutter test test/calibration_provider_test.dart`
Expected: PASS (all 4 cases)

- [ ] **Step 5: Register the provider in the app's provider tree**

Find where `SettingsProvider` is registered (it will be a
`ChangeNotifierProvider` in `main.dart` or an app-root widget — check via
`grep -n "SettingsProvider(" lib/main.dart`). Add `CalibrationProvider` the
same way, immediately after `SettingsProvider`, and call
`context.read<CalibrationProvider>().load()` wherever
`SettingsProvider.load()` is already called at startup (same file,
same pattern).

- [ ] **Step 6: Commit**

```bash
git add lib/providers/calibration_provider.dart test/calibration_provider_test.dart lib/main.dart
git commit -m "feat(voice_guard): add CalibrationProvider for per-speaker alert threshold"
```

---

### Task 2: Calibration capture in `AudioService`

**Files:**
- Modify: `lib/services/audio_service.dart`
- Test: `test/audio_service_calibration_test.dart`

**Interfaces:**
- Consumes: `AudioService.scoreStream` (existing, `Stream<double>`, raw
  pre-EMA scores — `audio_service.dart:89-90`).
- Produces: `Future<CalibrationSample> AudioService.captureCalibrationSample({int windows = 8, Duration perWindowTimeout = const Duration(seconds: 2)})`
  returning a new small value type `CalibrationSample { double mean; double std; int windowsCaptured; }`.

- [ ] **Step 1: Write the failing test**

`AudioService.scoreStream` is driven by `_scoreCtrl`, which is private —
the test drives it through the service's public surface: feed synthetic PCM
via `ingestBytes`/the internal timer is awkward to unit-test directly, so
instead test the **pure aggregation logic** in isolation (mean/std over a
list of raw scores), which is what actually matters for calibration
correctness; the wiring itself is covered by the on-device manual check in
Task 5.

```dart
// test/audio_service_calibration_test.dart
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/audio_service.dart';

void main() {
  group('CalibrationSample.fromScores', () {
    test('computes mean and population std over collected raw scores', () {
      final sample = CalibrationSample.fromScores([0.10, 0.20, 0.12, 0.18]);
      expect(sample.windowsCaptured, 4);
      expect(sample.mean, closeTo(0.15, 1e-9));
      // population std of [0.10,0.20,0.12,0.18] around mean 0.15
      expect(sample.std, closeTo(0.0412310563, 1e-6));
    });

    test('empty input yields a zero sample rather than throwing', () {
      final sample = CalibrationSample.fromScores(const []);
      expect(sample.windowsCaptured, 0);
      expect(sample.mean, 0.0);
      expect(sample.std, 0.0);
    });
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `flutter test test/audio_service_calibration_test.dart`
Expected: FAIL — `CalibrationSample` doesn't exist yet.

- [ ] **Step 3: Implement `CalibrationSample` and `captureCalibrationSample`**

Add to `lib/services/audio_service.dart` (new top-level class above
`AudioService`, plus one new method inside the class):

```dart
/// Result of a calibration capture — mean/std of raw (pre-EMA) model
/// scores over N consecutive 3s windows of the user's own voice.
class CalibrationSample {
  final double mean;
  final double std;
  final int windowsCaptured;
  const CalibrationSample({required this.mean, required this.std, required this.windowsCaptured});

  factory CalibrationSample.fromScores(List<double> scores) {
    if (scores.isEmpty) {
      return const CalibrationSample(mean: 0.0, std: 0.0, windowsCaptured: 0);
    }
    final mean = scores.reduce((a, b) => a + b) / scores.length;
    final variance = scores.map((s) => (s - mean) * (s - mean)).reduce((a, b) => a + b) / scores.length;
    return CalibrationSample(mean: mean, std: math.sqrt(variance), windowsCaptured: scores.length);
  }
}
```

Then, inside `AudioService`, add:

```dart
  /// Collects `windows` consecutive raw scores from scoreStream, without
  /// touching RiskScoreProvider (deliberately — see plan Global
  /// Constraints: calibration must never pollute call logs/risk history).
  /// Caller is responsible for having already started native capture
  /// (`CallService.startCallDetection()`) and `startScoring()` — this
  /// method only *listens*, it doesn't start capture, mirroring how
  /// call_screen.dart's Live Mic Test already separates those concerns.
  Future<CalibrationSample> captureCalibrationSample({
    int windows = 8,
    Duration perWindowTimeout = const Duration(seconds: 2),
  }) async {
    final scores = <double>[];
    final sub = scoreStream.listen(scores.add);
    try {
      final deadline = DateTime.now().add(perWindowTimeout * windows);
      while (scores.length < windows && DateTime.now().isBefore(deadline)) {
        await Future.delayed(const Duration(milliseconds: 200));
      }
    } finally {
      await sub.cancel();
    }
    return CalibrationSample.fromScores(scores);
  }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `flutter test test/audio_service_calibration_test.dart`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add lib/services/audio_service.dart test/audio_service_calibration_test.dart
git commit -m "feat(voice_guard): add calibration sample capture to AudioService"
```

---

### Task 3: Wire the calibrated threshold into `call_screen.dart`

**Files:**
- Modify: `lib/screens/call_screen.dart:71-96` (`_bindPipeline`)

**Interfaces:**
- Consumes: `CalibrationProvider.effectiveThreshold(double)` (Task 1),
  `AudioService.captureCalibrationSample()` (Task 2).

- [ ] **Step 1: Read the calibration provider in `_bindPipeline` and use it for the threshold**

Modify the existing `scoreStream.listen` callback
(`call_screen.dart:78-96`) — change:

```dart
    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = riskProvider.isAlert;
      riskProvider.update(score);
```

to:

```dart
    final calibration = context.read<CalibrationProvider>();
    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = riskProvider.isAlert;
      final effectiveThreshold = calibration.effectiveThreshold(settings.sensitivity);
      riskProvider.update(score, alertThreshold: effectiveThreshold);
```

and update the two remaining `settings.sensitivity` reads a few lines below
(`if (cur != null && cur.score > settings.sensitivity)`) to
`effectiveThreshold` as well, so the overlay/notification gating uses the
same calibrated bar as the alert state machine — leaving them on
`settings.sensitivity` would make the UI internally inconsistent (alert
fires at one threshold, overlay/notification at another).

Add the import at the top of the file:
```dart
import '../providers/calibration_provider.dart';
```

- [ ] **Step 2: Manually verify no behavior change when uncalibrated**

Run: `flutter analyze` (must be clean) then run the existing widget test
suite: `flutter test`
Expected: all pre-existing tests still pass — this task changes no
observable behavior for an uncalibrated install (Task 1's `computeThreshold`
returns `sensitivity` unchanged when `!isCalibrated`, and every fresh
`CalibrationProvider` starts uncalibrated).

- [ ] **Step 3: Commit**

```bash
git add lib/screens/call_screen.dart
git commit -m "feat(voice_guard): use per-speaker calibrated threshold in call screen"
```

---

### Task 4: Calibration UI in Settings

**Files:**
- Modify: `lib/screens/settings_screen.dart`

**Interfaces:**
- Consumes: `CalibrationProvider` (Task 1), `AudioService.captureCalibrationSample()`
  (Task 2), `CallService.startCallDetection()`/`stopCallDetection()`
  (existing — same calls `call_screen.dart:186`/`202` already make to drive
  native capture for Live Mic Test).

- [ ] **Step 1: Add a "Voice Calibration" section**

Add a new section, following the existing `_section(...)` / widget pattern
already used in this file (see `settings_screen.dart:111` for `_section`,
and the `sensitivity` `Slider` at line 57 for the surrounding
`Consumer`/`context.watch` pattern this file already uses). Insert after the
sensitivity slider's section:

```dart
              _section('Voice Calibration'),
              Consumer<CalibrationProvider>(
                builder: (context, calib, _) => Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16),
                    child: Column(crossAxisAlignment: CrossAxisAlignment.start, children: [
                      Text(
                        calib.isCalibrated
                            ? 'Calibrated — alerts are tuned to your voice.'
                            : 'Not calibrated — using the default alert threshold for everyone.',
                        style: const TextStyle(fontSize: 13, fontWeight: FontWeight.w700),
                      ),
                      const SizedBox(height: 4),
                      const Text(
                        'Records ~10 seconds of you speaking normally, on-device only, '
                        'to tune the alert line to your natural speaking style. '
                        'Nothing is uploaded.',
                        style: TextStyle(fontSize: 11, color: Colors.black54),
                      ),
                      const SizedBox(height: 12),
                      Row(children: [
                        FilledButton.icon(
                          onPressed: _calibrating ? null : _runCalibration,
                          icon: _calibrating
                              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                              : const Icon(Icons.mic),
                          label: Text(_calibrating
                              ? 'Listening...'
                              : (calib.isCalibrated ? 'Recalibrate' : 'Calibrate My Voice')),
                        ),
                        if (calib.isCalibrated) ...[
                          const SizedBox(width: 8),
                          TextButton(
                            onPressed: _calibrating ? null : () => calib.clearBaseline(),
                            child: const Text('Reset'),
                          ),
                        ],
                      ]),
                    ]),
                  ),
                ),
              ),
```

- [ ] **Step 2: Add the calibration-run state and method to `_SettingsScreenState`**

```dart
  bool _calibrating = false;

  Future<void> _runCalibration() async {
    setState(() => _calibrating = true);
    final calls = context.read<CallService>();
    final audio = context.read<AudioService>();
    final calib = context.read<CalibrationProvider>();
    try {
      audio.clearBuffer();
      audio.startScoring();
      await calls.startCallDetection();
      final sample = await audio.captureCalibrationSample();
      await calls.stopCallDetection();
      audio.stopScoring();
      if (sample.windowsCaptured < 4) {
        if (mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Not enough speech captured — try again somewhere quieter.')),
          );
        }
        return;
      }
      await calib.setBaseline(sample.mean, sample.std);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Voice calibration saved.')),
        );
      }
    } finally {
      if (mounted) setState(() => _calibrating = false);
    }
  }
```

Add the required imports (`CallService`, `AudioService`,
`CalibrationProvider`) if not already present in this file — check with
`grep -n "^import" lib/screens/settings_screen.dart` first.

- [ ] **Step 3: Manual on-device check (no automated test — this method drives real native audio capture)**

Build and install (`flutter build apk --flavor privileged --release` or the
flavor already in use per `voice_guard/state.md`), open Settings, tap
"Calibrate My Voice", speak normally for ~10s, confirm the snackbar reports
success and the section text switches to "Calibrated — alerts are tuned to
your voice." Then start a Live Mic Test and confirm (via `adb logcat`, same
`Monitor:` log lines already used throughout `voice_guard/state.md`) that
the EMA/alert log line's implicit threshold behavior changed for a
deliberately monotone reading vs. before calibration — this is the same
manual verification style already used for every other on-device claim in
this project; do not claim this works without running it.

- [ ] **Step 4: Commit**

```bash
git add lib/screens/settings_screen.dart
git commit -m "feat(voice_guard): add voice calibration UI to settings"
```

---

### Task 5: Update `voice_guard/state.md`

**Files:**
- Modify: `voice_guard/state.md`

- [ ] **Step 1: Add a dated section recording what was implemented, what's verified vs. not**

Follow this file's own documented convention (see its final "How to keep
this file useful" section) — record: track 1 implemented (calibration
provider, capture, UI, threshold wiring), what's unit-tested (Tasks 1-2)
vs. only manually-verifiable (Tasks 3-4, since they drive real native audio
capture), and explicitly that this does not touch the model or fix the
confound's root cause (per the CRITICAL doc, track 1 is explicitly partial
— "does not fix the underlying feature confound").

- [ ] **Step 2: Commit**

```bash
git add voice_guard/state.md
git commit -m "docs(voice_guard): record track 1 (per-speaker calibration) implementation"
```
