# VoIP Call UX & In-Call AI Alerts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Protected Call actually show live AI risk during a call
(currently `AudioService.scoreStream` is never consumed by
`ProtectedCallScreen`, so the risk meter never updates — see Task 1) and
add the in-call alert UX `Voip.md` Phase 5 calls for: a color-coded banner,
haptic feedback on a synthetic-voice alert, and a forensic 5-second PCM
dump saved alongside the call log.

**Architecture:** Reuse the existing app-wide alert state machine
(`RiskScoreProvider` — EMA smoothing + 2-consecutive-window alert gating
already implemented and used by the carrier-call flow in `call_screen.dart`)
rather than building a new one — `Voip.md`'s Phase 5 policy (EMA + 2
consecutive windows over threshold) is already what this class does. This
plan wires Protected Call into that existing machinery and adds the pieces
that don't exist yet: the banner, haptics, and forensic dump. Deliberately
**not** in scope: a ringing/accept call-state machine (`Voip.md` Phase
1.2's `idle → outgoingRinging → connecting → active → ended`) — the
signaling protocol has no INVITE/RINGING concept (it's a dumb 2-peer SDP/ICE
relay, per sub-project 1's spec), and building one means changing
`signaling.py`'s protocol, which sub-project 1's spec explicitly deferred
alongside the rest of the full user-directory/session design.

**Tech Stack:** Flutter/Dart (`provider`, `flutter/services.dart` for
haptics — no new package needed), `path_provider` (already a dependency,
unused so far).

**Spec:** `Voip.md` §"Phase 5: In-Call AI Shield & Alert UX" (lines
194-204), applied on top of the already-implemented alert state machine in
`voice_guard/lib/providers/risk_score_provider.dart`. As with the resampling
plan, no separate design doc was written for this sub-project — moving
straight to a plan tonight per explicit user direction.

## Global Constraints

- Do not modify `RiskScoreProvider`'s EMA/threshold logic (lines 17-18,
  45-63) — it's shared with the carrier-call flow (`call_screen.dart`) and
  already matches `Voip.md`'s policy (EMA smoothing, alert only after 2+
  consecutive over-threshold windows). This plan only adds a *consumer* of
  it for Protected Call.
- No ringing/accept call-state machine (see Architecture above) — out of
  scope, already deferred in sub-project 1's spec.
- Reuse `ShadTokens` colors already defined for this exact purpose:
  `verified`/`verifiedBg` (normal), `suspicious`/`suspiciousBg` (warn),
  `detected`/`detectedBg` (alert) — `voice_guard/lib/design/tokens.dart:51-68`.
  Do not invent new colors.

---

### Task 1: Wire `AudioService.scoreStream` into `RiskScoreProvider` for Protected Call

**Files:**
- Modify: `voice_guard/lib/screens/protected_call_screen.dart`

**Interfaces:**
- Consumes: `AudioService.scoreStream` (`Stream<double>`,
  `audio_service.dart:45`), `SettingsProvider.sensitivity`
  (`settings_provider.dart:14`), `RiskScoreProvider.update(double rawScore, {double alertThreshold})`
  (`risk_score_provider.dart:45`).
- Produces: a `StreamSubscription<double>? _scoreSub` field on
  `_ProtectedCallScreenState`, cancelled in `dispose()` and in `_hangUp()`.

This is UI wiring with no new pure logic to unit-test (the logic being
exercised — EMA/threshold — is already covered wherever
`RiskScoreProvider`'s existing tests live); verification is manual,
matching how `call_screen.dart`'s equivalent wiring is verified in this
codebase.

- [ ] **Step 1: Add the subscription field and wire it in `_connect`**

In `voice_guard/lib/screens/protected_call_screen.dart`, add a field:

```dart
  StreamSubscription<double>? _scoreSub;
```

(add `import 'dart:async';` if not already present via a transitive import —
check first).

In `_connect()` (currently lines 27-43), after `audio.startScoring();` and
before `final signaling = ...`, add:

```dart
    final settings = context.read<SettingsProvider>();
    _scoreSub?.cancel();
    _scoreSub = audio.scoreStream.listen((score) {
      if (!mounted) return;
      risk.update(score, alertThreshold: settings.sensitivity);
    });
```

Add the import: `import '../providers/settings_provider.dart';`

(Note: this `settings` local also satisfies Task 3 of the signaling plan,
which reads `SettingsProvider` in the same method for signaling
host/port — if that task already landed, reuse its `settings` variable
instead of declaring a second one.)

- [ ] **Step 2: Cancel the subscription on hangup and dispose**

In `_hangUp()` (currently lines 45-50), add `await _scoreSub?.cancel();`
before `audio.stopScoring();`. In `dispose()` (currently lines 52-57), add
`_scoreSub?.cancel();` alongside the existing `_call?.dispose()`.

- [ ] **Step 3: Manual verification**

Run: `cd voice_guard && flutter run`. Start a Protected Call (two
instances, or one against a peer answering — same-LAN default is fine for
this check). Speak into the mic; confirm the risk meter
(`ShadRiskMeter`, already rendered at line 202) actually moves instead of
staying at 0 for the whole call — this is the bug this task fixes.

- [ ] **Step 4: Commit**

```bash
git add voice_guard/lib/screens/protected_call_screen.dart
git commit -m "fix(voice_guard): wire live AI score into RiskScoreProvider during Protected Call"
```

---

### Task 2: In-call alert banner (Emerald / Amber / Crimson)

**Files:**
- Create: `voice_guard/lib/widgets/shad_alert_banner.dart`
- Modify: `voice_guard/lib/screens/protected_call_screen.dart`
- Test: `voice_guard/test/shad_alert_banner_test.dart` (new)

**Interfaces:**
- Produces: `class ShadAlertBanner extends StatelessWidget` with
  `const ShadAlertBanner({required this.state, super.key})` where
  `state` is the `String` from `RiskScoreProvider.state`
  (`'normal' | 'warn' | 'alert'`). Renders label + color per state.
- Consumes: `RiskScoreProvider.state` (`risk_score_provider.dart:23`).

- [ ] **Step 1: Write the failing widget test**

```dart
// voice_guard/test/shad_alert_banner_test.dart
import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/design/tokens.dart';
import 'package:voice_guard/widgets/shad_alert_banner.dart';

void main() {
  testWidgets('normal state shows Verified Human in the verified color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'normal')));
    expect(find.text('Verified Human'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.verifiedBg);
  });

  testWidgets('warn state shows Suspicious in the suspicious color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'warn')));
    expect(find.text('Suspicious'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.suspiciousBg);
  });

  testWidgets('alert state shows Synthetic Clone Detected in the detected color', (tester) async {
    await tester.pumpWidget(const MaterialApp(home: ShadAlertBanner(state: 'alert')));
    expect(find.text('Synthetic Clone Detected'), findsOneWidget);
    final container = tester.widget<Container>(find.byType(Container).first);
    expect((container.decoration as BoxDecoration).color, ShadTokens.detectedBg);
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd voice_guard && flutter test test/shad_alert_banner_test.dart`
Expected: FAIL — `package:voice_guard/widgets/shad_alert_banner.dart` doesn't exist.

- [ ] **Step 3: Implement `ShadAlertBanner`**

```dart
// voice_guard/lib/widgets/shad_alert_banner.dart
import 'package:flutter/material.dart';
import '../design/tokens.dart';

/// In-call risk banner for Protected Call — Voip.md Phase 5: Emerald
/// ("Verified Human") -> Amber ("Suspicious") -> Crimson ("Synthetic Clone
/// Detected"), driven directly by [RiskScoreProvider.state].
class ShadAlertBanner extends StatelessWidget {
  final String state; // 'normal' | 'warn' | 'alert'
  const ShadAlertBanner({required this.state, super.key});

  static const _copy = {
    'normal': 'Verified Human',
    'warn': 'Suspicious',
    'alert': 'Synthetic Clone Detected',
  };

  Color _bg() => switch (state) {
        'alert' => ShadTokens.detectedBg,
        'warn' => ShadTokens.suspiciousBg,
        _ => ShadTokens.verifiedBg,
      };

  Color _fg() => switch (state) {
        'alert' => ShadTokens.detected,
        'warn' => ShadTokens.suspicious,
        _ => ShadTokens.verified,
      };

  Color _border() => switch (state) {
        'alert' => ShadTokens.detectedBorder,
        'warn' => ShadTokens.suspiciousBorder,
        _ => ShadTokens.verifiedBorder,
      };

  IconData _icon() => switch (state) {
        'alert' => Icons.gpp_bad,
        'warn' => Icons.warning_amber_rounded,
        _ => Icons.verified_user,
      };

  @override
  Widget build(BuildContext context) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
      decoration: BoxDecoration(
        color: _bg(),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: _border()),
      ),
      child: Row(
        children: [
          Icon(_icon(), color: _fg(), size: 20),
          const SizedBox(width: 10),
          Text(
            _copy[state] ?? _copy['normal']!,
            style: TextStyle(color: _fg(), fontWeight: FontWeight.w800, fontSize: 14),
          ),
        ],
      ),
    );
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd voice_guard && flutter test test/shad_alert_banner_test.dart`
Expected: PASS (3 tests)

- [ ] **Step 5: Render the banner in `ProtectedCallScreen`**

In the "Live Connected Call State" section (`protected_call_screen.dart`,
currently around lines 163-198, right after the room/peer-status
`ShadCard` and before the `ShadRiskMeter`), add:

```dart
            const SizedBox(height: ShadTokens.space3),
            ShadAlertBanner(state: context.watch<RiskScoreProvider>().state),
```

Add the import: `import '../widgets/shad_alert_banner.dart';`

- [ ] **Step 6: Manual verification**

Run: `cd voice_guard && flutter run`, start a Protected Call, use
`AudioService.injectBenchmarkTest(isAiVoice: true/false)` (already exists,
`audio_service.dart:290`) if there's a debug trigger for it, or simply
observe the banner during a real call — confirm it starts "Verified Human"
(emerald) and, per the existing 2-consecutive-window gate, only flips to
"Synthetic Clone Detected" (crimson) after sustained high scores, not a
single spike.

- [ ] **Step 7: Commit**

```bash
git add voice_guard/lib/widgets/shad_alert_banner.dart voice_guard/test/shad_alert_banner_test.dart voice_guard/lib/screens/protected_call_screen.dart
git commit -m "feat(voice_guard): add color-coded in-call risk banner to Protected Call"
```

---

### Task 3: Haptic feedback on alert transition

**Files:**
- Modify: `voice_guard/lib/screens/protected_call_screen.dart`

**Interfaces:**
- Consumes: `RiskScoreProvider.isAlert` (`risk_score_provider.dart:24`),
  `HapticFeedback.heavyImpact()` (`package:flutter/services.dart`, already
  transitively available — no new pubspec dependency).

- [ ] **Step 1: Trigger haptics only on the normal/warn → alert transition**

In Task 1's `_scoreSub` listener (inside `_connect()`), capture whether the
state was already in alert before calling `update`, and fire haptics only
on the transition — mirroring the `wasAlert`/`riskProvider.isAlert` pattern
already used in `call_screen.dart:92,101` (`_bindPipeline`):

```dart
    _scoreSub = audio.scoreStream.listen((score) {
      if (!mounted) return;
      final wasAlert = risk.isAlert;
      risk.update(score, alertThreshold: settings.sensitivity);
      if (!wasAlert && risk.isAlert) {
        HapticFeedback.heavyImpact();
      }
    });
```

Add the import: `import 'package:flutter/services.dart';` (if not already
present via another import in this file).

- [ ] **Step 2: Manual verification**

Run: `cd voice_guard && flutter run` on a physical device (haptics don't
fire on emulators/desktop). Drive the score into alert range (real
synthetic-audio test clip, or `injectBenchmarkTest(isAiVoice: true)` called
repeatedly if a debug hook exists) and confirm a single distinct vibration
fires exactly once on the transition, not repeatedly while alert stays
active.

- [ ] **Step 3: Commit**

```bash
git add voice_guard/lib/screens/protected_call_screen.dart
git commit -m "feat(voice_guard): haptic feedback on synthetic-voice alert transition"
```

---

### Task 4: Forensic 5-second PCM dump on alert

**Files:**
- Create: `voice_guard/lib/utils/wav_encoder.dart`
- Test: `voice_guard/test/wav_encoder_test.dart` (new)
- Modify: `voice_guard/lib/services/audio_service.dart`
- Modify: `voice_guard/lib/screens/protected_call_screen.dart`

**Interfaces:**
- Produces (Step 4): `WavEncoder.encodePcm16Mono(List<double> samples, {int sampleRate = 16000}) -> Uint8List`
  — inverse of the existing `AudioProcessor.decodePcm16Wav`.
- Produces (Step 8): `AudioService.snapshotBuffer() -> List<double>` — a
  copy of the existing 5-second ring buffer (`_buffer`, already capped at
  `maxSamples = 80000` in `ingestBytes`, `audio_service.dart:72-75` — this
  task adds no new buffering, just exposes what's already held).
- Consumes (Step 9): both of the above, plus `RiskScoreProvider.addCallLog`
  (`risk_score_provider.dart:78`) and `path_provider`'s
  `getApplicationDocumentsDirectory()`.

- [ ] **Step 1: Write the failing round-trip test**

```dart
// voice_guard/test/wav_encoder_test.dart
import 'dart:math' as math;
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/utils/audio_processor.dart';
import 'package:voice_guard/utils/wav_encoder.dart';

void main() {
  test('encodePcm16Mono round-trips through AudioProcessor.decodePcm16Wav', () {
    final samples = List<double>.generate(1600, (i) => 0.5 * math.sin(2 * math.pi * 220 * i / 16000));

    final wavBytes = WavEncoder.encodePcm16Mono(samples, sampleRate: 16000);
    final decoded = AudioProcessor.decodePcm16Wav(wavBytes);

    expect(decoded, isNotNull);
    expect(decoded!.sampleRate, 16000);
    expect(decoded.samples.length, samples.length);
    for (var i = 0; i < samples.length; i++) {
      // PCM16 quantization: within one quantization step of the original.
      expect((decoded.samples[i] - samples[i]).abs(), lessThan(1.0 / 32768.0 * 1.5));
    }
  });

  test('encodePcm16Mono produces a canonical 44-byte header', () {
    final bytes = WavEncoder.encodePcm16Mono([0.0, 0.1, -0.1], sampleRate: 16000);
    expect(bytes.length, 44 + 3 * 2);
    expect(String.fromCharCodes(bytes.sublist(0, 4)), 'RIFF');
    expect(String.fromCharCodes(bytes.sublist(8, 12)), 'WAVE');
    expect(String.fromCharCodes(bytes.sublist(12, 16)), 'fmt ');
    expect(String.fromCharCodes(bytes.sublist(36, 40)), 'data');
  });
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd voice_guard && flutter test test/wav_encoder_test.dart`
Expected: FAIL — `package:voice_guard/utils/wav_encoder.dart` doesn't exist.

- [ ] **Step 3: Implement `WavEncoder`**

```dart
// voice_guard/lib/utils/wav_encoder.dart
import 'dart:typed_data';

/// Encodes normalized [-1, 1] samples as a canonical 16-bit PCM mono .wav —
/// the inverse of AudioProcessor.decodePcm16Wav. Used for the Protected
/// Call forensic dump (Voip.md Phase 5).
class WavEncoder {
  static Uint8List encodePcm16Mono(List<double> samples, {int sampleRate = 16000}) {
    const bitsPerSample = 16;
    const channels = 1;
    final byteRate = sampleRate * channels * bitsPerSample ~/ 8;
    final blockAlign = channels * bitsPerSample ~/ 8;
    final dataSize = samples.length * 2;

    final buffer = ByteData(44 + dataSize);
    void writeAscii(int offset, String s) {
      for (var i = 0; i < s.length; i++) {
        buffer.setUint8(offset + i, s.codeUnitAt(i));
      }
    }

    writeAscii(0, 'RIFF');
    buffer.setUint32(4, 36 + dataSize, Endian.little);
    writeAscii(8, 'WAVE');
    writeAscii(12, 'fmt ');
    buffer.setUint32(16, 16, Endian.little); // fmt chunk size
    buffer.setUint16(20, 1, Endian.little); // audioFormat = PCM
    buffer.setUint16(22, channels, Endian.little);
    buffer.setUint32(24, sampleRate, Endian.little);
    buffer.setUint32(28, byteRate, Endian.little);
    buffer.setUint16(32, blockAlign, Endian.little);
    buffer.setUint16(34, bitsPerSample, Endian.little);
    writeAscii(36, 'data');
    buffer.setUint32(40, dataSize, Endian.little);

    for (var i = 0; i < samples.length; i++) {
      final clamped = samples[i].clamp(-1.0, 1.0);
      final intSample = (clamped * 32767.0).round().clamp(-32768, 32767);
      buffer.setInt16(44 + i * 2, intSample, Endian.little);
    }

    return buffer.buffer.asUint8List();
  }
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd voice_guard && flutter test test/wav_encoder_test.dart`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit the encoder**

```bash
git add voice_guard/lib/utils/wav_encoder.dart voice_guard/test/wav_encoder_test.dart
git commit -m "feat(voice_guard): add PCM16 WAV encoder for forensic audio dumps"
```

- [ ] **Step 6: Expose a buffer snapshot from `AudioService`**

In `voice_guard/lib/services/audio_service.dart`, add a public method near
`clearBuffer()` (line 140):

```dart
  /// Copy of the last ~5s of raw PCM (see [ingestBytes]'s maxSamples cap) —
  /// used for the Protected Call forensic dump when an alert fires.
  List<double> snapshotBuffer() => List.unmodifiable(_buffer);
```

- [ ] **Step 7: Write and run a unit test for the snapshot**

```dart
// append to voice_guard/test/audio_service_test.dart if it exists;
// otherwise create it — check first with:
// find voice_guard/test -iname "audio_service_test.dart"
```
If the file exists, add:
```dart
  test('snapshotBuffer returns a copy of ingested samples, capped at 5s', () {
    final service = AudioService(/* existing test double/tflite arg — match this file's existing setUp */);
    final onesSecond = Uint8List(32000); // 1s of 16kHz mono PCM16 zero bytes
    service.ingestBytes(onesSecond);
    expect(service.snapshotBuffer().length, 16000);
  });
```
If `audio_service_test.dart` doesn't exist yet, skip writing a standalone
test file for this one method — `AudioService`'s constructor requires a
`TFLiteService`, and standing up a proper test double for it is
disproportionate to a one-line accessor; instead fold verification into
Step 9's manual check, which exercises `snapshotBuffer()` for real during
an actual call.

- [ ] **Step 8: Commit**

```bash
git add voice_guard/lib/services/audio_service.dart
git commit -m "feat(voice_guard): expose a snapshot of the recent-audio ring buffer"
```

- [ ] **Step 9: Write the dump to disk on alert and log it**

In `protected_call_screen.dart`, extend Task 3's transition check:

```dart
    _scoreSub = audio.scoreStream.listen((score) async {
      if (!mounted) return;
      final wasAlert = risk.isAlert;
      risk.update(score, alertThreshold: settings.sensitivity);
      if (!wasAlert && risk.isAlert) {
        HapticFeedback.heavyImpact();
        await _dumpForensicAudio(audio, risk);
      }
    });
```

Add the method:

```dart
  Future<void> _dumpForensicAudio(AudioService audio, RiskScoreProvider risk) async {
    try {
      final samples = audio.snapshotBuffer();
      final wavBytes = WavEncoder.encodePcm16Mono(samples, sampleRate: 16000);
      final dir = await getApplicationDocumentsDirectory();
      final path = '${dir.path}/forensic_${DateTime.now().millisecondsSinceEpoch}.wav';
      await File(path).writeAsBytes(wavBytes);
      risk.addCallLog(CallLog(
        id: DateTime.now().millisecondsSinceEpoch.toString(),
        timestamp: DateTime.now(),
        number: 'Protected Call: ${_roomController.text.trim()}',
        riskScore: risk.current?.score ?? 0.0,
        verdict: Verdict.detected,
        recordingPath: path,
      ));
    } catch (e) {
      debugPrint('Protected Call: forensic dump failed: $e');
    }
  }
```

Add imports: `import 'dart:io';`, `import 'package:path_provider/path_provider.dart';`,
`import '../utils/wav_encoder.dart';`, `import '../models/call_log.dart';`
(check `Verdict`'s import path — it's referenced via `CallLog` already in
`call_screen.dart`; use the same import that file uses).

- [ ] **Step 10: Manual verification**

Run: `cd voice_guard && flutter run` on a physical device. Drive a call
into alert state (Task 3's verification setup). After the alert fires,
open the Logs screen (`logs_screen.dart`) and confirm a new entry appears
with a non-null `recordingPath`; confirm the file exists at that path (via
`adb shell run-as com.voiceguard.voice_guard ls files/` or similar) and
that it's a valid, playable WAV (copy it off-device and open it in any
audio player).

- [ ] **Step 11: Commit**

```bash
git add voice_guard/lib/screens/protected_call_screen.dart
git commit -m "feat(voice_guard): dump forensic 5s audio + call log entry when an alert fires"
```
