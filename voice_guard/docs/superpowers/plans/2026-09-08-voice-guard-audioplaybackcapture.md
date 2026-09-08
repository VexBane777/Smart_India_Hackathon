# VoIP Call Audio Capture (AudioPlaybackCapture) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let VoiceGuard analyze the caller's voice on a VoIP app call (WhatsApp, Telegram, Zoom, Google Meet) in real time, using Android's public `AudioPlaybackCapture` API — no privileged permission, no telecom hack, works today.

**Architecture:** A new native capture path (`PlaybackCaptureManager`) uses `MediaProjection` + `AudioPlaybackCaptureConfiguration` to tap the PCM another app is currently playing back (the far end's voice, since that's what the OS routes to the speaker/earpiece). It feeds those bytes into the *same* `AudioService.ingestBytes()` → TFLite scoring pipeline the existing Live Call screen already uses — this plan adds a new **source**, not a new pipeline. A one-time system consent dialog (the same "Start recording or casting your screen?" dialog used by screen recorders) grants the `MediaProjection` token; a foreground service keeps capture alive while the VoIP app runs in the background.

**Tech Stack:** Kotlin (`android.media.projection.MediaProjection`, `android.media.AudioPlaybackCaptureConfiguration`, `android.media.AudioRecord.Builder`), existing Flutter/Dart `AudioService`/`CallService` pattern.

**Spec:** No separate spec doc exists — this plan's own Architecture section *is* the spec, derived directly from the debugging investigation earlier in this conversation (see the "What went wrong" findings: carrier-call mic capture is blocked at the OS/OEM level regardless of `AudioSource`, confirmed by WAV sample analysis on a ColorOS-family device; `AudioPlaybackCapture` is the public-API alternative that sidesteps that specific wall for VoIP-app calls).

## Global Constraints

- `AudioPlaybackCaptureConfiguration` requires **API 29+ (Android 10)**. The app's `minSdk` is 24 (confirmed via `adb shell dumpsys package com.voiceguard.voice_guard` → `minSdk=24`), so every entry point into this feature MUST runtime-guard on `Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q` and be hidden/disabled below that.
- On **API 34+ (Android 14)**, a foreground service using `MediaProjection` must declare `android:foregroundServiceType="mediaProjection"` and the app must hold `android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION`, or the service throws `MissingForegroundServiceTypeException` at `startForeground()`.
- Capture only works for apps that have **not** called `setAllowedCapturePolicy(AudioAttributes.ALLOW_CAPTURE_BY_NONE)` on their playback `AudioAttributes`. This is per-app, undocumented in advance, and must be verified empirically per target app (Task 6).
- This path captures **only the far end's voice** (what's being played back) — it does not and cannot capture the local user's own microphone input. That's fine for this product's actual requirement (classify the incoming/caller voice), but do not build any UI copy that implies both sides are analyzed.
- No new third-party package dependency. Pure Android platform API — do not reach for a plugin off pub.dev for this; the raw platform channel is small enough to hand-write, matching how `CallService`/`AudioCaptureManager` already work in this codebase.
- Reuse `AudioService.ingestBytes()` (in `lib/services/audio_service.dart`) as the single point where PCM enters the scoring pipeline — do not create a second buffering/scoring path.

---

## File Structure

New files:
- `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureManager.kt` — owns the `MediaProjection` token, `AudioRecord` (playback-capture mode), and the capture read-loop. Mirrors `AudioCaptureManager`'s shape (a Kotlin `object` singleton with `start`/`stop`/`setSink`) so a reader of one already understands the other.
- `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureForegroundService.kt` — minimal foreground `Service` that exists only to satisfy the Android 10+ "must be running in a foreground service" requirement for playback capture; delegates all real work to `PlaybackCaptureManager`.
- `lib/services/playback_capture_service.dart` — Dart-side `MethodChannel` wrapper, same shape as `CallService`.
- `lib/screens/voip_protection_screen.dart` — new screen: "Protect a VoIP Call" entry point, consent flow, live risk meter (reuses `RiskMeter` widget from `lib/widgets/risk_meter.dart`).

Modified files:
- `android/app/src/main/AndroidManifest.xml` — add `FOREGROUND_SERVICE_MEDIA_PROJECTION` permission (API 34+) and the new foreground service declaration.
- `android/app/src/main/kotlin/com/voiceguard/voice_guard/MainActivity.kt` — add method-channel handlers (`requestPlaybackCaptureConsent`, `startPlaybackCapture`, `stopPlaybackCapture`) and an `onActivityResult` override to receive the `MediaProjection` consent result.
- `lib/main.dart` — register the new screen's route/nav entry and provide `PlaybackCaptureService` alongside the existing `CallService`.

---

## Task 1: Manifest permissions and foreground service declaration

**Files:**
- Modify: `android/app/src/main/AndroidManifest.xml`
- Create: `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureForegroundService.kt`

**Interfaces:**
- Produces: a declared, buildable (but not yet functional) `PlaybackCaptureForegroundService` class other tasks will drive from `PlaybackCaptureManager`.

- [ ] **Step 1: Add the foreground-service-type permission**

In `android/app/src/main/AndroidManifest.xml`, inside the existing `<manifest>` block (alongside the other `<uses-permission>` entries near the top), add:

```xml
    <uses-permission android:name="android.permission.FOREGROUND_SERVICE_MEDIA_PROJECTION" />
```

- [ ] **Step 2: Write the minimal foreground service class**

Create `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureForegroundService.kt`:

```kotlin
package com.voiceguard.voice_guard

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder

/**
 * Exists only to satisfy Android 10+'s requirement that AudioPlaybackCapture
 * run inside a foreground service. All real capture logic lives in
 * PlaybackCaptureManager; this class just keeps the process alive and shows
 * the mandatory "VoiceGuard is monitoring a call" notification.
 */
class PlaybackCaptureForegroundService : Service() {
    companion object {
        private const val CHANNEL_ID = "playback_capture_channel"
        private const val NOTIFICATION_ID = 4201
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onCreate() {
        super.onCreate()
        val nm = getSystemService(NotificationManager::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            nm.createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "VoIP Call Protection", NotificationManager.IMPORTANCE_LOW)
            )
        }
        val notification: Notification = Notification.Builder(this, CHANNEL_ID)
            .setContentTitle("VoiceGuard is monitoring this call")
            .setContentText("Analyzing the caller's voice for AI cloning risk")
            .setSmallIcon(android.R.drawable.stat_notify_call_mute)
            .build()
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.UPSIDE_DOWN_CAKE) {
            startForeground(
                NOTIFICATION_ID,
                notification,
                android.content.pm.ServiceInfo.FOREGROUND_SERVICE_TYPE_MEDIA_PROJECTION
            )
        } else {
            startForeground(NOTIFICATION_ID, notification)
        }
    }

    override fun onDestroy() {
        PlaybackCaptureManager.stop()
        super.onDestroy()
    }
}
```

- [ ] **Step 3: Declare the service in the manifest**

In `android/app/src/main/AndroidManifest.xml`, inside `<application>`, alongside the existing `<service android:name=".InCallServiceImpl" .../>` block, add:

```xml
        <service
            android:name=".PlaybackCaptureForegroundService"
            android:exported="false"
            android:foregroundServiceType="mediaProjection" />
```

- [ ] **Step 4: Verify the project still builds**

Run: `cd voice_guard && flutter build apk --debug`
Expected: `✓ Built build\app\outputs\flutter-apk\app-debug.apk` — a service that's declared but never started doesn't change runtime behavior, so this is a pure compile/manifest-merge check.

- [ ] **Step 5: Commit**

```bash
git add android/app/src/main/AndroidManifest.xml android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureForegroundService.kt
git commit -m "feat(voice_guard): scaffold foreground service for playback capture"
```

---

## Task 2: PlaybackCaptureManager — the actual capture logic

**Files:**
- Create: `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureManager.kt`

**Interfaces:**
- Consumes: nothing from other tasks yet (receives its `MediaProjection` token as a parameter — Task 3 is what obtains that token from the system dialog).
- Produces:
  - `PlaybackCaptureManager.start(context: Context, projection: MediaProjection, onBytes: (ByteArray) -> Unit): Boolean` — returns `false` if capture could not start (e.g. below API 29, or `AudioRecord` failed to initialize).
  - `PlaybackCaptureManager.stop()`
  - `PlaybackCaptureManager.isRunning: Boolean` (read-only property)

- [ ] **Step 1: Write the capture manager**

Create `android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureManager.kt`:

```kotlin
package com.voiceguard.voice_guard

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioPlaybackCaptureConfiguration
import android.media.AudioRecord
import android.media.projection.MediaProjection
import android.os.Build
import android.util.Log

/**
 * Captures 16kHz 16-bit mono PCM of another app's call-audio *playback* —
 * i.e. the far end's voice as the OS routes it to speaker/earpiece — via the
 * public AudioPlaybackCapture API (Android 10+). This exists because a
 * regular app cannot read live cellular-call mic audio (see
 * AudioCaptureManager's docstring for the measured evidence); VoIP apps'
 * own call audio is a legitimate, documented alternative source that needs
 * only user consent, not a privileged permission.
 *
 * Deliberately captures playback only, not the local mic: for this
 * product's actual requirement (classify the INCOMING/caller voice), that
 * is exactly the signal needed, and it arrives undistorted by any
 * speaker-to-mic acoustic loop.
 */
object PlaybackCaptureManager {
    private const val TAG = "PlaybackCaptureManager"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    private var recorder: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var sink: ((ByteArray) -> Unit)? = null

    val isRunning: Boolean get() = running

    fun start(context: Context, projection: MediaProjection, onBytes: (ByteArray) -> Unit): Boolean {
        if (Build.VERSION.SDK_INT < Build.VERSION_CODES.Q) {
            Log.e(TAG, "AudioPlaybackCapture requires API 29+, device is ${Build.VERSION.SDK_INT}")
            return false
        }
        if (running) return true
        sink = onBytes

        val config = AudioPlaybackCaptureConfiguration.Builder(projection)
            .addMatchingUsage(AudioAttributes.USAGE_VOICE_COMMUNICATION)
            .addMatchingUsage(AudioAttributes.USAGE_MEDIA)
            .build()

        val format = AudioFormat.Builder()
            .setEncoding(ENCODING)
            .setSampleRate(SAMPLE_RATE)
            .setChannelMask(CHANNEL)
            .build()

        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) { Log.e(TAG, "Invalid min buffer size: $minBuf"); return false }

        recorder = try {
            AudioRecord.Builder()
                .setAudioPlaybackCaptureConfig(config)
                .setAudioFormat(format)
                .setBufferSizeInBytes(minBuf * 4)
                .build()
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord.Builder failed", e)
            null
        }

        if (recorder?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "Playback-capture AudioRecord not initialized")
            recorder?.release()
            recorder = null
            return false
        }

        recorder?.startRecording()
        running = true
        thread = Thread {
            val buf = ByteArray(3200) // 100ms @16kHz mono 16-bit
            while (running) {
                val n = recorder?.read(buf, 0, buf.size) ?: 0
                if (n > 0) {
                    val copy = buf.copyOf(n)
                    try { sink?.invoke(copy) } catch (_: Exception) {}
                }
            }
        }.also { it.isDaemon = true; it.start() }

        Log.i(TAG, "Playback capture started (16kHz mono)")
        return true
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
        try { recorder?.stop() } catch (_: Exception) {}
        try { recorder?.release() } catch (_: Exception) {}
        recorder = null
        Log.i(TAG, "Playback capture stopped")
    }
}
```

- [ ] **Step 2: Verify the project builds**

Run: `cd voice_guard && flutter build apk --debug`
Expected: `✓ Built build\app\outputs\flutter-apk\app-debug.apk`. `PlaybackCaptureManager` isn't called from anywhere yet, so this is purely a compile check — confirm no `AudioPlaybackCaptureConfiguration`/`AudioRecord.Builder` API usage errors (these APIs are API 29+, but referencing them in Kotlin source compiles fine regardless of `minSdk`; only the runtime guard in Step 1 above matters for correctness on old devices).

- [ ] **Step 3: Commit**

```bash
git add android/app/src/main/kotlin/com/voiceguard/voice_guard/PlaybackCaptureManager.kt
git commit -m "feat(voice_guard): add AudioPlaybackCapture-based capture manager"
```

---

## Task 3: MediaProjection consent flow + MethodChannel wiring

**Files:**
- Modify: `android/app/src/main/kotlin/com/voiceguard/voice_guard/MainActivity.kt`

**Interfaces:**
- Consumes: `PlaybackCaptureManager.start(context, projection, onBytes): Boolean`, `PlaybackCaptureManager.stop()`, `PlaybackCaptureManager.isRunning` (Task 2).
- Produces: three new MethodChannel methods callable from Dart on `com.voiceguard/calls`:
  - `"requestPlaybackCaptureConsent"` → `Boolean` (`true` once the system dialog was shown; the actual grant/deny arrives asynchronously via `onCallStateChanged`-style push — see Step 3)
  - `"startPlaybackCapture"` → `Boolean` (only meaningful after consent was granted)
  - `"stopPlaybackCapture"` → `null`
  - a new async push event on the same channel: `"onPlaybackCaptureConsent"` with `{"granted": Boolean}` — needed because the consent result arrives via `onActivityResult`, not synchronously with the method call.

- [ ] **Step 1: Add state fields and the consent request launcher**

In `MainActivity.kt`, add these imports near the top:

```kotlin
import android.app.Activity
import android.content.Intent
import android.media.projection.MediaProjectionManager
```

Add this field inside the `MainActivity` class (near `private var eventSink`):

```kotlin
    private var pendingProjectionResult: MethodChannel.Result? = null
    private val PLAYBACK_CAPTURE_REQUEST_CODE = 2001
```

- [ ] **Step 2: Add the three method-channel cases**

In `MainActivity.kt`'s `mc.setMethodCallHandler { call, result -> when (call.method) { ... } }` block, add these cases (alongside the existing `"toggleMicMute"` case, before `"showOverlay"`):

```kotlin
                "requestPlaybackCaptureConsent" -> {
                    if (android.os.Build.VERSION.SDK_INT < android.os.Build.VERSION_CODES.Q) {
                        result.error("UNSUPPORTED", "Requires Android 10+", null)
                    } else {
                        val mpm = getSystemService(MediaProjectionManager::class.java)
                        pendingProjectionResult = result
                        startActivityForResult(mpm.createScreenCaptureIntent(), PLAYBACK_CAPTURE_REQUEST_CODE)
                    }
                }
                "startPlaybackCapture" -> {
                    val projection = pendingMediaProjection
                    if (projection == null) {
                        result.success(false)
                    } else {
                        val started = PlaybackCaptureManager.start(this, projection) { bytes ->
                            eventSink?.success(bytes)
                        }
                        result.success(started)
                    }
                }
                "stopPlaybackCapture" -> {
                    PlaybackCaptureManager.stop()
                    pendingMediaProjection?.stop()
                    pendingMediaProjection = null
                    result.success(null)
                }
```

Add the field this references (near `pendingProjectionResult`):

```kotlin
    private var pendingMediaProjection: android.media.projection.MediaProjection? = null
```

- [ ] **Step 3: Handle the consent result and start the foreground service**

Add this method to the `MainActivity` class:

```kotlin
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode != PLAYBACK_CAPTURE_REQUEST_CODE) return
        val granted = resultCode == Activity.RESULT_OK && data != null
        if (granted) {
            val mpm = getSystemService(MediaProjectionManager::class.java)
            pendingMediaProjection = mpm.getMediaProjection(resultCode, data!!)
            startForegroundService(Intent(this, PlaybackCaptureForegroundService::class.java))
        }
        pendingProjectionResult?.success(granted)
        pendingProjectionResult = null
        channel?.invokeMethod("onPlaybackCaptureConsent", mapOf("granted" to granted))
    }
```

- [ ] **Step 4: Stop capture and the foreground service on activity destroy**

In `MainActivity.kt`'s existing `onDestroy()`, add before `super.onDestroy()`:

```kotlin
        PlaybackCaptureManager.stop()
        pendingMediaProjection?.stop()
```

- [ ] **Step 5: Verify the project builds**

Run: `cd voice_guard && flutter build apk --debug`
Expected: `✓ Built build\app\outputs\flutter-apk\app-debug.apk`

- [ ] **Step 6: Commit**

```bash
git add android/app/src/main/kotlin/com/voiceguard/voice_guard/MainActivity.kt
git commit -m "feat(voice_guard): wire MediaProjection consent flow for playback capture"
```

---

## Task 4: Dart PlaybackCaptureService

**Files:**
- Create: `lib/services/playback_capture_service.dart`
- Test: `test/playback_capture_service_test.dart`

**Interfaces:**
- Consumes: the three MethodChannel methods + `"onPlaybackCaptureConsent"` push event from Task 3.
- Produces: `PlaybackCaptureService` class with:
  - `Future<bool> requestConsent()`
  - `Future<bool> startCapture()`
  - `Future<void> stopCapture()`
  - `Stream<bool> get consentResult` — broadcast stream Task 5's UI listens to for the async grant/deny.

- [ ] **Step 1: Write the failing test**

Create `test/playback_capture_service_test.dart`:

```dart
import 'package:flutter/services.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:voice_guard/services/playback_capture_service.dart';

void main() {
  TestWidgetsFlutterBinding.ensureInitialized();
  const channel = MethodChannel('com.voiceguard/calls');

  tearDown(() {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, null);
  });

  test('requestConsent returns the native result', () async {
    TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger
        .setMockMethodCallHandler(channel, (call) async {
      expect(call.method, 'requestPlaybackCaptureConsent');
      return true;
    });
    final service = PlaybackCaptureService();
    expect(await service.requestConsent(), true);
  });

  test('consentResult stream emits pushed onPlaybackCaptureConsent events', () async {
    final service = PlaybackCaptureService();
    final events = <bool>[];
    final sub = service.consentResult.listen(events.add);

    final handler = TestDefaultBinaryMessengerBinding.instance.defaultBinaryMessenger;
    final call = MethodCall('onPlaybackCaptureConsent', {'granted': true});
    final data = const StandardMethodCodec().encodeMethodCall(call);
    await handler.handlePlatformMessage('com.voiceguard/calls', data, (_) {});

    await Future.delayed(Duration.zero);
    expect(events, [true]);
    await sub.cancel();
  });
}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `cd voice_guard && flutter test test/playback_capture_service_test.dart`
Expected: FAIL — `Error: Not found: 'package:voice_guard/services/playback_capture_service.dart'`

- [ ] **Step 3: Write the implementation**

Create `lib/services/playback_capture_service.dart`:

```dart
import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

/// Dart-side wrapper for the native AudioPlaybackCapture flow (see
/// PlaybackCaptureManager.kt's docstring for why this path exists: it
/// captures a VoIP app's call-playback audio via the public MediaProjection
/// API, as the alternative to mic capture on a real cellular call, which
/// Android blocks for non-privileged apps).
class PlaybackCaptureService {
  static const _method = MethodChannel('com.voiceguard/calls');
  final _consentCtrl = StreamController<bool>.broadcast();

  PlaybackCaptureService() {
    _method.setMethodCallHandler(_onMethodCall);
  }

  Future<dynamic> _onMethodCall(MethodCall call) async {
    if (call.method == 'onPlaybackCaptureConsent') {
      final args = call.arguments is Map ? call.arguments as Map : null;
      _consentCtrl.add(args?['granted'] as bool? ?? false);
    }
    return null;
  }

  Stream<bool> get consentResult => _consentCtrl.stream;

  Future<bool> requestConsent() async {
    try {
      final res = await _method.invokeMethod<bool>('requestPlaybackCaptureConsent');
      return res ?? false;
    } catch (e) {
      debugPrint('requestPlaybackCaptureConsent failed: $e');
      return false;
    }
  }

  Future<bool> startCapture() async {
    try {
      final res = await _method.invokeMethod<bool>('startPlaybackCapture');
      return res ?? false;
    } catch (e) {
      debugPrint('startPlaybackCapture failed: $e');
      return false;
    }
  }

  Future<void> stopCapture() async {
    try {
      await _method.invokeMethod('stopPlaybackCapture');
    } catch (e) {
      debugPrint('stopPlaybackCapture failed: $e');
    }
  }

  void dispose() => _consentCtrl.close();
}
```

**Note:** `PlaybackCaptureService` sets its own `setMethodCallHandler` on the shared `com.voiceguard/calls` channel. `CallService.setCallStateCallback()` (in `lib/services/call_service.dart`) also does this. Flutter's `MethodChannel.setMethodCallHandler` only keeps the *last* handler registered — if both services are constructed and both call `setMethodCallHandler`, whichever runs second silently wins and the other's incoming calls stop arriving. Task 5 must not call `CallService.setCallStateCallback()` and construct `PlaybackCaptureService` in a way where both are live at once without accounting for this — either merge them into one handler, or only register `PlaybackCaptureService`'s handler while the VoIP-protection screen is actually mounted (`initState`/`dispose`).

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd voice_guard && flutter test test/playback_capture_service_test.dart`
Expected: `00:0X +2: All tests passed!`

- [ ] **Step 5: Commit**

```bash
git add lib/services/playback_capture_service.dart test/playback_capture_service_test.dart
git commit -m "feat(voice_guard): add Dart PlaybackCaptureService with consent stream"
```

---

## Task 5: VoIP Protection screen + pipeline wiring

**Files:**
- Create: `lib/screens/voip_protection_screen.dart`
- Modify: `lib/main.dart` — provide `PlaybackCaptureService`, add navigation entry.
- Modify: `lib/services/call_service.dart` — fix the shared-channel handler collision noted in Task 4 (see Step 1).

**Interfaces:**
- Consumes: `PlaybackCaptureService` (Task 4), `AudioService.ingestBytes()` / `AudioService.scoreStream` (existing, in `lib/services/audio_service.dart`), `RiskMeter` widget (existing, `lib/widgets/risk_meter.dart`).
- Produces: a reachable screen at a new bottom-nav or home quick-action entry titled "VoIP Call Protection".

- [ ] **Step 1: Resolve the shared MethodChannel handler collision**

Open `lib/services/call_service.dart` and `lib/services/playback_capture_service.dart`. Change `CallService.setCallStateCallback()` so it no longer calls `_method.setMethodCallHandler` directly. Instead, introduce one shared dispatcher. In `lib/services/call_service.dart`, replace:

```dart
  void setCallStateCallback(void Function(String status, String? number) cb) {
    _method.setMethodCallHandler((call) async {
      if (call.method == 'onCallStateChanged') {
        final args = call.arguments is Map ? call.arguments as Map : null;
        final status = args?['status'] as String? ?? 'idle';
        final number = args?['number'] as String?;
        cb(status, number);
      }
    });
  }
```

with:

```dart
  void Function(String status, String? number)? _callStateCb;

  void setCallStateCallback(void Function(String status, String? number) cb) {
    _callStateCb = cb;
  }

  Future<dynamic> handleMethodCall(MethodCall call) async {
    if (call.method == 'onCallStateChanged') {
      final args = call.arguments is Map ? call.arguments as Map : null;
      final status = args?['status'] as String? ?? 'idle';
      final number = args?['number'] as String?;
      _callStateCb?.call(status, number);
    }
    return null;
  }
```

Add `import 'package:flutter/services.dart';` at the top if not already present (it already is, for `MethodChannel`).

In `lib/services/playback_capture_service.dart`, change `_onMethodCall` to also forward unrecognized calls, and accept a `CallService` to delegate to:

```dart
class PlaybackCaptureService {
  static const _method = MethodChannel('com.voiceguard/calls');
  final _consentCtrl = StreamController<bool>.broadcast();
  final CallService? _callService;

  PlaybackCaptureService({CallService? callService}) : _callService = callService {
    _method.setMethodCallHandler(_onMethodCall);
  }

  Future<dynamic> _onMethodCall(MethodCall call) async {
    if (call.method == 'onPlaybackCaptureConsent') {
      final args = call.arguments is Map ? call.arguments as Map : null;
      _consentCtrl.add(args?['granted'] as bool? ?? false);
      return null;
    }
    return _callService?.handleMethodCall(call);
  }
  // ... rest unchanged
```

Add `import 'call_service.dart';` at the top of `playback_capture_service.dart`.

In `lib/main.dart`, find where `CallService` is constructed and `setCallStateCallback` is wired (this is also where `Provider`s are set up), and construct `PlaybackCaptureService(callService: callService)` there so it becomes the single registered handler, forwarding to `CallService` for call-state events. Add it to the widget tree's `MultiProvider` alongside the existing services:

```dart
        Provider<PlaybackCaptureService>(create: (_) => PlaybackCaptureService(callService: calls)),
```

(matching whatever variable name the existing `CallService` instance uses in that file — read the surrounding `MultiProvider` block first to match the exact pattern already there, e.g. how `AudioService`/`CallService` are provided.)

- [ ] **Step 2: Run the existing test suite to confirm nothing broke**

Run: `cd voice_guard && flutter test`
Expected: all existing tests still pass, plus the two from Task 4.

- [ ] **Step 3: Write the VoIP Protection screen**

Create `lib/screens/voip_protection_screen.dart`:

```dart
import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../services/playback_capture_service.dart';
import '../services/audio_service.dart';
import '../providers/risk_score_provider.dart';
import '../widgets/risk_meter.dart';

class VoipProtectionScreen extends StatefulWidget {
  const VoipProtectionScreen({super.key});
  @override
  State<VoipProtectionScreen> createState() => _VoipProtectionScreenState();
}

class _VoipProtectionScreenState extends State<VoipProtectionScreen> {
  bool _capturing = false;
  StreamSubscription<bool>? _consentSub;

  @override
  void initState() {
    super.initState();
    final capture = context.read<PlaybackCaptureService>();
    _consentSub = capture.consentResult.listen(_onConsent);
  }

  Future<void> _onConsent(bool granted) async {
    if (!granted) return;
    final capture = context.read<PlaybackCaptureService>();
    final audio = context.read<AudioService>();
    final risk = context.read<RiskScoreProvider>();
    risk.reset();
    audio.clearBuffer();
    audio.startScoring();
    final started = await capture.startCapture();
    if (mounted) setState(() => _capturing = started);
  }

  Future<void> _start() async {
    final capture = context.read<PlaybackCaptureService>();
    await capture.requestConsent();
    // startCapture() runs from _onConsent once the system dialog resolves.
  }

  Future<void> _stop() async {
    final capture = context.read<PlaybackCaptureService>();
    final audio = context.read<AudioService>();
    await capture.stopCapture();
    audio.stopScoring();
    if (mounted) setState(() => _capturing = false);
  }

  @override
  void dispose() {
    _consentSub?.cancel();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final risk = context.watch<RiskScoreProvider>().current;
    final score = risk?.score ?? 0.0;

    return Scaffold(
      appBar: AppBar(title: const Text('VoIP Call Protection')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(children: [
          const Text(
            'Analyzes the other side\'s voice on a WhatsApp, Telegram, or '
            'Zoom call while it plays. This does not analyze your own '
            'microphone input.',
            style: TextStyle(fontSize: 12),
          ),
          const SizedBox(height: 20),
          RiskMeter(score: score),
          const SizedBox(height: 20),
          ElevatedButton(
            onPressed: _capturing ? _stop : _start,
            child: Text(_capturing ? 'Stop Protection' : 'Start Protection'),
          ),
        ]),
      ),
    );
  }
}
```

- [ ] **Step 4: Add a navigation entry point**

In `lib/main.dart`, find the `Quick Actions` row on the home screen (search for `'Live Call'` — the existing quick-action card that navigates to `CallScreen`). Add a fourth card alongside it that pushes `VoipProtectionScreen`:

```dart
              _QuickActionCard(
                icon: Icons.videocam_outlined,
                title: 'VoIP Protection',
                subtitle: 'WhatsApp / Zoom',
                onTap: () => Navigator.of(context).push(
                  MaterialPageRoute(builder: (_) => const VoipProtectionScreen()),
                ),
              ),
```

(Match whichever private widget class the existing quick-action cards use — read the surrounding code for the exact name and constructor signature before copying this in; it is very likely named something like `_QuickActionCard` given the pattern of the rest of the home screen, but confirm against the actual source rather than assuming.)

Add `import 'screens/voip_protection_screen.dart';` at the top of `lib/main.dart`.

- [ ] **Step 5: Manual verification on-device**

Run: `cd voice_guard && flutter build apk --debug && adb install -r build/app/outputs/flutter-apk/app-debug.apk`

Then, with a WhatsApp voice call active in the background on the test device:
1. Open VoiceGuard → VoIP Protection → Start Protection.
2. Confirm the system "Start recording or casting your screen?" dialog appears and grant it.
3. Watch `adb logcat -s PlaybackCaptureManager:I` — expect `Playback capture started (16kHz mono)`.
4. Confirm the risk meter moves off its resting value while the other party speaks (unlike the carrier-call case, this should show real, moving scores — if it stays flat, capture is not receiving real bytes; check `AudioRecord.STATE_INITIALIZED` logs for a rejection).

Record the actual per-app result (WhatsApp / Telegram / Zoom / Meet) in this plan file's Task 6 below — do not assume all four behave the same.

- [ ] **Step 6: Commit**

```bash
git add lib/screens/voip_protection_screen.dart lib/main.dart lib/services/call_service.dart lib/services/playback_capture_service.dart
git commit -m "feat(voice_guard): add VoIP Call Protection screen using AudioPlaybackCapture"
```

---

## Task 6: Per-app capture policy verification (manual QA, no code)

**Files:** None — this task is a verification log, append results directly into this plan document under this task once run.

- [ ] **Step 1:** With Task 5 built and installed, start a real WhatsApp voice call between two devices (or a WhatsApp call to a landline/voicemail so it actually connects). Start VoIP Protection. Confirm real, moving risk scores (not stuck at a constant value) for at least 15 seconds of the other party's actual speech. Record: PASS / FAIL / partial (specify what happened).
- [ ] **Step 2:** Repeat with a Telegram voice call. Record result.
- [ ] **Step 3:** Repeat with a Zoom call (audio-only is fine). Record result.
- [ ] **Step 4:** Repeat with Google Meet. Record result.
- [ ] **Step 5:** For any app that FAILs, check `adb logcat -s PlaybackCaptureManager:I PlaybackCaptureManager:E` during that attempt — if `AudioRecord.STATE_INITIALIZED` check fails or the read loop only ever produces exact-zero samples (same measurement technique used earlier in this conversation: pull output and check per-second RMS with a small Python script), that app has very likely called `setAllowedCapturePolicy(ALLOW_CAPTURE_BY_NONE)` on its call-audio stream and cannot be supported via this method — note that as a documented limitation for the demo pitch, don't keep debugging that specific app's policy from the VoiceGuard side (there is nothing to fix on our end if the source app opted out).

---

## Self-Review Notes

- **Spec coverage:** every element of the Architecture section maps to a task — consent flow (Task 3), capture manager (Task 2), foreground service requirement (Task 1), Dart wrapper (Task 4), UI + pipeline reuse (Task 5), per-app verification (Task 6).
- **Placeholder scan:** no TBD/TODO markers were introduced (the pre-existing `// TODO:` lines in `build.gradle.kts` predate this plan and are out of scope).
- **Type consistency:** `PlaybackCaptureService.startCapture()`/`stopCapture()`/`requestConsent()` names match what Task 5's screen calls; `consentResult` stream type (`Stream<bool>`) matches the `bool granted` payload sent from native in Task 3.
