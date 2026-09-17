# VoIP Native Audio Resampling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix `RemoteAudioTap.kt` so the AI detector receives correctly
formatted 16kHz mono PCM16 from a live Protected Call, instead of raw
48kHz (likely stereo) WebRTC buffers — which currently warp frequency by
~3x and make LFCC feature extraction invalid (confirmed defect,
`Voip.md` §1 "Audio Resampling" row).

**Architecture:** Add a small, pure-Kotlin `AudioResampler16k` (no Android
or WebRTC types — takes `ByteBuffer`/ints in, emits 100ms `ByteArray`
chunks via callback) that does stereo-downmix + point-sample downsampling.
`RemoteAudioTap`'s existing `AudioTrackSink` callback feeds it instead of
forwarding raw bytes untouched. This is a pure-function core wrapped by the
minimum native-audio-plumbing glue, which is what makes it unit-testable at
all without an Android instrumented-test harness (none exists in this repo
today).

**Tech Stack:** Kotlin (Android app module), JUnit4 for the new JVM unit
test.

**Spec:** `Voip.md` §"Phase 3: Native Audio Processing & Resampling
Engine" (lines 115-171) and §"Phase 6: Testing & Validation Matrix" item 1
("Resampler audio integrity: Verify 48kHz sine wave input produces 16kHz
sine wave output without aliasing", line 212). No separate design doc was
written for this sub-project — per the user's explicit direction to move
straight to plans tonight, `Voip.md`'s own Phase 3 section (which already
includes working reference code) serves as this plan's spec.

## Global Constraints

- Output chunks must be exactly 3200 bytes (100ms @ 16kHz mono 16-bit) —
  this is what the existing Flutter-side pipeline
  (`AudioService.ingestBytes` → `AudioProcessor`) already assumes
  downstream; changing it is out of scope.
- `AudioResampler16k` must have zero Android/WebRTC framework
  dependencies — this is what makes Task 1 pure-JVM-testable without
  Robolectric or an emulator.
- `RemoteAudioTap`'s public `attach`/`detach` signatures
  (`RemoteAudioTap.kt:29,43`) do not change — `WebRtcCallService.dart`'s
  method-channel contract depends on them and is out of this
  sub-project's scope (sub-project 3).

---

### Task 1: `AudioResampler16k` — stereo downmix + 48kHz→16kHz resample, unit-tested

**Files:**
- Create: `voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/AudioResampler16k.kt`
- Create: `voice_guard/android/app/src/test/kotlin/com/voiceguard/voice_guard/AudioResampler16kTest.kt`
- Modify: `voice_guard/android/app/build.gradle.kts` (add JUnit test dependency)

**Interfaces:**
- Produces: `class AudioResampler16k(onChunkReady: (ByteArray) -> Unit)` with
  `fun processAudio(buffer: java.nio.ByteBuffer, sampleRate: Int, channels: Int, frames: Int)`.
  Each call to `onChunkReady` delivers exactly 3200 bytes (100ms @ 16kHz
  mono PCM16 little-endian). Consumed by Task 2's `RemoteAudioTap`.

- [ ] **Step 1: Add the JUnit dependency**

In `voice_guard/android/app/build.gradle.kts`, in the `dependencies {}`
block, add:

```kotlin
    testImplementation("junit:junit:4.13.2")
```

- [ ] **Step 2: Write the failing test**

```kotlin
// voice_guard/android/app/src/test/kotlin/com/voiceguard/voice_guard/AudioResampler16kTest.kt
package com.voiceguard.voice_guard

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.PI
import kotlin.math.roundToInt
import kotlin.math.sin

class AudioResampler16kTest {

    /** Builds an interleaved PCM16LE buffer of a sine wave at [freqHz], [sampleRate], [channels]. */
    private fun sineBuffer(freqHz: Double, sampleRate: Int, channels: Int, frames: Int): ByteBuffer {
        val buf = ByteBuffer.allocate(frames * channels * 2).order(ByteOrder.LITTLE_ENDIAN)
        for (i in 0 until frames) {
            val sample = (0.5 * Short.MAX_VALUE * sin(2 * PI * freqHz * i / sampleRate)).roundToInt().toShort()
            for (c in 0 until channels) buf.putShort(sample)
        }
        buf.flip()
        return buf
    }

    @Test
    fun `mono 48kHz input downsamples to 16kHz output at one-third the sample count`() {
        val chunks = mutableListOf<ByteArray>()
        val resampler = AudioResampler16k(onChunkReady = { chunks.add(it) })

        // 300ms of 48kHz mono audio -> should yield 3 full 100ms/3200-byte chunks.
        val frames = 48000 * 300 / 1000
        resampler.processAudio(sineBuffer(220.0, 48000, 1, frames), sampleRate = 48000, channels = 1, frames = frames)

        assertEquals(3, chunks.size)
        chunks.forEach { assertEquals(3200, it.size) }
    }

    @Test
    fun `stereo input is downmixed to mono by averaging channels`() {
        // Left channel silent, right channel full-scale square-ish tone: downmix
        // should land at roughly half the right channel's amplitude, never at
        // either channel's amplitude alone (that would mean downmix is broken —
        // e.g. only reading the left channel).
        val frames = 48000 * 100 / 1000 // 100ms -> exactly 1 output chunk
        val buf = ByteBuffer.allocate(frames * 2 * 2).order(ByteOrder.LITTLE_ENDIAN)
        val rightAmplitude = 20000.toShort()
        for (i in 0 until frames) {
            buf.putShort(0) // left
            buf.putShort(rightAmplitude) // right
        }
        buf.flip()

        val chunks = mutableListOf<ByteArray>()
        val resampler = AudioResampler16k(onChunkReady = { chunks.add(it) })
        resampler.processAudio(buf, sampleRate = 48000, channels = 2, frames = frames)

        assertEquals(1, chunks.size)
        val out = ByteBuffer.wrap(chunks[0]).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer()
        val outSamples = ShortArray(out.remaining()) { out.get(it) }
        assertTrue(outSamples.all { it.toInt() == rightAmplitude / 2 })
    }

    @Test
    fun `output chunk size is always exactly 3200 bytes`() {
        val chunks = mutableListOf<ByteArray>()
        val resampler = AudioResampler16k(onChunkReady = { chunks.add(it) })
        val frames = 48000 * 1000 / 1000 // 1 full second, at 48kHz
        resampler.processAudio(sineBuffer(150.0, 48000, 1, frames), sampleRate = 48000, channels = 1, frames = frames)

        assertTrue(chunks.isNotEmpty())
        chunks.forEach { assertEquals(3200, it.size) }
    }
}
```

- [ ] **Step 3: Run test to verify it fails**

Run: `cd voice_guard/android && ./gradlew testStandardDebugUnitTest --tests "com.voiceguard.voice_guard.AudioResampler16kTest"`
Expected: FAIL to compile — `AudioResampler16k` doesn't exist yet.

- [ ] **Step 4: Implement `AudioResampler16k`**

```kotlin
// voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/AudioResampler16k.kt
package com.voiceguard.voice_guard

import java.nio.ByteBuffer

/**
 * Downmixes (stereo -> mono) and downsamples (any input rate -> 16kHz)
 * WebRTC PCM16 buffers, batching output into fixed 3200-byte (100ms @
 * 16kHz mono 16-bit) chunks. Point-sampling, not bandlimited resampling —
 * adequate for speech feature extraction, matching the reference design in
 * Voip.md Phase 3. Pure Kotlin, no Android/WebRTC types, so it's testable
 * as a plain JVM unit.
 */
class AudioResampler16k(private val onChunkReady: (ByteArray) -> Unit) {
    private val outChunkSize = 3200 // 100ms @ 16kHz mono 16-bit
    private var chunkBuffer = ByteArray(outChunkSize)
    private var chunkPos = 0

    fun processAudio(buffer: ByteBuffer, sampleRate: Int, channels: Int, frames: Int) {
        val shortBuf = buffer.asShortBuffer()
        if (shortBuf.remaining() == 0 || frames == 0) return

        val step = sampleRate.toDouble() / 16000.0
        var inIndex = 0.0

        while (inIndex < frames) {
            val frameIdx = inIndex.toInt()
            val sample = when (channels) {
                1 -> shortBuf.get(frameIdx).toInt()
                2 -> {
                    val left = shortBuf.get(frameIdx * 2).toInt()
                    val right = shortBuf.get(frameIdx * 2 + 1).toInt()
                    (left + right) / 2
                }
                else -> shortBuf.get(frameIdx * channels).toInt() // best-effort: first channel only
            }

            chunkBuffer[chunkPos++] = (sample and 0xFF).toByte()
            chunkBuffer[chunkPos++] = ((sample shr 8) and 0xFF).toByte()

            if (chunkPos >= outChunkSize) {
                onChunkReady(chunkBuffer.copyOf())
                chunkPos = 0
            }

            inIndex += step
        }
    }
}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `cd voice_guard/android && ./gradlew testStandardDebugUnitTest --tests "com.voiceguard.voice_guard.AudioResampler16kTest"`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add voice_guard/android/app/build.gradle.kts \
        voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/AudioResampler16k.kt \
        voice_guard/android/app/src/test/kotlin/com/voiceguard/voice_guard/AudioResampler16kTest.kt
git commit -m "feat(voice_guard): add unit-tested 48kHz->16kHz mono audio resampler"
```

---

### Task 2: Wire `AudioResampler16k` into `RemoteAudioTap`'s sink

**Files:**
- Modify: `voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/RemoteAudioTap.kt`

**Interfaces:**
- Consumes: `AudioResampler16k` (Task 1).
- Produces: unchanged public surface — `RemoteAudioTap.attach(plugin, trackId, onData): Boolean`,
  `RemoteAudioTap.detach()` (lines 29, 43) — `onData` now receives resampled
  16kHz mono chunks instead of raw buffer bytes. No change needed on the
  Dart side (`WebRtcCallService._attachRemoteAudioTap`,
  `webrtc_call_service.dart:86-92`) since it already just forwards whatever
  bytes arrive to `audioService.ingestBytes`.

- [ ] **Step 1: Confirm the `AudioTrackSink` callback's actual parameter order before relying on it**

The current code (`RemoteAudioTap.kt:32`) discards all but the first
`AudioTrackSink` callback parameter:
```kotlin
val newSink = AudioTrackSink { buffer, _, _, _, _, _ -> ... }
```
`org.webrtc.AudioTrackSink.onData` is `(audioData: ByteBuffer, bitsPerSample: Int, sampleRate: Int, numberOfChannels: Int, numberOfFrames: Int, absoluteCaptureTimestampMs: Long)`
— confirm this against the actual `io.github.webrtc-sdk:android:125.6422.03`
artifact (e.g. via Android Studio "Go to Declaration" on `AudioTrackSink`,
or by temporarily logging all six values from a live call and checking
`sampleRate` prints `48000` and `bitsPerSample` prints `16`) before wiring
Step 2 — a silent parameter-order mismatch here (e.g. swapping
`sampleRate`/`numberOfChannels`) would make the resampler silently produce
garbage rather than fail loudly.

- [ ] **Step 2: Route the sink through the resampler**

Replace the `attach` function's body:

```kotlin
    fun attach(plugin: FlutterWebRTCPlugin, trackId: String, onData: (ByteArray) -> Unit): Boolean {
        val remoteTrack = plugin.getRemoteTrack(trackId) as? AudioTrack ?: return false
        detach()
        val resampler = AudioResampler16k(onChunkReady = onData)
        val newSink = AudioTrackSink { buffer, _, sampleRate, numberOfChannels, numberOfFrames, _ ->
            resampler.processAudio(buffer, sampleRate, numberOfChannels, numberOfFrames)
        }
        remoteTrack.addSink(newSink)
        sink = newSink
        track = remoteTrack
        return true
    }
```

Update the class doc comment (lines 7-23) to note it now resamples rather
than passing raw buffers through — the existing comment's technical
justification for *why* `addSink` is used at all stays accurate and should
be kept, just add a line noting the resampling step.

- [ ] **Step 3: Manual verification on-device**

This requires a live Protected Call (needs sub-project 1's signaling
working, or the existing `10.0.2.2` emulator-loopback path for a quick same
network sanity check). Run: `cd voice_guard && flutter run`, start a
Protected Call between two instances, and confirm via `adb logcat` (filter
on the app's tag) that no exceptions are thrown from `RemoteAudioTap` and
that `AudioService.ingestBytes` (Dart side) keeps receiving data — its
existing `debugPrint('Monitor: chunkRms=...')` log (audio_service.dart:112)
should show plausible non-zero RMS values once someone speaks, confirming
the resampled stream isn't degenerate.

- [ ] **Step 4: Commit**

```bash
git add voice_guard/android/app/src/main/kotlin/com/voiceguard/voice_guard/RemoteAudioTap.kt
git commit -m "feat(voice_guard): resample remote WebRTC audio to 16kHz mono before scoring"
```
