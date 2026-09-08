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

        // Only USAGE_UNKNOWN/USAGE_GAME/USAGE_MEDIA are legal match targets for
        // capturing another app's audio (see android.media.AudioPlaybackCaptureConfiguration.Builder
        // reference) — USAGE_VOICE_COMMUNICATION can never be matched here, by
        // platform design, regardless of MediaProjection consent. WhatsApp/Zoom/
        // Telegram/Meet correctly tag call audio as USAGE_VOICE_COMMUNICATION, so
        // this capture path cannot see live third-party VoIP call audio at all;
        // it only ever captures USAGE_MEDIA playback from the target app.
        val config = AudioPlaybackCaptureConfiguration.Builder(projection)
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
