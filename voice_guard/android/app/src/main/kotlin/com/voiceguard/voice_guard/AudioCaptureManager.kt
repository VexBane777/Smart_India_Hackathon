package com.voiceguard.voice_guard

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log

/**
 * Captures PCM16 mono 16kHz via MIC source (Android 10+ blocks VOICE_CALL for
 * non-privileged apps — POC uses MIC + speakerphone; production needs telecom-side
 * media forking). Streams raw bytes to Flutter via EventChannel.
 * Raw PCM is held only in RAM and never persisted.
 */
object AudioCaptureManager {
    private const val TAG = "AudioCaptureManager"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    private var recorder: AudioRecord? = null
    private var thread: Thread? = null
    @Volatile private var running = false
    private var sink: ((ByteArray) -> Unit)? = null

    fun setSink(s: ((ByteArray) -> Unit)?) { sink = s }

    fun start(context: Context, onBytes: (ByteArray) -> Unit) {
        if (running) return
        sink = onBytes
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) { Log.e(TAG, "Invalid min buffer size: $minBuf"); return }
        val bufSize = minBuf * 4
        try {
            recorder = AudioRecord(
                MediaRecorder.AudioSource.MIC,
                SAMPLE_RATE, CHANNEL, ENCODING, bufSize
            )
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord creation failed", e); return
        }
        if (recorder?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized"); return
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
        Log.i(TAG, "Audio capture started (MIC, 16kHz mono)")
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
        try { recorder?.stop() } catch (_: Exception) {}
        try { recorder?.release() } catch (_: Exception) {}
        recorder = null
        Log.i(TAG, "Audio capture stopped")
    }
}
