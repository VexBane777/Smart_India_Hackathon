package com.voiceguard.voice_guard

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioTrack
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Plays 16kHz mono 16-bit PCM audio through the phone's speaker while the
 * "Test with audio file" scan runs, so the user can hear what the model is
 * analyzing. Ported from vaani/mobile's AudioPlaybackBridge.kt.
 *
 * Offers mute/unmute so scoring continues but audio is silenced.
 *
 * AudioTrack.write() in MODE_STREAM blocks once its internal buffer fills,
 * which is what naturally paces AudioService.scanAudioFile's otherwise-fast
 * scoring loop to real playback speed while unmuted — no extra timing code
 * needed on the Dart side for that.
 */
object AudioPlaybackBridge {
    private const val TAG = "AudioPlaybackBridge"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_OUT = AudioFormat.CHANNEL_OUT_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    @Volatile private var track: AudioTrack? = null
    @Volatile private var muted = false
    @Volatile private var playing = false

    /** Initialize and start the AudioTrack for streaming playback. */
    fun start(): Boolean {
        if (track != null && playing) return true

        val minBuf = AudioTrack.getMinBufferSize(SAMPLE_RATE, CHANNEL_OUT, ENCODING)
        if (minBuf <= 0) {
            android.util.Log.e(TAG, "Invalid min buffer size: $minBuf")
            return false
        }

        val bufSize = maxOf(minBuf * 4, SAMPLE_RATE * 2) // 2s buffer

        track = AudioTrack.Builder()
            .setAudioAttributes(
                AudioAttributes.Builder()
                    .setUsage(AudioAttributes.USAGE_MEDIA)
                    .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                    .build()
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setChannelMask(CHANNEL_OUT)
                    .setEncoding(ENCODING)
                    .build()
            )
            .setBufferSizeInBytes(bufSize)
            .setTransferMode(AudioTrack.MODE_STREAM)
            .build()

        track?.play()
        playing = true
        muted = false
        android.util.Log.i(TAG, "Audio playback started (16kHz mono)")
        return true
    }

    /** Play a chunk of 16-bit PCM mono audio ([-32768, 32767] samples). */
    fun playChunk(samples: List<Int>) {
        if (muted || track == null || !playing) return

        val buffer = ShortArray(samples.size)
        for (i in samples.indices) {
            buffer[i] = samples[i].coerceIn(-32768, 32767).toShort()
        }

        val byteBuffer = ByteBuffer.allocate(buffer.size * 2)
        byteBuffer.order(ByteOrder.LITTLE_ENDIAN)
        for (s in buffer) {
            byteBuffer.putShort(s)
        }

        val audioBuffer = byteBuffer.array()
        val writeResult = track?.write(audioBuffer, 0, audioBuffer.size) ?: 0
        if (writeResult < 0) {
            android.util.Log.e(TAG, "AudioTrack.write failed: $writeResult")
        }
    }

    /** Mute or unmute playback. Scoring continues regardless. */
    fun setMuted(mute: Boolean) {
        muted = mute
        android.util.Log.i(TAG, "Playback muted: $mute")
    }

    /** Stop playback and release the AudioTrack. */
    fun stop() {
        playing = false
        try {
            track?.stop()
        } catch (_: Exception) {}
        try {
            track?.release()
        } catch (_: Exception) {}
        track = null
        android.util.Log.i(TAG, "Audio playback stopped")
    }

    fun isPlaying(): Boolean = playing
    fun isMuted(): Boolean = muted
}
