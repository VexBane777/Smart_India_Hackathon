package org.vaani.mobile

import android.media.AudioAttributes
import android.media.AudioFormat
import android.media.AudioManager
import android.media.AudioTrack
import io.flutter.plugin.common.MethodChannel
import java.nio.ByteBuffer
import java.nio.ByteOrder

/**
 * Plays 16kHz mono 16-bit PCM audio through the phone's speaker (or earpiece
 * depending on audio route). Used by the file-import / playback pipeline so
 * the user can hear what the model is analyzing in real time.
 *
 * Offers mute/unmute so scoring continues but audio is silenced — useful when
 * testing in shared spaces or when the audio is already known.
 *
 * All chunks must be 16-bit PCM mono at 16000 Hz. Chunks are queued and
 * played in order; the caller is responsible for feeding chunks at the
 * correct rate (0.5s chunks every 500ms for real-time playback).
 */
object AudioPlaybackBridge {
    private const val TAG = "AudioPlaybackBridge"
    private const val SAMPLE_RATE = 16000
    private const val CHANNEL_OUT = AudioFormat.CHANNEL_OUT_MONO
    private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT

    @Volatile private var track: AudioTrack? = null
    @Volatile private var muted = false
    @Volatile private var playing = false

    /**
     * Initialize and start the AudioTrack for streaming playback.
     * Must be called on the main thread (Flutter plugin handler context).
     */
    fun start(): Boolean {
        if (track != null && playing) return true

        val minBuf = AudioTrack.getMinBufferSize(
            SAMPLE_RATE, CHANNEL_OUT, ENCODING
        )
        if (minBuf <= 0) {
            android.util.Log.e(TAG, "Invalid min buffer size: $minBuf")
            return false
        }

        // Use a buffer large enough for ~1s of audio to avoid underruns
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

    /**
     * Play a chunk of 16-bit PCM mono audio.
     * [samples] should be a List<Int> of 16-bit PCM samples in [-32768, 32767].
     */
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

    /**
     * Mute or unmute playback. Scoring continues regardless.
     */
    fun setMuted(mute: Boolean) {
        muted = mute
        android.util.Log.i(TAG, "Playback muted: $mute")
    }

    /**
     * Stop playback and release the AudioTrack.
     */
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
