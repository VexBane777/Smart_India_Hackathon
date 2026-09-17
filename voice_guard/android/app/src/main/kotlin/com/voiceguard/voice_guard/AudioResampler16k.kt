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
