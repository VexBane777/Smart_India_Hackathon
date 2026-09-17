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
