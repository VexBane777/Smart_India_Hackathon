package org.vaani.mobile

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import kotlin.math.PI
import kotlin.math.sin

class MelBridgeTest {
    @Test
    fun `output shape is N_MELS by expected frame count`() {
        val sr = 16000
        val durationS = 2.0
        val pcm = FloatArray((sr * durationS).toInt()) { 0f }
        val mel = MelBridge.computeMelDb(pcm, sr)
        assertEquals(48, mel.size) // N_MELS
        val expectedFrames = 1 + (pcm.size - 512) / 160
        assertEquals(expectedFrames, mel[0].size)
    }

    @Test
    fun `a pure tone concentrates energy in a narrow mel band`() {
        val sr = 16000
        val freq = 1000.0
        val n = sr // 1 s
        val pcm = FloatArray(n) { i -> sin(2 * PI * freq * i / sr).toFloat() }
        val mel = MelBridge.computeMelDb(pcm, sr)
        // Average energy per mel bin across all frames.
        val bandEnergy = mel.map { row -> row.average() }
        val maxBin = bandEnergy.indices.maxByOrNull { bandEnergy[it] }!!
        val totalAboveFloor = bandEnergy.count { it > bandEnergy[maxBin] - 10.0 }
        // The peak bin plus its immediate neighbors should dominate; a pure
        // tone should not spread energy evenly across all 48 bins.
        assertTrue("expected a concentrated peak, got $totalAboveFloor bins within 10dB of max",
            totalAboveFloor <= 6)
    }

    @Test
    fun `short input is zero-padded rather than throwing`() {
        val mel = MelBridge.computeMelDb(FloatArray(100) { 0f }, 16000)
        assertEquals(48, mel.size)
        assertEquals(1, mel[0].size)
    }
}
