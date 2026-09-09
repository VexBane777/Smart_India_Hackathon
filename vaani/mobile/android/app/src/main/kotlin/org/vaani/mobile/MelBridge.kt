package org.vaani.mobile

import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.log10
import kotlin.math.max
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Pure-Kotlin port of app/components/spectrogram.py::compute_mel_db.
 * A visual-aid mel spectrogram (triangular filterbank approximation), not
 * a model feature extractor — matches the Python original's math exactly
 * (N_FFT=512, HOP=160 i.e. 10ms @ 16kHz, N_MELS=48) so the two platforms'
 * spectrograms look identical.
 */
object MelBridge {
    private const val N_FFT = 512
    private const val HOP = 160
    private const val N_MELS = 48

    private var cachedSr: Int = -1
    private var cachedBank: Array<DoubleArray>? = null

    fun computeMelDb(pcm: FloatArray, sr: Int): Array<DoubleArray> {
        val padded = if (pcm.size < N_FFT) pcm.copyOf(N_FFT) else pcm
        val nFrames = 1 + (padded.size - N_FFT) / HOP
        val bank = melFilterbank(sr)
        val nBins = N_FFT / 2 + 1
        val melDb = Array(N_MELS) { DoubleArray(nFrames) }
        val window = hanningWindow(N_FFT)

        for (f in 0 until nFrames) {
            val start = f * HOP
            val frame = DoubleArray(N_FFT) { i -> padded[start + i] * window[i] }
            val mags = rfftMagnitude(frame) // size nBins
            for (m in 0 until N_MELS) {
                var acc = 0.0
                for (k in 0 until nBins) acc += bank[m][k] * mags[k]
                melDb[m][f] = 20.0 * log10(max(acc, 1e-10))
            }
        }
        return melDb
    }

    private fun hanningWindow(n: Int): DoubleArray =
        DoubleArray(n) { i -> 0.5 - 0.5 * cos(2 * PI * i / (n - 1)) }

    private fun hzToMel(f: Double) = 2595.0 * log10(1.0 + f / 700.0)
    private fun melToHz(m: Double) = 700.0 * (10.0.pow(m / 2595.0) - 1.0)

    private fun melFilterbank(sr: Int): Array<DoubleArray> {
        cachedBank?.let { if (cachedSr == sr) return it }
        val nBins = N_FFT / 2 + 1
        val melLo = hzToMel(0.0)
        val melHi = hzToMel(sr / 2.0)
        val mPts = DoubleArray(N_MELS + 2) { i -> melLo + (melHi - melLo) * i / (N_MELS + 1) }
        val hzPts = mPts.map { melToHz(it) }
        val bins = hzPts.map { it / sr * N_FFT }
        val bank = Array(N_MELS) { DoubleArray(nBins) }
        for (i in 0 until N_MELS) {
            val left = bins[i]; val center = bins[i + 1]; val right = bins[i + 2]
            for (k in 0 until nBins) {
                when {
                    k > left && k < center -> bank[i][k] = (k - left) / (center - left)
                    k.toDouble() in center..right && k < right -> bank[i][k] = (right - k) / (right - center)
                }
            }
        }
        cachedSr = sr
        cachedBank = bank
        return bank
    }

    /** Magnitude of the real FFT (rfft) of a real-valued frame, via a
     *  straightforward iterative radix-2 Cooley-Tukey FFT (N_FFT=512 is a
     *  power of two, so no Bluestein fallback is needed). */
    private fun rfftMagnitude(frame: DoubleArray): DoubleArray {
        val n = frame.size
        val re = frame.copyOf()
        val im = DoubleArray(n)
        fft(re, im)
        val nBins = n / 2 + 1
        return DoubleArray(nBins) { k -> sqrt(re[k] * re[k] + im[k] * im[k]) }
    }

    private fun fft(re: DoubleArray, im: DoubleArray) {
        val n = re.size
        // bit-reversal permutation
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) {
                j = j xor bit
                bit = bit shr 1
            }
            j = j or bit
            if (i < j) {
                var tmp = re[i]; re[i] = re[j]; re[j] = tmp
                tmp = im[i]; im[i] = im[j]; im[j] = tmp
            }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wReal = cos(ang)
            val wImag = sin(ang)
            var i = 0
            while (i < n) {
                var curReal = 1.0
                var curImag = 0.0
                for (k in 0 until len / 2) {
                    val uRe = re[i + k]; val uIm = im[i + k]
                    val vRe = re[i + k + len / 2] * curReal - im[i + k + len / 2] * curImag
                    val vIm = re[i + k + len / 2] * curImag + im[i + k + len / 2] * curReal
                    re[i + k] = uRe + vRe; im[i + k] = uIm + vIm
                    re[i + k + len / 2] = uRe - vRe; im[i + k + len / 2] = uIm - vIm
                    val nextReal = curReal * wReal - curImag * wImag
                    val nextImag = curReal * wImag + curImag * wReal
                    curReal = nextReal; curImag = nextImag
                }
                i += len
            }
            len = len shl 1
        }
    }
}
