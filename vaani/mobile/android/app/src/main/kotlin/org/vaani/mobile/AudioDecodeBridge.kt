package org.vaani.mobile

import android.media.MediaCodec
import android.media.MediaExtractor
import android.media.MediaFormat
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Decodes any Android-supported audio file (WAV/MP3/M4A/OGG/...) to mono
 *  16kHz float PCM in [-1, 1], mirroring server.py's _load_mono_float
 *  (channel-average, linear resample) so both platforms treat imported
 *  audio identically. */
object AudioDecodeBridge {
    private const val TARGET_SR = 16000

    fun decodeToPcm16kMono(path: String): FloatArray {
        val extractor = MediaExtractor()
        extractor.setDataSource(path)
        var trackIndex = -1
        var format: MediaFormat? = null
        for (i in 0 until extractor.trackCount) {
            val f = extractor.getTrackFormat(i)
            val mime = f.getString(MediaFormat.KEY_MIME) ?: ""
            if (mime.startsWith("audio/")) {
                trackIndex = i; format = f; break
            }
        }
        require(trackIndex >= 0 && format != null) { "no audio track found in $path" }
        extractor.selectTrack(trackIndex)

        val mime = format.getString(MediaFormat.KEY_MIME)!!
        val codec = MediaCodec.createDecoderByType(mime)
        codec.configure(format, null, null, 0)
        codec.start()

        val sourceChannels = format.getInteger(MediaFormat.KEY_CHANNEL_COUNT)
        val sourceSr = format.getInteger(MediaFormat.KEY_SAMPLE_RATE)

        val pcmOut = ArrayList<Short>()
        val bufferInfo = MediaCodec.BufferInfo()
        var sawInputEOS = false
        var sawOutputEOS = false

        while (!sawOutputEOS) {
            if (!sawInputEOS) {
                val inIndex = codec.dequeueInputBuffer(10_000)
                if (inIndex >= 0) {
                    val inBuffer = codec.getInputBuffer(inIndex)!!
                    val sampleSize = extractor.readSampleData(inBuffer, 0)
                    if (sampleSize < 0) {
                        codec.queueInputBuffer(inIndex, 0, 0, 0, MediaCodec.BUFFER_FLAG_END_OF_STREAM)
                        sawInputEOS = true
                    } else {
                        codec.queueInputBuffer(inIndex, 0, sampleSize, extractor.sampleTime, 0)
                        extractor.advance()
                    }
                }
            }

            val outIndex = codec.dequeueOutputBuffer(bufferInfo, 10_000)
            if (outIndex >= 0) {
                val outBuffer = codec.getOutputBuffer(outIndex)!!
                outBuffer.order(ByteOrder.LITTLE_ENDIAN)
                val shortBuf = outBuffer.asShortBuffer()
                val chunk = ShortArray(shortBuf.remaining())
                shortBuf.get(chunk)
                pcmOut.addAll(chunk.toList())
                codec.releaseOutputBuffer(outIndex, false)
                if (bufferInfo.flags and MediaCodec.BUFFER_FLAG_END_OF_STREAM != 0) {
                    sawOutputEOS = true
                }
            }
        }

        codec.stop()
        codec.release()
        extractor.release()

        // Channel-average to mono.
        val monoSamples = if (sourceChannels <= 1) {
            pcmOut.map { it / 32768.0f }
        } else {
            val mono = ArrayList<Float>(pcmOut.size / sourceChannels)
            var i = 0
            while (i + sourceChannels <= pcmOut.size) {
                var sum = 0f
                for (c in 0 until sourceChannels) sum += pcmOut[i + c] / 32768.0f
                mono.add(sum / sourceChannels)
                i += sourceChannels
            }
            mono
        }

        // Linear resample to TARGET_SR.
        if (sourceSr == TARGET_SR || monoSamples.isEmpty()) {
            return monoSamples.toFloatArray()
        }
        val ratio = TARGET_SR.toDouble() / sourceSr.toDouble()
        val outLen = (monoSamples.size * ratio).toInt()
        val resampled = FloatArray(outLen)
        for (i in 0 until outLen) {
            val srcPos = i / ratio
            val i0 = srcPos.toInt().coerceIn(0, monoSamples.size - 1)
            val i1 = (i0 + 1).coerceAtMost(monoSamples.size - 1)
            val frac = (srcPos - i0).toFloat()
            resampled[i] = monoSamples[i0] * (1 - frac) + monoSamples[i1] * frac
        }
        return resampled
    }
}
