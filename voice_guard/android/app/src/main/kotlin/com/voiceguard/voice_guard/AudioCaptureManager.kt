package com.voiceguard.voice_guard

import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * AudioCaptureManager — Captures 16kHz 16-bit mono PCM from VOICE_COMMUNICATION / MIC.
 * Real Call Recording: Writes PCM chunks to a standard 44-byte WAV file in the app's
 * external files directory so calls are permanently recorded and playable.
 * Real-Time Stream: Pushes PCM chunks to Flutter via EventChannel for on-device TFLite inference.
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

    var lastRecordingPath: String? = null
        private set

    private var currentRecordingFile: File? = null
    private var recordingStream: FileOutputStream? = null
    private var totalBytesRecorded = 0L

    fun setSink(s: ((ByteArray) -> Unit)?) { sink = s }

    fun start(context: Context, onBytes: ((ByteArray) -> Unit)? = null) {
        if (onBytes != null) {
            sink = onBytes
        }
        if (running) return
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        if (minBuf <= 0) { Log.e(TAG, "Invalid min buffer size: $minBuf"); return }
        val bufSize = minBuf * 4
        try {
            // VOICE_COMMUNICATION activates hardware AEC & beamforming for calls; fallback to MIC
            recorder = try {
                AudioRecord(
                    MediaRecorder.AudioSource.VOICE_COMMUNICATION,
                    SAMPLE_RATE, CHANNEL, ENCODING, bufSize
                )
            } catch (_: Exception) {
                AudioRecord(
                    MediaRecorder.AudioSource.MIC,
                    SAMPLE_RATE, CHANNEL, ENCODING, bufSize
                )
            }
        } catch (e: Exception) {
            Log.e(TAG, "AudioRecord creation failed", e); return
        }
        if (recorder?.state != AudioRecord.STATE_INITIALIZED) {
            Log.e(TAG, "AudioRecord not initialized"); return
        }

        // Initialize Call Recording WAV File
        try {
            val recDir = File(context.getExternalFilesDir(null), "Recordings").apply { mkdirs() }
            val timeStamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
            val recFile = File(recDir, "call_${timeStamp}.wav")
            currentRecordingFile = recFile
            totalBytesRecorded = 0L
            val fos = FileOutputStream(recFile)
            fos.write(ByteArray(44)) // placeholder for WAV header
            recordingStream = fos
            Log.i(TAG, "Call recording started: ${recFile.absolutePath}")
        } catch (e: Exception) {
            Log.e(TAG, "Failed to initialize call recording file", e)
        }

        recorder?.startRecording()
        running = true
        thread = Thread {
            val buf = ByteArray(3200) // 100ms @16kHz mono 16-bit
            while (running) {
                val n = recorder?.read(buf, 0, buf.size) ?: 0
                if (n > 0) {
                    val copy = buf.copyOf(n)
                    // 1. Write audio chunk to WAV file on disk
                    try {
                        recordingStream?.write(buf, 0, n)
                        totalBytesRecorded += n
                    } catch (e: Exception) {
                        Log.w(TAG, "Error writing audio chunk to recording", e)
                    }
                    // 2. Stream chunk to Flutter for real-time model scoring & waveform
                    try { sink?.invoke(copy) } catch (_: Exception) {}
                }
            }
        }.also { it.isDaemon = true; it.start() }
        Log.i(TAG, "Audio capture and call recording started (16kHz mono)")
    }

    fun stop() {
        running = false
        thread?.interrupt()
        thread = null
        try { recorder?.stop() } catch (_: Exception) {}
        try { recorder?.release() } catch (_: Exception) {}
        recorder = null

        // Finalize WAV Header on disk
        try {
            recordingStream?.flush()
            recordingStream?.close()
            recordingStream = null

            val recFile = currentRecordingFile
            if (recFile != null && recFile.exists() && totalBytesRecorded > 0) {
                RandomAccessFile(recFile, "rw").use { raf ->
                    writeWavHeader(
                        raf,
                        totalBytesRecorded,
                        totalBytesRecorded + 36,
                        SAMPLE_RATE.toLong(),
                        1,
                        (SAMPLE_RATE * 2).toLong()
                    )
                }
                lastRecordingPath = recFile.absolutePath
                Log.i(TAG, "Saved call recording to: ${recFile.absolutePath} ($totalBytesRecorded bytes)")
            }
        } catch (e: Exception) {
            Log.e(TAG, "Error finalizing WAV file", e)
        }

        Log.i(TAG, "Audio capture stopped")
    }

    private fun writeWavHeader(
        out: RandomAccessFile,
        totalAudioLen: Long,
        totalDataLen: Long,
        sampleRate: Long,
        channels: Int,
        byteRate: Long
    ) {
        val header = ByteArray(44)
        header[0] = 'R'.code.toByte() // RIFF/WAVE header
        header[1] = 'I'.code.toByte()
        header[2] = 'F'.code.toByte()
        header[3] = 'F'.code.toByte()
        header[4] = (totalDataLen and 0xff).toByte()
        header[5] = ((totalDataLen shr 8) and 0xff).toByte()
        header[6] = ((totalDataLen shr 16) and 0xff).toByte()
        header[7] = ((totalDataLen shr 24) and 0xff).toByte()
        header[8] = 'W'.code.toByte()
        header[9] = 'A'.code.toByte()
        header[10] = 'V'.code.toByte()
        header[11] = 'E'.code.toByte()
        header[12] = 'f'.code.toByte() // 'fmt ' chunk
        header[13] = 'm'.code.toByte()
        header[14] = 't'.code.toByte()
        header[15] = ' '.code.toByte()
        header[16] = 16
        header[17] = 0
        header[18] = 0
        header[19] = 0
        header[20] = 1 // format = 1 (PCM)
        header[21] = 0
        header[22] = channels.toByte()
        header[23] = 0
        header[24] = (sampleRate and 0xff).toByte()
        header[25] = ((sampleRate shr 8) and 0xff).toByte()
        header[26] = ((sampleRate shr 16) and 0xff).toByte()
        header[27] = ((sampleRate shr 24) and 0xff).toByte()
        header[28] = (byteRate and 0xff).toByte()
        header[29] = ((byteRate shr 8) and 0xff).toByte()
        header[30] = ((byteRate shr 16) and 0xff).toByte()
        header[31] = ((byteRate shr 24) and 0xff).toByte()
        header[32] = (channels * 2).toByte() // block align
        header[33] = 0
        header[34] = 16 // bits per sample
        header[35] = 0
        header[36] = 'd'.code.toByte()
        header[37] = 'a'.code.toByte()
        header[38] = 't'.code.toByte()
        header[39] = 'a'.code.toByte()
        header[40] = (totalAudioLen and 0xff).toByte()
        header[41] = ((totalAudioLen shr 8) and 0xff).toByte()
        header[42] = ((totalAudioLen shr 16) and 0xff).toByte()
        header[43] = ((totalAudioLen shr 24) and 0xff).toByte()
        out.seek(0)
        out.write(header, 0, 44)
    }
}
